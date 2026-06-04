"""
Coding sub-graph: CodeGenLimitGate → CodeGen → SemgrepScan → E2BExecute.

SafetyJudge and IOAllowlistJudge are temporarily disabled (Semgrep-only pipeline).
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Literal, NotRequired, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.types import TraceContext
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, NodeError
from langgraph.graph import END, StateGraph
from langgraph.types import Command, RetryPolicy
from session_paths import ensure_session_dirs, session_dir_for_paths, session_root

from observability.langfuse_handler import add_trace_context_to_config, current_trace_context, safe_reset_contextvar, serialize_messages, trace_context_from_runnable_config, traced_generation, traced_span
from sub_agents.coding_sub_agent.config import (
    CODING_GRAPH_RECURSION_LIMIT,
    CODING_MODEL,
    CODING_MODEL_LAST_ATTEMPT,
    CODING_NODE_RETRY_BACKOFF_FACTOR,
    CODING_NODE_RETRY_INITIAL_INTERVAL,
    CODING_NODE_RETRY_MAX_ATTEMPTS,
    E2B_API_KEY,
    E2B_EXECUTION_TIMEOUT_SECONDS,
    E2B_KILL_SANDBOX,
    E2B_SANDBOX_TIMEOUT_SECONDS,
    E2B_TEMPLATE_NAME,
    MAX_CODEGEN_ATTEMPTS,
    PLOT_FILE_EXTENSIONS,
)
# DISABLED: LLM judges — CODE_JUDGE_MODEL, IO_JUDGE_MODEL
from sub_agents.coding_sub_agent.prompts import CODE_GENERATION_SYSTEM_PROMPT, CODEGEN_FAILURE_SYSTEM_PROMPT
# DISABLED: LLM judges — CODE_JUDGE_SYSTEM_PROMPT, IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT
from sub_agents.coding_sub_agent.code_scan.semgrep_scan import run_semgrep_scan
from sub_agents.coding_sub_agent.validation import CodeGenFailureOutput, CodeGenerationOutput, sanitize_run_id
# DISABLED: LLM judges — JudgeOutput

# Parent trace snapshot for nested invokes (coding_tool → run); LangGraph drops OTel context.
coding_trace_ctx: ContextVar[TraceContext | None] = ContextVar("coding_trace_ctx", default=None)


def get_coding_trace_context() -> TraceContext | None:
    """Return the trace context set for the current coding_tool invoke."""
    return coding_trace_ctx.get()


PIPELINE_NODE_ERROR_MESSAGE = (
    "There were some errors executing a pipeline node. "
    "Please retry again with updated code requirements."
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CodeExecutionResult(TypedDict):
    """Captured E2B run outcome."""

    stdout: str
    stderr: str
    copied_outputs: list[str]
    plots: list[str]


class CodingAgentState(TypedDict):
    """State for the coding pipeline sub-graph."""

    requirements: str  # task text from coding_tool
    input_files: list[str]  # CSV/XLSX basenames to upload into the sandbox
    output_files: list[str]  # expected output basenames (tables + plots)
    session_id: str
    tool_call_id: str
    data_profile: list[Any]  # session workspace profile from main graph (CodeGen context)
    codegen_count: NotRequired[int]  # 1-based attempts completed by CodeGen; CodeGenLimitGate reads before next run
    code: str
    semgrep_feedback: str  # set by SemgrepScan; cleared after a successful CodeGen
    judge_feedback: str  # set by SafetyJudge; cleared after a successful CodeGen
    io_feedback: str  # set by IOAllowlistJudge; cleared after a successful CodeGen
    e2b_feedback: str  # set by E2BExecute on failure; cleared after a successful CodeGen
    codegen_failure_feedback: NotRequired[str]  # set by CodeGenFailure when retries exhausted
    code_execution_result: NotRequired[CodeExecutionResult]
    e2b_execution_status: NotRequired[Literal["success", "failed"]]
    graph_failure: NotRequired[dict[str, Any]]  # node RetryPolicy exhaustion from handle_node_failure


# ---------------------------------------------------------------------------
# Node retry exhaustion (transient failures)
# ---------------------------------------------------------------------------


def handle_node_failure(state: CodingAgentState, error: NodeError) -> Command:
    """Route retry-exhausted node failures to ``PipelineNodeError``."""
    del state
    return Command(
        update={"graph_failure": {"failed_node": error.node, "detail": str(error.error)}},
        goto="PipelineNodeError",
    )


def pipeline_node_error_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Emit ``code_execution_result`` for the orchestrator after node retries are exhausted."""
    gf = state.get("graph_failure") or {}
    failed_node = gf.get("failed_node") or "unknown"
    detail = gf.get("detail") or "Unknown error"
    msg = f"{PIPELINE_NODE_ERROR_MESSAGE} (failed_node={failed_node}; detail={detail})"
    exec_result: CodeExecutionResult = {
        "stdout": "",
        "stderr": msg,
        "copied_outputs": [],
        "plots": [],
    }
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    with traced_span("PipelineNodeError", trace_context=ctx) as span:
        if span is not None:
            span.update(output={"failed_node": failed_node, "graph_failure": gf, "message_preview": msg[:200]})
    return {
        "code_execution_result": exec_result,
        "e2b_execution_status": "failed",
    }


# ---------------------------------------------------------------------------
# CodeGen — structured Python generation
# ---------------------------------------------------------------------------


def codegen_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate Python from requirements; on retry, prior gate feedback is included in context.

    Model selection: early attempts use ``CODING_MODEL``; the last permitted attempt
    (when this run's count equals ``MAX_CODEGEN_ATTEMPTS``) may use ``CODING_MODEL_LAST_ATTEMPT``
    if set in root ``.env``.
    """
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    codegen_count = state.get("codegen_count", 0) + 1
    if CODING_MODEL_LAST_ATTEMPT and codegen_count == MAX_CODEGEN_ATTEMPTS:
        model = CODING_MODEL_LAST_ATTEMPT
    else:
        model = CODING_MODEL
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(CodeGenerationOutput)

    context_content = (
        f"## Requirements\n{state['requirements']}\n\n"
        f"## Input files\n{json.dumps(state['input_files'], ensure_ascii=False)}\n\n"
        f"## Output files\n{json.dumps(state['output_files'], ensure_ascii=False)}\n\n"
        f"## Session workspace (data_profile)\n"
        f"{json.dumps(state['data_profile'], indent=2, ensure_ascii=False, default=str)}"
    )
    semgrep_fb = (state["semgrep_feedback"] or "").strip()
    if semgrep_fb:
        context_content += (
            "\n\n## Semgrep failure feedback (previous generated code)\n"
            "The code below failed the static Semgrep scan. Rewrite the script to fix "
            "every violation; do not repeat blocked patterns.\n\n"
            f"{semgrep_fb}"
        )
    e2b_fb = (state["e2b_feedback"] or "").strip()
    if e2b_fb:
        context_content += (
            "\n\n## E2B execution failure feedback (previous generated code)\n"
            "The script below ran in the sandbox but failed (non-zero exit, stderr, missing "
            "outputs, or runtime error). Rewrite the code to satisfy the requirements and "
            "produce every listed output file.\n\n"
            f"{e2b_fb}"
        )

    prompt_messages = [
        SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=context_content),
    ]
    with traced_span("CodeGen", trace_context=ctx, metadata={"codegen_count": codegen_count, "model": model}) as node_span:
        with traced_generation("CodeGen-llm", model=model, trace_context=ctx) as gen:
            parsed: CodeGenerationOutput = llm.invoke(prompt_messages, config=config)
            code = (parsed.code or "").strip()
            if gen is not None:
                gen.update(
                    model=model,
                    input=serialize_messages(prompt_messages),
                    output={"filename": parsed.filename, "code": code},
                )
            result = {
                "codegen_count": codegen_count,
                "code": code,
                "semgrep_feedback": "",
                "judge_feedback": "",
                "io_feedback": "",
                "e2b_feedback": "",
            }
            if node_span is not None:
                node_span.update(output=result)
    return result


def semgrep_scan_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Static Semgrep scan."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    with traced_span("SemgrepScan", trace_context=ctx) as span:
        code = (state.get("code") or "").strip()
        if not code:
            feedback = "No code to scan."
            if span is not None:
                span.update(output={"passed": False, "semgrep_feedback": feedback, "violations": []})
            return {"semgrep_feedback": feedback}

        result = run_semgrep_scan(code)
        passed = result.passed
        detail = result.detail if not passed else ""
        violations = result.violations or []
        if span is not None:
            span.update(output={"passed": passed, "semgrep_feedback": detail, "violations": violations})
        return {"semgrep_feedback": detail}


# ---------------------------------------------------------------------------
# DISABLED: SafetyJudge — LLM policy review (re-enable with LLM judges)
# ---------------------------------------------------------------------------
#
# def safety_judge_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
#     """Ask the judge model to accept or reject the script for safety policy."""
#     from sub_agents.coding_sub_agent.config import CODE_JUDGE_MODEL
#     from sub_agents.coding_sub_agent.prompts import CODE_JUDGE_SYSTEM_PROMPT
#     from sub_agents.coding_sub_agent.validation import JudgeOutput
#     ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
#     code = (state.get("code") or "").strip()
#     llm = ChatOpenAI(model=CODE_JUDGE_MODEL, temperature=0).with_structured_output(JudgeOutput)
#     task_spec = json.dumps(
#         {"requirements": state["requirements"], "input_files": state.get("input_files") or [], "output_files": state.get("output_files") or []},
#         ensure_ascii=False,
#         indent=2,
#     )
#     system_content = CODE_JUDGE_SYSTEM_PROMPT + "\n\n## Task\n" + task_spec.strip()
#     human_content = f"```python\n{code}\n```"
#     try:
#         with traced_span("SafetyJudge", trace_context=ctx) as node_span:
#             with traced_generation("SafetyJudge-llm", model=CODE_JUDGE_MODEL, trace_context=ctx) as gen:
#                 resp: JudgeOutput = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=human_content)])
#                 if gen is not None:
#                     update_llm_generation(gen, model=CODE_JUDGE_MODEL, raw=None)
#                 passed = resp.passed
#                 if node_span is not None:
#                     node_span.update(output={"passed": passed})
#     except Exception as exc:
#         return {"judge_feedback": f"Judge model failed to produce valid structured output: {exc}"}
#     if resp.passed:
#         return {"judge_feedback": ""}
#     detail = (resp.detail or "").strip() or "LLM judge rejected the code (no reason provided)."
#     return {"judge_feedback": detail}


# ---------------------------------------------------------------------------
# DISABLED: IOAllowlistJudge — declared basename check (re-enable with LLM judges)
# ---------------------------------------------------------------------------
#
# def io_allowlist_judge_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
#     """LLM-only check that reads/writes match declared input/output basenames."""
#     from sub_agents.coding_sub_agent.config import IO_JUDGE_MODEL
#     from sub_agents.coding_sub_agent.prompts import IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT
#     from sub_agents.coding_sub_agent.validation import JudgeOutput
#     ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
#     code = (state.get("code") or "").strip()
#     llm = ChatOpenAI(model=IO_JUDGE_MODEL, temperature=0).with_structured_output(JudgeOutput)
#     spec = json.dumps({"input_files": state.get("input_files") or [], "output_files": state.get("output_files") or []}, ensure_ascii=False, indent=2)
#     system_content = IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT + "\n\n## Declared files\n" + spec
#     human_content = f"```python\n{code}\n```"
#     try:
#         with traced_span("IOAllowlistJudge", trace_context=ctx) as node_span:
#             with traced_generation("IOAllowlistJudge-llm", model=IO_JUDGE_MODEL, trace_context=ctx) as gen:
#                 resp: JudgeOutput = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=human_content)])
#                 if gen is not None:
#                     update_llm_generation(gen, model=IO_JUDGE_MODEL, raw=None)
#                 passed = resp.passed
#                 if node_span is not None:
#                     node_span.update(output={"passed": passed})
#     except Exception as exc:
#         return {"io_feedback": f"IO judge failed to produce valid structured output: {exc}"}
#     if resp.passed:
#         return {"io_feedback": ""}
#     detail = (resp.detail or "").strip() or "IO allowlist judge rejected the code."
#     return {"io_feedback": detail}


# ---------------------------------------------------------------------------
# E2BExecute — sandbox run + local copy-back (inlined)
# CSV/XLSX → session root; PNG/SVG → run_<tool_call_id>/ for API image discovery
# ---------------------------------------------------------------------------


def e2b_execute_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Run generated_code.py in E2B; one try/except maps failures to ``e2b_feedback`` (semantic retry).

    Uncaught raises (e.g. missing template before the try) still use ``RetryPolicy`` → ``PipelineNodeError``.
    """
    from e2b_code_interpreter import Sandbox

    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    if not E2B_TEMPLATE_NAME:
        raise ValueError(
            "CODING_E2B_TEMPLATE_NAME is not set. Run e2b/build_e2b_template.py and add it to "
            "root .env"
        )

    code = (state.get("code") or "").strip()
    session_id = state["session_id"]
    tool_call_id = state["tool_call_id"]
    input_files = state.get("input_files") or []
    output_files = state.get("output_files") or []

    ensure_session_dirs(session_id)
    local_root = session_root(session_id)
    run_id = sanitize_run_id(tool_call_id)
    run_workspace = local_root / f"run_{run_id}"
    run_workspace.mkdir(parents=True, exist_ok=True)

    workspace_dir = "/home/user/workspace"
    sandbox = None
    stdout = ""
    stderr = ""
    error = ""
    copied_outputs: list[str] = []
    plots: list[str] = []

    with traced_span("E2BExecute", trace_context=ctx) as span:
        try:
            sandbox = Sandbox.create(
                template=E2B_TEMPLATE_NAME,
                api_key=E2B_API_KEY or None,
                timeout=E2B_SANDBOX_TIMEOUT_SECONDS,
                allow_internet_access=False,
                lifecycle={"on_timeout": "pause"},
            )
            sandbox.commands.run(f"mkdir -p {workspace_dir}")

            for basename in input_files:
                local_path = local_root / basename
                remote_path = f"{workspace_dir}/{basename}"
                with open(local_path, "rb") as handle:
                    sandbox.files.write(remote_path, handle.read())

            script_path = f"{workspace_dir}/generated_code.py"
            sandbox.files.write(script_path, code)

            cmd = f"cd {workspace_dir} && python generated_code.py"
            result = sandbox.commands.run(cmd, timeout=E2B_EXECUTION_TIMEOUT_SECONDS)
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""

            for basename in output_files:
                remote_path = f"{workspace_dir}/{basename}"
                ext = Path(basename).suffix.lower()
                data = sandbox.files.read(remote_path, format="bytes")
                if ext in PLOT_FILE_EXTENSIONS:
                    dest = run_workspace / basename
                    dest.write_bytes(data)
                    sid = session_dir_for_paths(session_id)
                    plots.append(f"agent_filesystem/{sid}/run_{run_id}/{basename}")
                else:
                    dest = local_root / basename
                    dest.write_bytes(data)
                copied_outputs.append(basename)
        except Exception as exc:
            error = str(exc)
        finally:
            if sandbox is not None and E2B_KILL_SANDBOX:
                try:
                    sandbox.kill()
                except Exception:
                    pass

        e2b_succeeded = not error.strip() and not stderr.strip()
        result = {
            "code_execution_result": {
                "stdout": stdout,
                "stderr": stderr,
                "copied_outputs": copied_outputs,
                "plots": plots,
            },
            "e2b_feedback": error.strip() or stderr.strip(),
            "e2b_execution_status": "success" if e2b_succeeded else "failed",
        }
        if span is not None:
            span.update(output=result)
    return result


# ---------------------------------------------------------------------------
# CodeGenLimitGate — passthrough; routing in route_codegen_limit_gate
# ---------------------------------------------------------------------------


def codegen_limit_gate_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Passthrough; ``route_codegen_limit_gate`` enforces ``MAX_CODEGEN_ATTEMPTS`` before CodeGen."""
    del state, config
    return {}


# ---------------------------------------------------------------------------
# CodeGenFailure — LLM summary when codegen retries are exhausted
# ---------------------------------------------------------------------------


def codegen_failure_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize pipeline failures for the orchestrator when codegen_count reaches the cap."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    llm = ChatOpenAI(model=CODING_MODEL, temperature=0).with_structured_output(CodeGenFailureOutput)
    summary = {
        "requirements": state["requirements"],
        "input_files": state.get("input_files") or [],
        "output_files": state.get("output_files") or [],
        "codegen_count": state.get("codegen_count", 0),
        "semgrep_feedback": state.get("semgrep_feedback") or "",
        "judge_feedback": state.get("judge_feedback") or "",
        "io_feedback": state.get("io_feedback") or "",
        "e2b_feedback": state.get("e2b_feedback") or "",
        "code": state.get("code") or "",
    }
    human_content = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    try:
        with traced_span("CodeGenFailure", trace_context=ctx) as node_span:
            with traced_generation("CodeGenFailure-llm", model=CODING_MODEL, trace_context=ctx) as gen:
                parsed: CodeGenFailureOutput = llm.invoke(
                    [SystemMessage(content=CODEGEN_FAILURE_SYSTEM_PROMPT), HumanMessage(content=human_content)],
                    config=config,
                )
                if gen is not None:
                    gen.update(model=CODING_MODEL, output={"codegen_failure_feedback": parsed.codegen_failure_feedback})
                feedback = parsed.codegen_failure_feedback.strip() or f"Coding pipeline exhausted {MAX_CODEGEN_ATTEMPTS} codegen attempts."
                result = {"codegen_failure_feedback": feedback}
                if node_span is not None:
                    node_span.update(output=result)
        return result
    except Exception as exc:
        return {
            "codegen_failure_feedback": (
                f"Coding pipeline exhausted {MAX_CODEGEN_ATTEMPTS} codegen attempts. "
                f"Failure summary unavailable: {exc}"
            ),
        }


# ---------------------------------------------------------------------------
# Gate routing — failures retry via CodeGenLimitGate (max MAX_CODEGEN_ATTEMPTS CodeGen runs)
# ---------------------------------------------------------------------------


def route_codegen_limit_gate(state: CodingAgentState) -> str:
    """Send to CodeGenFailure once ``codegen_count`` reaches ``MAX_CODEGEN_ATTEMPTS``.

    Count is set by the previous CodeGen run (1-based after each visit). Example with MAX=3:
    state 0,1,2 → CodeGen; state 3 → CodeGenFailure (no further LLM codegen).
    """
    if int(state.get("codegen_count") or 0) >= MAX_CODEGEN_ATTEMPTS:
        return "CodeGenFailure"
    return "CodeGen"


def route_after_semgrep(state: CodingAgentState) -> str:
    if (state.get("semgrep_feedback") or "").strip():
        return "CodeGenLimitGate"
    return "E2BExecute"


# DISABLED: LLM judges — re-enable with SafetyJudge / IOAllowlistJudge nodes
#
# def route_after_safety_judge(state: CodingAgentState) -> str:
#     if (state.get("judge_feedback") or "").strip():
#         return "CodeGenLimitGate"
#     return "IOAllowlistJudge"
#
#
# def route_after_io_judge(state: CodingAgentState) -> str:
#     if (state.get("io_feedback") or "").strip():
#         return "CodeGenLimitGate"
#     return "E2BExecute"


def route_after_e2b(state: CodingAgentState) -> str:
    if state.get("e2b_execution_status") == "success":
        return END
    return "CodeGenLimitGate"


def build_coding_tool_response(result: dict[str, Any]) -> dict[str, Any]:
    """Map sub-graph state to the JSON body returned by ``coding_tool``."""
    codegen_failure = (result.get("codegen_failure_feedback") or "").strip()
    e2b_ok = result.get("e2b_execution_status") == "success"
    tool_success = e2b_ok and not codegen_failure
    exec_result = result.get("code_execution_result")
    code_violation: dict[str, str] | None = None
    if not tool_success:
        viol: dict[str, str] = {}
        if (result.get("semgrep_feedback") or "").strip():
            viol["semgrep"] = result["semgrep_feedback"]
        if (result.get("e2b_feedback") or "").strip():
            viol["e2b"] = result["e2b_feedback"]
        if codegen_failure:
            viol["codegen_failure"] = codegen_failure
        gf = result.get("graph_failure") or {}
        if gf.get("failed_node"):
            viol["node_error"] = f"{gf.get('failed_node')}: {gf.get('detail', '')}"
        if not viol and not (result.get("code") or "").strip():
            viol["codegen"] = "No code produced."
        code_violation = viol or {"pipeline": "Coding pipeline failed."}

    stdout = (exec_result or {}).get("stdout") or ""
    stderr = (exec_result or {}).get("stderr") or ""
    body: dict[str, Any] = {
        "status": "success" if tool_success else "failed",
        "stdout": stdout,
        "stderr": stderr,
        "code_violation": code_violation if not tool_success else None,
        "outputs": list((exec_result or {}).get("copied_outputs") or []),
        "plots": list((exec_result or {}).get("plots") or []),
    }
    if exec_result is not None:
        body["code_execution_result"] = exec_result
    if not tool_success and (result.get("code") or "").strip():
        body["code"] = result["code"]
    return body


class CodingGraph:
    """Wrapper around the coding pipeline LangGraph (invoked by ``coding_tool``)."""

    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.graph = self.build_graph()

    def build_graph(self) -> Any:
        """Construct and compile the coding pipeline ``StateGraph`` (in-memory checkpointer for ``get_state`` on limit)."""
        builder = StateGraph(CodingAgentState)
        node_retry = RetryPolicy(
            max_attempts=CODING_NODE_RETRY_MAX_ATTEMPTS,
            initial_interval=CODING_NODE_RETRY_INITIAL_INTERVAL,
            backoff_factor=CODING_NODE_RETRY_BACKOFF_FACTOR,
        )

        builder.add_node("CodeGenLimitGate", codegen_limit_gate_node)
        builder.add_node("CodeGen", codegen_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("SemgrepScan", semgrep_scan_node, retry_policy=node_retry, error_handler=handle_node_failure)
        # DISABLED: LLM judges — builder.add_node("SafetyJudge", safety_judge_node)
        # DISABLED: LLM judges — builder.add_node("IOAllowlistJudge", io_allowlist_judge_node)
        builder.add_node("E2BExecute", e2b_execute_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("CodeGenFailure", codegen_failure_node)
        builder.add_node("PipelineNodeError", pipeline_node_error_node)

        builder.set_entry_point("CodeGenLimitGate")
        builder.add_conditional_edges(
            "CodeGenLimitGate",
            route_codegen_limit_gate,
            {"CodeGen": "CodeGen", "CodeGenFailure": "CodeGenFailure"},
        )
        builder.add_edge("CodeGen", "SemgrepScan")
        builder.add_conditional_edges(
            "SemgrepScan",
            route_after_semgrep,
            {"E2BExecute": "E2BExecute", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        # DISABLED: LLM judges — SafetyJudge / IOAllowlistJudge conditional edges
        builder.add_conditional_edges(
            "E2BExecute",
            route_after_e2b,
            {END: END, "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_edge("CodeGenFailure", END)
        builder.add_edge("PipelineNodeError", END)
        return builder.compile(checkpointer=self.checkpointer)

    def run(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        requirements: str,
        input_files: list[str],
        output_files: list[str],
        data_profile: list,
        trace_context: TraceContext | None = None,
    ) -> dict[str, Any]:
        """Run the coding sub-graph and return a tool JSON-shaped dict.

        One checkpoint thread per ``tool_call_id``; in-memory checkpointer supports ``get_state`` on recursion limit.
        """
        initial: CodingAgentState = {
            "requirements": requirements,
            "input_files": input_files,
            "output_files": output_files,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "data_profile": data_profile,
            "codegen_count": 0,
            "code": "",
            "semgrep_feedback": "",
            "judge_feedback": "",
            "io_feedback": "",
            "e2b_feedback": "",
            "codegen_failure_feedback": "",
            "graph_failure": {},
        }
        safe_id = (tool_call_id or "unknown").replace("/", "_")
        config = {
            "configurable": {"thread_id": f"coding_sub_agent_{session_id}_{safe_id}"},
            "recursion_limit": CODING_GRAPH_RECURSION_LIMIT,
        }
        meta = {"session_id": session_id, "tool_call_id": tool_call_id}
        token = coding_trace_ctx.set(trace_context)  # nodes read via get_coding_trace_context()
        try:
            with traced_span("coding_pipeline", trace_context=trace_context, metadata=meta):
                # LangGraph may execute coding nodes outside the current OTel context;
                # carrying the active pipeline span in config keeps each gate nested.
                config = add_trace_context_to_config(config, current_trace_context())
                try:
                    result = self.graph.invoke(initial, config=config)
                except GraphRecursionError:
                    # Safety net: return last checkpoint so coding_tool gets JSON instead of crashing.
                    snap = self.graph.get_state(config)
                    result = dict(snap.values) if snap and snap.values else dict(initial)
                    if not (result.get("codegen_failure_feedback") or "").strip():
                        result["codegen_failure_feedback"] = (
                            f"Coding pipeline hit recursion limit ({CODING_GRAPH_RECURSION_LIMIT})."
                        )
        finally:
            safe_reset_contextvar(coding_trace_ctx, token)

        return build_coding_tool_response(result)
