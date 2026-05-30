"""
Coding sub-graph: CodeGenLimitGate → CodeGen → SemgrepScan → SafetyJudge → IOAllowlistJudge → E2BExecute.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.types import TraceContext
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from session_paths import ensure_session_dirs, session_dir_for_paths, session_root

from observability.langfuse_handler import add_trace_context_to_config, current_trace_context, safe_reset_contextvar, trace_context_from_runnable_config, traced_generation, traced_span, update_llm_generation
from sub_agents.coding_sub_agent.config import CODE_JUDGE_MODEL, CODING_MODEL, CODING_RECURSION_LIMIT, E2B_API_KEY, E2B_EXECUTION_TIMEOUT_SECONDS, E2B_KILL_SANDBOX, E2B_SANDBOX_TIMEOUT_SECONDS, E2B_TEMPLATE_NAME, IO_JUDGE_MODEL, MAX_CODEGEN_ATTEMPTS, PLOT_FILE_EXTENSIONS, TABULAR_OUTPUT_EXTENSIONS
from sub_agents.coding_sub_agent.prompts import CODE_GENERATION_SYSTEM_PROMPT, CODE_JUDGE_SYSTEM_PROMPT, CODEGEN_FAILURE_SYSTEM_PROMPT, IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT
from sub_agents.coding_sub_agent.code_scan.semgrep_scan import run_semgrep_scan
from sub_agents.coding_sub_agent.validation import CodeGenFailureOutput, CodeGenerationOutput, JudgeOutput, sanitize_run_id

# Parent trace snapshot for nested invokes (coding_tool → run); LangGraph drops OTel context.
coding_trace_ctx: ContextVar[TraceContext | None] = ContextVar("coding_trace_ctx", default=None)


def get_coding_trace_context() -> TraceContext | None:
    """Return the trace context set for the current coding_tool invoke."""
    return coding_trace_ctx.get()


def parse_structured_output(raw: Any, schema: type) -> tuple[Any, AIMessage | None]:
    if isinstance(raw, dict) and "parsed" in raw:
        parsed = raw["parsed"]
        msg = raw.get("raw")
        if isinstance(msg, AIMessage):
            return parsed, msg
        return parsed, None
    if isinstance(raw, schema):
        return raw, None
    return schema.model_validate(raw), None

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CodeExecutionResult(TypedDict):
    """Captured E2B run outcome."""

    stdout: str
    stderr: str
    exit_code: int
    error: NotRequired[str]
    copied_outputs: list[str]
    missing_outputs: list[str]
    plots: list[str]


class CodingAgentState(TypedDict):
    """State for the coding pipeline sub-graph."""

    requirements: str  # task text from coding_tool
    input_files: list[str]  # CSV/XLSX basenames to upload into the sandbox
    output_files: list[str]  # expected output basenames (tables + plots)
    session_id: str
    tool_call_id: str
    data_profile: list[Any]  # session workspace profile from main graph (CodeGen context)
    codegen_count: NotRequired[int]  # incremented by CodeGen; used by CodeGenLimitGate
    code: str
    semgrep_feedback: str  # set by SemgrepScan; cleared after a successful CodeGen
    judge_feedback: str  # set by SafetyJudge; cleared after a successful CodeGen
    io_feedback: str  # set by IOAllowlistJudge; cleared after a successful CodeGen
    e2b_feedback: str  # set by E2BExecute on failure; cleared after a successful CodeGen
    codegen_failure_feedback: NotRequired[str]  # set by CodeGenFailure when retries exhausted
    code_execution_result: NotRequired[CodeExecutionResult]
    e2b_execution_status: NotRequired[Literal["success", "failed"]]
    sandbox_id: NotRequired[str]


# ---------------------------------------------------------------------------
# CodeGen — structured Python generation
# ---------------------------------------------------------------------------


def codegen_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate Python from requirements; on retry, prior gate feedback is included in context."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    llm = ChatOpenAI(model=CODING_MODEL, temperature=0).with_structured_output(CodeGenerationOutput, include_raw=True)
    codegen_count = state.get("codegen_count", 0) + 1

    # Ephemeral LLM context (mirrors planner orchestrator): task, files, workspace, then gate failures.
    context_content = (
        f"## Requirements\n{state['requirements']}\n\n"
        f"## Input files\n{json.dumps(state['input_files'], ensure_ascii=False)}\n\n"
        f"## Output files\n{json.dumps(state['output_files'], ensure_ascii=False)}\n\n"
        f"## Session workspace (data_profile)\n"
        f"{json.dumps(state['data_profile'], indent=2, ensure_ascii=False, default=str)}"
    )
    semgrep_fb = (state["semgrep_feedback"] or "").strip()
    if semgrep_fb:
        context_content += f"\n\n## Semgrep\n{semgrep_fb}"
    judge_fb = (state["judge_feedback"] or "").strip()
    if judge_fb:
        context_content += f"\n\n## Safety judge\n{judge_fb}"
    io_fb = (state["io_feedback"] or "").strip()
    if io_fb:
        context_content += f"\n\n## IO allowlist\n{io_fb}"
    e2b_fb = (state["e2b_feedback"] or "").strip()
    if e2b_fb:
        context_content += f"\n\n## E2B\n{e2b_fb}"
    # First pass: only requirements, files, and data_profile. Retries append non-empty feedback above.

    prompt_messages = [
        SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=context_content),
    ]
    try:
        with traced_span("CodeGen", trace_context=ctx) as node_span:
            with traced_generation("CodeGen-llm", model=CODING_MODEL, trace_context=ctx) as gen:
                raw = llm.invoke(prompt_messages, config=config)
                parsed, raw_msg = parse_structured_output(raw, CodeGenerationOutput)
                if gen is not None:
                    update_llm_generation(gen, model=CODING_MODEL, raw=raw_msg)
                if isinstance(parsed, dict):
                    parsed = CodeGenerationOutput.model_validate(parsed)
                code = (parsed.code or "").strip()
                # New code attempt: drop stale feedback so downstream gates only see this revision.
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
    except Exception as exc:
        return {
            "codegen_count": codegen_count,
            "code": "",
            "semgrep_feedback": f"Code generation failed: {exc}",
        }


def semgrep_scan_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Static Semgrep scan."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    code = (state.get("code") or "").strip()
    if not code:
        return {"semgrep_feedback": "No code to scan."}
    with traced_span("SemgrepScan", trace_context=ctx) as span:
        result = run_semgrep_scan(code)
        passed = result.passed
        detail = result.detail if not passed else ""
        if span is not None:
            span.update(output={"passed": passed, "finding_count": 0 if passed else 1})
    if passed:
        return {"semgrep_feedback": ""}
    return {"semgrep_feedback": detail}


# ---------------------------------------------------------------------------
# SafetyJudge — LLM policy review (inlined)
# ---------------------------------------------------------------------------


def safety_judge_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Ask the judge model to accept or reject the script for safety policy."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    code = (state.get("code") or "").strip()
    llm = ChatOpenAI(model=CODE_JUDGE_MODEL, temperature=0).with_structured_output(JudgeOutput, include_raw=True)
    task_spec = json.dumps(
        {"requirements": state["requirements"], "input_files": state.get("input_files") or [], "output_files": state.get("output_files") or []},
        ensure_ascii=False,
        indent=2,
    )
    system_content = CODE_JUDGE_SYSTEM_PROMPT + "\n\n## Task\n" + task_spec.strip()
    human_content = f"```python\n{code}\n```"
    try:
        with traced_span("SafetyJudge", trace_context=ctx) as node_span:
            with traced_generation("SafetyJudge-llm", model=CODE_JUDGE_MODEL, trace_context=ctx) as gen:
                raw = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=human_content)])
                resp, raw_msg = parse_structured_output(raw, JudgeOutput)
                if gen is not None:
                    update_llm_generation(gen, model=CODE_JUDGE_MODEL, raw=raw_msg)
                if isinstance(resp, dict):
                    resp = JudgeOutput.model_validate(resp)
                passed = resp.passed
                if node_span is not None:
                    node_span.update(output={"passed": passed})
    except Exception as exc:
        return {"judge_feedback": f"Judge model failed to produce valid structured output: {exc}"}

    if resp.passed:
        return {"judge_feedback": ""}
    detail = (resp.detail or "").strip() or "LLM judge rejected the code (no reason provided)."
    return {"judge_feedback": detail}


# ---------------------------------------------------------------------------
# IOAllowlistJudge — declared basename check (inlined)
# ---------------------------------------------------------------------------


def io_allowlist_judge_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """LLM-only check that reads/writes match declared input/output basenames."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    code = (state.get("code") or "").strip()
    llm = ChatOpenAI(model=IO_JUDGE_MODEL, temperature=0).with_structured_output(JudgeOutput, include_raw=True)
    spec = json.dumps({"input_files": state.get("input_files") or [], "output_files": state.get("output_files") or []}, ensure_ascii=False, indent=2)
    system_content = IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT + "\n\n## Declared files\n" + spec
    human_content = f"```python\n{code}\n```"
    try:
        with traced_span("IOAllowlistJudge", trace_context=ctx) as node_span:
            with traced_generation("IOAllowlistJudge-llm", model=IO_JUDGE_MODEL, trace_context=ctx) as gen:
                raw = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=human_content)])
                resp, raw_msg = parse_structured_output(raw, JudgeOutput)
                if gen is not None:
                    update_llm_generation(gen, model=IO_JUDGE_MODEL, raw=raw_msg)
                if isinstance(resp, dict):
                    resp = JudgeOutput.model_validate(resp)
                passed = resp.passed
                if node_span is not None:
                    node_span.update(output={"passed": passed})
    except Exception as exc:
        return {"io_feedback": f"IO judge failed to produce valid structured output: {exc}"}

    if resp.passed:
        return {"io_feedback": ""}
    detail = (resp.detail or "").strip() or "IO allowlist judge rejected the code."
    return {"io_feedback": detail}


# ---------------------------------------------------------------------------
# E2BExecute — sandbox run + local copy-back (inlined)
# CSV/XLSX → session root; PNG/SVG → run_<tool_call_id>/ for API image discovery
# ---------------------------------------------------------------------------


def resolve_sandbox_id(sandbox: Any) -> str:
    """E2B SDK exposes ``sandbox_id`` on current ``e2b-code-interpreter`` builds."""
    for attr in ("sandbox_id", "id"):
        value = getattr(sandbox, attr, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_command_exit_code(result: Any) -> int:
    """Read E2B command exit code; preserve ``0`` (do not use ``or`` — ``0 or -1`` is ``-1``)."""
    raw = getattr(result, "exit_code", None)
    if raw is None:
        raw = getattr(result, "exitCode", None)
    if raw is None:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def e2b_execute_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Run generated_code.py in a fresh E2B sandbox and copy outputs locally."""
    from e2b_code_interpreter import Sandbox

    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    if not E2B_TEMPLATE_NAME:
        raise ValueError(
            "E2B_TEMPLATE_NAME is not set. Run e2b/build_e2b_template.py and add it to "
            "sub_agents/coding_sub_agent/.env"
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
    sandbox_id = ""
    stdout = ""
    stderr = ""
    exit_code = -1
    error = ""
    copied_outputs: list[str] = []
    missing_outputs: list[str] = []
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
            sandbox_id = resolve_sandbox_id(sandbox)
            sandbox.commands.run(f"mkdir -p {workspace_dir}")

            for basename in input_files:
                local_path = local_root / basename
                if not local_path.is_file():
                    missing_outputs.append(basename)
                    continue
                remote_path = f"{workspace_dir}/{basename}"
                with open(local_path, "rb") as handle:
                    sandbox.files.write(remote_path, handle.read())

            script_path = f"{workspace_dir}/generated_code.py"
            sandbox.files.write(script_path, code)

            cmd = f"cd {workspace_dir} && python generated_code.py"
            result = sandbox.commands.run(cmd, timeout=E2B_EXECUTION_TIMEOUT_SECONDS)
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            exit_code = parse_command_exit_code(result)

            if output_files:
                for basename in output_files:
                    remote_path = f"{workspace_dir}/{basename}"
                    ext = Path(basename).suffix.lower()
                    try:
                        data = sandbox.files.read(remote_path, format="bytes")
                        if ext in PLOT_FILE_EXTENSIONS:
                            dest = run_workspace / basename
                            dest.write_bytes(data)
                            sid = session_dir_for_paths(session_id)
                            plots.append(f"agent_filesystem/{sid}/run_{run_id}/{basename}")
                        elif ext in TABULAR_OUTPUT_EXTENSIONS:
                            dest = local_root / basename
                            dest.write_bytes(data)
                        else:
                            dest = local_root / basename
                            dest.write_bytes(data)
                        copied_outputs.append(basename)
                    except Exception:
                        missing_outputs.append(basename)

        except Exception as exc:
            error = str(exc)
            if exit_code == -1:
                exit_code = 1
        finally:
            if sandbox is not None and E2B_KILL_SANDBOX:
                try:
                    sandbox.kill()
                    sandbox_id = ""
                except Exception:
                    pass

        if span is not None:
            span.update(
                output={
                    "exit_code": exit_code,
                    "plot_count": len(plots),
                    "copied_outputs": copied_outputs,
                    "stderr": stderr,
                    "sandbox_id": sandbox_id or None,
                    "sandbox_killed": E2B_KILL_SANDBOX,
                    "error": error or None,
                }
            )

    exec_result: CodeExecutionResult = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "copied_outputs": copied_outputs,
        "missing_outputs": missing_outputs,
        "plots": plots,
    }
    if error:
        exec_result["error"] = error

    stderr_stripped = stderr.strip()
    err_stripped = error.strip()
    if exit_code == 0 and not err_stripped and not stderr_stripped and not missing_outputs:
        return {
            "code_execution_result": exec_result,
            "e2b_feedback": "",
            "e2b_execution_status": "success",
            "sandbox_id": sandbox_id,
        }

    parts: list[str] = []
    if err_stripped:
        parts.append(err_stripped)
    if stderr_stripped:
        parts.append(stderr_stripped)
    if exit_code not in (0, None):
        parts.append(f"exit_code={exit_code}")
    if missing_outputs:
        parts.append(f"missing outputs: {', '.join(missing_outputs)}")
    feedback = "\n".join(parts) or "Execution failed."
    return {
        "code_execution_result": exec_result,
        "e2b_feedback": feedback,
        "e2b_execution_status": "failed",
        "sandbox_id": sandbox_id,
    }


# ---------------------------------------------------------------------------
# CodeGenLimitGate — passthrough; routing in route_codegen_limit_gate
# ---------------------------------------------------------------------------


def codegen_limit_gate_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Passthrough before CodeGen; limit check in route_codegen_limit_gate."""
    del state, config
    return {}


# ---------------------------------------------------------------------------
# CodeGenFailure — LLM summary when codegen retries are exhausted
# ---------------------------------------------------------------------------


def codegen_failure_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize pipeline failures for the orchestrator when codegen_count reaches the cap."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    llm = ChatOpenAI(model=CODING_MODEL, temperature=0).with_structured_output(
        CodeGenFailureOutput, include_raw=True
    )
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
                raw = llm.invoke(
                    [SystemMessage(content=CODEGEN_FAILURE_SYSTEM_PROMPT), HumanMessage(content=human_content)],
                    config=config,
                )
                parsed, raw_msg = parse_structured_output(raw, CodeGenFailureOutput)
                if gen is not None:
                    update_llm_generation(gen, model=CODING_MODEL, raw=raw_msg)
                if isinstance(parsed, dict):
                    parsed = CodeGenFailureOutput.model_validate(parsed)
                feedback = (parsed.codegen_failure_feedback or "").strip() or (
                    f"Coding pipeline exhausted {MAX_CODEGEN_ATTEMPTS} codegen attempts."
                )
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
    if state.get("codegen_count", 0) >= MAX_CODEGEN_ATTEMPTS:
        return "CodeGenFailure"
    return "CodeGen"


def route_after_semgrep(state: CodingAgentState) -> str:
    if (state.get("semgrep_feedback") or "").strip():
        return "CodeGenLimitGate"
    return "SafetyJudge"


def route_after_safety_judge(state: CodingAgentState) -> str:
    if (state.get("judge_feedback") or "").strip():
        return "CodeGenLimitGate"
    return "IOAllowlistJudge"


def route_after_io_judge(state: CodingAgentState) -> str:
    if (state.get("io_feedback") or "").strip():
        return "CodeGenLimitGate"
    return "E2BExecute"


def route_after_e2b(state: CodingAgentState) -> str:
    if state.get("e2b_execution_status") == "success":
        return END
    return "CodeGenLimitGate"


def build_coding_tool_response(result: dict[str, Any]) -> dict[str, Any]:
    """Map sub-graph state to the JSON body returned by ``coding_tool``."""
    codegen_failure = (result.get("codegen_failure_feedback") or "").strip()
    e2b_ok = result.get("e2b_execution_status") == "success"
    tool_success = e2b_ok and not codegen_failure
    exec_result = result.get("code_execution_result") or {}
    code_violation: dict[str, str] | None = None
    if not tool_success:
        viol: dict[str, str] = {}
        if (result.get("semgrep_feedback") or "").strip():
            viol["semgrep"] = result["semgrep_feedback"]
        if (result.get("judge_feedback") or "").strip():
            viol["judge"] = result["judge_feedback"]
        if (result.get("io_feedback") or "").strip():
            viol["io"] = result["io_feedback"]
        if (result.get("e2b_feedback") or "").strip():
            viol["e2b"] = result["e2b_feedback"]
        if codegen_failure:
            viol["codegen_failure"] = codegen_failure
        if not viol and not (result.get("code") or "").strip():
            viol["codegen"] = "No code produced."
        code_violation = viol or {"pipeline": "Coding pipeline failed."}

    body: dict[str, Any] = {
        "status": "success" if tool_success else "failed",
        "stdout": exec_result.get("stdout") or "",
        "stderr": exec_result.get("stderr") or "",
        "code_violation": code_violation if not tool_success else None,
        "outputs": list(exec_result.get("copied_outputs") or []),
        "plots": list(exec_result.get("plots") or []),
    }
    if not tool_success and (result.get("code") or "").strip():
        body["code"] = result["code"]
    if not tool_success and exec_result:
        body["code_execution_result"] = exec_result
    return body


class CodingGraph:
    """Wrapper around the coding pipeline LangGraph (invoked by ``coding_tool``)."""

    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.graph = self.build_graph()

    def build_graph(self) -> Any:
        """Construct and compile the coding pipeline ``StateGraph`` (in-memory checkpointer for ``get_state`` on limit)."""
        builder = StateGraph(CodingAgentState)

        builder.add_node("CodeGenLimitGate", codegen_limit_gate_node)
        builder.add_node("CodeGen", codegen_node)
        builder.add_node("SemgrepScan", semgrep_scan_node)
        builder.add_node("SafetyJudge", safety_judge_node)
        builder.add_node("IOAllowlistJudge", io_allowlist_judge_node)
        builder.add_node("E2BExecute", e2b_execute_node)
        builder.add_node("CodeGenFailure", codegen_failure_node)

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
            {"SafetyJudge": "SafetyJudge", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_conditional_edges(
            "SafetyJudge",
            route_after_safety_judge,
            {"IOAllowlistJudge": "IOAllowlistJudge", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_conditional_edges(
            "IOAllowlistJudge",
            route_after_io_judge,
            {"E2BExecute": "E2BExecute", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_conditional_edges(
            "E2BExecute",
            route_after_e2b,
            {END: END, "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_edge("CodeGenFailure", END)
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
        }
        safe_id = (tool_call_id or "unknown").replace("/", "_")
        config = {
            "configurable": {"thread_id": f"coding_sub_agent_{session_id}_{safe_id}"},
            "recursion_limit": CODING_RECURSION_LIMIT,
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
                            f"Coding pipeline hit recursion limit ({CODING_RECURSION_LIMIT})."
                        )
        finally:
            safe_reset_contextvar(coding_trace_ctx, token)

        return build_coding_tool_response(result)
