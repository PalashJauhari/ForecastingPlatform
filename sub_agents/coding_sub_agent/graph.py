"""
Coding sub-graph: CodeGen → SemgrepScan → SafetyJudge → IOAllowlistJudge → E2BExecute.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.types import TraceContext
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph, add_messages
from session_paths import ensure_session_dirs, session_dir_for_paths, session_root

from observability.langfuse_handler import (
    safe_reset_contextvar,
    traced_generation,
    traced_span,
    truncate_preview,
    update_llm_generation,
)

from sub_agents.coding_sub_agent.config import (
    CODE_JUDGE_MODEL,
    CODING_MODEL,
    CODING_RECURSION_LIMIT,
    E2B_API_KEY,
    E2B_EXECUTION_TIMEOUT_SECONDS,
    E2B_KILL_SANDBOX,
    E2B_SANDBOX_TIMEOUT_SECONDS,
    E2B_TEMPLATE_NAME,
    IO_JUDGE_MODEL,
    PLOT_FILE_EXTENSIONS,
    TABULAR_OUTPUT_EXTENSIONS,
)
from sub_agents.coding_sub_agent.prompts import (
    CODE_GENERATION_SYSTEM_PROMPT,
    CODE_JUDGE_SYSTEM_PROMPT,
    IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT,
)
from sub_agents.coding_sub_agent.code_scan.semgrep_scan import run_semgrep_scan
from sub_agents.coding_sub_agent.validation import CodeGenerationOutput, JudgeOutput, sanitize_run_id

coding_trace_ctx: ContextVar[TraceContext | None] = ContextVar("coding_trace_ctx", default=None)


def get_coding_trace_context() -> TraceContext | None:
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

    requirements: str
    input_files: list[str]
    output_files: list[str]
    session_id: str
    tool_call_id: str
    messages: Annotated[list, add_messages]
    code: str
    explanation: NotRequired[str]
    semgrep_feedback: str
    judge_feedback: str
    io_feedback: str
    execution_feedback: str
    code_execution_result: NotRequired[CodeExecutionResult]
    status: NotRequired[Literal["success", "failed"]]
    sandbox_id: NotRequired[str]


def build_feedback_block(state: CodingAgentState) -> str:
    """Combine prior gate failures for codegen retry."""
    parts: list[str] = []
    for label, key in (
        ("Semgrep", "semgrep_feedback"),
        ("Safety judge", "judge_feedback"),
        ("IO allowlist", "io_feedback"),
        ("Execution", "execution_feedback"),
    ):
        text = (state.get(key) or "").strip()
        if text:
            parts.append(f"## {label}\n{text}")
    return "\n\n".join(parts)


def codegen_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """LLM structured codegen with requirements, file lists, and accumulated feedback."""
    ctx = get_coding_trace_context()
    llm = ChatOpenAI(model=CODING_MODEL, temperature=0).with_structured_output(CodeGenerationOutput, include_raw=True)
    feedback = build_feedback_block(state)
    blocks = [
        f"## Requirements\n{state['requirements']}",
        f"## Input files\n{json.dumps(state.get('input_files') or [], ensure_ascii=False)}",
        f"## Output files\n{json.dumps(state.get('output_files') or [], ensure_ascii=False)}",
    ]
    if feedback:
        blocks.append(feedback)
    if state.get("messages"):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "data_profile" in content:
                    blocks.append(content)
                    break
    user_payload = "\n\n".join(blocks)
    prompt_messages = [SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT), HumanMessage(content=user_payload)]
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
                if node_span is not None:
                    node_span.update(output={"code_preview": truncate_preview(code), "code_len": len(code)})
        return {
            "code": code,
            "explanation": parsed.explanation or "",
            "semgrep_feedback": "",
            "judge_feedback": "",
            "io_feedback": "",
            "execution_feedback": "",
        }
    except Exception as exc:
        return {"code": "", "explanation": "", "semgrep_feedback": f"Code generation failed: {exc}", "status": "failed"}


def semgrep_scan_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Static Semgrep scan."""
    del config
    ctx = get_coding_trace_context()
    code = (state.get("code") or "").strip()
    if not code:
        return {"semgrep_feedback": "No code to scan.", "status": "failed"}
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
    del config
    ctx = get_coding_trace_context()
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
    del config
    ctx = get_coding_trace_context()
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
    del config
    from e2b_code_interpreter import Sandbox

    ctx = get_coding_trace_context()
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
                    "stderr_preview": truncate_preview(stderr),
                    "sandbox_id": sandbox_id or None,
                    "sandbox_killed": E2B_KILL_SANDBOX,
                    "error": truncate_preview(error) if error else None,
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
            "execution_feedback": "",
            "status": "success",
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
        "execution_feedback": feedback,
        "status": "failed",
        "sandbox_id": sandbox_id,
    }


# ---------------------------------------------------------------------------
# Gate routing — any failure retries CodeGen until CODING_RECURSION_LIMIT
# (~6 graph steps per full pass: CodeGen → … → E2B → retry)
# ---------------------------------------------------------------------------


def route_after_semgrep(state: CodingAgentState) -> str:
    if (state.get("semgrep_feedback") or "").strip():
        return "CodeGen"
    return "SafetyJudge"


def route_after_safety_judge(state: CodingAgentState) -> str:
    if (state.get("judge_feedback") or "").strip():
        return "CodeGen"
    return "IOAllowlistJudge"


def route_after_io_judge(state: CodingAgentState) -> str:
    if (state.get("io_feedback") or "").strip():
        return "CodeGen"
    return "E2BExecute"


def route_after_e2b(state: CodingAgentState) -> str:
    if state.get("status") == "success":
        return END
    return "CodeGen"


def build_coding_tool_response(result: dict[str, Any]) -> dict[str, Any]:
    """Map sub-graph state to the JSON body returned by ``coding_tool``."""
    status = result.get("status") or "failed"
    exec_result = result.get("code_execution_result") or {}
    code_violation: dict[str, str] | None = None
    if status != "success":
        viol: dict[str, str] = {}
        if (result.get("semgrep_feedback") or "").strip():
            viol["semgrep"] = result["semgrep_feedback"]
        if (result.get("judge_feedback") or "").strip():
            viol["judge"] = result["judge_feedback"]
        if (result.get("io_feedback") or "").strip():
            viol["io"] = result["io_feedback"]
        if (result.get("execution_feedback") or "").strip():
            viol["execution"] = result["execution_feedback"]
        if not viol and not (result.get("code") or "").strip():
            viol["codegen"] = "No code produced."
        code_violation = viol or {"pipeline": "Coding pipeline failed."}

    body: dict[str, Any] = {
        "status": "success" if status == "success" else "failed",
        "stdout": exec_result.get("stdout") or "",
        "stderr": exec_result.get("stderr") or "",
        "code_violation": code_violation if status != "success" else None,
        "outputs": list(exec_result.get("copied_outputs") or []),
        "plots": list(exec_result.get("plots") or []),
    }
    if status != "success" and (result.get("code") or "").strip():
        body["code"] = result["code"]
    if status != "success" and exec_result:
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

        builder.add_node("CodeGen", codegen_node)
        builder.add_node("SemgrepScan", semgrep_scan_node)
        builder.add_node("SafetyJudge", safety_judge_node)
        builder.add_node("IOAllowlistJudge", io_allowlist_judge_node)
        builder.add_node("E2BExecute", e2b_execute_node)

        builder.set_entry_point("CodeGen")
        builder.add_edge("CodeGen", "SemgrepScan")
        builder.add_conditional_edges(
            "SemgrepScan",
            route_after_semgrep,
            {"SafetyJudge": "SafetyJudge", "CodeGen": "CodeGen"},
        )
        builder.add_conditional_edges(
            "SafetyJudge",
            route_after_safety_judge,
            {"IOAllowlistJudge": "IOAllowlistJudge", "CodeGen": "CodeGen"},
        )
        builder.add_conditional_edges(
            "IOAllowlistJudge",
            route_after_io_judge,
            {"E2BExecute": "E2BExecute", "CodeGen": "CodeGen"},
        )
        builder.add_conditional_edges(
            "E2BExecute",
            route_after_e2b,
            {END: END, "CodeGen": "CodeGen"},
        )
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def coding_thread_id(session_id: str, tool_call_id: str) -> str:
        """One checkpoint thread per ``coding_tool`` call (isolates retries from prior calls)."""
        safe_id = (tool_call_id or "unknown").replace("/", "_")
        return f"coding_sub_agent_{session_id}_{safe_id}"

    def run_config(self, session_id: str, tool_call_id: str) -> Dict[str, Any]:
        return {
            "configurable": {"thread_id": self.coding_thread_id(session_id, tool_call_id)},
            "recursion_limit": CODING_RECURSION_LIMIT,
        }

    def invoke_pipeline(
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

        Separate ``thread_id`` per ``tool_call_id``; in-memory checkpointer supports ``get_state`` on recursion limit.
        """
        profile_block = (
            "## Session workspace (data_profile)\n"
            f"{json.dumps(data_profile, indent=2, ensure_ascii=False, default=str)}"
        )
        initial: CodingAgentState = {
            "requirements": requirements,
            "input_files": input_files,
            "output_files": output_files,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "messages": [HumanMessage(content=profile_block)],
            "code": "",
            "semgrep_feedback": "",
            "judge_feedback": "",
            "io_feedback": "",
            "execution_feedback": "",
        }
        config = self.run_config(session_id, tool_call_id)
        meta = {"session_id": session_id, "tool_call_id": tool_call_id}
        token = coding_trace_ctx.set(trace_context)
        try:
            with traced_span("coding_pipeline", trace_context=trace_context, metadata=meta):
                try:
                    result = self.graph.invoke(initial, config=config)
                except GraphRecursionError:
                    snap = self.graph.get_state(config)
                    result = dict(snap.values) if snap and snap.values else dict(initial)
                    if result.get("status") != "success":
                        result.setdefault("execution_feedback", "")
                        if not (result.get("execution_feedback") or "").strip():
                            result["execution_feedback"] = (
                                f"Coding pipeline hit recursion limit ({CODING_RECURSION_LIMIT})."
                            )
        finally:
            safe_reset_contextvar(coding_trace_ctx, token)

        return build_coding_tool_response(result)


coding_graph_instance: CodingGraph | None = None


def get_coding_graph() -> Any:
    """Return the compiled coding subgraph (lazy singleton)."""
    global coding_graph_instance
    if coding_graph_instance is None:
        coding_graph_instance = CodingGraph()
    return coding_graph_instance.graph


get_graph = get_coding_graph


def invoke_coding_pipeline(
    *,
    session_id: str,
    tool_call_id: str,
    requirements: str,
    input_files: list[str],
    output_files: list[str],
    data_profile: list,
    trace_context=None,
) -> dict[str, Any]:
    """Backward-compatible entrypoint for ``coding_tool``."""
    global coding_graph_instance
    if coding_graph_instance is None:
        coding_graph_instance = CodingGraph()
    return coding_graph_instance.invoke_pipeline(
        session_id=session_id,
        tool_call_id=tool_call_id,
        requirements=requirements,
        input_files=input_files,
        output_files=output_files,
        data_profile=data_profile,
        trace_context=trace_context,
    )
