"""
Coding sub-graph: CodeGenLimitGate → CodeGen → SemgrepScan → IOAllowlistScan → InputFilesCheck
→ E2BExecute; all terminal paths → PrepareResponse → END.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, NotRequired, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse.types import TraceContext
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, RetryPolicy
from session_paths import ensure_session_dirs, logical_input_file, session_dir_for_paths, session_root

from observability.langfuse_handler import (
    add_trace_context_to_config,
    current_trace_context,
    safe_reset_contextvar,
    serialize_messages,
    trace_context_from_runnable_config,
    traced_generation,
    traced_span,
)
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
from sub_agents.coding_sub_agent.prompts import CODE_GENERATION_SYSTEM_PROMPT
from sub_agents.coding_sub_agent.code_scan.io_allowlist_scan import run_io_allowlist_scan
from sub_agents.coding_sub_agent.code_scan.semgrep_scan import run_semgrep_scan
from sub_agents.coding_sub_agent.validation import CodeGenerationOutput, sanitize_run_id

coding_trace_ctx: ContextVar[TraceContext | None] = ContextVar("coding_trace_ctx", default=None)

PIPELINE_NODE_ERROR_MESSAGE = (
    "There were some errors executing a pipeline node. "
    "Please retry again with updated code requirements."
)


def get_coding_trace_context() -> TraceContext | None:
    """Return the trace context set for the current coding_tool invoke."""
    return coding_trace_ctx.get()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CodingAgentState(TypedDict):
    """State for the coding pipeline sub-graph."""

    requirements: str
    input_files: list[str]
    output_files: list[str]
    session_id: str
    tool_call_id: str
    data_profile: list[Any]
    codegen_count: NotRequired[int]
    code: str
    pipeline_violation: NotRequired[dict[str, Any]]
    code_execution_result: NotRequired[dict[str, Any]]
    tool_response: NotRequired[dict[str, Any]]


# ---------------------------------------------------------------------------
# Node retry exhaustion (transient failures)
# ---------------------------------------------------------------------------


def handle_node_failure(state: CodingAgentState, error: Any) -> Command:
    """Route retry-exhausted node failures to ``PrepareResponse``."""
    del state
    failed_node = getattr(error, "node", "unknown")
    detail = getattr(error, "error", error)
    msg = f"{PIPELINE_NODE_ERROR_MESSAGE} (failed_node={failed_node}; detail={detail})"
    return Command(
        update={
            "pipeline_violation": {"stage": "node_error", "message": msg},
            "code": "",
            "code_execution_result": {},
        },
        goto="PrepareResponse",
    )


# ---------------------------------------------------------------------------
# CodeGen
# ---------------------------------------------------------------------------


def codegen_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate Python from requirements; on retry, include ``pipeline_violation`` in context."""
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
    violation = state.get("pipeline_violation") or {}
    failure_msg = (violation.get("message") or "").strip()
    if failure_msg:
        stage = (violation.get("stage") or "").strip()
        context_content += (
            f"\n\n## Pipeline failure ({stage})\n"
            "The script below did not pass the previous pipeline step. "
            "Rewrite the code to fix the issue.\n\n"
            f"{failure_msg}"
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
            result = {"codegen_count": codegen_count, "code": code}
            if node_span is not None:
                node_span.update(output=result)
    return result


def semgrep_scan_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Static Semgrep scan."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    with traced_span("SemgrepScan", trace_context=ctx) as span:
        code = (state.get("code") or "").strip()
        if not code:
            detail = "No code to scan."
            if span is not None:
                span.update(output={"passed": False, "pipeline_violation": {"stage": "semgrep", "message": detail}})
            return {"pipeline_violation": {"stage": "semgrep", "message": detail}}

        result = run_semgrep_scan(code)
        if result.passed:
            if span is not None:
                span.update(output={"passed": True, "pipeline_violation": {}})
            return {"pipeline_violation": {}}

        detail = result.detail or "Semgrep rejected the generated code."
        if span is not None:
            span.update(output={"passed": False, "pipeline_violation": {"stage": "semgrep", "message": detail}})
        return {"pipeline_violation": {"stage": "semgrep", "message": detail}}


def io_allowlist_scan_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Static IO allowlist scan: declared basenames must match code reads/writes."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    with traced_span("IOAllowlistScan", trace_context=ctx) as span:
        code = (state.get("code") or "").strip()
        if not code:
            detail = "No code to scan."
            if span is not None:
                span.update(output={"passed": False, "pipeline_violation": {"stage": "io_allowlist", "message": detail}})
            return {"pipeline_violation": {"stage": "io_allowlist", "message": detail}}

        result = run_io_allowlist_scan(
            code,
            state.get("input_files") or [],
            state.get("output_files") or [],
        )
        if result.passed:
            if span is not None:
                span.update(output={"passed": True, "pipeline_violation": {}})
            return {"pipeline_violation": {}}

        detail = result.detail or "IO allowlist rejected the generated code."
        if span is not None:
            span.update(output={"passed": False, "pipeline_violation": {"stage": "io_allowlist", "message": detail}})
        return {"pipeline_violation": {"stage": "io_allowlist", "message": detail}}


def input_files_check_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Verify declared input_files exist under the session workspace before E2B."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    session_id = state["session_id"]
    input_files = state.get("input_files") or []
    with traced_span("InputFilesCheck", trace_context=ctx) as span:
        if not input_files:
            if span is not None:
                span.update(output={"passed": True, "pipeline_violation": {}})
            return {"pipeline_violation": {}}

        missing: list[str] = []
        for basename in input_files:
            name = Path(basename).name
            if not (session_root(session_id) / name).is_file():
                missing.append(name)

        if missing:
            lines = [
                "Declared input file(s) missing from session workspace:",
                "",
            ]
            for name in missing:
                lines.append(f"  - {name} (expected at {logical_input_file(session_id, name)})")
            lines.append("")
            lines.append("Upload the file(s) or fix input_files before calling coding_tool again.")
            detail = "\n".join(lines)
            if span is not None:
                span.update(output={"passed": False, "missing": missing})
            return {"pipeline_violation": {"stage": "missing_inputs", "message": detail}}

        if span is not None:
            span.update(output={"passed": True, "pipeline_violation": {}})
        return {"pipeline_violation": {}}


# ---------------------------------------------------------------------------
# E2BExecute
# ---------------------------------------------------------------------------


def e2b_execute_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Run generated_code.py in E2B; failures set ``pipeline_violation`` stage ``e2b``."""
    from e2b_code_interpreter import Sandbox

    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    if not E2B_TEMPLATE_NAME:
        raise ValueError(
            "CODING_E2B_TEMPLATE_NAME is not set. Run e2b/build_e2b_template.py and add it to root .env"
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
            run_result = sandbox.commands.run(cmd, timeout=E2B_EXECUTION_TIMEOUT_SECONDS)
            stdout = getattr(run_result, "stdout", "") or ""
            stderr = getattr(run_result, "stderr", "") or ""

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
        node_result: Dict[str, Any] = {
            "code_execution_result": {
                "stdout": stdout,
                "stderr": stderr,
                "copied_outputs": copied_outputs,
                "plots": plots,
            },
        }
        if e2b_succeeded:
            node_result["pipeline_violation"] = {}
        else:
            node_result["pipeline_violation"] = {
                "stage": "e2b",
                "message": error.strip() or stderr.strip() or "E2B execution failed.",
            }

        if span is not None:
            span.update(output=node_result)
    return node_result


# ---------------------------------------------------------------------------
# CodeGenLimitGate / terminal nodes
# ---------------------------------------------------------------------------


def codegen_limit_gate_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Passthrough; ``route_codegen_limit_gate`` enforces ``MAX_CODEGEN_ATTEMPTS`` before CodeGen."""
    del state, config
    return {}


def codegen_exhausted_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Set ``pipeline_violation`` when codegen retries are exhausted (no LLM)."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    result = {
        "pipeline_violation": {**(state.get("pipeline_violation") or {}), "stage": "codegen_exhausted"},
    }
    with traced_span("CodegenExhausted", trace_context=ctx) as span:
        if span is not None:
            span.update(output=result)
    return result


def prepare_response_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Build ``tool_response`` from state before END."""
    ctx = trace_context_from_runnable_config(config) or get_coding_trace_context()
    violation = state.get("pipeline_violation") or {}
    exec_result = state.get("code_execution_result") or {}
    tool_response = {
        "status": "success" if not violation else "failed",
        "failure": violation or None,
        "execution": {
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", ""),
        },
        "artifacts": {
            "outputs": exec_result.get("copied_outputs", []),
            "plots": exec_result.get("plots", []),
        },
        "code": state.get("code"),
    }
    result = {"tool_response": tool_response}
    with traced_span("PrepareResponse", trace_context=ctx) as span:
        if span is not None:
            span.update(output=result)
    return result


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_codegen_limit_gate(state: CodingAgentState) -> str:
    if int(state.get("codegen_count") or 0) >= MAX_CODEGEN_ATTEMPTS:
        return "CodegenExhausted"
    return "CodeGen"


def route_after_semgrep(state: CodingAgentState) -> str:
    if (state.get("pipeline_violation") or {}).get("message", "").strip():
        return "CodeGenLimitGate"
    return "IOAllowlistScan"


def route_after_io(state: CodingAgentState) -> str:
    if (state.get("pipeline_violation") or {}).get("message", "").strip():
        return "CodeGenLimitGate"
    return "InputFilesCheck"


def route_after_input_check(state: CodingAgentState) -> str:
    if (state.get("pipeline_violation") or {}).get("message", "").strip():
        return "PrepareResponse"
    return "E2BExecute"


def route_after_e2b(state: CodingAgentState) -> str:
    if (state.get("pipeline_violation") or {}).get("message", "").strip():
        return "CodeGenLimitGate"
    return "PrepareResponse"


class CodingGraph:
    """Wrapper around the coding pipeline LangGraph (invoked by ``coding_tool``)."""

    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.graph = self.build_graph()

    def build_graph(self) -> Any:
        builder = StateGraph(CodingAgentState)
        node_retry = RetryPolicy(
            max_attempts=CODING_NODE_RETRY_MAX_ATTEMPTS,
            initial_interval=CODING_NODE_RETRY_INITIAL_INTERVAL,
            backoff_factor=CODING_NODE_RETRY_BACKOFF_FACTOR,
        )

        builder.add_node("CodeGenLimitGate", codegen_limit_gate_node)
        builder.add_node("CodeGen", codegen_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("SemgrepScan", semgrep_scan_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("IOAllowlistScan", io_allowlist_scan_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("InputFilesCheck", input_files_check_node)
        builder.add_node("E2BExecute", e2b_execute_node, retry_policy=node_retry, error_handler=handle_node_failure)
        builder.add_node("PrepareResponse", prepare_response_node)
        builder.add_node("CodegenExhausted", codegen_exhausted_node)

        builder.set_entry_point("CodeGenLimitGate")
        builder.add_conditional_edges(
            "CodeGenLimitGate",
            route_codegen_limit_gate,
            {"CodeGen": "CodeGen", "CodegenExhausted": "CodegenExhausted"},
        )
        builder.add_edge("CodeGen", "SemgrepScan")
        builder.add_conditional_edges(
            "SemgrepScan",
            route_after_semgrep,
            {"IOAllowlistScan": "IOAllowlistScan", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_conditional_edges(
            "IOAllowlistScan",
            route_after_io,
            {"InputFilesCheck": "InputFilesCheck", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_conditional_edges(
            "InputFilesCheck",
            route_after_input_check,
            {"PrepareResponse": "PrepareResponse", "E2BExecute": "E2BExecute"},
        )
        builder.add_conditional_edges(
            "E2BExecute",
            route_after_e2b,
            {"PrepareResponse": "PrepareResponse", "CodeGenLimitGate": "CodeGenLimitGate"},
        )
        builder.add_edge("CodegenExhausted", "PrepareResponse")
        builder.add_edge("PrepareResponse", END)
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
        initial: CodingAgentState = {
            "requirements": requirements,
            "input_files": input_files,
            "output_files": output_files,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "data_profile": data_profile,
            "codegen_count": 0,
            "code": "",
            "pipeline_violation": {},
        }
        safe_id = (tool_call_id or "unknown").replace("/", "_")
        config = {
            "configurable": {"thread_id": f"coding_sub_agent_{session_id}_{safe_id}"},
            "recursion_limit": CODING_GRAPH_RECURSION_LIMIT,
        }
        meta = {"session_id": session_id, "tool_call_id": tool_call_id}
        token = coding_trace_ctx.set(trace_context)
        try:
            with traced_span("coding_pipeline", trace_context=trace_context, metadata=meta):
                config = add_trace_context_to_config(config, current_trace_context())
                result = self.graph.invoke(initial, config=config)
        finally:
            safe_reset_contextvar(coding_trace_ctx, token)

        return result.get("tool_response") or {}
