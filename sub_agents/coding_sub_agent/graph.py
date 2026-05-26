"""
Coding sub-graph: CodeGen → SemgrepScan → SafetyJudge → IOAllowlistJudge → E2BExecute.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, NotRequired, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph, add_messages
from session_paths import ensure_session_dirs, session_dir_for_paths, session_root

from sub_agents.coding_sub_agent.config import (
    CODE_JUDGE_MODEL,
    CODING_MODEL,
    CODING_RECURSION_LIMIT,
    E2B_API_KEY,
    E2B_EXECUTION_TIMEOUT_SECONDS,
    E2B_SANDBOX_TIMEOUT_SECONDS,
    IO_JUDGE_MODEL,
    PLOT_FILE_EXTENSIONS,
    TABULAR_OUTPUT_EXTENSIONS,
)
from sub_agents.coding_sub_agent.prompts import (
    CODE_GENERATION_SYSTEM_PROMPT,
    CODE_JUDGE_SYSTEM_PROMPT,
    IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT,
)
from sub_agents.coding_sub_agent.semgrep_scan import run_semgrep_scan
from sub_agents.coding_sub_agent.validation import CodeGenerationOutput, JudgeOutput, sanitize_run_id

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


checkpointer = InMemorySaver()
compiled_graph = None


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
    llm = ChatOpenAI(model=CODING_MODEL, temperature=0).with_structured_output(CodeGenerationOutput)
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
    prompt_messages = [
        SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_payload),
    ]
    try:
        resp = llm.invoke(prompt_messages, config=config)
        if isinstance(resp, dict):
            resp = CodeGenerationOutput.model_validate(resp)
        code = (resp.code or "").strip()
        return {
            "code": code,
            "explanation": resp.explanation or "",
            "semgrep_feedback": "",
            "judge_feedback": "",
            "io_feedback": "",
            "execution_feedback": "",
        }
    except Exception as exc:
        return {
            "code": "",
            "explanation": "",
            "semgrep_feedback": f"Code generation failed: {exc}",
            "status": "failed",
        }


def semgrep_scan_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Static Semgrep scan."""
    del config
    code = (state.get("code") or "").strip()
    if not code:
        return {"semgrep_feedback": "No code to scan.", "status": "failed"}
    result = run_semgrep_scan(code)
    if result.passed:
        return {"semgrep_feedback": ""}
    return {"semgrep_feedback": result.detail}


# ---------------------------------------------------------------------------
# SafetyJudge — LLM policy review (inlined)
# ---------------------------------------------------------------------------


def safety_judge_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Ask the judge model to accept or reject the script for safety policy."""
    del config
    code = (state.get("code") or "").strip()
    llm = ChatOpenAI(model=CODE_JUDGE_MODEL, temperature=0).with_structured_output(JudgeOutput)
    task_spec = json.dumps(
        {
            "requirements": state["requirements"],
            "input_files": state.get("input_files") or [],
            "output_files": state.get("output_files") or [],
        },
        ensure_ascii=False,
        indent=2,
    )
    system_content = CODE_JUDGE_SYSTEM_PROMPT + "\n\n## Task\n" + task_spec.strip()
    human_content = f"```python\n{code}\n```"
    try:
        resp = llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content),
            ]
        )
    except Exception as exc:
        return {"judge_feedback": f"Judge model failed to produce valid structured output: {exc}"}

    if isinstance(resp, dict):
        resp = JudgeOutput.model_validate(resp)
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
    code = (state.get("code") or "").strip()
    llm = ChatOpenAI(model=IO_JUDGE_MODEL, temperature=0).with_structured_output(JudgeOutput)
    spec = json.dumps(
        {
            "input_files": state.get("input_files") or [],
            "output_files": state.get("output_files") or [],
        },
        ensure_ascii=False,
        indent=2,
    )
    system_content = IO_ALLOWLIST_JUDGE_SYSTEM_PROMPT + "\n\n## Declared files\n" + spec
    human_content = f"```python\n{code}\n```"
    try:
        resp = llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content),
            ]
        )
    except Exception as exc:
        return {"io_feedback": f"IO judge failed to produce valid structured output: {exc}"}

    if isinstance(resp, dict):
        resp = JudgeOutput.model_validate(resp)
    if resp.passed:
        return {"io_feedback": ""}

    detail = (resp.detail or "").strip() or "IO allowlist judge rejected the code."
    return {"io_feedback": detail}


# ---------------------------------------------------------------------------
# E2BExecute — sandbox run + local copy-back (inlined)
# CSV/XLSX → session root; PNG/SVG → run_<tool_call_id>/ for API image discovery
# ---------------------------------------------------------------------------


def e2b_execute_node(state: CodingAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Run generated_code.py in a fresh E2B sandbox and copy outputs locally."""
    del config
    from e2b_code_interpreter import Sandbox

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
    exit_code = -1
    error = ""
    copied_outputs: list[str] = []
    missing_outputs: list[str] = []
    plots: list[str] = []

    try:
        sandbox = Sandbox.create(
            api_key=E2B_API_KEY or None,
            timeout=E2B_SANDBOX_TIMEOUT_SECONDS,
        )
        sandbox.commands.run(f"mkdir -p {workspace_dir}")

        # Upload session CSV/XLSX inputs into the sandbox workspace.
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
        exit_code = int(getattr(result, "exit_code", getattr(result, "exitCode", -1)) or -1)

        # Download declared outputs; tabular files land at session root, plots under run_<id>/.
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
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:
                pass

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
    }


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


def build_graph():
    """Construct and compile the coding pipeline ``StateGraph``."""
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
    return builder.compile(checkpointer=checkpointer)


def get_graph():
    """Return the compiled coding sub-graph (lazy singleton)."""
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = build_graph()
    return compiled_graph


def coding_thread_id(session_id: str) -> str:
    """Checkpoint thread id for the coding sub-agent."""
    return f"coding_sub_agent_{session_id}"


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


def invoke_coding_pipeline(
    *,
    session_id: str,
    tool_call_id: str,
    requirements: str,
    input_files: list[str],
    output_files: list[str],
    data_profile: list,
) -> dict[str, Any]:
    """Run the coding sub-graph and return a tool JSON-shaped dict."""
    graph = get_graph()
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
    config: Dict[str, Any] = {
        "configurable": {"thread_id": coding_thread_id(session_id)},
        "recursion_limit": CODING_RECURSION_LIMIT,
    }
    try:
        result = graph.invoke(initial, config=config)
    except GraphRecursionError:
        snap = graph.get_state(config)
        result = dict(snap.values) if snap and snap.values else dict(initial)
        if result.get("status") != "success":
            result.setdefault("execution_feedback", "")
            if not (result.get("execution_feedback") or "").strip():
                result["execution_feedback"] = (
                    f"Coding pipeline hit recursion limit ({CODING_RECURSION_LIMIT})."
                )

    return build_coding_tool_response(result)
