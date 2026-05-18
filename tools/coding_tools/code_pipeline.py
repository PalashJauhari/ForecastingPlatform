"""
``code_pipeline`` tool: codegen → Semgrep → LLM judge → save → run (sandboxed).

See module docstring in the LangChain ``@tool`` wrapper for orchestrator-facing behavior.
"""

# Do NOT add ``from __future__ import annotations``. LangChain's ``@tool`` introspects the
# ``ToolRuntime`` parameter for injection; with PEP 563 postponed evaluation the annotation
# becomes the string ``"ToolRuntime"`` and injection is skipped → missing ``runtime`` at call time.

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langfuse import observe

from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)
from output_validation.code_generation import (
    CodeGenerationOutput,
    CodePipelineInput,
    CodePipelineTask,
)
from prompts.code_generation_prompt import CODE_GENERATION_SYSTEM_PROMPT
from session_paths import (
    ensure_session_dirs,
    logical_pipeline_run_path,
    resolve_agent_path,
    session_dir_for_paths,
    session_id_from_config,
    session_root,
)
from skills.loader import LoadPatternSkills

from .code_scan.llm_judge import run_llm_judge
from .code_scan.safety_check import SafetyCheckResult
from .code_scan.semgrep_scan import run_semgrep_scan

PLOT_FILE_EXTENSIONS = frozenset({".png", ".svg"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

CODING_MODEL = cfg["models"].get("code_generation", "gpt-4o-mini")
JUDGE_MODEL = cfg["models"].get("code_judge", cfg["models"].get("code_generation", "gpt-4o-mini"))
_pipe = cfg.get("code_pipeline") or cfg.get("run_python_file") or {}
TIMEOUT = float(_pipe.get("timeout_seconds", 120))

SANDBOX_RUNNER = Path(__file__).resolve().parent / "code_scan" / "run_pipeline_sandboxed.py"
langfuse = get_langfuse_client()

_SANDBOX_ENV_DENY_EXACT: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
        "GOOGLE_API_KEY",
    }
)
_SANDBOX_ENV_DENY_PREFIX: tuple[str, ...] = ("LANGFUSE_",)


def env_for_sandbox_subprocess() -> dict[str, str]:
    """Parent ``os.environ`` minus credentials the generated script must not see."""
    out = dict(os.environ)
    for key in list(out):
        if key in _SANDBOX_ENV_DENY_EXACT or key.startswith(_SANDBOX_ENV_DENY_PREFIX):
            out.pop(key, None)
    return out


def merge_gate_violations(sem: SafetyCheckResult, judge: SafetyCheckResult) -> dict[str, str]:
    """Combine Semgrep and judge failures into one ``code_violation`` map (keys ``semgrep`` / ``judge``)."""
    out: dict[str, str] = {}
    if not sem.passed and sem.detail:
        out["semgrep"] = sem.detail
    if not judge.passed and judge.detail:
        out["judge"] = judge.detail
    return out


def execution_needs_code_attachment(execution: dict[str, Any]) -> bool:
    """True when the tool should echo generated ``code`` so the model can fix timeout/errors/nonzero exit."""
    if execution.get("returncode") == 124:
        return True
    if execution.get("error"):
        return True
    if (execution.get("stderr") or "").strip():
        return True
    if execution.get("returncode", 0) not in (0, None):
        return True
    return False


@observe(name="tool.code_pipeline", as_type="tool")
def code_pipeline_impl(
    task: CodePipelineTask,
    data_profile: str,
    active_skills: list[str],
    previous_code_violation: str,
    session_id: str,
    tool_call_id: str,
    runtime: ToolRuntime,
) -> str:
    """
    Run codegen, Semgrep, LLM judge, then save ``pipeline_run.py`` and execute in the sandbox.

    Returns a JSON string with ``stdout`` / ``stderr`` / ``code_violation`` / ``plots``,
    and ``code`` only when safety gates fail or execution needs a retry patch.
    The structured ``task`` is not repeated (it is already in the tool call message).
    """
    del runtime  # Signature matches LangChain ``ToolRuntime``; reserved for future hooks.

    # --- Codegen: static system prompt; Human carries task JSON, optional retry text, profile, patterns ---
    llm = make_llm(
        model=CODING_MODEL,
        temperature=0,
        output_schema=CodeGenerationOutput,
    )
    pattern_guidance = LoadPatternSkills(active_skills)
    violation = previous_code_violation.strip()
    data_block = data_profile.strip() or "(none)"
    patterns_block = pattern_guidance.strip() if pattern_guidance else "(none)"
    blocks = [f"## Task (structured)\n{json.dumps(task.model_dump(), ensure_ascii=False, indent=2)}"]
    if violation:
        blocks.append(f"## Previous code policy violations\n{violation}")
    blocks.append(f"## Data Profile\n{data_block}")
    blocks.append(f"## Vetted Code Patterns\n{patterns_block}")
    user_payload = "\n\n".join(blocks)
    prompt_messages = [
        SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_payload),
    ]

    with langfuse.start_as_current_observation(name="code_pipeline.codegen", as_type="generation", model=CODING_MODEL, input=[serialize_message(message) for message in prompt_messages]) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            generation.update(
                output={"error": str(e)},
                metadata={
                    "has_data_profile": bool(data_profile.strip()),
                    "has_previous_code_violation": bool(previous_code_violation.strip()),
                },
            )
            langfuse.update_current_span(metadata={"final_stage": "codegen", "status": "invalid_structured_output"})
            return json.dumps(
                {
                    "code": "",
                    "stdout": None,
                    "stderr": None,
                    "code_violation": {"codegen": f"Code generation failed to produce valid structured output: {e}"},
                    "plots": [],
                },
                default=str,
            )
        generation.update(
            output=resp.model_dump(),
            metadata={
                "has_data_profile": bool(data_profile.strip()),
                "has_previous_code_violation": bool(previous_code_violation.strip()),
            },
        )

    code = resp.code.strip()

    if not code:
        langfuse.update_current_span(metadata={"final_stage": "codegen", "status": "no_code"})
        return json.dumps(
            {
                "code": "",
                "stdout": None,
                "stderr": None,
                "code_violation": {"codegen": "No code returned from model."},
                "plots": [],
            },
            default=str,
        )

    # --- Static scan + LLM judge (both run when code is non-empty); on failure return code + violations ---
    with langfuse.start_as_current_observation(name="code_pipeline.semgrep", as_type="span", input={"code_len": len(code)}) as semgrep_span:
        sem = run_semgrep_scan(code)
        semgrep_span.update(
            output={"passed": sem.passed, "detail": sem.detail[:2000] if sem.detail else ""},
            metadata={
                "passed": sem.passed,
                "violation_count": len(sem.violations or []),
            },
        )

    judge = run_llm_judge(code=code, task=task, model_name=JUDGE_MODEL)

    langfuse.update_current_span(
        metadata={
            "semgrep_passed": str(sem.passed).lower(),
            "judge_passed": str(judge.passed).lower(),
        }
    )

    if not sem.passed or not judge.passed:
        viol = merge_gate_violations(sem, judge)
        return json.dumps(
            {
                "code": code,
                "stdout": None,
                "stderr": None,
                "code_violation": viol,
                "plots": [],
            },
            default=str,
        )

    # --- Persist script and run sandbox child; optional IO allowlist as 4th CLI arg JSON ---
    ensure_session_dirs(session_id)
    run_logical = logical_pipeline_run_path(session_id)
    out_path = resolve_agent_path(session_id, run_logical)
    session_workspace = session_root(session_id)

    with langfuse.start_as_current_observation(name="code_pipeline.save_script", as_type="span", input={"logical_path": run_logical}) as save_span:
        out_path.write_text(code, encoding="utf-8")
        save_span.update(
            output={"path": str(out_path)},
            metadata={"bytes_written": len(code.encode("utf-8"))},
        )

    run_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(tool_call_id or "")).strip("_") or "unknown_run"
    run_workspace = session_workspace / f"run_{run_id}"
    run_workspace.mkdir(parents=True, exist_ok=True)

    io_allowlist_path: str | None = None
    if task.input or task.output:
        # Ephemeral JSON passed to run_pipeline_sandboxed so reads/writes stay on declared basenames.
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"input": list(task.input), "output": list(task.output)}, tf, ensure_ascii=False)
        tf.close()
        io_allowlist_path = tf.name

    cmd: list[str] = [
        sys.executable,
        str(SANDBOX_RUNNER),
        str(out_path),
        str(session_workspace),
        str(run_workspace),
    ]
    if io_allowlist_path:
        cmd.append(io_allowlist_path)

    execution: dict[str, Any] = {"stdout": "", "stderr": "", "returncode": -1}
    try:
        with langfuse.start_as_current_observation(name="code_pipeline.execute_subprocess", as_type="span", input={"command": cmd, "cwd": str(PROJECT_ROOT), "timeout_seconds": TIMEOUT, "session_id": session_id, "run_id": run_id}) as execution_span:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env_for_sandbox_subprocess(),
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
                execution = {
                    "stdout": proc.stdout or "",
                    "stderr": proc.stderr or "",
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired as e:
                err = f"Execution exceeded timeout ({TIMEOUT}s). Process was terminated."
                out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
                err_out = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
                combined = (err_out + "\n" + err).strip() if err_out else err
                execution = {"error": err, "stdout": out, "stderr": combined, "returncode": 124}
            execution_span.update(
                output=execution,
                metadata={
                    "returncode": execution["returncode"],
                    "timed_out": execution["returncode"] == 124,
                    "stdout_length": len(execution.get("stdout", "") or ""),
                    "stderr_length": len(execution.get("stderr", "") or ""),
                },
            )
    finally:
        if io_allowlist_path:
            try:
                Path(io_allowlist_path).unlink(missing_ok=True)
            except OSError:
                pass

    # Collect plot paths written under this run_* folder (PNG/SVG only).
    plots: list[str] = []
    if run_workspace.exists():
        sid = session_dir_for_paths(session_id)
        for f in sorted(run_workspace.iterdir()):
            if f.is_file() and f.suffix.lower() in PLOT_FILE_EXTENSIONS:
                plots.append(f"agent_filesystem/{sid}/run_{run_id}/{f.name}")

    # Success path: ``task`` is not echoed (already in the tool call). Omit ``code`` unless execution needs a retry patch.
    out_body: dict[str, Any] = {
        "stdout": execution.get("stdout") or "",
        "stderr": execution.get("stderr") or "",
        "code_violation": None,
        "plots": plots,
    }

    if execution_needs_code_attachment(execution):
        out_body["code"] = code
        rc = execution.get("returncode")
        err_msg = execution.get("error")
        parts: dict[str, str] = {}
        if (execution.get("stderr") or "").strip():
            parts["stderr"] = (execution.get("stderr") or "").strip()
        if err_msg:
            parts["timeout"] = str(err_msg)
        if rc not in (None, 0) and rc != 124 and not parts.get("stderr"):
            parts["returncode"] = f"non-zero exit: {rc}"
        out_body["code_violation"] = parts if parts else {"runtime": "Execution completed with issues; see stderr."}

    langfuse.update_current_span(
        output={"execution": execution, "plot_count": len(plots)},
        metadata={"final_stage": "execute_subprocess", "status": "completed", "run_id": run_id},
    )
    return json.dumps(out_body, default=str)


@tool(args_schema=CodePipelineInput)
def code_pipeline(
    runtime: ToolRuntime,
    task: CodePipelineTask,
    data_profile: str = "",
    previous_code_violation: str = "",
) -> str:
    """Generate Python with Semgrep + LLM judge, save ``pipeline_run.py``, run in sandbox. Returns JSON (no task echo)."""
    session_id = session_id_from_config(runtime.config)
    active_skills = (runtime.state or {}).get("active_skills", [])
    tool_call_id = getattr(runtime, "tool_call_id", "") or ""

    return code_pipeline_impl(
        task=task,
        data_profile=data_profile,
        active_skills=active_skills,
        previous_code_violation=previous_code_violation,
        session_id=session_id,
        tool_call_id=tool_call_id,
        runtime=runtime,
    )
