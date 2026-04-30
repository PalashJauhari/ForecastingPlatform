"""
``code_pipeline`` tool implementation (generate → Semgrep → judge → save → run).

Pipeline order
    1. **Codegen** — structured output with ``CODE_GENERATION_SYSTEM_PROMPT``; user message
       may include task, data profile, and optional ``## Previous code policy violations`` on retry.
    2. **Semgrep** — static rules in ``code_scan/codegen_scan_semgrep.yaml``.
    3. **LLM judge** — semantic/policy check (``run_llm_judge``).
    4. **Save** — ``pipeline_run.py`` at ``agent_filesystem/<session>/pipeline_run.py`` (overwrites).
    5. **Execute** — same Python interpreter, project root as cwd, via ``code_scan/run_pipeline_sandboxed.py``
       (runtime I/O patches, 200 MiB RLIMIT_AS, BLAS single-thread env, Linux CPU‑0 affinity); wall-clock timeout in parent.
       The subprocess receives a **sanitized** copy of the parent environment (LLM and Langfuse secrets removed)
       so generated scripts cannot read those variables even if static checks were bypassed.

On Semgrep or judge failure, the tool returns the generated source in JSON for review but does **not**
write to disk or run the script. Callers should pass ``previous_code_violation`` on the next attempt.

All session files, including generated ``pipeline_run.py``, live at ``./agent_filesystem/<session-folder>/``.

The LangChain ``@tool`` docstring on ``code_pipeline`` is what the orchestrator model sees; this
module docstring is for developers maintaining the implementation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langfuse import observe
from pydantic import BaseModel, Field

from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)
from output_validation.code_generation import CodeGenerationOutput
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
from .code_scan.semgrep_scan import format_semgrep_issues, run_semgrep_scan

# Plot artifacts the per-question run folder may contain. Anything else (csv/xlsx) lands at
# session root via the ``safe_to_*`` patches, so this set is intentionally tiny.
PLOT_FILE_EXTENSIONS = frozenset({".png", ".svg"})

# ---------------------------------------------------------------------------
# Config (loaded once at import)
# ---------------------------------------------------------------------------

# Repo root (``ForecastingPlatform/``): parents are coding_tools → tools → project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

# Model names (``models.*`` in config.yaml).
CODING_MODEL = cfg["models"].get("code_generation", "gpt-4o-mini")
JUDGE_MODEL = cfg["models"].get("code_judge", cfg["models"].get("code_generation", "gpt-4o-mini"))
# Execution limits: prefer ``code_pipeline``; fall back to legacy ``run_python_file`` for older configs.
_pipe = cfg.get("code_pipeline") or cfg.get("run_python_file") or {}
# Hard stop for the child process (seconds); raises ``TimeoutExpired`` in the parent.
TIMEOUT = float(_pipe.get("timeout_seconds", 120))

# Subprocess always runs ``run_pipeline_sandboxed.py`` so ``runtime_patch_scan`` wraps I/O before user code.
SANDBOX_RUNNER = Path(__file__).resolve().parent / "code_scan" / "run_pipeline_sandboxed.py"
langfuse = get_langfuse_client()

# Strip these from the codegen subprocess env (denylist — not a scan of user source).
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


def _env_for_sandbox_subprocess() -> dict[str, str]:
    """Parent ``os.environ`` minus credentials the generated script must not see."""
    out = dict(os.environ)
    for key in list(out):
        if key in _SANDBOX_ENV_DENY_EXACT or key.startswith(_SANDBOX_ENV_DENY_PREFIX):
            out.pop(key, None)
    return out


class CodePipelineInput(BaseModel):
    """
    Arguments exposed to the orchestrator for ``code_pipeline``.

    Field descriptions are shown in the tool schema LangChain binds to the model; keep them aligned
    with ``graph_prompts`` (paths, retries) and with ``CODE_GENERATION_SYSTEM_PROMPT``.
    """

    task: str = Field(
        description=(
            "Required. Natural-language analysis spec. Name every dataset by **filename** only (.csv / .xlsx), "
            "e.g. `Input: sales.csv` or `Input1: a.csv | Input 2: b.xlsx`. "
            "If saving, give the **output filename** (e.g. `Output: forecast.csv`). "
            "If only printing stats, omit output. No full workspace paths, no `./`. "
            "Examples: "
            "'Input: sales.csv | Output: forecast.csv | Fit a regression and save predictions.' "
            "'Input: sales.csv | Print summary statistics.' "
            "'Input 1: orders.csv | Input 2: returns.csv | Output: merged.csv | Merge on date.'"
        ),
    )
    data_profile: str = Field(
        default="",
        description=(
            "Optional. Structured description of each input file the script will read: column names, dtypes, "
            "date formats, nullability, and sample rows if helpful. "
            "Derive this from the **Session workspace** ``data_profile`` list in context (and prior tool results); "
            "or run a small exploratory **code_pipeline** step first and summarize findings here. "
            "Use an empty string only when the profile is unknown."
        ),
    )
    previous_code_violation: str = Field(
        default="",
        description=(
            "Optional. When this is a **retry** after Semgrep or the LLM judge rejected an earlier attempt, "
            "paste the prior **coding policy / safety** failure (e.g. forbidden import, bad path, judge detail). "
            "Empty on the first attempt. Helps the model fix the issue without repeating the violation."
        ),
    )
@observe(name="tool.code_pipeline", as_type="tool")
def _code_pipeline_impl(
    task: str,
    tool_call_id: str,
    data_profile: str = "",
    previous_code_violation: str = "",
    session_id: str = "default",
    active_skills: list[str] = [],
) -> str:
    """Generate, validate, save, and run Python for a tabular-data task in one tool call.

    Use when the user needs **new** Python code written and executed against files under
    ``agent_filesystem/`` (transformations, models, reports). The tool runs a dedicated
    codegen model, applies a **Semgrep** scan and an **LLM judge** before saving, writes the script to
    ``agent_filesystem/<session>/pipeline_run.py`` (overwritten each time), then executes it in a
    subprocess with a wall-clock timeout and resource-related environment limits.

    Do not use for: answering from already-loaded data alone (use read tools), or when the
    user only needs a file listing.

    Args:
        task: Natural-language job naming inputs/outputs by **filename** (see tool schema); codegen expands to full paths in code.
        data_profile: Optional per-file structure and samples; empty if unknown.
        previous_code_violation: Optional text describing a **prior** Semgrep/judge failure on a previous codegen attempt;
            included in the model prompt so the retry respects policy. Empty on first attempt.

    Returns:
        A **JSON string** (parse it) with two keys:

        **code_generation** — Always present.
        - If the model returns no code: ``code`` is empty; ``detail`` explains; ``execution`` is null.
        - If **Semgrep** or the **LLM judge** fails: ``code`` still contains the **generated source** so you can
          review it against ``code_safety_evaluation.detail``; ``execution`` is null; nothing is saved to disk.
        - If codegen, Semgrep, and the judge all pass: ``code`` is the generated source, ``explanation``
          summarizes it, ``code_safety_evaluation`` has ``passed: true``; the same source is written to ``pipeline_run.py``.

        **execution** — null when the script was not run (failures above). Otherwise an object
        with ``stdout``, ``stderr``, and ``returncode`` from the subprocess. If the run hits
        the configured timeout, ``returncode`` is 124 and ``error`` describes the timeout.

    After any safety failure, the **next** call should pass **previous_code_violation** with the
    prior attempt’s ``code_safety_evaluation.detail`` (Semgrep/judge text). You may still refine
    **task** if needed;     do not retry with an empty **previous_code_violation** as if nothing failed.
    """
    # ------------------------------------------------------------------
    # Step 1 — Codegen (structured object: filename, explanation, code)
    # ------------------------------------------------------------------
    llm = make_llm(
        model=CODING_MODEL,
        temperature=0,
        output_schema=CodeGenerationOutput,
    )
    pattern_guidance = LoadPatternSkills(active_skills)
    # Keep implementation guidance in the user payload so codegen sees one fully-assembled spec.
    user_payload = (
        "## Task\n"
        f"{task.strip()}\n\n"
        "## Data Profile\n"
        f"{data_profile.strip() or '(none)'}\n\n"
        "## Vetted Code Patterns\n"
        f"{pattern_guidance.strip() if pattern_guidance else '(none)'}"
    )

    prompt_messages = [
        SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_payload),
    ]
    with langfuse.start_as_current_observation(name="code_pipeline.codegen", as_type="generation", model=CODING_MODEL, input=[serialize_message(message) for message in prompt_messages]) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            generation.update(output={"error": str(e)}, metadata={"has_data_profile": bool(data_profile.strip()), "has_previous_code_violation": bool(previous_code_violation.strip())})
            result = json.dumps(
                {
                    "code_generation": {
                        "code": "",
                        "explanation": "",
                        "code_safety_evaluation": {
                            "passed": False,
                            "detail": f"Code generation failed to produce valid structured output: {e}",
                        },
                    },
                    "execution": None,
                },
                default=str,
            )
            langfuse.update_current_span(metadata={"final_stage": "codegen", "status": "invalid_structured_output"})
            return result
        generation.update(output=resp.model_dump(), metadata={"has_data_profile": bool(data_profile.strip()), "has_previous_code_violation": bool(previous_code_violation.strip())})

    code = resp.code.strip()
    explanation = resp.explanation.strip()

    # ------------------------------------------------------------------
    # Step 2a — Empty codegen: skip expensive checks and disk I/O
    # ------------------------------------------------------------------
    if not code:
        result = json.dumps(
            {
                "code_generation": {
                    "code": "",
                    "explanation": explanation,
                    "code_safety_evaluation": {"passed": False, "detail": "No code returned from model."},
                },
                "execution": None,
            },
            default=str,
        )
        langfuse.update_current_span(metadata={"final_stage": "codegen", "status": "no_code"})
        return result

    # ------------------------------------------------------------------
    # Step 2b — Semgrep (static patterns; see codegen_scan_semgrep.yaml)
    # ------------------------------------------------------------------
    with langfuse.start_as_current_observation(name="code_pipeline.semgrep", as_type="span", input={"code": code}) as semgrep_span:
        semgrep_report = run_semgrep_scan(code)
        semgrep_span.update(output=semgrep_report, metadata={"passed": semgrep_report["passed"], "violation_count": len(semgrep_report["violations"])})
    if not semgrep_report["passed"]:
        # Return the blocked source so the caller can inspect it and retry with `previous_code_violation`.
        result = json.dumps(
            {
                "code_generation": {
                    "code": code,
                    "explanation": explanation,
                    "code_safety_evaluation": {
                        "passed": False,
                        "detail": format_semgrep_issues(semgrep_report["violations"]),
                    },
                },
                "execution": None,
            },
            default=str,
        )
        langfuse.update_current_span(metadata={"final_stage": "semgrep", "status": "blocked"})
        return result

    # ------------------------------------------------------------------
    # Step 2c — LLM judge (task alignment, policy gaps Semgrep can miss)
    # ------------------------------------------------------------------
    judge_ok, judge_detail = run_llm_judge(
        code=code,
        task=task,
        model_name=JUDGE_MODEL,
    )
    if not judge_ok:
        # Mirror the Semgrep failure shape so the orchestrator can handle both safety gates the same way.
        result = json.dumps(
            {
                "code_generation": {
                    "code": code,
                    "explanation": explanation,
                    "code_safety_evaluation": {"passed": False, "detail": judge_detail},
                },
                "execution": None,
            },
            default=str,
        )
        langfuse.update_current_span(metadata={"final_stage": "llm_judge", "status": "blocked"})
        return result

    # ------------------------------------------------------------------
    # Step 3 — Persist only after both gates pass (fixed name for the runner)
    # ------------------------------------------------------------------
    ensure_session_dirs(session_id)
    run_logical = logical_pipeline_run_path(session_id)
    out_path = resolve_agent_path(session_id, run_logical)
    session_workspace = session_root(session_id)
    with langfuse.start_as_current_observation(name="code_pipeline.save_script", as_type="span", input={"logical_path": run_logical}) as save_span:
        out_path.write_text(code, encoding="utf-8")
        save_span.update(output={"path": str(out_path)}, metadata={"bytes_written": len(code.encode("utf-8"))})
    gen = {
        "code": code,
        "explanation": explanation,
        "code_safety_evaluation": {"passed": True},
    }

    # ------------------------------------------------------------------
    # Step 4 — Allocate per-question plot folder
    #
    # Each ``code_pipeline`` invocation gets its own ``run_<run_id>/`` subfolder under the
    # session workspace. ``plt.savefig`` writes are routed there by the runtime patch, so
    # the API can find the plots produced by this specific tool call by looking up the
    # tool call id and listing the matching run folder. Tabular outputs (csv/xlsx) still
    # land at session root and remain reusable across questions.
    #
    # ``run_id`` is the LangChain ``tool_call_id`` (sanitized for filesystem use).
    # ``sarima_tool`` and ``prophet_tool`` do **not** write plot files — only this tool
    # does — so aligning folder names with each tool-call id keeps per-turn plots
    # discoverable alongside other tools' ``ToolMessage`` ids without guessing paths.
    # ------------------------------------------------------------------
    run_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(tool_call_id or "")).strip("_") or "unknown_run"
    run_workspace = session_workspace / f"run_{run_id}"
    # ``apply_patches`` will mkdir again inside the subprocess; we create here too so the
    # parent can scan the folder after execution even if the script wrote nothing.
    run_workspace.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 5 — Run script: subprocess, no shell, project root as cwd
    # (memory / BLAS / CPU affinity: see ``run_pipeline_sandboxed.py``).
    # The third argv carries the per-question plot folder across the process boundary;
    # RunnableConfig does not propagate to subprocesses, so we pass it explicitly.
    # ------------------------------------------------------------------
    cmd = [
        sys.executable,
        str(SANDBOX_RUNNER),
        str(out_path),
        str(session_workspace),
        str(run_workspace),
    ]
    with langfuse.start_as_current_observation(name="code_pipeline.execute_subprocess", as_type="span", input={"command": cmd, "cwd": str(PROJECT_ROOT), "timeout_seconds": TIMEOUT, "session_id": session_id, "run_id": run_id}) as execution_span:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=_env_for_sandbox_subprocess(),
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
            # Non-zero returncode is still a successful tool result (caller inspects stderr/stdout).
            execution = {"stdout": proc.stdout or "", "stderr": proc.stderr or "", "returncode": proc.returncode}
        except subprocess.TimeoutExpired as e:
            # SIGKILL path: returncode 124 convention; surface partial streams if the OS attached them.
            err = f"Execution exceeded timeout ({TIMEOUT}s). Process was terminated."
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
            err_out = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
            combined = (err_out + "\n" + err).strip() if err_out else err
            execution = {"error": err, "stdout": out, "stderr": combined, "returncode": 124}
        execution_span.update(output=execution, metadata={"returncode": execution["returncode"], "timed_out": execution["returncode"] == 124, "stdout_length": len(execution.get("stdout", "")), "stderr_length": len(execution.get("stderr", ""))})

    # ------------------------------------------------------------------
    # Step 6 — Collect plot artifacts
    #
    # Scan the per-question run folder for plot files. Returned as logical
    # ``agent_filesystem/<session>/run_<run_id>/<file>`` paths so the UI can fetch them
    # via the ``GET /artifact`` endpoint. By listing only the run folder we get perfect
    # per-message isolation: previous questions live under different ``run_<id>/`` and
    # never leak into the current tool result.
    # ------------------------------------------------------------------
    plots: list[str] = []
    if run_workspace.exists():
        sid = session_dir_for_paths(session_id)
        for f in sorted(run_workspace.iterdir()):
            if f.is_file() and f.suffix.lower() in PLOT_FILE_EXTENSIONS:
                plots.append(f"agent_filesystem/{sid}/run_{run_id}/{f.name}")

    result = json.dumps({"code_generation": gen, "execution": execution, "plots": plots}, default=str)
    langfuse.update_current_span(output={"code_generation_passed": True, "execution": execution, "plot_count": len(plots)}, metadata={"final_stage": "execute_subprocess", "status": "completed", "run_id": run_id})
    return result


@tool(args_schema=CodePipelineInput)
def code_pipeline(
    runtime: ToolRuntime,
    task: str,
    data_profile: str = "",
    previous_code_violation: str = "",
) -> str:
    """LangChain wrapper for the traced code pipeline implementation."""
    # ``ToolRuntime`` first (required, no default) so it injects from LangGraph; defaults follow for Python syntax.
    session_id = session_id_from_config(runtime.config)
    active_skills = (runtime.state or {}).get("active_skills", [])
    # The injected LangChain tool-call id becomes the per-run folder id so the API
    # can find the plots produced by this exact call without scanning tool JSON.
    tool_call_id = getattr(runtime, "tool_call_id", "") or ""

    return _code_pipeline_impl(
        task=task,
        tool_call_id=tool_call_id,
        data_profile=data_profile,
        previous_code_violation=previous_code_violation,
        session_id=session_id,
        active_skills=active_skills,
    )
