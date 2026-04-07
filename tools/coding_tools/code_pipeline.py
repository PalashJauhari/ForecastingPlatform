"""
``code_pipeline`` tool implementation (generate → Semgrep → judge → save → run).

Pipeline order
    1. **Codegen** — structured output with ``CODE_GENERATION_SYSTEM_PROMPT``; user message
       may include task, data schema, and optional ``## Previous code policy violations`` on retry.
    2. **Semgrep** — static rules in ``code_scan/codegen_scan_semgrep.yaml``.
    3. **LLM judge** — semantic/policy check (``run_llm_judge``).
    4. **Save** — ``pipeline_run.py`` under ``paths.code`` (overwrites).
    5. **Execute** — same Python interpreter, project root as cwd, via ``code_scan/run_pipeline_sandboxed.py``
       (runtime I/O patches, 200 MiB RLIMIT_AS, BLAS single-thread env, Linux CPU‑0 affinity); wall-clock timeout in parent.

On Semgrep or judge failure, the tool returns the generated source in JSON for review but does **not**
write to disk or run the script. Callers should pass ``previous_code_violation`` on the next attempt.

The LangChain ``@tool`` docstring on ``code_pipeline`` is what the orchestrator model sees; this
module docstring is for developers maintaining the implementation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_openai import ChatOpenAI
from langfuse import observe
from pydantic import BaseModel, Field

from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)
from prompts.code_generation_prompt import CODE_GENERATION_SYSTEM_PROMPT
from session_paths import ensure_session_dirs, resolve_agent_path, session_id_from_config, session_root

from .code_scan.llm_judge import run_llm_judge
from .code_scan.semgrep_scan import format_semgrep_issues, run_semgrep_scan

# ---------------------------------------------------------------------------
# Config (loaded once at import)
# ---------------------------------------------------------------------------

# Repo root (``ForecastingPlatform/``): parents are coding_tools → tools → project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

# Model names (``models.*`` in config.yaml).
CODING_MODEL = cfg["models"].get("code_generation", "gpt-4o-mini")
JUDGE_MODEL = cfg["models"].get("code_judge", cfg["models"].get("code_generation", "gpt-4o-mini"))
# Logical location for generated ``.py`` files. The real directory is session-scoped.
CODE_DIR = cfg["paths"]["code"]

# Execution limits: prefer ``code_pipeline``; fall back to legacy ``run_python_file`` for older configs.
_pipe = cfg.get("code_pipeline") or cfg.get("run_python_file") or {}
# Hard stop for the child process (seconds); raises ``TimeoutExpired`` in the parent.
TIMEOUT = float(_pipe.get("timeout_seconds", 120))

# Subprocess always runs ``run_pipeline_sandboxed.py`` so ``runtime_patch_scan`` wraps I/O before user code.
SANDBOX_RUNNER = Path(__file__).resolve().parent / "code_scan" / "run_pipeline_sandboxed.py"
langfuse = get_langfuse_client()


class CodePipelineInput(BaseModel):
    """
    Arguments exposed to the orchestrator for ``code_pipeline``.

    Field descriptions are shown in the tool schema LangChain binds to the model; keep them aligned
    with ``graph_prompts`` (paths, retries) and with ``CODE_GENERATION_SYSTEM_PROMPT``.
    """

    task: str = Field(
        description=(
            "Required. Natural-language specification of the analysis: what to read, compute, print, and optionally save. "
            "Rules: every CSV/Excel path the generated script must use must appear as a full string starting with "
            "agent_filesystem/ (e.g. agent_filesystem/input/data.csv). Label multiple inputs clearly (Input 1:, Input 2:). "
            "If the script writes a file, include the full output path under agent_filesystem/output/ or processed/. "
            "If the script only prints statistics, no output path is required. "
            "Do not use bare filenames, ./, or paths outside agent_filesystem/. "
            "Examples: "
            "'Input: agent_filesystem/input/sales.csv | Output: agent_filesystem/output/forecast.csv | Fit a regression and save predictions.' "
            "'Input: agent_filesystem/input/sales.csv | Print summary statistics.' "
            "'Input 1: … | Input 2: … | Output: … | Merge on date and compute variance.'"
        ),
    )
    data_schema: str = Field(
        default="",
        description=(
            "Optional. Structured description of each input file the script will read: column names, dtypes, "
            "date formats, nullability, and sample rows if helpful. "
            "When possible, call read_agent_filesystem_data on each input path and paste the returned columns/samples here. "
            "Use an empty string only when the schema is unknown and inspection was not possible."
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


class CodeGenerationOutput(BaseModel):
    """Structured response returned by the code generation model."""

    filename: str = Field(description="Descriptive Python filename ending in .py.")
    explanation: str = Field(description="Short explanation of the generated script.")
    code: str = Field(description="Full runnable Python source.")


@observe(name="tool.code_pipeline", as_type="tool")
def _code_pipeline_impl(
    task: str,
    data_schema: str = "",
    previous_code_violation: str = "",
    session_id: str = "default",
) -> str:
    """Generate, validate, save, and run Python for a tabular-data task in one tool call.

    Use when the user needs **new** Python code written and executed against files under
    ``agent_filesystem/`` (transformations, models, reports). The tool runs a dedicated
    codegen model, applies a **Semgrep** scan and an **LLM judge** before saving, writes the script to
    ``agent_filesystem/code/pipeline_run.py`` (overwritten each time), then executes it in a
    subprocess with a wall-clock timeout and resource-related environment limits.

    Do not use for: answering from already-loaded data alone (use read tools), or when the
    user only needs a file listing.

    Args:
        task: Natural-language job plus all data paths (full ``agent_filesystem/...`` strings).
        data_schema: Optional per-file schema and samples; empty if unknown.
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
    llm = ChatOpenAI(
        model=CODING_MODEL,
        temperature=0,
    ).with_structured_output(CodeGenerationOutput)
    # Markdown sections must stay in sync with ``CODE_GENERATION_SYSTEM_PROMPT`` (schema + retries).
    user_payload = "## Task\n" + task.strip() + "\n\n## Data schema\n" + data_schema.strip()
    if previous_code_violation.strip():
        user_payload += "\n\n## Previous code policy violations\n" + previous_code_violation.strip()
    prompt_messages = [
        SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_payload),
    ]
    with langfuse.start_as_current_observation(name="code_pipeline.codegen", as_type="generation", model=CODING_MODEL, input=[serialize_message(message) for message in prompt_messages]) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            generation.update(output={"error": str(e)}, metadata={"has_data_schema": bool(data_schema.strip()), "has_previous_code_violation": bool(previous_code_violation.strip())})
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
        generation.update(output=resp.model_dump(), metadata={"has_data_schema": bool(data_schema.strip()), "has_previous_code_violation": bool(previous_code_violation.strip())})

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
    judge_ok, judge_detail = run_llm_judge(code=code, task=task, model_name=JUDGE_MODEL)
    if not judge_ok:
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
    code_dir = resolve_agent_path(session_id, CODE_DIR)
    session_workspace = session_root(session_id)
    with langfuse.start_as_current_observation(name="code_pipeline.save_script", as_type="span", input={"target_dir": str(CODE_DIR)}) as save_span:
        code_dir.mkdir(parents=True, exist_ok=True)
        out_path = code_dir / "pipeline_run.py"
        out_path.write_text(code, encoding="utf-8")
        save_span.update(output={"path": str(out_path)}, metadata={"bytes_written": len(code.encode("utf-8"))})
    gen = {
        "code": code,
        "explanation": explanation,
        "code_safety_evaluation": {"passed": True},
    }

    # ------------------------------------------------------------------
    # Step 4 — Run script: subprocess, no shell, project root as cwd
    # (memory / BLAS / CPU affinity: see ``run_pipeline_sandboxed.py``).
    # ------------------------------------------------------------------
    cmd = [sys.executable, str(SANDBOX_RUNNER), str(out_path), str(session_workspace)]
    with langfuse.start_as_current_observation(name="code_pipeline.execute_subprocess", as_type="span", input={"command": cmd, "cwd": str(PROJECT_ROOT), "timeout_seconds": TIMEOUT, "session_id": session_id}) as execution_span:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
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

    result = json.dumps({"code_generation": gen, "execution": execution}, default=str)
    langfuse.update_current_span(output={"code_generation_passed": True, "execution": execution}, metadata={"final_stage": "execute_subprocess", "status": "completed"})
    return result


@tool(args_schema=CodePipelineInput)
def code_pipeline(
    task: str,
    data_schema: str = "",
    previous_code_violation: str = "",
    runtime: ToolRuntime | None = None,
) -> str:
    """LangChain wrapper for the traced code pipeline implementation."""
    session_id = session_id_from_config(runtime.config if runtime is not None else None)
    return _code_pipeline_impl(
        task=task,
        data_schema=data_schema,
        previous_code_violation=previous_code_violation,
        session_id=session_id,
    )
