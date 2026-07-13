"""
Pydantic schemas for the coding sub-agent: tool input and codegen output.

``CodingToolInput`` and basename validators gate E2B I/O at the orchestrator tool.
``build_coding_tool_response`` shapes the JSON returned to the orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from session_paths import session_root

ALLOWED_INPUT_EXT = frozenset({".csv", ".xlsx", ".xls"})
ALLOWED_OUTPUT_EXT = frozenset({".csv", ".xlsx", ".png", ".svg"})
ALLOWED_READ_FILE_EXT = frozenset({".csv", ".xlsx"})

SUMMARY_BY_STAGE: dict[str | None, str] = {
    None: "Script ran successfully.",
    "": "Script ran successfully.",
    "missing_inputs": "Failed: declared input file(s) not in session.",
    "semgrep": "Failed: code did not pass safety scan.",
    "io_allowlist": "Failed: code used undeclared file I/O.",
    "e2b": "Failed: sandbox execution error.",
    "codegen_exhausted": "Failed: codegen retry limit reached.",
    "node_error": "Failed: pipeline node error.",
}


def validate_input_basename(name: str) -> str:
    """Single-segment input file name; .csv / .xlsx / .xls only."""
    raw = str(name).strip()
    base = Path(raw).name
    if not base or base != raw:
        raise ValueError("Input must be a bare filename (no directories).")
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_INPUT_EXT:
        raise ValueError(f"Input {base!r}: extension must be one of {sorted(ALLOWED_INPUT_EXT)}")
    return base


def validate_read_file_basename(name: str) -> str:
    """Single-segment read_file name; .csv / .xlsx only."""
    raw = str(name).strip()
    base = Path(raw).name
    if not base or base != raw:
        raise ValueError("file_name must be a bare filename (no directories).")
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_READ_FILE_EXT:
        raise ValueError(f"file_name {base!r}: extension must be .csv or .xlsx.")
    return base


def validate_output_basename(name: str) -> str:
    """Single-segment output file name; tabular or plot extensions only."""
    raw = str(name).strip()
    base = Path(raw).name
    if not base or base != raw:
        raise ValueError("Output must be a bare filename (no directories).")
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_OUTPUT_EXT:
        raise ValueError(f"Output {base!r}: extension must be one of {sorted(ALLOWED_OUTPUT_EXT)}")
    if base.lower() == "generated_code.py":
        raise ValueError("generated_code.py cannot be used as a data or plot path.")
    return base


def find_missing_input_files(session_id: str, input_files: list[str]) -> list[str]:
    """Return basenames in input_files that are not present under the session workspace."""
    missing: list[str] = []
    root = session_root(session_id)
    for basename in input_files:
        name = Path(basename).name
        if not (root / name).is_file():
            missing.append(name)
    return missing


def build_missing_inputs_message(missing: list[str]) -> str:
    """Plain-text failure detail for missing declared input files (basename only)."""
    lines = ["Declared input file(s) missing from session workspace:", ""]
    for name in missing:
        lines.append(f"  - {name}")
    lines.append("")
    lines.append("Upload the file(s) or fix input_files before calling coding_tool again.")
    return "\n".join(lines)


def report_summary_for_stage(stage: str | None) -> str:
    """One-line summary for the coding tool report block."""
    if not stage:
        return SUMMARY_BY_STAGE[None]
    return SUMMARY_BY_STAGE.get(stage, f"Failed: {stage}.")


def build_coding_tool_response(
    *,
    violation: dict[str, Any] | None = None,
    exec_result: dict[str, Any] | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build the JSON body returned by coding_tool and prepare_response_node."""
    violation = violation or {}
    exec_result = exec_result or {}
    failed = bool(violation.get("message", "").strip() or violation.get("stage"))
    stage = (violation.get("stage") or "").strip() or None
    status = "failed" if failed else "success"
    plots = [Path(p).name for p in exec_result.get("plots", []) or []]
    return {
        "status": status,
        "report": {
            "status": status,
            "summary": report_summary_for_stage(stage),
        },
        "failure": violation if failed else None,
        "execution": {
            "stdout": exec_result.get("stdout", "") or "",
            "stderr": exec_result.get("stderr", "") or "",
        },
        "artifacts": {
            "outputs": list(exec_result.get("copied_outputs", []) or []),
            "plots": plots,
        },
        "code": code if code else None,
    }


class CodingToolInput(BaseModel):
    """Arguments LangChain exposes for the ``coding_tool`` on the main orchestrator."""

    requirements: str = Field(
        min_length=1,
        description="Detailed natural-language spec for what the script must do.",
    )
    input_files: list[str] = Field(
        default_factory=list,
        description=(
            "Basenames from data_profile the script may read (.csv/.xlsx); "
            "[] only if the script needs no inputs."
        ),
    )
    output_files: list[str] = Field(
        default_factory=list,
        description=(
            "Every basename the script may write: .csv / .xlsx / .png / .svg; "
            "list all outputs the script creates."
        ),
    )

    @field_validator("input_files", mode="before")
    @classmethod
    def validate_inputs(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("input_files must be a list of filename strings.")
        return [validate_input_basename(x) for x in v]

    @field_validator("output_files", mode="before")
    @classmethod
    def validate_outputs(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("output_files must be a list of filename strings.")
        return [validate_output_basename(x) for x in v]


class CodeGenerationOutput(BaseModel):
    """Structured response returned by the code generation model."""

    filename: str = Field(description="Descriptive Python filename ending in .py.")
    code: str = Field(description="Full runnable Python source.")


def sanitize_run_id(raw: str) -> str:
    """Convert a tool_call_id into a filesystem-safe run folder name."""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw or ""))
    return safe.strip("_") or "unknown_run"
