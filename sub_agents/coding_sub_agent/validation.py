"""
Pydantic schemas for the coding sub-agent: tool input, codegen output, and judge responses.

Tool-input group: ``CodingToolInput`` and basename validators for E2B I/O.
Judge-output group: structured schemas for safety / IO allowlist LLM gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

_ALLOWED_INPUT_EXT = frozenset({".csv", ".xlsx", ".xls"})
_ALLOWED_OUTPUT_EXT = frozenset({".csv", ".xlsx", ".png", ".svg"})


def validate_input_basename(name: str) -> str:
    """Single-segment input file name; .csv / .xlsx / .xls only."""
    raw = str(name).strip()
    base = Path(raw).name
    if not base or base != raw:
        raise ValueError("Input must be a bare filename (no directories).")
    ext = Path(base).suffix.lower()
    if ext not in _ALLOWED_INPUT_EXT:
        raise ValueError(f"Input {base!r}: extension must be one of {sorted(_ALLOWED_INPUT_EXT)}")
    return base


def validate_output_basename(name: str) -> str:
    """Single-segment output file name; tabular or plot extensions only."""
    raw = str(name).strip()
    base = Path(raw).name
    if not base or base != raw:
        raise ValueError("Output must be a bare filename (no directories).")
    ext = Path(base).suffix.lower()
    if ext not in _ALLOWED_OUTPUT_EXT:
        raise ValueError(f"Output {base!r}: extension must be one of {sorted(_ALLOWED_OUTPUT_EXT)}")
    if base.lower() == "generated_code.py":
        raise ValueError("generated_code.py cannot be used as a data or plot path.")
    return base


class CodingToolInput(BaseModel):
    """Arguments LangChain exposes for the ``coding_tool`` on the main orchestrator."""

    requirements: str = Field(
        min_length=1,
        description="Detailed natural-language spec for what the script must do.",
    )
    input_files: list[str] = Field(
        default_factory=list,
        description="Basenames of CSV/XLSX files this run may read.",
    )
    output_files: list[str] = Field(
        default_factory=list,
        description="Basenames of files this run may write: .csv / .xlsx / .png / .svg.",
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
    explanation: str = Field(
        description=(
            "Concise prose for the orchestrator: problem solved, logic, and every "
            "input/output basename touched."
        ),
    )
    code: str = Field(description="Full runnable Python source.")


class JudgeOutput(BaseModel):
    """Structured accept/reject response from a judge model."""

    passed: bool = Field(description="Whether the generated code is safe and on-spec.")
    detail: str = Field(default="", description="Short rejection reason when passed is false.")


def sanitize_run_id(raw: str) -> str:
    """Convert a tool_call_id into a filesystem-safe run folder name."""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw or ""))
    return safe.strip("_") or "unknown_run"
