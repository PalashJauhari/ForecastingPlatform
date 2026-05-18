"""
Structured output validation for the code generation model inside ``code_pipeline``.
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
    if base.lower() == "pipeline_run.py":
        raise ValueError("pipeline_run.py cannot be used as a data or plot path.")
    return base


class CodePipelineTask(BaseModel):
    """
    Structured task for ``code_pipeline``: detailed requirements plus explicit input/output basenames.

    ``requirements`` is the full natural-language spec. ``input`` / ``output`` list only filenames
    (no folders or session prefixes), matching codegen and runtime enforcement.
    """

    requirements: str = Field(
        min_length=1,
        description="Detailed natural-language requirements for what the script must do.",
    )
    input: list[str] = Field(
        default_factory=list,
        description="Basenames of CSV/XLSX files this run may read (empty = runtime does not restrict reads to a list).",
    )
    output: list[str] = Field(
        default_factory=list,
        description="Basenames of files this run may write: .csv / .xlsx / .png / .svg (empty = runtime does not restrict writes to a list).",
    )

    @field_validator("input", mode="before")
    @classmethod
    def validate_inputs(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("input must be a list of filename strings.")
        return [validate_input_basename(x) for x in v]

    @field_validator("output", mode="before")
    @classmethod
    def validate_outputs(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("output must be a list of filename strings.")
        return [validate_output_basename(x) for x in v]


class CodeGenerationOutput(BaseModel):
    """Structured response returned by the code generation model."""

    filename: str = Field(description="Descriptive Python filename ending in .py.")
    explanation: str = Field(
        description=(
            "Concise prose for the orchestrator when the script is not echoed after safety passes: "
            "(1) problem solved, "
            "(2) high-level logic / steps, "
            "(3) every **input** and **output** basename from the Task that the script touches. "
            "Align with the Task's ``input`` / ``output`` lists where applicable."
        ),
    )
    code: str = Field(description="Full runnable Python source.")


class CodePipelineInput(BaseModel):
    """Arguments LangChain exposes for the ``code_pipeline`` tool."""

    task: CodePipelineTask = Field(
        description=(
            "Required. Object with keys: **requirements** (detailed natural-language spec); "
            "**input** (list of basenames — .csv / .xlsx / .xls — the script may read); "
            "**output** (list of basenames — .csv / .xlsx / .png / .svg — the script may write). "
            "Use empty lists when no explicit allowlist is needed."
        ),
    )
    data_profile: str = Field(
        default="",
        description=(
            "Optional. Per-input file structure and samples from Session workspace ``data_profile``; "
            "empty if unknown."
        ),
    )
    previous_code_violation: str = Field(
        default="",
        description=(
            "On retry after safety or runtime failure, paste prior **code_violation** / stderr detail; "
            "it is included under `## Previous code policy violations` in the Human message."
        ),
    )
