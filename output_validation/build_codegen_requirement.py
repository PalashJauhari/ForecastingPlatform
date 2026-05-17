"""
Structured output for the codegen preflight step (runs inside ``code_pipeline`` when enabled).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from session_paths import LOGICAL_AGENT_PREFIX


class CodegenPreflightOutput(BaseModel):
    """Validated preflight planner output."""

    detailed_requirement: str = Field(
        description=(
            "Structured execution brief for code generation. "
            "Any saved figure filenames must end in .png or .svg only (not .pdf/.jpg/etc.). "
            "When execution needs a script, the brief should instruct **only 2–3 short print lines** "
            "describing pipeline purpose (not verbose logging), unless printed numeric output is explicitly required."
        ),
    )
    dataset_paths: list[str] = Field(
        default_factory=list,
        description="Dataset basenames (.csv / .xlsx), or legacy full logical paths for compatibility.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions made while drafting the requirement.",
    )

    @field_validator("detailed_requirement")
    @classmethod
    def validate_detailed_requirement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("detailed_requirement must not be empty.")
        return value

    @field_validator("dataset_paths")
    @classmethod
    def validate_dataset_paths(cls, value: list[str]) -> list[str]:
        allowed_suffixes = {".csv", ".xlsx"}
        cleaned: list[str] = []
        seen: set[str] = set()

        for path in value:
            path = path.strip()
            if not path:
                continue
            if path.startswith(LOGICAL_AGENT_PREFIX):
                pass
            else:
                if any(sep in path for sep in ("/", "\\")) or ".." in path:
                    raise ValueError(
                        f"dataset_paths must be a bare filename (e.g. sales.csv) or full logical path: {path!r}",
                    )
                name = Path(path).name
                if name != path:
                    raise ValueError(f"dataset_paths filename must be a single basename: {path!r}")
                if Path(path).suffix.lower() not in allowed_suffixes:
                    raise ValueError(
                        f"dataset_paths filename must end with .csv or .xlsx: {path!r}",
                    )
            if path not in seen:
                cleaned.append(path)
                seen.add(path)

        return cleaned

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


# Back-compat for existing imports during refactors / external references
BuildCodegenRequirementOutput = CodegenPreflightOutput
