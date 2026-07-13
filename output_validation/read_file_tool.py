"""
Pydantic schemas for ``read_file_tool`` — ``ReadFileToolInput`` only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from sub_agents.coding_sub_agent.validation import validate_read_file_basename


class ReadFileToolInput(BaseModel):
    """Arguments exposed to the orchestrator for ``read_file_tool``."""

    file_name: str = Field(
        description="Basename of a session CSV/XLSX file from data_profile (e.g. monthly_revenue.csv).",
    )
    columns: list[str] = Field(
        default_factory=list,
        description=(
            "Column names to return; use [] to return all columns. "
            "Names must match column_profiles[].name in data_profile."
        ),
    )

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        return validate_read_file_basename(v)

    @field_validator("columns", mode="before")
    @classmethod
    def validate_columns(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("columns must be a list of column name strings.")
        return [str(x).strip() for x in v if str(x).strip()]
