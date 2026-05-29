"""
Shared Pydantic fields for forecasting tool inputs.

``BaseForecastToolInput`` holds arguments common to all forecasting tools.
Field descriptions are the orchestrator contract (hyperparameter impact lives here).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class BaseForecastToolInput(BaseModel):
    """Arguments shared by all session forecasting tools."""

    experiment_name: str = Field(
        description=(
            "Unique label for this forecast run; prefixes all output CSV/PNG files "
            "(e.g. 'revenue_q1_hw'). Use letters, numbers, underscores, dots, hyphens only."
        ),
    )
    file_name: str = Field(
        description=(
            "Basename of an uploaded session file, e.g. 'sales.csv'. "
            "Must be .csv or .xlsx; not a path or session prefix."
        ),
    )
    date_column: str = Field(
        description=(
            "Column name with parseable dates; one row per period; "
            "duplicate dates are dropped (first row kept)."
        ),
    )
    target_column: str = Field(
        description="Numeric column to forecast.",
    )
    horizon: int = Field(
        gt=0,
        description=(
            "Number of future periods at the inferred frequency "
            "(e.g. 12 = next 12 months if data is monthly)."
        ),
    )

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, v: str) -> str:
        name = str(v).strip()
        if not name:
            raise ValueError("experiment_name must not be empty.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(
                "experiment_name may only contain letters, numbers, underscores, dots, and hyphens."
            )
        return name


class InterpretationSummaryOutput(BaseModel):
    """Single business-readable summary for the forecasting tool response."""

    summary: str = Field(
        description="2-4 sentences for the user: fit quality, residual concerns, forecast direction, and caveats.",
    )
