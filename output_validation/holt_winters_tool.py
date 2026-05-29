"""
Pydantic schemas for ``holt_winters_tool`` — ``HoltWintersToolInput`` only.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator

from output_validation.forecasting_common import BaseForecastToolInput


class HoltWintersToolInput(BaseForecastToolInput):
    """Arguments exposed to the orchestrator for ``holt_winters_tool``."""

    seasonal_period: Optional[int] = Field(
        ...,
        description=(
            "Seasonal length m (e.g. 12 for monthly). Pass null when seasonal='none'. "
            "Required for add/mul seasonal."
        ),
    )
    trend: Literal["none", "add", "mul"] = Field(
        description=(
            "none: level/seasonal only. add: linear trend. mul: trend scales with level."
        ),
    )
    seasonal: Literal["none", "add", "mul"] = Field(
        description=(
            "none: no seasonality. add: fixed seasonal bumps. mul: seasonal bumps scale with level."
        ),
    )
    damped_trend: bool = Field(
        default=False,
        description=(
            "If true, trend effect fades over the forecast horizon (use when growth should level off)."
        ),
    )

    @field_validator("seasonal_period")
    @classmethod
    def validate_seasonal_period(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 2:
            raise ValueError("seasonal_period must be at least 2 or null.")
        return v
