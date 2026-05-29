"""
Pydantic schemas for ``prophet_tool`` — ``ProphetToolInput`` only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from output_validation.forecasting_common import BaseForecastToolInput


class ProphetToolInput(BaseForecastToolInput):
    """Arguments exposed to the orchestrator for ``prophet_tool``."""

    changepoint_prior_scale: float = Field(
        gt=0,
        description=(
            "Trend flexibility at automatic changepoints. Higher (e.g. 0.5) reacts faster to shifts; "
            "lower (e.g. 0.01) yields a smoother trend. Default-like value is ~0.05."
        ),
    )
    seasonality_mode: Literal["additive", "multiplicative"] = Field(
        description=(
            "additive: fixed seasonal swing size. multiplicative: seasonal swing scales with level "
            "(use when seasonality grows with the series)."
        ),
    )
    weekly_seasonality: bool = Field(
        description="Turn on weekly seasonal pattern (useful for daily data).",
    )
    monthly_seasonality: bool = Field(
        description="Turn on custom monthly seasonality (30.5-day period).",
    )
    yearly_seasonality: bool = Field(
        description="Turn on yearly seasonal pattern.",
    )
