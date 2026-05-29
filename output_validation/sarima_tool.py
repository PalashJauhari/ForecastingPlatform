"""
Pydantic schemas for ``sarima_tool`` — ``SarimaToolInput`` only.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from output_validation.forecasting_common import BaseForecastToolInput


class SarimaToolInput(BaseForecastToolInput):
    """Arguments exposed to the orchestrator for ``sarima_tool``."""

    seasonal_period: Optional[int] = Field(
        ...,
        description=(
            "Seasonal period m (e.g. 12 for monthly+yearly). Pass null for non-seasonal. "
            "Required when seasonal_order is set."
        ),
    )
    use_auto_arima: bool = Field(
        default=False,
        description=(
            "If true, pmdarima selects (p,d,q)(P,D,Q) by AICc and ignores manual order. "
            "Use when you want data-driven order search instead of fixed ARIMA terms."
        ),
    )
    order: Optional[list[int]] = Field(
        default=None,
        description=(
            "Manual non-seasonal ARIMA order [p,d,q]. Higher p adds AR memory; d differences for trend; "
            "q adds MA smoothing. Ignored when use_auto_arima is true."
        ),
    )
    seasonal_order: Optional[list[int]] = Field(
        default=None,
        description=(
            "Manual seasonal order [P,D,Q] (m is seasonal_period). Adds seasonal AR/I/MA; "
            "ignored when use_auto_arima is true."
        ),
    )

    @field_validator("seasonal_period")
    @classmethod
    def validate_seasonal_period(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 1:
            raise ValueError("seasonal_period must be a positive integer or null.")
        return v

    @field_validator("order", "seasonal_order")
    @classmethod
    def validate_order_shape(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is None:
            return v
        if len(v) != 3 or any((not isinstance(x, int) or x < 0) for x in v):
            raise ValueError("order/seasonal_order must be a list of three non-negative integers.")
        return v
