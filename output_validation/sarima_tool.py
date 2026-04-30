"""
Pydantic schemas for ``sarima_tool``.

Two groups live here:

* ``SarimaToolInput`` — the LangChain tool argument schema. Field descriptions
  are what the orchestrator model sees, so they must match the tool docstring.
* ``ResidualAnalysisOutput`` / ``FitQualityOutput`` / ``ForecastSummaryOutput``
  — structured outputs for the three internal LLM calls that translate the
  deterministic JSON into business-readable text.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Tool input
# ---------------------------------------------------------------------------


class SarimaToolInput(BaseModel):
    """Arguments exposed to the orchestrator for ``sarima_tool``."""

    file_name: str = Field(
        description="Bare CSV/XLSX filename in the session workspace, e.g. 'sales.csv'.",
    )
    date_column: str = Field(
        description="Column name in the file containing dates (parsed and used as the time index).",
    )
    target_column: str = Field(
        description="Numeric column name to forecast.",
    )
    horizon: int = Field(
        gt=0,
        description="Number of future periods to forecast (positive integer).",
    )
    # ``seasonal_period`` is intentionally a required field whose value can be ``null``:
    # this forces the orchestrator to make an explicit seasonality choice (or ask the user)
    # rather than relying on a hidden default.
    seasonal_period: Optional[int] = Field(
        ...,
        description=(
            "Seasonal period m (observations per season). "
            "Pass null for non-seasonal modelling, or an integer (e.g. 12 for monthly with yearly cycle, "
            "4 for quarterly, 7 for daily with weekly cycle). Must be provided when seasonal_order is set. "
            "If unsure, ask the user via ask_user before calling this tool."
        ),
    )
    forecast_output_file: str = Field(
        description="Bare filename for the forecast table; must end with .csv or .xlsx.",
    )
    use_auto_arima: bool = Field(
        default=False,
        description=(
            "If true, the tool runs pmdarima.auto_arima (AICc selection) and ignores any provided order. "
            "If false but order is missing, auto_arima is used and a warning is added to the response."
        ),
    )
    order: Optional[list[int]] = Field(
        default=None,
        description="Optional non-seasonal order [p, d, q] (three non-negative integers).",
    )
    seasonal_order: Optional[list[int]] = Field(
        default=None,
        description=(
            "Optional seasonal order [P, D, Q] (three non-negative integers). "
            "Do NOT include m here; m is given separately as seasonal_period."
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


# ---------------------------------------------------------------------------
# LLM interpretation outputs
# ---------------------------------------------------------------------------


class ResidualAnalysisOutput(BaseModel):
    """Plain-English interpretation of residual diagnostics."""

    status: Literal["ok", "warn"] = Field(
        description="'ok' when residuals look acceptable; 'warn' when assumptions are violated.",
    )
    summary: str = Field(
        description="1-2 sentences explaining whether residuals look like white noise.",
    )
    caveat: str = Field(
        description="1 sentence stating the practical implication for forecast trust.",
    )


class FitQualityOutput(BaseModel):
    """Plain-English interpretation of fit quality (information criteria + convergence)."""

    status: Literal["ok", "warn"] = Field(
        description="'ok' when the model converged and fit looks reasonable; 'warn' otherwise.",
    )
    summary: str = Field(
        description="1-2 sentences on whether the fit is reasonable, referencing AICc/AIC where useful.",
    )
    caveat: str = Field(
        description="1 sentence reminding that information criteria do not guarantee out-of-sample accuracy.",
    )


class ForecastSummaryOutput(BaseModel):
    """Plain-English interpretation of the forecast trajectory and uncertainty."""

    status: Literal["ok", "warn"] = Field(
        description="'ok' by default; 'warn' if intervals are extremely wide or values look implausible.",
    )
    summary: str = Field(
        description="1-2 sentences describing direction (rising/falling/stable) over the horizon.",
    )
    uncertainty: str = Field(
        description="1 sentence describing how the prediction interval changes over the horizon.",
    )
    business_readout: str = Field(
        description="1 sentence translating the forecast into a business takeaway.",
    )


class ModelImprovementGuidanceOutput(BaseModel):
    """Plain-English suggestions for tuning the model further, grounded in provided diagnostics."""

    summary: str = Field(
        description="1-2 sentences describing whether the current model looks sufficient or has room to improve, citing the diagnostic that drives this view.",
    )
    possible_next_steps: list[str] = Field(
        description="2-5 concrete, actionable tuning suggestions tied to the provided diagnostics (e.g. enable auto_arima, adjust p/q, add seasonal_order). Do not promise improvement.",
    )
    caution: str = Field(
        description="1 sentence reminding the reader these are hypotheses to validate, not proven improvements.",
    )
