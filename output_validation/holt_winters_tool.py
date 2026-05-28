"""
Pydantic schemas for ``holt_winters_tool``.

* ``HoltWintersToolInput`` — LangChain tool argument schema.
* LLM interpretation output models for the five internal structured calls.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from output_validation.forecasting_common import BaseForecastToolInput


# ---------------------------------------------------------------------------
# Tool input
# ---------------------------------------------------------------------------


class HoltWintersToolInput(BaseForecastToolInput):
    """Arguments exposed to the orchestrator for ``holt_winters_tool``."""

    decomposition_output_file: str = Field(
        description=(
            "Output basename for the combined fitted + forecast decomposition table "
            "(level/trend/seasonal where available); must end with .csv or .xlsx."
        ),
    )
    seasonal_period: Optional[int] = Field(
        ...,
        description=(
            "Seasonal period m (observations per season). Pass null when seasonal='none'. "
            "Required when seasonal is 'add' or 'mul' (e.g. 12 for monthly+yearly cycle). "
            "If unsure, ask the user before calling."
        ),
    )
    trend: Literal["none", "add", "mul"] = Field(
        description=(
            "Trend component: 'none' (level-only or seasonal-only), 'add' (additive trend), "
            "or 'mul' (multiplicative trend). Full Holt-Winters typically uses 'add'."
        ),
    )
    seasonal: Literal["none", "add", "mul"] = Field(
        description=(
            "Seasonal component: 'none', 'add' (constant seasonal amplitude), or 'mul' "
            "(seasonal amplitude scales with level)."
        ),
    )
    damped_trend: bool = Field(
        default=False,
        description="If true, apply damped trend (trend effects decay over the forecast horizon).",
    )

    @field_validator("seasonal_period")
    @classmethod
    def validate_seasonal_period(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 2:
            raise ValueError("seasonal_period must be at least 2 or null.")
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
    """Plain-English interpretation of fit quality (MAE / RMSE / SMAPE)."""

    status: Literal["ok", "warn"] = Field(
        description="'ok' when fit metrics look reasonable for the target's scale; 'warn' otherwise.",
    )
    summary: str = Field(
        description="1-2 sentences on whether the fit looks reasonable, referencing MAE/RMSE/SMAPE where useful.",
    )
    caveat: str = Field(
        description="1 sentence reminding that in-sample errors do not guarantee out-of-sample accuracy.",
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


class ComponentAnalysisOutput(BaseModel):
    """Plain-English interpretation of level / trend / seasonal components."""

    summary: str = Field(
        description="1-2 sentences describing the level path and which components carry meaningful signal.",
    )
    component_signals: list[str] = Field(
        description="2-5 short notes citing level, trend, or seasonal components from the decomposition preview.",
    )
    changepoint_summary: str = Field(
        description="1 sentence on whether the level/trend path is smooth or shifts noticeably over history.",
    )
    caveat: str = Field(
        description="1 sentence reminding that components are model-implied attributions, not causal explanations.",
    )


class ModelImprovementGuidanceOutput(BaseModel):
    """Plain-English suggestions for tuning Holt-Winters further."""

    summary: str = Field(
        description="1-2 sentences on whether the current fit looks sufficient or has room to improve.",
    )
    possible_next_steps: list[str] = Field(
        description="2-5 concrete tuning suggestions (trend/seasonal/damped/seasonal_period). Do not promise improvement.",
    )
    caution: str = Field(
        description="1 sentence reminding the reader these are hypotheses to validate, not proven improvements.",
    )
