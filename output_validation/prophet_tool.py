"""
Pydantic schemas for ``prophet_tool``.

Two groups live here:

* ``ProphetToolInput`` — the LangChain tool argument schema. Field descriptions
  are what the orchestrator model sees, so they must match the tool docstring.
* ``ResidualAnalysisOutput`` / ``FitQualityOutput`` / ``ForecastSummaryOutput`` /
  ``ComponentAnalysisOutput`` / ``ModelImprovementGuidanceOutput`` — structured
  outputs for the five internal LLM calls that translate the deterministic
  Prophet JSON into business-readable text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tool input
# ---------------------------------------------------------------------------


class ProphetToolInput(BaseModel):
    """Arguments exposed to the orchestrator for ``prophet_tool``."""

    file_name: str = Field(
        description="Bare CSV/XLSX filename in the session workspace, e.g. 'sales.csv'.",
    )
    date_column: str = Field(
        description="Column name in the file containing dates (parsed and renamed to Prophet's required ``ds``).",
    )
    target_column: str = Field(
        description="Numeric column name to forecast (renamed to Prophet's required ``y``).",
    )
    horizon: int = Field(
        gt=0,
        description="Number of future periods to forecast (positive integer).",
    )
    changepoint_prior_scale: float = Field(
        gt=0,
        description=(
            "Prior scale on Prophet's automatic trend changepoints. "
            "Higher (e.g. 0.5) makes the trend more flexible / reactive; lower (e.g. 0.01) makes the trend smoother. "
            "Prophet's default is 0.05; tune up if the trend underfits real shifts, down if it overreacts to noise."
        ),
    )
    seasonality_mode: Literal["additive", "multiplicative"] = Field(
        description=(
            "Prophet seasonality mode. ``additive``: seasonal effects have constant absolute size. "
            "``multiplicative``: seasonal effects scale with the trend level (use when amplitude grows with trend)."
        ),
    )
    weekly_seasonality: bool = Field(
        description="Enable Prophet's built-in weekly seasonality.",
    )
    monthly_seasonality: bool = Field(
        description="Enable an added monthly custom seasonality (period=30.5, fourier_order=5).",
    )
    yearly_seasonality: bool = Field(
        description="Enable Prophet's built-in yearly seasonality.",
    )
    forecast_output_file: str = Field(
        description="Bare filename for the future-only forecast table; must end with .csv or .xlsx.",
    )
    fitted_output_file: str = Field(
        description="Bare filename for the in-sample fitted table (actual + fitted + residual + components); must end with .csv or .xlsx.",
    )
    decomposition_output_file: str = Field(
        description="Bare filename for the combined fitted + forecast decomposition table; must end with .csv or .xlsx.",
    )
    forecast_image_file: str = Field(
        description="Bare filename for the history + future forecast plot; must end with .png or .svg.",
    )


# ---------------------------------------------------------------------------
# LLM interpretation outputs
# ---------------------------------------------------------------------------


class ResidualAnalysisOutput(BaseModel):
    """Plain-English interpretation of residual diagnostics."""

    status: Literal["ok", "warn"] = Field(
        description="'ok' when residuals look acceptable; 'warn' when assumptions are violated.",
    )
    summary: str = Field(
        description="1-2 sentences explaining whether residuals look like noise around the fit.",
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
        description="1-2 sentences on whether the fit looks reasonable, referencing MAE / RMSE / SMAPE where useful.",
    )
    caveat: str = Field(
        description="1 sentence reminding that in-sample errors do not guarantee out-of-sample accuracy.",
    )


class ForecastSummaryOutput(BaseModel):
    """Plain-English interpretation of the forecast trajectory (Prophet has no intervals here)."""

    status: Literal["ok", "warn"] = Field(
        description="'ok' by default; 'warn' if the forecast direction looks implausible (e.g. negative where target is non-negative).",
    )
    summary: str = Field(
        description="1-2 sentences describing the forecast direction (rising / falling / stable / cyclical) over the horizon.",
    )
    business_readout: str = Field(
        description="1 sentence translating the forecast into a clear business takeaway.",
    )


class ComponentAnalysisOutput(BaseModel):
    """Plain-English interpretation of Prophet trend / seasonality / changepoint components."""

    summary: str = Field(
        description="1-2 sentences describing the trend direction and which seasonalities carry meaningful signal.",
    )
    component_signals: list[str] = Field(
        description="2-5 short notes citing the components that drive the forecast (e.g. 'trend rises gradually', 'yearly cycle peaks in December', 'monthly effect is small').",
    )
    changepoint_summary: str = Field(
        description="1 sentence summarising whether the trend is stable or driven by a few notable changepoints, referencing dates if useful.",
    )
    caveat: str = Field(
        description="1 sentence reminding that decompositions are model-implied attributions, not causal explanations.",
    )


class ModelImprovementGuidanceOutput(BaseModel):
    """Plain-English suggestions for tuning the Prophet model further, grounded in provided diagnostics."""

    summary: str = Field(
        description="1-2 sentences describing whether the current fit looks sufficient or has room to improve, citing the diagnostic that drives this view.",
    )
    possible_next_steps: list[str] = Field(
        description="2-5 concrete, actionable tuning suggestions tied to the provided diagnostics (e.g. raise/lower changepoint_prior_scale, toggle a seasonality, switch seasonality_mode). Do not promise improvement.",
    )
    caution: str = Field(
        description="1 sentence reminding the reader these are hypotheses to validate, not proven improvements.",
    )
