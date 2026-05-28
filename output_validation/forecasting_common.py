"""
Shared Pydantic fields for forecasting tool inputs.

``BaseForecastToolInput`` holds arguments common to ``sarima_tool``,
``prophet_tool``, and ``holt_winters_tool``. Model-specific schemas inherit and add hyperparameters.
Field descriptions are part of the orchestrator tool contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BaseForecastToolInput(BaseModel):
    """Arguments shared by all session forecasting tools."""

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
    forecast_output_file: str = Field(
        description=(
            "Output basename for the future forecast table; "
            "must end with .csv or .xlsx, e.g. 'revenue_forecast.csv'."
        ),
    )
    fitted_output_file: str = Field(
        description=(
            "Output basename for the in-sample actual/fitted/residual table; "
            "must end with .csv or .xlsx, e.g. 'revenue_fitted.csv'."
        ),
    )
