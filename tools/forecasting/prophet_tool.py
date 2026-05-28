# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

from typing import Optional

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.prophet_tool import ProphetToolInput
from tools.forecasting.prophet_model import ProphetModel


@tool(args_schema=ProphetToolInput)
def prophet_tool(
    runtime: ToolRuntime,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    changepoint_prior_scale: float,
    seasonality_mode: str,
    weekly_seasonality: bool,
    monthly_seasonality: bool,
    yearly_seasonality: bool,
    forecast_output_file: str,
    fitted_output_file: str,
    decomposition_output_file: str,
) -> str:
    """Fit Facebook Prophet on one univariate time series and return forecast, fitted, decomposition, diagnostics, and business interpretation as JSON.

    ## Purpose
    Deterministic Prophet pipeline (validate → fit → residuals → forecast → save tables → interpret).
    Use instead of ``coding_tool`` when Prophet trend + seasonality decomposition is appropriate.

    ## When to use
    - One numeric target over time with roughly regular dates.
    - User wants Prophet, trend/seasonality breakdown, or changepoint-aware forecasting.
    - Some missing target values are OK (Prophet handles NaN in ``y``).

    ## When NOT to use
    - ARIMA/SARIMA → ``sarima_tool``.
    - Holt-Winters / exponential smoothing / ETS → ``holt_winters_tool``.
    - Cleaning/joining/resampling needed → ``coding_tool`` first.
    - Multiple series → out of scope.
    - Charts → tables only; follow with ``coding_tool``.

    ## Input data format (session CSV/XLSX)
    - File must exist in the session workspace (upload first).
    - One row per period; duplicate dates keep the first row.
    - Regular calendar frequency required (tool infers freq). Irregular spacing → ``frequency_not_inferred``.
    - At least 10 non-missing numeric target values; not constant.
    - Example columns: ``month`` (dates), ``revenue`` (numeric target).

    ## Arguments
    - ``file_name`` — basename only, e.g. ``sales.csv``.
    - ``date_column`` / ``target_column`` — column names in the file.
    - ``horizon`` — future periods at inferred frequency (12 = next 12 months if monthly).
    - ``changepoint_prior_scale`` — trend flexibility; 0.05 default-like; higher = more reactive.
    - ``seasonality_mode`` — ``additive`` or ``multiplicative``.
    - ``weekly_seasonality`` / ``monthly_seasonality`` / ``yearly_seasonality`` — booleans.
    - ``forecast_output_file`` — future forecast table (.csv or .xlsx).
    - ``fitted_output_file`` — in-sample actual/fitted/residual table.
    - ``decomposition_output_file`` — combined fitted + forecast decomposition.

    ## Output (JSON string)
    **Success:** ``status``, ``model_type`` ``prophet``, ``frequency``, ``model``, ``fit_quality`` (MAE/RMSE/SMAPE),
    ``residual_diagnostics`` (MAD bounds), ``changepoints``, three output file paths, previews,
    ``llm_interpretation`` (+ ``component_analysis``), ``warnings``.

    **Error:** ``status`` ``error``; ``stage``; ``error.code`` / ``error.message``.
    Common codes: ``file_not_found``, ``missing_required_columns``, ``frequency_not_inferred``, ``constant_target``.

    Present ``llm_interpretation`` to the user; do not narrate file paths.

    ## Example
    ```
    prophet_tool(
      file_name="sales.csv", date_column="month", target_column="revenue", horizon=6,
      changepoint_prior_scale=0.05, seasonality_mode="multiplicative",
      weekly_seasonality=false, monthly_seasonality=true, yearly_seasonality=true,
      forecast_output_file="revenue_forecast.csv",
      fitted_output_file="revenue_fitted.csv",
      decomposition_output_file="revenue_decomposition.csv"
    )
    ```
    Then chart via ``coding_tool`` reading the forecast/fitted CSVs.
    """
    params = ProphetToolInput(
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly_seasonality,
        monthly_seasonality=monthly_seasonality,
        yearly_seasonality=yearly_seasonality,
        forecast_output_file=forecast_output_file,
        fitted_output_file=fitted_output_file,
        decomposition_output_file=decomposition_output_file,
    )
    return ProphetModel().run(runtime=runtime, params=params)
