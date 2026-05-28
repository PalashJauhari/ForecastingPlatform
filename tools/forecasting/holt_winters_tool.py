# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

from typing import Optional

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.holt_winters_tool import HoltWintersToolInput
from tools.forecasting.holt_winters_model import HoltWintersModel


@tool(
    args_schema=HoltWintersToolInput,
    description="Holt-Winters / exponential smoothing forecast on one regular time series; JSON + CSV tables (no charts).",
)
def holt_winters_tool(
    runtime: ToolRuntime,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    forecast_output_file: str,
    fitted_output_file: str,
    decomposition_output_file: str,
    seasonal_period: Optional[int],
    trend: str,
    seasonal: str,
    damped_trend: bool = False,
) -> str:
    """Fit Holt-Winters / exponential smoothing on one univariate time series and return forecast, fitted, decomposition, diagnostics, and business interpretation as JSON.

    ## Purpose
    Deterministic Holt-Winters pipeline (validate → fit → residuals → forecast with 95% intervals → save tables → interpret).
    Use instead of ``coding_tool`` when the user wants Holt-Winters, Holt's linear trend, or additive/multiplicative exponential smoothing.

    ## When to use
    - One numeric target over time with regular dates (daily/weekly/monthly).
    - User asks for Holt-Winters, exponential smoothing, ETS-style trend + seasonality.
    - Smooth seasonal patterns without ARIMA or Prophet complexity.

    ## When NOT to use
    - ARIMA/SARIMA → ``sarima_tool``.
    - Facebook Prophet decomposition → ``prophet_tool``.
    - Neural nets, custom models, or irregular data → ``coding_tool`` first.
    - Multiple series → out of scope.
    - Charts → tables only; follow with ``coding_tool``.

    ## Input data format (session CSV/XLSX)
    - File in session workspace; basename only in ``file_name``.
    - One row per period; duplicate dates keep first.
    - Regular frequency required. Irregular → ``frequency_not_inferred``; resample first.
    - Target: numeric, no missing values, at least 10 observations; seasonal models need ≥ 2×``seasonal_period`` rows.

    ## Arguments
    - ``file_name`` — e.g. ``sales.csv``.
    - ``date_column`` / ``target_column`` — column names.
    - ``horizon`` — future periods at inferred frequency.
    - ``forecast_output_file`` / ``fitted_output_file`` / ``decomposition_output_file`` — output basenames (.csv or .xlsx).
    - ``seasonal_period`` — required field; ``null`` when ``seasonal='none'``; e.g. 12 for monthly+yearly cycle.
    - ``trend`` — ``none``, ``add``, or ``mul``.
    - ``seasonal`` — ``none``, ``add``, or ``mul``.
    - ``damped_trend`` — default ``false``; set ``true`` to damp trend over the forecast horizon.

    ## Output (JSON string)
    **Success:** ``status``, ``model_type`` ``holt_winters``, ``frequency``, ``model``, ``fit_quality`` (MAE/RMSE/SMAPE),
    ``residual_diagnostics`` (Ljung-Box, Jarque-Bera), three output file paths, previews,
    ``llm_interpretation`` (+ ``component_analysis``), ``warnings``. Forecast columns include ``lower_95`` / ``upper_95``.
    Forecast-period decomposition rows have point forecast only (level/trend/seasonal null).

    **Error:** ``status`` ``error``; ``stage``; ``error.code`` / ``error.message``.

    Present ``llm_interpretation`` to the user; do not narrate file paths.

    ## Example
    ```
    holt_winters_tool(
      file_name="sales.csv", date_column="month", target_column="revenue", horizon=6,
      trend="add", seasonal="add", seasonal_period=12, damped_trend=false,
      forecast_output_file="revenue_forecast.csv",
      fitted_output_file="revenue_fitted.csv",
      decomposition_output_file="revenue_decomposition.csv"
    )
    ```
    """
    params = HoltWintersToolInput(
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        forecast_output_file=forecast_output_file,
        fitted_output_file=fitted_output_file,
        decomposition_output_file=decomposition_output_file,
        seasonal_period=seasonal_period,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
    )
    return HoltWintersModel().run(runtime=runtime, params=params)
