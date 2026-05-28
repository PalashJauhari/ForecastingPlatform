# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

from typing import Optional

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.sarima_tool import SarimaToolInput
from tools.forecasting.sarima_model import SarimaModel


@tool(
    args_schema=SarimaToolInput,
    description="ARIMA/SARIMA forecast on one regular time series; returns JSON + CSV tables (no charts).",
)
def sarima_tool(
    runtime: ToolRuntime,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    seasonal_period: Optional[int],
    forecast_output_file: str,
    fitted_output_file: str,
    use_auto_arima: bool = False,
    order: Optional[list[int]] = None,
    seasonal_order: Optional[list[int]] = None,
) -> str:
    """Fit ARIMA/SARIMA on one univariate time series and return forecast, fitted, diagnostics, and business interpretation as JSON.

    ## Purpose
    Deterministic SARIMAX pipeline (validate → order selection → fit → residuals → forecast with 95% intervals → interpret).
    Use instead of ``coding_tool`` when the user wants ARIMA/SARIMA specifically.

    ## When to use
    - One numeric target over time, regular spacing (daily/weekly/monthly).
    - User asks for ARIMA, SARIMA, auto-ARIMA, or classical forecast with prediction intervals.

    ## When NOT to use
    - Prophet / Holt-Winters / ETS → ``prophet_tool`` or ``holt_winters_tool``.
    - Data needs cleaning or resampling → ``coding_tool`` first.
    - Multiple targets → out of scope.
    - Charts → tables only; follow with ``coding_tool``.

    ## Input data format (session CSV/XLSX)
    - File in session workspace; basename only in ``file_name``.
    - One row per period; duplicate dates keep first.
    - Regular frequency required. Irregular → ``frequency_not_inferred``; resample first.
    - Target: numeric, **no missing values**, at least 10 observations, not constant.

    ## Arguments
    - ``file_name`` — e.g. ``sales.csv``.
    - ``date_column`` / ``target_column`` — column names.
    - ``horizon`` — future periods at inferred frequency.
    - ``seasonal_period`` — required; ``null`` for non-seasonal, or ``m`` (12 monthly+yearly, 4 quarterly, etc.).
    - ``forecast_output_file`` / ``fitted_output_file`` — output basenames (.csv or .xlsx).
    - ``use_auto_arima`` — if true, runs ``pmdarima.auto_arima`` (AICc).
    - ``order`` — optional ``[p, d, q]``.
    - ``seasonal_order`` — optional ``[P, D, Q]``; requires ``seasonal_period``.

    ## Output (JSON string)
    **Success:** ``status``, ``model_type`` ``sarima``, ``frequency``, ``model``, ``fit_quality`` (AIC/AICc/BIC),
    ``residual_diagnostics`` (Ljung-Box, Jarque-Bera), forecast + fitted file paths, previews,
    ``llm_interpretation``, ``warnings``. Forecast table columns: ``calendar_date``, ``forecast``, ``lower_95``, ``upper_95``.

    **Error:** ``status`` ``error``; ``stage``; ``error.code`` / ``error.message``.

    Present ``llm_interpretation`` to the user; do not narrate file paths.

    ## Example
    ```
    sarima_tool(
      file_name="sales.csv", date_column="month", target_column="revenue", horizon=12,
      seasonal_period=12, forecast_output_file="revenue_forecast.csv",
      fitted_output_file="revenue_fitted.csv", use_auto_arima=true
    )
    ```
    """
    params = SarimaToolInput(
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        seasonal_period=seasonal_period,
        forecast_output_file=forecast_output_file,
        fitted_output_file=fitted_output_file,
        use_auto_arima=use_auto_arima,
        order=order,
        seasonal_order=seasonal_order,
    )
    return SarimaModel().run(runtime=runtime, params=params)
