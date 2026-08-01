# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

from typing import Optional

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.sarima_tool import SarimaToolInput
from tools.forecasting.sarima_model import SarimaModel

SARIMA_TOOL_DESCRIPTION = """Fit ARIMA/SARIMA on one univariate time series; return lean pipeline JSON with CSV basenames.

    ## Purpose
    Deterministic SARIMAX pipeline (validate → order selection → fit → residuals → forecast with 95% intervals → summary).
    Use instead of ``coding_tool`` when the user wants ARIMA/SARIMA specifically.

    ## When to use
    - One numeric target over time, regular spacing (daily/weekly/monthly).
    - User asks for ARIMA, SARIMA, auto-ARIMA, or classical forecast with prediction intervals.

    ## When NOT to use
    - Prophet / Holt-Winters → ``prophet_tool`` or ``holt_winters_tool``.
    - Data needs cleaning or resampling → ``coding_tool`` first.
    - Multiple targets → out of scope.

    ## Input data format (session CSV/XLSX)
    - File in session workspace; basename only in ``file_name``.
    - One row per period; duplicate dates keep first.
    - Regular frequency required. Irregular → ``frequency_not_inferred``; resample first.
    - Target: numeric, **no missing values**, at least 10 observations, not constant.

    ## Arguments
    - ``experiment_name`` — unique run label; prefixes all outputs (e.g. ``revenue_q1_sarima``).
    - ``file_name`` — e.g. ``sales.csv``.
    - ``date_column`` / ``target_column`` — column names.
    - ``horizon`` — future periods at inferred frequency.
    - ``seasonal_period`` — required; ``null`` for non-seasonal, or ``m`` (12 monthly+yearly, 4 quarterly, etc.).
    - ``use_auto_arima`` — if true, runs ``pmdarima.auto_arima`` (AICc).
    - ``order`` — optional ``[p, d, q]``.
    - ``seasonal_order`` — optional ``[P, D, Q]``; requires ``seasonal_period``.

    ## Output (JSON string)
    **Success:** ``status``, ``model_type`` ``sarima``, ``experiment_name``, ``frequency``, ``warnings``,
    ``pipeline`` (stages: data_validation, model_fit, fitted_values, forecast_values, residual_analysis).
    File references are basenames only. Use ``coding_tool`` for charts.
    LLM summary stage is not emitted yet — use ``pipeline`` metrics and previews.

    **Error:** ``status`` ``error``; ``stage``; ``error.code`` / ``error.message``.

    ## Example
    ```
    sarima_tool(
      experiment_name="revenue_sarima_may2026",
      file_name="sales.csv", date_column="month", target_column="revenue", horizon=12,
      seasonal_period=12, use_auto_arima=true
    )
    ```"""


@tool(description=SARIMA_TOOL_DESCRIPTION, args_schema=SarimaToolInput)
def sarima_tool(
    runtime: ToolRuntime,
    experiment_name: str,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    seasonal_period: Optional[int],
    use_auto_arima: bool = False,
    order: Optional[list[int]] = None,
    seasonal_order: Optional[list[int]] = None,
) -> str:
    params = SarimaToolInput(
        experiment_name=experiment_name,
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        seasonal_period=seasonal_period,
        use_auto_arima=use_auto_arima,
        order=order,
        seasonal_order=seasonal_order,
    )
    return SarimaModel().run(runtime=runtime, params=params)
