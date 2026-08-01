# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

from typing import Optional

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.holt_winters_tool import HoltWintersToolInput
from tools.forecasting.holt_winters_model import HoltWintersModel

HOLT_WINTERS_TOOL_DESCRIPTION = """Fit Holt-Winters / exponential smoothing; return lean pipeline JSON with CSV basenames.

    ## Purpose
    Deterministic Holt-Winters pipeline (validate → fit → residuals → forecast with 95% intervals → decomposition → summary).
    Use instead of ``coding_tool`` when the user wants Holt-Winters or additive/multiplicative exponential smoothing.

    ## When to use
    - One numeric target over time with regular dates (daily/weekly/monthly).
    - User asks for Holt-Winters, exponential smoothing, ETS-style trend + seasonality.
    - Smooth seasonal patterns without ARIMA or Prophet complexity.

    ## When NOT to use
    - ARIMA/SARIMA → ``sarima_tool``.
    - Facebook Prophet decomposition → ``prophet_tool``.
    - Neural nets, custom models, or irregular data → ``coding_tool`` first.
    - Multiple series → out of scope.

    ## Input data format (session CSV/XLSX)
    - File in session workspace; basename only in ``file_name``.
    - One row per period; duplicate dates keep first.
    - Regular frequency required. Irregular → ``frequency_not_inferred``; resample first.
    - Target: numeric, no missing values, at least 10 observations; seasonal models need ≥ 2×``seasonal_period`` rows.

    ## Arguments
    - ``experiment_name`` — unique run label; prefixes all outputs.
    - ``file_name`` — e.g. ``sales.csv``.
    - ``date_column`` / ``target_column`` — column names.
    - ``horizon`` — future periods at inferred frequency.
    - ``seasonal_period`` — required field; ``null`` when ``seasonal='none'``; e.g. 12 for monthly+yearly cycle.
    - ``trend`` — ``none``, ``add``, or ``mul``.
    - ``seasonal`` — ``none``, ``add``, or ``mul``.
    - ``damped_trend`` — default ``false``; set ``true`` to damp trend over the forecast horizon.

    ## Output (JSON string)
    **Success:** ``status``, ``model_type`` ``holt_winters``, ``experiment_name``, ``frequency``, ``warnings``,
    ``pipeline`` including decomposition stages. Forecast columns include ``lower_95`` / ``upper_95``.
    Use ``coding_tool`` for charts. LLM summary stage is not emitted yet — use ``pipeline`` metrics and previews.

    **Error:** ``status`` ``error``; ``stage``; ``error.code`` / ``error.message``.

    ## Example
    ```
    holt_winters_tool(
      experiment_name="revenue_hw_may2026",
      file_name="sales.csv", date_column="month", target_column="revenue", horizon=6,
      trend="add", seasonal="add", seasonal_period=12, damped_trend=false
    )
    ```"""


@tool(description=HOLT_WINTERS_TOOL_DESCRIPTION, args_schema=HoltWintersToolInput)
def holt_winters_tool(
    runtime: ToolRuntime,
    experiment_name: str,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    seasonal_period: Optional[int],
    trend: str,
    seasonal: str,
    damped_trend: bool = False,
) -> str:
    params = HoltWintersToolInput(
        experiment_name=experiment_name,
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        seasonal_period=seasonal_period,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
    )
    return HoltWintersModel().run(runtime=runtime, params=params)
