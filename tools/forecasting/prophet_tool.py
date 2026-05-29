# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.prophet_tool import ProphetToolInput
from tools.forecasting.prophet_model import ProphetModel


@tool(args_schema=ProphetToolInput)
def prophet_tool(
    runtime: ToolRuntime,
    experiment_name: str,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    changepoint_prior_scale: float,
    seasonality_mode: str,
    weekly_seasonality: bool,
    monthly_seasonality: bool,
    yearly_seasonality: bool,
) -> str:
    """Fit Facebook Prophet on one univariate time series; return lean pipeline JSON with CSV/PNG basenames.

    ## Purpose
    Deterministic Prophet pipeline (validate → fit → residuals → forecast → decomposition → plots → summary).
    Use instead of ``coding_tool`` when Prophet trend + seasonality decomposition is appropriate.

    ## When to use
    - One numeric target over time with roughly regular dates.
    - User wants Prophet, trend/seasonality breakdown, or changepoint-aware forecasting.
    - Some missing target values are OK (Prophet handles NaN in ``y``).

    ## When NOT to use
    - ARIMA/SARIMA → ``sarima_tool``.
    - Holt-Winters / exponential smoothing → ``holt_winters_tool``.
    - Cleaning/joining/resampling needed → ``coding_tool`` first.
    - Multiple series → out of scope.

    ## Input data format (session CSV/XLSX)
    - File must exist in the session workspace (upload first).
    - One row per period; duplicate dates keep the first row.
    - Regular calendar frequency required (tool infers freq). Irregular spacing → ``frequency_not_inferred``.
    - At least 10 non-missing numeric target values; not constant.

    ## Arguments
    - ``experiment_name`` — unique run label; prefixes all outputs.
    - ``file_name`` — basename only, e.g. ``sales.csv``.
    - ``date_column`` / ``target_column`` — column names in the file.
    - ``horizon`` — future periods at inferred frequency (12 = next 12 months if monthly).
    - ``changepoint_prior_scale`` — trend flexibility; 0.05 default-like; higher = more reactive.
    - ``seasonality_mode`` — ``additive`` or ``multiplicative``.
    - ``weekly_seasonality`` / ``monthly_seasonality`` / ``yearly_seasonality`` — booleans.

    ## Output (JSON string)
    **Success:** ``status``, ``model_type`` ``prophet``, ``experiment_name``, ``frequency``, ``warnings``,
    ``pipeline`` including fitted/forecast decomposition previews and decomposition plot.
    Artifacts: ``{experiment_name}_fitted.csv``, ``_forecast.csv``, ``_decomposition.csv``, and matching PNGs.
    LLM summary stage is not emitted yet — use ``pipeline`` metrics, previews, and plots.

    **Error:** ``status`` ``error``; ``stage``; ``error.code`` / ``error.message``.

    ## Example
    ```
    prophet_tool(
      experiment_name="revenue_prophet_may2026",
      file_name="sales.csv", date_column="month", target_column="revenue", horizon=6,
      changepoint_prior_scale=0.05, seasonality_mode="multiplicative",
      weekly_seasonality=false, monthly_seasonality=true, yearly_seasonality=true
    )
    ```
    """
    params = ProphetToolInput(
        experiment_name=experiment_name,
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly_seasonality,
        monthly_seasonality=monthly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    return ProphetModel().run(runtime=runtime, params=params)
