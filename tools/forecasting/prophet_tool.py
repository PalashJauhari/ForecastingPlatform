"""
Prophet (univariate) forecasting tool for the analysis agent.

Given a CSV/XLSX file in the session workspace plus a date column and a numeric
target column, this tool fits a Prophet model with caller-controlled trend
flexibility (``changepoint_prior_scale``), seasonality mode, and weekly /
monthly / yearly seasonality flags. It then produces:

* an in-sample fitted table (actual / fitted / residual + decomposition),
* a future-only forecast table (forecast + decomposition),
* a combined fitted+forecast decomposition table,
* deterministic fit-quality (MAE / RMSE / SMAPE) and MAD-based residual outlier
  diagnostics, and
* five structured LLM interpretations: residual analysis, fit quality,
  forecast summary, component analysis, and improvement guidance.

This tool deliberately does **not** create images. If the user wants a plot,
the orchestrator should follow up with ``coding_tool``: that tool already
patches ``plt.savefig`` / ``Figure.savefig`` to land under
``agent_filesystem/<session>/run_<tool_call_id>/`` so the UI can show only
the plots produced by that turn.

Design rules:

* The deterministic Prophet pipeline is the source of truth. LLMs only
  translate the resulting JSON into business-readable text.
* Hard input/data errors short-circuit early and return a clean error JSON so
  the orchestrator can decide what to do next.
* Bad residual diagnostics are *warnings*, not errors — the forecast is still
  produced and the LLM interpretation surfaces the caveats.

Outputs (logical paths):

* Forecast table at ``agent_filesystem/<session>/<forecast_output_file>``.
* Fitted table at ``agent_filesystem/<session>/<fitted_output_file>``.
* Decomposition table at ``agent_filesystem/<session>/<decomposition_output_file>``.
* JSON tool response containing model spec, fit quality, residual
  diagnostics, changepoints, output paths, previews, warnings, and the
  ``llm_interpretation`` block.
"""

# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph. With PEP 563
# postponed evaluation, the annotation becomes the string ``"ToolRuntime"``
# instead of the class object, and LangChain silently skips injection — the
# tool then fails at call time with "missing 1 required positional argument:
# 'runtime'". Keep annotations evaluated eagerly here.

import json
import logging
import warnings as _warns
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from prophet import Prophet

from middleware.llm_client import make_llm
from observability.langfuse_handler import trace_context_from_runnable_config, traced_generation, traced_span, update_llm_generation
from output_validation.prophet_tool import (
    ComponentAnalysisOutput,
    FitQualityOutput,
    ForecastSummaryOutput,
    ModelImprovementGuidanceOutput,
    ProphetToolInput,
    ResidualAnalysisOutput,
)
from prompts.prophet_interpretation_prompts import (
    COMPONENT_ANALYSIS_SYSTEM_PROMPT,
    FIT_QUALITY_SYSTEM_PROMPT,
    FORECAST_SUMMARY_SYSTEM_PROMPT,
    MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT,
    RESIDUAL_ANALYSIS_SYSTEM_PROMPT,
)
from session_paths import (
    ensure_session_dirs,
    session_dir_for_paths,
    session_id_from_config,
    session_root,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

# Reuse the orchestrator model for interpretation: it is already configured,
# rate-limited, and available in every environment that runs this tool.
INTERPRETATION_MODEL = cfg["models"].get("orchestrator", "gpt-4o-mini")

# Lower bound for a meaningful Prophet fit; below this we fail fast with a
# friendly message instead of letting Prophet error obscurely.
MIN_OBS = 10

ALLOWED_TABLE_EXTS = {".csv", ".xlsx"}

# Fixed Prophet parameters kept off the tool surface so the orchestrator does
# not have to reason about them on every call. Tunable hyperparameters are
# exposed as tool inputs.
CHANGEPOINT_RANGE = 0.8
SEASONALITY_PRIOR_SCALE = 10.0
MONTHLY_FOURIER_ORDER = 5
DAILY_SEASONALITY = False

# Prophet decomposition columns we surface in fitted / forecast / decomposition
# tables, in the order they should appear when present.
COMPONENT_COLS = ["trend", "weekly", "monthly", "yearly", "additive_terms", "multiplicative_terms"]

# Quiet Prophet / cmdstanpy fit chatter so the tool log stays clean.
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

class ProphetToolError(Exception):
    """Domain error raised by pipeline stages so the wrapper can return clean JSON."""

    # Brief: Store a stable error code, user-facing message, and pipeline stage.
    def __init__(self, code: str, message: str, stage: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


def _structured_llm_call(name: str, schema: type, messages: list, fallback: dict) -> dict:
    llm = make_llm(model=INTERPRETATION_MODEL, temperature=0, output_schema=schema, include_raw=True)
    with traced_generation(name, model=INTERPRETATION_MODEL) as gen:
        try:
            raw = llm.invoke(messages)
            if isinstance(raw, dict) and "parsed" in raw:
                parsed = raw["parsed"]
                raw_msg = raw.get("raw")
            else:
                parsed = raw
                raw_msg = None
            if gen is not None:
                update_llm_generation(gen, model=INTERPRETATION_MODEL, raw=raw_msg if isinstance(raw_msg, AIMessage) else None)
            return parsed.model_dump()
        except Exception as exc:
            if gen is not None:
                gen.update(output={"error": str(exc)})
            return fallback


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


# Brief: Load the file, prepare the Prophet ``ds`` / ``y`` frame, and sanity-check the target.
def data_validation(
    *,
    file_name: str,
    date_column: str,
    target_column: str,
    session_id: str,
) -> tuple[pd.DataFrame, str]:
    """
    End-to-end input preparation for the Prophet pipeline.

    Steps:
    - Check the file exists in the session workspace; otherwise raise a tool error.
    - Read the file via pandas based on its .csv / .xlsx extension.
    - Check the requested date and target columns exist.
    - Parse dates, sort by date, and drop duplicate dates (keep first).
    - Rename to Prophet's required columns: ``ds`` (datetime) and ``y`` (numeric).
    - Infer a regular frequency (needed for ``make_future_dataframe``).
    - Reject a target series with no usable numeric values, too few usable values,
      or a constant usable target. Missing ``y`` rows are kept because Prophet
      tolerates them natively.

    Returns ``(df, freq)`` ready for Prophet fitting.
    """

    # 1. Locate the file inside the session workspace; error if missing.
    name = Path(file_name).name
    path = session_root(session_id) / name
    if not path.exists() or not path.is_file():
        raise ProphetToolError(
            "file_not_found",
            f"File '{name}' was not found in the session workspace.",
            "data_validation",
        )

    # 2. Read the file based on its extension.
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ProphetToolError(
            "unsupported_file_extension",
            f"file_name must end with .csv or .xlsx (got '{suffix or 'no extension'}').",
            "data_validation",
        )

    # 3. Confirm the required date and target columns are present.
    missing_cols = [c for c in (date_column, target_column) if c not in df.columns]
    if missing_cols:
        raise ProphetToolError(
            "missing_required_columns",
            f"Required columns missing in '{name}': {missing_cols}. Available: {list(df.columns)[:20]}",
            "data_validation",
        )

    # 4. Rename to Prophet's required columns and coerce types.
    df = df[[date_column, target_column]].copy()
    df = df.rename(columns={date_column: "ds", target_column: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # 5. Drop unparseable dates, sort by date, drop duplicate dates (keep first).
    df = (
        df.dropna(subset=["ds"])
        .sort_values("ds")
        .drop_duplicates(subset=["ds"], keep="first")
        .reset_index(drop=True)
    )

    # 6. Infer a regular cadence; Prophet's make_future_dataframe needs a freq.
    date_idx = pd.DatetimeIndex(df["ds"].values)
    freq = pd.infer_freq(date_idx)
    if freq is None:
        raise ProphetToolError(
            "frequency_not_inferred",
            "Could not infer a regular frequency from the date column. Resample/clean first or ask the user.",
            "data_validation",
        )

    # 7. Sanity-check the target series. Prophet handles missing y, so we only
    #    require enough non-missing, non-constant numeric values to fit.
    usable = df["y"].dropna()
    if usable.empty:
        raise ProphetToolError(
            "all_target_missing",
            "Target column has no usable numeric values.",
            "data_validation",
        )
    if len(usable) < MIN_OBS:
        raise ProphetToolError(
            "too_few_target_values",
            f"Need at least {MIN_OBS} non-missing target values; got {len(usable)}.",
            "data_validation",
        )
    if usable.nunique() <= 1:
        raise ProphetToolError(
            "constant_target",
            "Target column is constant; Prophet cannot be fitted.",
            "data_validation",
        )

    return df, freq


# Brief: Fit Prophet, build in-sample predictions, and compute fit quality + residual + changepoint summaries.
def fit_prophet_model(
    df: pd.DataFrame,
    *,
    changepoint_prior_scale: float,
    seasonality_mode: str,
    weekly_seasonality: bool,
    monthly_seasonality: bool,
    yearly_seasonality: bool,
) -> tuple[Any, pd.DataFrame, dict, dict, dict]:
    """
    Fit Prophet with the chosen hyperparameters and return everything downstream needs.

    Combines four concerns so the pipeline can move from "we have a series" to
    "we have a fitted model with quality, residual, and changepoint evidence" in
    one step:

    1. Fit: instantiate Prophet with caller-supplied trend/seasonality
       configuration plus fixed internal defaults (``changepoint_range=0.8``,
       ``seasonality_prior_scale=10.0``, ``daily_seasonality=False``); add a
       monthly custom seasonality with ``period=30.5`` and
       ``fourier_order=5`` when requested.
    2. In-sample predict: call ``model.predict(df[["ds"]])`` so the rest of the
       pipeline can build fitted, forecast, and decomposition tables from a
       consistent prediction object.
    3. Fit quality: MAE, RMSE, SMAPE on rows where actual ``y`` is present,
       with inline ``definitions`` so the orchestrator/LLM never has to guess
       what each metric means.
    4. Residual diagnostics: median absolute deviation (MAD), robust sigma,
       ``median ± 3*sigma`` bounds, and the dated residual points outside those
       bounds. Outliers never fail the pipeline — they are surfaced as warnings
       only.
    5. Changepoints: dates of all changepoints Prophet considered, plus the
       top-5 by absolute ``delta`` if Prophet exposes the parameter.

    Returns ``(model, in_sample_forecast, fit_quality, residual_diagnostics, changepoints)``.
    """

    # 1. Configure and fit the Prophet model.
    with traced_span("prophet_tool.fit_model", input={"changepoint_prior_scale": changepoint_prior_scale, "seasonality_mode": seasonality_mode, "weekly_seasonality": weekly_seasonality, "monthly_seasonality": monthly_seasonality, "yearly_seasonality": yearly_seasonality, "n_obs": int(len(df))}) as span:
        try:
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                changepoint_range=CHANGEPOINT_RANGE,
                seasonality_prior_scale=SEASONALITY_PRIOR_SCALE,
                seasonality_mode=seasonality_mode,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                daily_seasonality=DAILY_SEASONALITY,
            )
            if monthly_seasonality:
                model.add_seasonality(name="monthly", period=30.5, fourier_order=MONTHLY_FOURIER_ORDER)

            with _warns.catch_warnings():
                _warns.simplefilter("ignore")
                model.fit(df)

            # In-sample prediction over training dates so we can build fitted,
            # residual, and decomposition tables from one consistent object.
            in_sample = model.predict(df[["ds"]])
            if span is not None:
                span.update(output={"n_obs": int(len(df)), "n_changepoints": int(len(model.changepoints))})
        except Exception as exc:
            if span is not None:
                span.update(output={"error": str(exc)})
            raise ProphetToolError(
                "fit_failed",
                f"Prophet fit failed: {exc}",
                "fit_prophet_model",
            ) from exc

    # 2. Compute fit-quality metrics on rows where actual ``y`` is present.
    actuals = df["y"].to_numpy()
    fitted = in_sample["yhat"].to_numpy()
    mask = ~np.isnan(actuals) & ~np.isnan(fitted)
    actuals_v = actuals[mask]
    fitted_v = fitted[mask]
    n = int(len(actuals_v))

    if n == 0:
        # Should not happen because data_validation enforces usable values, but keep
        # a defensive branch so we still return a coherent payload.
        residuals = np.array([], dtype=float)
        mae = rmse = smape = float("nan")
    else:
        residuals = actuals_v - fitted_v
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        # Symmetric MAPE in percent. Rows where both actual and fitted are zero
        # contribute 0 to avoid division-by-zero noise.
        denom = (np.abs(actuals_v) + np.abs(fitted_v)) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            per_row = np.where(denom == 0, 0.0, np.abs(residuals) / denom)
        smape = float(np.mean(per_row) * 100.0)

    fit_quality = {
        "mae": None if np.isnan(mae) else round(mae, 4),
        "rmse": None if np.isnan(rmse) else round(rmse, 4),
        "smape": None if np.isnan(smape) else round(smape, 4),
        "n_observations": n,
        "definitions": {
            "mae": "Mean absolute error between actual and fitted values; same units as the target; lower is better.",
            "rmse": "Root mean squared error between actual and fitted values; penalises large errors more than MAE; lower is better.",
            "smape": "Symmetric mean absolute percentage error in percent; scale-free comparison; lower is better. Rows where both actual and fitted are zero contribute 0.",
            "n_observations": "Number of in-sample observations used to compute the metrics (rows with non-missing actual and fitted).",
        },
    }

    # 3. Residual diagnostics. Use MAD to flag dated residual points outside
    #    median +/- 3 robust sigma.
    dates_v = pd.to_datetime(df["ds"].to_numpy()[mask])
    if n == 0:
        residual_median = residual_mad = robust_sigma = lower_bound = upper_bound = float("nan")
        outlier_points: list[dict] = []
    else:
        residual_median = float(np.median(residuals))
        absolute_deviation = np.abs(residuals - residual_median)
        residual_mad = float(np.median(absolute_deviation))
        robust_sigma = float(1.4826 * residual_mad)
        lower_bound = float(residual_median - 3.0 * robust_sigma)
        upper_bound = float(residual_median + 3.0 * robust_sigma)
        outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
        outlier_points = [
            {
                "calendar_date": pd.Timestamp(dates_v[idx]).date().isoformat(),
                "actual": None if np.isnan(actuals_v[idx]) else round(float(actuals_v[idx]), 6),
                "fitted": None if np.isnan(fitted_v[idx]) else round(float(fitted_v[idx]), 6),
                "residual": round(float(residuals[idx]), 6),
            }
            for idx in np.where(outlier_mask)[0]
        ]

    diagnostic_warnings: list[str] = []
    if outlier_points:
        diagnostic_warnings.append(
            f"{len(outlier_points)} residual point(s) fall outside the MAD bounds [{lower_bound:.4f}, {upper_bound:.4f}]."
        )

    residual_diagnostics = {
        "status": "warn" if diagnostic_warnings else "pass",
        "n_residuals": n,
        "residual_median": None if np.isnan(residual_median) else round(residual_median, 6),
        "residual_mad": None if np.isnan(residual_mad) else round(residual_mad, 6),
        "robust_sigma": None if np.isnan(robust_sigma) else round(robust_sigma, 6),
        "lower_bound": None if np.isnan(lower_bound) else round(lower_bound, 6),
        "upper_bound": None if np.isnan(upper_bound) else round(upper_bound, 6),
        "outlier_count": len(outlier_points),
        "outlier_fraction": None if n == 0 else round(len(outlier_points) / n, 6),
        "outlier_points": outlier_points,
        "warnings": diagnostic_warnings,
        "definitions": {
            "status": "'pass' when no residual points fall outside the MAD bounds; 'warn' when one or more residual outliers are found.",
            "n_residuals": "Number of residuals available for the MAD outlier check.",
            "residual_median": "Median residual; the robust center of model errors.",
            "residual_mad": "Median absolute deviation of residuals from the residual median; robust spread measure.",
            "robust_sigma": "MAD converted to an approximate standard-deviation scale using 1.4826 * MAD.",
            "lower_bound": "Lower residual outlier bound: residual_median - 3 * robust_sigma.",
            "upper_bound": "Upper residual outlier bound: residual_median + 3 * robust_sigma.",
            "outlier_count": "Number of fitted residual points outside the MAD bounds.",
            "outlier_fraction": "Share of residual points outside the MAD bounds.",
            "outlier_points": "Dated fitted observations whose residuals fall outside the MAD bounds.",
            "warnings": "Plain-English notes about MAD residual outliers, if any.",
        },
    }

    # 4. Extract changepoint summary. Prophet stores ``delta`` for trend rate
    #    changes per changepoint; for MAP fits the array is shape (1, n).
    changepoint_dates: list[str] = []
    largest_delta_changepoints: list[dict] = []
    try:
        changepoint_dates = [pd.Timestamp(c).date().isoformat() for c in model.changepoints]
        delta_raw = model.params.get("delta")
        if delta_raw is not None and changepoint_dates:
            deltas_arr = np.asarray(delta_raw, dtype=float)
            if deltas_arr.ndim > 1:
                deltas = deltas_arr.mean(axis=0)
            else:
                deltas = deltas_arr
            if len(deltas) == len(changepoint_dates):
                # Top-5 by absolute delta so the response stays compact.
                order_idx = np.argsort(-np.abs(deltas))[:5]
                for idx in order_idx:
                    largest_delta_changepoints.append(
                        {
                            "date": changepoint_dates[int(idx)],
                            "delta": float(round(float(deltas[int(idx)]), 6)),
                        }
                    )
    except Exception:
        # If anything fails, keep the dates we already collected and skip the ranking.
        pass

    changepoints = {
        "changepoint_range": CHANGEPOINT_RANGE,
        "changepoint_dates": changepoint_dates,
        "largest_delta_changepoints": largest_delta_changepoints,
    }

    return model, in_sample, fit_quality, residual_diagnostics, changepoints


# Brief: Build the future-only Prophet forecast frame for the requested horizon.
def generate_forecast(model: Any, *, df: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    """Return Prophet's future-only ``predict`` frame at the inferred frequency."""
    try:
        future = model.make_future_dataframe(periods=int(horizon), freq=freq, include_history=False)
        forecast = model.predict(future)
        return forecast
    except Exception as exc:
        raise ProphetToolError(
            "forecast_failed",
            f"Failed to generate forecast: {exc}",
            "generate_forecast",
        ) from exc


# Brief: Save a tabular DataFrame at the session root and return its logical path.
def save_table(table: pd.DataFrame, *, session_id: str, output_file: str, stage: str) -> str:
    """Write ``table`` as CSV/XLSX at the session root only — never PNG/SVG (use ``coding_tool`` for charts)."""
    name = Path(output_file).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_TABLE_EXTS:
        raise ProphetToolError(
            "invalid_output_file",
            f"{stage} output file must end with .csv or .xlsx (got '{suffix or 'no extension'}').",
            stage,
        )

    ensure_session_dirs(session_id)
    out_path = session_root(session_id) / name
    if suffix == ".csv":
        table.to_csv(out_path, index=False)
    else:
        table.to_excel(out_path, index=False)

    sid = session_dir_for_paths(session_id)
    return f"agent_filesystem/{sid}/{name}"


# Brief: Ask the LLM to turn deterministic Prophet diagnostics into user-facing summaries.
def run_llm_interpretations(
    *,
    fit_quality: dict,
    residual_diagnostics: dict,
    forecast_rows: list[dict],
    decomposition_rows: list[dict],
    changepoints: dict,
    model_spec: dict,
) -> dict:
    """
    Five small structured LLM calls that translate deterministic JSON into prose.

    If a single call fails (rate limit, schema parse error, …) the tool still
    returns a usable response — the failed sub-call gets a deterministic
    placeholder so the orchestrator can surface the rest.
    """
    interpretations: dict[str, dict] = {}

    residual_payload = json.dumps({"model": model_spec, "residual_diagnostics": residual_diagnostics}, indent=2, default=str)
    residual_messages = [SystemMessage(content=RESIDUAL_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=residual_payload)]
    interpretations["residual_analysis"] = _structured_llm_call(
        "prophet_tool.llm.residual_analysis",
        ResidualAnalysisOutput,
        residual_messages,
        {"status": residual_diagnostics.get("status", "warn"), "summary": "LLM interpretation failed; review the raw residual diagnostics.", "caveat": ""},
    )

    fit_payload = json.dumps({"model": model_spec, "fit_quality": fit_quality, "residual_diagnostics": residual_diagnostics}, indent=2, default=str)
    fit_messages = [SystemMessage(content=FIT_QUALITY_SYSTEM_PROMPT), HumanMessage(content=fit_payload)]
    interpretations["fit_quality"] = _structured_llm_call(
        "prophet_tool.llm.fit_quality",
        FitQualityOutput,
        fit_messages,
        {"status": "warn", "summary": "LLM interpretation failed; review the raw fit-quality metrics.", "caveat": ""},
    )

    forecast_preview = forecast_rows[:24]
    forecast_payload = json.dumps({"model": model_spec, "horizon": len(forecast_rows), "forecast_preview": forecast_preview}, indent=2, default=str)
    forecast_messages = [SystemMessage(content=FORECAST_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=forecast_payload)]
    interpretations["forecast_summary"] = _structured_llm_call(
        "prophet_tool.llm.forecast_summary",
        ForecastSummaryOutput,
        forecast_messages,
        {"status": "warn", "summary": "LLM interpretation failed; review the raw forecast values.", "business_readout": ""},
    )

    decomposition_preview = decomposition_rows[:24]
    component_payload = json.dumps({"model": model_spec, "changepoints": changepoints, "decomposition_preview": decomposition_preview}, indent=2, default=str)
    component_messages = [SystemMessage(content=COMPONENT_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=component_payload)]
    interpretations["component_analysis"] = _structured_llm_call(
        "prophet_tool.llm.component_analysis",
        ComponentAnalysisOutput,
        component_messages,
        {"summary": "LLM interpretation failed; review the raw decomposition table.", "component_signals": [], "changepoint_summary": "", "caveat": ""},
    )

    guidance_payload = json.dumps(
        {"model": model_spec, "fit_quality": fit_quality, "residual_diagnostics": residual_diagnostics, "changepoints": changepoints, "forecast_summary": {"horizon": len(forecast_rows), "forecast_preview": forecast_preview}},
        indent=2,
        default=str,
    )
    guidance_messages = [SystemMessage(content=MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT), HumanMessage(content=guidance_payload)]
    interpretations["model_improvement_guidance"] = _structured_llm_call(
        "prophet_tool.llm.model_improvement_guidance",
        ModelImprovementGuidanceOutput,
        guidance_messages,
        {"summary": "LLM guidance failed; review the raw fit quality, residual diagnostics, and changepoints to decide next steps.", "possible_next_steps": [], "caution": ""},
    )

    return interpretations


# Brief: Assemble the final JSON payload returned to the orchestrator.
def build_response(
    *,
    file_name: str,
    date_column: str,
    target_column: str,
    freq: str,
    model_spec: dict,
    fit_quality: dict,
    residual_diagnostics: dict,
    changepoints: dict,
    forecast_rows: list[dict],
    fitted_rows: list[dict],
    decomposition_rows: list[dict],
    forecast_path: str,
    fitted_path: str,
    decomposition_path: str,
    llm_interpretation: dict,
    warnings_out: list[dict],
) -> str:
    """Pack everything into the JSON string the orchestrator will see as the tool result."""
    has_warnings = bool(warnings_out)
    response = {
        "status": "success_with_warnings" if has_warnings else "success",
        "file_name": file_name,
        "date_column": date_column,
        "target_column": target_column,
        "frequency": freq,
        "model": model_spec,
        "fit_quality": fit_quality,
        "residual_diagnostics": residual_diagnostics,
        "changepoints": changepoints,
        "forecast_output_file": forecast_path,
        "fitted_output_file": fitted_path,
        "decomposition_output_file": decomposition_path,
        "horizon": len(forecast_rows),
        # Keep the wire small: orchestrator gets short previews; full tables are on disk.
        "forecast_preview": forecast_rows[:12],
        "fitted_preview": fitted_rows[-12:],
        "decomposition_preview": decomposition_rows[-12:],
        "llm_interpretation": llm_interpretation,
        "warnings": warnings_out,
    }
    return json.dumps(response, default=str)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_prophet_pipeline(
    *,
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
    runtime: ToolRuntime,
) -> str:
    """
    End-to-end Prophet pipeline. Returns a JSON string (success or error).

    Any pipeline-level error short-circuits to a clean error JSON so the
    orchestrator can decide what to do next (e.g. ask the user for a missing
    column, request a different frequency, fall back to ``coding_tool``).
    """
    session_id = session_id_from_config(runtime.config)
    trace_context = trace_context_from_runnable_config(runtime.config)

    with traced_span("prophet_tool", trace_context=trace_context, metadata={"session_id": session_id}) as tool_span:
        warnings_out: list[dict] = []
        try:
            # 1. Load and validate the source file, then turn it into a Prophet-ready frame.
            #    data_validation performs all of the following:
            #    - Confirms the file exists in the session workspace; otherwise raises a tool error.
            #    - Reads the file via pandas based on its .csv / .xlsx extension.
            #    - Confirms the requested date and target columns are present.
            #    - Renames them to Prophet's required ``ds`` and ``y`` columns and coerces types.
            #    - Sorts by date and drops duplicate dates (keeps the first row per date).
            #    - Infers a regular cadence with pd.infer_freq; needed for make_future_dataframe.
            #    - Rejects a target series with no usable numeric values, too few usable values,
            #      or a constant target. Missing ``y`` rows are kept because Prophet handles them.
            #    - Returns the model-ready DataFrame and its inferred frequency string.
            df, freq = data_validation(file_name=file_name, date_column=date_column, target_column=target_column, session_id=session_id)

            # 2. Fit Prophet and collect fit-quality + residual + changepoint summaries in one stage.
            model, in_sample, fit_quality, residual_diagnostics, changepoints = fit_prophet_model(
                df,
                changepoint_prior_scale=float(changepoint_prior_scale),
                seasonality_mode=seasonality_mode,
                weekly_seasonality=bool(weekly_seasonality),
                monthly_seasonality=bool(monthly_seasonality),
                yearly_seasonality=bool(yearly_seasonality),
            )

            # 3. Build the model-spec block surfaced in the response (visible config).
            model_spec = {
                "model_type": "prophet",
                "changepoint_prior_scale": float(changepoint_prior_scale),
                "changepoint_range": CHANGEPOINT_RANGE,
                "seasonality_prior_scale": SEASONALITY_PRIOR_SCALE,
                "seasonality_mode": seasonality_mode,
                "weekly_seasonality": bool(weekly_seasonality),
                "monthly_seasonality": bool(monthly_seasonality),
                "yearly_seasonality": bool(yearly_seasonality),
                "monthly_fourier_order": MONTHLY_FOURIER_ORDER if monthly_seasonality else None,
                "daily_seasonality": DAILY_SEASONALITY,
            }

            # 4. Generate the requested future-only forecast at the inferred frequency.
            forecast_future = generate_forecast(model, df=df, horizon=int(horizon), freq=freq)

            # 5. Build the three tables (forecast / fitted / decomposition).
            #    Component columns are intersected with what Prophet actually returned so we
            #    do not write empty columns when a seasonality is disabled.
            fitted_components_present = [c for c in COMPONENT_COLS if c in in_sample.columns]
            forecast_components_present = [c for c in COMPONENT_COLS if c in forecast_future.columns]

            # 5a. Forecast rows: future only, with components.
            forecast_table = forecast_future[["ds", "yhat"] + forecast_components_present].copy()
            forecast_table = forecast_table.rename(columns={"ds": "calendar_date", "yhat": "forecast"})
            forecast_table["calendar_date"] = pd.to_datetime(forecast_table["calendar_date"]).dt.date.astype(str)

            # 5b. Fitted rows: in-sample with actual / fitted / residual + components.
            fitted_table = pd.merge(df[["ds", "y"]], in_sample[["ds", "yhat"] + fitted_components_present], on="ds", how="left")
            fitted_table["residual"] = fitted_table["y"] - fitted_table["yhat"]
            fitted_table = fitted_table.rename(columns={"ds": "calendar_date", "y": "actual", "yhat": "fitted"})
            fitted_table = fitted_table[["calendar_date", "actual", "fitted", "residual"] + fitted_components_present]
            fitted_table["calendar_date"] = pd.to_datetime(fitted_table["calendar_date"]).dt.date.astype(str)

            # 5c. Decomposition rows: combined fitted + forecast for direct comparison.
            all_components_present = [c for c in COMPONENT_COLS if c in fitted_components_present or c in forecast_components_present]
            decomposition_fit = pd.merge(df[["ds", "y"]], in_sample[["ds", "yhat"] + fitted_components_present], on="ds", how="left")
            decomposition_fit = decomposition_fit.rename(columns={"y": "actual"})
            decomposition_fit["period_type"] = "fitted"
            decomposition_fcst = forecast_future[["ds", "yhat"] + forecast_components_present].copy()
            decomposition_fcst["actual"] = np.nan
            decomposition_fcst["period_type"] = "forecast"
            decomposition_table = pd.concat([decomposition_fit, decomposition_fcst], ignore_index=True)
            decomposition_table = decomposition_table.reindex(columns=["ds", "period_type", "actual", "yhat"] + all_components_present)
            decomposition_table = decomposition_table.rename(columns={"ds": "calendar_date"})
            decomposition_table["calendar_date"] = pd.to_datetime(decomposition_table["calendar_date"]).dt.date.astype(str)

            # 6. Persist the three tables for downstream display/download. Charts are
            #    intentionally not produced here; ``coding_tool`` is the single source of
            #    plot artifacts (it routes plt.savefig into a per-tool-call run folder).
            forecast_path = save_table(forecast_table, session_id=session_id, output_file=forecast_output_file, stage="save_forecast")
            fitted_path = save_table(fitted_table, session_id=session_id, output_file=fitted_output_file, stage="save_fitted")
            decomposition_path = save_table(decomposition_table, session_id=session_id, output_file=decomposition_output_file, stage="save_decomposition")

            # 7. Convert the tables to JSON-friendly row lists for previews and LLM prompts.
            forecast_rows = forecast_table.to_dict(orient="records")
            fitted_rows = fitted_table.to_dict(orient="records")
            decomposition_rows = decomposition_table.to_dict(orient="records")

            # 8. Ask the LLM to summarize residuals, fit quality, forecast, components, and improvement guidance.
            llm_interpretation = run_llm_interpretations(
                fit_quality=fit_quality,
                residual_diagnostics=residual_diagnostics,
                forecast_rows=forecast_rows,
                decomposition_rows=decomposition_rows,
                changepoints=changepoints,
                model_spec=model_spec,
            )

            # 9. Bubble residual diagnostic warnings up to the top-level warnings array so the
            #    orchestrator does not have to dig into nested diagnostics.
            for w in residual_diagnostics.get("warnings", []) or []:
                warnings_out.append({"code": "residual_assumption", "message": str(w)})

            # 10. Pack everything into the JSON response the orchestrator will see.
            response = build_response(
                file_name=file_name,
                date_column=date_column,
                target_column=target_column,
                freq=freq,
                model_spec=model_spec,
                fit_quality=fit_quality,
                residual_diagnostics=residual_diagnostics,
                changepoints=changepoints,
                forecast_rows=forecast_rows,
                fitted_rows=fitted_rows,
                decomposition_rows=decomposition_rows,
                forecast_path=forecast_path,
                fitted_path=fitted_path,
                decomposition_path=decomposition_path,
                llm_interpretation=llm_interpretation,
                warnings_out=warnings_out,
            )

            if tool_span is not None:
                tool_span.update(metadata={"changepoint_prior_scale": str(changepoint_prior_scale), "seasonality_mode": seasonality_mode, "weekly": str(weekly_seasonality), "monthly": str(monthly_seasonality), "yearly": str(yearly_seasonality), "warnings": str(len(warnings_out))})
            return response

        except ProphetToolError as exc:
            if tool_span is not None:
                tool_span.update(metadata={"status": "error", "stage": exc.stage, "code": exc.code})
            return json.dumps(
                {
                    "status": "error",
                    "stage": exc.stage,
                    "error": {"code": exc.code, "message": exc.message},
                },
                default=str,
            )
        except Exception as exc:
            if tool_span is not None:
                tool_span.update(metadata={"status": "error", "stage": "unknown", "code": "internal_error"})
            return json.dumps(
                {
                    "status": "error",
                    "stage": "unknown",
                    "error": {"code": "internal_error", "message": str(exc)[:500]},
                },
                default=str,
            )


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------


@tool(args_schema=ProphetToolInput)
# Brief: Public LangChain tool wrapper that exposes Prophet forecasting to the agent.
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
    """Fit a univariate Prophet model on a tabular session time series and return forecast + fitted + decomposition + diagnostics + LLM interpretation as JSON.

    ## When to use
    The user wants a univariate time-series forecast from a clean tabular file
    with one date column and one numeric target column, and Prophet's
    trend + seasonality decomposition (weekly / monthly / yearly) is a good
    fit. Prefer this over ``coding_tool`` when the analysis is "Prophet
    forecast + decomposition" rather than custom modelling.

    ## When NOT to use
    - User wants ARIMA/SARIMA — use ``sarima_tool``.
    - User wants ETS, neural nets, or any non-Prophet model — write code via ``coding_tool``.
    - Data still needs cleaning, joining, or resampling before fitting — handle that in ``coding_tool`` first, then call this tool on the cleaned file.
    - Multiple targets or panel structure — out of scope for this tool.
    - User wants a chart of the forecast — this tool does not generate images;
      follow up with ``coding_tool`` (it is the only image-producing tool).

    ## Required inputs
    - ``file_name`` — bare CSV/XLSX filename in the session workspace.
    - ``date_column`` — column with dates (renamed to Prophet's ``ds``).
    - ``target_column`` — numeric column to forecast (renamed to Prophet's ``y``).
    - ``horizon`` — positive integer number of future periods.
    - ``changepoint_prior_scale`` — positive float controlling trend flexibility (Prophet default 0.05). Higher = more reactive trend; lower = smoother trend.
    - ``seasonality_mode`` — ``"additive"`` (constant seasonal amplitude) or ``"multiplicative"`` (amplitude scales with trend level).
    - ``weekly_seasonality`` / ``monthly_seasonality`` / ``yearly_seasonality`` — booleans that enable Prophet's built-in weekly / yearly cycles and an added monthly custom seasonality (period=30.5, fourier_order=5).
    - ``forecast_output_file`` — bare filename ending in ``.csv`` or ``.xlsx`` for the future forecast table (with components).
    - ``fitted_output_file`` — bare filename ending in ``.csv`` or ``.xlsx`` for the in-sample fitted table (actual + fitted + residual + components).
    - ``decomposition_output_file`` — bare filename ending in ``.csv`` or ``.xlsx`` for the combined fitted + forecast decomposition.

    ## Behaviour
    - ``changepoint_range`` is fixed at 0.8 (changepoints sought only in the first 80% of history).
    - ``seasonality_prior_scale`` is fixed at 10.0 (Prophet default).
    - ``daily_seasonality`` is fixed off; ``monthly_fourier_order`` is fixed at 5.
    - Hard data errors (missing file, missing columns, non-regular spacing, no usable numeric target, constant target) return a clean error JSON so you can ask the user or run cleaning first. Missing target rows are tolerated.
    - Bad residual diagnostics are returned as warnings, not errors — the forecast is still produced.

    ## Output
    A JSON string with: ``status``, ``file_name``, ``date_column``, ``target_column``, ``frequency``, ``model``, ``fit_quality`` (MAE/RMSE/SMAPE/n_observations plus inline ``definitions``), ``residual_diagnostics`` (MAD residual bounds, outlier count/fraction, dated outlier points, plain-English ``warnings``, plus inline ``definitions``), ``changepoints`` (``changepoint_range``, dates, top-5 by absolute delta), ``forecast_output_file``, ``fitted_output_file``, ``decomposition_output_file``, ``horizon``, ``forecast_preview``, ``fitted_preview``, ``decomposition_preview``, ``llm_interpretation`` (``residual_analysis``, ``fit_quality``, ``forecast_summary``, ``component_analysis``, ``model_improvement_guidance`` with concrete ``possible_next_steps``), and ``warnings``. When showing results to the user, present the LLM interpretation; do not narrate file paths.
    """
    return run_prophet_pipeline(
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
        runtime=runtime,
    )
