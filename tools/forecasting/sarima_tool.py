"""
SARIMA/ARIMA forecasting tool for the analysis agent.

Given a CSV/XLSX file in the session workspace plus a date column and a numeric
target column, this tool fits an ARIMA or SARIMA model (manual order or auto
via ``pmdarima.auto_arima``), runs residual diagnostics, generates a forecast
with 95% prediction intervals, and produces an LLM-written interpretation of
residuals, fit quality, and the forecast.

This tool deliberately does **not** create images. If the user wants a plot,
the orchestrator should follow up with ``coding_tool``: that tool already
patches ``plt.savefig`` / ``Figure.savefig`` to land under
``agent_filesystem/<session>/run_<tool_call_id>/`` so the UI can show only
the plots produced by that turn.

Design rules:

* The deterministic statistics pipeline is the source of truth. LLMs only
  translate the resulting JSON into business-readable text.
* Hard input/data errors short-circuit early and return a clean error JSON so
  the orchestrator can decide what to do next (e.g. ask the user).
* Residual-assumption violations are *warnings*, not errors — the forecast is
  still produced and the LLM interpretation surfaces the caveats.

Outputs:

* Forecast table at ``agent_filesystem/<session>/<forecast_output_file>``
  (CSV or XLSX).
* JSON tool response containing model spec, fit quality, residual
  diagnostics, forecast preview, output path, warnings, and the
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
import warnings as _warns
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pmdarima as pm
import yaml
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

from middleware.llm_client import make_llm
from observability.langfuse_handler import traced_generation, traced_span, update_llm_generation
from output_validation.sarima_tool import (
    FitQualityOutput,
    ForecastSummaryOutput,
    ModelImprovementGuidanceOutput,
    ResidualAnalysisOutput,
    SarimaToolInput,
)
from prompts.sarima_interpretation_prompts import (
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

# Lower bound for a meaningful ARIMA fit. Statsmodels will surface its own
# error if the chosen order is too rich for the available history; this is
# just a cheap up-front guard so we fail fast with a friendly message.
MIN_OBS = 10

# 95% prediction intervals.
DEFAULT_ALPHA = 0.05

ALLOWED_TABLE_EXTS = {".csv", ".xlsx"}


class SarimaToolError(Exception):
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


# Brief: Load the file, prepare a clean time index, and sanity-check the target series.
def data_validation(
    *,
    file_name: str,
    date_column: str,
    target_column: str,
    session_id: str,
) -> tuple[pd.Series, str]:
    """
    End-to-end input preparation for the SARIMA pipeline.

    Steps:
    - Check the file exists in the session workspace; otherwise raise a tool error.
    - Check the requested date and target columns exist.
    - Sort by date and drop duplicate dates (keep first).
    - Infer a regular frequency (ARIMA needs evenly spaced data).
    - Reject a target series with no numeric values, any missing values, too few
      values, or a constant target.

    Returns ``(series, freq)`` ready for model selection and fitting.
    """

    # 1. Locate the file inside the session workspace; error if missing.
    name = Path(file_name).name
    path = session_root(session_id) / name
    if not path.exists() or not path.is_file():
        raise SarimaToolError(
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
        raise SarimaToolError(
            "unsupported_file_extension",
            f"file_name must end with .csv or .xlsx (got '{suffix or 'no extension'}').",
            "data_validation",
        )

    # 3. Confirm the required date and target columns are present.
    missing_cols = [c for c in (date_column, target_column) if c not in df.columns]
    if missing_cols:
        raise SarimaToolError(
            "missing_required_columns",
            f"Required columns missing in '{name}': {missing_cols}. Available: {list(df.columns)[:20]}",
            "data_validation",
        )

    df = df[[date_column, target_column]].copy()

    # 4. Parse dates, drop unparseable rows, sort, and drop duplicate dates (keep first).
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = (
        df.dropna(subset=[date_column])
        .sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="first")
        .reset_index(drop=True)
    )

    # 5. Infer a regular cadence; ARIMA needs evenly spaced data.
    date_idx = pd.DatetimeIndex(df[date_column].values)
    freq = pd.infer_freq(date_idx)
    if freq is None:
        raise SarimaToolError(
            "frequency_not_inferred",
            "Could not infer a regular frequency from the date column. Resample/clean first or ask the user.",
            "data_validation",
        )

    # 6. Build the numeric target series indexed by the regular DatetimeIndex.
    series = pd.Series(
        pd.to_numeric(df[target_column], errors="coerce").values,
        index=pd.DatetimeIndex(date_idx, freq=freq),
        name=target_column,
    )

    # 7. Sanity-check the target series.
    n_obs = int(len(series))
    missing_target = int(series.isna().sum())

    if missing_target == n_obs:
        raise SarimaToolError(
            "all_target_missing",
            "Target column has no usable numeric values.",
            "data_validation",
        )
    if missing_target > 0:
        raise SarimaToolError(
            "missing_target_values",
            f"Target column has {missing_target} missing values. Clean/impute the data before fitting ARIMA.",
            "data_validation",
        )
    if n_obs < MIN_OBS:
        raise SarimaToolError(
            "too_few_target_values",
            f"Need at least {MIN_OBS} target values; got {n_obs}.",
            "data_validation",
        )
    if series.nunique() <= 1:
        raise SarimaToolError(
            "constant_target",
            "Target column is constant; ARIMA cannot be fitted.",
            "data_validation",
        )

    return series, freq


# Brief: Resolve manual or auto-selected ARIMA/SARIMA orders into one model spec.
def select_or_prepare_model_order(
    *,
    series: pd.Series,
    use_auto_arima: bool,
    order: Optional[list[int]],
    seasonal_order: Optional[list[int]],
    seasonal_period: Optional[int],
) -> tuple[dict, list[dict]]:
    """
    Decide the final SARIMA spec ``(p,d,q) x (P,D,Q,m)``.

    Branching rules:

    * If ``seasonal_order`` is provided without ``seasonal_period`` → error
      (we will not guess m).
    * If ``use_auto_arima`` is true → call ``pmdarima.auto_arima`` (AICc).
      Any ``order`` / ``seasonal_order`` passed alongside is ignored, with a
      warning so the orchestrator knows.
    * If ``use_auto_arima`` is false and ``order`` is provided → use it.
      ``seasonal_order`` is optional and only applied when both it and
      ``seasonal_period`` are set.
    * If ``use_auto_arima`` is false and ``order`` is missing → fall back to
      ``auto_arima`` with a warning. (We never silently fail; we always pick
      something.)
    """
    warnings_out: list[dict] = []

    if seasonal_order is not None and seasonal_period is None:
        raise SarimaToolError(
            "seasonal_period_required",
            "seasonal_period must be provided when seasonal_order is set.",
            "select_or_prepare_model_order",
        )

    if use_auto_arima and (order is not None or seasonal_order is not None):
        warnings_out.append(
            {
                "code": "auto_arima_overrides_manual_order",
                "message": "use_auto_arima is true; provided order/seasonal_order were ignored.",
            }
        )

    if not use_auto_arima and order is None:
        warnings_out.append(
            {
                "code": "manual_order_missing_used_auto_arima",
                "message": "use_auto_arima is false but order was not provided; auto_arima was used.",
            }
        )
        use_auto_arima = True

    if use_auto_arima:
        # ``m=1`` and ``seasonal=False`` are pmdarima's "non-seasonal" signals.
        seasonal = seasonal_period is not None and int(seasonal_period) > 1
        m = int(seasonal_period) if seasonal else 1
        with traced_span("sarima_tool.auto_arima", input={"seasonal": seasonal, "m": m, "n_obs": int(len(series))}) as span:
            try:
                model = pm.auto_arima(
                    series,
                    seasonal=seasonal,
                    m=m,
                    information_criterion="aicc",
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True,
                )
            except Exception as exc:
                if span is not None:
                    span.update(output={"error": str(exc)})
                raise SarimaToolError(
                    "auto_arima_failed",
                    f"pmdarima.auto_arima failed: {exc}",
                    "select_or_prepare_model_order",
                ) from exc

            chosen_order = [int(x) for x in model.order]
            full_seasonal = list(model.seasonal_order)
            if seasonal and len(full_seasonal) == 4 and full_seasonal[3] > 1:
                chosen_seasonal_order: Optional[list[int]] = [int(x) for x in full_seasonal[:3]]
                chosen_m: Optional[int] = int(full_seasonal[3])
            else:
                chosen_seasonal_order = None
                chosen_m = None

            if span is not None:
                span.update(output={"order": chosen_order, "seasonal_order": chosen_seasonal_order, "seasonal_period": chosen_m})

        return (
            {
                "selection_method": "auto_arima",
                "order": chosen_order,
                "seasonal_order": chosen_seasonal_order,
                "seasonal_period": chosen_m,
                "information_criterion": "aicc",
            },
            warnings_out,
        )

    # Manual path: use the order(s) the orchestrator (or user) provided as-is.
    chosen_order = [int(x) for x in order]  # type: ignore[arg-type]
    if seasonal_order is not None and seasonal_period is not None and int(seasonal_period) > 1:
        chosen_seasonal_order = [int(x) for x in seasonal_order]
        chosen_m = int(seasonal_period)
    else:
        chosen_seasonal_order = None
        chosen_m = None

    return (
        {
            "selection_method": "manual",
            "order": chosen_order,
            "seasonal_order": chosen_seasonal_order,
            "seasonal_period": chosen_m,
            "information_criterion": None,
        },
        warnings_out,
    )


# Brief: Fit SARIMAX, summarise fit quality, and diagnose residuals in one stage.
def fit_model(series: pd.Series, spec: dict) -> tuple[Any, dict, dict]:
    """
    Fit SARIMAX with the chosen spec and return everything downstream needs.

    Combines three concerns so the pipeline can move from "we have a series"
    to "we have a fitted model with quality and residual evidence" in one step:

    1. Fit: instantiate SARIMAX with the chosen ``(p,d,q) x (P,D,Q,m)`` spec
       and fit by MLE. Stationarity / invertibility constraints are relaxed so
       user-supplied near-non-stationary orders still fit; diagnostics surface
       any practical issues.
    2. Fit quality: AIC, AICc (small-sample-corrected), BIC, log-likelihood,
       convergence flag, sample size, and parameter count, plus inline
       ``definitions`` so the orchestrator/LLM never has to guess what each
       metric means.
    3. Residual diagnostics: residual mean/std, Ljung-Box p-value (with a
       seasonal-aware lag), Jarque-Bera normality p-value, and plain-English
       ``warnings`` when assumptions look violated. Bad diagnostics never fail
       the pipeline — they are surfaced as warnings only.

    Returns ``(fit, fit_quality, residual_diagnostics)``.
    """
    order = tuple(spec["order"])
    seasonal_period = spec.get("seasonal_period")
    if spec["seasonal_order"] is not None and seasonal_period:
        seasonal_order = tuple(spec["seasonal_order"]) + (int(seasonal_period),)
    else:
        seasonal_order = (0, 0, 0, 0)

    # 1. Fit the SARIMAX model.
    with traced_span("sarima_tool.fit_model", input={"order": list(order), "seasonal_order": list(seasonal_order)}) as span:
        try:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            with _warns.catch_warnings():
                _warns.simplefilter("ignore")
                fit = model.fit(disp=False)
            if span is not None:
                span.update(output={"converged": bool(fit.mle_retvals.get("converged", True)), "aic": float(fit.aic)})
        except Exception as exc:
            if span is not None:
                span.update(output={"error": str(exc)})
            raise SarimaToolError(
                "fit_failed",
                f"SARIMAX fit failed for order={list(order)} seasonal_order={list(seasonal_order)}: {exc}",
                "fit_model",
            ) from exc

    # 2. Summarise fit quality metrics for the response and LLM prompts.
    nobs = int(getattr(fit, "nobs", 0)) or len(getattr(fit, "fittedvalues", []))
    k = int(getattr(fit, "df_model", 0)) + 1
    aic = float(fit.aic)
    bic = float(fit.bic)
    if nobs - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (nobs - k - 1)
    else:
        aicc = aic
    log_lik = float(getattr(fit, "llf", float("nan")))
    converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))

    fit_quality = {
        "aic": round(aic, 4),
        "aicc": round(aicc, 4),
        "bic": round(bic, 4),
        "log_likelihood": round(log_lik, 4),
        "converged": converged,
        "n_observations": nobs,
        "n_parameters": k,
        "definitions": {
            "aic": "Akaike Information Criterion. Estimates relative in-sample model quality; lower is better when comparing models on the same data.",
            "aicc": "AIC corrected for small sample sizes; lower is better. Preferred when n_observations is small relative to n_parameters.",
            "bic": "Bayesian Information Criterion. Penalises model complexity more strongly than AIC; lower is better.",
            "log_likelihood": "Log of the likelihood of the data given the fitted model; higher is better, but usually compared via AIC/BIC.",
            "converged": "Whether the optimizer successfully found model parameters. False suggests the fit may be unreliable.",
            "n_observations": "Number of observations used to fit the model.",
            "n_parameters": "Number of free parameters in the fitted model (including the variance term).",
        },
    }

    # 3. Diagnose residuals. Bad diagnostics become warnings, not errors.
    resid = pd.Series(fit.resid).dropna()
    n = int(len(resid))

    if n < 5:
        residual_diagnostics = {
            "status": "warn",
            "n_residuals": n,
            "warnings": ["Too few residuals for diagnostic tests."],
            "definitions": {
                "status": "'pass' when residuals look like white noise; 'warn' when at least one assumption looks violated.",
                "n_residuals": "Number of residuals available for diagnostic tests.",
            },
        }
        return fit, fit_quality, residual_diagnostics

    residual_mean = float(resid.mean())
    residual_std = float(resid.std(ddof=1)) if n > 1 else 0.0

    if seasonal_period and seasonal_period > 1:
        lb_lags = min(2 * int(seasonal_period), max(1, n // 5))
    else:
        lb_lags = min(10, max(1, n // 5))

    try:
        lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
        ljung_box_pvalue = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        ljung_box_pvalue = float("nan")

    try:
        _, jb_pvalue, _, _ = jarque_bera(resid.values)
        normality_pvalue = float(jb_pvalue)
    except Exception:
        normality_pvalue = float("nan")

    diagnostic_warnings: list[str] = []
    if not np.isnan(ljung_box_pvalue) and ljung_box_pvalue < 0.05:
        diagnostic_warnings.append(
            f"Residuals show statistically significant autocorrelation (Ljung-Box p={ljung_box_pvalue:.4f} at lag {lb_lags})."
        )
    if not np.isnan(normality_pvalue) and normality_pvalue < 0.05:
        diagnostic_warnings.append(
            f"Residuals deviate from normality (Jarque-Bera p={normality_pvalue:.4f}); prediction intervals may be inaccurate."
        )
    if residual_std > 0 and abs(residual_mean) > 2.0 * residual_std / np.sqrt(n):
        diagnostic_warnings.append(
            f"Residual mean {residual_mean:.4f} is more than 2 standard errors from zero — possible bias."
        )

    residual_diagnostics = {
        "status": "warn" if diagnostic_warnings else "pass",
        "n_residuals": n,
        "residual_mean": round(residual_mean, 6),
        "residual_std": round(residual_std, 6),
        "ljung_box_pvalue": None if np.isnan(ljung_box_pvalue) else round(ljung_box_pvalue, 6),
        "ljung_box_lags": int(lb_lags),
        "normality_pvalue": None if np.isnan(normality_pvalue) else round(normality_pvalue, 6),
        "warnings": diagnostic_warnings,
        "definitions": {
            "status": "'pass' when residuals look like white noise; 'warn' when at least one assumption looks violated.",
            "n_residuals": "Number of residuals available for diagnostic tests.",
            "residual_mean": "Average residual. Should be close to zero; large absolute values suggest forecast bias.",
            "residual_std": "Sample standard deviation of residuals; rough scale of unexplained variation.",
            "ljung_box_pvalue": "Ljung-Box test p-value for residual autocorrelation. Low values (<0.05) suggest the model has not captured all time dependence.",
            "ljung_box_lags": "Number of lags used by the Ljung-Box test (set heuristically from sample size or seasonal period).",
            "normality_pvalue": "Jarque-Bera test p-value for residual normality. Low values (<0.05) suggest non-normal residuals; prediction intervals may be less reliable.",
            "warnings": "Plain-English notes about which residual assumptions look violated, if any.",
        },
    }

    return fit, fit_quality, residual_diagnostics


# Brief: Produce future point forecasts and 95% prediction intervals.
def generate_forecast(
    fit: Any,
    *,
    horizon: int,
    freq: str,
    last_date: pd.Timestamp,
) -> list[dict]:
    """Out-of-sample point forecasts plus 95% prediction intervals as JSON-friendly rows."""
    try:
        pred = fit.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        conf = pred.conf_int(alpha=DEFAULT_ALPHA)
    except Exception as exc:
        raise SarimaToolError(
            "forecast_failed",
            f"Failed to generate forecast: {exc}",
            "generate_forecast",
        ) from exc

    # Rebuild future dates explicitly so the response uses ISO strings the UI handles
    # uniformly (statsmodels can return either a DatetimeIndex or an integer index
    # depending on how the input series was constructed).
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    rows: list[dict] = []
    for ts, m, lo, hi in zip(future_dates, mean.values, conf.iloc[:, 0].values, conf.iloc[:, 1].values):
        rows.append(
            {
                "calendar_date": pd.Timestamp(ts).date().isoformat(),
                "forecast": float(m),
                "lower_95": float(lo),
                "upper_95": float(hi),
            }
        )
    return rows


# Brief: Save the forecast table in the session workspace.
def save_forecast(rows: list[dict], *, session_id: str, output_file: str) -> str:
    """Write the forecast table as CSV/XLSX at the session root only — never PNG/SVG (use ``coding_tool`` for charts)."""
    name = Path(output_file).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_TABLE_EXTS:
        raise SarimaToolError(
            "invalid_forecast_output_file",
            f"forecast_output_file must end with .csv or .xlsx (got '{suffix or 'no extension'}').",
            "save_forecast",
        )

    ensure_session_dirs(session_id)
    out_path = session_root(session_id) / name
    df = pd.DataFrame(rows)
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)

    sid = session_dir_for_paths(session_id)
    return f"agent_filesystem/{sid}/{name}"


# Brief: Ask the LLM to turn deterministic diagnostics into user-facing summaries.
def run_llm_interpretations(
    *,
    fit_quality: dict,
    residual_diagnostics: dict,
    forecast_rows: list[dict],
    model_spec: dict,
) -> dict:
    """
    Three small structured LLM calls that translate deterministic JSON into prose.

    If a single call fails (rate limit, schema parse error, …) the tool still
    returns a usable response — the failed sub-call gets a deterministic
    placeholder so the orchestrator can surface the rest.
    """
    interpretations: dict[str, dict] = {}

    residual_payload = json.dumps({"model": model_spec, "residual_diagnostics": residual_diagnostics}, indent=2, default=str)
    residual_messages = [SystemMessage(content=RESIDUAL_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=residual_payload)]
    interpretations["residual_analysis"] = _structured_llm_call(
        "sarima_tool.llm.residual_analysis",
        ResidualAnalysisOutput,
        residual_messages,
        {"status": residual_diagnostics.get("status", "warn"), "summary": "LLM interpretation failed; review the raw residual diagnostics.", "caveat": ""},
    )

    fit_payload = json.dumps({"model": model_spec, "fit_quality": fit_quality, "residual_diagnostics": residual_diagnostics}, indent=2, default=str)
    fit_messages = [SystemMessage(content=FIT_QUALITY_SYSTEM_PROMPT), HumanMessage(content=fit_payload)]
    interpretations["fit_quality"] = _structured_llm_call(
        "sarima_tool.llm.fit_quality",
        FitQualityOutput,
        fit_messages,
        {"status": "warn", "summary": "LLM interpretation failed; review the raw fit-quality metrics.", "caveat": ""},
    )

    preview = forecast_rows[:24]
    forecast_payload = json.dumps({"model": model_spec, "horizon": len(forecast_rows), "forecast_preview": preview}, indent=2, default=str)
    forecast_messages = [SystemMessage(content=FORECAST_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=forecast_payload)]
    interpretations["forecast_summary"] = _structured_llm_call(
        "sarima_tool.llm.forecast_summary",
        ForecastSummaryOutput,
        forecast_messages,
        {"status": "warn", "summary": "LLM interpretation failed; review the raw forecast values.", "uncertainty": "", "business_readout": ""},
    )

    guidance_payload = json.dumps(
        {"model": model_spec, "fit_quality": fit_quality, "residual_diagnostics": residual_diagnostics, "forecast_summary": {"horizon": len(forecast_rows), "forecast_preview": preview}},
        indent=2,
        default=str,
    )
    guidance_messages = [SystemMessage(content=MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT), HumanMessage(content=guidance_payload)]
    interpretations["model_improvement_guidance"] = _structured_llm_call(
        "sarima_tool.llm.model_improvement_guidance",
        ModelImprovementGuidanceOutput,
        guidance_messages,
        {"summary": "LLM guidance failed; review the raw fit quality and residual diagnostics to decide next steps.", "possible_next_steps": [], "caution": ""},
    )

    return interpretations


# Brief: Assemble the final JSON payload returned to the orchestrator.
def build_response(
    *,
    freq: str,
    model_spec: dict,
    fit_quality: dict,
    residual_diagnostics: dict,
    forecast_rows: list[dict],
    forecast_path: str,
    llm_interpretation: dict,
    warnings_out: list[dict],
) -> str:
    """Pack everything into the JSON string the orchestrator will see as the tool result."""
    has_warnings = bool(warnings_out)
    response = {
        "status": "success_with_warnings" if has_warnings else "success",
        "frequency": freq,
        "model": model_spec,
        "fit_quality": fit_quality,
        "residual_diagnostics": residual_diagnostics,
        "forecast_output_file": forecast_path,
        "horizon": len(forecast_rows),
        # Keep the wire small: orchestrator gets a 12-row preview, full table is on disk.
        "forecast_preview": forecast_rows[:12],
        "llm_interpretation": llm_interpretation,
        "warnings": warnings_out,
    }
    return json.dumps(response, default=str)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_sarima_pipeline(
    *,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    use_auto_arima: bool,
    order: Optional[list[int]],
    seasonal_order: Optional[list[int]],
    seasonal_period: Optional[int],
    forecast_output_file: str,
    runtime: ToolRuntime,
) -> str:
    """End-to-end SARIMA pipeline. Returns a JSON string (success or error)."""
    session_id = session_id_from_config(runtime.config)

    with traced_span("sarima_tool", metadata={"session_id": session_id}) as tool_span:
        try:
            series, freq = data_validation(file_name=file_name, date_column=date_column, target_column=target_column, session_id=session_id)
            model_spec, warnings_out = select_or_prepare_model_order(series=series, use_auto_arima=use_auto_arima, order=order, seasonal_order=seasonal_order, seasonal_period=seasonal_period)
            fit, fit_quality, residual_diagnostics = fit_model(series, model_spec)
            forecast_rows = generate_forecast(fit, horizon=int(horizon), freq=freq, last_date=series.index[-1])
            forecast_path = save_forecast(forecast_rows, session_id=session_id, output_file=forecast_output_file)
            llm_interpretation = run_llm_interpretations(fit_quality=fit_quality, residual_diagnostics=residual_diagnostics, forecast_rows=forecast_rows, model_spec=model_spec)
            for w in residual_diagnostics.get("warnings", []) or []:
                warnings_out.append({"code": "residual_assumption", "message": str(w)})
            response = build_response(
                freq=freq,
                model_spec=model_spec,
                fit_quality=fit_quality,
                residual_diagnostics=residual_diagnostics,
                forecast_rows=forecast_rows,
                forecast_path=forecast_path,
                llm_interpretation=llm_interpretation,
                warnings_out=warnings_out,
            )
            if tool_span is not None:
                tool_span.update(metadata={"selection_method": model_spec["selection_method"], "order": str(model_spec["order"]), "warnings": str(len(warnings_out))})
            return response
        except SarimaToolError as exc:
            if tool_span is not None:
                tool_span.update(metadata={"status": "error", "stage": exc.stage, "code": exc.code})
            return json.dumps({"status": "error", "stage": exc.stage, "error": {"code": exc.code, "message": exc.message}}, default=str)
        except Exception as exc:
            if tool_span is not None:
                tool_span.update(metadata={"status": "error", "stage": "unknown", "code": "internal_error"})
            return json.dumps({"status": "error", "stage": "unknown", "error": {"code": "internal_error", "message": str(exc)[:500]}}, default=str)


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------


@tool(args_schema=SarimaToolInput)
# Brief: Public LangChain tool wrapper that exposes SARIMA forecasting to the agent.
def sarima_tool(
    runtime: ToolRuntime,
    file_name: str,
    date_column: str,
    target_column: str,
    horizon: int,
    seasonal_period: Optional[int],
    forecast_output_file: str,
    use_auto_arima: bool = False,
    order: Optional[list[int]] = None,
    seasonal_order: Optional[list[int]] = None,
) -> str:
    """Fit ARIMA/SARIMA on a tabular session time series and return forecast + diagnostics + LLM interpretation as JSON.

    ## When to use
    The user wants a time-series forecast from a clean tabular file with one
    date column and one numeric target column. Prefer this over ``coding_tool``
    for ARIMA/SARIMA: it runs a deterministic statistical pipeline (validate →
    time index → order selection → fit → diagnostics → forecast → interpret)
    and returns structured JSON the orchestrator can present directly.

    ## When NOT to use
    - User wants Prophet, ETS, neural nets, or any non-ARIMA model — write code via ``coding_tool``.
    - Data still needs cleaning, joining, or resampling before fitting — handle that in ``coding_tool`` first, then call this tool on the cleaned file.
    - Multiple targets or panel structure — out of scope for this tool.
    - User wants a chart of the forecast — this tool does not generate images;
      follow up with ``coding_tool`` (it is the only image-producing tool).

    ## Required inputs
    - ``file_name`` — bare CSV/XLSX filename in the session workspace.
    - ``date_column`` — column with dates.
    - ``target_column`` — numeric column to forecast.
    - ``horizon`` — positive integer number of future periods.
    - ``seasonal_period`` — required field. Pass ``null`` for non-seasonal modelling, or an integer m (12 monthly/yearly cycle, 4 quarterly, 7 daily/weekly cycle, 24 hourly/daily cycle, etc.). If unsure, call ``ask_user`` first.
    - ``forecast_output_file`` — bare filename ending in ``.csv`` or ``.xlsx``.

    ## Optional inputs
    - ``use_auto_arima`` (default ``false``) — if ``true``, the tool runs ``pmdarima.auto_arima`` (AICc) and ignores any provided order.
    - ``order`` — ``[p, d, q]`` if you already know the model order.
    - ``seasonal_order`` — ``[P, D, Q]`` (no m); only valid when ``seasonal_period`` is also set.

    ## Behaviour
    - If ``use_auto_arima`` is ``false`` and ``order`` is missing, the tool falls back to ``auto_arima`` and adds a warning.
    - If ``seasonal_order`` is provided without ``seasonal_period``, the tool returns an error.
    - Hard data errors (missing file, missing columns, non-regular spacing, missing target values, constant target) return a clean error JSON so you can ask the user or run cleaning first.
    - Bad residual diagnostics are returned as warnings, not errors — the forecast is still produced.

    ## Output
    A JSON string with: ``status``, ``frequency``, ``model``, ``fit_quality`` (AIC/AICc/BIC/log-likelihood/converged/n_observations/n_parameters plus inline ``definitions`` for each metric), ``residual_diagnostics`` (status, Ljung-Box and Jarque-Bera p-values, residual mean/std, plain-English ``warnings``, plus inline ``definitions``), ``forecast_output_file``, ``horizon``, ``forecast_preview`` (first 12 rows), ``llm_interpretation`` (``residual_analysis``, ``fit_quality``, ``forecast_summary``, ``model_improvement_guidance`` with concrete ``possible_next_steps``), and ``warnings``. Inputs (``file_name``, ``date_column``, ``target_column``) are not echoed in the result — they appear on the tool call. When showing results to the user, present the LLM interpretation (residual analysis + fit quality + forecast summary + improvement guidance); do not narrate file paths.
    """
    return run_sarima_pipeline(
        file_name=file_name,
        date_column=date_column,
        target_column=target_column,
        horizon=horizon,
        use_auto_arima=use_auto_arima,
        order=order,
        seasonal_order=seasonal_order,
        seasonal_period=seasonal_period,
        forecast_output_file=forecast_output_file,
        runtime=runtime,
    )
