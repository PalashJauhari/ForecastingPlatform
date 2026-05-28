"""
Holt-Winters / exponential smoothing model — inherits ``ForecastingModel``.

Fits statsmodels ``ExponentialSmoothing`` with caller-controlled trend/seasonal
components, produces forecast with 95% intervals, fitted/residual table,
level/trend/seasonal decomposition, Ljung-Box/Jarque-Bera diagnostics, and
five LLM interpretation blocks.
"""

from __future__ import annotations

import json
import warnings as warns_module
from typing import Any, Optional

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from observability.langfuse_handler import traced_span
from output_validation.forecasting_common import BaseForecastToolInput
from output_validation.holt_winters_tool import (
    ComponentAnalysisOutput,
    FitQualityOutput,
    ForecastSummaryOutput,
    HoltWintersToolInput,
    ModelImprovementGuidanceOutput,
    ResidualAnalysisOutput,
)
from prompts.holt_winters_interpretation_prompts import (
    COMPONENT_ANALYSIS_SYSTEM_PROMPT,
    FIT_QUALITY_SYSTEM_PROMPT,
    FORECAST_SUMMARY_SYSTEM_PROMPT,
    MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT,
    RESIDUAL_ANALYSIS_SYSTEM_PROMPT,
)
from tools.forecasting.base import ForecastingModel, ForecastingToolError

DEFAULT_ALPHA = 0.05
INITIALIZATION_METHOD = "estimated"
SIMULATION_REPETITIONS = 500


class HoltWintersModel(ForecastingModel):
    """Holt-Winters backend for ``holt_winters_tool``."""

    @property
    def model_type(self) -> str:
        return "holt_winters"

    @property
    def allow_missing_target(self) -> bool:
        return False

    def map_component(self, value: str) -> Optional[str]:
        """Map tool arg ``none``/``add``/``mul`` to statsmodels trend/seasonal arg."""
        if value == "none":
            return None
        return value

    def build_series(self, df: pd.DataFrame, freq: str, target_column: str) -> pd.Series:
        """Build a regular-frequency series from normalized ``ds``/``y``."""
        date_idx = pd.DatetimeIndex(df["ds"].values, freq=freq)
        return pd.Series(df["y"].values, index=date_idx, name=target_column)

    def validate_holt_winters_config(self, hw_params: HoltWintersToolInput, n_obs: int) -> None:
        """Pre-fit checks for trend/seasonal/period combinations."""
        seasonal_sm = self.map_component(hw_params.seasonal)
        if seasonal_sm is not None:
            if hw_params.seasonal_period is None:
                raise ForecastingToolError(
                    "seasonal_period_required",
                    "seasonal_period must be provided when seasonal is 'add' or 'mul'.",
                    "fit",
                )
            m = int(hw_params.seasonal_period)
            if n_obs < 2 * m:
                raise ForecastingToolError(
                    "too_few_observations_for_seasonal",
                    f"Need at least {2 * m} observations for seasonal_period={m}; got {n_obs}.",
                    "fit",
                )

    def fit(
        self,
        df: pd.DataFrame,
        freq: str,
        file_name: str,
        date_column: str,
        target_column: str,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict], dict]:
        """Fit exponential smoothing / Holt-Winters and compute in-sample fit quality."""
        hw_params = HoltWintersToolInput.model_validate(params.model_dump())
        series = self.build_series(df, freq, target_column)
        self.validate_holt_winters_config(hw_params, int(len(series)))

        trend_sm = self.map_component(hw_params.trend)
        seasonal_sm = self.map_component(hw_params.seasonal)
        seasonal_periods = int(hw_params.seasonal_period) if hw_params.seasonal_period is not None else None

        with traced_span(
            "holt_winters_tool.fit_model",
            trace_context=self.trace_context,
            input={
                "trend": hw_params.trend,
                "seasonal": hw_params.seasonal,
                "seasonal_period": seasonal_periods,
                "damped_trend": hw_params.damped_trend,
                "n_obs": int(len(series)),
            },
        ) as span:
            try:
                es_model = ExponentialSmoothing(
                    series,
                    trend=trend_sm,
                    damped_trend=bool(hw_params.damped_trend),
                    seasonal=seasonal_sm,
                    seasonal_periods=seasonal_periods,
                    initialization_method=INITIALIZATION_METHOD,
                )
                with warns_module.catch_warnings():
                    warns_module.simplefilter("ignore")
                    es_fit = es_model.fit(optimized=True)
                if span is not None:
                    span.update(output={"n_obs": int(len(series)), "aic": float(es_fit.aic)})
            except Exception as exc:
                if span is not None:
                    span.update(output={"error": str(exc)})
                raise ForecastingToolError(
                    "fit_failed",
                    f"Holt-Winters fit failed: {exc}",
                    "fit",
                ) from exc

        fitted_vals = es_fit.fittedvalues.reindex(series.index)
        actuals = series.values
        fitted_arr = fitted_vals.values
        mask = ~np.isnan(actuals) & ~np.isnan(fitted_arr)
        actuals_v = actuals[mask]
        fitted_v = fitted_arr[mask]
        n = int(len(actuals_v))

        if n == 0:
            mae = rmse = smape = float("nan")
        else:
            residuals = actuals_v - fitted_v
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals**2)))
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
                "mae": "Mean absolute error between actual and fitted values; lower is better.",
                "rmse": "Root mean squared error; penalises large errors more than MAE; lower is better.",
                "smape": "Symmetric mean absolute percentage error in percent; lower is better.",
                "n_observations": "Number of in-sample observations used to compute the metrics.",
            },
        }

        model_spec = {
            "model_type": "holt_winters",
            "trend": hw_params.trend,
            "seasonal": hw_params.seasonal,
            "seasonal_period": hw_params.seasonal_period,
            "damped_trend": bool(hw_params.damped_trend),
            "initialization_method": INITIALIZATION_METHOD,
        }

        return es_fit, model_spec, fit_quality, [], {
            "series": series,
            "trend_enabled": trend_sm is not None,
            "seasonal_enabled": seasonal_sm is not None,
        }

    def component_arrays(
        self,
        es_fit: Any,
        fit_extras: dict,
    ) -> dict[str, Optional[np.ndarray]]:
        """Extract level/trend/seasonal state arrays from the statsmodels fit."""
        components: dict[str, Optional[np.ndarray]] = {"level": None, "trend": None, "seasonal": None}
        try:
            components["level"] = np.asarray(es_fit.level, dtype=float)
        except Exception:
            pass
        if fit_extras.get("trend_enabled"):
            try:
                components["trend"] = np.asarray(es_fit.trend, dtype=float)
            except Exception:
                pass
        if fit_extras.get("seasonal_enabled"):
            try:
                components["seasonal"] = np.asarray(es_fit.season, dtype=float)
            except Exception:
                pass
        return components

    def analyze_residuals(
        self,
        fitted_table: pd.DataFrame,
        model_spec: dict,
        fit_extras: dict,
    ) -> dict:
        """Ljung-Box and Jarque-Bera diagnostics on in-sample residuals."""
        seasonal_period = model_spec.get("seasonal_period")
        resid = pd.to_numeric(fitted_table["residual"], errors="coerce").dropna()
        n = int(len(resid))

        if n < 5:
            return {
                "status": "warn",
                "n_residuals": n,
                "warnings": ["Too few residuals for diagnostic tests."],
                "definitions": {
                    "status": "'pass' when residuals look like white noise; 'warn' when assumptions are violated.",
                    "n_residuals": "Number of residuals available for diagnostic tests.",
                },
            }

        residual_mean = float(resid.mean())
        residual_std = float(resid.std(ddof=1)) if n > 1 else 0.0

        if seasonal_period and int(seasonal_period) > 1:
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

        return {
            "status": "warn" if diagnostic_warnings else "pass",
            "n_residuals": n,
            "residual_mean": round(residual_mean, 6),
            "residual_std": round(residual_std, 6),
            "ljung_box_pvalue": None if np.isnan(ljung_box_pvalue) else round(ljung_box_pvalue, 6),
            "ljung_box_lags": int(lb_lags),
            "normality_pvalue": None if np.isnan(normality_pvalue) else round(normality_pvalue, 6),
            "warnings": diagnostic_warnings,
            "definitions": {
                "status": "'pass' when residuals look like white noise; 'warn' when assumptions are violated.",
                "n_residuals": "Number of residuals available for diagnostic tests.",
                "residual_mean": "Average residual; should be close to zero.",
                "residual_std": "Sample standard deviation of residuals.",
                "ljung_box_pvalue": "Ljung-Box test p-value for residual autocorrelation.",
                "ljung_box_lags": "Number of lags used by the Ljung-Box test.",
                "normality_pvalue": "Jarque-Bera test p-value for residual normality.",
                "warnings": "Plain-English notes about violated residual assumptions, if any.",
            },
        }

    def build_fitted_table(
        self,
        df: pd.DataFrame,
        target_column: str,
        model: Any,
        fit_extras: dict,
    ) -> pd.DataFrame:
        """In-sample actual, fitted, and residual."""
        es_fit = model
        series: pd.Series = fit_extras["series"]
        fitted_vals = es_fit.fittedvalues.reindex(series.index)
        fitted_table = pd.DataFrame(
            {
                "calendar_date": [pd.Timestamp(ts).date().isoformat() for ts in series.index],
                "actual": series.values,
                "fitted": fitted_vals.values,
            }
        )
        fitted_table["residual"] = fitted_table["actual"] - fitted_table["fitted"]
        return fitted_table

    def generate_forecast(
        self,
        df: pd.DataFrame,
        freq: str,
        model: Any,
        fit_extras: dict,
        params: BaseForecastToolInput,
    ) -> pd.DataFrame:
        """Out-of-sample forecasts with 95% prediction intervals."""
        es_fit = model
        series: pd.Series = fit_extras["series"]
        horizon = int(params.horizon)

        try:
            mean = es_fit.forecast(steps=horizon)
            mean_arr = np.asarray(mean, dtype=float).reshape(-1)
            sim = es_fit.simulate(
                nsimulations=horizon,
                anchor="end",
                repetitions=SIMULATION_REPETITIONS,
                random_state=42,
                random_errors="bootstrap",
            )
            sim_arr = np.asarray(sim, dtype=float)
            if sim_arr.ndim == 1:
                sim_arr = sim_arr.reshape(-1, 1)
            lower = np.quantile(sim_arr, DEFAULT_ALPHA / 2.0, axis=1)
            upper = np.quantile(sim_arr, 1.0 - DEFAULT_ALPHA / 2.0, axis=1)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast: {exc}",
                "generate_forecast",
            ) from exc

        future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=freq)[1:]
        rows: list[dict] = []
        for ts, m, lo, hi in zip(future_dates, mean_arr, lower, upper):
            rows.append(
                {
                    "calendar_date": pd.Timestamp(ts).date().isoformat(),
                    "forecast": float(m),
                    "lower_95": float(lo),
                    "upper_95": float(hi),
                }
            )
        return pd.DataFrame(rows)

    def build_decomposition(
        self,
        df: pd.DataFrame,
        model: Any,
        fit_extras: dict,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Combined fitted + forecast decomposition with level/trend/seasonal where available."""
        es_fit = model
        components = self.component_arrays(es_fit, fit_extras)

        fitted_part = fitted_table.copy()
        fitted_part["period_type"] = "fitted"
        fitted_part = fitted_part.rename(columns={"fitted": "forecast"})

        if components["level"] is not None and len(components["level"]) == len(fitted_part):
            fitted_part["level"] = components["level"]
        if components["trend"] is not None and len(components["trend"]) == len(fitted_part):
            fitted_part["trend"] = components["trend"]
        if components["seasonal"] is not None and len(components["seasonal"]) == len(fitted_part):
            fitted_part["seasonal"] = components["seasonal"]

        forecast_part = forecast_table.copy()
        forecast_part["period_type"] = "forecast"
        forecast_part["actual"] = np.nan
        forecast_part["residual"] = np.nan
        for col in ("level", "trend", "seasonal"):
            if col in fitted_part.columns:
                forecast_part[col] = np.nan

        base_cols = ["calendar_date", "period_type", "actual", "forecast", "residual"]
        extra_cols = [c for c in ("level", "trend", "seasonal") if c in fitted_part.columns]
        decomposition_table = pd.concat([fitted_part, forecast_part], ignore_index=True)
        decomposition_table = decomposition_table.reindex(columns=base_cols + extra_cols)
        return decomposition_table

    def run_llm_interpretations(
        self,
        model_spec: dict,
        fit_quality: dict,
        fit_extras: dict,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
        residual_diagnostics: dict,
        params: BaseForecastToolInput,
        decomposition_table: pd.DataFrame | None,
    ) -> dict:
        """Five structured LLM calls for Holt-Winters results."""
        forecast_rows = forecast_table.to_dict(orient="records")
        decomposition_rows = decomposition_table.to_dict(orient="records") if decomposition_table is not None else []
        interpretations: dict[str, dict] = {}

        residual_payload = json.dumps(
            {"model": model_spec, "residual_diagnostics": residual_diagnostics},
            indent=2,
            default=str,
        )
        interpretations["residual_analysis"] = self.structured_llm_call(
            "holt_winters_tool.llm.residual_analysis",
            ResidualAnalysisOutput,
            [SystemMessage(content=RESIDUAL_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=residual_payload)],
            {
                "status": "warn" if residual_diagnostics.get("status") == "warn" else "ok",
                "summary": "LLM interpretation failed; review the raw residual diagnostics.",
                "caveat": "",
            },
        )

        fit_payload = json.dumps(
            {"model": model_spec, "fit_quality": fit_quality, "residual_diagnostics": residual_diagnostics},
            indent=2,
            default=str,
        )
        interpretations["fit_quality"] = self.structured_llm_call(
            "holt_winters_tool.llm.fit_quality",
            FitQualityOutput,
            [SystemMessage(content=FIT_QUALITY_SYSTEM_PROMPT), HumanMessage(content=fit_payload)],
            {"status": "warn", "summary": "LLM interpretation failed; review the raw fit-quality metrics.", "caveat": ""},
        )

        forecast_preview = forecast_rows[:24]
        forecast_payload = json.dumps(
            {"model": model_spec, "horizon": len(forecast_rows), "forecast_preview": forecast_preview},
            indent=2,
            default=str,
        )
        interpretations["forecast_summary"] = self.structured_llm_call(
            "holt_winters_tool.llm.forecast_summary",
            ForecastSummaryOutput,
            [SystemMessage(content=FORECAST_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=forecast_payload)],
            {
                "status": "warn",
                "summary": "LLM interpretation failed; review the raw forecast values.",
                "uncertainty": "",
                "business_readout": "",
            },
        )

        decomposition_preview = decomposition_rows[:24]
        component_payload = json.dumps(
            {"model": model_spec, "decomposition_preview": decomposition_preview},
            indent=2,
            default=str,
        )
        interpretations["component_analysis"] = self.structured_llm_call(
            "holt_winters_tool.llm.component_analysis",
            ComponentAnalysisOutput,
            [SystemMessage(content=COMPONENT_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=component_payload)],
            {
                "summary": "LLM interpretation failed; review the raw decomposition table.",
                "component_signals": [],
                "changepoint_summary": "",
                "caveat": "",
            },
        )

        guidance_payload = json.dumps(
            {
                "model": model_spec,
                "fit_quality": fit_quality,
                "residual_diagnostics": residual_diagnostics,
                "forecast_summary": {"horizon": len(forecast_rows), "forecast_preview": forecast_preview},
            },
            indent=2,
            default=str,
        )
        interpretations["model_improvement_guidance"] = self.structured_llm_call(
            "holt_winters_tool.llm.model_improvement_guidance",
            ModelImprovementGuidanceOutput,
            [SystemMessage(content=MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT), HumanMessage(content=guidance_payload)],
            {
                "summary": "LLM guidance failed; review diagnostics to decide next steps.",
                "possible_next_steps": [],
                "caution": "",
            },
        )
        return interpretations
