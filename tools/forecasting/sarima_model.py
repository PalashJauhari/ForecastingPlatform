"""
SARIMA/ARIMA forecasting model — inherits the shared ``ForecastingModel`` pipeline.

Fits SARIMAX (manual or auto-ARIMA order search), produces forecast with 95%
intervals, fitted/residual table, Ljung-Box/Jarque-Bera diagnostics, and four
LLM interpretation blocks.
"""

from __future__ import annotations

import json
import warnings as warns_module
from typing import Any, Optional

import numpy as np
import pandas as pd
import pmdarima as pm
from langchain_core.messages import HumanMessage, SystemMessage
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

from observability.langfuse_handler import traced_span
from output_validation.forecasting_common import BaseForecastToolInput
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
from tools.forecasting.base import FitBundle, ForecastingModel, ForecastingToolError, ValidatedData

DEFAULT_ALPHA = 0.05


class SarimaModel(ForecastingModel):
    """SARIMA backend for ``sarima_tool``."""

    @property
    def model_type(self) -> str:
        return "sarima"

    @property
    def allow_missing_target(self) -> bool:
        return False

    def build_series(self, validated: ValidatedData) -> pd.Series:
        """Build a regular DatetimeIndex series from validated ``ds``/``y``."""
        df = validated.df
        date_idx = pd.DatetimeIndex(df["ds"].values, freq=validated.freq)
        return pd.Series(
            df["y"].values,
            index=date_idx,
            name=validated.target_column,
        )

    def select_or_prepare_model_order(
        self,
        series: pd.Series,
        sarima_params: SarimaToolInput,
    ) -> tuple[dict, list[dict]]:
        """Resolve manual or auto-selected ARIMA/SARIMA orders into one model spec."""
        warnings_out: list[dict] = []
        use_auto_arima = sarima_params.use_auto_arima
        order = sarima_params.order
        seasonal_order = sarima_params.seasonal_order
        seasonal_period = sarima_params.seasonal_period

        if seasonal_order is not None and seasonal_period is None:
            raise ForecastingToolError(
                "seasonal_period_required",
                "seasonal_period must be provided when seasonal_order is set.",
                "fit",
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
            seasonal = seasonal_period is not None and int(seasonal_period) > 1
            m = int(seasonal_period) if seasonal else 1
            with traced_span(
                "sarima_tool.auto_arima",
                trace_context=self.trace_context,
                input={"seasonal": seasonal, "m": m, "n_obs": int(len(series))},
            ) as span:
                try:
                    auto_model = pm.auto_arima(
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
                    raise ForecastingToolError(
                        "auto_arima_failed",
                        f"pmdarima.auto_arima failed: {exc}",
                        "fit",
                    ) from exc

                chosen_order = [int(x) for x in auto_model.order]
                full_seasonal = list(auto_model.seasonal_order)
                if seasonal and len(full_seasonal) == 4 and full_seasonal[3] > 1:
                    chosen_seasonal_order: Optional[list[int]] = [int(x) for x in full_seasonal[:3]]
                    chosen_m: Optional[int] = int(full_seasonal[3])
                else:
                    chosen_seasonal_order = None
                    chosen_m = None

                if span is not None:
                    span.update(
                        output={
                            "order": chosen_order,
                            "seasonal_order": chosen_seasonal_order,
                            "seasonal_period": chosen_m,
                        }
                    )

            spec = {
                "selection_method": "auto_arima",
                "order": chosen_order,
                "seasonal_order": chosen_seasonal_order,
                "seasonal_period": chosen_m,
                "information_criterion": "aicc",
            }
            return spec, warnings_out

        chosen_order = [int(x) for x in order]  # type: ignore[arg-type]
        if seasonal_order is not None and seasonal_period is not None and int(seasonal_period) > 1:
            chosen_seasonal_order = [int(x) for x in seasonal_order]
            chosen_m = int(seasonal_period)
        else:
            chosen_seasonal_order = None
            chosen_m = None

        spec = {
            "selection_method": "manual",
            "order": chosen_order,
            "seasonal_order": chosen_seasonal_order,
            "seasonal_period": chosen_m,
            "information_criterion": None,
        }
        return spec, warnings_out

    def fit(self, validated: ValidatedData, params: BaseForecastToolInput) -> FitBundle:
        """Select order, fit SARIMAX, and summarise fit-quality metrics."""
        sarima_params = SarimaToolInput.model_validate(params.model_dump())
        series = self.build_series(validated)
        spec, warnings_out = self.select_or_prepare_model_order(series, sarima_params)

        order = tuple(spec["order"])
        seasonal_period = spec.get("seasonal_period")
        if spec["seasonal_order"] is not None and seasonal_period:
            seasonal_order = tuple(spec["seasonal_order"]) + (int(seasonal_period),)
        else:
            seasonal_order = (0, 0, 0, 0)

        with traced_span(
            "sarima_tool.fit_model",
            trace_context=self.trace_context,
            input={"order": list(order), "seasonal_order": list(seasonal_order)},
        ) as span:
            try:
                sm_model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                with warns_module.catch_warnings():
                    warns_module.simplefilter("ignore")
                    sm_fit = sm_model.fit(disp=False)
                if span is not None:
                    span.update(
                        output={
                            "converged": bool(sm_fit.mle_retvals.get("converged", True)),
                            "aic": float(sm_fit.aic),
                        }
                    )
            except Exception as exc:
                if span is not None:
                    span.update(output={"error": str(exc)})
                raise ForecastingToolError(
                    "fit_failed",
                    f"SARIMAX fit failed for order={list(order)} seasonal_order={list(seasonal_order)}: {exc}",
                    "fit",
                ) from exc

        nobs = int(getattr(sm_fit, "nobs", 0)) or len(getattr(sm_fit, "fittedvalues", []))
        k = int(getattr(sm_fit, "df_model", 0)) + 1
        aic = float(sm_fit.aic)
        bic = float(sm_fit.bic)
        if nobs - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / (nobs - k - 1)
        else:
            aicc = aic
        log_lik = float(getattr(sm_fit, "llf", float("nan")))
        converged = bool(getattr(sm_fit, "mle_retvals", {}).get("converged", True))

        fit_quality = {
            "aic": round(aic, 4),
            "aicc": round(aicc, 4),
            "bic": round(bic, 4),
            "log_likelihood": round(log_lik, 4),
            "converged": converged,
            "n_observations": nobs,
            "n_parameters": k,
            "definitions": {
                "aic": "Akaike Information Criterion; lower is better when comparing models on the same data.",
                "aicc": "AIC corrected for small sample sizes; lower is better.",
                "bic": "Bayesian Information Criterion; penalises complexity more than AIC.",
                "log_likelihood": "Log-likelihood of the data given the fitted model.",
                "converged": "Whether the optimizer successfully found model parameters.",
                "n_observations": "Number of observations used to fit the model.",
                "n_parameters": "Number of free parameters in the fitted model.",
            },
        }

        model_spec = {"model_type": "sarima", **spec}

        return FitBundle(
            model=sm_fit,
            model_spec=model_spec,
            fit_quality=fit_quality,
            warnings=warnings_out,
            extras={"spec": spec, "series": series},
        )

    def analyze_residuals(self, fitted_table: pd.DataFrame, fit: FitBundle) -> dict:
        """Ljung-Box and Jarque-Bera diagnostics on in-sample residuals."""
        spec = fit.extras.get("spec") or {}
        seasonal_period = spec.get("seasonal_period")
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

    def build_fitted_table(self, validated: ValidatedData, fit: FitBundle) -> pd.DataFrame:
        """In-sample actual, fitted, and residual from the SARIMAX fit."""
        sm_fit = fit.model
        series: pd.Series = fit.extras["series"]
        fitted_vals = sm_fit.fittedvalues.reindex(series.index)
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
        validated: ValidatedData,
        fit: FitBundle,
        params: BaseForecastToolInput,
    ) -> pd.DataFrame:
        """Out-of-sample point forecasts with 95% prediction intervals."""
        sm_fit = fit.model
        series: pd.Series = fit.extras["series"]
        horizon = int(params.horizon)

        try:
            pred = sm_fit.get_forecast(steps=horizon)
            mean = pred.predicted_mean
            conf = pred.conf_int(alpha=DEFAULT_ALPHA)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast: {exc}",
                "generate_forecast",
            ) from exc

        future_dates = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=validated.freq)[1:]
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
        return pd.DataFrame(rows)

    def run_llm_interpretations(
        self,
        validated: ValidatedData,
        fit: FitBundle,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
        residual_diagnostics: dict,
        params: BaseForecastToolInput,
        decomposition_table: pd.DataFrame | None,
    ) -> dict:
        """Four structured LLM calls for SARIMA results."""
        model_spec = fit.model_spec
        forecast_rows = forecast_table.to_dict(orient="records")
        interpretations: dict[str, dict] = {}

        residual_payload = json.dumps(
            {"model": model_spec, "residual_diagnostics": residual_diagnostics},
            indent=2,
            default=str,
        )
        interpretations["residual_analysis"] = self.structured_llm_call(
            "sarima_tool.llm.residual_analysis",
            ResidualAnalysisOutput,
            [SystemMessage(content=RESIDUAL_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=residual_payload)],
            {
                "status": residual_diagnostics.get("status", "warn"),
                "summary": "LLM interpretation failed; review the raw residual diagnostics.",
                "caveat": "",
            },
        )

        fit_payload = json.dumps(
            {"model": model_spec, "fit_quality": fit.fit_quality, "residual_diagnostics": residual_diagnostics},
            indent=2,
            default=str,
        )
        interpretations["fit_quality"] = self.structured_llm_call(
            "sarima_tool.llm.fit_quality",
            FitQualityOutput,
            [SystemMessage(content=FIT_QUALITY_SYSTEM_PROMPT), HumanMessage(content=fit_payload)],
            {"status": "warn", "summary": "LLM interpretation failed; review the raw fit-quality metrics.", "caveat": ""},
        )

        preview = forecast_rows[:24]
        forecast_payload = json.dumps(
            {"model": model_spec, "horizon": len(forecast_rows), "forecast_preview": preview},
            indent=2,
            default=str,
        )
        interpretations["forecast_summary"] = self.structured_llm_call(
            "sarima_tool.llm.forecast_summary",
            ForecastSummaryOutput,
            [SystemMessage(content=FORECAST_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=forecast_payload)],
            {
                "status": "warn",
                "summary": "LLM interpretation failed; review the raw forecast values.",
                "uncertainty": "",
                "business_readout": "",
            },
        )

        guidance_payload = json.dumps(
            {
                "model": model_spec,
                "fit_quality": fit.fit_quality,
                "residual_diagnostics": residual_diagnostics,
                "forecast_summary": {"horizon": len(forecast_rows), "forecast_preview": preview},
            },
            indent=2,
            default=str,
        )
        interpretations["model_improvement_guidance"] = self.structured_llm_call(
            "sarima_tool.llm.model_improvement_guidance",
            ModelImprovementGuidanceOutput,
            [SystemMessage(content=MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT), HumanMessage(content=guidance_payload)],
            {
                "summary": "LLM guidance failed; review diagnostics to decide next steps.",
                "possible_next_steps": [],
                "caution": "",
            },
        )
        return interpretations
