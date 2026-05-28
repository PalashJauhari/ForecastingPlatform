"""
Prophet forecasting model — inherits the shared ``ForecastingModel`` pipeline.

Fits Facebook Prophet with caller-controlled trend/seasonality, produces forecast,
fitted, and decomposition tables, MAD residual diagnostics, and five LLM
interpretation blocks.
"""

from __future__ import annotations

import json
import logging
import warnings as warns_module
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from prophet import Prophet

from observability.langfuse_handler import traced_span
from output_validation.forecasting_common import BaseForecastToolInput
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
from tools.forecasting.base import ForecastingModel, ForecastingToolError

# Fixed Prophet parameters kept off the tool surface.
CHANGEPOINT_RANGE = 0.8
SEASONALITY_PRIOR_SCALE = 10.0
MONTHLY_FOURIER_ORDER = 5
DAILY_SEASONALITY = False

COMPONENT_COLS = ["trend", "weekly", "monthly", "yearly", "additive_terms", "multiplicative_terms"]

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetModel(ForecastingModel):
    """Prophet backend for ``prophet_tool``."""

    @property
    def model_type(self) -> str:
        return "prophet"

    @property
    def allow_missing_target(self) -> bool:
        return True

    def changepoints_for_response(self, fit_extras: dict) -> dict | None:
        return fit_extras.get("changepoints")

    def fit(
        self,
        df: pd.DataFrame,
        freq: str,
        file_name: str,
        date_column: str,
        target_column: str,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict], dict]:
        """Configure Prophet, fit on ``ds``/``y``, and compute in-sample predictions."""
        prophet_params = ProphetToolInput.model_validate(params.model_dump())

        with traced_span(
            "prophet_tool.fit_model",
            trace_context=self.trace_context,
            input={
                "changepoint_prior_scale": prophet_params.changepoint_prior_scale,
                "seasonality_mode": prophet_params.seasonality_mode,
                "weekly_seasonality": prophet_params.weekly_seasonality,
                "monthly_seasonality": prophet_params.monthly_seasonality,
                "yearly_seasonality": prophet_params.yearly_seasonality,
                "n_obs": int(len(df)),
            },
        ) as span:
            try:
                model = Prophet(
                    changepoint_prior_scale=prophet_params.changepoint_prior_scale,
                    changepoint_range=CHANGEPOINT_RANGE,
                    seasonality_prior_scale=SEASONALITY_PRIOR_SCALE,
                    seasonality_mode=prophet_params.seasonality_mode,
                    weekly_seasonality=prophet_params.weekly_seasonality,
                    yearly_seasonality=prophet_params.yearly_seasonality,
                    daily_seasonality=DAILY_SEASONALITY,
                )
                if prophet_params.monthly_seasonality:
                    model.add_seasonality(name="monthly", period=30.5, fourier_order=MONTHLY_FOURIER_ORDER)

                with warns_module.catch_warnings():
                    warns_module.simplefilter("ignore")
                    model.fit(df)

                in_sample = model.predict(df[["ds"]])
                if span is not None:
                    span.update(output={"n_obs": int(len(df)), "n_changepoints": int(len(model.changepoints))})
            except Exception as exc:
                if span is not None:
                    span.update(output={"error": str(exc)})
                raise ForecastingToolError(
                    "fit_failed",
                    f"Prophet fit failed: {exc}",
                    "fit",
                ) from exc

        # Fit-quality metrics on rows where actual y is present.
        actuals = df["y"].to_numpy()
        fitted_vals = in_sample["yhat"].to_numpy()
        mask = ~np.isnan(actuals) & ~np.isnan(fitted_vals)
        actuals_v = actuals[mask]
        fitted_v = fitted_vals[mask]
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
                "mae": "Mean absolute error between actual and fitted values; same units as the target; lower is better.",
                "rmse": "Root mean squared error between actual and fitted values; penalises large errors more than MAE; lower is better.",
                "smape": "Symmetric mean absolute percentage error in percent; scale-free comparison; lower is better.",
                "n_observations": "Number of in-sample observations used to compute the metrics.",
            },
        }

        changepoints = self.extract_changepoints(model)
        model_spec = {
            "model_type": "prophet",
            "changepoint_prior_scale": float(prophet_params.changepoint_prior_scale),
            "changepoint_range": CHANGEPOINT_RANGE,
            "seasonality_prior_scale": SEASONALITY_PRIOR_SCALE,
            "seasonality_mode": prophet_params.seasonality_mode,
            "weekly_seasonality": bool(prophet_params.weekly_seasonality),
            "monthly_seasonality": bool(prophet_params.monthly_seasonality),
            "yearly_seasonality": bool(prophet_params.yearly_seasonality),
            "monthly_fourier_order": MONTHLY_FOURIER_ORDER if prophet_params.monthly_seasonality else None,
            "daily_seasonality": DAILY_SEASONALITY,
        }

        return model, model_spec, fit_quality, [], {
            "in_sample": in_sample,
            "changepoints": changepoints,
        }

    def extract_changepoints(self, model: Any) -> dict:
        """Summarise Prophet changepoint dates and top deltas."""
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
                    order_idx = np.argsort(-np.abs(deltas))[:5]
                    for idx in order_idx:
                        largest_delta_changepoints.append(
                            {
                                "date": changepoint_dates[int(idx)],
                                "delta": float(round(float(deltas[int(idx)]), 6)),
                            }
                        )
        except Exception:
            pass
        return {
            "changepoint_range": CHANGEPOINT_RANGE,
            "changepoint_dates": changepoint_dates,
            "largest_delta_changepoints": largest_delta_changepoints,
        }

    def analyze_residuals(
        self,
        fitted_table: pd.DataFrame,
        model_spec: dict,
        fit_extras: dict,
    ) -> dict:
        """MAD-based residual outlier diagnostics on the fitted table."""
        actuals = pd.to_numeric(fitted_table["actual"], errors="coerce").to_numpy()
        fitted_vals = pd.to_numeric(fitted_table["fitted"], errors="coerce").to_numpy()
        dates = pd.to_datetime(fitted_table["calendar_date"], errors="coerce")
        mask = ~np.isnan(actuals) & ~np.isnan(fitted_vals)
        actuals_v = actuals[mask]
        fitted_v = fitted_vals[mask]
        dates_v = dates.to_numpy()[mask]
        n = int(len(actuals_v))

        if n == 0:
            residual_median = residual_mad = robust_sigma = lower_bound = upper_bound = float("nan")
            outlier_points: list[dict] = []
        else:
            residuals = actuals_v - fitted_v
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

        return {
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
                "status": "'pass' when no residual points fall outside the MAD bounds; 'warn' when outliers are found.",
                "n_residuals": "Number of residuals available for the MAD outlier check.",
                "residual_median": "Median residual; robust center of model errors.",
                "residual_mad": "Median absolute deviation of residuals from the residual median.",
                "robust_sigma": "MAD converted to an approximate standard-deviation scale using 1.4826 * MAD.",
                "lower_bound": "Lower residual outlier bound: residual_median - 3 * robust_sigma.",
                "upper_bound": "Upper residual outlier bound: residual_median + 3 * robust_sigma.",
                "outlier_count": "Number of fitted residual points outside the MAD bounds.",
                "outlier_fraction": "Share of residual points outside the MAD bounds.",
                "outlier_points": "Dated fitted observations whose residuals fall outside the MAD bounds.",
                "warnings": "Plain-English notes about MAD residual outliers, if any.",
            },
        }

    def generate_forecast(
        self,
        df: pd.DataFrame,
        freq: str,
        model: Any,
        fit_extras: dict,
        params: BaseForecastToolInput,
    ) -> pd.DataFrame:
        """Future-only Prophet forecast at the inferred frequency."""
        try:
            future = model.make_future_dataframe(
                periods=int(params.horizon),
                freq=freq,
                include_history=False,
            )
            forecast_future = model.predict(future)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast: {exc}",
                "generate_forecast",
            ) from exc

        in_sample = fit_extras["in_sample"]
        fitted_components = [c for c in COMPONENT_COLS if c in in_sample.columns]
        forecast_components = [c for c in COMPONENT_COLS if c in forecast_future.columns]
        component_cols = forecast_components or fitted_components

        forecast_table = forecast_future[["ds", "yhat"] + component_cols].copy()
        forecast_table = forecast_table.rename(columns={"ds": "calendar_date", "yhat": "forecast"})
        forecast_table["calendar_date"] = pd.to_datetime(forecast_table["calendar_date"]).dt.date.astype(str)
        fit_extras["forecast_components"] = component_cols
        fit_extras["fitted_components"] = fitted_components
        fit_extras["forecast_future"] = forecast_future
        return forecast_table

    def build_fitted_table(
        self,
        df: pd.DataFrame,
        target_column: str,
        model: Any,
        fit_extras: dict,
    ) -> pd.DataFrame:
        """In-sample actual, fitted, residual, and component columns."""
        in_sample = fit_extras["in_sample"]
        fitted_components = [c for c in COMPONENT_COLS if c in in_sample.columns]
        fit_extras["fitted_components"] = fitted_components

        fitted_table = pd.merge(
            df[["ds", "y"]],
            in_sample[["ds", "yhat"] + fitted_components],
            on="ds",
            how="left",
        )
        fitted_table["residual"] = fitted_table["y"] - fitted_table["yhat"]
        fitted_table = fitted_table.rename(columns={"ds": "calendar_date", "y": "actual", "yhat": "fitted"})
        fitted_table = fitted_table[["calendar_date", "actual", "fitted", "residual"] + fitted_components]
        fitted_table["calendar_date"] = pd.to_datetime(fitted_table["calendar_date"]).dt.date.astype(str)
        return fitted_table

    def build_decomposition(
        self,
        df: pd.DataFrame,
        model: Any,
        fit_extras: dict,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Combined fitted + forecast decomposition table."""
        in_sample = fit_extras["in_sample"]
        forecast_future = fit_extras.get("forecast_future")
        if forecast_future is None:
            return None

        fitted_components = fit_extras.get("fitted_components") or []
        forecast_components = fit_extras.get("forecast_components") or []
        all_components = [c for c in COMPONENT_COLS if c in fitted_components or c in forecast_components]

        decomposition_fit = pd.merge(
            df[["ds", "y"]],
            in_sample[["ds", "yhat"] + fitted_components],
            on="ds",
            how="left",
        )
        decomposition_fit = decomposition_fit.rename(columns={"y": "actual"})
        decomposition_fit["period_type"] = "fitted"

        decomposition_fcst = forecast_future[["ds", "yhat"] + forecast_components].copy()
        decomposition_fcst["actual"] = np.nan
        decomposition_fcst["period_type"] = "forecast"

        decomposition_table = pd.concat([decomposition_fit, decomposition_fcst], ignore_index=True)
        decomposition_table = decomposition_table.reindex(columns=["ds", "period_type", "actual", "yhat"] + all_components)
        decomposition_table = decomposition_table.rename(columns={"ds": "calendar_date"})
        decomposition_table["calendar_date"] = pd.to_datetime(decomposition_table["calendar_date"]).dt.date.astype(str)
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
        """Five structured LLM calls for Prophet results."""
        changepoints = fit_extras.get("changepoints") or {}
        forecast_rows = forecast_table.to_dict(orient="records")
        decomposition_rows = decomposition_table.to_dict(orient="records") if decomposition_table is not None else []
        interpretations: dict[str, dict] = {}

        residual_payload = json.dumps(
            {"model": model_spec, "residual_diagnostics": residual_diagnostics},
            indent=2,
            default=str,
        )
        interpretations["residual_analysis"] = self.structured_llm_call(
            "prophet_tool.llm.residual_analysis",
            ResidualAnalysisOutput,
            [SystemMessage(content=RESIDUAL_ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=residual_payload)],
            {
                "status": residual_diagnostics.get("status", "warn"),
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
            "prophet_tool.llm.fit_quality",
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
            "prophet_tool.llm.forecast_summary",
            ForecastSummaryOutput,
            [SystemMessage(content=FORECAST_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=forecast_payload)],
            {"status": "warn", "summary": "LLM interpretation failed; review the raw forecast values.", "business_readout": ""},
        )

        decomposition_preview = decomposition_rows[:24]
        component_payload = json.dumps(
            {"model": model_spec, "changepoints": changepoints, "decomposition_preview": decomposition_preview},
            indent=2,
            default=str,
        )
        interpretations["component_analysis"] = self.structured_llm_call(
            "prophet_tool.llm.component_analysis",
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
                "changepoints": changepoints,
                "forecast_summary": {"horizon": len(forecast_rows), "forecast_preview": forecast_preview},
            },
            indent=2,
            default=str,
        )
        interpretations["model_improvement_guidance"] = self.structured_llm_call(
            "prophet_tool.llm.model_improvement_guidance",
            ModelImprovementGuidanceOutput,
            [SystemMessage(content=MODEL_IMPROVEMENT_GUIDANCE_SYSTEM_PROMPT), HumanMessage(content=guidance_payload)],
            {
                "summary": "LLM guidance failed; review diagnostics to decide next steps.",
                "possible_next_steps": [],
                "caution": "",
            },
        )
        return interpretations
