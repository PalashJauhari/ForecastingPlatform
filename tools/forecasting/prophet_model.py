"""
Prophet forecasting model — inherits the shared ``ForecastingUnivariateModel`` pipeline.
"""

from __future__ import annotations

import logging
import warnings as warns_module
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from output_validation.forecasting_common import BaseForecastToolInput
from output_validation.prophet_tool import ProphetToolInput
from tools.forecasting.base import ForecastingToolError, ForecastingUnivariateModel

# Fixed Prophet parameters kept off the tool surface.
CHANGEPOINT_RANGE = 0.8
SEASONALITY_PRIOR_SCALE = 10.0
MONTHLY_FOURIER_ORDER = 5
DAILY_SEASONALITY = False

COMPONENT_COLS = ["trend", "weekly", "monthly", "yearly", "additive_terms", "multiplicative_terms"]

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetModel(ForecastingUnivariateModel):
    """Prophet backend for ``prophet_tool``."""

    @property
    def model_type(self) -> str:
        return "prophet"

    @property
    def allow_missing_target(self) -> bool:
        return True

    def fit(
        self,
        df: pd.DataFrame,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict]]:
        """Configure Prophet and fit on ``ds``/``y``."""
        prophet_params = ProphetToolInput.model_validate(params.model_dump())

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
        except Exception as exc:
            raise ForecastingToolError(
                "fit_failed",
                f"Prophet fit failed: {exc}",
                "fit",
            ) from exc

        in_sample = model.predict(df[["ds"]])
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
            **changepoints,
        }

        return model, model_spec, fit_quality, []

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
            "changepoint_dates": changepoint_dates,
            "largest_delta_changepoints": largest_delta_changepoints,
        }

    def analyze_residuals(self, residual_table: pd.DataFrame) -> dict:
        """MAD-based residual outlier diagnostics on the residual table."""
        actuals = pd.to_numeric(residual_table["actual"], errors="coerce").to_numpy()
        fitted_vals = pd.to_numeric(residual_table["fitted"], errors="coerce").to_numpy()
        residuals = pd.to_numeric(residual_table["residual"], errors="coerce").to_numpy()
        dates = pd.to_datetime(residual_table["calendar_date"], errors="coerce")
        mask = ~np.isnan(actuals) & ~np.isnan(fitted_vals) & ~np.isnan(residuals)
        actuals_v = actuals[mask]
        fitted_v = fitted_vals[mask]
        residuals_v = residuals[mask]
        dates_v = dates.to_numpy()[mask]
        n = int(len(actuals_v))

        if n == 0:
            residual_median = residual_mad = robust_sigma = lower_bound = upper_bound = float("nan")
            outlier_points: list[dict] = []
        else:
            residual_median = float(np.median(residuals_v))
            absolute_deviation = np.abs(residuals_v - residual_median)
            residual_mad = float(np.median(absolute_deviation))
            robust_sigma = float(1.4826 * residual_mad)
            lower_bound = float(residual_median - 3.0 * robust_sigma)
            upper_bound = float(residual_median + 3.0 * robust_sigma)
            outlier_mask = (residuals_v < lower_bound) | (residuals_v > upper_bound)
            outlier_points = [
                {
                    "calendar_date": pd.Timestamp(dates_v[idx]).date().isoformat(),
                    "actual": None if np.isnan(actuals_v[idx]) else round(float(actuals_v[idx]), 6),
                    "fitted": None if np.isnan(fitted_v[idx]) else round(float(fitted_v[idx]), 6),
                    "residual": round(float(residuals_v[idx]), 6),
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

    def build_fitted_table(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        """In-sample fitted values from Prophet predict."""
        in_sample = model.predict(df[["ds"]])
        fitted_table = in_sample[["ds", "yhat"]].copy()
        fitted_table = fitted_table.rename(columns={"ds": "calendar_date", "yhat": "fitted"})
        fitted_table["calendar_date"] = pd.to_datetime(fitted_table["calendar_date"]).dt.date.astype(str)
        return fitted_table[["calendar_date", "fitted"]]

    def build_forecast_table(self, model: Any, horizon: int, freq: str) -> pd.DataFrame:
        """Future-only Prophet forecast at the inferred frequency."""
        try:
            future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
            forecast_future = model.predict(future)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast: {exc}",
                "build_forecast_table",
            ) from exc

        forecast_table = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_table = forecast_table.rename(
            columns={"ds": "calendar_date", "yhat": "forecast", "yhat_lower": "lower_95", "yhat_upper": "upper_95"}
        )
        forecast_table["calendar_date"] = pd.to_datetime(forecast_table["calendar_date"]).dt.date.astype(str)
        return forecast_table

    def build_fitted_decomposition(self, df: pd.DataFrame, model: Any) -> pd.DataFrame | None:
        """In-sample Prophet component breakdown."""
        in_sample = model.predict(df[["ds"]])
        component_cols = [c for c in COMPONENT_COLS if c in in_sample.columns]
        if not component_cols:
            return None

        decomposition = in_sample[["ds", "yhat"] + component_cols].copy()
        decomposition = decomposition.rename(columns={"ds": "calendar_date", "yhat": "fitted"})
        decomposition["calendar_date"] = pd.to_datetime(decomposition["calendar_date"]).dt.date.astype(str)
        return decomposition

    def build_forecast_decomposition(self, model: Any, horizon: int, freq: str) -> pd.DataFrame | None:
        """Forecast-period Prophet component breakdown."""
        try:
            future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
            forecast_future = model.predict(future)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast decomposition: {exc}",
                "build_forecast_decomposition",
            ) from exc

        component_cols = [c for c in COMPONENT_COLS if c in forecast_future.columns]
        if not component_cols:
            return None

        decomposition = forecast_future[["ds", "yhat"] + component_cols].copy()
        decomposition = decomposition.rename(columns={"ds": "calendar_date", "yhat": "forecast"})
        decomposition["calendar_date"] = pd.to_datetime(decomposition["calendar_date"]).dt.date.astype(str)
        return decomposition
