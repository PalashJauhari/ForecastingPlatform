"""
Holt-Winters / exponential smoothing model — inherits ``ForecastingUnivariateModel``.
"""

from __future__ import annotations

import warnings as warns_module
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from output_validation.forecasting_common import BaseForecastToolInput
from output_validation.holt_winters_tool import HoltWintersToolInput
from tools.forecasting.base import ForecastingToolError, ForecastingUnivariateModel

DEFAULT_ALPHA = 0.05
INITIALIZATION_METHOD = "estimated"
SIMULATION_REPETITIONS = 500


class HoltWintersModel(ForecastingUnivariateModel):
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

    def build_series(self, df: pd.DataFrame, freq: str) -> pd.Series:
        """Build a regular-frequency series from normalized ``ds``/``y``."""
        date_idx = pd.DatetimeIndex(df["ds"].values, freq=freq)
        return pd.Series(df["y"].values, index=date_idx, name="y")

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
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict]]:
        """Fit exponential smoothing / Holt-Winters and compute in-sample fit quality."""
        hw_params = HoltWintersToolInput.model_validate(params.model_dump())
        series = self.build_series(df, self.inferred_freq)
        self.validate_holt_winters_config(hw_params, int(len(series)))

        trend_sm = self.map_component(hw_params.trend)
        seasonal_sm = self.map_component(hw_params.seasonal)
        seasonal_periods = int(hw_params.seasonal_period) if hw_params.seasonal_period is not None else None

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
        except Exception as exc:
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

        return es_fit, model_spec, fit_quality, []

    def component_arrays(self, es_fit: Any) -> dict[str, Optional[np.ndarray]]:
        """Extract level/trend/seasonal state arrays from the statsmodels fit."""
        components: dict[str, Optional[np.ndarray]] = {"level": None, "trend": None, "seasonal": None}
        try:
            components["level"] = np.asarray(es_fit.level, dtype=float)
        except Exception:
            pass
        try:
            components["trend"] = np.asarray(es_fit.trend, dtype=float)
        except Exception:
            pass
        try:
            components["seasonal"] = np.asarray(es_fit.season, dtype=float)
        except Exception:
            pass
        return components

    def analyze_residuals(self, residual_table: pd.DataFrame) -> dict:
        """Ljung-Box and Jarque-Bera diagnostics on in-sample residuals."""
        seasonal_period = self.fitted_model_spec.get("seasonal_period")
        resid = pd.to_numeric(residual_table["residual"], errors="coerce").dropna()
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

    def build_fitted_table(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        """In-sample fitted values."""
        series = self.build_series(df, self.inferred_freq)
        fitted_vals = model.fittedvalues.reindex(series.index)
        return pd.DataFrame(
            {
                "calendar_date": [pd.Timestamp(ts).date().isoformat() for ts in series.index],
                "fitted": fitted_vals.values,
            }
        )

    def build_forecast_table(self, model: Any, horizon: int, freq: str) -> pd.DataFrame:
        """Out-of-sample forecasts with 95% prediction intervals."""
        try:
            mean = model.forecast(steps=horizon)
            mean_arr = np.asarray(mean, dtype=float).reshape(-1)
            sim = model.simulate(
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
                "build_forecast_table",
            ) from exc

        last_index = model.fittedvalues.index[-1]
        future_dates = pd.date_range(start=last_index, periods=horizon + 1, freq=freq)[1:]
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

    def build_fitted_decomposition(self, df: pd.DataFrame, model: Any) -> pd.DataFrame | None:
        """In-sample level/trend/seasonal breakdown where available."""
        series = self.build_series(df, self.inferred_freq)
        fitted_vals = model.fittedvalues.reindex(series.index)
        fitted_part = pd.DataFrame(
            {
                "calendar_date": [pd.Timestamp(ts).date().isoformat() for ts in series.index],
                "fitted": fitted_vals.values,
            }
        )
        components = self.component_arrays(model)
        has_components = False
        if components["level"] is not None and len(components["level"]) == len(fitted_part):
            fitted_part["level"] = components["level"]
            has_components = True
        if components["trend"] is not None and len(components["trend"]) == len(fitted_part):
            fitted_part["trend"] = components["trend"]
            has_components = True
        if components["seasonal"] is not None and len(components["seasonal"]) == len(fitted_part):
            fitted_part["seasonal"] = components["seasonal"]
            has_components = True
        return fitted_part if has_components else None

    def build_forecast_decomposition(self, model: Any, horizon: int, freq: str) -> pd.DataFrame | None:
        """Forecast-period decomposition with null component columns."""
        try:
            mean = model.forecast(steps=horizon)
            mean_arr = np.asarray(mean, dtype=float).reshape(-1)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast decomposition: {exc}",
                "build_forecast_decomposition",
            ) from exc

        last_index = model.fittedvalues.index[-1]
        future_dates = pd.date_range(start=last_index, periods=horizon + 1, freq=freq)[1:]
        forecast_part = pd.DataFrame(
            {
                "calendar_date": [pd.Timestamp(ts).date().isoformat() for ts in future_dates],
                "forecast": mean_arr,
            }
        )
        for col in ("level", "trend", "seasonal"):
            forecast_part[col] = np.nan
        return forecast_part
