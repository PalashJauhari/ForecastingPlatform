"""
SARIMA/ARIMA forecasting model — inherits the shared ``ForecastingUnivariateModel`` pipeline.
"""

from __future__ import annotations

import warnings as warns_module
from typing import Any, Optional

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

from output_validation.forecasting_common import BaseForecastToolInput
from output_validation.sarima_tool import SarimaToolInput
from tools.forecasting.base import ForecastingToolError, ForecastingUnivariateModel

DEFAULT_ALPHA = 0.05


class SarimaModel(ForecastingUnivariateModel):
    """SARIMA backend for ``sarima_tool``."""

    @property
    def model_type(self) -> str:
        return "sarima"

    @property
    def allow_missing_target(self) -> bool:
        return False

    def build_series(self, df: pd.DataFrame, freq: str) -> pd.Series:
        """Build a regular DatetimeIndex series from normalized ``ds``/``y``."""
        date_idx = pd.DatetimeIndex(df["ds"].values, freq=freq)
        return pd.Series(df["y"].values, index=date_idx, name="y")

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

    def fit(
        self,
        df: pd.DataFrame,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict]]:
        """Select order, fit SARIMAX, and summarise fit-quality metrics."""
        sarima_params = SarimaToolInput.model_validate(params.model_dump())
        series = self.build_series(df, self.inferred_freq)
        spec, warnings_out = self.select_or_prepare_model_order(series, sarima_params)

        order = tuple(spec["order"])
        seasonal_period = spec.get("seasonal_period")
        if spec["seasonal_order"] is not None and seasonal_period:
            seasonal_order = tuple(spec["seasonal_order"]) + (int(seasonal_period),)
        else:
            seasonal_order = (0, 0, 0, 0)

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
        except Exception as exc:
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
        return sm_fit, model_spec, fit_quality, warnings_out

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

    def build_fitted_table(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        """In-sample fitted values from the SARIMAX fit."""
        series = self.build_series(df, self.inferred_freq)
        fitted_vals = model.fittedvalues.reindex(series.index)
        return pd.DataFrame(
            {
                "calendar_date": [pd.Timestamp(ts).date().isoformat() for ts in series.index],
                "fitted": fitted_vals.values,
            }
        )

    def build_forecast_table(self, model: Any, horizon: int, freq: str) -> pd.DataFrame:
        """Out-of-sample point forecasts with 95% prediction intervals."""
        try:
            pred = model.get_forecast(steps=horizon)
            mean = pred.predicted_mean
            conf = pred.conf_int(alpha=DEFAULT_ALPHA)
        except Exception as exc:
            raise ForecastingToolError(
                "forecast_failed",
                f"Failed to generate forecast: {exc}",
                "build_forecast_table",
            ) from exc

        last_index = model.fittedvalues.index[-1]
        future_dates = pd.date_range(start=last_index, periods=horizon + 1, freq=freq)[1:]
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
