"""
Base forecasting pipeline for session tabular time-series tools.

``ForecastingUnivariateModel`` owns validate → fit → fitted → residuals → forecast → save tables → lean JSON.
Subclasses override model-specific fit, forecast, and optional decomposition hooks.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime

from observability.langfuse_handler import trace_context_from_runnable_config, traced_span
from output_validation.forecasting_common import BaseForecastToolInput
from session_paths import ensure_session_dirs, session_id_from_config, session_root

MIN_OBS = 10
PREVIEW_HEAD_ROWS = 5
ALLOWED_TABLE_EXTS = {".csv", ".xlsx"}


class ForecastingToolError(Exception):
    """Domain error raised by pipeline stages so ``run()`` can return clean JSON."""

    def __init__(self, code: str, message: str, stage: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


class ForecastingUnivariateModel(ABC):
    """Template-method base for univariate forecasting tools."""

    session_id: str = ""
    trace_context: Any = None
    inferred_freq: str = ""
    fitted_model_spec: dict = {}

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short model id surfaced in JSON."""

    @property
    @abstractmethod
    def allow_missing_target(self) -> bool:
        """If false, any missing target row fails validation."""

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict]]:
        """Return (model, model_spec, fit_quality, fit_warnings)."""

    @abstractmethod
    def analyze_residuals(self, residual_table: pd.DataFrame) -> dict:
        """Return residual diagnostics (may include definitions; stripped before JSON)."""

    @abstractmethod
    def build_fitted_table(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        """Return in-sample table: calendar_date, fitted."""

    @abstractmethod
    def build_forecast_table(self, model: Any, horizon: int, freq: str) -> pd.DataFrame:
        """Return forecast table with calendar_date and forecast (+ intervals when available)."""

    def build_fitted_decomposition(self, df: pd.DataFrame, model: Any) -> pd.DataFrame | None:
        """Optional in-sample decomposition; default none."""
        return None

    def build_forecast_decomposition(self, model: Any, horizon: int, freq: str) -> pd.DataFrame | None:
        """Optional forecast-period decomposition; default none."""
        return None

    def build_residual_table(self, fitted_table: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
        """Merge fitted values with actuals on calendar_date; compute residual."""
        merged = fitted_table.merge(actuals, on="calendar_date", how="inner")
        merged["residual"] = merged["actual"] - merged["fitted"]
        return merged[["calendar_date", "actual", "fitted", "residual"]]

    # ---------------------------------------------------------------------------
    # Artifact names and persistence
    # ---------------------------------------------------------------------------

    def artifact_basename(self, experiment_name: str, stem: str, suffix: str) -> str:
        """Build output basename ``{experiment_name}_{stem}.{suffix}``."""
        return f"{experiment_name}_{stem}.{suffix}"

    def save_table(
        self,
        table: pd.DataFrame,
        session_id: str,
        basename: str,
        stage: str,
    ) -> str:
        """Write table at session root; return basename only."""
        name = Path(basename).name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_TABLE_EXTS:
            raise ForecastingToolError(
                "invalid_output_file",
                f"{stage} file must end with .csv or .xlsx (got '{ext or 'no extension'}').",
                stage,
            )
        ensure_session_dirs(session_id)
        out_path = session_root(session_id) / name
        if ext == ".csv":
            table.to_csv(out_path, index=False)
        else:
            table.to_excel(out_path, index=False)
        return name

    # ---------------------------------------------------------------------------
    # JSON helpers
    # ---------------------------------------------------------------------------

    def preview_head(self, table: pd.DataFrame) -> list[dict]:
        """First N rows for LLM-facing JSON."""
        return table.head(PREVIEW_HEAD_ROWS).to_dict(orient="records")

    def lean_residual_output(self, residual_diagnostics: dict) -> dict:
        """Drop definitions and other clutter from residual diagnostics for the tool JSON."""
        skip = {"definitions"}
        out: dict[str, Any] = {}
        for key, val in residual_diagnostics.items():
            if key in skip:
                continue
            out[key] = val
        return out

    def lean_metrics(self, fit_quality: dict) -> dict:
        """Fit metrics only (no definitions block)."""
        skip = {"definitions"}
        return {k: v for k, v in fit_quality.items() if k not in skip}

    def hyperparameters_from_spec(self, model_spec: dict) -> dict:
        """Model hyperparameters for JSON (exclude model_type)."""
        return {k: v for k, v in model_spec.items() if k != "model_type"}

    def warning_messages(self, warnings_out: list) -> list[str]:
        """Flatten warning entries to plain strings for the top-level JSON."""
        messages: list[str] = []
        for w in warnings_out:
            if isinstance(w, dict):
                msg = w.get("message")
                if msg:
                    messages.append(str(msg))
            elif w:
                messages.append(str(w))
        return messages

    # ---------------------------------------------------------------------------
    # Data validation
    # ---------------------------------------------------------------------------

    def validate_data(
        self,
        session_id: str,
        file_name: str,
        date_column: str,
        target_column: str,
    ) -> tuple[pd.DataFrame, str]:
        """Load session file, normalize to ds/y, infer frequency; return df and freq."""
        name = Path(file_name).name
        path = session_root(session_id) / name
        if not path.exists() or not path.is_file():
            raise ForecastingToolError(
                "file_not_found",
                f"File '{name}' was not found in the session workspace.",
                "data_validation",
            )

        suffix = Path(name).suffix.lower()
        if suffix == ".csv":
            raw_df = pd.read_csv(path)
        elif suffix == ".xlsx":
            raw_df = pd.read_excel(path)
        else:
            raise ForecastingToolError(
                "unsupported_file_extension",
                f"file_name must end with .csv or .xlsx (got '{suffix or 'no extension'}').",
                "data_validation",
            )

        missing_cols = [c for c in (date_column, target_column) if c not in raw_df.columns]
        if missing_cols:
            raise ForecastingToolError(
                "missing_required_columns",
                f"Required columns missing in '{name}': {missing_cols}.",
                "data_validation",
            )

        df = raw_df[[date_column, target_column]].copy()
        df = df.rename(columns={date_column: "ds", target_column: "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = (
            df.dropna(subset=["ds"])
            .sort_values("ds")
            .drop_duplicates(subset=["ds"], keep="first")
            .reset_index(drop=True)
        )

        freq = pd.infer_freq(pd.DatetimeIndex(df["ds"].values))
        if freq is None:
            raise ForecastingToolError(
                "frequency_not_inferred",
                "Could not infer a regular frequency from the date column.",
                "data_validation",
            )

        if self.allow_missing_target:
            usable = df["y"].dropna()
            if usable.empty or len(usable) < MIN_OBS or usable.nunique() <= 1:
                raise ForecastingToolError(
                    "too_few_target_values",
                    "Need at least 10 non-missing non-constant target values.",
                    "data_validation",
                )
        else:
            n_obs = int(len(df))
            missing_target = int(df["y"].isna().sum())
            if missing_target > 0:
                raise ForecastingToolError(
                    "missing_target_values",
                    f"Target has {missing_target} missing values.",
                    "data_validation",
                )
            if n_obs < MIN_OBS or df["y"].nunique() <= 1:
                raise ForecastingToolError(
                    "too_few_target_values",
                    f"Need at least {MIN_OBS} target values.",
                    "data_validation",
                )

        return df, freq

    def error_dict(self, exc: ForecastingToolError) -> dict:
        return {
            "status": "error",
            "stage": exc.stage,
            "error": {"code": exc.code, "message": exc.message},
        }

    def error_dict_unknown(self, exc: Exception) -> dict:
        return {
            "status": "error",
            "stage": "unknown",
            "error": {"code": "internal_error", "message": str(exc)[:500]},
        }

    def run(self, runtime: ToolRuntime, params: BaseForecastToolInput) -> str:
        """End-to-end pipeline; returns lean JSON string for the orchestrator."""
        self.session_id = session_id_from_config(runtime.config)
        self.trace_context = trace_context_from_runnable_config(runtime.config)
        experiment_name = params.experiment_name
        span_name = f"{self.model_type}_tool"
        span_input = {
            "experiment_name": experiment_name,
            "file_name": params.file_name,
            "horizon": params.horizon,
        }

        with traced_span(
            span_name,
            trace_context=self.trace_context,
            input=span_input,
            metadata={"session_id": self.session_id},
        ) as tool_span:
            warnings_out: list = []
            try:
                df, freq = self.validate_data(
                    self.session_id,
                    params.file_name,
                    params.date_column,
                    params.target_column,
                )
                self.inferred_freq = freq
                n_obs = int(len(df))

                model, model_spec, fit_quality, fit_warnings = self.fit(df, params)
                self.fitted_model_spec = model_spec
                warnings_out.extend(fit_warnings)

                fitted_table = self.build_fitted_table(df, model)
                actuals = df[["ds", "y"]].copy()
                actuals = actuals.rename(columns={"ds": "calendar_date", "y": "actual"})
                actuals["calendar_date"] = actuals["calendar_date"].dt.date.astype(str)
                residual_table = self.build_residual_table(fitted_table, actuals)

                residual_raw = self.analyze_residuals(residual_table)
                residual_lean = self.lean_residual_output(residual_raw)
                for w in residual_raw.get("warnings", []) or []:
                    warnings_out.append({"code": "residual_assumption", "message": str(w)})

                horizon = int(params.horizon)
                forecast_table = self.build_forecast_table(model, horizon, freq)
                fitted_decomp = self.build_fitted_decomposition(df, model)
                forecast_decomp = self.build_forecast_decomposition(model, horizon, freq)

                fitted_csv = self.artifact_basename(experiment_name, "fitted", "csv")
                forecast_csv = self.artifact_basename(experiment_name, "forecast", "csv")

                self.save_table(residual_table, self.session_id, fitted_csv, "save_fitted")
                self.save_table(forecast_table, self.session_id, forecast_csv, "save_forecast")

                pipeline: dict[str, Any] = {
                    "data_validation": {
                        "description": "Loaded and checked regular time series.",
                        "status": "success",
                        "output": {"n_obs": n_obs, "frequency": freq},
                    },
                    "model_fit": {
                        "description": "Fitted model and in-sample error metrics.",
                        "status": "success",
                        "output": {
                            "hyperparameters": self.hyperparameters_from_spec(model_spec),
                            "metrics": self.lean_metrics(fit_quality),
                        },
                    },
                    "fitted_values": {
                        "description": "In-sample actual, fitted, and residual.",
                        "status": "success",
                        "output": {
                            "file_name": fitted_csv,
                            "preview_head": self.preview_head(residual_table),
                        },
                    },
                    "forecast_values": {
                        "description": "Out-of-sample forecast.",
                        "status": "success",
                        "output": {
                            "file_name": forecast_csv,
                            "preview_head": self.preview_head(forecast_table),
                        },
                    },
                    "residual_analysis": {
                        "description": "Residual diagnostic checks.",
                        "status": residual_lean.get("status", "pass"),
                        "output": residual_lean,
                    },
                }

                if fitted_decomp is not None:
                    fitted_decomp_csv = self.artifact_basename(experiment_name, "fitted_decomposition", "csv")
                    self.save_table(fitted_decomp, self.session_id, fitted_decomp_csv, "save_fitted_decomposition")
                    pipeline["fitted_decomposition"] = {
                        "description": "In-sample level/trend/seasonal breakdown.",
                        "status": "success",
                        "output": {
                            "file_name": fitted_decomp_csv,
                            "preview_head": self.preview_head(fitted_decomp),
                        },
                    }

                if forecast_decomp is not None:
                    forecast_decomp_csv = self.artifact_basename(experiment_name, "forecast_decomposition", "csv")
                    self.save_table(
                        forecast_decomp, self.session_id, forecast_decomp_csv, "save_forecast_decomposition"
                    )
                    pipeline["forecast_decomposition"] = {
                        "description": "Forecast-period decomposition (components may be null).",
                        "status": "success",
                        "output": {
                            "file_name": forecast_decomp_csv,
                            "preview_head": self.preview_head(forecast_decomp),
                        },
                    }

                warn_msgs = self.warning_messages(warnings_out)
                response: dict[str, Any] = {
                    "status": "success_with_warnings" if warn_msgs else "success",
                    "model_type": self.model_type,
                    "experiment_name": experiment_name,
                    "frequency": freq,
                    "warnings": warn_msgs,
                    "pipeline": pipeline,
                }
                if tool_span is not None:
                    tool_span.update(output=response)
                return json.dumps(response, default=str)

            except ForecastingToolError as exc:
                err = self.error_dict(exc)
                if tool_span is not None:
                    tool_span.update(output=err)
                return json.dumps(err, default=str)
            except Exception as exc:
                err = self.error_dict_unknown(exc)
                if tool_span is not None:
                    tool_span.update(output=err)
                return json.dumps(err, default=str)
