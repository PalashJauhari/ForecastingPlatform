"""
Base forecasting pipeline for session tabular time-series tools.

``ForecastingModel`` owns validate → fit → fitted → residuals → forecast → save tables/plots → interpret → lean JSON.
Subclasses override model-specific fit, forecast, and optional decomposition hooks.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage

from middleware.llm_client import make_llm
from observability.langfuse_handler import trace_context_from_runnable_config, traced_span
from output_validation.forecasting_common import BaseForecastToolInput, InterpretationSummaryOutput
from prompts.forecasting_interpretation_prompt import FORECASTING_INTERPRETATION_SYSTEM_PROMPT
from session_paths import ensure_session_dirs, session_id_from_config, session_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

INTERPRETATION_MODEL = cfg["models"].get("orchestrator", "gpt-4o-mini")

MIN_OBS = 10
PREVIEW_HEAD_ROWS = 5
ALLOWED_TABLE_EXTS = {".csv", ".xlsx"}
ALLOWED_PLOT_EXTS = {".png"}


class ForecastingToolError(Exception):
    """Domain error raised by pipeline stages so ``run()`` can return clean JSON."""

    def __init__(self, code: str, message: str, stage: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


class ForecastingModel(ABC):
    """Template-method base for forecasting tools."""

    session_id: str = ""
    trace_context: Any = None

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
        freq: str,
        file_name: str,
        date_column: str,
        target_column: str,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict], dict]:
        """Return (model, model_spec, fit_quality, fit_warnings, fit_extras)."""

    @abstractmethod
    def analyze_residuals(
        self,
        fitted_table: pd.DataFrame,
        model_spec: dict,
        fit_extras: dict,
    ) -> dict:
        """Return residual diagnostics (may include definitions; stripped before JSON)."""

    @abstractmethod
    def generate_forecast(
        self,
        df: pd.DataFrame,
        freq: str,
        model: Any,
        fit_extras: dict,
        params: BaseForecastToolInput,
    ) -> pd.DataFrame:
        """Return forecast table with calendar_date and forecast (+ intervals when available)."""

    @abstractmethod
    def build_fitted_table(
        self,
        df: pd.DataFrame,
        target_column: str,
        model: Any,
        fit_extras: dict,
    ) -> pd.DataFrame:
        """Return in-sample table: calendar_date, actual, fitted, residual."""

    def build_decomposition(
        self,
        df: pd.DataFrame,
        model: Any,
        fit_extras: dict,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Optional combined decomposition with period_type column; default none."""
        return None

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

    def save_plot(self, fig: plt.Figure, session_id: str, basename: str, stage: str) -> str:
        """Save matplotlib figure as PNG; return basename only."""
        name = Path(basename).name
        if Path(name).suffix.lower() not in ALLOWED_PLOT_EXTS:
            raise ForecastingToolError(
                "invalid_output_file",
                f"{stage} plot must end with .png.",
                stage,
            )
        ensure_session_dirs(session_id)
        out_path = session_root(session_id) / name
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return name

    # ---------------------------------------------------------------------------
    # Default plots
    # ---------------------------------------------------------------------------

    def plot_fitted(self, fitted_table: pd.DataFrame) -> plt.Figure:
        """Actual vs fitted line chart."""
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = pd.to_datetime(fitted_table["calendar_date"])
        ax.plot(dates, fitted_table["actual"], label="actual", marker="o", markersize=3)
        ax.plot(dates, fitted_table["fitted"], label="fitted", marker="o", markersize=3)
        ax.legend()
        ax.set_title("In-sample fit")
        fig.autofmt_xdate()
        return fig

    def plot_forecast(self, forecast_table: pd.DataFrame) -> plt.Figure:
        """Forecast line with optional 95% interval band."""
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = pd.to_datetime(forecast_table["calendar_date"])
        ax.plot(dates, forecast_table["forecast"], label="forecast", marker="o", markersize=3)
        if "lower_95" in forecast_table.columns and "upper_95" in forecast_table.columns:
            ax.fill_between(
                dates,
                forecast_table["lower_95"],
                forecast_table["upper_95"],
                alpha=0.2,
                label="95% interval",
            )
        ax.legend()
        ax.set_title("Forecast")
        fig.autofmt_xdate()
        return fig

    def plot_residuals(self, fitted_table: pd.DataFrame) -> plt.Figure:
        """Residuals over time."""
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = pd.to_datetime(fitted_table["calendar_date"])
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.plot(dates, fitted_table["residual"], marker="o", markersize=3)
        ax.set_title("Residuals")
        fig.autofmt_xdate()
        return fig

    def plot_decomposition(self, decomposition_table: pd.DataFrame) -> plt.Figure:
        """Plot main level/trend/seasonal or forecast series from decomposition."""
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = pd.to_datetime(decomposition_table["calendar_date"])
        value_col = "forecast" if "forecast" in decomposition_table.columns else "fitted"
        if "trend" in decomposition_table.columns:
            ax.plot(dates, decomposition_table["trend"], label="trend")
        elif value_col in decomposition_table.columns:
            ax.plot(dates, decomposition_table[value_col], label=value_col)
        if "seasonal" in decomposition_table.columns:
            ax.plot(dates, decomposition_table["seasonal"], label="seasonal", alpha=0.8)
        ax.legend()
        ax.set_title("Decomposition")
        fig.autofmt_xdate()
        return fig

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

    def interpretation_llm_invoke(self, messages: list) -> dict:
        """Single structured interpretation call (no Langfuse generation span)."""
        llm = make_llm(model=INTERPRETATION_MODEL, temperature=0, output_schema=InterpretationSummaryOutput)
        try:
            raw = llm.invoke(messages)
            if isinstance(raw, dict) and "parsed" in raw:
                parsed = raw["parsed"]
            else:
                parsed = raw
            return parsed.model_dump()
        except Exception:
            return {"summary": "Interpretation unavailable; review pipeline metrics and previews."}

    def run_llm_interpretation_summary(
        self,
        experiment_name: str,
        freq: str,
        model_spec: dict,
        fit_quality: dict,
        residual_lean: dict,
        forecast_table: pd.DataFrame,
        warnings_out: list,
    ) -> str:
        """One LLM call returning a short user-facing summary."""
        payload = json.dumps(
            {
                "model_type": self.model_type,
                "experiment_name": experiment_name,
                "frequency": freq,
                "hyperparameters": self.hyperparameters_from_spec(model_spec),
                "metrics": self.lean_metrics(fit_quality),
                "residual_analysis": residual_lean,
                "forecast_preview": self.preview_head(forecast_table),
                "warnings": self.warning_messages(warnings_out),
            },
            indent=2,
            default=str,
        )
        result = self.interpretation_llm_invoke(
            [
                SystemMessage(content=FORECASTING_INTERPRETATION_SYSTEM_PROMPT),
                HumanMessage(content=payload),
            ]
        )
        return str(result.get("summary", ""))

    # ---------------------------------------------------------------------------
    # Data validation
    # ---------------------------------------------------------------------------

    def validate_data(
        self,
        session_id: str,
        file_name: str,
        date_column: str,
        target_column: str,
    ) -> tuple[pd.DataFrame, str, str, str, str]:
        """Load session file, normalize to ds/y, infer frequency."""
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

        return df, freq, name, date_column, target_column

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
                df, freq, file_name, date_column, target_column = self.validate_data(
                    self.session_id,
                    params.file_name,
                    params.date_column,
                    params.target_column,
                )
                n_obs = int(len(df))

                model, model_spec, fit_quality, fit_warnings, fit_extras = self.fit(
                    df, freq, file_name, date_column, target_column, params
                )
                warnings_out.extend(fit_warnings)

                fitted_table = self.build_fitted_table(df, target_column, model, fit_extras)
                residual_raw = self.analyze_residuals(fitted_table, model_spec, fit_extras)
                residual_lean = self.lean_residual_output(residual_raw)
                for w in residual_raw.get("warnings", []) or []:
                    warnings_out.append({"code": "residual_assumption", "message": str(w)})

                forecast_table = self.generate_forecast(df, freq, model, fit_extras, params)
                decomposition_table = self.build_decomposition(
                    df, model, fit_extras, fitted_table, forecast_table
                )

                fitted_csv = self.artifact_basename(experiment_name, "fitted", "csv")
                forecast_csv = self.artifact_basename(experiment_name, "forecast", "csv")
                fitted_plot = self.artifact_basename(experiment_name, "fitted", "png")
                forecast_plot = self.artifact_basename(experiment_name, "forecast", "png")
                residual_plot = self.artifact_basename(experiment_name, "residuals", "png")

                self.save_table(fitted_table, self.session_id, fitted_csv, "save_fitted")
                self.save_table(forecast_table, self.session_id, forecast_csv, "save_forecast")
                self.save_plot(self.plot_fitted(fitted_table), self.session_id, fitted_plot, "save_fitted_plot")
                self.save_plot(self.plot_forecast(forecast_table), self.session_id, forecast_plot, "save_forecast_plot")
                self.save_plot(self.plot_residuals(fitted_table), self.session_id, residual_plot, "save_residual_plot")

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
                            "preview_head": self.preview_head(fitted_table),
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
                    "fitted_plot": {
                        "description": "Chart of actual vs fitted.",
                        "status": "success",
                        "output": {"file_name": fitted_plot},
                    },
                    "forecast_plot": {
                        "description": "Chart of forecast.",
                        "status": "success",
                        "output": {"file_name": forecast_plot},
                    },
                    "residual_plot": {
                        "description": "Chart of residuals over time.",
                        "status": "success",
                        "output": {"file_name": residual_plot},
                    },
                }

                if decomposition_table is not None:
                    decomp_csv = self.artifact_basename(experiment_name, "decomposition", "csv")
                    decomp_plot = self.artifact_basename(experiment_name, "decomposition", "png")
                    self.save_table(decomposition_table, self.session_id, decomp_csv, "save_decomposition")
                    self.save_plot(
                        self.plot_decomposition(decomposition_table),
                        self.session_id,
                        decomp_plot,
                        "save_decomposition_plot",
                    )
                    fitted_decomp = decomposition_table[
                        decomposition_table["period_type"] == "fitted"
                    ]
                    forecast_decomp = decomposition_table[
                        decomposition_table["period_type"] == "forecast"
                    ]
                    pipeline["fitted_decomposition"] = {
                        "description": "In-sample level/trend/seasonal breakdown.",
                        "status": "success",
                        "output": {
                            "file_name": decomp_csv,
                            "preview_head": self.preview_head(fitted_decomp),
                        },
                    }
                    pipeline["forecast_decomposition"] = {
                        "description": "Forecast-period decomposition (components may be null).",
                        "status": "success",
                        "output": {
                            "preview_head": self.preview_head(forecast_decomp),
                        },
                    }
                    pipeline["decomposition_plot"] = {
                        "description": "Decomposition chart.",
                        "status": "success",
                        "output": {"file_name": decomp_plot},
                    }

                # Brief: LLM summary deferred; run_llm_interpretation_summary kept on base for later.

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
