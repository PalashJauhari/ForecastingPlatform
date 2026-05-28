"""
Base forecasting pipeline for session tabular time-series tools.

``ForecastingModel`` owns the shared template-method pipeline (validate → fit →
fitted table → residual analysis → forecast → save → LLM interpret → JSON).
``SarimaModel``, ``ProphetModel``, and ``HoltWintersModel`` inherit and override model-specific stages.

Design rules:

* Deterministic statistics are the source of truth; LLMs only interpret JSON.
* Hard data errors return clean error JSON; bad residual diagnostics are warnings.
* This module does not create images — use ``coding_tool`` for charts.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from middleware.llm_client import make_llm
from observability.langfuse_handler import trace_context_from_runnable_config, traced_generation, traced_span, update_llm_generation
from output_validation.forecasting_common import BaseForecastToolInput
from session_paths import ensure_session_dirs, session_dir_for_paths, session_id_from_config, session_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

# Reuse the orchestrator model for interpretation: already configured and rate-limited.
INTERPRETATION_MODEL = cfg["models"].get("orchestrator", "gpt-4o-mini")

# Lower bound for a meaningful fit; fail fast below this.
MIN_OBS = 10

ALLOWED_TABLE_EXTS = {".csv", ".xlsx"}

FORECAST_PREVIEW_ROWS = 12
FITTED_PREVIEW_ROWS = 12


class ForecastingToolError(Exception):
    """Domain error raised by pipeline stages so ``run()`` can return clean JSON."""

    # Brief: Store a stable error code, user-facing message, and pipeline stage.
    def __init__(self, code: str, message: str, stage: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


@dataclass
class ValidatedData:
    """Normalized session input after ``validate_data``."""

    df: pd.DataFrame
    freq: str
    file_name: str
    date_column: str
    target_column: str


@dataclass
class FitBundle:
    """Output of ``fit`` — everything downstream needs except forecast tables."""

    model: Any
    model_spec: dict
    fit_quality: dict
    warnings: list[dict] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


@dataclass
class SavedPaths:
    """Logical paths written under ``agent_filesystem/<session>/``."""

    forecast_output_file: str
    fitted_output_file: str
    decomposition_output_file: str | None = None


class ForecastingModel(ABC):
    """
    Template-method base for ``sarima_tool`` and ``prophet_tool``.

    Subclasses override ``fit``, ``analyze_residuals``, ``generate_forecast``,
    ``build_fitted_table``, and ``run_llm_interpretations``.
    """

    session_id: str = ""
    trace_context: Any = None

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short model id surfaced in JSON (``sarima`` or ``prophet``)."""

    @property
    @abstractmethod
    def allow_missing_target(self) -> bool:
        """If false, any missing target row fails validation (SARIMA)."""

    @abstractmethod
    def fit(self, validated: ValidatedData, params: BaseForecastToolInput) -> FitBundle:
        """Fit the model and return spec + fit-quality metrics."""

    @abstractmethod
    def analyze_residuals(self, fitted_table: pd.DataFrame, fit: FitBundle) -> dict:
        """Return residual diagnostics dict with status, metrics, warnings, definitions."""

    @abstractmethod
    def generate_forecast(
        self,
        validated: ValidatedData,
        fit: FitBundle,
        params: BaseForecastToolInput,
    ) -> pd.DataFrame:
        """Return forecast table with at least calendar_date and forecast columns."""

    @abstractmethod
    def build_fitted_table(self, validated: ValidatedData, fit: FitBundle) -> pd.DataFrame:
        """Return in-sample table: calendar_date, actual, fitted, residual."""

    @abstractmethod
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
        """Run structured LLM calls and return the llm_interpretation block."""

    def build_decomposition(
        self,
        validated: ValidatedData,
        fit: FitBundle,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Optional combined fitted + forecast decomposition; default none."""
        return None

    def changepoints_for_response(self, fit: FitBundle) -> dict | None:
        """Optional changepoint block for unified JSON; default none."""
        return None

    # ---------------------------------------------------------------------------
    # Shared pipeline stages
    # ---------------------------------------------------------------------------

    def validate_data(
        self,
        session_id: str,
        file_name: str,
        date_column: str,
        target_column: str,
    ) -> ValidatedData:
        """
        Load CSV/XLSX from the session workspace and normalize to ``ds`` / ``y``.

        Steps: locate file, read by extension, check columns, parse dates, sort,
        dedupe dates, infer regular frequency, sanity-check target column.
        """
        # 1. Locate the file inside the session workspace; error if missing.
        name = Path(file_name).name
        path = session_root(session_id) / name
        if not path.exists() or not path.is_file():
            raise ForecastingToolError(
                "file_not_found",
                f"File '{name}' was not found in the session workspace.",
                "data_validation",
            )

        # 2. Read the file based on its extension.
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

        # 3. Confirm the required date and target columns are present.
        missing_cols = [c for c in (date_column, target_column) if c not in raw_df.columns]
        if missing_cols:
            raise ForecastingToolError(
                "missing_required_columns",
                f"Required columns missing in '{name}': {missing_cols}. Available: {list(raw_df.columns)[:20]}",
                "data_validation",
            )

        df = raw_df[[date_column, target_column]].copy()
        df = df.rename(columns={date_column: "ds", target_column: "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        # 4. Drop unparseable dates, sort, dedupe (keep first).
        df = (
            df.dropna(subset=["ds"])
            .sort_values("ds")
            .drop_duplicates(subset=["ds"], keep="first")
            .reset_index(drop=True)
        )

        # 5. Infer a regular cadence; forecasting needs evenly spaced dates.
        date_idx = pd.DatetimeIndex(df["ds"].values)
        freq = pd.infer_freq(date_idx)
        if freq is None:
            raise ForecastingToolError(
                "frequency_not_inferred",
                "Could not infer a regular frequency from the date column. Resample/clean first or ask the user.",
                "data_validation",
            )

        # 6. Sanity-check the target column.
        if self.allow_missing_target:
            usable = df["y"].dropna()
            if usable.empty:
                raise ForecastingToolError(
                    "all_target_missing",
                    "Target column has no usable numeric values.",
                    "data_validation",
                )
            if len(usable) < MIN_OBS:
                raise ForecastingToolError(
                    "too_few_target_values",
                    f"Need at least {MIN_OBS} non-missing target values; got {len(usable)}.",
                    "data_validation",
                )
            if usable.nunique() <= 1:
                raise ForecastingToolError(
                    "constant_target",
                    "Target column is constant; model cannot be fitted.",
                    "data_validation",
                )
        else:
            n_obs = int(len(df))
            missing_target = int(df["y"].isna().sum())
            if missing_target == n_obs:
                raise ForecastingToolError(
                    "all_target_missing",
                    "Target column has no usable numeric values.",
                    "data_validation",
                )
            if missing_target > 0:
                raise ForecastingToolError(
                    "missing_target_values",
                    f"Target column has {missing_target} missing values. Clean/impute the data before fitting.",
                    "data_validation",
                )
            if n_obs < MIN_OBS:
                raise ForecastingToolError(
                    "too_few_target_values",
                    f"Need at least {MIN_OBS} target values; got {n_obs}.",
                    "data_validation",
                )
            if df["y"].nunique() <= 1:
                raise ForecastingToolError(
                    "constant_target",
                    "Target column is constant; model cannot be fitted.",
                    "data_validation",
                )

        return ValidatedData(
            df=df,
            freq=freq,
            file_name=name,
            date_column=date_column,
            target_column=target_column,
        )

    def save_table(
        self,
        table: pd.DataFrame,
        session_id: str,
        output_file: str,
        stage: str,
    ) -> str:
        """Write ``table`` as CSV/XLSX at the session root; return logical path."""
        name = Path(output_file).name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_TABLE_EXTS:
            raise ForecastingToolError(
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

    def structured_llm_call(
        self,
        name: str,
        schema: type,
        messages: list,
        fallback: dict,
    ) -> dict:
        """One structured interpretation call with Langfuse generation span."""
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
                    update_llm_generation(
                        gen,
                        model=INTERPRETATION_MODEL,
                        raw=raw_msg if isinstance(raw_msg, AIMessage) else None,
                    )
                return parsed.model_dump()
            except Exception as exc:
                if gen is not None:
                    gen.update(output={"error": str(exc)})
                return fallback

    def save_outputs(
        self,
        session_id: str,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
        params: BaseForecastToolInput,
        decomposition_table: pd.DataFrame | None,
    ) -> SavedPaths:
        """Persist forecast + fitted (+ optional decomposition) tables."""
        forecast_path = self.save_table(
            forecast_table,
            session_id,
            params.forecast_output_file,
            "save_forecast",
        )
        fitted_path = self.save_table(
            fitted_table,
            session_id,
            params.fitted_output_file,
            "save_fitted",
        )
        decomposition_path = None
        decomp_file = getattr(params, "decomposition_output_file", None)
        if decomposition_table is not None and decomp_file:
            decomposition_path = self.save_table(
                decomposition_table,
                session_id,
                decomp_file,
                "save_decomposition",
            )
        return SavedPaths(
            forecast_output_file=forecast_path,
            fitted_output_file=fitted_path,
            decomposition_output_file=decomposition_path,
        )

    def build_response(
        self,
        validated: ValidatedData,
        fit: FitBundle,
        fitted_table: pd.DataFrame,
        forecast_table: pd.DataFrame,
        residual_diagnostics: dict,
        paths: SavedPaths,
        llm_interpretation: dict,
        warnings_out: list[dict],
        decomposition_table: pd.DataFrame | None,
    ) -> str:
        """Pack the unified JSON envelope returned to the orchestrator."""
        forecast_rows = forecast_table.to_dict(orient="records")
        fitted_rows = fitted_table.to_dict(orient="records")
        decomposition_rows = (
            decomposition_table.to_dict(orient="records") if decomposition_table is not None else None
        )
        has_warnings = bool(warnings_out)
        response: dict[str, Any] = {
            "status": "success_with_warnings" if has_warnings else "success",
            "model_type": self.model_type,
            "frequency": validated.freq,
            "model": fit.model_spec,
            "fit_quality": fit.fit_quality,
            "residual_diagnostics": residual_diagnostics,
            "changepoints": self.changepoints_for_response(fit),
            "forecast_output_file": paths.forecast_output_file,
            "fitted_output_file": paths.fitted_output_file,
            "decomposition_output_file": paths.decomposition_output_file,
            "horizon": len(forecast_rows),
            "forecast_preview": forecast_rows[:FORECAST_PREVIEW_ROWS],
            "fitted_preview": fitted_rows[-FITTED_PREVIEW_ROWS:],
            "decomposition_preview": (
                decomposition_rows[-FITTED_PREVIEW_ROWS:] if decomposition_rows else None
            ),
            "llm_interpretation": llm_interpretation,
            "warnings": warnings_out,
        }
        return json.dumps(response, default=str)

    def error_json(self, exc: ForecastingToolError) -> str:
        """Serialize a domain error for the orchestrator."""
        return json.dumps(
            {
                "status": "error",
                "stage": exc.stage,
                "error": {"code": exc.code, "message": exc.message},
            },
            default=str,
        )

    def error_json_unknown(self, exc: Exception) -> str:
        """Serialize an unexpected failure."""
        return json.dumps(
            {
                "status": "error",
                "stage": "unknown",
                "error": {"code": "internal_error", "message": str(exc)[:500]},
            },
            default=str,
        )

    def run(self, runtime: ToolRuntime, params: BaseForecastToolInput) -> str:
        """
        End-to-end forecasting pipeline. Returns a JSON string (success or error).

        Subclasses do not override this — they override stage hooks instead.
        """
        self.session_id = session_id_from_config(runtime.config)
        self.trace_context = trace_context_from_runnable_config(runtime.config)
        span_name = f"{self.model_type}_tool"

        with traced_span(span_name, trace_context=self.trace_context, metadata={"session_id": self.session_id}) as tool_span:
            warnings_out: list[dict] = []
            try:
                # 1. Validate and normalize input.
                validated = self.validate_data(
                    self.session_id,
                    params.file_name,
                    params.date_column,
                    params.target_column,
                )

                # 2. Fit the model.
                fit = self.fit(validated, params)
                warnings_out.extend(fit.warnings)

                # 3. Build in-sample fitted table.
                fitted_table = self.build_fitted_table(validated, fit)

                # 4. Residual diagnostics (warnings only, not errors).
                residual_diagnostics = self.analyze_residuals(fitted_table, fit)
                for w in residual_diagnostics.get("warnings", []) or []:
                    warnings_out.append({"code": "residual_assumption", "message": str(w)})

                # 5. Future forecast table.
                forecast_table = self.generate_forecast(validated, fit, params)

                # 6. Optional decomposition (Prophet).
                decomposition_table = self.build_decomposition(validated, fit, fitted_table, forecast_table)

                # 7. Save artifacts to session workspace.
                paths = self.save_outputs(
                    self.session_id,
                    fitted_table,
                    forecast_table,
                    params,
                    decomposition_table,
                )

                # 8. LLM interpretation of deterministic results.
                llm_interpretation = self.run_llm_interpretations(
                    validated,
                    fit,
                    fitted_table,
                    forecast_table,
                    residual_diagnostics,
                    params,
                    decomposition_table,
                )

                # 9. Unified JSON response.
                response = self.build_response(
                    validated,
                    fit,
                    fitted_table,
                    forecast_table,
                    residual_diagnostics,
                    paths,
                    llm_interpretation,
                    warnings_out,
                    decomposition_table,
                )
                if tool_span is not None:
                    tool_span.update(metadata={"warnings": str(len(warnings_out))})
                return response

            except ForecastingToolError as exc:
                if tool_span is not None:
                    tool_span.update(metadata={"status": "error", "stage": exc.stage, "code": exc.code})
                return self.error_json(exc)
            except Exception as exc:
                if tool_span is not None:
                    tool_span.update(metadata={"status": "error", "stage": "unknown", "code": "internal_error"})
                return self.error_json_unknown(exc)
