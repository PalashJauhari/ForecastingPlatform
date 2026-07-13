"""Unit tests for ForecastingUnivariateModel pipeline and model smoke runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from output_validation.forecasting_common import BaseForecastToolInput
from output_validation.holt_winters_tool import HoltWintersToolInput
from output_validation.sarima_tool import SarimaToolInput
from tools.forecasting.base import ForecastingUnivariateModel
from tools.forecasting.holt_winters_model import HoltWintersModel
from tools.forecasting.sarima_model import SarimaModel

try:
    import prophet  # noqa: F401

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

if HAS_PROPHET:
    from output_validation.prophet_tool import ProphetToolInput
    from tools.forecasting.prophet_model import ProphetModel

SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data" / "monthly_revenue.csv"


class StubForecastModel(ForecastingUnivariateModel):
    """Minimal concrete model for pipeline JSON shape tests."""

    @property
    def model_type(self) -> str:
        return "stub"

    @property
    def allow_missing_target(self) -> bool:
        return False

    def fit(
        self,
        df: pd.DataFrame,
        params: BaseForecastToolInput,
    ) -> tuple[Any, dict, dict, list[dict]]:
        return (
            {"n": len(df)},
            {"model_type": "stub", "seasonal_period": 12},
            {"mae": 1.0, "definitions": {"mae": "test"}},
            [],
        )

    def analyze_residuals(self, residual_table: pd.DataFrame) -> dict:
        return {"status": "pass", "n_residuals": len(residual_table), "warnings": []}

    def build_fitted_table(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "calendar_date": df["ds"].dt.date.astype(str),
                "fitted": [float(i) for i in range(len(df))],
            }
        )

    def build_forecast_table(self, model: Any, horizon: int, freq: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "calendar_date": ["2099-01-01"],
                "forecast": [42.0],
            }
        )


def make_runtime(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(config={"configurable": {"thread_id": session_id}})


def patch_session_paths(monkeypatch: pytest.MonkeyPatch, session_dir: Path) -> str:
    session_id = "test_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    def fake_session_root(sid: str) -> Path:
        assert sid == session_id
        return session_dir

    def fake_ensure_session_dirs(sid: str) -> Path:
        return fake_session_root(sid)

    monkeypatch.setattr("tools.forecasting.base.session_root", fake_session_root)
    monkeypatch.setattr("tools.forecasting.base.ensure_session_dirs", fake_ensure_session_dirs)
    return session_id


def test_build_residual_table_merge():
    model = StubForecastModel()
    fitted_table = pd.DataFrame(
        {
            "calendar_date": ["2020-01-01", "2020-02-01"],
            "fitted": [10.0, 20.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "calendar_date": ["2020-01-01", "2020-02-01"],
            "actual": [12.0, 18.0],
        }
    )
    result = model.build_residual_table(fitted_table, actuals)
    assert list(result.columns) == ["calendar_date", "actual", "fitted", "residual"]
    assert result["residual"].tolist() == [2.0, -2.0]


def test_build_residual_table_missing_dates_inner_merge():
    model = StubForecastModel()
    fitted_table = pd.DataFrame({"calendar_date": ["2020-01-01"], "fitted": [10.0]})
    actuals = pd.DataFrame(
        {
            "calendar_date": ["2020-01-01", "2020-02-01"],
            "actual": [12.0, 99.0],
        }
    )
    result = model.build_residual_table(fitted_table, actuals)
    assert len(result) == 1
    assert result["calendar_date"].iloc[0] == "2020-01-01"


def test_lean_residual_output_strips_definitions():
    model = StubForecastModel()
    raw = {"status": "pass", "definitions": {"status": "meta"}, "n_residuals": 3}
    lean = model.lean_residual_output(raw)
    assert "definitions" not in lean
    assert lean["n_residuals"] == 3


def test_lean_metrics_strips_definitions():
    model = StubForecastModel()
    raw = {"mae": 1.5, "definitions": {"mae": "meta"}}
    lean = model.lean_metrics(raw)
    assert "definitions" not in lean
    assert lean["mae"] == 1.5


def test_validate_data_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    session_id = patch_session_paths(monkeypatch, tmp_path)
    model = StubForecastModel()
    params = BaseForecastToolInput(
        experiment_name="exp1",
        file_name="missing.csv",
        date_column="month",
        target_column="revenue",
        horizon=3,
    )
    result = json.loads(model.run(runtime=make_runtime(session_id), params=params))
    assert result["status"] == "error"
    assert result["stage"] == "data_validation"
    assert result["error"]["code"] == "file_not_found"


def test_validate_data_bad_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    session_id = patch_session_paths(monkeypatch, tmp_path)
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("month,revenue\n2020-01,1\n")
    model = StubForecastModel()
    params = BaseForecastToolInput(
        experiment_name="exp1",
        file_name="bad.csv",
        date_column="date",
        target_column="revenue",
        horizon=3,
    )
    result = json.loads(model.run(runtime=make_runtime(session_id), params=params))
    assert result["status"] == "error"
    assert result["error"]["code"] == "missing_required_columns"


def test_stub_run_pipeline_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    session_id = patch_session_paths(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_DATA, tmp_path / "monthly_revenue.csv")

    model = StubForecastModel()
    params = BaseForecastToolInput(
        experiment_name="exp_stub",
        file_name="monthly_revenue.csv",
        date_column="month",
        target_column="revenue",
        horizon=3,
    )
    result = json.loads(model.run(runtime=make_runtime(session_id), params=params))

    assert result["status"] == "success"
    pipeline = result["pipeline"]
    assert "fitted_values" in pipeline
    assert "forecast_values" in pipeline
    assert "residual_analysis" in pipeline
    assert "decomposition" not in pipeline
    assert pipeline["fitted_values"]["output"]["file_name"] == "exp_stub_fitted.csv"
    assert pipeline["forecast_values"]["output"]["file_name"] == "exp_stub_forecast.csv"

    fitted_csv = tmp_path / "exp_stub_fitted.csv"
    forecast_csv = tmp_path / "exp_stub_forecast.csv"
    assert fitted_csv.exists()
    assert forecast_csv.exists()

    fitted_df = pd.read_csv(fitted_csv)
    assert list(fitted_df.columns) == ["calendar_date", "actual", "fitted", "residual"]


def test_sarima_run_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    session_id = patch_session_paths(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_DATA, tmp_path / "monthly_revenue.csv")

    params = SarimaToolInput(
        experiment_name="exp_sarima",
        file_name="monthly_revenue.csv",
        date_column="month",
        target_column="revenue",
        horizon=3,
        seasonal_period=12,
        use_auto_arima=True,
    )
    result = json.loads(SarimaModel().run(runtime=make_runtime(session_id), params=params))

    assert result["status"] in {"success", "success_with_warnings"}
    assert result["model_type"] == "sarima"
    assert (tmp_path / "exp_sarima_fitted.csv").exists()
    assert (tmp_path / "exp_sarima_forecast.csv").exists()

    fitted_df = pd.read_csv(tmp_path / "exp_sarima_fitted.csv")
    forecast_df = pd.read_csv(tmp_path / "exp_sarima_forecast.csv")
    assert list(fitted_df.columns) == ["calendar_date", "actual", "fitted", "residual"]
    assert "forecast" in forecast_df.columns
    assert len(forecast_df) == 3


def test_holt_winters_run_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    session_id = patch_session_paths(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_DATA, tmp_path / "monthly_revenue.csv")

    params = HoltWintersToolInput(
        experiment_name="exp_hw",
        file_name="monthly_revenue.csv",
        date_column="month",
        target_column="revenue",
        horizon=3,
        seasonal_period=12,
        trend="add",
        seasonal="add",
        damped_trend=False,
    )
    result = json.loads(HoltWintersModel().run(runtime=make_runtime(session_id), params=params))

    assert result["status"] in {"success", "success_with_warnings"}
    assert result["model_type"] == "holt_winters"
    assert (tmp_path / "exp_hw_fitted.csv").exists()
    assert (tmp_path / "exp_hw_forecast.csv").exists()
    assert "fitted_decomposition" in result["pipeline"]
    assert "forecast_decomposition" in result["pipeline"]


@pytest.mark.skipif(not HAS_PROPHET, reason="prophet not installed")
def test_prophet_run_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    session_id = patch_session_paths(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_DATA, tmp_path / "monthly_revenue.csv")

    params = ProphetToolInput(
        experiment_name="exp_prophet",
        file_name="monthly_revenue.csv",
        date_column="month",
        target_column="revenue",
        horizon=3,
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        weekly_seasonality=False,
        monthly_seasonality=True,
        yearly_seasonality=True,
    )
    result = json.loads(ProphetModel().run(runtime=make_runtime(session_id), params=params))

    assert result["status"] in {"success", "success_with_warnings"}
    assert result["model_type"] == "prophet"
    assert (tmp_path / "exp_prophet_fitted.csv").exists()
    assert (tmp_path / "exp_prophet_forecast.csv").exists()
    assert "fitted_decomposition" in result["pipeline"]
    assert "forecast_decomposition" in result["pipeline"]
