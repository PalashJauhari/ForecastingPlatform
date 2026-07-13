"""Unit tests for read_file_tool."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from output_validation.read_file_tool import ReadFileToolInput
from tools.file_management_tools.read_file_tool import read_file_tool

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "sample_data" / "monthly_revenue.csv"


def make_runtime(session_id: str = "read-file-session") -> SimpleNamespace:
    return SimpleNamespace(config={"configurable": {"thread_id": session_id}})


def patch_session_root(monkeypatch: pytest.MonkeyPatch, session_dir: Path, session_id: str = "read-file-session") -> str:
    session_dir.mkdir(parents=True, exist_ok=True)

    def fake_session_root(sid: str) -> Path:
        assert sid == session_id
        return session_dir

    monkeypatch.setattr(
        "tools.file_management_tools.read_file_tool.session_root",
        fake_session_root,
    )
    return session_id


def test_read_file_success_all_columns(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_CSV, tmp_path / "monthly_revenue.csv")

    raw = read_file_tool.func(
        file_name="monthly_revenue.csv",
        columns=[],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    assert body["status"] == "success"
    assert body["file"] == "monthly_revenue.csv"
    assert body["columns"] == ["month", "revenue"]
    assert body["rows_total"] > 0
    assert body["rows_returned"] == body["rows_total"]
    assert body["truncated"] is False
    assert len(body["data"]) == body["rows_returned"]


def test_read_file_success_subset_columns(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_CSV, tmp_path / "monthly_revenue.csv")

    raw = read_file_tool.func(
        file_name="monthly_revenue.csv",
        columns=["revenue"],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    assert body["status"] == "success"
    assert body["columns"] == ["revenue"]
    assert all(set(row.keys()) == {"revenue"} for row in body["data"])


def test_read_file_missing_column_error(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_CSV, tmp_path / "monthly_revenue.csv")

    raw = read_file_tool.func(
        file_name="monthly_revenue.csv",
        columns=["not_a_column"],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    assert body["status"] == "error"
    assert body["stage"] == "data_validation"
    assert body["error"]["code"] == "missing_required_columns"


def test_read_file_missing_file_error(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)

    raw = read_file_tool.func(
        file_name="absent.csv",
        columns=[],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    assert body["status"] == "error"
    assert body["error"]["code"] == "file_not_found"


def test_read_file_rejects_path():
    with pytest.raises(ValidationError):
        ReadFileToolInput(file_name="agent_filesystem/monthly_revenue.csv", columns=[])


def test_read_file_truncated_when_over_max_rows(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    shutil.copy(SAMPLE_CSV, tmp_path / "monthly_revenue.csv")
    monkeypatch.setenv("READ_FILE_MAX_ROWS", "3")

    raw = read_file_tool.func(
        file_name="monthly_revenue.csv",
        columns=[],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    assert body["status"] == "success"
    assert body["rows_total"] > 3
    assert body["rows_returned"] == 3
    assert body["truncated"] is True


def test_read_file_xlsx(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    xlsx_path = tmp_path / "sheet.xlsx"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(xlsx_path, index=False)

    raw = read_file_tool.func(
        file_name="sheet.xlsx",
        columns=["a"],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    assert body["status"] == "success"
    assert body["columns"] == ["a"]
    assert len(body["data"]) == 2
