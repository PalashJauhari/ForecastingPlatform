"""Unit tests for coding_tool entry precheck before subgraph invoke."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from sub_agents.coding_sub_agent.validation import CodingToolInput
from tools.coding_tools.coding_tool import coding_tool

coding_tool_module = importlib.import_module("tools.coding_tools.coding_tool")


def make_runtime(session_id: str = "precheck-session") -> SimpleNamespace:
    return SimpleNamespace(
        config={"configurable": {"thread_id": session_id}},
        state={"data_profile": []},
        tool_call_id="call-precheck-1",
    )


def patch_session_root(monkeypatch: pytest.MonkeyPatch, session_dir, session_id: str = "precheck-session") -> str:
    session_dir.mkdir(parents=True, exist_ok=True)

    def fake_session_root(sid: str):
        assert sid == session_id
        return session_dir

    monkeypatch.setattr(
        "sub_agents.coding_sub_agent.validation.session_root",
        fake_session_root,
    )
    return session_id


def test_precheck_missing_input_skips_subgraph(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    mock_run = MagicMock()
    monkeypatch.setattr(coding_tool_module, "coding_graph", SimpleNamespace(run=mock_run))

    raw = coding_tool.func(
        requirements="Plot sales",
        input_files=["missing.csv"],
        output_files=["plot.png"],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    mock_run.assert_not_called()
    assert body["status"] == "failed"
    assert body["failure"]["stage"] == "missing_inputs"
    assert body["report"]["summary"] == "Failed: declared input file(s) not in session."


def test_precheck_ok_invokes_subgraph(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    (tmp_path / "sales.csv").write_text("x\n1\n", encoding="utf-8")
    success_body = {"status": "success", "report": {"status": "success", "summary": "Script ran successfully."}}
    mock_run = MagicMock(return_value=success_body)
    monkeypatch.setattr(coding_tool_module, "coding_graph", SimpleNamespace(run=mock_run))

    raw = coding_tool.func(
        requirements="Summarize sales",
        input_files=["sales.csv"],
        output_files=[],
        runtime=make_runtime(session_id),
    )
    body = json.loads(raw)

    mock_run.assert_called_once()
    assert body["status"] == "success"


def test_precheck_empty_input_files_skips_existence_check(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    mock_run = MagicMock(return_value={"status": "success"})
    monkeypatch.setattr(coding_tool_module, "coding_graph", SimpleNamespace(run=mock_run))

    coding_tool.func(
        requirements="Generate constants",
        input_files=[],
        output_files=[],
        runtime=make_runtime(session_id),
    )

    mock_run.assert_called_once()


def test_pydantic_rejects_bad_input_path():
    with pytest.raises(ValidationError):
        CodingToolInput(
            requirements="x",
            input_files=["agent_filesystem/x.csv"],
            output_files=[],
        )
