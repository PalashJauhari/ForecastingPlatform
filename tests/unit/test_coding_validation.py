"""Unit tests for coding sub-agent validation helpers."""

import pytest

from sub_agents.coding_sub_agent.validation import (
    build_coding_tool_response,
    find_missing_input_files,
    validate_input_basename,
    validate_output_basename,
)


def patch_session_root(monkeypatch: pytest.MonkeyPatch, session_dir, session_id: str = "sess-1") -> str:
    session_dir.mkdir(parents=True, exist_ok=True)

    def fake_session_root(sid: str):
        assert sid == session_id
        return session_dir

    monkeypatch.setattr(
        "sub_agents.coding_sub_agent.validation.session_root",
        fake_session_root,
    )
    return session_id


def test_find_missing_input_files_none_missing(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.xlsx").write_bytes(b"")
    assert find_missing_input_files(session_id, ["a.csv", "b.xlsx"]) == []


def test_find_missing_input_files_some_missing(tmp_path, monkeypatch):
    session_id = patch_session_root(monkeypatch, tmp_path)
    (tmp_path / "present.csv").write_text("x\n1\n", encoding="utf-8")
    assert find_missing_input_files(session_id, ["present.csv", "gone.csv"]) == ["gone.csv"]


def test_build_coding_tool_response_success():
    body = build_coding_tool_response(
        exec_result={"stdout": "ok", "stderr": "", "copied_outputs": ["out.csv"], "plots": []},
        code="print(1)",
    )
    assert body["status"] == "success"
    assert body["report"]["status"] == "success"
    assert body["report"]["summary"] == "Script ran successfully."
    assert body["failure"] is None
    assert "outputs" in body["artifacts"]
    assert "plots" in body["artifacts"]
    assert body["code"] == "print(1)"


def test_build_coding_tool_response_missing_inputs():
    body = build_coding_tool_response(
        violation={"stage": "missing_inputs", "message": "missing sales.csv"},
    )
    assert body["status"] == "failed"
    assert body["report"]["status"] == "failed"
    assert body["report"]["summary"] == "Failed: declared input file(s) not in session."
    assert body["failure"]["stage"] == "missing_inputs"
    assert body["execution"]["stdout"] == ""
    assert body["artifacts"]["outputs"] == []


@pytest.mark.parametrize(
    "stage,expected_summary",
    [
        ("semgrep", "Failed: code did not pass safety scan."),
        ("io_allowlist", "Failed: code used undeclared file I/O."),
        ("e2b", "Failed: sandbox execution error."),
        ("codegen_exhausted", "Failed: codegen retry limit reached."),
        ("node_error", "Failed: pipeline node error."),
    ],
)
def test_build_coding_tool_response_each_failure_stage(stage, expected_summary):
    body = build_coding_tool_response(violation={"stage": stage, "message": "detail"})
    assert body["status"] == "failed"
    assert body["report"]["summary"] == expected_summary


def test_validate_input_basename_rejects_path():
    with pytest.raises(ValueError, match="bare filename"):
        validate_input_basename("foo/bar.csv")


def test_validate_output_basename_rejects_path():
    with pytest.raises(ValueError, match="bare filename"):
        validate_output_basename("plots/chart.png")
