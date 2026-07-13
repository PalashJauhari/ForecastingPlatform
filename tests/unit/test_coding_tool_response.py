"""Unit tests for coding_tool JSON from ``prepare_response_node``."""

from sub_agents.coding_sub_agent.graph import (
    PIPELINE_NODE_ERROR_MESSAGE,
    prepare_response_node,
)


def test_prepare_response_node_error() -> None:
    msg = f"{PIPELINE_NODE_ERROR_MESSAGE} (failed_node=CodeGen; detail=timeout)"
    state = {
        "pipeline_violation": {"stage": "node_error", "message": msg},
        "code_execution_result": {},
        "code": "",
    }
    body = prepare_response_node(state, {})["tool_response"]
    assert body["status"] == "failed"
    assert body["report"]["status"] == "failed"
    assert body["failure"]["stage"] == "node_error"
    assert body["failure"]["message"] == msg
    assert body["execution"]["stderr"] == ""
    assert len(body["report"]["summary"].split("\n")) == 1


def test_prepare_response_node_success() -> None:
    state = {
        "pipeline_violation": {},
        "code_execution_result": {
            "stdout": "ok\n",
            "stderr": "",
            "copied_outputs": ["out.csv"],
            "plots": [],
        },
    }
    body = prepare_response_node(state, {})["tool_response"]
    assert body["status"] == "success"
    assert body["report"]["status"] == "success"
    assert body["failure"] is None
    assert body["execution"]["stdout"] == "ok\n"
    assert body["artifacts"]["outputs"] == ["out.csv"]
    assert body["artifacts"]["plots"] == []
    assert body["report"]["summary"] == "Script ran successfully."


def test_prepare_response_node_missing_inputs_failed() -> None:
    detail = "Declared input file(s) missing from session workspace:\n\n  - sales.csv"
    state = {"pipeline_violation": {"stage": "missing_inputs", "message": detail}}
    body = prepare_response_node(state, {})["tool_response"]
    assert body["status"] == "failed"
    assert body["failure"]["stage"] == "missing_inputs"
    assert body["failure"]["message"] == detail
    assert body["report"]["summary"] == "Failed: declared input file(s) not in session."


def test_prepare_response_node_io_allowlist_violation() -> None:
    detail = "IO allowlist rejected the generated code."
    state = {"pipeline_violation": {"stage": "io_allowlist", "message": detail}}
    body = prepare_response_node(state, {})["tool_response"]
    assert body["status"] == "failed"
    assert body["failure"]["stage"] == "io_allowlist"
    assert body["failure"]["message"] == detail
    assert body["report"]["summary"] == "Failed: code used undeclared file I/O."


def test_prepare_response_plots_basename_only() -> None:
    state = {
        "pipeline_violation": {},
        "code_execution_result": {
            "stdout": "",
            "stderr": "",
            "copied_outputs": [],
            "plots": ["agent_filesystem/sess/run_1/trend.png"],
        },
    }
    body = prepare_response_node(state, {})["tool_response"]
    assert body["artifacts"]["plots"] == ["trend.png"]


def test_prepare_response_summary_one_line() -> None:
    state = {
        "pipeline_violation": {"stage": "e2b", "message": "sandbox failed"},
        "code_execution_result": {"stdout": "", "stderr": "traceback"},
    }
    body = prepare_response_node(state, {})["tool_response"]
    assert len(body["report"]["summary"].split("\n")) == 1
    assert body["report"]["status"] == "failed"


def test_prepare_response_e2b_failure() -> None:
    state = {
        "pipeline_violation": {"stage": "e2b", "message": "exit code 1"},
        "code_execution_result": {"stdout": "", "stderr": "ValueError: bad data"},
    }
    body = prepare_response_node(state, {})["tool_response"]
    assert body["report"]["status"] == "failed"
    assert body["execution"]["stderr"] == "ValueError: bad data"
