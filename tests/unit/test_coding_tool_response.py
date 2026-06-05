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
    assert body["failure"]["stage"] == "node_error"
    assert body["failure"]["message"] == msg
    assert body["execution"]["stderr"] == ""


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
    assert body["failure"] is None
    assert body["execution"]["stdout"] == "ok\n"
    assert body["artifacts"]["outputs"] == ["out.csv"]
    assert body["artifacts"]["plots"] == []


def test_prepare_response_node_missing_inputs_failed() -> None:
    detail = "Declared input file(s) missing from session workspace:\n\n  - sales.csv"
    state = {"pipeline_violation": {"stage": "missing_inputs", "message": detail}}
    body = prepare_response_node(state, {})["tool_response"]
    assert body["status"] == "failed"
    assert body["failure"]["stage"] == "missing_inputs"
    assert body["failure"]["message"] == detail


def test_prepare_response_node_io_allowlist_violation() -> None:
    detail = "IO allowlist rejected the generated code."
    state = {"pipeline_violation": {"stage": "io_allowlist", "message": detail}}
    body = prepare_response_node(state, {})["tool_response"]
    assert body["status"] == "failed"
    assert body["failure"]["stage"] == "io_allowlist"
    assert body["failure"]["message"] == detail
