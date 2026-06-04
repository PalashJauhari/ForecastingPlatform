"""Unit tests for coding_tool JSON mapping (``build_coding_tool_response``)."""

from sub_agents.coding_sub_agent.graph import (
    PIPELINE_NODE_ERROR_MESSAGE,
    build_coding_tool_response,
)


def test_build_coding_tool_response_includes_code_execution_result_on_node_error() -> None:
    msg = f"{PIPELINE_NODE_ERROR_MESSAGE} (failed_node=CodeGen; detail=timeout)"
    result = {
        "e2b_execution_status": "failed",
        "graph_failure": {"failed_node": "CodeGen", "detail": "timeout"},
        "code_execution_result": {
            "stdout": "",
            "stderr": msg,
            "copied_outputs": [],
            "plots": [],
        },
    }
    body = build_coding_tool_response(result)
    assert body["status"] == "failed"
    assert body["code_execution_result"]["stderr"] == msg
    assert body["stderr"] == msg
    assert body["code_violation"]["node_error"] == "CodeGen: timeout"


def test_build_coding_tool_response_success_includes_code_execution_result() -> None:
    result = {
        "e2b_execution_status": "success",
        "code_execution_result": {
            "stdout": "ok\n",
            "stderr": "",
            "copied_outputs": ["out.csv"],
            "plots": [],
        },
    }
    body = build_coding_tool_response(result)
    assert body["status"] == "success"
    assert body["code_violation"] is None
    assert body["stdout"] == "ok\n"
    assert body["outputs"] == ["out.csv"]
