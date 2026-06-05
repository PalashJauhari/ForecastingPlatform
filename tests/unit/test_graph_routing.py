"""Unit tests for main, planner, and coding sub-agent graph routing helpers."""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from graph.graph import route_after_orchestrator
from sub_agents.coding_sub_agent.graph import (
    route_after_e2b,
    route_after_input_check,
    route_after_io,
    route_after_semgrep,
    route_codegen_limit_gate,
)
from sub_agents.planner_sub_agent.graph import route_after_planner


def test_route_orchestrator_to_run_tools():
    """Main orchestrator routes to RunTools when the last AIMessage has tool_calls."""
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "update_todo", "args": {}, "id": "c1"}]),
        ],
    }
    assert route_after_orchestrator(state) == "RunTools"


def test_route_orchestrator_to_final_answer():
    """Main orchestrator routes to FinalAnswer when the last AIMessage has no tool_calls."""
    state = {"messages": [AIMessage(content="Done for now.")]}
    assert route_after_orchestrator(state) == "FinalAnswer"


def test_planner_routes_to_tools_when_tool_calls():
    """Planner orchestrator routes to RunTools when the last AIMessage has tool_calls."""
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "write_todo", "args": {}, "id": "c1"}]),
        ],
    }
    assert route_after_planner(state) == "RunTools"


def test_planner_ends_on_no_tool_reply():
    """Planner subgraph ends when the orchestrator replies with no tool_calls."""
    state = {"messages": [AIMessage(content="Plan is ready.")]}
    assert route_after_planner(state) == END


def test_route_codegen_limit_gate_under_limit():
    """CodeGenLimitGate sends to CodeGen when codegen_count is below MAX (default 3)."""
    assert route_codegen_limit_gate({"codegen_count": 0}) == "CodeGen"
    assert route_codegen_limit_gate({"codegen_count": 1}) == "CodeGen"
    assert route_codegen_limit_gate({"codegen_count": 2}) == "CodeGen"


def test_route_codegen_limit_gate_at_limit():
    """CodeGenLimitGate sends to CodegenExhausted when codegen_count reached MAX."""
    assert route_codegen_limit_gate({"codegen_count": 3}) == "CodegenExhausted"


def test_route_codegen_limit_gate_over_limit():
    """CodeGenLimitGate sends to CodegenExhausted when codegen_count exceeds MAX."""
    assert route_codegen_limit_gate({"codegen_count": 4}) == "CodegenExhausted"


def test_route_codegen_limit_gate_missing_count():
    """CodeGenLimitGate treats missing codegen_count as zero attempts."""
    assert route_codegen_limit_gate({}) == "CodeGen"


def test_route_after_semgrep_to_io_scan_when_clean() -> None:
    assert route_after_semgrep({"pipeline_violation": {}}) == "IOAllowlistScan"


def test_route_after_semgrep_to_gate_on_failure() -> None:
    assert route_after_semgrep({"pipeline_violation": {"stage": "semgrep", "message": "blocked"}}) == "CodeGenLimitGate"


def test_route_after_io_to_input_check_when_clean() -> None:
    assert route_after_io({"pipeline_violation": {}}) == "InputFilesCheck"


def test_route_after_io_to_gate_on_failure() -> None:
    assert route_after_io({"pipeline_violation": {"stage": "io_allowlist", "message": "undeclared read"}}) == "CodeGenLimitGate"


def test_route_after_input_check_to_e2b_when_present() -> None:
    assert route_after_input_check({"pipeline_violation": {}}) == "E2BExecute"


def test_route_after_input_check_to_prepare_response_when_missing() -> None:
    assert route_after_input_check({"pipeline_violation": {"stage": "missing_inputs", "message": "missing sales.csv"}}) == "PrepareResponse"


def test_route_after_e2b_to_prepare_response_on_success() -> None:
    assert route_after_e2b({"pipeline_violation": {}}) == "PrepareResponse"


def test_route_after_e2b_to_gate_on_failure() -> None:
    assert route_after_e2b({"pipeline_violation": {"stage": "e2b", "message": "stderr"}}) == "CodeGenLimitGate"
