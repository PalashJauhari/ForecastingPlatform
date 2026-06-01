"""Unit tests for main, planner, and coding sub-agent graph routing helpers."""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from graph.graph import route_after_orchestrator
from sub_agents.coding_sub_agent.graph import route_codegen_limit_gate
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
    """CodeGenLimitGate sends to CodeGenFailure when codegen_count reached MAX."""
    assert route_codegen_limit_gate({"codegen_count": 3}) == "CodeGenFailure"


def test_route_codegen_limit_gate_over_limit():
    """CodeGenLimitGate sends to CodeGenFailure when codegen_count exceeds MAX."""
    assert route_codegen_limit_gate({"codegen_count": 4}) == "CodeGenFailure"


def test_route_codegen_limit_gate_missing_count():
    """CodeGenLimitGate treats missing codegen_count as zero attempts."""
    assert route_codegen_limit_gate({}) == "CodeGen"
