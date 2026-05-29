"""Unit tests for main and planner graph routing helpers."""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from graph.graph import (
    route_after_orchestrator,
    route_after_todo_gate,
    todo_gate,
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


def test_route_orchestrator_to_todo_gate():
    """Main orchestrator routes to TodoGate when the last AIMessage has no tool_calls."""
    state = {"messages": [AIMessage(content="Done for now.")]}
    assert route_after_orchestrator(state) == "TodoGate"


def test_route_after_todo_gate_passed():
    """TodoGate router sends to FinalAnswer when todo_gate_passed is True."""
    assert route_after_todo_gate({"todo_gate_passed": True}) == "FinalAnswer"


def test_route_after_todo_gate_failed():
    """TodoGate router loops back to Orchestrator when todo_gate_passed is False."""
    assert route_after_todo_gate({"todo_gate_passed": False}) == "Orchestrator"


def test_todo_gate_empty_todos():
    """TodoGate passes with 'No todos to work on.' when the todo list is empty."""
    config = {"configurable": {"thread_id": "pytest-session-unit"}}
    result = todo_gate({"todos": []}, config)
    assert result["todo_gate_passed"] is True
    assert "No todos to work on." in result["messages"][0].content


def test_todo_gate_all_completed():
    """TodoGate passes with 'All todos completed.' when every row is completed."""
    config = {"configurable": {"thread_id": "pytest-session-unit"}}
    todos = [
        {"id": "1", "content": "a", "status": "completed"},
        {"id": "2", "content": "b", "status": "completed"},
    ]
    result = todo_gate({"todos": todos}, config)
    assert result["todo_gate_passed"] is True
    assert "All todos completed." in result["messages"][0].content


def test_todo_gate_incomplete():
    """TodoGate blocks and lists pending todos when any row is not completed."""
    config = {"configurable": {"thread_id": "pytest-session-unit"}}
    todos = [
        {"id": "1", "content": "a", "status": "completed"},
        {"id": "2", "content": "Run forecast", "status": "pending"},
    ]
    result = todo_gate({"todos": todos}, config)
    assert result["todo_gate_passed"] is False
    content = result["messages"][0].content
    assert "Pending todo: 2 — Run forecast (pending)" in content


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
