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
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "update_todo", "args": {}, "id": "c1"}]),
        ],
    }
    assert route_after_orchestrator(state) == "RunTools"


def test_route_orchestrator_to_todo_gate():
    state = {"messages": [AIMessage(content="Done for now.")]}
    assert route_after_orchestrator(state) == "TodoGate"


def test_route_after_todo_gate_passed():
    assert route_after_todo_gate({"todo_gate_passed": True}) == "FinalAnswer"


def test_route_after_todo_gate_failed():
    assert route_after_todo_gate({"todo_gate_passed": False}) == "Orchestrator"


def test_todo_gate_empty_todos():
    config = {"configurable": {"thread_id": "pytest-session-unit"}}
    result = todo_gate({"todos": []}, config)
    assert result["todo_gate_passed"] is True
    assert "No todos to work on." in result["messages"][0].content


def test_todo_gate_all_completed():
    config = {"configurable": {"thread_id": "pytest-session-unit"}}
    todos = [
        {"id": "1", "content": "a", "status": "completed"},
        {"id": "2", "content": "b", "status": "completed"},
    ]
    result = todo_gate({"todos": todos}, config)
    assert result["todo_gate_passed"] is True
    assert "All todos completed." in result["messages"][0].content


def test_todo_gate_incomplete():
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
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "write_todo", "args": {}, "id": "c1"}]),
        ],
    }
    assert route_after_planner(state) == "RunTools"


def test_planner_ends_on_no_tool_reply():
    state = {"messages": [AIMessage(content="Plan is ready.")]}
    assert route_after_planner(state) == END
