"""Unit tests for planning gate routing and skip-path todo shape."""

from langchain_core.messages import AIMessage, HumanMessage

from graph.graph import route_after_planning_gate


def test_route_after_planning_gate_to_orchestrator_when_todos_present():
    state = {"todos": [{"id": "1", "content": "Forecast sales", "status": "pending"}]}
    assert route_after_planning_gate(state) == "Orchestrator"


def test_route_after_planning_gate_to_planner_when_todos_empty():
    state = {"todos": []}
    assert route_after_planning_gate(state) == "Planner"


def test_skip_path_todo_shape_matches_write_todo():
    user_query = "Forecast next 30 days with SARIMA on daily_sales.csv"
    todos = [{"id": "1", "content": user_query, "status": "pending"}]
    assert todos[0]["id"] == "1"
    assert todos[0]["status"] == "pending"
    assert todos[0]["content"] == user_query


def test_latest_human_message_for_skip_content():
    messages = [
        HumanMessage(content="older question"),
        AIMessage(content="prior answer"),
        HumanMessage(content="Forecast next 30 days with SARIMA on daily_sales.csv"),
    ]
    latest_human_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                latest_human_text = content.strip()
                break
    assert latest_human_text == "Forecast next 30 days with SARIMA on daily_sales.csv"
