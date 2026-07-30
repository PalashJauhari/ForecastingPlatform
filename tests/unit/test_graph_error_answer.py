"""Unit tests for main-graph error_answer handling."""

import asyncio

from langchain_core.messages import AIMessage

from api.main import user_facing_answer_from_messages
from graph.graph import (
    ERROR_ANSWER_NODE_NAME,
    ERROR_ANSWER_USER_MESSAGE,
    FINAL_ANSWER_NODE_NAME,
    error_answer_node,
    handle_node_failure,
)


class FakeNodeError:
    node = "IsPlanningRequired"
    error = ValueError("json_invalid")


def test_handle_node_failure_routes_to_error_answer():
    cmd = handle_node_failure({}, FakeNodeError())  # type: ignore[arg-type]
    assert cmd.goto == "error_answer"
    assert cmd.update["graph_failure"]["failed_node"] == "IsPlanningRequired"
    assert "json_invalid" in cmd.update["graph_failure"]["detail"]


def test_user_facing_answer_from_error_answer_node():
    msg = AIMessage(content="Please try again later.", name=ERROR_ANSWER_NODE_NAME)
    text = user_facing_answer_from_messages([msg])
    assert text == "Please try again later."


def test_user_facing_answer_prefers_last_ai_message():
    messages = [
        AIMessage(content="older reply", name=FINAL_ANSWER_NODE_NAME),
        AIMessage(content=ERROR_ANSWER_USER_MESSAGE, name=ERROR_ANSWER_NODE_NAME),
    ]
    text = user_facing_answer_from_messages(messages)
    assert text == ERROR_ANSWER_USER_MESSAGE


def test_error_answer_node_returns_message():
    result = asyncio.run(
        error_answer_node(
            {"graph_failure": {"failed_node": "Orchestrator", "detail": "timeout"}},
            {},
        )
    )
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.name == ERROR_ANSWER_NODE_NAME
    assert msg.content == ERROR_ANSWER_USER_MESSAGE
