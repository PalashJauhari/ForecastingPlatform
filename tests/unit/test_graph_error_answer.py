"""Unit tests for main-graph error_answer handling (RAG-style)."""

import asyncio
import json

from langchain_core.messages import AIMessage

from api.main import user_facing_answer_from_messages
from graph.graph import (
    ERROR_ANSWER_NODE_NAME,
    ERROR_ANSWER_USER_MESSAGE,
    build_error_answer_message,
    error_answer_node,
    handle_node_failure,
)
from output_validation.error_answer import ErrorAnswerOutput


class FakeNodeError:
    node = "IsPlanningRequired"
    error = ValueError("json_invalid")


def test_handle_node_failure_routes_to_error_answer():
    cmd = handle_node_failure({}, FakeNodeError())  # type: ignore[arg-type]
    assert cmd.goto == "error_answer"
    assert cmd.update["graph_failure"]["failed_node"] == "IsPlanningRequired"
    assert "json_invalid" in cmd.update["graph_failure"]["detail"]


def test_build_error_answer_message_json_shape():
    msg = build_error_answer_message()
    assert msg.name == ERROR_ANSWER_NODE_NAME
    payload = json.loads(str(msg.content))
    assert payload["answer"] == ERROR_ANSWER_USER_MESSAGE
    assert payload["confidence"] == "low"
    assert payload["sources"] == []


def test_user_facing_answer_from_error_answer_node():
    msg = build_error_answer_message(
        payload=ErrorAnswerOutput(
            answer="Please try again later.",
            sources=[],
            confidence="low",
        ).model_dump(),
    )
    text = user_facing_answer_from_messages([msg])
    assert text == "Please try again later."


def test_user_facing_answer_prefers_error_answer_over_older_ai():
    messages = [
        AIMessage(content="older reply"),
        build_error_answer_message(),
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
    assert json.loads(msg.content)["answer"] == ERROR_ANSWER_USER_MESSAGE
