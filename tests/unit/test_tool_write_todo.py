"""Unit tests for planner ``write_todo`` tool and ``merge_todos`` reducer."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sub_agents.planner_sub_agent.tools.write_todo import write_todo
from tools.planning.update_todo import merge_todos


def make_runtime(state_todos: list | None = None) -> SimpleNamespace:
    """Minimal ToolRuntime stub; state todos are not read when building the new list."""
    return SimpleNamespace(
        state={"todos": state_todos or []},
        config={"configurable": {"thread_id": "pytest-session-unit"}},
        tool_call_id="call-1",
    )


def test_merge_todos_right_none_keeps_left():
    """When right is None, keep a copy of the existing todo list."""
    left = [{"id": "1", "content": "A", "status": "pending"}]
    result = merge_todos(left, None)
    assert result == left
    assert result is not left


def test_merge_todos_both_none():
    """Empty state when both left and right are None."""
    assert merge_todos(None, None) == []


def test_merge_todos_right_replaces_left():
    """When right is set, last write wins (full list replace)."""
    left = [{"id": "1", "content": "Old", "status": "pending"}]
    right = [{"id": "1", "content": "New", "status": "pending"}]
    assert merge_todos(left, right) == right


def test_write_todo_assigns_sequential_ids():
    """Assigns string ids '1', '2', … in list order with task text as content."""
    runtime = make_runtime()
    cmd = write_todo.func(
        todos=[{"task": "Load data"}, {"task": "Forecast"}],
        runtime=runtime,
    )
    assert cmd.update["todos"] == [
        {"id": "1", "content": "Load data", "status": "pending"},
        {"id": "2", "content": "Forecast", "status": "pending"},
    ]


def test_write_todo_all_pending():
    """Every row written by the planner starts with status pending."""
    runtime = make_runtime()
    cmd = write_todo.func(todos=[{"task": "Run model"}], runtime=runtime)
    assert all(row["status"] == "pending" for row in cmd.update["todos"])


def test_write_todo_ack_message():
    """Returns a short ToolMessage ack; authoritative list is in todos state."""
    runtime = make_runtime()
    cmd = write_todo.func(todos=[{"task": "Load data"}], runtime=runtime)
    assert "full list replaced" in cmd.update["messages"][0].content


def test_write_todo_overwrites_old_state():
    """Command todos come only from tool args (full replace), not from runtime.state."""
    runtime = make_runtime(
        [{"id": "1", "content": "Old task", "status": "pending"}],
    )
    cmd = write_todo.func(todos=[{"task": "New only"}], runtime=runtime)
    assert len(cmd.update["todos"]) == 1
    assert cmd.update["todos"][0]["content"] == "New only"


def test_write_todo_empty_list_raises():
    """Rejects empty todos list (at least one task required)."""
    runtime = make_runtime()
    with pytest.raises(ValidationError):
        write_todo.func(todos=[], runtime=runtime)


def test_write_todo_empty_task_raises():
    """Rejects tasks with empty string content."""
    runtime = make_runtime()
    with pytest.raises(ValidationError):
        write_todo.func(todos=[{"task": ""}], runtime=runtime)


def test_write_todo_missing_task_raises():
    """Rejects todo items missing the required task field."""
    runtime = make_runtime()
    with pytest.raises(ValidationError):
        write_todo.func(todos=[{"wrong": "field"}], runtime=runtime)


def test_write_todo_task_too_long_raises():
    """Rejects tasks longer than 2000 characters."""
    runtime = make_runtime()
    with pytest.raises(ValidationError):
        write_todo.func(todos=[{"task": "x" * 2001}], runtime=runtime)
