"""Unit tests for ``update_todo_impl``."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tools.planning.update_todo import update_todo_impl


def make_runtime(todos: list) -> SimpleNamespace:
    """Minimal ToolRuntime stub for update_todo_impl tests."""
    return SimpleNamespace(
        state={"todos": todos},
        config={"configurable": {"thread_id": "pytest-session-unit"}},
        tool_call_id="call-1",
    )


def test_update_todo_patches_matching_id():
    """Patches one row by id and leaves other todos unchanged in the full returned list."""
    runtime = make_runtime(
        [
            {"id": "1", "content": "Load data", "status": "pending"},
            {"id": "2", "content": "Forecast", "status": "pending"},
        ],
    )
    cmd = update_todo_impl(todo_id="1", status="in_progress", runtime=runtime)
    assert cmd.update["todos"] == [
        {"id": "1", "content": "Load data", "status": "in_progress"},
        {"id": "2", "content": "Forecast", "status": "pending"},
    ]


def test_update_todo_to_completed():
    """Sets status to completed on a matching todo id."""
    runtime = make_runtime([{"id": "1", "content": "Load data", "status": "pending"}])
    cmd = update_todo_impl(todo_id="1", status="completed", runtime=runtime)
    assert cmd.update["todos"][0]["status"] == "completed"


def test_update_todo_success_message():
    """Success path returns a ToolMessage confirming the id and new status."""
    runtime = make_runtime([{"id": "1", "content": "Load data", "status": "pending"}])
    cmd = update_todo_impl(todo_id="1", status="completed", runtime=runtime)
    assert "Updated todo `1`" in cmd.update["messages"][0].content


def test_update_todo_unknown_id():
    """Unknown id omits todos from the Command and returns an error ToolMessage."""
    runtime = make_runtime([{"id": "1", "content": "Load data", "status": "pending"}])
    cmd = update_todo_impl(todo_id="99", status="completed", runtime=runtime)
    assert "todos" not in cmd.update
    assert "No todo with id" in cmd.update["messages"][0].content


def test_update_todo_empty_todos():
    """Empty state todo list behaves like unknown id: no todos update."""
    runtime = make_runtime([])
    cmd = update_todo_impl(todo_id="1", status="completed", runtime=runtime)
    assert "todos" not in cmd.update
    assert "No todo with id" in cmd.update["messages"][0].content


def test_update_todo_invalid_status():
    """Rejects status values outside pending / in_progress / completed."""
    runtime = make_runtime([{"id": "1", "content": "Load data", "status": "pending"}])
    with pytest.raises(ValidationError):
        update_todo_impl(todo_id="1", status="done", runtime=runtime)


def test_update_todo_empty_todo_id():
    """Rejects empty todo_id before patching state."""
    runtime = make_runtime([{"id": "1", "content": "Load data", "status": "pending"}])
    with pytest.raises(ValidationError):
        update_todo_impl(todo_id="", status="completed", runtime=runtime)


def test_update_todo_skips_malformed_rows():
    """Malformed rows are dropped; valid rows are still patched and returned."""
    runtime = make_runtime(
        [
            {"id": "1", "content": "A", "status": "pending"},
            {},
            {"id": "2", "content": "B", "status": "pending"},
        ],
    )
    cmd = update_todo_impl(todo_id="1", status="in_progress", runtime=runtime)
    assert len(cmd.update["todos"]) == 2
    assert cmd.update["todos"][0]["status"] == "in_progress"


def test_update_todo_skips_empty_content_row():
    """Rows with empty content are not included in the patched todo list."""
    runtime = make_runtime(
        [
            {"id": "1", "content": "A", "status": "pending"},
            {"id": "3", "content": "", "status": "pending"},
        ],
    )
    cmd = update_todo_impl(todo_id="1", status="completed", runtime=runtime)
    assert len(cmd.update["todos"]) == 1
    assert cmd.update["todos"][0]["id"] == "1"


def test_update_todo_skips_invalid_status_row():
    """Rows with non-literal status values are dropped during normalization."""
    runtime = make_runtime(
        [
            {"id": "1", "content": "A", "status": "pending"},
            {"id": "3", "content": "Bad", "status": "bogus"},
        ],
    )
    cmd = update_todo_impl(todo_id="1", status="completed", runtime=runtime)
    assert len(cmd.update["todos"]) == 1


def test_update_todo_id_with_spaces():
    """Id match is exact: spaced id in state matches spaced todo_id arg only."""
    runtime = make_runtime([{"id": " 2 ", "content": "B", "status": "pending"}])
    cmd = update_todo_impl(todo_id=" 2 ", status="in_progress", runtime=runtime)
    assert cmd.update["todos"][0]["status"] == "in_progress"

    runtime = make_runtime([{"id": " 2 ", "content": "B", "status": "pending"}])
    cmd = update_todo_impl(todo_id="2", status="in_progress", runtime=runtime)
    assert "todos" not in cmd.update


def test_update_todo_failed_patch_preserves_state():
    """Failed patch omits todos so malformed-only state is not written as an empty list."""
    runtime = make_runtime([{}, {"bad": "row"}])
    cmd = update_todo_impl(todo_id="1", status="completed", runtime=runtime)
    assert "todos" not in cmd.update
