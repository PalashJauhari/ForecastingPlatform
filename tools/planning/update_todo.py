# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

"""
Patch one todo's status in graph state (orchestrator bookkeeping).

Planner assigns sequential string ids (``"1"``, ``"2"``, …). This tool updates
``status`` only via ``Command`` patch — it does not replace the full todo list
(see planner ``write_todo`` for full replace).
"""

from typing import Any, get_args

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from observability.langfuse_handler import trace_context_from_runnable_config, traced_span
from output_validation.update_todo import UpdateTodoInput
from output_validation.write_todos import TodoStatus

ALLOWED_TODO_STATUS = frozenset(get_args(TodoStatus))


def merge_todos(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """LangGraph reducer for ``todos``: last write in a step wins (``write_todo`` / ``update_todo``)."""
    if right is None:
        return list(left or [])
    return list(right)


def normalize_todo_row(row: object) -> dict | None:
    """Return a validated todo dict, or ``None`` if the row is missing required fields."""
    if not isinstance(row, dict):
        return None
    content = row.get("content")
    status = row.get("status")
    todo_id = row.get("id")
    # Reject missing or blank id, content, or status (id keeps internal spaces if present).
    if (
        content is None
        or status is None
        or todo_id is None
        or not str(content).strip()
        or not str(status).strip()
        or not str(todo_id).strip()
    ):
        return None
    status_str = str(status).strip()
    if status_str not in ALLOWED_TODO_STATUS:
        return None
    return {"id": str(todo_id), "content": str(content).strip(), "status": status_str}


def update_todo_impl(todo_id: str, status: str, runtime: ToolRuntime) -> Command:
    trace_context = trace_context_from_runnable_config(runtime.config)
    with traced_span("update_todo", trace_context=trace_context, input={"todo_id": todo_id, "status": status}) as outer_span:
        validated = UpdateTodoInput(todo_id=todo_id, status=status)  # type: ignore[arg-type]
        state = runtime.state or {}
        raw_todos = list(state.get("todos") or [])
        # Match ``todo_id`` exactly (planner ids may include spaces; no strip on target).
        target_todo_id = validated.todo_id
        matched = False
        new_rows: list[dict] = []
        for raw_row in raw_todos:
            todo_row = normalize_todo_row(raw_row)
            if todo_row is None:
                continue  # Drop malformed rows; they are not echoed back.
            if todo_row["id"] == target_todo_id:
                new_rows.append({**todo_row, "status": validated.status})
                matched = True
            else:
                new_rows.append(todo_row)

        if matched:
            update: dict[str, Any] = {
                "todos": new_rows,
                "messages": [
                    ToolMessage(
                        content=f"Updated todo `{target_todo_id}` to status `{validated.status}`.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
            span_output: dict[str, Any] = {
                "matched": True,
                "todo_id": target_todo_id,
                "status": validated.status,
            }
        else:
            # Omit ``todos`` so a failed patch does not wipe state (e.g. all rows were malformed).
            update = {
                "messages": [
                    ToolMessage(
                        content=f"No todo with id `{target_todo_id}`.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
            span_output = {"matched": False, "todo_id": target_todo_id}

        if outer_span is not None:
            outer_span.update(output=span_output)

        return Command(update=update)


@tool(args_schema=UpdateTodoInput)
def update_todo(runtime: ToolRuntime, todo_id: str, status: str) -> Command:
    """Update one todo status by planner-assigned ``todo_id`` (``\"1\"``, ``\"2\"``, …) from Current Todo List."""
    return update_todo_impl(todo_id=todo_id, status=status, runtime=runtime)
