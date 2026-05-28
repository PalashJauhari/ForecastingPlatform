"""
Patch one todo's status in graph state (orchestrator bookkeeping).

Planner assigns sequential string ids (``"1"``, ``"2"``, …). This tool updates
``status`` only via ``Command`` patch — it does not replace the full todo list
(see planner ``write_todo`` for full replace).
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from observability.langfuse_handler import trace_context_from_runnable_config, traced_span
from output_validation.update_todo import UpdateTodoInput


def normalize_todo_row(row: object) -> dict | None:
    """Keep rows that match the planner schema: id, content, status."""
    if not isinstance(row, dict):
        return None
    content = row.get("content")
    status = row.get("status")
    tid = row.get("id")
    if content is None or status is None:
        return None
    if tid is None or str(tid).strip() == "":
        return None
    return {"id": str(tid), "content": str(content), "status": str(status)}


def update_todo_impl(todo_id: str, status: str, runtime: ToolRuntime) -> Command:
    trace_context = trace_context_from_runnable_config(runtime.config)
    with traced_span("update_todo", trace_context=trace_context, input={"todo_id": todo_id, "status": status}) as outer_span:
        validated = UpdateTodoInput(todo_id=todo_id, status=status)  # type: ignore[arg-type]
        state = runtime.state or {}
        raw = list(state.get("todos") or [])
        rows: list[dict] = []
        for item in raw:
            norm = normalize_todo_row(item)
            if norm is not None:
                rows.append(norm)
            # Rows missing id/content/status are skipped (not echoed back).

        # Match exact planner id string (e.g. "1", not integer 1).
        target = validated.todo_id.strip()
        patched = False
        new_rows: list[dict] = []
        for row in rows:
            if row["id"] == target:
                new_rows.append({**row, "status": validated.status})
                patched = True
            else:
                new_rows.append(dict(row))

        if not patched:
            tool_content = (
                f"No todo with id `{target}`; Current Todo List ids may have been replaced by a new user turn."
            )
            if outer_span is not None:
                outer_span.update(output={"patched": False, "todo_id": target})
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=tool_content, tool_call_id=runtime.tool_call_id),
                    ],
                },
            )

        with traced_span("update_todo.state_update", input={"todo_id": target, "status": validated.status}) as span:
            if span is not None:
                span.update(output={"todos": new_rows, "patched": True})
            if outer_span is not None:
                outer_span.update(output={"patched": True, "todo_id": target, "status": validated.status})

        return Command(
            update={
                "todos": new_rows,
                "messages": [
                    ToolMessage(
                        content=f"Updated todo `{target}` to status `{validated.status}`.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            },
        )


@tool(args_schema=UpdateTodoInput)
def update_todo(todo_id: str, status: str, runtime: ToolRuntime) -> Command:
    """Update one todo status by planner-assigned ``todo_id`` (``\"1\"``, ``\"2\"``, …) from Current Todo List."""
    return update_todo_impl(todo_id=todo_id, status=status, runtime=runtime)
