"""
Patch one todo's status in graph state (orchestrator bookkeeping).
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langfuse import observe
from langgraph.types import Command

from observability.langfuse_handler import get_langfuse_client
from output_validation.update_todo import UpdateTodoInput

langfuse = get_langfuse_client()


def _normalize_todos_row(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    content = row.get("content")
    status = row.get("status")
    tid = row.get("id")
    if content is None or status is None:
        return None
    if tid is None or str(tid).strip() == "":
        tid = f"legacy-{hash(str(content)) & 0xFFFFFFFF:x}"
    return {"id": str(tid), "content": str(content), "status": str(status)}


@observe(name="tool.update_todo", as_type="tool")
def _update_todo_impl(todo_id: str, status: str, runtime: ToolRuntime) -> Command:
    validated = UpdateTodoInput(todo_id=todo_id, status=status)  # type: ignore[arg-type]
    state = runtime.state or {}
    raw = list(state.get("todos") or [])
    rows: list[dict] = []
    for item in raw:
        norm = _normalize_todos_row(item)
        if norm is not None:
            rows.append(norm)
    target = validated.todo_id.strip()
    patched = False
    new_rows: list[dict] = []
    for r in rows:
        if r["id"] == target:
            new_rows.append({**r, "status": validated.status})
            patched = True
        else:
            new_rows.append(dict(r))
    if not patched:
        tool_content = f"No todo with id `{target}`; Current Todo List ids may have been replaced by a new user turn."
        return Command(
            update={
                "messages": [
                    ToolMessage(content=tool_content, tool_call_id=runtime.tool_call_id),
                ],
            },
        )

    with langfuse.start_as_current_observation(
        name="tool.update_todo.state_update",
        as_type="span",
        input={"todo_id": target, "status": validated.status},
    ) as span:
        span.update(output={"todos": new_rows}, metadata={"patched": True})

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
    """Update **one** todo **status** by **id** from Current Todo List (required for downstream completion checks)."""
    return _update_todo_impl(todo_id=todo_id, status=status, runtime=runtime)
