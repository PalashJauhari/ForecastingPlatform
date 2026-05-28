"""
Replace the planner sub-graph todo list with a renumbered, all-pending checklist.

Full list replace each planner turn; server assigns ids ``"1"``, ``"2"``, …
Authoritative list is ``todos`` state; ``ToolMessage`` is a short acknowledgment only.
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from observability.langfuse_handler import trace_context_from_runnable_config, traced_span
from sub_agents.planner_sub_agent.validation import WriteTodoInput

WRITE_TODO_ACK = "Todos updated for this turn (full list replaced in state)."
MAX_SPAN_TODOS = 50
MAX_SPAN_TODO_CONTENT = 400


@tool(args_schema=WriteTodoInput)
def write_todo(todos: list, runtime: ToolRuntime) -> Command:
    """Replace the session todo list for this user turn with an ordered checklist."""
    trace_context = trace_context_from_runnable_config(runtime.config)
    validated = WriteTodoInput(todos=todos)
    rows: list[dict[str, str]] = []
    for index, item in enumerate(validated.todos, start=1):
        rows.append({"id": str(index), "content": item.task.strip(), "status": "pending"})
    span_todos: list[dict[str, str]] = []
    for row in rows[:MAX_SPAN_TODOS]:
        content = row["content"]
        if len(content) > MAX_SPAN_TODO_CONTENT:
            content = content[: MAX_SPAN_TODO_CONTENT - 1] + "…"
        span_todos.append({"id": row["id"], "content": content, "status": row["status"]})
    span_output: dict[str, object] = {"todos": span_todos, "todo_count": len(rows)}
    if len(rows) > MAX_SPAN_TODOS:
        span_output["todos_truncated"] = True
    with traced_span("write_todo", trace_context=trace_context) as span:
        if span is not None:
            span.update(output=span_output)
    return Command(
        update={
            "todos": rows,
            "messages": [ToolMessage(content=WRITE_TODO_ACK, tool_call_id=runtime.tool_call_id)],
        },
    )
