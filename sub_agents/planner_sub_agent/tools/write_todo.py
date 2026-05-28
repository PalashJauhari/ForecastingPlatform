"""
Replace the planner sub-graph todo list with a renumbered, all-pending checklist.

Full list replace each planner turn; server assigns ids ``"1"``, ``"2"``, …
Updates ``todos`` state and a ``ToolMessage`` the planner orchestrator reads on the next turn.
"""

import json

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from observability.langfuse_handler import traced_span
from sub_agents.planner_sub_agent.validation import WriteTodoInput


def format_write_todo_tool_message(rows: list[dict[str, str]]) -> str:
    """Tool result for ``messages``: summary plus committed todo rows for the orchestrator."""
    count = len(rows)
    list_block = json.dumps(rows, indent=2, ensure_ascii=False)
    return (
        f"Wrote {count} todo(s) for this turn.\n\n"
        f"## Current Todo List\n{list_block}"
    )


@tool(args_schema=WriteTodoInput)
def write_todo(todos: list, runtime: ToolRuntime) -> Command:
    """Replace the session todo list for this user turn with an ordered checklist."""
    validated = WriteTodoInput(todos=todos)
    rows: list[dict[str, str]] = []
    for index, item in enumerate(validated.todos, start=1):
        rows.append({"id": str(index), "content": item.task.strip(), "status": "pending"})
    tool_content = format_write_todo_tool_message(rows)
    with traced_span("write_todo") as span:
        if span is not None:
            span.update(output={"todo_count": len(rows)})
    return Command(
        update={
            "todos": rows,
            "messages": [ToolMessage(content=tool_content, tool_call_id=runtime.tool_call_id)],
        },
    )
