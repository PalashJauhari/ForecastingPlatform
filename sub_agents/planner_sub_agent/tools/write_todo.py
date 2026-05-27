"""
Replace the planner sub-graph todo list with a renumbered, all-pending checklist.

Full list replace each planner turn; server assigns ids ``"1"``, ``"2"``, …
Writes through shared ``todos`` channel visible to parent Orchestrator.
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from sub_agents.planner_sub_agent.validation import WriteTodoInput


@tool(args_schema=WriteTodoInput)
def write_todo(todos: list, runtime: ToolRuntime) -> Command:
    """Replace the session todo list for this user turn with an ordered checklist."""
    validated = WriteTodoInput(todos=todos)
    rows: list[dict[str, str]] = []
    for index, item in enumerate(validated.todos, start=1):
        rows.append(
            {
                "id": str(index),
                "content": item.task.strip(),
                "status": "pending",
            }
        )
    summary = f"Wrote {len(rows)} todo(s) for this turn."
    return Command(
        update={
            "todos": rows,
            "messages": [
                ToolMessage(content=summary, tool_call_id=runtime.tool_call_id),
            ],
        },
    )
