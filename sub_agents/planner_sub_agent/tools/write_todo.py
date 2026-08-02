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


WRITE_TODO_DESCRIPTION = """Full replacement checklist for this turn; pass the entire list when revising."""


@tool(description=WRITE_TODO_DESCRIPTION, args_schema=WriteTodoInput)
def write_todo(todos: list, runtime: ToolRuntime) -> Command:
    trace_context = trace_context_from_runnable_config(runtime.config)
    # Full replace: sequential ids and all rows start pending (planner never sets status).
    # ``todos`` already validated by args_schema=WriteTodoInput before this body runs.
    rows: list[dict[str, str]] = []
    for index, item in enumerate(todos, start=1):
        rows.append({"id": str(index), "content": item.task.strip(), "status": "pending"})
    ack = "Todos updated for this turn (full list replaced in state)."
    with traced_span("PlannerSubAgent - write_todo", trace_context=trace_context) as span:
        if span is not None:
            span.update(
                input={"todos": [{"task": item.task} for item in todos]},
                output={"todos": rows, "ack": ack},
            )
    # ``todos`` in state is source of truth; ToolMessage is ack only (see planner prompt).
    return Command(
        update={
            "todos": rows,
            "messages": [ToolMessage(content=ack, tool_call_id=runtime.tool_call_id)],
        },
    )
