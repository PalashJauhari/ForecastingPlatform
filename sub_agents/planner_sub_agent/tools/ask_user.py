"""
Planner sub-graph tool: pause for user clarification via LangGraph ``interrupt``.

Interrupt pauses the **main** graph thread; resume via ``AnalysisGraph.resume``.
Multiple calls per planning turn are allowed.
"""

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import interrupt

from observability.langfuse_handler import trace_context_from_runnable_config, traced_span
from sub_agents.planner_sub_agent.validation import PlannerAskUserInput

ASK_USER_DESCRIPTION = """Ask the user for clarification before finishing the plan. May be called multiple times."""


@tool(description=ASK_USER_DESCRIPTION, args_schema=PlannerAskUserInput)
def ask_user(clarification_required: str, runtime: ToolRuntime) -> str:
    """Pause planning until the user answers the clarification question."""
    trace_context = trace_context_from_runnable_config(runtime.config)
    # Input is set at span creation so the span paused on ``interrupt`` still shows the question.
    with traced_span(
        "PlannerSubAgent - ask_user",
        trace_context=trace_context,
        input={"clarification_required": clarification_required},
        metadata={"phase": "planner"},
    ) as span:
        response = interrupt({"phase": "planner", "question": clarification_required})
        if span is not None:
            span.update(output={"response": response})
    return response
