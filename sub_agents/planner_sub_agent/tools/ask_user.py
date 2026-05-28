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


def ask_user_impl(clarification_required: str, runtime: ToolRuntime) -> str:
    """Pause planning until the user answers the clarification question."""
    trace_context = trace_context_from_runnable_config(runtime.config)
    with traced_span("ask_user", trace_context=trace_context, metadata={"phase": "planner"}) as span:
        response = interrupt({"phase": "planner", "question": clarification_required})
        if span is not None:
            span.update(output={"phase": "planner", "question_preview": clarification_required[:120]})
    return response


@tool(args_schema=PlannerAskUserInput)
def ask_user(clarification_required: str, runtime: ToolRuntime) -> str:
    """Ask the user for clarification before finishing the plan. May be called multiple times."""
    return ask_user_impl(clarification_required=clarification_required, runtime=runtime)
