"""
Planner sub-graph tool: pause for user clarification via LangGraph ``interrupt``.

Interrupt pauses the **main** graph thread; resume via ``AnalysisGraph.resume``.
Multiple calls per planning turn are allowed.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langfuse import observe
from langgraph.types import interrupt

from sub_agents.planner_sub_agent.validation import PlannerAskUserInput


@observe(name="tool.planner.ask_user", as_type="tool")
def _ask_user_impl(clarification_required: str) -> str:
    """Pause planning until the user answers the clarification question."""
    response = interrupt({"phase": "planner", "question": clarification_required})
    return response


@tool(args_schema=PlannerAskUserInput)
def ask_user(clarification_required: str) -> str:
    """Ask the user for clarification before finishing the plan. May be called multiple times."""
    return _ask_user_impl(clarification_required=clarification_required)
