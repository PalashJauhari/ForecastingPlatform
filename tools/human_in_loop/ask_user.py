"""
LangChain tool: ask the user a clarifying question via LangGraph ``interrupt``.

**Orphan:** not registered on main ``AnalysisGraph.TOOLS``. The planner uses
``sub_agents/planner_sub_agent/tools/ask_user.py`` (``phase: planner``).
This copy remains for orchestrator-phase interrupts if wired later.
"""

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from observability.langfuse_handler import trace_context_from_runnable_config, traced_span


class AskUserInput(BaseModel):
    """Arguments for ``ask_user``."""

    question: str = Field(
        description="The clarifying question to present to the user.",
    )


def ask_user_impl(question: str, runtime: ToolRuntime) -> str:
    """
    Ask the user a clarifying question and pause until they respond.

    Use this when information is missing or ambiguous and you cannot
    proceed without user input.  **This must be the only tool call in the
    step** — do NOT call ``ask_user`` alongside other tools.

    Returns the user's response as a plain string.
    """
    trace_context = trace_context_from_runnable_config(runtime.config)
    with traced_span("ask_user", trace_context=trace_context, metadata={"phase": "orchestrator"}) as span:
        response = interrupt({"phase": "orchestrator", "question": question})
        if span is not None:
            span.update(output={"phase": "orchestrator", "question_preview": question[:120]})
    return response


@tool(args_schema=AskUserInput)
def ask_user(question: str, runtime: ToolRuntime) -> str:
    """LangChain wrapper for the traced interrupt tool implementation."""
    return ask_user_impl(question=question, runtime=runtime)
