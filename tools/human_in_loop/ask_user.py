"""
LangChain tool: ask the user a clarifying question via LangGraph ``interrupt``.

**Orphan:** not registered on main ``AnalysisGraph.TOOLS``. The planner uses
``sub_agents/planner_sub_agent/tools/ask_user.py`` (``phase: planner``).
This copy remains for orchestrator-phase interrupts if wired later.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from observability.langfuse_handler import traced_span


class AskUserInput(BaseModel):
    """Arguments for ``ask_user``."""

    question: str = Field(
        description="The clarifying question to present to the user.",
    )


def _ask_user_impl(question: str) -> str:
    """
    Ask the user a clarifying question and pause until they respond.

    Use this when information is missing or ambiguous and you cannot
    proceed without user input.  **This must be the only tool call in the
    step** — do NOT call ``ask_user`` alongside other tools.

    Returns the user's response as a plain string.
    """
    with traced_span("ask_user", metadata={"phase": "orchestrator"}) as span:
        response = interrupt({"phase": "orchestrator", "question": question})
        if span is not None:
            span.update(output={"phase": "orchestrator", "question_preview": question[:120]})
    return response


@tool(args_schema=AskUserInput)
def ask_user(question: str) -> str:
    """LangChain wrapper for the traced interrupt tool implementation."""
    return _ask_user_impl(question=question)
