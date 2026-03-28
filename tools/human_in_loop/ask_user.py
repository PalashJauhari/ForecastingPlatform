"""
LangChain tool: ask the user a clarifying question via LangGraph ``interrupt``.

When invoked, the graph pauses and surfaces the question to the API
caller.  Execution resumes when the API sends ``Command(resume=<answer>)``.
The user's answer is returned as the tool result (a ``ToolMessage``).
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


class AskUserInput(BaseModel):
    """Arguments for ``ask_user``."""

    question: str = Field(
        description="The clarifying question to present to the user.",
    )


@tool(args_schema=AskUserInput)
def ask_user(question: str) -> str:
    """
    Ask the user a clarifying question and pause until they respond.

    Use this when information is missing or ambiguous and you cannot
    proceed without user input.  **This must be the only tool call in the
    step** — do NOT call ``ask_user`` alongside other tools.

    Returns the user's response as a plain string.
    """
    response = interrupt({"question": question})
    return response
