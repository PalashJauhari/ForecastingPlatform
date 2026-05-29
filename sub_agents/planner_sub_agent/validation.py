"""
Pydantic schemas for the planner sub-agent ``write_todo`` tool.

Tool-input validation only (planner LLM output shapes live in prompts, not here).
"""

from pydantic import BaseModel, Field


class TodoDraftInput(BaseModel):
    """One planned task row before server-side id assignment."""

    task: str = Field(
        min_length=1,
        max_length=2000,
        description="What the orchestrator should accomplish for this step.",
    )


class PlannerAskUserInput(BaseModel):
    """Arguments for planner ``ask_user``."""

    clarification_required: str = Field(
        min_length=1,
        max_length=4000,
        description="The clarifying question to present to the user before planning continues.",
    )


class WriteTodoInput(BaseModel):
    """Arguments for ``write_todo``: full replacement list for the current user turn."""

    todos: list[TodoDraftInput] = Field(
        min_length=1,
        description="Ordered tasks for this turn; at least one task required.",
    )
