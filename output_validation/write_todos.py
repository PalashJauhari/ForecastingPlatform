"""
Structured validation for session todos (legacy ``write_todos`` tool and shared fields).
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """One row in the session todo list."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=128,
        description="Stable id; required for orchestrator update_todo after Planner runs.",
    )
    content: str = Field(
        min_length=1,
        max_length=2000,
        description="Short, actionable description of the task.",
    )
    status: TodoStatus = Field(
        description='One of "pending", "in_progress", or "completed".',
    )

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("content must not be empty or whitespace-only")
        return s


class WriteTodosInput(BaseModel):
    """Arguments the orchestrator passes to ``write_todos``."""

    todos: list[TodoItem] = Field(
        description=(
            "Complete replacement for the session todo list. "
            "Include every task you want to keep, with up-to-date statuses. "
            "Use an empty list only when intentionally clearing the list."
        ),
    )

    @field_validator("todos")
    @classmethod
    def cap_list_length(cls, v: list[TodoItem]) -> list[TodoItem]:
        if len(v) > 40:
            raise ValueError("At most 40 todo items are allowed")
        return v
