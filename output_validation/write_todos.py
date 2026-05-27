"""
Shared todo typing for Planner output and ``update_todo``.

``TodoRow`` is the canonical shape in graph ``todos`` after ``write_todo``.
"""

from typing import Literal

from pydantic import BaseModel, Field

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoRow(BaseModel):
    """One todo row in main graph state after planner ``write_todo``."""

    id: str = Field(description='Sequential planner id: "1", "2", "3", …')
    content: str = Field(min_length=1, max_length=2000)
    status: TodoStatus = Field(default="pending")
