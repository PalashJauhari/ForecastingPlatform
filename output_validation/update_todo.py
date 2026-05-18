"""Validate ``update_todo`` tool arguments."""

from __future__ import annotations

from pydantic import BaseModel, Field

from output_validation.write_todos import TodoStatus


class UpdateTodoInput(BaseModel):
    """Update one todo status by id (required for TodoCompletionGate)."""

    todo_id: str = Field(min_length=1, max_length=128, description="Must match ``id`` from Current Todo List.")
    status: TodoStatus = Field(description='New status: "pending", "in_progress", or "completed".')

    @staticmethod
    def tool_description() -> str:
        return (
            "Update the **status** of **one** todo by its ``id`` from **Current Todo List**. "
            "Call whenever you complete meaningful work on an item or start it (`in_progress`). "
            "Skipping updates breaks downstream completion checks and forces extra turns."
        )
