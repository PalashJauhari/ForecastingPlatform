"""Validate ``update_todo`` tool arguments."""

from __future__ import annotations

from pydantic import BaseModel, Field

from output_validation.write_todos import TodoStatus


class UpdateTodoInput(BaseModel):
    """Update one todo status by id from the planner-provided list."""

    todo_id: str = Field(
        min_length=1,
        max_length=128,
        description='Must match ``id`` from Current Todo List — sequential strings "1", "2", "3", … from Planner.',
    )
    status: TodoStatus = Field(description='New status: "pending", "in_progress", or "completed".')
