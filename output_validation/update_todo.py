"""Validate ``update_todo`` tool arguments.

``tool_description()`` is consumed by graph prompt / tool doc generation — keep in sync with Field text.
"""

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

    @staticmethod
    def tool_description() -> str:
        return (
            "Update the **status** of **one** todo by its ``todo_id`` from **Current Todo List**. "
            "Planner ids are sequential strings like ``\"1\"``, ``\"2\"``. "
            "Call whenever you complete meaningful work on an item or start it (`in_progress`). "
            "Skipping updates breaks downstream completion checks and forces extra turns."
        )
