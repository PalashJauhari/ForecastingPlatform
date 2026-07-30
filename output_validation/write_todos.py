"""Shared todo typing for Planner output and ``update_todo``."""

from typing import Literal

TodoStatus = Literal["pending", "in_progress", "completed"]
