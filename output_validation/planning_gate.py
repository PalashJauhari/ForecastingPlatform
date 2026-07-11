"""Structured output schema for the planning gate node."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanningGateOutput(BaseModel):
    """Whether the latest user turn needs multi-step planning."""

    decision: Literal["skip", "plan"] = Field(
        description=(
            "skip = single-step task; orchestrator can execute directly. "
            "plan = multi-step or ambiguous; route to Planner subgraph."
        ),
    )
    reason: str = Field(description="Brief justification for the decision.")
