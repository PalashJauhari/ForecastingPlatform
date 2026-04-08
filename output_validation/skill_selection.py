"""
Structured output validation for orchestrator skill selection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillSelection(BaseModel):
    """Structured response for the skill selector."""

    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Subset of the optional overlays: forecasting_strategy, visualization_strategy, "
            "results_communication, data_processing_strategy."
        ),
    )
