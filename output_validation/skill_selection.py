from pydantic import BaseModel, Field


class OrchestratorSkillPick(BaseModel):
    """Structured output: orchestrator skill ids only."""

    selected_skills: list[str] = Field(
        default_factory=list,
        description="Canonical orchestrator skill id strings from the catalog; most important first.",
    )


class PlannerSkillPick(BaseModel):
    """Structured output: planner skill ids only."""

    selected_planner_skills: list[str] = Field(
        default_factory=list,
        description="Canonical planner skill id strings from the catalog; most important first.",
    )
