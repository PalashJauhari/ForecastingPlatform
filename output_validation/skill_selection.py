from pydantic import BaseModel, Field

class SkillSelection(BaseModel):
    """Structured output for LLM-based skill identification."""
    selected_skills: list[str] = Field(
        description="List of skill IDs to activate (e.g., ['forecasting', 'data_integrity'])."
    )
