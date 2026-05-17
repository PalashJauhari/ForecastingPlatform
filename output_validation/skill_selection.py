from pydantic import BaseModel, Field


class SkillSelection(BaseModel):
    """Structured output for LLM-based skill identification."""

    selected_skills: list[str] = Field(
        description=(
            "Canonical skill **id** strings from the catalog (e.g. `forecasting`, `data-grain-and-integrity`). "
            "Most important first; length at most the configured max_selected."
        ),
    )
