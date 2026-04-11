"""
Structured output validation for ``build_codegen_requirement``.

The planner tool returns a compact JSON object that the orchestrator can inspect
before deciding whether to answer directly, ask the user for clarification, or
call ``code_pipeline``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from session_paths import LOGICAL_AGENT_PREFIX


class BuildCodegenRequirementOutput(BaseModel):
    """Validated output returned by the requirement-building tool."""

    detailed_requirement: str = Field(
        description="Structured execution brief that can be reviewed before code generation.",
    )
    dataset_paths: list[str] = Field(
        default_factory=list,
        description="Logical workspace paths relevant to the planned code generation task.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions made while drafting the requirement.",
    )
    needs_clarification: bool = Field(
        description="Whether a blocking clarification is still needed before safe execution.",
    )
    clarification_question: str = Field(
        default="",
        description="Single blocking clarification question when clarification is required.",
    )

    @field_validator("detailed_requirement")
    @classmethod
    def validate_detailed_requirement(cls, value: str) -> str:
        """The planner must always return a usable execution brief."""
        value = value.strip()
        if not value:
            raise ValueError("detailed_requirement must not be empty.")
        return value

    @field_validator("dataset_paths")
    @classmethod
    def validate_dataset_paths(cls, value: list[str]) -> list[str]:
        """Paths must be ``agent_filesystem/<session-folder>/input|output/...`` (see planner context)."""
        cleaned: list[str] = []
        seen: set[str] = set()

        for path in value:
            path = path.strip()
            if not path.startswith(LOGICAL_AGENT_PREFIX):
                raise ValueError(
                    f"Dataset path must start with '{LOGICAL_AGENT_PREFIX}': {path!r}",
                )
            if path not in seen:
                cleaned.append(path)
                seen.add(path)

        return cleaned

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, value: list[str]) -> list[str]:
        """Keep assumptions compact and remove empty items."""
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_clarification_fields(self) -> "BuildCodegenRequirementOutput":
        """
        Clarification fields must agree with each other.

        - when clarification is needed, a non-empty question is required
        - when clarification is not needed, the question must be empty
        """
        question = self.clarification_question.strip()

        if self.needs_clarification and not question:
            raise ValueError(
                "clarification_question must be provided when needs_clarification is true.",
            )

        if not self.needs_clarification and question:
            raise ValueError(
                "clarification_question must be empty when needs_clarification is false.",
            )

        self.clarification_question = question
        return self

