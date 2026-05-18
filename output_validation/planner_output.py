"""Structured output for the Planner node (plan vs clarification)."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from output_validation.write_todos import TodoStatus


class PlannerTodoOutput(BaseModel):
    """One todo row emitted by the Planner (requires stable ``id`` for ``update_todo``)."""

    id: str = Field(min_length=1, max_length=128, description="Stable id for update_todo (slug or uuid).")
    content: str = Field(min_length=1, max_length=2000)
    status: TodoStatus = Field(default="pending")


class PlanReady(BaseModel):
    kind: Literal["plan_ready"] = "plan_ready"
    todos: list[PlannerTodoOutput] = Field(
        default_factory=list,
        description="Full replacement list for this user turn; empty if no structured steps.",
    )


class NeedsPlanningClarification(BaseModel):
    kind: Literal["needs_planning_clarification"] = "needs_planning_clarification"
    question: str = Field(min_length=1, max_length=2000)


class PlannerStructuredResponse(BaseModel):
    """Single flattened schema for ``with_structured_output`` (LangChain rejects ``PlanReady | …`` unions)."""

    kind: Literal["plan_ready", "needs_planning_clarification"]
    todos: list[PlannerTodoOutput] = Field(default_factory=list)
    question: str = Field(default="", max_length=2000)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _coherent_by_kind(self) -> PlannerStructuredResponse:
        if self.kind == "needs_planning_clarification":
            if not (self.question or "").strip():
                raise ValueError("needs_planning_clarification requires a non-empty question")
        return self

    def to_step(self) -> PlanReady | NeedsPlanningClarification:
        if self.kind == "plan_ready":
            return PlanReady(todos=list(self.todos))
        return NeedsPlanningClarification(question=(self.question or "").strip())


# App-level discriminated typing (LLM binds ``PlannerStructuredResponse`` above).
PlannerStepOutput = Annotated[
    Union[PlanReady, NeedsPlanningClarification],
    Field(discriminator="kind"),
]
