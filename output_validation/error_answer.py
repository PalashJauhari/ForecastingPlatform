"""Structured output for deterministic ``error_answer`` fallback on the main graph."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorAnswerOutput(BaseModel):
    """User-facing fallback when a main-graph LLM node fails after retries."""

    answer: str = Field(description="Plain-language message shown to the user.")
    sources: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(default="low")
