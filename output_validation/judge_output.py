"""
Structured output validation for the LLM judge in ``code_pipeline``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeOutput(BaseModel):
    """Structured accept/reject response from the judge model."""

    passed: bool = Field(description="Whether the generated code is safe and on-spec.")
    detail: str = Field(default="", description="Short rejection reason when passed is false.")

