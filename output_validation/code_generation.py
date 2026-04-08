"""
Structured output validation for the code generation model inside ``code_pipeline``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeGenerationOutput(BaseModel):
    """Structured response returned by the code generation model."""

    filename: str = Field(description="Descriptive Python filename ending in .py.")
    explanation: str = Field(description="Short explanation of the generated script.")
    code: str = Field(description="Full runnable Python source.")

