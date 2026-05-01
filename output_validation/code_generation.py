"""
Structured output validation for the code generation model inside ``code_pipeline``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeGenerationOutput(BaseModel):
    """Structured response returned by the code generation model."""

    filename: str = Field(description="Descriptive Python filename ending in .py.")
    explanation: str = Field(
        description=(
            "Concise prose for the orchestrator when the script is not echoed: "
            "(1) the user problem solved, "
            "(2) high-level logic / steps, "
            "(3) every **input** table filename read and **output** filenames written or plots saved (**names only**, no paths). "
            "All mandatory; keep it readable in a few sentences."
        ),
    )
    code: str = Field(description="Full runnable Python source.")

