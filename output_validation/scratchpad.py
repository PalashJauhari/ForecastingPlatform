"""Validation for ``write_scratchpad`` (session state notes)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class WriteScratchpadInput(BaseModel):
    """Arguments for writing one note to graph state ``scratchpad`` (appends via reducer)."""

    note: str = Field(
        description=(
            "Full scratchpad entry: detailed insight, finding, assumption, decision, or reminder kept across turns "
            "(shown verbatim in orchestrator **Scratchpad** JSON). Not for raw CSV/Excel—use workspace files for data."
        ),
    )
    summary: str = Field(
        description=(
            "At most **two short lines** (one line break allowed), shown as this tool's **result message** in chat. "
            "Capture the gist; full detail lives in ``note`` / state."
        ),
    )

    @field_validator("note")
    @classmethod
    def strip_note(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("note must not be empty or whitespace-only")
        if len(s) > 16_000:
            raise ValueError("note must be at most 16000 characters")
        return s

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("summary must not be empty or whitespace-only")
        if s.count("\n") > 1:
            raise ValueError("summary must be at most 2 lines (at most one line break)")
        if len(s) > 400:
            raise ValueError("summary must be at most 400 characters")
        return s
