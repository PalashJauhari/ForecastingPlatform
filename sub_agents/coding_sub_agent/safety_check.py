"""Unified result type for Semgrep and LLM judge gates in the coding sub-agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class SafetyCheckResult:
    """One safety gate outcome (Semgrep static scan or LLM judge)."""

    passed: bool
    source: Literal["semgrep", "judge", "io"]
    detail: str = ""
    violations: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.passed:
            object.__setattr__(self, "detail", self.detail or "")
            object.__setattr__(self, "violations", None)
        elif self.source == "semgrep":
            object.__setattr__(self, "violations", self.violations if self.violations is not None else [])
