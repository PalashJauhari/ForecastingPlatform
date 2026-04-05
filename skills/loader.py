"""
Load curated skill text (``approach.md`` + reference ``*.py``) for orchestrator context.

Pure stdlib + pathlib + yaml (reads ``config.yaml`` for token budget).
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = Path(__file__).resolve().parent

PRIORITY_ORDER = [
    "eda",
    "data_processing",
    "feature_engineering",
    "modeling",
    "visualisation",
]


def _max_skill_tokens() -> int:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    return int(cfg.get("skills", {}).get("max_skill_context_tokens", 3000))


def load_skills(skill_names: list[str]) -> str:
    """
    Load ``approach.md`` and reference ``.py`` files for each requested skill.
    Returns a single assembled string for the orchestrator context.
    Truncates when estimated tokens exceed the configured budget.
    """
    if not skill_names:
        return ""

    max_tokens = _max_skill_tokens()
    ordered = [s for s in PRIORITY_ORDER if s in skill_names]
    sections: list[str] = []
    estimated_tokens = 0

    for skill in ordered:
        skill_dir = SKILLS_ROOT / skill
        if not skill_dir.is_dir():
            continue

        approach_path = skill_dir / "approach.md"
        if approach_path.exists():
            approach_text = approach_path.read_text(encoding="utf-8")
            section = f"## Skill: {skill}\n\n{approach_text}"
            section_tokens = len(section) // 4
            if estimated_tokens + section_tokens > max_tokens:
                sections.append(f"## Skill: {skill}\n[Approach truncated — token budget reached]")
                break
            sections.append(section)
            estimated_tokens += section_tokens

        for py_file in sorted(skill_dir.glob("*.py")):
            ref_text = py_file.read_text(encoding="utf-8")
            ref_section = f"### Reference: {py_file.name}\n```python\n{ref_text}\n```"
            ref_tokens = len(ref_section) // 4
            if estimated_tokens + ref_tokens > max_tokens:
                sections.append(f"### Reference: {py_file.name}\n[Truncated — token budget]")
                break
            sections.append(ref_section)
            estimated_tokens += ref_tokens

    return "\n\n".join(sections)
