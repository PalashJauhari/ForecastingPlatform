"""Load the core workflow skill plus any requested overlay skills for the orchestrator."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = Path(__file__).resolve().parent

ALWAYS_LOADED_SKILLS = [
    "data_science_workflow",
]

OVERLAY_PRIORITY = [
    "forecasting_strategy",
    "visualization_strategy",
    "results_communication",
    "data_processing_strategy",
]


def _max_skill_tokens() -> int:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    return int(cfg.get("skills", {}).get("max_skill_context_tokens", 3000))


def load_skill_context(skill_names: list[str]) -> str:
    """
    Load the always-on workflow skill plus requested overlay skills.

    Returns a single assembled string for the orchestrator context.
    Stops adding more skill sections when the configured budget is reached.
    """
    max_tokens = _max_skill_tokens()
    ordered = ALWAYS_LOADED_SKILLS + [s for s in OVERLAY_PRIORITY if s in skill_names]
    sections: list[str] = []
    estimated_tokens = 0

    for skill in ordered:
        skill_dir = SKILLS_ROOT / skill
        if not skill_dir.is_dir():
            continue

        approach_path = skill_dir / "approach.md"
        if not approach_path.exists():
            continue

        approach_text = approach_path.read_text(encoding="utf-8")
        section = f"## Skill: {skill}\n\n{approach_text}"
        section_tokens = len(section) // 4
        if estimated_tokens + section_tokens > max_tokens:
            sections.append(f"## Skill: {skill}\n[Omitted — skill context budget reached]")
            break
        sections.append(section)
        estimated_tokens += section_tokens

    return "\n\n".join(sections)
