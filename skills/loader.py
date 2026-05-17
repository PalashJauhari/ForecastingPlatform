""" Expert Skill Loader: Ingests Reasoning (.md) and Patterns (.py) from the skills/ library. """

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from middleware.llm_client import make_llm
from prompts.skill_identification_prompt import SKILL_IDENTIFIER_SYSTEM_PROMPT
from observability.langfuse_handler import get_langfuse_client
from output_validation.skill_selection import SkillSelection
from skills.registry import (
    catalog_for_prompt,
    normalize_selected_skills,
    resolve_skill_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
langfuse = get_langfuse_client()


@observe(name="skills.LoadReasoningSkills")
def LoadReasoningSkills(skill_names: list[str]) -> str:
    """
    Ingest ``approach.md`` for each requested skill and assemble one context block.
    All selected skills are included in full (no token budget).
    Resolves legacy **aliases** (e.g. ``data_integrity`` → ``data-grain-and-integrity``).
    """
    sections: list[str] = []

    for name in skill_names:
        skill_path = resolve_skill_dir(name)
        if skill_path is None:
            continue
        approach_path = skill_path / "approach.md"
        text = approach_path.read_text(encoding="utf-8")
        sections.append(f"## SKILL: {skill_path.name.upper()}\n{text}")

    return "\n\n".join(sections)


@observe(name="skills.LoadPatternSkills")
def LoadPatternSkills(skill_names: list[str]) -> str:
    """
    Ingest 'patterns.py' for requested skills.
    Assembles a block of vetted code snippets for the Code Generator.
    """
    sections: list[str] = []

    for name in skill_names:
        skill_path = resolve_skill_dir(name)
        if skill_path is None:
            continue
        pattern_path = skill_path / "patterns.py"
        if not pattern_path.is_file():
            continue

        code = pattern_path.read_text(encoding="utf-8")
        sections.append(f"# Pattern Library: {skill_path.name.upper()}\n{code}")

    return "\n\n".join(sections)


@observe(name="skills.IdentifySkills", as_type="generation")
def IdentifySkills(messages: list, data_profile: list) -> list[str]:
    """
    LLM-based skill selection using ``skill.yaml`` catalog (names + descriptions).
    """
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_selected = int((cfg.get("skills") or {}).get("max_selected", 8))
    catalog = catalog_for_prompt()
    if not catalog:
        return []

    model_name = cfg["models"].get("orchestrator", "gpt-4o-mini")
    llm = make_llm(model=model_name, temperature=0, output_schema=SkillSelection)

    user_context = (
        "## Available Skills\n"
        "Each skill lists an **id** (use this exact string in `selected_skills`), a name, and a description.\n\n"
        f"{catalog}\n\n"
        f"Select **at most {max_selected}** skills, **ordered most important first**. "
        "Use only **id** values from the list above.\n\n"
        "## Current Data Profile\n"
        f"{json.dumps(data_profile, indent=2, default=str)}\n\n"
        "## Task\n"
        "Choose skills relevant to the next assistant turn using the conversation messages below and the data profile."
    )

    prompt_messages = [
        SystemMessage(content=SKILL_IDENTIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_context),
        *messages,
    ]

    try:
        response = llm.invoke(prompt_messages)
        selected = normalize_selected_skills(list(response.selected_skills or []), max_selected)
        langfuse.update_current_span(metadata={"skills_selected_count": len(selected), "skills_selected": ",".join(selected)})
        return selected
    except Exception:
        return []
