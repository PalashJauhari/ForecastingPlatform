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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = Path(__file__).resolve().parent
langfuse = get_langfuse_client()


@observe(name="skills.LoadReasoningSkills")
def LoadReasoningSkills(skill_names: list[str]) -> str:
    """
    Ingest ``approach.md`` for each requested skill and assemble one context block.
    All selected skills are included in full (no token budget).
    """
    sections: list[str] = []

    for name in skill_names:
        skill_dir = SKILLS_ROOT / name
        if not skill_dir.is_dir():
            continue

        approach_path = skill_dir / "approach.md"
        if not approach_path.exists():
            continue

        text = approach_path.read_text(encoding="utf-8")
        sections.append(f"## SKILL: {name.upper()}\n{text}")

    return "\n\n".join(sections)


@observe(name="skills.LoadPatternSkills")
def LoadPatternSkills(skill_names: list[str]) -> str:
    """
    Ingest 'patterns.py' for requested skills.
    Assembles a block of vetted code snippets for the Code Generator.
    """
    sections: list[str] = []

    for name in skill_names:
        skill_dir = SKILLS_ROOT / name
        if not skill_dir.is_dir():
            continue

        pattern_path = skill_dir / "patterns.py"
        if not pattern_path.exists():
            continue

        code = pattern_path.read_text(encoding="utf-8")
        sections.append(f"# Pattern Library: {name.upper()}\n{code}")

    return "\n\n".join(sections)


@observe(name="skills.IdentifySkills", as_type="generation")
def IdentifySkills(messages: list, data_profile: list) -> list[str]:
    """
    LLM-based skill selection. 
    Analyzes messages and data context to pick specialized skills from the registry.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Discover available skills on disk
    available_skills = [d.name for d in SKILLS_ROOT.iterdir() if d.is_dir()]
    skills_block = "\n".join([f"- {s}" for s in available_skills])

    # 2. Setup LLM
    model_name = cfg["models"].get("orchestrator", "gpt-4o-mini")
    llm = make_llm(model=model_name, temperature=0, output_schema=SkillSelection)

    # Give the selector the registry plus the profiled workspace; the raw conversation is appended below.
    user_context = (
        "## Available Skills\n"
        f"{skills_block}\n\n"
        "## Current Data Profile\n"
        f"{json.dumps(data_profile, indent=2, default=str)}\n\n"
        "## Task\n"
        "Select the appropriate skills from the list using the conversation messages below and the current data profile."
    )

    prompt_messages = [
        SystemMessage(content=SKILL_IDENTIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_context),
        *messages,
    ]

    # 4. Invoke
    try:
        response = llm.invoke(prompt_messages)
        selected = response.selected_skills
        # Guard against hallucinated skill ids from structured output.
        return [s for s in selected if s in available_skills]
    except Exception:
        # Fallback to empty list if LLM fails
        return []
