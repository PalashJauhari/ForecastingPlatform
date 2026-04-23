""" Expert Skill Loader: Ingests Reasoning (.md) and Patterns (.py) from the skills/ library. """

from __future__ import annotations

import os
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from middleware.llm_rate_limit import OPENAI_RATE_LIMITER
from prompts.skill_identification_prompt import SKILL_IDENTIFIER_SYSTEM_PROMPT
from observability.langfuse_handler import serialize_messages, get_langfuse_client
from output_validation.skill_selection import SkillSelection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = Path(__file__).resolve().parent
langfuse = get_langfuse_client()


def GetMaxSkillTokens() -> int:
    """Read the context budget from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg.get("skills", {}).get("max_skill_context_tokens", 3000))


@observe(name="skills.LoadReasoningSkills")
def LoadReasoningSkills(skill_names: list[str]) -> str:
    """
    Ingest 'approach.md' for requested skills. 
    Assembles a single context block for the Orchestrator.
    """
    max_tokens = GetMaxSkillTokens()
    sections: list[str] = []
    current_tokens = 0

    for name in skill_names:
        skill_dir = SKILLS_ROOT / name
        if not skill_dir.is_dir():
            continue

        approach_path = skill_dir / "approach.md"
        if not approach_path.exists():
            continue

        text = approach_path.read_text(encoding="utf-8")
        # Simple token estimation (chars / 4)
        est_tokens = len(text) // 4
        
        if current_tokens + est_tokens > max_tokens:
            sections.append(f"## SKILL: {name.upper()}\n[Context budget reached, skill omitted]")
            break
            
        sections.append(f"## SKILL: {name.upper()}\n{text}")
        current_tokens += est_tokens

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
    _kw = {"model": model_name, "temperature": 0}
    if OPENAI_RATE_LIMITER is not None:
        _kw["rate_limiter"] = OPENAI_RATE_LIMITER
    
    llm = ChatOpenAI(**_kw).with_structured_output(SkillSelection)

    # 3. Build Prompt
    conversation_list = []
    for m in messages:
        role = type(m).__name__.replace("Message", "").replace("AI", "Assistant")
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        conversation_list.append(f"{role}: {content}")
    conversation_text = "\n".join(conversation_list)

    # We separate the Brain (System) from the Data (Human)
    user_context = (
        f"## Conversation History\n{conversation_text}\n\n"
        f"## Available Skills\n{skills_block}\n\n"
        f"## Current Data Profile\n{json.dumps(data_profile, indent=2, default=str)}\n\n"
        "Analyze the context and data profile above, then select the appropriate skills from the list."
    )
    
    prompt_messages = [
        SystemMessage(content=SKILL_IDENTIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_context)
    ]

    # 4. Invoke
    try:
        response = llm.invoke(prompt_messages)
        selected = response.selected_skills
        # Filter to ensure we only return skills that actually exist
        return [s for s in selected if s in available_skills]
    except Exception:
        # Fallback to empty list if LLM fails
        return []
