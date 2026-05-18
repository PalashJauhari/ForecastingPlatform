"""Expert Skill Loader: Ingests Reasoning (.md) and Patterns (.py) from the skills/ library."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from middleware.llm_client import make_llm
from observability.langfuse_handler import get_langfuse_client
from output_validation.skill_selection import OrchestratorSkillPick, PlannerSkillPick
from prompts.skill_identification_prompt import (
    ORCHESTRATOR_SKILL_ROUTER_PROMPT,
    PLANNER_SKILL_ROUTER_PROMPT,
)
from skills.registry import (
    allowed_orchestrator_skill_ids,
    allowed_planner_skill_ids,
    normalize_skill_pick,
    orchestrator_catalog_text,
    path_for_orchestrator_skill,
    path_for_planner_skill,
    planner_catalog_text,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
langfuse = get_langfuse_client()


@observe(name="skills.LoadReasoningSkills")
def LoadReasoningSkills(skill_names: list[str]) -> str:
    """Ingest ``approach.md`` for each requested **orchestrator** skill and assemble one context block."""
    sections: list[str] = []
    for name in skill_names:
        skill_path = path_for_orchestrator_skill(name)
        if skill_path is None:
            continue
        text = (skill_path / "approach.md").read_text(encoding="utf-8")
        sections.append(f"## SKILL: {skill_path.name.upper()}\n{text}")
    return "\n\n".join(sections)


@observe(name="skills.LoadPlannerSkills")
def LoadPlannerSkills(skill_names: list[str]) -> str:
    """Ingest ``approach.md`` for each requested **planner** skill."""
    sections: list[str] = []
    for name in skill_names:
        skill_path = path_for_planner_skill(name)
        if skill_path is None:
            continue
        text = (skill_path / "approach.md").read_text(encoding="utf-8")
        sections.append(f"## PLANNER SKILL: {skill_path.name.upper()}\n{text}")
    return "\n\n".join(sections)


@observe(name="skills.LoadPatternSkills")
def LoadPatternSkills(skill_names: list[str]) -> str:
    """Ingest ``patterns.py`` for requested **orchestrator** skills (codegen pattern library)."""
    sections: list[str] = []
    for name in skill_names:
        skill_path = path_for_orchestrator_skill(name)
        if skill_path is None:
            continue
        pattern_path = skill_path / "patterns.py"
        if not pattern_path.is_file():
            continue
        code = pattern_path.read_text(encoding="utf-8")
        sections.append(f"# Pattern Library: {skill_path.name.upper()}\n{code}")
    return "\n\n".join(sections)


def _skill_router_settings() -> tuple[str, str, int, int]:
    """Return `(planner_model, orchestrator_model, max_plan, max_orch)` from config; required keys only."""
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse to a mapping at the root.")
    models = cfg.get("models")
    skills_cfg = cfg.get("skills")
    if not isinstance(models, dict):
        raise ValueError("config.yaml missing required top-level key `models` (mapping).")
    if not isinstance(skills_cfg, dict):
        raise ValueError("config.yaml missing required top-level key `skills` (mapping).")

    for key in ("identify_planner_skills", "identify_orchestrator_skills"):
        if key not in models:
            raise ValueError(
                f"config.yaml missing required models.{key} (explicit planner vs orchestrator skill-router models)."
            )
        if models[key] is None or not str(models[key]).strip():
            raise ValueError(f"config.yaml models.{key} must be a non-empty string.")

    planner_model = str(models["identify_planner_skills"]).strip()
    orchestrator_model = str(models["identify_orchestrator_skills"]).strip()

    missing_caps = [k for k in ("max_planner_selected", "max_orchestrator_selected") if k not in skills_cfg]
    if missing_caps:
        raise ValueError(
            "config.yaml skills section missing required key(s): "
            + ", ".join(missing_caps)
            + " (skills.max_planner_selected and skills.max_orchestrator_selected are required)."
        )

    try:
        max_plan = int(skills_cfg["max_planner_selected"])
        max_orch = int(skills_cfg["max_orchestrator_selected"])
    except (TypeError, ValueError) as e:
        raise ValueError(
            "skills.max_planner_selected and skills.max_orchestrator_selected must be integers."
        ) from e
    if max_plan < 1 or max_orch < 1:
        raise ValueError(
            "skills.max_planner_selected and skills.max_orchestrator_selected must be positive integers."
        )
    return planner_model, orchestrator_model, max_plan, max_orch


@observe(name="skills.SelectPlannerSkills", as_type="generation")
def invoke_planner_skill_pick(messages: list, data_profile: list) -> list[str]:
    model_name, _, max_plan, _ = _skill_router_settings()
    catalog = planner_catalog_text()
    if not catalog.strip():
        return []
    allowed = allowed_planner_skill_ids()
    llm = make_llm(model=model_name, temperature=0, output_schema=PlannerSkillPick)
    user_ctx = (
        "## Planner skills (choose by **id** only)\n\n"
        f"{catalog}\n\n"
        f"Select **at most {max_plan}** planner skills.\n\n"
        "## Current Data Profile\n"
        f"{json.dumps(data_profile, indent=2, default=str)}\n\n"
        "## Conversation\nUse messages below plus profile to choose skills for planning this session."
    )
    prompt_msgs = [
        SystemMessage(content=PLANNER_SKILL_ROUTER_PROMPT),
        HumanMessage(content=user_ctx),
        *messages,
    ]
    try:
        raw = llm.invoke(prompt_msgs)
        if isinstance(raw, PlannerSkillPick):
            ids = list(raw.selected_planner_skills or [])
        elif isinstance(raw, dict):
            ids = list(raw.get("selected_planner_skills") or [])
        else:
            ids = list(getattr(raw, "selected_planner_skills", None) or [])
        out = normalize_skill_pick(ids, allowed_ids=allowed, max_n=max_plan)
        langfuse.update_current_span(
            metadata={"planner_skills_selected_count": len(out), "planner_skills_selected": ",".join(out)}
        )
        return out
    except Exception:
        return []


@observe(name="skills.SelectOrchestratorSkills", as_type="generation")
def invoke_orchestrator_skill_pick(messages: list, data_profile: list) -> list[str]:
    _, model_name, _, max_orch = _skill_router_settings()
    catalog = orchestrator_catalog_text()
    if not catalog.strip():
        return []
    allowed = allowed_orchestrator_skill_ids()
    llm = make_llm(model=model_name, temperature=0, output_schema=OrchestratorSkillPick)
    user_ctx = (
        "## Orchestrator skills (choose by **id** only)\n\n"
        f"{catalog}\n\n"
        f"Select **at most {max_orch}** orchestrator skills.\n\n"
        "## Current Data Profile\n"
        f"{json.dumps(data_profile, indent=2, default=str)}\n\n"
        "## Conversation\nUse messages below plus profile to choose skills for execution."
    )
    prompt_msgs = [
        SystemMessage(content=ORCHESTRATOR_SKILL_ROUTER_PROMPT),
        HumanMessage(content=user_ctx),
        *messages,
    ]
    try:
        raw = llm.invoke(prompt_msgs)
        if isinstance(raw, OrchestratorSkillPick):
            ids = list(raw.selected_skills or [])
        elif isinstance(raw, dict):
            ids = list(raw.get("selected_skills") or [])
        else:
            ids = list(getattr(raw, "selected_skills", None) or [])
        out = normalize_skill_pick(ids, allowed_ids=allowed, max_n=max_orch)
        langfuse.update_current_span(
            metadata={"skills_selected_count": len(out), "skills_selected": ",".join(out)}
        )
        return out
    except Exception:
        return []
