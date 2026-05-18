"""System prompts for planner vs orchestrator skill-router LLMs."""

PLANNER_SKILL_ROUTER_PROMPT = """\
You pick **planner** skills for task decomposition (todos) from one catalog below.

Rules:
1. Each catalog entry has an **id** (backticks). Return only those exact id strings in `selected_planner_skills`.
2. Most important skill first; respect the caller’s maximum count cap.
3. If none apply, return an empty list. Never invent ids.
"""

ORCHESTRATOR_SKILL_ROUTER_PROMPT = """\
You pick **orchestrator** (execution / reasoning) skills from one catalog below.

Rules:
1. Each catalog entry has an **id** (backticks). Return only those exact id strings in `selected_skills`.
2. Most important skill first; respect the caller’s maximum count cap.
3. If none apply, return an empty list. Never invent ids.
"""
