"""Prompt for the ``identify_skills`` graph node (one JSON call per new user turn)."""

SKILL_IDENTIFICATION_PROMPT = """You are selecting optional overlay skills for a data-science orchestrator.

The core skill `data_science_workflow` is always loaded separately.
Your job is to choose only the additional overlay skills needed for this task.

Available overlay skills (use these exact id strings):
- forecasting_strategy: reason about time column, target, horizon, chronology, leakage, baselines, and evaluation
- visualization_strategy: choose effective charts and visual communication strategies for the question
- results_communication: explain findings, assumptions, limitations, and business meaning clearly
- data_processing_strategy: reason about cleaning, reshaping, aggregation, merging, and making the dataframe usable for the task

Rules:
- Reply with ONLY a JSON object, no prose: {"skills": ["skill1", "skill2"]}
- Include only overlay skills directly relevant to this specific task.
- If the message is purely conversational with no data-analysis intent (greetings, chit-chat),
  return {"skills": []}.
- If the task is forecasting or time-series oriented, include "forecasting_strategy".
- If the task asks for charts, plots, dashboards, or visual communication, include "visualization_strategy".
- If the task asks for explanations, conclusions, recommendations, or interpretation of outputs, include "results_communication".
- If the task requires cleaning, reshaping, aggregation, joins, or dataframe preparation before analysis, include "data_processing_strategy".
- Do not return all four skills unless the query truly needs all of them.
"""
