"""Unit tests for planner system prompt tool-boundary rules."""

from sub_agents.planner_sub_agent.prompts import PLANNER_SYSTEM_PROMPT


def test_planner_prompt_forbids_tool_names_in_todos():
    assert "NEVER" in PLANNER_SYSTEM_PROMPT
    assert "reference tool names" not in PLANNER_SYSTEM_PROMPT.lower()


def test_planner_prompt_allows_write_todo_ask_user():
    assert "write_todo" in PLANNER_SYSTEM_PROMPT
    assert "ask_user" in PLANNER_SYSTEM_PROMPT


def test_planner_prompt_outcome_example():
    good_example = "Forecast monthly revenue for 12 periods using monthly_revenue.csv"
    assert good_example in PLANNER_SYSTEM_PROMPT
    assert "prophet_tool" not in good_example
