"""Unit tests for orchestrator execution tool catalog."""

from tools.planning.update_todo import update_todo
from tools.tool_catalog import ORCHESTRATOR_EXECUTION_TOOLS, TOOL_CATALOG_TEXT


def test_orchestrator_execution_tools_count():
    assert len(ORCHESTRATOR_EXECUTION_TOOLS) == 4


def test_orchestrator_execution_tools_exclude_update_todo():
    names = {tool.name for tool in ORCHESTRATOR_EXECUTION_TOOLS}
    assert "update_todo" not in names
    assert update_todo.name == "update_todo"


def test_tool_catalog_text_includes_all_execution_tools():
    for tool in ORCHESTRATOR_EXECUTION_TOOLS:
        assert f"### {tool.name}" in TOOL_CATALOG_TEXT
        assert tool.description in TOOL_CATALOG_TEXT
        assert "Args schema:" in TOOL_CATALOG_TEXT
