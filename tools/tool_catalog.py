"""Read-only catalog of orchestrator execution tools for planner context."""

from __future__ import annotations

import json

from tools.coding_tools.coding_tool import coding_tool
from tools.file_management_tools.read_file_tool import read_file_tool
from tools.forecasting.holt_winters_tool import holt_winters_tool
from tools.forecasting.prophet_tool import prophet_tool
from tools.forecasting.sarima_tool import sarima_tool

ORCHESTRATOR_EXECUTION_TOOLS = [
    coding_tool,
    read_file_tool,
    sarima_tool,
    prophet_tool,
    holt_winters_tool,
]

catalog_parts: list[str] = []
for tool in ORCHESTRATOR_EXECUTION_TOOLS:
    args_schema = getattr(tool, "args_schema", None)
    args_json = (
        json.dumps(args_schema.model_json_schema(), indent=2, ensure_ascii=False)
        if args_schema is not None
        else "{}"
    )
    catalog_parts.append(
        f"### {tool.name}\n"
        f"{tool.description or ''}\n\n"
        f"**Args schema:**\n```json\n{args_json}\n```\n"
    )

TOOL_CATALOG_TEXT = "\n".join(catalog_parts)
