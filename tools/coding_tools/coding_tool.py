"""
Main-graph LangChain tool: validate args, invoke coding sub-agent, return JSON.

``ToolRuntime`` supplies ``session_id``, ``tool_call_id``, and ``data_profile``
from the parent graph; return value is JSON string for the orchestrator ToolMessage.
"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from session_paths import session_id_from_config
from sub_agents.coding_sub_agent.graph import invoke_coding_pipeline
from sub_agents.coding_sub_agent.validation import CodingToolInput


@tool(args_schema=CodingToolInput)
def coding_tool(
    requirements: str,
    input_files: list[str],
    output_files: list[str],
    runtime: ToolRuntime,
) -> str:
    """Generate Python, scan with Semgrep and LLM judges, run in E2B sandbox. Returns JSON."""
    validated = CodingToolInput(
        requirements=requirements,
        input_files=input_files,
        output_files=output_files,
    )
    session_id = session_id_from_config(runtime.config)
    tool_call_id = getattr(runtime, "tool_call_id", "") or ""  # drives run_<id>/ plot paths
    state = runtime.state or {}
    data_profile = state.get("data_profile") or []

    body = invoke_coding_pipeline(
        session_id=session_id,
        tool_call_id=tool_call_id,
        requirements=validated.requirements,
        input_files=list(validated.input_files),
        output_files=list(validated.output_files),
        data_profile=list(data_profile),
    )
    return json.dumps(body, default=str)
