"""
Main-graph LangChain tool: validate args, invoke coding sub-agent, return JSON.

``ToolRuntime`` supplies ``session_id``, ``tool_call_id``, and ``data_profile``
from the parent graph; return value is JSON string for the orchestrator ToolMessage.

Stays a plain sync ``def`` (like the forecasting tools) rather than ``async def``:
LangChain's ``ToolNode.ainvoke()`` offloads sync tool functions to a worker thread,
so the parent graph's event loop is never blocked. The coding subgraph is fully sync
(``CodingGraph.run()`` + ``graph.invoke()``) — no ``asyncio.run`` and no second
event loop.
"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from observability.langfuse_handler import trace_context_for_nested_invoke, trace_context_from_runnable_config
from session_paths import session_id_from_config
from sub_agents.coding_sub_agent.graph import CodingGraph
from sub_agents.coding_sub_agent.validation import CodingToolInput

coding_graph = CodingGraph()


@tool(args_schema=CodingToolInput)
def coding_tool(
    requirements: str,
    input_files: list[str],
    output_files: list[str],
    runtime: ToolRuntime,
) -> str:
    """Generate Python, scan with Semgrep, run in E2B sandbox. Returns JSON."""
    validated = CodingToolInput(
        requirements=requirements,
        input_files=input_files,
        output_files=output_files,
    )
    session_id = session_id_from_config(runtime.config)
    tool_call_id = getattr(runtime, "tool_call_id", "") or ""
    state = runtime.state or {}
    data_profile = state.get("data_profile") or []
    trace_context = trace_context_for_nested_invoke() or trace_context_from_runnable_config(runtime.config)

    body = coding_graph.run(
        session_id=session_id,
        tool_call_id=tool_call_id,
        requirements=validated.requirements,
        input_files=list(validated.input_files),
        output_files=list(validated.output_files),
        data_profile=list(data_profile),
        trace_context=trace_context,
    )
    return json.dumps(body, default=str)
