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
from sub_agents.coding_sub_agent.validation import (
    CodingToolInput,
    build_coding_tool_response,
    build_missing_inputs_message,
    find_missing_input_files,
)

coding_graph = CodingGraph()


@tool(args_schema=CodingToolInput)
def coding_tool(
    requirements: str,
    input_files: list[str],
    output_files: list[str],
    runtime: ToolRuntime,
) -> str:
    """Flexible Python sandbox for plots, data processing, exploration, and custom analysis.

    ## Purpose
    This is the **flexible** tool when dedicated forecast or read tools are not enough.
    It generates Python, runs it in a sandbox, and saves declared outputs. Use it as
    the general-purpose data workbench: plots, transforms, stats, and custom logic.

    ## When to use
    - **Plots and charts** — line/bar/scatter/decomposition visuals (``.png`` / ``.svg``).
    - **Data in the right format** — clean, resample, merge, or reshape CSV/XLSX so
      ``sarima_tool``, ``prophet_tool``, or ``holt_winters_tool`` can run.
    - **Data exploration** — inspect patterns, filters, groupbys beyond a raw row read.
    - **Data processing** — joins, pivots, missing-value handling, feature engineering.
    - **Statistics** — aggregations, correlations, summaries that ``read_file_tool`` cannot
      compute (read_file returns rows only; coding_tool runs arbitrary analysis).
    - **Custom modeling or logic** — any analysis the built-in forecast tools do not cover.

    ## When NOT to use
    - Standard univariate forecast → ``sarima_tool``, ``prophet_tool``, ``holt_winters_tool``.
    - Simple row preview from a file → ``read_file_tool``.
    - Answer already in ``data_profile`` (filenames, dtypes, 5-row head, column stats).

    ## How to call
    1. Read ``data_profile`` first — pick ``file`` basenames and ``column_profiles[].name``.
    2. ``requirements`` — clear natural-language spec of what the script must do.
    3. ``input_files`` — every ``.csv`` / ``.xlsx`` basename the script may **read**
       (from ``data_profile``); use ``[]`` only if the script needs no inputs.
    4. ``output_files`` — every ``.csv`` / ``.xlsx`` / ``.png`` / ``.svg`` basename the
       script may **write**; list all outputs the script creates.
    5. **Basename only** — e.g. ``monthly_revenue.csv``; never paths or folder prefixes.

    ## Output (JSON string)
    - ``status`` — ``success`` or ``failed``.
    - ``report`` — run status and one-line summary.
    - ``failure`` — ``stage`` + ``message`` when failed.
    - ``execution`` — sandbox ``stdout`` / ``stderr``.
    - ``artifacts`` — saved filenames only: ``outputs`` (tabular), ``plots`` (images).
    - ``code`` — generated source (especially useful on failure).
    On ``missing_inputs``: fix files or ``input_files``; do not retry with same inputs.
    On ``semgrep`` / ``io_allowlist`` / ``e2b``: refine ``requirements`` and retry.
    """
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

    if validated.input_files:
        missing = find_missing_input_files(session_id, list(validated.input_files))
        if missing:
            body = build_coding_tool_response(
                violation={
                    "stage": "missing_inputs",
                    "message": build_missing_inputs_message(missing),
                },
            )
            return json.dumps(body, default=str)

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
