"""
Planner sub-graph: PlannerOrchestrator ↔ RunTools → END after write_todo.

Mounted on ``AnalysisGraph`` via ``get_planner_graph()`` — no local checkpointer.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from observability.langfuse_handler import trace_context_from_runnable_config, traced_generation, traced_span, update_llm_generation
from tools.planning.update_todo import merge_todos
from tools.tool_catalog import TOOL_CATALOG_TEXT
from sub_agents.planner_sub_agent.config import PLANNER_MODEL
from sub_agents.planner_sub_agent.prompts import PLANNER_SYSTEM_PROMPT
from sub_agents.planner_sub_agent.tools.ask_user import ask_user
from sub_agents.planner_sub_agent.tools.write_todo import write_todo

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class PlannerTodoEntry(TypedDict):
    """One todo row stored in planner sub-graph state."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"]


class PlannerAgentState(TypedDict):
    """State for the planner sub-graph (shared keys with main ``AgentState``)."""

    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    todos: Annotated[List[PlannerTodoEntry], merge_todos]


# ---------------------------------------------------------------------------
# Nodes + routing
# ---------------------------------------------------------------------------

TOOLS = [write_todo, ask_user]
llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


async def planner_orchestrator(state: PlannerAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Invoke planner LLM with messages, summary, workspace, and orchestrator tool catalog."""
    profile_block = json.dumps(
        state.get("data_profile") or [],
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    summary = state.get("message_summary", "")
    context_content = (
        f"## Conversation Summary\n{summary}\n\n"
        f"## Session workspace (data_profile)\n{profile_block}\n\n"
        f"## Available execution tools (orchestrator-only — do not name these in todos)\n"
        f"{TOOL_CATALOG_TEXT}"
    )
    planner_messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=context_content),
    ]
    trace_context = trace_context_from_runnable_config(config)
    with traced_span("PlannerOrchestrator", trace_context=trace_context) as node_span:
        with traced_generation("PlannerOrchestrator-llm", model=PLANNER_MODEL) as gen:
            response = await llm_with_tools.ainvoke(planner_messages, config=config)
            if gen is not None:
                update_llm_generation(gen, model=PLANNER_MODEL, raw=response)
        tool_calls = response.tool_calls or []
        if node_span is not None:
            node_span.update(output={"tool_calls": [tc["name"] for tc in tool_calls]})
    return {"messages": [response]}


def route_after_planner(state: PlannerAgentState) -> str:
    """Only ``PlannerOrchestrator`` ends planning: no tool calls → ``END``; else ``RunTools``."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return END


def route_after_planner_tools(state: PlannerAgentState) -> str:
    """End subgraph after ``write_todo``; loop back for ``ask_user`` clarifications."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_names = {tc.get("name") for tc in msg.tool_calls}
            if "write_todo" in tool_names:
                return END
            return "PlannerOrchestrator"
    return "PlannerOrchestrator"


# ---------------------------------------------------------------------------
# PlannerGraph
# ---------------------------------------------------------------------------


class PlannerGraph:
    """Wrapper around the planner LangGraph subgraph (mounted on ``AnalysisGraph``).

    Compiled without a checkpointer so ``ask_user`` interrupts use the parent
    ``session_id`` thread (``AnalysisGraph.resume``).
    """

    def __init__(self) -> None:
        builder = StateGraph(PlannerAgentState)
        tool_node = ToolNode(TOOLS)

        builder.add_node("PlannerOrchestrator", planner_orchestrator)
        builder.add_node("RunTools", tool_node)
        builder.set_entry_point("PlannerOrchestrator")
        builder.add_conditional_edges(
            "PlannerOrchestrator",
            route_after_planner,
            {"RunTools": "RunTools", END: END},
        )
        builder.add_conditional_edges(
            "RunTools",
            route_after_planner_tools,
            {"PlannerOrchestrator": "PlannerOrchestrator", END: END},
        )
        self.graph = builder.compile()


planner_graph_instance: PlannerGraph | None = None


def get_planner_graph() -> Any:
    """Return the compiled planner subgraph (lazy singleton)."""
    global planner_graph_instance
    if planner_graph_instance is None:
        planner_graph_instance = PlannerGraph()
    return planner_graph_instance.graph
