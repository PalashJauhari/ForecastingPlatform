"""
Planner sub-graph: PlannerOrchestrator ↔ RunTools (ask_user, write_todo) → END.

Mounted on ``AnalysisGraph`` via ``get_planner_graph()`` — no local checkpointer.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from observability.langfuse_handler import traced_generation, traced_span, update_llm_generation
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
    data_profile: List[Any]
    todos: List[PlannerTodoEntry]


# ---------------------------------------------------------------------------
# Nodes + routing
# ---------------------------------------------------------------------------

TOOLS = [write_todo, ask_user]
llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


def planner_orchestrator(state: PlannerAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Invoke planner LLM; ``write_todo`` commits todos via ``ToolMessage`` + ``todos`` state."""
    profile_block = json.dumps(
        state.get("data_profile") or [],
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    planner_messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=f"## Session workspace (data_profile)\n{profile_block}"),
    ]
    with traced_span("PlannerOrchestrator") as node_span:
        with traced_generation("PlannerOrchestrator-llm", model=PLANNER_MODEL) as gen:
            response = llm_with_tools.invoke(planner_messages, config=config)
            if gen is not None:
                update_llm_generation(gen, model=PLANNER_MODEL, raw=response)
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if node_span is not None:
            node_span.update(output={"tool_calls_count": len(tool_calls)})
    return {"messages": [response]}


def route_after_planner(state: PlannerAgentState) -> str:
    """Only ``PlannerOrchestrator`` ends planning: no tool calls → ``END``; else ``RunTools``."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return END


# ---------------------------------------------------------------------------
# PlannerGraph
# ---------------------------------------------------------------------------


class PlannerGraph:
    """Wrapper around the planner LangGraph subgraph (mounted on ``AnalysisGraph``).

    Compiled without a checkpointer so ``ask_user`` interrupts use the parent
    ``session_id`` thread (``AnalysisGraph.resume``).
    """

    def __init__(self) -> None:
        self.graph = self.build_graph()

    def build_graph(self) -> Any:
        """Construct and compile the planner ``StateGraph`` (no checkpointer)."""
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
        builder.add_edge("RunTools", "PlannerOrchestrator")
        return builder.compile()


planner_graph_instance: PlannerGraph | None = None


def get_planner_graph() -> Any:
    """Return the compiled planner subgraph (lazy singleton)."""
    global planner_graph_instance
    if planner_graph_instance is None:
        planner_graph_instance = PlannerGraph()
    return planner_graph_instance.graph


get_graph = get_planner_graph
