"""
Planner sub-graph: PlannerOrchestrator ↔ RunTools (write_todo) → END.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from sub_agents.planner_sub_agent.config import (
    PLANNER_MAX_CONCURRENCY,
    PLANNER_MODEL,
    PLANNER_RECURSION_LIMIT,
)
from sub_agents.planner_sub_agent.prompts import PLANNER_SYSTEM_PROMPT
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
    """State for the planner sub-graph."""

    messages: Annotated[list, add_messages]
    todos: List[PlannerTodoEntry]


# ---------------------------------------------------------------------------
# LLM + compiled graph (module singleton)
# ---------------------------------------------------------------------------

TOOLS = [write_todo]
llm = ChatOpenAI(model=PLANNER_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
checkpointer = InMemorySaver()
compiled_graph = None


def build_graph():
    """Construct and compile the planner ``StateGraph``."""
    builder = StateGraph(PlannerAgentState)
    tool_node = ToolNode(TOOLS)

    def planner_orchestrator(state: PlannerAgentState, config: RunnableConfig) -> Dict[str, Any]:
        planner_messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(planner_messages, config=config)
        return {"messages": [response]}

    def route_after_planner(state: PlannerAgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "RunTools"
        return END

    builder.add_node("PlannerOrchestrator", planner_orchestrator)
    builder.add_node("RunTools", tool_node)
    builder.set_entry_point("PlannerOrchestrator")
    builder.add_conditional_edges(
        "PlannerOrchestrator",
        route_after_planner,
        {"RunTools": "RunTools", END: END},
    )
    builder.add_edge("RunTools", "PlannerOrchestrator")
    return builder.compile(checkpointer=checkpointer)


def get_graph():
    """Return the compiled planner sub-graph (lazy singleton)."""
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = build_graph()
    return compiled_graph


def planner_thread_id(session_id: str) -> str:
    """Checkpoint thread id for the planner sub-agent."""
    return f"planner_agent_{session_id}"


def build_transcript(messages: list) -> str:
    """Format HumanMessage and AIMessage turns as plain text (skip ToolMessage)."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"User: {text}")
        elif isinstance(msg, AIMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if text.strip():
                lines.append(f"Assistant: {text}")
    return "\n".join(lines) if lines else "(no prior turns)"


def invoke_planner(
    session_id: str,
    messages: list,
    data_profile: list,
) -> list[dict[str, str]]:
    """
    Run the planner sub-graph and return todo rows for the main graph.

    Returns a list of ``{id, content, status}`` dicts (may be empty).
    """
    graph = get_graph()
    transcript = build_transcript(messages)
    profile_block = (
        "## Session workspace (data_profile)\n"
        f"{json.dumps(data_profile, indent=2, ensure_ascii=False, default=str)}"
    )
    invoke_messages = [
        HumanMessage(content=transcript),
        HumanMessage(content=profile_block),
    ]
    config: Dict[str, Any] = {
        "configurable": {"thread_id": planner_thread_id(session_id)},
        "recursion_limit": PLANNER_RECURSION_LIMIT,
        "max_concurrency": PLANNER_MAX_CONCURRENCY,
    }
    result = graph.invoke({"messages": invoke_messages, "todos": []}, config=config)
    raw = result.get("todos") or []
    out: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        content = row.get("content")
        if content is None:
            continue
        out.append(
            {
                "id": str(row.get("id", "")).strip() or str(len(out) + 1),
                "content": str(content),
                "status": str(row.get("status", "pending")),
            }
        )
    return out
