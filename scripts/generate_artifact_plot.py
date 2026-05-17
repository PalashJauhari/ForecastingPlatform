#!/usr/bin/env python3
"""
Render the ForecastingPlatform LangGraph topology to a PNG under ``artifact/``.

Mirrors ``graph.graph.AnalysisGraph.build_graph``:
``BeginTurn`` → parallel ``ProfileSavedData``, ``DescribePlots``, ``SummariseMessages`` →
``IdentifySkills`` → ``Orchestrator`` → (``RunTools`` | ``FinalAnswer`` → END);
``RunTools`` → ``BeginTurn``.

We build a standalone ``StateGraph`` with no-op nodes so this script never imports
``graph.graph`` (which configures LLMs and Langfuse at import time).

Requires: LangGraph stack from ``requirements.txt`` (uses ``draw_mermaid_png``).
"""

from __future__ import annotations

from operator import add
from pathlib import Path
from typing import Annotated, Any, List

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, add_messages
from typing_extensions import NotRequired, TypedDict

ROOT = Path(__file__).resolve().parents[1]


class StubAgentState(TypedDict):
    """Matches ``graph.graph.AgentState`` channel reducers so parallel prep can render."""

    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    skill_guidance: NotRequired[str]
    plot_descriptions: NotRequired[List[str]]
    todos: NotRequired[list[Any]]
    scratchpad: Annotated[list[str], add]
    active_skills: List[str]


ARTIFACT_DIR = ROOT / "artifact"
OUTPUT_PNG = ARTIFACT_DIR / "forecasting_agent_langgraph.png"


def noop(state):  # noqa: ANN001
    """Stub node — topology only."""
    return {}


def route_after_orchestrator(state):  # noqa: ANN001
    """Match ``graph.graph.route_after_orchestrator`` routing labels for visualization."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return "FinalAnswer"


def build_topology_graph():
    builder = StateGraph(StubAgentState)
    builder.add_node("BeginTurn", noop)
    builder.add_node("ProfileSavedData", noop)
    builder.add_node("DescribePlots", noop)
    builder.add_node("SummariseMessages", noop)
    builder.add_node("IdentifySkills", noop)
    builder.add_node("Orchestrator", noop)
    builder.add_node("RunTools", noop)
    builder.add_node("FinalAnswer", noop)

    builder.set_entry_point("BeginTurn")
    builder.add_edge("BeginTurn", "ProfileSavedData")
    builder.add_edge("BeginTurn", "DescribePlots")
    builder.add_edge("BeginTurn", "SummariseMessages")
    builder.add_edge("ProfileSavedData", "IdentifySkills")
    builder.add_edge("DescribePlots", "IdentifySkills")
    builder.add_edge("SummariseMessages", "IdentifySkills")
    builder.add_edge("IdentifySkills", "Orchestrator")
    builder.add_conditional_edges(
        "Orchestrator",
        route_after_orchestrator,
        {"RunTools": "RunTools", "FinalAnswer": "FinalAnswer"},
    )
    builder.add_edge("RunTools", "BeginTurn")
    builder.add_edge("FinalAnswer", END)
    return builder.compile()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    compiled = build_topology_graph()
    png_bytes = compiled.get_graph().draw_mermaid_png()
    OUTPUT_PNG.write_bytes(png_bytes)
    print(f"Saved LangGraph plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
