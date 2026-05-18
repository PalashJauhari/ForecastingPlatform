#!/usr/bin/env python3
"""
Render the ForecastingPlatform LangGraph topology to a PNG under ``artifact/``.

Mirrors ``graph.graph.AnalysisGraph.build_graph`` topology (no imports of ``graph.graph`` —
LLMs / Langfuse are configured there at import time).

Writes ``artifact/langgraph_analysis_flow.png`` (+ Mermaid ``.mermaid``) and mirrors
``artifact/forecasting_agent_langgraph.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, List

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, add_messages
from typing_extensions import NotRequired, TypedDict

ROOT = Path(__file__).resolve().parents[1]


class StubAgentState(TypedDict):
    """Minimal state shape aligned with ``graph.graph.AgentState`` reducers."""

    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    skill_guidance: NotRequired[str]
    planner_skill_guidance: NotRequired[str]
    todos: NotRequired[list[Any]]
    active_skills: List[str]
    active_planner_skills: List[str]


ARTIFACT_DIR = ROOT / "artifact"
LANGGRAPH_FLOW_PNG = ARTIFACT_DIR / "langgraph_analysis_flow.png"
LANGGRAPH_FLOW_MERMAID = ARTIFACT_DIR / "langgraph_analysis_flow.mermaid"
LEGACY_LANGGRAPH_PNG = ARTIFACT_DIR / "forecasting_agent_langgraph.png"


def noop(state):  # noqa: ANN001
    return {}


def route_after_orchestrator(state):  # noqa: ANN001
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return "TodoCompletionGate"


def route_after_todo_gate(state):  # noqa: ANN001
    todos_raw = state.get("todos") or []
    todos = [t for t in todos_raw if isinstance(t, dict)]
    if not todos:
        return "FinalAnswer"
    if all(t.get("status") == "completed" for t in todos):
        return "FinalAnswer"
    return "Orchestrator"


def build_topology_graph():
    builder = StateGraph(StubAgentState)
    for name in (
        "BeginTurn",
        "ProfileSavedData",
        "SummariseConversationalSummary",
        "MergePrep",
        "SelectPlannerSkills",
        "Planner",
        "SelectOrchestratorSkills",
        "Orchestrator",
        "RunTools",
        "ProfileSavedData_PostTools",
        "MergeTools",
        "TodoCompletionGate",
        "FinalAnswer",
    ):
        builder.add_node(name, noop)

    builder.set_entry_point("BeginTurn")
    builder.add_edge("BeginTurn", "ProfileSavedData")
    builder.add_edge("ProfileSavedData", "SummariseConversationalSummary")
    builder.add_edge("SummariseConversationalSummary", "MergePrep")
    builder.add_edge("MergePrep", "SelectPlannerSkills")
    builder.add_edge("SelectPlannerSkills", "Planner")
    builder.add_edge("Planner", "SelectOrchestratorSkills")
    builder.add_edge("SelectOrchestratorSkills", "Orchestrator")
    builder.add_conditional_edges(
        "Orchestrator",
        route_after_orchestrator,
        {"RunTools": "RunTools", "TodoCompletionGate": "TodoCompletionGate"},
    )
    builder.add_conditional_edges(
        "TodoCompletionGate",
        route_after_todo_gate,
        {"Orchestrator": "Orchestrator", "FinalAnswer": "FinalAnswer"},
    )
    builder.add_edge("RunTools", "ProfileSavedData_PostTools")
    builder.add_edge("ProfileSavedData_PostTools", "MergeTools")
    builder.add_edge("MergeTools", "Orchestrator")
    builder.add_edge("FinalAnswer", END)
    return builder.compile()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    compiled = build_topology_graph()
    graph_visual = compiled.get_graph()
    png_bytes = graph_visual.draw_mermaid_png()

    LANGGRAPH_FLOW_PNG.write_bytes(png_bytes)
    LEGACY_LANGGRAPH_PNG.write_bytes(png_bytes)

    mermaid = graph_visual.draw_mermaid()
    LANGGRAPH_FLOW_MERMAID.write_text(
        mermaid + ("\n" if not mermaid.endswith("\n") else ""),
        encoding="utf-8",
    )
    print(f"Saved LangGraph PNG → {LANGGRAPH_FLOW_PNG}")
    print(f"Saved LangGraph Mermaid → {LANGGRAPH_FLOW_MERMAID}")
    print(f"Mirrored legacy PNG → {LEGACY_LANGGRAPH_PNG}")


if __name__ == "__main__":
    main()
