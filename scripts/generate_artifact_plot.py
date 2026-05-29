#!/usr/bin/env python3
"""
Render LangGraph topologies to PNG under ``artifact/``.

Uses stub graphs only — no imports of sub-agent modules that load LLMs at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, List

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, add_messages
from typing_extensions import NotRequired, TypedDict

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifact"

MAIN_PNG = ARTIFACT_DIR / "main_graph_langgraph.png"
MAIN_MERMAID = ARTIFACT_DIR / "main_graph_langgraph.mermaid"
PLANNER_PNG = ARTIFACT_DIR / "planner_sub_agent_langgraph.png"
PLANNER_MERMAID = ARTIFACT_DIR / "planner_sub_agent_langgraph.mermaid"
CODING_PNG = ARTIFACT_DIR / "coding_sub_agent_langgraph.png"
CODING_MERMAID = ARTIFACT_DIR / "coding_sub_agent_langgraph.mermaid"


def noop(state):  # noqa: ANN001
    return {}


def write_artifact(compiled, png_path: Path, mermaid_path: Path) -> None:
    """Write PNG + Mermaid for one compiled graph."""
    graph_visual = compiled.get_graph()
    png_bytes = graph_visual.draw_mermaid_png()
    png_path.write_bytes(png_bytes)
    mermaid = graph_visual.draw_mermaid()
    mermaid_path.write_text(mermaid + ("\n" if not mermaid.endswith("\n") else ""), encoding="utf-8")
    print(f"Saved PNG → {png_path}")
    print(f"Saved Mermaid → {mermaid_path}")


class MainStubState(TypedDict):
    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    todos: NotRequired[list[Any]]


def build_main_graph():
    def route_after_orchestrator(state):  # noqa: ANN001
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "RunTools"
        return "TodoCompletionCheck"

    def route_after_todo_check(state):  # noqa: ANN001
        todos_raw = state.get("todos") or []
        todos = [t for t in todos_raw if isinstance(t, dict)]
        if not todos:
            return "FinalAnswer"
        if all(t.get("status") == "completed" for t in todos):
            return "FinalAnswer"
        return "Orchestrator"

    builder = StateGraph(MainStubState)
    for name in (
        "ProfileSavedData",
        "SummariseConversationalSummary",
        "Planner",
        "Orchestrator",
        "RunTools",
        "ProfileSavedData_PostTools",
        "TodoCompletionCheck",
        "FinalAnswer",
    ):
        builder.add_node(name, noop)

    builder.set_entry_point("ProfileSavedData")
    builder.add_edge("ProfileSavedData", "SummariseConversationalSummary")
    builder.add_edge("SummariseConversationalSummary", "Planner")
    builder.add_edge("Planner", "Orchestrator")
    builder.add_conditional_edges(
        "Orchestrator",
        route_after_orchestrator,
        {"RunTools": "RunTools", "TodoCompletionCheck": "TodoCompletionCheck"},
    )
    builder.add_conditional_edges(
        "TodoCompletionCheck",
        route_after_todo_check,
        {"Orchestrator": "Orchestrator", "FinalAnswer": "FinalAnswer"},
    )
    builder.add_edge("RunTools", "ProfileSavedData_PostTools")
    builder.add_edge("ProfileSavedData_PostTools", "Orchestrator")
    builder.add_edge("FinalAnswer", END)
    return builder.compile()


class PlannerStubState(TypedDict):
    messages: Annotated[list, add_messages]
    todos: List[Any]


def build_planner_graph():
    def route_after_planner(state):  # noqa: ANN001
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "RunTools"
        return END

    builder = StateGraph(PlannerStubState)
    builder.add_node("PlannerOrchestrator", noop)
    builder.add_node("RunTools", noop)
    builder.set_entry_point("PlannerOrchestrator")
    builder.add_conditional_edges(
        "PlannerOrchestrator",
        route_after_planner,
        {"RunTools": "RunTools", END: END},
    )
    builder.add_edge("RunTools", "PlannerOrchestrator")
    return builder.compile()


class CodingStubState(TypedDict):
    code: str
    semgrep_feedback: NotRequired[str]
    judge_feedback: NotRequired[str]
    io_feedback: NotRequired[str]
    execution_feedback: NotRequired[str]
    status: NotRequired[str]


def build_coding_graph():
    def route_after_semgrep(state: CodingStubState) -> str:
        if (state.get("semgrep_feedback") or "").strip():
            return "CodeGen"
        return "SafetyJudge"

    def route_after_safety_judge(state: CodingStubState) -> str:
        if (state.get("judge_feedback") or "").strip():
            return "CodeGen"
        return "IOAllowlistJudge"

    def route_after_io_judge(state: CodingStubState) -> str:
        if (state.get("io_feedback") or "").strip():
            return "CodeGen"
        return "E2BExecute"

    def route_after_e2b(state: CodingStubState) -> str:
        if state.get("status") == "success":
            return END
        return "CodeGen"

    builder = StateGraph(CodingStubState)
    for name in (
        "CodeGen",
        "SemgrepScan",
        "SafetyJudge",
        "IOAllowlistJudge",
        "E2BExecute",
    ):
        builder.add_node(name, noop)

    builder.set_entry_point("CodeGen")
    builder.add_edge("CodeGen", "SemgrepScan")
    builder.add_conditional_edges(
        "SemgrepScan",
        route_after_semgrep,
        {"SafetyJudge": "SafetyJudge", "CodeGen": "CodeGen"},
    )
    builder.add_conditional_edges(
        "SafetyJudge",
        route_after_safety_judge,
        {"IOAllowlistJudge": "IOAllowlistJudge", "CodeGen": "CodeGen"},
    )
    builder.add_conditional_edges(
        "IOAllowlistJudge",
        route_after_io_judge,
        {"E2BExecute": "E2BExecute", "CodeGen": "CodeGen"},
    )
    builder.add_conditional_edges(
        "E2BExecute",
        route_after_e2b,
        {END: END, "CodeGen": "CodeGen"},
    )
    return builder.compile()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_artifact(build_main_graph(), MAIN_PNG, MAIN_MERMAID)
    write_artifact(build_planner_graph(), PLANNER_PNG, PLANNER_MERMAID)
    write_artifact(build_coding_graph(), CODING_PNG, CODING_MERMAID)


if __name__ == "__main__":
    main()
