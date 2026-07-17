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
        return "FinalAnswer"

    builder = StateGraph(MainStubState)
    for name in (
        "ProfileSavedData",
        "SummariseConversationalSummary",
        "IsPlanningRequired",
        "Planner",
        "Orchestrator",
        "RunTools",
        "ProfileSavedData_PostTools",
        "FinalAnswer",
        "error_answer",
    ):
        builder.add_node(name, noop)

    builder.set_entry_point("ProfileSavedData")
    builder.add_edge("ProfileSavedData", "SummariseConversationalSummary")
    builder.add_edge("SummariseConversationalSummary", "IsPlanningRequired")
    builder.add_edge("IsPlanningRequired", "Planner")
    builder.add_edge("Planner", "Orchestrator")
    builder.add_conditional_edges(
        "Orchestrator",
        route_after_orchestrator,
        {"RunTools": "RunTools", "FinalAnswer": "FinalAnswer"},
    )
    builder.add_edge("RunTools", "ProfileSavedData_PostTools")
    builder.add_edge("ProfileSavedData_PostTools", "Orchestrator")
    builder.add_edge("FinalAnswer", END)
    builder.add_edge("error_answer", END)
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


class PipelineViolationStub(TypedDict):
    stage: str
    message: str


class CodingStubState(TypedDict):
    code: str
    codegen_count: NotRequired[int]
    pipeline_violation: NotRequired[PipelineViolationStub]


def build_coding_graph():
    def has_violation(state: CodingStubState) -> bool:
        v = state.get("pipeline_violation") or {}
        return bool((v.get("message") or "").strip())

    def route_codegen_limit_gate(state: CodingStubState) -> str:
        if int(state.get("codegen_count") or 0) >= 3:
            return "CodegenExhausted"
        return "CodeGen"

    def route_after_semgrep(state: CodingStubState) -> str:
        if has_violation(state):
            return "CodeGenLimitGate"
        return "IOAllowlistScan"

    def route_after_io(state: CodingStubState) -> str:
        if has_violation(state):
            return "CodeGenLimitGate"
        return "E2BExecute"

    def route_after_e2b(state: CodingStubState) -> str:
        if has_violation(state):
            return "CodeGenLimitGate"
        return "PrepareResponse"

    builder = StateGraph(CodingStubState)
    for name in (
        "CodeGenLimitGate",
        "CodeGen",
        "SemgrepScan",
        "IOAllowlistScan",
        "E2BExecute",
        "PrepareResponse",
        "CodegenExhausted",
    ):
        builder.add_node(name, noop)

    builder.set_entry_point("CodeGenLimitGate")
    builder.add_conditional_edges(
        "CodeGenLimitGate",
        route_codegen_limit_gate,
        {"CodeGen": "CodeGen", "CodegenExhausted": "CodegenExhausted"},
    )
    builder.add_edge("CodeGen", "SemgrepScan")
    builder.add_conditional_edges(
        "SemgrepScan",
        route_after_semgrep,
        {"IOAllowlistScan": "IOAllowlistScan", "CodeGenLimitGate": "CodeGenLimitGate"},
    )
    builder.add_conditional_edges(
        "IOAllowlistScan",
        route_after_io,
        {"E2BExecute": "E2BExecute", "CodeGenLimitGate": "CodeGenLimitGate"},
    )
    builder.add_conditional_edges(
        "E2BExecute",
        route_after_e2b,
        {"PrepareResponse": "PrepareResponse", "CodeGenLimitGate": "CodeGenLimitGate"},
    )
    builder.add_edge("CodegenExhausted", "PrepareResponse")
    builder.add_edge("PrepareResponse", END)
    return builder.compile()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_artifact(build_main_graph(), MAIN_PNG, MAIN_MERMAID)
    write_artifact(build_planner_graph(), PLANNER_PNG, PLANNER_MERMAID)
    write_artifact(build_coding_graph(), CODING_PNG, CODING_MERMAID)


if __name__ == "__main__":
    main()
