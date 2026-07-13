"""
LangGraph entrypoint for the data-analysis agent: prep, Planner sub-agent,
Orchestrator, tools, final synthesis, checkpointing.

Prep path: **ProfileSavedData** → **SummariseConversationalSummary** → **IsPlanningRequired** → **Planner** (if plan) or **Orchestrator** (if skip) → …

After **RunTools**: **ProfileSavedData_PostTools** → **Orchestrator**.
"""

from __future__ import annotations
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, Literal

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import propagate_attributes
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from typing_extensions import TypedDict

from middleware.context_editing import truncate_and_summarize
from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    add_trace_context_to_config,
    flush_langfuse,
    is_tracing_enabled,
    serialize_messages,
    trace_context_from_runnable_config,
    traced_generation,
    traced_span,
    tracing_root,
    update_llm_generation,
)
from output_validation.planning_gate import PlanningGateOutput
from prompts.graph_prompts import FINAL_ANSWER_PROMPT, PLANNING_GATE_SYSTEM_PROMPT, SYSTEM_PROMPT
from session_paths import session_id_from_config
from tools.coding_tools.coding_tool import coding_tool
from tools.file_management_tools.read_file_tool import read_file_tool
from sub_agents.planner_sub_agent.graph import get_planner_graph
from tools.file_management_tools.profiling_data import profile_session_workspace
from tools.forecasting.holt_winters_tool import holt_winters_tool
from tools.forecasting.prophet_tool import prophet_tool
from tools.forecasting.sarima_tool import sarima_tool
from tools.planning.update_todo import merge_todos, update_todo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env flag with safe fallback."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Parse integer env value; fallback when missing/invalid."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    """Read non-empty env value; fallback when unset/blank."""
    value = (os.environ.get(name) or "").strip()
    return value or default


_USE_NEON = _env_bool("MAIN_CHECKPOINTER_USE_NEON", default=False)
GRAPH_RECURSION_LIMIT = _env_int("MAIN_GRAPH_RECURSION_LIMIT", default=100)
GRAPH_MAX_CONCURRENCY = _env_int("MAIN_GRAPH_MAX_CONCURRENCY", default=2)
ORCHESTRATOR_MODEL = _env_str("MAIN_MODEL_ORCHESTRATOR", default="gpt-5.4-mini")
PLANNING_GATE_MODEL = _env_str("MAIN_MODEL_PLANNING_GATE", default="gpt-4o-mini")
CONTEXT_KEEP_RECENT_HUMAN_MESSAGES = _env_int("MAIN_CONTEXT_KEEP_RECENT_HUMAN_MESSAGES", default=10)
CONTEXT_SUMMARY_TOKEN_THRESHOLD = _env_int("MAIN_SUMMARY_TOKEN_THRESHOLD", default=100000)


class TodoEntry(TypedDict):
    """Single item in ``AgentState["todos"]``."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Graph state consumed by every node.

    Keys
        messages        — full conversation (HumanMessage, AIMessage, ToolMessage).
        message_summary — running summary of evicted messages.
        data_profile    — per-file profiling dicts from ``ProfileSavedData``.
        todos           — session task list; Planner replaces on each new user turn.

    Shared with ``PlannerAgentState`` for mounted subgraph: ``messages``,
    ``message_summary``, ``data_profile``, ``todos`` (LangGraph merges channels on the parent thread).
    """

    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    todos: Annotated[list[TodoEntry], merge_todos]


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
    coding_tool,
    read_file_tool,
    sarima_tool,
    prophet_tool,
    holt_winters_tool,
    update_todo,
]  # Main-graph tier only; planner tools live under sub_agents/planner_sub_agent/tools/

llm = make_llm(model=ORCHESTRATOR_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
planning_gate_llm = make_llm(
    model=PLANNING_GATE_MODEL,
    temperature=0,
    output_schema=PlanningGateOutput,
    include_raw=True,
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def profile_saved_data(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Delegate to ``profiling_data.profile_session_workspace`` and set ``data_profile``."""
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)
    with traced_span("ProfileSavedData", trace_context=trace_context, metadata={"session_id": session_id}) as span:
        rows: List[Any] = await asyncio.to_thread(profile_session_workspace, session_id)
        if span is not None:
            span.update(output={"profile_entries": len(rows), "session_id": session_id})
        update: Dict[str, Any] = {"data_profile": rows}
        if "todos" not in state:
            update["todos"] = []
        return update


async def profile_saved_data_post_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Re-profile workspace after ``RunTools`` so new CSV/XLSX from tools appear in ``data_profile``."""
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)
    with traced_span("ProfileSavedData_PostTools", trace_context=trace_context, metadata={"session_id": session_id}) as span:
        rows: List[Any] = await asyncio.to_thread(profile_session_workspace, session_id)
        if span is not None:
            span.update(output={"profile_entries": len(rows), "session_id": session_id})
        return {"data_profile": rows}


async def summarise_conversational_summary(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Evict old turns into ``message_summary`` when over token threshold, keeping the last N human turns."""
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)
    messages = state["messages"]
    previous_summary = state.get("message_summary", "")

    with traced_span("SummariseConversationalSummary", trace_context=trace_context, metadata={"session_id": session_id}) as span:
        updated_summary, _kept_messages, remove_ops = await truncate_and_summarize(
            messages,
            previous_summary,
            CONTEXT_KEEP_RECENT_HUMAN_MESSAGES,
            CONTEXT_SUMMARY_TOKEN_THRESHOLD,
        )
        if span is not None:
            span.update(output={"evicted_count": len(remove_ops), "truncated": bool(remove_ops)})

    if not remove_ops:
        return {}
    return {"messages": remove_ops, "message_summary": updated_summary}


async def is_planning_required(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classify whether Planner is needed; on skip, write a single todo from the latest user message."""
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)
    messages = state.get("messages") or []
    summary = state.get("message_summary", "")
    data_profile_rows = state.get("data_profile") or []

    latest_human_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                latest_human_text = content.strip()
                break

    context = (
        "## Session Workspace\n"
        f"{json.dumps(data_profile_rows, indent=2, ensure_ascii=False, default=str)}\n\n"
        "## Conversation Summary\n"
        f"{summary}\n\n"
        "## Latest user message\n"
        f"{latest_human_text}\n"
    )
    gate_messages = [
        SystemMessage(content=PLANNING_GATE_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]
    model = PLANNING_GATE_MODEL

    with traced_span("IsPlanningRequired", trace_context=trace_context, metadata={"session_id": session_id}) as node_span:
        with traced_generation("IsPlanningRequired-llm", model=model) as gen:
            result = await planning_gate_llm.ainvoke(gate_messages, config=config)
            parsed = result["parsed"] if isinstance(result, dict) else result
            raw = result.get("raw") if isinstance(result, dict) else None
            if gen is not None:
                update_llm_generation(gen, model=model, raw=raw)
        decision = parsed.decision if parsed is not None else "plan"
        reason = parsed.reason if parsed is not None else ""
        if node_span is not None:
            node_span.update(output={"decision": decision, "reason": reason})

    if decision == "skip" and latest_human_text:
        return {
            "todos": [{"id": "1", "content": latest_human_text, "status": "pending"}],
        }
    return {}


def route_after_planning_gate(state: AgentState) -> str:
    """Route to Orchestrator when skip path wrote todos; else Planner."""
    if state.get("todos"):
        return "Orchestrator"
    return "Planner"


async def orchestrator(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Build context from state and invoke the tool-bound LLM."""
    messages = state["messages"]
    summary = state.get("message_summary", "")
    raw_todos = state.get("todos") or []
    data_profile_rows = state.get("data_profile", [])

    context = (
        "## File Rules\n"
        "Use only `file` basenames from data_profile in todos and reasoning. "
        "Use column names from column_profiles[].name. "
        "Never write folder paths or session prefixes.\n\n"
        "## Session Workspace\n"
        f"{json.dumps(data_profile_rows, indent=2, ensure_ascii=False, default=str)}\n\n"
        "## Current Todo List\n"
        f"{json.dumps(raw_todos, indent=2)}\n\n"
        "## Conversation Summary\n"
        f"{summary}\n\n"
    )
    orchestrator_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)] + messages
    model = ORCHESTRATOR_MODEL
    trace_context = trace_context_from_runnable_config(config)

    with traced_span("Orchestrator", trace_context=trace_context) as node_span:
        with traced_generation("Orchestrator-llm", model=model) as gen:
            response = await llm_with_tools.ainvoke(orchestrator_messages, config=config)
            if gen is not None:
                update_llm_generation(gen, model=model, raw=response)
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if node_span is not None:
            node_span.update(output={"tool_calls_this_step": len(tool_calls), "had_summary_context": bool(summary)})

    return {"messages": [response]}


async def final_answer(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Synthesize the user-facing reply from session context and this turn's messages."""
    messages = state["messages"]
    summary = state.get("message_summary", "")
    raw_todos = state.get("todos") or []
    data_profile_rows = state.get("data_profile", [])

    context = (
        "## File Rules\n"
        "Refer to every CSV/Excel by filename only (for example `sales.csv`). "
        "Do not write `agent_filesystem/`, session ids, or path prefixes.\n\n"
        "## Session Workspace\n"
        f"{json.dumps(data_profile_rows, indent=2, ensure_ascii=False, default=str)}\n\n"
        "## Current Todo List\n"
        f"{json.dumps(raw_todos, indent=2)}\n\n"
        "## Conversation Summary\n"
        f"{summary}\n\n"
        "## Messages\n"
        f"{json.dumps(serialize_messages(messages), indent=2, ensure_ascii=False, default=str)}\n\n"
        "## Task\n"
        "Write the final reply to the user's latest request using the conversation above.\n"
    )
    final_messages = [SystemMessage(content=FINAL_ANSWER_PROMPT), HumanMessage(content=context)]
    model = ORCHESTRATOR_MODEL
    trace_context = trace_context_from_runnable_config(config)

    with traced_span("FinalAnswer", trace_context=trace_context, metadata={"session_id": session_id_from_config(config)}) as node_span:
        with traced_generation("FinalAnswer-llm", model=model) as gen:
            response = await llm.ainvoke(final_messages, config=config)
            if gen is not None:
                update_llm_generation(gen, model=model, raw=response)
        if node_span is not None:
            reply = response.content if isinstance(response.content, str) else str(response.content or "")
            node_span.update(output={"reply_preview": reply, "had_summary_context": bool(summary)})

    return {"messages": [response]}


def route_after_orchestrator(state: AgentState) -> str:
    """Route to ``RunTools`` when the last ``AIMessage`` has tool calls, else final answer."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return "FinalAnswer"


# ---------------------------------------------------------------------------
# AnalysisGraph
# ---------------------------------------------------------------------------


class AnalysisGraph:
    """
    Wrapper around the compiled LangGraph ``StateGraph``.

    Notes
        * **Tools** — ``coding_tool``, ``read_file_tool``, ``sarima_tool``, ``prophet_tool``, ``holt_winters_tool``, ``update_todo``.
        * **Prep** — **ProfileSavedData** → **SummariseConversationalSummary** → **IsPlanningRequired** → **Planner** or **Orchestrator**.
        * **Streaming** — :meth:`stream_graph` / :meth:`stream_resume` yield ``stream_mode="updates"`` chunks.
        * **Construction** — plain ``AnalysisGraph()`` only supports the in-memory
          checkpointer. When ``MAIN_CHECKPOINTER_USE_NEON=true``, build via the async
          factory ``await AnalysisGraph.acreate()`` instead, since the Postgres
          checkpointer's connection + ``setup()`` are async-only (``AsyncPostgresSaver``).
    """

    def __init__(self, checkpointer: Any | None = None, pg_conn: AsyncConnection | None = None) -> None:
        self._pg_conn = pg_conn
        if checkpointer is None:
            if _USE_NEON:
                raise RuntimeError(
                    "MAIN_CHECKPOINTER_USE_NEON is true — construct via "
                    "'await AnalysisGraph.acreate()' instead of 'AnalysisGraph()' "
                    "(Postgres checkpointer setup is async-only)."
                )
            checkpointer = InMemorySaver()
            print("GaussianBlurr checkpointer: InMemorySaver", flush=True)
        self.checkpointer = checkpointer

        builder = StateGraph(AgentState)
        tool_node = ToolNode(TOOLS)

        builder.add_node("ProfileSavedData", profile_saved_data)
        builder.add_node("ProfileSavedData_PostTools", profile_saved_data_post_tools)
        builder.add_node("SummariseConversationalSummary", summarise_conversational_summary)
        builder.add_node("IsPlanningRequired", is_planning_required)
        builder.add_node("Planner", get_planner_graph())
        builder.add_node("Orchestrator", orchestrator)
        builder.add_node("RunTools", tool_node)
        builder.add_node("FinalAnswer", final_answer)

        builder.set_entry_point("ProfileSavedData")
        builder.add_edge("ProfileSavedData", "SummariseConversationalSummary")
        builder.add_edge("SummariseConversationalSummary", "IsPlanningRequired")
        builder.add_conditional_edges(
            "IsPlanningRequired",
            route_after_planning_gate,
            {"Planner": "Planner", "Orchestrator": "Orchestrator"},
        )
        builder.add_edge("Planner", "Orchestrator")
        builder.add_conditional_edges(
            "Orchestrator",
            route_after_orchestrator,
            {"RunTools": "RunTools", "FinalAnswer": "FinalAnswer"},
        )
        builder.add_edge("RunTools", "ProfileSavedData_PostTools")
        builder.add_edge("ProfileSavedData_PostTools", "Orchestrator")
        builder.add_edge("FinalAnswer", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

    @classmethod
    async def acreate(cls) -> "AnalysisGraph":
        """
        Async factory — the only supported construction path when
        ``MAIN_CHECKPOINTER_USE_NEON=true``, since ``AsyncPostgresSaver`` requires an
        awaited connection and an awaited ``setup()`` call. Falls back to the plain
        (sync-safe) in-memory constructor otherwise.
        """
        if not _USE_NEON:
            return cls()

        uri = os.environ.get("DATABASE_URL", "").strip()
        if not uri:
            raise ValueError("MAIN_CHECKPOINTER_USE_NEON is true but DATABASE_URL is missing in the environment.")
        pg_conn = await AsyncConnection.connect(uri, autocommit=True, row_factory=dict_row)
        checkpointer = AsyncPostgresSaver(pg_conn)
        await checkpointer.setup()
        print("GaussianBlurr checkpointer: Postgres (Neon / DATABASE_URL)", flush=True)
        return cls(checkpointer, pg_conn)

    def build_graph(self) -> Any:
        """Return the compiled main graph."""
        return self.graph

    def thread_config(self, session_id: str, trace_context: Any | None = None) -> Dict[str, Any]:
        """Runnable config aligned with ``run_graph`` / ``resume``."""
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "max_concurrency": GRAPH_MAX_CONCURRENCY,
        }
        return add_trace_context_to_config(config, trace_context)

    async def stream_graph(self, session_id: str, user_query: str) -> AsyncIterator[Dict[str, Any]]:
        """Yield graph progress as ``updates`` payloads."""
        invoke_input = {
            "messages": [HumanMessage(content=user_query, id=f"user_input-{uuid.uuid4().hex}")],
            "todos": [],
        }
        if not is_tracing_enabled():
            async for update in self.graph.astream(invoke_input, config=self.thread_config(session_id), stream_mode="updates"):
                yield update
            return
        try:
            with tracing_root("stream_run", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                async for update in self.graph.astream(invoke_input, config=config, stream_mode="updates"):
                    yield update
        finally:
            flush_langfuse()

    async def stream_resume(self, session_id: str, value: Any) -> AsyncIterator[Dict[str, Any]]:
        """Stream after an ``interrupt``, using ``Command(resume=…)``."""
        if not is_tracing_enabled():
            async for update in self.graph.astream(Command(resume=value), config=self.thread_config(session_id), stream_mode="updates"):
                yield update
            return
        try:
            with tracing_root("resume", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                async for update in self.graph.astream(Command(resume=value), config=config, stream_mode="updates"):
                    yield update
        finally:
            flush_langfuse()

    async def run_graph(self, session_id: str, user_query: str) -> Dict[str, Any]:
        invoke_input = {
            "messages": [HumanMessage(content=user_query, id=f"user_input-{uuid.uuid4().hex}")],
            "todos": [],
        }
        if not is_tracing_enabled():
            return await self.graph.ainvoke(invoke_input, config=self.thread_config(session_id))
        try:
            with tracing_root("run", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                with propagate_attributes(session_id=session_id):
                    return await self.graph.ainvoke(invoke_input, config=config)
        finally:
            flush_langfuse()

    async def resume(self, session_id: str, value: Any) -> Dict[str, Any]:
        """Resume after planner ``ask_user`` interrupt; same ``thread_id`` as ``run_graph``."""
        if not is_tracing_enabled():
            return await self.graph.ainvoke(Command(resume=value), config=self.thread_config(session_id))
        try:
            with tracing_root("resume", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                with propagate_attributes(session_id=session_id):
                    return await self.graph.ainvoke(Command(resume=value), config=config)
        finally:
            flush_langfuse()

    async def get_state(self, session_id: str) -> Any:
        """Return the current state snapshot for *session_id*."""
        return await self.graph.aget_state({"configurable": {"thread_id": session_id}})
