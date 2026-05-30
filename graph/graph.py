"""
LangGraph entrypoint for the data-analysis agent: prep, Planner sub-agent,
Orchestrator, tools, todo check, checkpointing.

Prep path: **ProfileSavedData** → **Planner** → **Orchestrator** → …
(SummariseConversationalSummary disabled for now — was ProfileSavedData → Summarise → Planner.)

After **RunTools**: **ProfileSavedData_PostTools** → **Orchestrator**.
"""

from __future__ import annotations
# import asyncio  # used by SummariseConversationalSummary (disabled)
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Literal

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import propagate_attributes
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from psycopg import Connection
from psycopg.rows import dict_row
from typing_extensions import NotRequired, TypedDict

# from middleware.context_editing import truncate_and_summarize  # disabled with Summarise node
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
from prompts.graph_prompts import FINAL_ANSWER_PROMPT, SYSTEM_PROMPT
from session_paths import session_id_from_config
from tools.coding_tools.coding_tool import coding_tool
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
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

# KEEP_RECENT = int(cfg["middleware"]["context_editing"]["keep_recent_messages"])
# TOKEN_THRESHOLD = int(cfg["middleware"]["message_summarisation"]["token_threshold"])
_USE_NEON = bool((cfg.get("checkpointer") or {}).get("use_neon"))
GRAPH_RECURSION_LIMIT = int((cfg.get("graph") or {}).get("recursion_limit", 100))
GRAPH_MAX_CONCURRENCY = int((cfg.get("graph") or {}).get("max_concurrency", 2))
ORCHESTRATOR_MODEL = cfg["models"]["orchestrator"]


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
    ``data_profile``, ``todos`` (LangGraph merges channels on the parent thread).
    """

    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    todos: Annotated[list[TodoEntry], merge_todos]
    todo_gate_passed: NotRequired[bool]  # set by TodoGate each visit


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
    coding_tool,
    sarima_tool,
    prophet_tool,
    holt_winters_tool,
    update_todo,
]  # Main-graph tier only; planner tools live under sub_agents/planner_sub_agent/tools/

llm = make_llm(model=ORCHESTRATOR_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def profile_saved_data(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Delegate to ``profiling_data.profile_session_workspace`` and set ``data_profile``."""
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)
    with traced_span("ProfileSavedData", trace_context=trace_context, metadata={"session_id": session_id}) as span:
        rows: List[Any] = profile_session_workspace(session_id)
        if span is not None:
            span.update(output={"profile_entries": len(rows), "session_id": session_id})
        update: Dict[str, Any] = {"data_profile": rows}
        if "todos" not in state:
            update["todos"] = []
        return update


def profile_saved_data_post_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Re-profile workspace after ``RunTools`` so new CSV/XLSX from tools appear in ``data_profile``."""
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)
    with traced_span("ProfileSavedData_PostTools", trace_context=trace_context, metadata={"session_id": session_id}) as span:
        rows: List[Any] = profile_session_workspace(session_id)
        if span is not None:
            span.update(output={"profile_entries": len(rows), "session_id": session_id})
        return {"data_profile": rows}


def summarise_conversational_summary(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Evict old turns into ``message_summary`` when over token threshold. (DISABLED)"""
    del state, config
    return {}


def orchestrator(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Build context from state and invoke the tool-bound LLM."""
    messages = state["messages"]
    summary = state.get("message_summary", "")
    raw_todos = state.get("todos") or []
    data_profile_rows = state.get("data_profile", [])

    context = (
        "## File Rules\n"
        "Refer to every CSV/Excel by filename only (for example `sales.csv`) in messages and tool arguments. "
        "Do not write `agent_filesystem/`, session ids, or path prefixes.\n\n"
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
            response = llm_with_tools.invoke(orchestrator_messages, config=config)
            if gen is not None:
                update_llm_generation(gen, model=model, raw=response)
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if node_span is not None:
            node_span.update(output={"tool_calls_this_step": len(tool_calls), "had_summary_context": bool(summary)})

    return {"messages": [response]}


def final_answer(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
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
            response = llm.invoke(final_messages, config=config)
            if gen is not None:
                update_llm_generation(gen, model=model, raw=response)
        if node_span is not None:
            reply = response.content if isinstance(response.content, str) else str(response.content or "")
            node_span.update(output={"reply_preview": reply, "had_summary_context": bool(summary)})

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def todo_gate(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Set ``todo_gate_passed`` and a short status ``AIMessage`` for the orchestrator."""
    todos_raw = state.get("todos") or []
    todos: List[dict[str, Any]] = [t for t in todos_raw if isinstance(t, dict)]
    incomplete = [t for t in todos if t.get("status") != "completed"]
    session_id = session_id_from_config(config)
    trace_context = trace_context_from_runnable_config(config)

    if not todos:
        passed = True
        content = "No todos to work on."
    elif not incomplete:
        passed = True
        content = "All todos completed."
    else:
        passed = False
        content = "\n".join(
            f"Pending todo: {t.get('id', '?')} — {t.get('content', '')} ({t.get('status', '')})"
            for t in incomplete
        )

    with traced_span("TodoGate", trace_context=trace_context, metadata={"session_id": session_id}) as span:
        if span is not None:
            span.update(output={"todo_gate_passed": passed, "content": content})

    return {
        "todo_gate_passed": passed,
        "messages": [AIMessage(content=content)],
    }


def route_after_todo_gate(state: AgentState) -> str:
    return "FinalAnswer" if state.get("todo_gate_passed") else "Orchestrator"


def route_after_orchestrator(state: AgentState) -> str:
    """Route to ``RunTools`` when the last ``AIMessage`` has tool calls, else todo check."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return "TodoGate"


# ---------------------------------------------------------------------------
# AnalysisGraph
# ---------------------------------------------------------------------------


class AnalysisGraph:
    """
    Wrapper around the compiled LangGraph ``StateGraph``.

    Notes
        * **Tools** — ``coding_tool``, ``sarima_tool``, ``prophet_tool``, ``holt_winters_tool``, ``update_todo``.
        * **Prep** — **ProfileSavedData** → **Planner** → **Orchestrator** (summarise disabled).
        * **Streaming** — :meth:`stream_graph` / :meth:`stream_resume` yield ``stream_mode="updates"`` chunks.
    """

    def __init__(self) -> None:
        self._pg_conn: Connection | None = None
        if _USE_NEON:
            uri = os.environ.get("DATABASE_URL", "").strip()
            if not uri:
                raise ValueError("checkpointer.use_neon is true in config.yaml but DATABASE_URL is missing in the environment.")
            self._pg_conn = Connection.connect(uri, autocommit=True, row_factory=dict_row)
            self.checkpointer = PostgresSaver(self._pg_conn)
            self.checkpointer.setup()
            print("GaussianBlurr checkpointer: Postgres (Neon / DATABASE_URL)", flush=True)
        else:
            self.checkpointer = InMemorySaver()
            print("GaussianBlurr checkpointer: InMemorySaver", flush=True)

        builder = StateGraph(AgentState)
        tool_node = ToolNode(TOOLS)

        builder.add_node("ProfileSavedData", profile_saved_data)
        builder.add_node("ProfileSavedData_PostTools", profile_saved_data_post_tools)
        builder.add_node("Planner", get_planner_graph())
        builder.add_node("Orchestrator", orchestrator)
        builder.add_node("RunTools", tool_node)
        builder.add_node("TodoGate", todo_gate)
        builder.add_node("FinalAnswer", final_answer)

        builder.set_entry_point("ProfileSavedData")
        builder.add_edge("ProfileSavedData", "Planner")
        builder.add_edge("Planner", "Orchestrator")
        builder.add_conditional_edges(
            "Orchestrator",
            route_after_orchestrator,
            {"RunTools": "RunTools", "TodoGate": "TodoGate"},
        )
        builder.add_conditional_edges(
            "TodoGate",
            route_after_todo_gate,
            {"Orchestrator": "Orchestrator", "FinalAnswer": "FinalAnswer"},
        )
        builder.add_edge("RunTools", "ProfileSavedData_PostTools")
        builder.add_edge("ProfileSavedData_PostTools", "Orchestrator")
        builder.add_edge("FinalAnswer", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

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

    def stream_graph(self, session_id: str, user_query: str) -> Iterator[Dict[str, Any]]:
        """Yield graph progress as ``updates`` payloads."""
        invoke_input = {
            "messages": [HumanMessage(content=user_query, id=f"user_input-{uuid.uuid4().hex}")],
            "todos": [],
        }
        if not is_tracing_enabled():
            yield from self.graph.stream(invoke_input, config=self.thread_config(session_id), stream_mode="updates")
            return
        try:
            with tracing_root("stream_run", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                yield from self.graph.stream(invoke_input, config=config, stream_mode="updates")
        finally:
            flush_langfuse()

    def stream_resume(self, session_id: str, value: Any) -> Iterator[Dict[str, Any]]:
        """Stream after an ``interrupt``, using ``Command(resume=…)``."""
        if not is_tracing_enabled():
            yield from self.graph.stream(Command(resume=value), config=self.thread_config(session_id), stream_mode="updates")
            return
        try:
            with tracing_root("resume", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                yield from self.graph.stream(Command(resume=value), config=config, stream_mode="updates")
        finally:
            flush_langfuse()

    def run_graph(self, session_id: str, user_query: str) -> Dict[str, Any]:
        invoke_input = {
            "messages": [HumanMessage(content=user_query, id=f"user_input-{uuid.uuid4().hex}")],
            "todos": [],
        }
        if not is_tracing_enabled():
            return self.graph.invoke(invoke_input, config=self.thread_config(session_id))
        try:
            with tracing_root("run", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                with propagate_attributes(session_id=session_id):
                    return self.graph.invoke(invoke_input, config=config)
        finally:
            flush_langfuse()

    def resume(self, session_id: str, value: Any) -> Dict[str, Any]:
        """Resume after planner ``ask_user`` interrupt; same ``thread_id`` as ``run_graph``."""
        if not is_tracing_enabled():
            return self.graph.invoke(Command(resume=value), config=self.thread_config(session_id))
        try:
            with tracing_root("resume", metadata={"session_id": session_id}) as trace_context:
                config = self.thread_config(session_id, trace_context=trace_context)
                with propagate_attributes(session_id=session_id):
                    return self.graph.invoke(Command(resume=value), config=config)
        finally:
            flush_langfuse()

    def get_state(self, session_id: str) -> Any:
        """Return the current state snapshot for *session_id*."""
        return self.graph.get_state({"configurable": {"thread_id": session_id}})
