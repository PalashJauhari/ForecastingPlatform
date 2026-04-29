"""
LangGraph entrypoint for the data-analysis agent: orchestrator + tools + checkpointing.

Flow: the ``data_profile`` node runs **immediately before** ``orchestrator`` on every
orchestrator turn: once from **START**, and again after **tools** (``tools`` →
``data_profile`` → ``orchestrator``). Then ``orchestrator`` → (optional) ``tools`` loop.
Code execution goes through ``code_pipeline`` (codegen, Semgrep, judge, run).
"""

from __future__ import annotations
import json
import os
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal

from dotenv import load_dotenv

# Ensure OPENAI_API_KEY (and other vars) from project-root .env are loaded before
# module-level ChatOpenAI clients are constructed — even when graph is imported
# without going through api.main (e.g. scripts, tests).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from psycopg import Connection
from psycopg.rows import dict_row
from typing_extensions import NotRequired, TypedDict

from middleware.context_editing import truncate_and_summarize
from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    extract_usage_details,
    get_langfuse_client,
    serialize_message,
    serialize_messages,
)
from prompts.graph_prompts import SYSTEM_PROMPT
from session_paths import session_id_from_config
from tools.human_in_loop.ask_user import ask_user
from tools.coding_tools.build_codegen_requirement import build_codegen_requirement
from tools.coding_tools.code_pipeline import code_pipeline
from tools.file_management_tools.profiling_data import profile_session_workspace
from tools.forecasting.prophet_tool import prophet_tool
from tools.forecasting.sarima_tool import sarima_tool
from tools.planning.write_scratchpad import write_scratchpad
from tools.planning.write_todos import write_todos
from skills.loader import LoadReasoningSkills, IdentifySkills

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

KEEP_RECENT = int(cfg["middleware"]["context_editing"]["keep_recent_messages"])
TOKEN_THRESHOLD = int(cfg["middleware"]["message_summarisation"]["token_threshold"])
_USE_NEON = bool((cfg.get("checkpointer") or {}).get("use_neon"))
GRAPH_RECURSION_LIMIT = int((cfg.get("graph") or {}).get("recursion_limit", 100))
GRAPH_MAX_CONCURRENCY = int((cfg.get("graph") or {}).get("max_concurrency", 2))


class TodoEntry(TypedDict):
    """Single item in ``AgentState["todos"]`` (mirrors ``output_validation.write_todos``)."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Graph state consumed by every node.

    Keys
        messages             — full conversation (HumanMessage, AIMessage, ToolMessage).
                               Uses ``add_messages`` reducer: append, deduplicate by id,
                               honour ``RemoveMessage`` for truncation.
        message_summary      — running summary of evicted messages, grows across
                               summarisation cycles.
        data_profile         — list of per-file profiling dicts from
                               ``profiling_data.profile_session_workspace`` (graph node ``data_profile``).
                               Each entry uses ``file`` = basename only and includes
                               ``row_count``, ``column_count``, ``columns``, ``dtypes``,
                               ``null_counts``, ``head``, ``column_profiles``, and
                               ``numeric_summary``. Empty list when no CSV/XLSX;
                               refreshed before every orchestrator call.
        todos                — session task list maintained via ``write_todos`` (full replace each call).
        scratchpad           — session notes; ``write_scratchpad`` sends ``[note]`` and ``operator.add`` concatenates lists.
    """
    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    todos: NotRequired[list[TodoEntry]]
    scratchpad: Annotated[list[str], add]
    active_skills: List[str]


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
    build_codegen_requirement,
    code_pipeline,
    sarima_tool,
    prophet_tool,
    ask_user,
    write_scratchpad,
    write_todos,
]

llm = make_llm(model=cfg["models"]["orchestrator"], temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
langfuse = get_langfuse_client()

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def data_profile(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Delegate to ``profiling_data.profile_session_workspace``: read every top-level
    session CSV/XLSX file, build a rich in-memory profile, and set state key ``data_profile``.
    Does not modify ``todos`` or ``scratchpad``. Tracing uses ``start_as_current_observation``
    so metadata is applied via ``obs.update`` on the open span (avoids ``update_current_span``
    missing the observation when the OTEL current span does not match the node span).
    """
    session_id = session_id_from_config(config)
    with langfuse.start_as_current_observation(name="graph.data_profile", as_type="span") as obs:
        rows: List[Any] = profile_session_workspace(session_id)
        obs.update(metadata={"profile_entries": len(rows), "session_id": session_id})
        return {"data_profile": rows}


@observe(name="graph.orchestrator", capture_input=False, capture_output=False)
def orchestrator(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Core agent node.

    Steps executed in order:
        0. **Summarisation + truncation** — if token estimate exceeds the
           threshold, evict old messages into a running summary and
           produce ``RemoveMessage`` ops for the ``add_messages`` reducer.
        1. **Invoke LLM** — ``SYSTEM_PROMPT``, then a ``HumanMessage`` with session
           workspace profile list (``data_profile`` as JSON) + conversation summary,
           then prior ``messages``; model has tools bound (``llm_with_tools``).

    ``config`` provides ``thread_id`` so the HumanMessage can spell the correct
    ``agent_filesystem/<session-folder>/...`` prefix for this checkpoint.
    """
    messages = state["messages"]
    summary = state.get("message_summary", "")

    remove_ops: list = []

    # 1. Evict old turns into ``message_summary`` when estimated tokens exceed threshold.
    summary, messages, remove_ops = truncate_and_summarize(
        messages, summary, KEEP_RECENT, TOKEN_THRESHOLD,
    )

    raw_todos = state.get("todos") or []
    raw_pad = state.get("scratchpad") or []
    data_profile = state.get("data_profile", [])
    # Select skills after truncation so routing uses the same message window the orchestrator sees.
    active_skills = IdentifySkills(messages, data_profile)
    skill_guidance = LoadReasoningSkills(active_skills)

    # Keep the dynamic context in one synthetic human turn; prior conversation remains as raw messages below.
    context = (
        "## File Rules\n"
        "Refer to every CSV/Excel by filename only (for example `sales.csv`) in messages and tool arguments. "
        "Do not write `agent_filesystem/`, session ids, or path prefixes.\n\n"
        "## Session Workspace\n"
        f"{json.dumps(data_profile, indent=2, ensure_ascii=False, default=str)}\n\n"
        "### EXPERT GUIDANCE (Reasoning Skills):\n"
        f"{skill_guidance or '(none)'}\n\n"
        "## Current Todo List\n"
        f"{json.dumps(raw_todos, indent=2)}\n\n"
        "## Scratchpad\n"
        f"{json.dumps(raw_pad, ensure_ascii=False, indent=2)}\n\n"
        "## Conversation Summary\n"
        f"{summary}\n\n"
    )
    orchestrator_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)] + messages
    with langfuse.start_as_current_observation(name="graph.orchestrator.llm", as_type="generation", model=cfg["models"]["orchestrator"], input=serialize_messages(orchestrator_messages)) as generation:
        response = llm_with_tools.invoke(orchestrator_messages)
        generation.update(output=serialize_message(response), usage_details=extract_usage_details(response), metadata={"tool_calls_requested": len(response.tool_calls)})

    tool_calls = list(getattr(response, "tool_calls", None) or [])
    ask_user_calls = [tool_call for tool_call in tool_calls if tool_call.get("name") == "ask_user"]
    ask_user_batch_rejected = bool(ask_user_calls and len(tool_calls) > 1)
    if ask_user_batch_rejected:
        # ``ask_user`` pauses the graph, so mixed batches are rejected and must be regenerated.
        response = AIMessage(
            content=(
                "Invalid tool batch: `ask_user` must be the only tool call in a step. "
                "Reissue either a single `ask_user` call or a tool batch that does not include `ask_user`."
            ),
        )
        tool_calls = []
    langfuse.update_current_span(metadata={"ask_user_batch_rejected": "true" if ask_user_batch_rejected else "false", "ask_user_batch_trimmed": "false"})

    langfuse.update_current_span(metadata={"tool_calls_this_step": len(tool_calls), "had_summary_context": "true" if bool(summary) else "false"})

    return {
        "messages": remove_ops + [response],
        "message_summary": summary,
        "active_skills": active_skills,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> str:
    """Route to ``tools`` if the last ``AIMessage`` has tool calls, else ``END``."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# AnalysisGraph
# ---------------------------------------------------------------------------


class AnalysisGraph:
    """
    Wrapper around the compiled LangGraph ``StateGraph``.

    Notes
        * **Orchestrator model** — ``models.orchestrator`` from ``config.yaml``.
        * **Tools** — ``build_codegen_requirement``, ``code_pipeline``, ask_user, ``write_scratchpad``, ``write_todos`` (tabular profiles live in state ``data_profile``, refreshed by the ``data_profile`` graph node).
        * **Middleware logic** — context editing, summarisation, and per-session tool-call budget run inside the orchestrator node.
        * **code_pipeline** — LLM codegen, Semgrep, judge, save ``pipeline_run.py`` under ``agent_filesystem/<session>/``, then sandbox runner.
        * **Skills** — the ``skills/`` package and loader remain in the repo for future use; the graph does not load skill overlays into the orchestrator for now.
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
        self.graph = self.build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> Any:
        """Construct and compile the ``StateGraph``."""
        builder = StateGraph(AgentState)

        builder.add_node("data_profile", data_profile)
        builder.add_node("orchestrator", orchestrator)
        builder.add_node("tools", ToolNode(TOOLS))

        # ``data_profile`` node always runs immediately before orchestrator:
        #   START → data_profile → orchestrator
        #   tools → data_profile → orchestrator
        builder.set_entry_point("data_profile")
        builder.add_edge("data_profile", "orchestrator")
        builder.add_conditional_edges(
            "orchestrator", should_continue, {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "data_profile")

        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @observe(name="graph.run_graph", as_type="chain", capture_input=False, capture_output=False)
    def run_graph(
        self,
        session_id: str,
        user_query: str,
    ) -> Dict[str, Any]:
        """
        Invoke the agent graph with one user message.

        Parameters
            session_id  — ``configurable.thread_id`` (conversation key).
            user_query  — user text (may include appended upload paths).

        Returns
            Graph invoke result dict (``messages``, ``message_summary``, etc.).
        """
        # ``thread_id`` ties every turn for a session to the same checkpointed graph state.
        # ``recursion_limit`` / ``max_concurrency`` are top-level RunnableConfig keys (see ``config.yaml`` → ``graph``).
        config: Dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "max_concurrency": GRAPH_MAX_CONCURRENCY,
        }
        # On each turn we add only the new user message and let the checkpointer load prior state.
        result = self.graph.invoke({"messages": [HumanMessage(content=user_query)]}, config=config)
        langfuse.update_current_span(input={"session_id": session_id, "user_query": user_query}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    @observe(name="graph.resume", as_type="chain", capture_input=False, capture_output=False)
    def resume(
        self,
        session_id: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Resume a paused graph (after ``ask_user`` interrupt).

        Parameters
            session_id — same session that was interrupted.
            value      — the user's answer to the clarifying question.

        Returns
            Graph invoke result dict.
        """
        config: Dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "max_concurrency": GRAPH_MAX_CONCURRENCY,
        }
        result = self.graph.invoke(Command(resume=value), config=config)
        langfuse.update_current_span(input={"session_id": session_id, "resume_value": value}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    def get_state(self, session_id: str) -> Any:
        """Return the current state snapshot for *session_id*."""
        return self.graph.get_state({"configurable": {"thread_id": session_id}})
