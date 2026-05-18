"""
LangGraph entrypoint for the data-analysis agent: parallel prep, skills, orchestrator, tools, checkpointing.

Flow on each turn (and after **RunTools**): **BeginTurn** → parallel **ProfileSavedData**,
**DescribePlots**, **SummariseConversationalSummary** → **IdentifySkills** → **Orchestrator** →
(**RunTools** | **FinalAnswer** → END). Tool execution loops back to **BeginTurn**.
Code execution goes through ``code_pipeline`` (optional preflight, codegen, Semgrep, judge, run).
"""

from __future__ import annotations
import asyncio
import json
import os
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Literal

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
    LANGFUSE_PARENT_OBS_METADATA_KEY,
    LANGFUSE_TRACE_ID_METADATA_KEY,
    extract_usage_details,
    get_langfuse_client,
    observation_parented_to_run,
    serialize_message,
    serialize_messages,
)
from prompts.graph_prompts import SYSTEM_PROMPT
from session_paths import session_id_from_config
from tools.human_in_loop.ask_user import ask_user
from tools.coding_tools.code_pipeline import code_pipeline
from tools.file_management_tools.profiling_data import profile_session_workspace
from tools.forecasting.prophet_tool import prophet_tool
from tools.forecasting.sarima_tool import sarima_tool
from tools.planning.write_scratchpad import write_scratchpad
from tools.planning.write_todos import write_todos
from skills.loader import LoadReasoningSkills, IdentifySkills as route_skills_llm

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
                               summarisation cycles (node ``SummariseConversationalSummary``).
        data_profile         — list of per-file profiling dicts from
                               ``profiling_data.profile_session_workspace`` (node ``ProfileSavedData``).
        skill_guidance       — concatenated ``approach.md`` text from ``LoadReasoningSkills`` (node ``IdentifySkills``).
        plot_descriptions    — reserved for node ``DescribePlots`` (e.g. plot captions); optional.
        todos                — session task list maintained via ``write_todos`` (full replace each call).
        scratchpad           — session notes; ``write_scratchpad`` sends ``[note]`` and ``operator.add`` concatenates lists.
        active_skills        — skill folder ids chosen on the latest ``IdentifySkills`` step.
    """
    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    skill_guidance: NotRequired[str]
    plot_descriptions: NotRequired[List[str]]
    todos: NotRequired[list[TodoEntry]]
    scratchpad: Annotated[list[str], add]
    active_skills: List[str]


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
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


def begin_turn(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Fan-out anchor for parallel prep; trace only."""
    with observation_parented_to_run(langfuse, config, name="graph.BeginTurn", as_type="span") as obs:
        obs.update(metadata={})
    return {}


def profile_saved_data(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Delegate to ``profiling_data.profile_session_workspace`` and set state key ``data_profile``.
    """
    session_id = session_id_from_config(config)
    with observation_parented_to_run(langfuse, config, name="graph.ProfileSavedData", as_type="span") as obs:
        rows: List[Any] = profile_session_workspace(session_id)
        obs.update(metadata={"profile_entries": len(rows), "session_id": session_id})
        return {"data_profile": rows}


def describe_plots(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Stub for future plot / artifact descriptions into state."""
    with observation_parented_to_run(langfuse, config, name="graph.DescribePlots", as_type="span") as obs:
        obs.update(metadata={})
    return {}


def summarise_conversational_summary(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Evict old turns into ``message_summary`` when over token threshold (async summarisation in a fresh event loop)."""

    def _run_truncation() -> tuple[str, list, list]:
        return asyncio.run(
            truncate_and_summarize(
                state["messages"],
                state.get("message_summary", ""),
                KEEP_RECENT,
                TOKEN_THRESHOLD,
                runnable_config=config,
            )
        )

    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.SummariseConversationalSummary",
        as_type="chain",
        capture_input=False,
        capture_output=False,
    ):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            summary, _kept, remove_ops = _run_truncation()
        else:
            raise RuntimeError(
                "SummariseConversationalSummary invoked under a running event loop; "
                "use asynchronous graph execution (ainvoke) for this stack."
            )
        return {"message_summary": summary, "messages": remove_ops}


def identify_skills_step(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Route to skill folders and load reasoning text for orchestrator context."""
    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.IdentifySkills",
        capture_input=False,
        capture_output=False,
        as_type="chain",
    ):
        messages = state["messages"]
        data_profile_rows = state.get("data_profile") or []
        active_skills = route_skills_llm(messages, data_profile_rows)
        skill_guidance = LoadReasoningSkills(active_skills)
        return {"active_skills": active_skills, "skill_guidance": skill_guidance}


def orchestrator(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Core agent node: builds context from state and invokes the tool-bound LLM.
    """
    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.Orchestrator",
        capture_input=False,
        capture_output=False,
        as_type="chain",
    ):
        messages = state["messages"]
        summary = state.get("message_summary", "")
        raw_todos = state.get("todos") or []
        raw_pad = state.get("scratchpad") or []
        data_profile_rows = state.get("data_profile", [])
        skill_guidance = state.get("skill_guidance") or ""

        context = (
            "## File Rules\n"
            "Refer to every CSV/Excel by filename only (for example `sales.csv`) in messages and tool arguments. "
            "Do not write `agent_filesystem/`, session ids, or path prefixes.\n\n"
            "## Session Workspace\n"
            f"{json.dumps(data_profile_rows, indent=2, ensure_ascii=False, default=str)}\n\n"
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
        with observation_parented_to_run(
            langfuse,
            config,
            name="graph.Orchestrator.llm",
            as_type="generation",
            model=cfg["models"]["orchestrator"],
            input=serialize_messages(orchestrator_messages),
        ) as generation:
            response = llm_with_tools.invoke(orchestrator_messages, config=config)
            generation.update(
                output=serialize_message(response),
                usage_details=extract_usage_details(response),
                metadata={"tool_calls_requested": len(getattr(response, "tool_calls", None) or [])},
            )

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        ask_user_calls = [tool_call for tool_call in tool_calls if tool_call.get("name") == "ask_user"]
        ask_user_batch_rejected = bool(ask_user_calls and len(tool_calls) > 1)
        if ask_user_batch_rejected:
            response = AIMessage(
                content=(
                    "Invalid tool batch: `ask_user` must be the only tool call in a step. "
                    "Reissue either a single `ask_user` call or a tool batch that does not include `ask_user`."
                ),
            )
            tool_calls = []
        langfuse.update_current_span(metadata={"ask_user_batch_rejected": "true" if ask_user_batch_rejected else "false", "ask_user_batch_trimmed": "false"})
        langfuse.update_current_span(metadata={"tool_calls_this_step": len(tool_calls), "had_summary_context": "true" if bool(summary) else "false"})

        return {"messages": [response]}


def final_answer(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Terminal hook after a non-tool assistant reply."""
    with observation_parented_to_run(langfuse, config, name="graph.FinalAnswer", as_type="span") as obs:
        obs.update(metadata={})
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_orchestrator(state: AgentState) -> str:
    """Route to ``RunTools`` when the last ``AIMessage`` has tool calls, else ``FinalAnswer``."""
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
        * **Orchestrator model** — ``models.orchestrator`` from ``config.yaml``.
        * **Tools** — ``code_pipeline``, ask_user, ``write_scratchpad``, ``write_todos``, forecasting tools.
        * **Prep** — ``ProfileSavedData``, ``DescribePlots``, and ``SummariseConversationalSummary`` run in parallel, then ``IdentifySkills``, then ``Orchestrator``.
        * **code_pipeline** — LLM codegen, Semgrep, judge, save ``pipeline_run.py`` under ``agent_filesystem/<session>/``, then sandbox runner.
        * **Streaming** — :meth:`stream_graph` / :meth:`stream_resume` yield LangGraph ``stream_mode="updates"`` chunks (one dict per finished node batch). After the iterator exits, read the checkpoint snapshot and merge ``__interrupt__`` when Human-in-the-loop pauses mid-turn (same semantics as terminal ``invoke``).
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

    def build_graph(self) -> Any:
        """Construct and compile the ``StateGraph``."""
        builder = StateGraph(AgentState)
        tool_node = ToolNode(TOOLS)

        builder.add_node("BeginTurn", begin_turn)
        builder.add_node("ProfileSavedData", profile_saved_data)
        builder.add_node("DescribePlots", describe_plots)
        builder.add_node("SummariseConversationalSummary", summarise_conversational_summary)
        builder.add_node("IdentifySkills", identify_skills_step)
        builder.add_node("Orchestrator", orchestrator)
        builder.add_node("RunTools", tool_node)
        builder.add_node("FinalAnswer", final_answer)

        builder.set_entry_point("BeginTurn")
        builder.add_edge("BeginTurn", "ProfileSavedData")
        builder.add_edge("BeginTurn", "DescribePlots")
        builder.add_edge("BeginTurn", "SummariseConversationalSummary")
        builder.add_edge("ProfileSavedData", "IdentifySkills")
        builder.add_edge("DescribePlots", "IdentifySkills")
        builder.add_edge("SummariseConversationalSummary", "IdentifySkills")
        builder.add_edge("IdentifySkills", "Orchestrator")
        builder.add_conditional_edges(
            "Orchestrator",
            route_after_orchestrator,
            {"RunTools": "RunTools", "FinalAnswer": "FinalAnswer"},
        )
        builder.add_edge("RunTools", "BeginTurn")
        builder.add_edge("FinalAnswer", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _langfuse_invoke_metadata_pins(self) -> dict[str, str] | None:
        """Pins current trace/parent IDs from the active span (caller must run under ``@observe``)."""
        trace_id = langfuse.get_current_trace_id()
        obs_id = langfuse.get_current_observation_id()
        if not trace_id and not obs_id:
            return None
        pins: dict[str, str] = {}
        if trace_id:
            pins[LANGFUSE_TRACE_ID_METADATA_KEY] = str(trace_id)
        if obs_id:
            pins[LANGFUSE_PARENT_OBS_METADATA_KEY] = str(obs_id)
        return pins or None

    def _thread_config(self, session_id: str, *, langfuse_pin: dict[str, str] | None = None) -> Dict[str, Any]:
        """Runnable config aligned with ``run_graph`` / ``resume`` (thread + worker limits)."""
        out: Dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "max_concurrency": GRAPH_MAX_CONCURRENCY,
        }
        if langfuse_pin:
            merged = dict(out.get("metadata") or {})
            merged.update(langfuse_pin)
            out["metadata"] = merged
        return out

    def stream_graph(
        self,
        session_id: str,
        user_query: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield graph progress as ``updates`` payloads (typically ``{node_name: delta}``).

        Parallel prep nodes (``ProfileSavedData``, …) may arrive in nondeterministic order.
        Combine with checkpoint ``get_state`` after exhaustion to reconstruct an invoke-shaped
        dict including ``__interrupt__`` when Human-in-the-loop pauses mid-turn.
        """
        pins = self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        yield from self.graph.stream(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
            stream_mode="updates",
        )

    def stream_resume(
        self,
        session_id: str,
        value: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Same as :meth:`stream_graph` after an ``interrupt``, using ``Command(resume=…)``."""
        pins = self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        yield from self.graph.stream(Command(resume=value), config=config, stream_mode="updates")

    @observe(name="graph.run_graph", as_type="chain", capture_input=False, capture_output=False)
    def run_graph(
        self,
        session_id: str,
        user_query: str,
    ) -> Dict[str, Any]:
        pins = self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        result = self.graph.invoke({"messages": [HumanMessage(content=user_query)]}, config=config)
        langfuse.update_current_span(input={"session_id": session_id, "user_query": user_query}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    @observe(name="graph.resume", as_type="chain", capture_input=False, capture_output=False)
    def resume(
        self,
        session_id: str,
        value: Any,
    ) -> Dict[str, Any]:
        pins = self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        result = self.graph.invoke(Command(resume=value), config=config)
        langfuse.update_current_span(input={"session_id": session_id, "resume_value": value}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    def get_state(self, session_id: str) -> Any:
        """Return the current state snapshot for *session_id*."""
        return self.graph.get_state({"configurable": {"thread_id": session_id}})
