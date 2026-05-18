"""
LangGraph entrypoint for the data-analysis agent: serial prep, skill nodes, Planner,
Orchestrator, tools, todo gate, checkpointing.

Prep path: **BeginTurn** → **ProfileSavedData** → **SummariseConversationalSummary**
→ **SelectPlannerSkills** → **Planner** → **SelectOrchestratorSkills** → **Orchestrator**
→ (**RunTools** | **TodoCompletionGate** → … | **FinalAnswer** → END).

After **RunTools**: **ProfileSavedData_PostTools** → **Orchestrator**
(resume with existing ``skill_guidance`` / ``active_skills`` from prep).
"""

from __future__ import annotations
import asyncio
import json
import os
import uuid
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
from langgraph.types import Command, interrupt
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
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from session_paths import session_id_from_config
from tools.coding_tools.code_pipeline import code_pipeline
from tools.file_management_tools.profiling_data import profile_session_workspace
from tools.forecasting.prophet_tool import prophet_tool
from tools.forecasting.sarima_tool import sarima_tool
from output_validation.planner_output import (
    NeedsPlanningClarification,
    PlanReady,
    PlannerStructuredResponse,
)
from tools.planning.update_todo import update_todo
from skills.loader import (
    LoadPlannerSkills,
    LoadReasoningSkills,
    invoke_orchestrator_skill_pick,
    invoke_planner_skill_pick,
)

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

_m = cfg["models"]
PLANNER_MODEL = _m.get("planner") or _m["orchestrator"]


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
        messages             — full conversation (HumanMessage, AIMessage, ToolMessage).
                               Uses ``add_messages`` reducer: append, deduplicate by id,
                               honour ``RemoveMessage`` for truncation. ``/run`` user text uses HumanMessage ids ``user_input-{uuid}``.
        message_summary      — running summary of evicted messages, grows across
                               summarisation cycles (node ``SummariseConversationalSummary``).
        data_profile         — list of per-file profiling dicts from
                               ``profiling_data.profile_session_workspace`` (node ``ProfileSavedData``).
        skill_guidance       — concatenated ``approach.md`` from orchestrator skills (node ``SelectOrchestratorSkills``).
        planner_skill_guidance — concatenated planner ``approach.md`` text (node ``SelectPlannerSkills``).
        todos                — session task list; **Planner** replaces on each new user turn; **update_todo** patches.
        active_skills        — orchestrator skill ids from ``SelectOrchestratorSkills``.
        active_planner_skills — planner skill ids from ``SelectPlannerSkills``.
    """
    messages: Annotated[list, add_messages]
    message_summary: str
    data_profile: List[Any]
    skill_guidance: NotRequired[str]
    planner_skill_guidance: NotRequired[str]
    todos: NotRequired[list[TodoEntry]]
    active_skills: List[str]
    active_planner_skills: List[str]


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
    code_pipeline,
    sarima_tool,
    prophet_tool,
    update_todo,
]

llm = make_llm(model=cfg["models"]["orchestrator"], temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
planner_llm = make_llm(
    model=PLANNER_MODEL,
    temperature=0,
    output_schema=PlannerStructuredResponse,
)
langfuse = get_langfuse_client()

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def begin_turn(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Turn entry hook; trace only."""
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


def profile_saved_data_post_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Re-profile workspace after ``RunTools`` (same handler as ``ProfileSavedData``)."""
    session_id = session_id_from_config(config)
    with observation_parented_to_run(
        langfuse, config, name="graph.ProfileSavedData_PostTools", as_type="span"
    ) as obs:
        rows: List[Any] = profile_session_workspace(session_id)
        obs.update(metadata={"profile_entries": len(rows), "session_id": session_id})
        return {"data_profile": rows}


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


def select_planner_skills_step(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """LLM picker + ``approach.md`` for planner skill ids."""
    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.SelectPlannerSkills",
        as_type="chain",
    ):
        messages = state["messages"]
        data_profile_rows = state.get("data_profile") or []
        ids = invoke_planner_skill_pick(messages, data_profile_rows)
        planner_skill_guidance = LoadPlannerSkills(ids)
        return {"active_planner_skills": ids, "planner_skill_guidance": planner_skill_guidance}


def select_orchestrator_skills_step(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """LLM picker + ``approach.md`` for orchestrator skill ids."""
    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.SelectOrchestratorSkills",
        as_type="chain",
    ):
        messages = state["messages"]
        data_profile_rows = state.get("data_profile") or []
        ids = invoke_orchestrator_skill_pick(messages, data_profile_rows)
        skill_guidance = LoadReasoningSkills(ids)
        return {"active_skills": ids, "skill_guidance": skill_guidance}


def _ensure_unique_todo_ids(rows: List[dict[str, Any]]) -> List[dict[str, Any]]:
    seen: set[str] = set()
    out: List[dict[str, Any]] = []
    for i, row in enumerate(rows):
        tid = str(row.get("id", "")).strip() or f"todo-{i + 1}"
        base = tid
        n = 2
        while tid in seen:
            tid = f"{base}-{n}"
            n += 1
        seen.add(tid)
        out.append(
            {
                "id": tid,
                "content": row["content"],
                "status": row["status"],
            }
        )
    return out


def planner_step(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Structured plan or clarification interrupt; replaces ``todos`` on ``PlanReady``."""
    messages = state["messages"]
    summary = state.get("message_summary", "")
    data_profile_rows = state.get("data_profile", [])
    planner_guidance = state.get("planner_skill_guidance") or ""

    clarification_extra = ""

    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.Planner",
        as_type="chain",
    ):
        while True:
            human_payload = (
                "## Session workspace (data_profile)\n"
                f"{json.dumps(data_profile_rows, indent=2, ensure_ascii=False, default=str)}\n\n"
                "### Planner skill guidance\n"
                f"{planner_guidance or '(none)'}\n\n"
                "## Conversation summary\n"
                f"{summary or '(none)'}\n"
                f"{clarification_extra}\n"
                "## Instruction\n"
                "Produce `plan_ready` with todos **for this user turn** (new ids), or `needs_planning_clarification`."
            )
            planner_messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=human_payload),
                *messages,
            ]
            with observation_parented_to_run(
                langfuse,
                config,
                name="graph.Planner.llm",
                as_type="generation",
                model=PLANNER_MODEL,
                input=serialize_messages(planner_messages),
            ) as generation:
                step_raw = planner_llm.invoke(planner_messages, config=config)
                if isinstance(step_raw, PlannerStructuredResponse):
                    step_out: PlanReady | NeedsPlanningClarification = step_raw.to_step()
                elif isinstance(step_raw, dict):
                    step_out = PlannerStructuredResponse.model_validate(step_raw).to_step()
                elif isinstance(step_raw, NeedsPlanningClarification):
                    step_out = step_raw
                elif isinstance(step_raw, PlanReady):
                    step_out = step_raw
                else:
                    raise TypeError(f"Unexpected planner output: {type(step_raw)}")
                generation.update(
                    output=repr(step_out),
                    metadata={"planner_kind": getattr(step_out, "kind", type(step_out).__name__)},
                )

            if isinstance(step_out, NeedsPlanningClarification):
                ans = interrupt({"phase": "planner", "question": step_out.question})
                clarification_extra += (
                    "\n## Prior clarification\n"
                    f"Planner asked: {step_out.question}\n"
                    f"User answered: {ans}\n"
                )
                continue

            if isinstance(step_out, PlanReady):
                raw_rows = [t.model_dump() for t in step_out.todos]
                todos_out = _ensure_unique_todo_ids(raw_rows)
                langfuse.update_current_span(metadata={"planner_todo_count": len(todos_out)})
                return {"todos": todos_out}

            raise TypeError(f"Unexpected planner output type: {type(step_out)}")


def orchestrator(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Core agent node: builds context from state and invokes the tool-bound LLM.
    """
    with observation_parented_to_run(
        langfuse,
        config,
        name="graph.Orchestrator",
        as_type="chain",
    ):
        messages = state["messages"]
        summary = state.get("message_summary", "")
        raw_todos = state.get("todos") or []
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
            "## Conversation Summary\n"
            f"{summary}\n\n"
        )
        orchestrator_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)] + messages
        with observation_parented_to_run(langfuse, config, name="graph.Orchestrator.llm", as_type="generation", model=cfg["models"]["orchestrator"], input=serialize_messages(orchestrator_messages)) as generation:
            response = llm_with_tools.invoke(orchestrator_messages, config=config)
            generation.update(output=serialize_message(response), usage_details=extract_usage_details(response), metadata={"tool_calls_requested": len(getattr(response, "tool_calls", None) or [])})

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        langfuse.update_current_span(
            metadata={"tool_calls_this_step": len(tool_calls), "had_summary_context": "true" if bool(summary) else "false"}
        )

        return {"messages": [response]}


def final_answer(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Terminal hook after a non-tool assistant reply."""
    with observation_parented_to_run(langfuse, config, name="graph.FinalAnswer", as_type="span") as obs:
        obs.update(metadata={})
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _pending_todos_message(incomplete: List[dict[str, Any]]) -> str:
    """Deterministic workflow text: one line per item not yet completed."""
    lines: List[str] = [
        "## Workflow instruction",
        "",
    ]
    for t in incomplete:
        tid = t.get("id", "?")
        content = t.get("content", "")
        st = t.get("status", "")
        lines.append(f"Pending todo: `{tid}` — {content} (status: {st})")
    return "\n".join(lines)


def todo_completion_gate(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """If todos remain incomplete, inject a workflow ``AIMessage`` listing pending items only.

    Uses ``AIMessage`` so the user's last ``HumanMessage`` stays the turn boundary for API
    helpers (e.g. plot discovery in ``api.main._collect_current_turn_tool_call_ids``).
    """
    todos_raw = state.get("todos") or []
    todos: List[dict[str, Any]] = [t for t in todos_raw if isinstance(t, dict)]
    incomplete = [t for t in todos if t.get("status") != "completed"]
    if not incomplete:
        with observation_parented_to_run(langfuse, config, name="graph.TodoCompletionGate", as_type="span") as obs:
            obs.update(metadata={"action": "pass"})
        return {}

    content = _pending_todos_message(incomplete)
    with observation_parented_to_run(langfuse, config, name="graph.TodoCompletionGate", as_type="span") as obs:
        obs.update(
            output=content[:4000],
            metadata={"action": "block", "incomplete_todos": len(incomplete)},
        )
    return {"messages": [AIMessage(content=content)]}


def route_after_todo_gate(state: AgentState) -> str:
    todos_raw = state.get("todos") or []
    todos = [t for t in todos_raw if isinstance(t, dict)]
    if not todos:
        return "FinalAnswer"
    if all(t.get("status") == "completed" for t in todos):
        return "FinalAnswer"
    return "Orchestrator"


def route_after_orchestrator(state: AgentState) -> str:
    """Route to ``RunTools`` when the last ``AIMessage`` has tool calls, else todo gate."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "RunTools"
    return "TodoCompletionGate"


# ---------------------------------------------------------------------------
# AnalysisGraph
# ---------------------------------------------------------------------------


class AnalysisGraph:
    """
    Wrapper around the compiled LangGraph ``StateGraph``.

    Notes
        * **Models** — ``models.orchestrator`` (tool agent), ``models.planner`` (structured plan / clarification); ``code_generation`` / ``code_judge`` from ``config.yaml`` apply inside ``code_pipeline``. Incomplete todos inject a deterministic workflow message (no extra model).
        * **Tools** — ``code_pipeline``, ``sarima_tool``, ``prophet_tool``, ``update_todo``.
        * **Prep** — serial **ProfileSavedData** → **SummariseConversationalSummary** → **SelectPlannerSkills** → **Planner** → **SelectOrchestratorSkills** → **Orchestrator**. After **RunTools**: **ProfileSavedData_PostTools** → **Orchestrator** (skills unchanged from prep).
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
        builder.add_node("SummariseConversationalSummary", summarise_conversational_summary)
        builder.add_node("SelectPlannerSkills", select_planner_skills_step)
        builder.add_node("SelectOrchestratorSkills", select_orchestrator_skills_step)
        builder.add_node("ProfileSavedData_PostTools", profile_saved_data_post_tools)
        builder.add_node("Planner", planner_step)
        builder.add_node("Orchestrator", orchestrator)
        builder.add_node("RunTools", tool_node)
        builder.add_node("TodoCompletionGate", todo_completion_gate)
        builder.add_node("FinalAnswer", final_answer)

        builder.set_entry_point("BeginTurn")
        builder.add_edge("BeginTurn", "ProfileSavedData")
        builder.add_edge("ProfileSavedData", "SummariseConversationalSummary")
        builder.add_edge("SummariseConversationalSummary", "SelectPlannerSkills")
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
        builder.add_edge("ProfileSavedData_PostTools", "Orchestrator")
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
        *,
        langfuse_pin: dict[str, str] | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield graph progress as ``updates`` payloads (typically ``{node_name: delta}``).

        Prep runs serially (see ``build_graph``); stream frames follow that node order.
        Combine with checkpoint ``get_state`` after exhaustion to reconstruct an invoke-shaped
        dict including ``__interrupt__`` when Human-in-the-loop pauses mid-turn.

        For ``/run/stream``, pass *langfuse_pin* from :func:`sse_stream_runnable_langfuse_pin`
        so Langfuse has a trace id without an HTTP iterator context manager.
        """
        pins = langfuse_pin if langfuse_pin is not None else self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        yield from self.graph.stream(
            {"messages": [HumanMessage(content=user_query, id=f"user_input-{uuid.uuid4().hex}")]},
            config=config,
            stream_mode="updates",
        )

    def stream_resume(
        self,
        session_id: str,
        value: Any,
        *,
        langfuse_pin: dict[str, str] | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """Same as :meth:`stream_graph` after an ``interrupt``, using ``Command(resume=…)``."""
        pins = langfuse_pin if langfuse_pin is not None else self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        yield from self.graph.stream(Command(resume=value), config=config, stream_mode="updates")

    @observe(name="graph.run_graph", as_type="chain")
    def run_graph(
        self,
        session_id: str,
        user_query: str,
    ) -> Dict[str, Any]:
        pins = self._langfuse_invoke_metadata_pins()
        config = self._thread_config(session_id, langfuse_pin=pins)
        result = self.graph.invoke({"messages": [HumanMessage(content=user_query, id=f"user_input-{uuid.uuid4().hex}")]}, config=config)
        langfuse.update_current_span(input={"session_id": session_id, "user_query": user_query}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    @observe(name="graph.resume", as_type="chain")
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
