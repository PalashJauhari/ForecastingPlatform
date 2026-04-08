"""
LangGraph entrypoint for the data-analysis agent: orchestrator + tools + checkpointing.

Flow: ``refresh_data_schema`` → ``identify_skills`` → ``reset_tool_budget`` → ``orchestrator`` → (optional) ``tools`` loop.
Code execution goes through ``code_pipeline`` (codegen, Semgrep, judge, run).
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Any, Dict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langfuse import observe
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from middleware.context_editing import truncate_and_summarize
from observability.langfuse_handler import (
    extract_usage_details,
    get_langfuse_client,
    serialize_message,
    serialize_messages,
)
from middleware.tool_call_limit import check_tool_call_limit
from output_validation.skill_selection import SkillSelection
from prompts.graph_prompts import SYSTEM_PROMPT
from prompts.skills_prompts import SKILL_IDENTIFICATION_PROMPT
from session_paths import session_id_from_config, session_root, to_agent_path
from skills.loader import PRIORITY_ORDER, load_skills
from tools.human_in_loop.ask_user import ask_user
from tools.coding_tools.build_codegen_requirement import build_codegen_requirement
from tools.coding_tools.code_pipeline import code_pipeline
from tools.file_management_tools.list_agent_filesystem_data import list_agent_filesystem_data
from tools.file_management_tools.profile_forecasting_data import profile_forecasting_data
from tools.file_management_tools.read_agent_filesystem_data import read_agent_filesystem_data
from tools.file_management_tools.read_scratchpad import read_scratchpad
from tools.file_management_tools.write_scratchpad import write_scratchpad

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

KEEP_RECENT = int(cfg["middleware"]["context_editing"]["keep_recent_messages"])
TOKEN_THRESHOLD = int(cfg["middleware"]["summarization"]["token_threshold"])
MAX_TOOL_CALLS = int(cfg["middleware"]["tool_call_limit"]["max_calls"])

_VALID_SKILL_IDS = frozenset(PRIORITY_ORDER)
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
        number_of_tool_calls — total individual tool invocations in the current
                               user turn; reset in ``reset_tool_budget`` when the
                               last message is a ``HumanMessage``.
        data_schema          — newline-separated listing of data files currently in
                               ``agent_filesystem/`` (excluding ``scratchpad/``).
                               Refreshed before every orchestrator call.
        latest_profile_result — latest structured result returned by
                               ``profile_forecasting_data``. Overwritten on each
                               new profiling call so planning tools can read a
                               normalized current profile from state.
        active_skills        — skill ids chosen by ``identify_skills`` on the latest user turn.
        skill_context        — assembled ``approach.md`` + reference snippets for those skills.
    """
    messages: Annotated[list, add_messages]
    message_summary: str
    number_of_tool_calls: int
    data_schema: str
    latest_profile_result: str
    active_skills: list[str]
    skill_context: str


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
    list_agent_filesystem_data,
    read_agent_filesystem_data,
    profile_forecasting_data,
    build_codegen_requirement,
    code_pipeline,
    ask_user,
    read_scratchpad,
    write_scratchpad,
]

llm = ChatOpenAI(model=cfg["models"]["orchestrator"], temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
langfuse = get_langfuse_client()

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@observe(name="graph.refresh_data_schema", capture_input=False, capture_output=False)
def refresh_data_schema(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Scan the current session workspace for ``.csv`` and ``.xlsx`` files and
    update ``state["data_schema"]`` with logical agent paths.
    """
    session_id = session_id_from_config(config)
    base = session_root(session_id)
    files = [
        to_agent_path(session_id, f)
        for f in sorted(base.rglob("*"))
        if f.is_file()
        and f.suffix.lower() in {".csv", ".xlsx"}
    ] if base.exists() else []

    output = {"data_schema": "\n".join(files) if files else "(no data files found)"}
    langfuse.update_current_span(metadata={"file_count": len(files), "session_id": session_id})
    return output


@observe(name="graph.identify_skills", capture_input=False, capture_output=False)
def identify_skills(state: AgentState) -> Dict[str, Any]:
    """
    One structured-output LLM call per **new user turn** (last message is ``HumanMessage``).
    Sets ``active_skills`` and ``skill_context``; on tool-loop steps returns ``{}`` (unchanged).
    """
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {}

    user_query = str(messages[-1].content)
    data_schema = state.get("data_schema", "")
    prompt = (
        f"User query: {user_query}\n\n"
        f"Available data files:\n{data_schema}\n\n"
        "Which skills are needed?"
    )
    llm_structured = ChatOpenAI(model=cfg["models"]["orchestrator"], temperature=0).with_structured_output(SkillSelection)
    generation_input = [
        serialize_message(SystemMessage(content=SKILL_IDENTIFICATION_PROMPT)),
        serialize_message(HumanMessage(content=prompt)),
    ]
    with langfuse.start_as_current_observation(name="graph.identify_skills.llm", as_type="generation", model=cfg["models"]["orchestrator"], input=generation_input) as generation:
        try:
            resp = llm_structured.invoke(
                [SystemMessage(content=SKILL_IDENTIFICATION_PROMPT), HumanMessage(content=prompt)],
            )
        except Exception as e:
            generation.update(output={"error": str(e)})
            langfuse.update_current_span(metadata={"selected_skills": "", "selected_skill_count": 0}, status_message="Skill identification returned invalid structured output.")
            return {"active_skills": [], "skill_context": ""}
        generation.update(output=resp.model_dump())

    if not isinstance(resp.skills, list):
        return {"active_skills": [], "skill_context": ""}

    skills = [str(s) for s in resp.skills if s in _VALID_SKILL_IDS]

    skill_context = load_skills(skills) if skills else ""
    langfuse.update_current_span(metadata={"selected_skills": ",".join(skills), "selected_skill_count": len(skills)})
    return {"active_skills": skills, "skill_context": skill_context}


@observe(name="graph.reset_tool_budget", capture_input=False, capture_output=False)
def reset_tool_budget(state: AgentState) -> Dict[str, Any]:
    """
    Zero ``number_of_tool_calls`` when the latest message is a new user turn.

    Runs after ``refresh_data_schema`` on every path into the orchestrator (initial
    invoke, post-tool loop). When the graph is mid-tool-loop the last message is not
    a ``HumanMessage``, so the counter is left unchanged.
    """
    messages = state["messages"]
    if messages and isinstance(messages[-1], HumanMessage):
        langfuse.update_current_span(metadata={"reset_applied": "true"})
        return {"number_of_tool_calls": 0}
    langfuse.update_current_span(metadata={"reset_applied": "false"})
    return {}


@observe(name="graph.orchestrator", capture_input=False, capture_output=False)
def orchestrator(state: AgentState) -> Dict[str, Any]:
    """
    Core agent node.

    Steps executed in order:
        1. **Tool call limit** — return early ``AIMessage`` if exceeded (budget set
           by ``reset_tool_budget`` on each new user message).
        2. **Summarisation + truncation** — if token estimate exceeds the
           threshold, evict old messages into a running summary and
           produce ``RemoveMessage`` ops for the ``add_messages`` reducer.
        3. **Invoke LLM** — ``SYSTEM_PROMPT``, then a ``HumanMessage`` with available
           files (``data_schema``) + conversation summary, then prior ``messages``;
           model has tools bound (``llm_with_tools``).
        4. **Count tool calls** — increment ``number_of_tool_calls`` by this response’s
           ``tool_calls`` length.
    """
    messages = state["messages"]
    summary = state.get("message_summary", "")

    tool_calls_so_far = state.get("number_of_tool_calls", 0)

    # 1. Hard cap on tool invocations per user turn (see ``reset_tool_budget``).
    limit_msg = check_tool_call_limit(tool_calls_so_far, MAX_TOOL_CALLS)
    if limit_msg:
        return {"messages": [limit_msg]}
    remove_ops: list = []

    # 2. Evict old turns into ``message_summary`` when estimated tokens exceed threshold.
    summary, messages, remove_ops = truncate_and_summarize(
        messages, summary, KEEP_RECENT, TOKEN_THRESHOLD,
    )

    # 3. System prompt + dynamic context (file list, summary, optional skill guidance) + messages.
    skill_ctx = state.get("skill_context", "")
    context = (
        f"Available data files:\n{state.get('data_schema', '')}\n\n"
        f"Conversation summary:\n{summary}\n\n"
        + (f"Skill guidance:\n{skill_ctx}" if skill_ctx else "")
    )
    orchestrator_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)] + messages
    with langfuse.start_as_current_observation(name="graph.orchestrator.llm", as_type="generation", model=cfg["models"]["orchestrator"], input=serialize_messages(orchestrator_messages)) as generation:
        response = llm_with_tools.invoke(orchestrator_messages)
        generation.update(output=serialize_message(response), usage_details=extract_usage_details(response), metadata={"tool_call_count": len(response.tool_calls)})

    tool_calls = list(getattr(response, "tool_calls", None) or [])
    ask_user_calls = [tool_call for tool_call in tool_calls if tool_call.get("name") == "ask_user"]
    if ask_user_calls and len(tool_calls) > 1:
        # ``ask_user`` pauses the graph, so we trim mixed batches down to one interrupt call.
        response = AIMessage(content="", tool_calls=[ask_user_calls[0]])
        tool_calls = [ask_user_calls[0]]
        langfuse.update_current_span(metadata={"ask_user_batch_trimmed": "true"})
    else:
        langfuse.update_current_span(metadata={"ask_user_batch_trimmed": "false"})

    # 4. Count tool calls in this orchestrator step (each tool_calls entry counts once).
    new_count = tool_calls_so_far + len(tool_calls)
    langfuse.update_current_span(metadata={"tool_calls_this_step": len(tool_calls), "tool_calls_total": new_count, "had_summary_context": "true" if bool(summary) else "false", "had_skill_context": "true" if bool(skill_ctx) else "false"})

    return {
        "messages": remove_ops + [response],
        "message_summary": summary,
        "number_of_tool_calls": new_count,
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
        * **Tools** — list/read filesystem, ``code_pipeline``, ask_user, read/write scratchpad.
        * **Middleware logic** — tool budget reset in ``reset_tool_budget``;
          context editing, summarisation, and tool call limit checks run inside
          the orchestrator node.
        * **code_pipeline** — LLM codegen, Semgrep, judge, save under ``agent_filesystem/code/``, then sandbox runner.
        * **Skills** — ``identify_skills`` injects ``skill_context`` (from ``skills/``) into the orchestrator context on new user turns.
    """

    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.graph = self.build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> Any:
        """Construct and compile the ``StateGraph``."""
        builder = StateGraph(AgentState)

        builder.add_node("refresh_data_schema", refresh_data_schema)
        builder.add_node("identify_skills", identify_skills)
        builder.add_node("reset_tool_budget", reset_tool_budget)
        builder.add_node("orchestrator", orchestrator)
        builder.add_node("tools", ToolNode(TOOLS))

        builder.set_entry_point("refresh_data_schema")
        builder.add_edge("refresh_data_schema", "identify_skills")
        builder.add_edge("identify_skills", "reset_tool_budget")
        builder.add_edge("reset_tool_budget", "orchestrator")
        builder.add_conditional_edges(
            "orchestrator", should_continue, {"tools": "tools", END: END},
        )
        # After tools run, refresh file listing so the next orchestrator turn sees new outputs.
        builder.add_edge("tools", "refresh_data_schema")

        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @observe(name="graph.run_graph", as_type="chain", capture_input=False, capture_output=False)
    def run_graph(
        self,
        session_id: str,
        user_query: str,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the agent graph with one user message.

        Parameters
            session_id  — ``configurable.thread_id`` (conversation key).
            user_query  — user text (may include appended upload paths).
            config      — invoke config with LangGraph runtime options.

        Returns
            Graph invoke result dict (``messages``, ``message_summary``, etc.).
        """
        config = config or {}
        # ``thread_id`` ties every turn for a session to the same checkpointed graph state.
        config["configurable"] = {"thread_id": session_id}
        # On each turn we add only the new user message and let the checkpointer load prior state.
        result = self.graph.invoke({"messages": [HumanMessage(content=user_query)]}, config=config)
        langfuse.update_current_span(input={"session_id": session_id, "user_query": user_query}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    @observe(name="graph.resume", as_type="chain", capture_input=False, capture_output=False)
    def resume(
        self,
        session_id: str,
        value: Any,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Resume a paused graph (after ``ask_user`` interrupt).

        Parameters
            session_id — same session that was interrupted.
            value      — the user's answer to the clarifying question.
            config     — invoke config with LangGraph runtime options.

        Returns
            Graph invoke result dict.
        """
        config = config or {}
        config["configurable"] = {"thread_id": session_id}
        result = self.graph.invoke(Command(resume=value), config=config)
        langfuse.update_current_span(input={"session_id": session_id, "resume_value": value}, output={"message_count": len(result.get("messages", []))}, metadata={"session_id": session_id})
        return result

    def get_state(self, session_id: str) -> Any:
        """Return the current state snapshot for *session_id*."""
        return self.graph.get_state({"configurable": {"thread_id": session_id}})
