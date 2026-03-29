from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import TypedDict

from middleware.context_editing import truncate_and_summarize
from middleware.tool_call_limit import check_tool_call_limit
from prompts.graph_prompts import SYSTEM_PROMPT
from tools.human_in_loop.ask_user import ask_user
from tools.coding_tools.generate_code import generate_code
from tools.coding_tools.run_python_file import run_python_file
from tools.file_management_tools.list_agent_filesystem_data import list_agent_filesystem_data
from tools.file_management_tools.read_agent_filesystem_data import read_agent_filesystem_data
from tools.file_management_tools.read_scratchpad import read_scratchpad
from tools.file_management_tools.write_scratchpad import write_scratchpad

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

AGENT_FS = (PROJECT_ROOT / cfg["paths"]["agent_filesystem"]).resolve()
SCRATCHPAD_DIR = (PROJECT_ROOT / cfg["paths"]["scratchpad"]).resolve()
SCRATCHPAD_FILE = SCRATCHPAD_DIR / "scratchpad.md"

KEEP_RECENT = int(cfg["middleware"]["context_editing"]["keep_recent_messages"])
TOKEN_THRESHOLD = int(cfg["middleware"]["summarization"]["token_threshold"])
MAX_TOOL_CALLS = int(cfg["middleware"]["tool_call_limit"]["max_calls"])

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
                               user-invocation (reset on each new HumanMessage).
        data_schema          — newline-separated listing of data files currently in
                               ``agent_filesystem/`` (excluding ``scratchpad/``).
                               Refreshed before every orchestrator call.
    """
    messages: Annotated[list, add_messages]
    message_summary: str
    number_of_tool_calls: int
    data_schema: str


# ---------------------------------------------------------------------------
# Tools and LLM (module-level, built once)
# ---------------------------------------------------------------------------

TOOLS = [
    list_agent_filesystem_data,
    read_agent_filesystem_data,
    generate_code,
    run_python_file,
    ask_user,
    read_scratchpad,
    write_scratchpad,
]

llm = ChatOpenAI(model=cfg["models"]["orchestrator"], temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def refresh_data_schema(state: AgentState) -> Dict[str, Any]:
    """
    Scan ``agent_filesystem/`` for ``.csv`` and ``.xlsx`` files (excluding
    ``scratchpad/``) and update ``state["data_schema"]``.
    """
    files = [
        f"agent_filesystem/{f.relative_to(AGENT_FS).as_posix()}"
        for f in sorted(AGENT_FS.rglob("*"))
        if f.is_file()
        and f.suffix.lower() in {".csv", ".xlsx"}
        and not str(f.resolve()).startswith(str(SCRATCHPAD_DIR))
    ] if AGENT_FS.exists() else []

    return {"data_schema": "\n".join(files) if files else "(no data files found)"}


def orchestrator(state: AgentState) -> Dict[str, Any]:
    """
    Core agent node.

    Steps executed in order:
        1. **Tool call limit** — return early ``AIMessage`` if exceeded.
        2. **Summarisation + truncation** — if token estimate exceeds the
           threshold, evict old messages into a running summary and
           produce ``RemoveMessage`` ops for the ``add_messages`` reducer.
        3. **Build system prompt** — ``SYSTEM_PROMPT`` + available files
           (``data_schema``) + conversation summary (``message_summary``).
        4. **Call LLM** with bound tools.
        5. **Count tool calls** — increment ``number_of_tool_calls``.
    """
    messages = state["messages"]
    summary  = state.get("message_summary", "")

    # 1. Reset tool call count on new user invocation
    is_new_invocation = messages and isinstance(messages[-1], HumanMessage)
    tool_calls_so_far = 0 if is_new_invocation else state.get("number_of_tool_calls", 0)

    # 2. Tool call limit
    limit_msg = check_tool_call_limit(tool_calls_so_far, MAX_TOOL_CALLS)
    if limit_msg:
        return {"messages": [limit_msg]}
    remove_ops: list = []

    # 2. Summarisation + truncation
    summary, messages, remove_ops = truncate_and_summarize(
        messages, summary, KEEP_RECENT, TOKEN_THRESHOLD,
    )

    # 3. Call LLM
    context = (
        f"Available data files:\n{state.get('data_schema', '')}\n\n"
        f"Conversation summary:\n{summary}"
    )
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)] + messages,
    )

    # 5. Count tool calls
    new_count = tool_calls_so_far + len(response.tool_calls)

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
        * **Tools** — list/read filesystem, generate_code, run_python_file,
          ask_user, read/write scratchpad.
        * **Middleware logic** (context editing, summarisation, tool call limit)
          runs inside the orchestrator node as plain function calls.
        * **Codegen safety** — semgrep + path scanner inside ``generate_code``.
        * **Layer 3** (runtime patch) is applied inside ``run_python_file``.
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
        builder.add_node("orchestrator", orchestrator)
        builder.add_node("tools", ToolNode(TOOLS))

        builder.set_entry_point("refresh_data_schema")
        builder.add_edge("refresh_data_schema", "orchestrator")
        builder.add_conditional_edges(
            "orchestrator", should_continue, {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "refresh_data_schema")

        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            config      — invoke config with Langfuse callbacks.

        Returns
            Graph invoke result dict (``messages``, ``message_summary``, etc.).
        """
        # thread_id drives InMemorySaver checkpointing per session
        config["configurable"] = {"thread_id": session_id}
        return self.graph.invoke(
            {"messages": [HumanMessage(content=user_query)]}, config=config,
        )

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
            config     — invoke config with Langfuse callbacks.

        Returns
            Graph invoke result dict.
        """
        config["configurable"] = {"thread_id": session_id}
        return self.graph.invoke(Command(resume=value), config=config)

    def get_state(self, session_id: str) -> Any:
        """Return the current state snapshot for *session_id*."""
        return self.graph.get_state({"configurable": {"thread_id": session_id}})
