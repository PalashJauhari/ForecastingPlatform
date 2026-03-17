"""
Forecast agent built with create_agent + middleware.

Architecture:
  create_agent loop: LLM → tool calls → tool execution → LLM → ... → final text response.

Middleware stack (in order):
  1. SessionContextMiddleware     – sets session_id/csv_path/data_dir context vars for tools
  2. LLMToolSelectorMiddleware    – picks relevant tools; filesystem tools always included
  3. ToolCallLimitMiddleware       – caps tool calls per run
  4. SummarizationMiddleware       – condenses history when tokens grow
  5. ContextEditingMiddleware      – clears old tool results to save context
  6. FilesystemMiddleware          – file read/write/edit/ls in session scope
  7. ShellToolMiddleware           – sandbox code execution (Docker or host fallback)
  8. CodeSafetyMiddleware          – inspects Python code before sandbox execution

State: AgentState (messages) + session_id, csv_path, data_dir (last-wins reducers).
Persistence: InMemorySaver, thread_id = session_id.
Observability: Langfuse callbacks passed via config.
"""
import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ClearToolUsesEdit,
    CodexSandboxExecutionPolicy,
    ContextEditingMiddleware,
    LLMToolSelectorMiddleware,
    ShellToolMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import NotRequired

from graph.middleware.code_safety import CodeSafetyMiddleware
from graph.middleware.session_context import SessionContextMiddleware
from prompts.graph_prompts import SYSTEM_PROMPT
from tools.plot_tool import plot_data

# ── Helpers ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("./data")


def _last_wins(a: str, b: str) -> str:
    """Reducer: newest value wins (used for scalar state fields)."""
    return b


def _ensure_session_dir(session_id: str) -> Path:
    safe = re.sub(r"[^\w\\-]", "", session_id or "default") or "default"
    d = DATA_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── State schema ─────────────────────────────────────────────────────────────

class ForecastAgentState(AgentState):
    """Extends AgentState with session-scoped fields (last-wins reducers)."""
    session_id: NotRequired[Annotated[str, _last_wins]]
    csv_path: NotRequired[Annotated[str, _last_wins]]
    data_dir: NotRequired[Annotated[str, _last_wins]]


# ── AnalysisGraph ────────────────────────────────────────────────────────────

class AnalysisGraph:
    """
    Single agent instance shared across sessions.
    thread_id = session_id provides session isolation via InMemorySaver.
    """

    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        self._session_csv: Dict[str, str] = {}
        self._agent = self._build_agent()

    def set_csv_path(self, session_id: str, path: str) -> None:
        self._session_csv[session_id] = path

    # ── Build agent ──────────────────────────────────────────────────────

    def _build_agent(self):
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        model_str = f"openai:{model}"

        tools = [plot_data]

        middleware: List[Any] = [
            # 1. Session context → sets context vars for tools
            SessionContextMiddleware(),

            # 2. MODEL middleware (runs around LLM calls)
            LLMToolSelectorMiddleware(
                model=model_str,
                max_tools=5,
                always_include=["ls", "read_file", "write_file", "edit_file"],
            ),
            SummarizationMiddleware(
                model=model_str,
                trigger=("tokens", 6000),
                keep=("messages", 20),
            ),
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=100_000,
                        keep=5,
                        exclude_tools=["write_file"],
                    ),
                ],
            ),

            # 3. TOOL middleware (runs around tool calls)
            ToolCallLimitMiddleware(run_limit=15, exit_behavior="continue"),
            FilesystemMiddleware(
                system_prompt=(
                    "Use the filesystem tools to store intermediate analysis results, "
                    "notes, and long outputs. Files persist within the session."
                ),
            ),
            ShellToolMiddleware(
                workspace_root=str(DATA_DIR / "sandbox"),
                execution_policy=CodexSandboxExecutionPolicy(),
            ),
            CodeSafetyMiddleware(),
        ]

        return create_agent(
            model=model_str,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=middleware,
            state_schema=ForecastAgentState,
            checkpointer=self._checkpointer,
            name="analytics_agent",
        )

    # ── Run ──────────────────────────────────────────────────────────────

    def run_graph(
        self,
        session_id: str,
        user_query: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke agent. Returns a parsed dict:
          session_id, csv_path, summary, clarification_question,
          tool_output, last_tool_result, messages.
        """
        csv_path = self._session_csv.get(session_id, "")
        data_dir = str(_ensure_session_dir(session_id))

        invoke_input = {
            "messages": [HumanMessage(content=user_query)],
            "session_id": session_id,
            "csv_path": csv_path,
            "data_dir": data_dir,
        }

        invoke_config: Dict[str, Any] = {
            "configurable": {"thread_id": session_id},
        }
        if config:
            if "callbacks" in config:
                invoke_config["callbacks"] = config["callbacks"]
            for k, v in config.items():
                if k != "callbacks":
                    invoke_config[k] = v

        result = self._agent.invoke(invoke_input, config=invoke_config)
        return self._parse_output(result, session_id, csv_path)

    # ── Parse output ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_output(result: Dict[str, Any], session_id: str, csv_path: str) -> Dict[str, Any]:
        """Extract summary, clarification, tool results from agent output messages."""
        messages = result.get("messages", [])

        last_ai_content = ""
        last_tool_result = None
        tool_output: Dict[str, Any] = {}

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not last_ai_content:
                last_ai_content = msg.content or ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tc = msg.tool_calls[-1]
                    tool_output = tc.get("args", {}) if isinstance(tc, dict) else {}

            if hasattr(msg, "type") and getattr(msg, "type", "") == "tool" and last_tool_result is None:
                content = msg.content or ""
                try:
                    last_tool_result = json.loads(content)
                except Exception:
                    last_tool_result = content

        summary = last_ai_content if last_ai_content else "Done."
        is_question = summary.rstrip().endswith("?")

        return {
            "session_id": session_id,
            "csv_path": csv_path,
            "summary": None if is_question else summary,
            "clarification_question": summary if is_question else None,
            "tool_output": tool_output,
            "last_tool_result": last_tool_result,
            "messages": messages,
        }
