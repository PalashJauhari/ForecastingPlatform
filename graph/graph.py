"""
LangGraph/LangChain agent construction: ``create_agent`` with tools and middleware.

Loads ``config.yaml`` from the project root for model names and middleware tuning.
Checkpointing uses :class:`langgraph.checkpoint.memory.InMemorySaver` with ``thread_id`` = session id.

Middleware stack (in order):
    1. ContextEditingMiddleware
    2. ToolCallLimitMiddleware

Codegen safety (semgrep + path check on generated source) runs **inside**
:func:`tools.coding_tools.generate_code.generate_code` using ``tools/coding_tools/code_scan/agent_sandbox.yaml``.

Layer 3 (runtime patch) lives inside ``run_python_file``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from prompts.graph_prompts import SYSTEM_PROMPT
from tools.coding_tools.generate_code import generate_code
from tools.coding_tools.run_python_file import run_python_file
from tools.file_management_tools.list_agent_filesystem_data import list_agent_filesystem_data
from tools.file_management_tools.read_agent_filesystem_data import read_agent_filesystem_data

_ROOT = Path(__file__).resolve().parent.parent
_cfg  = yaml.safe_load(open(_ROOT / "config.yaml"))


class AnalysisGraph:
    """
    Thin wrapper around a single LangChain ``create_agent`` instance.

    Notes
        * **Orchestrator model** — ``models.orchestrator`` from ``config.yaml``, OpenAI provider.
        * **Tools** — list_agent_filesystem_data, read_agent_filesystem_data, generate_code, run_python_file.
        * **Middleware** — context editing, tool-call limit.
        * **Codegen safety** — semgrep + path checker inside ``generate_code``.
        * **Layer 3** (runtime patch) is applied inside ``run_python_file``.
    """

    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        self._agent = self._build_agent()

    def _build_agent(self) -> Any:
        """Construct ``create_agent`` with tools, system prompt, and middleware stack."""
        model_str = f"openai:{_cfg['models']['orchestrator']}"
        mw  = _cfg.get("middleware", {})
        ctx  = mw.get("context_editing", {})
        tlim = mw.get("tool_call_limit", {})

        tools = [list_agent_filesystem_data, read_agent_filesystem_data, generate_code, run_python_file]

        middleware: List[Any] = [
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=int(ctx.get("clear_tool_uses_trigger", 100_000)),
                        keep=int(ctx.get("clear_tool_uses_keep", 5)),
                        exclude_tools=list(ctx.get("exclude_tools", [])),
                    ),
                ],
            ),
            ToolCallLimitMiddleware(
                run_limit=int(tlim.get("run_limit", 15)),
                exit_behavior="continue",
            ),
        ]

        return create_agent(
            model=model_str,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=middleware,
            checkpointer=self._checkpointer,
            name="analytics_agent",
        )

    def run_graph(
        self,
        session_id: str,
        user_query: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with one user message and session-scoped checkpointing.

        Parameters
            session_id  — ``configurable.thread_id`` (conversation key).
            user_query  — user text (may include appended upload paths from the API).
            config      — optional invoke config; ``callbacks`` (e.g. Langfuse) merged in.

        Returns
            Agent invoke result dict (typically includes ``messages`` list).
        """
        invoke_input = {"messages": [HumanMessage(content=user_query)]}
        invoke_config: Dict[str, Any] = {"configurable": {"thread_id": session_id}}
        if config:
            if "callbacks" in config:
                invoke_config["callbacks"] = config["callbacks"]
            for k, v in config.items():
                if k != "callbacks":
                    invoke_config[k] = v
        return self._agent.invoke(invoke_input, config=invoke_config)
