"""
Session context: thread-safe context variables for session_id, csv_path, data_dir.

SessionContextMiddleware sets these from agent state at the start of each run.
Tools read from them so they don't need session args in their signatures.
"""
import contextvars
from typing import Any, Dict, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState

session_id_var = contextvars.ContextVar("session_id", default="default")
csv_path_var = contextvars.ContextVar("csv_path", default="")
data_dir_var = contextvars.ContextVar("data_dir", default="data/default")


class SessionContextMiddleware(AgentMiddleware):
    """Reads session_id / csv_path / data_dir from agent state and sets context vars."""

    def before_agent(self, state: AgentState, **kwargs: Any) -> Optional[Dict[str, Any]]:

        session_id_var.set(state.get("session_id", "default"))
        csv_path_var.set(state.get("csv_path", ""))
        data_dir_var.set(state.get("data_dir", f"data/{sid}"))
        
        return None
