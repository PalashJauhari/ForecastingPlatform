"""
LangChain tool: list all .csv and .xlsx under this session (logical paths ``agent_filesystem/<session>/input|output/...``, on disk ``./agent_filesystem/<session>/...``).
"""

from __future__ import annotations

import json
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langfuse import observe

from session_paths import session_id_from_config, session_root, to_agent_path


@observe(name="tool.list_agent_filesystem_data", as_type="tool")
def _list_agent_filesystem_data_impl(runtime: ToolRuntime) -> str:
    """
    Scan the current session workspace and return all .csv and .xlsx files found.

    Call this first to discover what data files are available. Each path is
    ``agent_filesystem/<session-folder>/input|output/...`` for this session and can be passed to
    ``read_agent_filesystem_data`` or ``code_pipeline``.

    Returns a JSON string with two keys:
      - "files": list of matching file paths, e.g.
            [
              "agent_filesystem/my-session/input/sales.csv",
              "agent_filesystem/my-session/output/clean.xlsx"
            ]
      - "count": total number of files found (int).

    If no .csv or .xlsx files are found, returns {"files": [], "count": 0}.
    """
    session_id = session_id_from_config(runtime.config)
    base = session_root(session_id)
    if not base.exists():
        return json.dumps({"files": [], "count": 0})

    files = [
        to_agent_path(session_id, f)
        for f in sorted(base.rglob("*"))
        if f.is_file() and f.suffix.lower() in {".csv", ".xlsx"}
    ]

    return json.dumps({"files": files, "count": len(files)})


@tool
def list_agent_filesystem_data(runtime: ToolRuntime) -> str:
    """LangChain wrapper for the traced filesystem listing implementation."""
    return _list_agent_filesystem_data_impl(runtime=runtime)
