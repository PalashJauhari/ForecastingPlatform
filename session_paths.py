"""
Session-aware path resolution.

Everything for a session lives in one folder — no ``input/``, ``output/``, or ``code/`` subfolders::

    agent_filesystem/<session-folder>/<filename>   # uploads (POST /upload-data)
    agent_filesystem/<session-folder>/run_<tool_call_id>/plot.png  # coding plots

Example logical paths::

    agent_filesystem/default/sales.csv
    agent_filesystem/default/run_call_abc123/trend.png

On disk (under the repo)::

    <project>/agent_filesystem/<session-folder>/...

``MAIN_PATH_AGENT_FILESYSTEM`` in root ``.env`` sets the folder name (default ``agent_filesystem``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent

def _env_str(name: str, default: str) -> str:
    """Read non-empty env value; fallback when unset/blank."""
    value = (os.environ.get(name) or "").strip()
    return value or default


LOGICAL_AGENT_FS = _env_str("MAIN_PATH_AGENT_FILESYSTEM", default="agent_filesystem").rstrip("/")
LOGICAL_AGENT_PREFIX = f"{LOGICAL_AGENT_FS}/"
SESSIONS_ROOT = (PROJECT_ROOT / LOGICAL_AGENT_FS).resolve()


def session_id_from_config(config: Any = None) -> str:
    """Read the current session/thread id from LangGraph or LangChain runtime config."""
    configurable = (config or {}).get("configurable", {})
    session_id = configurable.get("thread_id") or configurable.get("session_id") or "default"
    return str(session_id)


def session_dir_for_paths(session_id: str) -> str:
    """Directory name used in logical paths and on disk under ``agent_filesystem/``."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id.strip())
    return cleaned or "default"


def session_root(session_id: str) -> Path:
    """Physical workspace root for one session."""
    return SESSIONS_ROOT / session_dir_for_paths(session_id)


def ensure_session_dirs(session_id: str) -> Path:
    """Create the session directory if missing."""
    root = session_root(session_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def logical_input_file(session_id: str, filename: str) -> str:
    """Logical path for an uploaded file at the session root (basename only)."""
    name = Path(filename).name or "data"
    return f"{LOGICAL_AGENT_PREFIX}{session_dir_for_paths(session_id)}/{name}"


def resolve_agent_path(session_id: str, logical_path: str) -> Path:
    """
    Map ``agent_filesystem/<session-folder>/...`` to a path under this session's
    physical root. The first folder after ``agent_filesystem/`` must match the
    current session (prevents cross-session access).
    """
    path = str(logical_path).strip()
    if path == LOGICAL_AGENT_FS:
        return session_root(session_id)
    if not path.startswith(LOGICAL_AGENT_PREFIX):
        raise ValueError(
            f"Path must start with '{LOGICAL_AGENT_PREFIX}' (got: {logical_path!r}).",
        )

    rel = path[len(LOGICAL_AGENT_PREFIX) :].strip("/")
    if not rel:
        raise ValueError(f"Missing session and file path after '{LOGICAL_AGENT_PREFIX}'.")

    parts = rel.split("/", 1)
    session_key = parts[0]
    expected = session_dir_for_paths(session_id)
    # Cross-session guard: logical path must name the current session folder.
    if session_key != expected:
        raise ValueError(
            f"Path session folder {session_key!r} must match current session {expected!r}.",
        )

    root = session_root(session_id).resolve()
    if len(parts) == 1:
        return root

    target = (root / parts[1]).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path must stay inside the current session workspace.") from exc
    return target
