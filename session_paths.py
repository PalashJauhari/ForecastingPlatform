"""
Session-aware path resolution for the agent workspace.

The model keeps using logical ``agent_filesystem/...`` paths. This module maps
those logical paths into a per-session physical workspace on disk so sessions do
not read or overwrite each other's files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

LOGICAL_AGENT_FS = str(cfg["paths"]["agent_filesystem"]).rstrip("/")
SESSIONS_ROOT = (PROJECT_ROOT / cfg["paths"].get("sessions_root", "agent_sessions")).resolve()
SESSION_SUBDIRS = ("input", "output", "processed", "scratchpad", "code")


def session_id_from_config(config: Any = None) -> str:
    """Read the current session/thread id from LangGraph or LangChain runtime config."""
    configurable = (config or {}).get("configurable", {})
    session_id = configurable.get("thread_id") or configurable.get("session_id") or "default"
    return str(session_id)


def _session_dir_name(session_id: str) -> str:
    """Sanitize session ids before they become directory names."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id.strip())
    return cleaned or "default"


def session_root(session_id: str) -> Path:
    """Return the physical workspace root for a session."""
    return SESSIONS_ROOT / _session_dir_name(session_id)


def ensure_session_dirs(session_id: str) -> Path:
    """Create the standard workspace folders for a session and return its root."""
    root = session_root(session_id)
    for subdir in SESSION_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def resolve_agent_path(session_id: str, logical_path: str) -> Path:
    """
    Translate a logical ``agent_filesystem/...`` path into a session-local physical path.

    The returned path is guaranteed to stay within that session's workspace root.
    """
    path = str(logical_path).strip()
    prefix = f"{LOGICAL_AGENT_FS}/"

    if path == LOGICAL_AGENT_FS:
        return session_root(session_id)
    if not path.startswith(prefix):
        raise ValueError(
            f"Path must start with '{LOGICAL_AGENT_FS}/' (got: {logical_path!r})."
        )

    relative = Path(path[len(prefix):])
    if relative.is_absolute():
        raise ValueError("Path must stay inside agent_filesystem/.")

    root = session_root(session_id).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path must stay inside the current session workspace.") from exc
    return target


def to_agent_path(session_id: str, physical_path: str | Path) -> str:
    """Convert a session-local physical path back into the logical agent path form."""
    root = session_root(session_id).resolve()
    target = Path(physical_path).resolve()
    relative = target.relative_to(root)
    return f"{LOGICAL_AGENT_FS}/{relative.as_posix()}"
