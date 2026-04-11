"""
Session-aware path resolution.

Logical paths (what the model and API use)::

    agent_filesystem/<session-folder>/input/...   # reads (uploads); do not write new artifacts here
    agent_filesystem/<session-folder>/output/...  # reads of prior results + all writes

On disk (same shape, under the repo)::

    <project>/agent_filesystem/<session-folder>/input/...
    <project>/agent_filesystem/<session-folder>/output/...

``paths.agent_filesystem`` in ``config.yaml`` sets the folder name (default ``agent_filesystem``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

LOGICAL_AGENT_FS = str(cfg["paths"]["agent_filesystem"]).rstrip("/")
LOGICAL_AGENT_PREFIX = f"{LOGICAL_AGENT_FS}/"
# All session workspaces live under this directory (same name as the logical root).
SESSIONS_ROOT = (PROJECT_ROOT / LOGICAL_AGENT_FS).resolve()
SESSION_SUBDIRS = ("input", "output")


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
    """Create ``input/`` and ``output/`` under the session root."""
    root = session_root(session_id)
    for subdir in SESSION_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def logical_input_file(session_id: str, filename: str) -> str:
    """Logical path for an uploaded file in ``input/`` (basename only)."""
    name = Path(filename).name or "data"
    return f"{LOGICAL_AGENT_PREFIX}{session_dir_for_paths(session_id)}/input/{name}"


def logical_output_code_dir(session_id: str) -> str:
    """Logical directory for generated ``pipeline_run.py``."""
    return f"{LOGICAL_AGENT_PREFIX}{session_dir_for_paths(session_id)}/output/code"


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


def to_agent_path(session_id: str, physical_path: str | Path) -> str:
    """Physical path under the session root → logical ``agent_filesystem/<session>/...``."""
    root = session_root(session_id).resolve()
    target = Path(physical_path).resolve()
    rel = target.relative_to(root)
    sid = session_dir_for_paths(session_id)
    return f"{LOGICAL_AGENT_FS}/{sid}/{rel.as_posix()}"
