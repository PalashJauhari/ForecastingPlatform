"""
Session tabular profiling: minimal preview (``pandas.DataFrame.head``) per file.

Used by ``profile_session_file`` in ``graph/graph.py``. Results live only in graph
state ``data_profile`` (no snapshot file on disk). Extend with richer stats later if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langfuse import observe

from session_paths import ensure_session_dirs, resolve_agent_path, session_root, to_agent_path

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
_HEAD_ROWS = 5


def _read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _jsonify_cell(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and (v != v or abs(v) == float("inf")):  # NaN / inf
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


def _head_records(df: pd.DataFrame, n: int = _HEAD_ROWS) -> list[dict[str, Any]]:
    sample = df.head(n)
    out: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        out.append({str(c): _jsonify_cell(row[c]) for c in df.columns})
    return out


def _profile_one_file(logical_path: str, df: pd.DataFrame) -> dict[str, Any]:
    """Minimal profile: path, row count, column names, first ``n`` rows only."""
    return {
        "file": logical_path,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "head": _head_records(df, _HEAD_ROWS),
    }


def _profile_path(session_id: str, logical_path: str) -> dict[str, Any]:
    clean = logical_path.strip()
    if not clean.startswith("agent_filesystem/"):
        return {"file": clean, "error": f"Path must start with 'agent_filesystem/': {clean}"}
    try:
        physical = resolve_agent_path(session_id, clean)
    except ValueError as e:
        return {"file": clean, "error": str(e)}
    if not physical.exists():
        return {"file": clean, "error": f"File not found: {clean}"}
    if physical.is_dir():
        return {"file": clean, "error": f"Path is a directory: {clean}"}
    if physical.suffix.lower() not in ALLOWED_EXTENSIONS:
        return {"file": clean, "error": f"Only .csv and .xlsx supported: {clean}"}
    try:
        df = _read_tabular_file(physical)
    except Exception as e:
        return {"file": clean, "error": f"Read failed: {e}"}
    return _profile_one_file(clean, df)


def profile_session_workspace(session_id: str) -> list[dict[str, Any]]:
    """
    Discover every ``.csv`` / ``.xlsx`` under the session workspace and profile each path.

    This is the entry point used by ``graph.profile_session_file``; it delegates to
    ``build_session_data_profile``.
    """
    base = session_root(session_id)
    logical_paths = [
        to_agent_path(session_id, f)
        for f in sorted(base.rglob("*"))
        if f.is_file()
        and f.suffix.lower() in {".csv", ".xlsx"}
    ] if base.exists() else []
    return build_session_data_profile(session_id, logical_paths)


@observe(name="profiling_data.build_session_data_profile", capture_input=False, capture_output=False)
def build_session_data_profile(session_id: str, logical_paths: list[str]) -> list[dict[str, Any]]:
    """
    Profile every logical path and return a **list** of per-file dicts (or single-entry error dicts).
    """
    ensure_session_dirs(session_id)
    profiles: list[dict[str, Any]] = []
    for p in logical_paths:
        profiles.append(_profile_path(session_id, p))

    return profiles
