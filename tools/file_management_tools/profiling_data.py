"""
Session tabular profiling: per-file head sample, dtypes, time-like columns,
cardinality, categorical hints, and numeric summaries.

Used by ``profile_session_file`` in ``graph/graph.py``. Results live only in graph
state ``data_profile`` (no snapshot file on disk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langfuse import observe

from session_paths import ensure_session_dirs, resolve_agent_path, session_root, to_agent_path

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


def _read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _jsonify_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def _head_records(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    sample = df.head(n)
    out: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        out.append({str(c): _jsonify_cell(row[c]) for c in df.columns})
    return out


def _datetime_candidate_columns(df: pd.DataFrame) -> list[str]:
    """Column names that look like time dimensions (dtype or parse/heuristic)."""
    found: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(str(col))
            continue
        sample = series.dropna().head(50)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        parse_rate = float(parsed.notna().mean())
        col_l = str(col).lower()
        name_hint = any(t in col_l for t in ("date", "time", "timestamp", "ds"))
        if parse_rate >= 0.7 or name_hint:
            found.append(str(col))
    return found


def _possible_categorical(series: pd.Series, n_rows: int, cardinality: int) -> bool:
    if n_rows == 0:
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        # Low-cardinality integers often encode categories
        if pd.api.types.is_integer_dtype(series) and cardinality <= min(50, max(5, n_rows // 20)):
            return True
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cap = min(100, max(5, n_rows // 20))
        return cardinality <= cap and cardinality < n_rows
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    return False


def _continuous_stats(series: pd.Series) -> dict[str, float] | None:
    if not pd.api.types.is_numeric_dtype(series):
        return None
    if pd.api.types.is_bool_dtype(series):
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
    }


def _profile_one_file(logical_path: str, df: pd.DataFrame) -> dict[str, Any]:
    n_rows = int(len(df))
    time_columns = _datetime_candidate_columns(df)
    time_set = set(time_columns)

    column_profiles: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        cardinality = int(s.nunique(dropna=True))
        null_pct = float(s.isna().mean() * 100.0) if n_rows else 0.0
        numeric_pct: float | None = None
        if pd.api.types.is_numeric_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            numeric_pct = float(coerced.notna().mean() * 100.0)
        else:
            coerced_num = pd.to_numeric(s, errors="coerce")
            if coerced_num.notna().any():
                numeric_pct = float(coerced_num.notna().mean() * 100.0)

        column_profiles.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "null_pct": round(null_pct, 4),
                "numeric_pct": None if numeric_pct is None else round(numeric_pct, 4),
                "cardinality": cardinality,
                "possible_categorical": _possible_categorical(s, n_rows, cardinality),
                "time_like": str(col) in time_set,
                "continuous_stats": _continuous_stats(s),
            }
        )

    return {
        "file": logical_path,
        "row_count": n_rows,
        "time_columns": time_columns,
        "head": _head_records(df, 5),
        "columns": column_profiles,
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
