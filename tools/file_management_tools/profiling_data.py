"""
Session tabular profiling for the flat per-session workspace.

Used by the ``data_profile`` graph node in ``graph/graph.py``. Profiling reads every
top-level ``.csv`` / ``.xlsx`` file in ``agent_filesystem/<session>/`` and
returns a list of per-file summaries stored only in graph state
(``data_profile``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langfuse import observe

from session_paths import ensure_session_dirs, session_root

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
_HEAD_ROWS = 5
_LOW_CARDINALITY_MAX_UNIQUE = 20


def _read_tabular_file(path: Path) -> pd.DataFrame:
    """Read one supported tabular file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _to_json_value(value: Any) -> Any:
    """Convert pandas / numpy values into JSON-friendly Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _head_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the first few rows as JSON-safe records."""
    records = df.head(_HEAD_ROWS).to_dict(orient="records")
    return [
        {str(column): _to_json_value(cell) for column, cell in row.items()}
        for row in records
    ]


def _numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return ``describe()`` output for numeric columns only."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    summary = numeric_df.describe().transpose()
    return {
        str(column_name): {
            str(metric): _to_json_value(metric_value)
            for metric, metric_value in row.items()
        }
        for column_name, row in summary.iterrows()
    }


def _column_profiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build per-column stats, including low-cardinality value lists."""
    row_count = int(len(df))
    profiles: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        null_count = int(series.isna().sum())
        non_null_count = int(series.notna().sum())
        unique_count = int(non_null.nunique(dropna=True))
        is_low_cardinality = unique_count <= _LOW_CARDINALITY_MAX_UNIQUE

        profile: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_fraction": round((null_count / row_count), 6) if row_count else 0.0,
            "unique_count": unique_count,
            "is_low_cardinality": is_low_cardinality,
        }
        if is_low_cardinality:
            unique_values = non_null.drop_duplicates().tolist()
            profile["unique_values"] = [_to_json_value(value) for value in unique_values]

        profiles.append(profile)

    return profiles


def _profile_one_file(file_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Return profile dict for graph state ``data_profile``: file, row_count, head, column_profiles, numeric_summary."""
    return {
        "file": file_name,
        "row_count": int(len(df)),
        "head": _head_records(df),
        "column_profiles": _column_profiles(df),
        "numeric_summary": _numeric_summary(df),
    }


def profile_session_workspace(session_id: str) -> list[dict[str, Any]]:
    """
    Profile every top-level CSV/XLSX file in the flat session workspace.

    The session layout is ``agent_filesystem/<session>/<filename>``, so this
    function reads only files directly under that folder and reports each file by
    basename only.
    """
    ensure_session_dirs(session_id)
    base = session_root(session_id)
    profiles: list[dict[str, Any]] = []

    for path in sorted(base.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            df = _read_tabular_file(path)
        except Exception as exc:
            profiles.append({"file": path.name, "error": f"Read failed: {exc}"})
            continue
        profiles.append(_profile_one_file(path.name, df))

    return profiles
