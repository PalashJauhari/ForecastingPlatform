"""
LangChain tool: profile one or more tabular files for forecasting readiness.

This tool inspects CSV/XLSX files, identifies likely time and target columns,
flags common forecasting blockers, and recommends the next step. It does not
modify source data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langfuse import observe
from langgraph.types import Command
from pydantic import BaseModel, Field

from session_paths import ensure_session_dirs, resolve_agent_path, session_id_from_config

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
PROFILE_OUTPUT_PATH = "agent_filesystem/processed/forecast_profile.json"


class ProfileForecastingDataInput(BaseModel):
    paths: list[str] = Field(
        description=(
            "One or more logical .csv/.xlsx file paths starting with 'agent_filesystem/'."
        ),
        min_length=1,
    )
    time_column: str = Field(
        default="",
        description="Optional user-provided time column hint. Leave empty if unknown.",
    )
    target_column: str = Field(
        default="",
        description="Optional user-provided target column hint. Leave empty if unknown.",
    )
    goal: str = Field(
        default="",
        description="Optional forecasting goal like 'forecast weekly revenue'.",
    )


def _read_tabular_file(path: Path) -> pd.DataFrame:
    """Read one CSV/XLSX file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _datetime_candidates(df: pd.DataFrame) -> list[str]:
    """
    Return columns that likely represent time.

    A column qualifies if it is already datetime-like, most sampled values parse
    as datetimes, or the column name strongly suggests time semantics.
    """
    candidates: list[str] = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(str(col))
            continue

        sample = series.dropna().head(50)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        parse_rate = float(parsed.notna().mean())
        col_name = str(col).lower()
        looks_like_time_name = any(
            token in col_name for token in {"date", "time", "timestamp", "ds"}
        )

        if parse_rate >= 0.7 or looks_like_time_name:
            candidates.append(str(col))

    return candidates


def _numeric_candidates(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that could plausibly be forecasting targets."""
    return [str(col) for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _likely_join_columns(df: pd.DataFrame, datetime_candidates: list[str]) -> list[str]:
    """
    Return columns that look usable as join keys.

    These are usually identifiers, categories, or date columns with repeated
    values, not high-cardinality free text.
    """
    candidates: list[str] = []

    for col in df.columns:
        series = df[col]
        nunique = int(series.nunique(dropna=True))

        if nunique == 0:
            continue

        is_datetime_like = str(col) in datetime_candidates
        is_object_like = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        is_reasonable_cardinality = 1 < nunique < max(1000, int(len(df) * 0.8))

        if is_datetime_like or (is_object_like and is_reasonable_cardinality):
            candidates.append(str(col))

    return candidates


def _infer_frequency(ts: pd.Series) -> str:
    """Infer a stable frequency from a parsed datetime series when possible."""
    clean = pd.to_datetime(ts, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(clean) < 3:
        return ""

    inferred = pd.infer_freq(clean)
    return str(inferred) if inferred else ""


def _issue(code: str, severity: str, message: str) -> dict[str, str]:
    """Build a small issue object for downstream routing and user explanation."""
    return {"code": code, "severity": severity, "message": message}


def _profile_single_file(
    logical_path: str,
    df: pd.DataFrame,
    provided_time_column: str,
    provided_target_column: str,
) -> dict[str, Any]:
    """Build a forecasting-oriented profile for one file."""
    datetime_candidates = _datetime_candidates(df)
    numeric_candidates = _numeric_candidates(df)

    detected_time_column = ""
    if provided_time_column and provided_time_column in df.columns:
        detected_time_column = provided_time_column
    elif len(datetime_candidates) == 1:
        detected_time_column = datetime_candidates[0]

    detected_target_column = ""
    if provided_target_column and provided_target_column in df.columns:
        detected_target_column = provided_target_column
    elif len(numeric_candidates) == 1:
        detected_target_column = numeric_candidates[0]

    issues: list[dict[str, str]] = []
    frequency = ""

    if not datetime_candidates:
        issues.append(_issue("missing_time_column", "high", "No clear time column was detected."))
    elif not detected_time_column:
        issues.append(_issue(
            "ambiguous_time_column",
            "high",
            f"Multiple time-like columns were found: {datetime_candidates}.",
        ))
    else:
        parsed_time = pd.to_datetime(df[detected_time_column], errors="coerce")

        if parsed_time.isna().all():
            issues.append(_issue(
                "invalid_time_column",
                "high",
                f"Column '{detected_time_column}' could not be parsed as datetime.",
            ))
        else:
            duplicate_timestamps = int(parsed_time.dropna().duplicated().sum())
            if duplicate_timestamps > 0:
                issues.append(_issue(
                    "duplicate_timestamps",
                    "medium",
                    f"{duplicate_timestamps} duplicate timestamps detected in '{detected_time_column}'.",
                ))

            if not parsed_time.dropna().is_monotonic_increasing:
                issues.append(_issue(
                    "not_sorted_by_time",
                    "medium",
                    f"Rows are not sorted by '{detected_time_column}'.",
                ))

            frequency = _infer_frequency(parsed_time)
            if not frequency:
                issues.append(_issue(
                    "irregular_or_unknown_frequency",
                    "medium",
                    f"Could not infer a stable frequency from '{detected_time_column}'.",
                ))

    if not numeric_candidates:
        issues.append(_issue(
            "missing_target_column",
            "high",
            "No clear numeric target column was detected.",
        ))
    elif not detected_target_column:
        issues.append(_issue(
            "ambiguous_target_column",
            "high",
            f"Multiple numeric columns could be the target: {numeric_candidates}.",
        ))
    else:
        target_series = df[detected_target_column]

        if not pd.api.types.is_numeric_dtype(target_series):
            issues.append(_issue(
                "target_not_numeric",
                "high",
                f"Column '{detected_target_column}' is not numeric.",
            ))

        target_nulls = int(target_series.isna().sum())
        if target_nulls > 0:
            issues.append(_issue(
                "target_has_nulls",
                "medium",
                f"Column '{detected_target_column}' contains {target_nulls} null values.",
            ))

    return {
        "path": logical_path,
        "rows": int(len(df)),
        "columns": [str(col) for col in df.columns],
        "column_dtypes": {str(col): str(df[col].dtype) for col in df.columns},
        "datetime_candidates": datetime_candidates,
        "numeric_candidates": numeric_candidates,
        "join_candidates": _likely_join_columns(df, datetime_candidates),
        "detected_time_column": detected_time_column,
        "detected_target_column": detected_target_column,
        "frequency": frequency,
        "issues": issues,
    }


def _profile_score(file_profile: dict[str, Any]) -> int:
    """
    Score a file as a possible primary forecasting dataset.

    Higher score means the file looks more forecast-ready.
    """
    score = 0

    if file_profile["detected_time_column"]:
        score += 3
    if file_profile["detected_target_column"]:
        score += 3
    if file_profile["frequency"]:
        score += 2
    if file_profile["rows"] >= 20:
        score += 1

    issue_codes = {issue["code"] for issue in file_profile["issues"]}
    if "missing_time_column" in issue_codes:
        score -= 3
    if "ambiguous_time_column" in issue_codes:
        score -= 2
    if "missing_target_column" in issue_codes:
        score -= 3
    if "ambiguous_target_column" in issue_codes:
        score -= 2
    if "invalid_time_column" in issue_codes:
        score -= 3

    return score


def _shared_join_candidates(file_profiles: list[dict[str, Any]]) -> list[str]:
    """Return join candidate column names that appear in more than one file."""
    counts: dict[str, int] = {}

    for profile in file_profiles:
        for col in profile["join_candidates"]:
            counts[col] = counts.get(col, 0) + 1

    return sorted([col for col, count in counts.items() if count >= 2])


@observe(name="tool.profile_forecasting_data", as_type="tool")
def _profile_forecasting_data_impl(
    paths: list[str],
    time_column: str = "",
    target_column: str = "",
    goal: str = "",
    runtime: ToolRuntime | None = None,
) -> str | Command:
    """
    Inspect one or more tabular files and assess forecasting readiness.

    Returns a JSON string with:
    - per-file profiles
    - selected primary dataset
    - likely join keys
    - issues
    - status
    - next_step
    - artifact_path
    """
    session_id = session_id_from_config(runtime.config if runtime is not None else None)
    ensure_session_dirs(session_id)

    dataframes: list[tuple[str, pd.DataFrame]] = []

    for logical_path in paths:
        clean_path = logical_path.strip()

        if not clean_path.startswith("agent_filesystem/"):
            return json.dumps({"error": f"Path must start with 'agent_filesystem/': {clean_path}"})

        try:
            physical_path = resolve_agent_path(session_id, clean_path)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        if not physical_path.exists():
            return json.dumps({"error": f"File not found: {clean_path}"})

        if physical_path.is_dir():
            return json.dumps({"error": f"Path is a directory, not a file: {clean_path}"})

        if physical_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return json.dumps({"error": f"Only .csv and .xlsx files are supported. Got: {clean_path}"})

        df = _read_tabular_file(physical_path)
        dataframes.append((clean_path, df))

    file_profiles = [
        _profile_single_file(
            logical_path=logical_path,
            df=df,
            provided_time_column=time_column.strip(),
            provided_target_column=target_column.strip(),
        )
        for logical_path, df in dataframes
    ]

    primary_profile = max(file_profiles, key=_profile_score)
    shared_join_keys = _shared_join_candidates(file_profiles)

    combined_issues: list[dict[str, str]] = []
    for profile in file_profiles:
        for issue in profile["issues"]:
            combined_issues.append({"file": profile["path"], **issue})

    if len(file_profiles) > 1 and not shared_join_keys:
        combined_issues.append({
            "file": "",
            "code": "no_clear_join_keys",
            "severity": "medium",
            "message": "Multiple files were provided, but no obvious shared join keys were detected.",
        })

    primary_issue_codes = {issue["code"] for issue in primary_profile["issues"]}

    if {
        "missing_time_column",
        "ambiguous_time_column",
        "invalid_time_column",
    } & primary_issue_codes:
        status = "needs_user_clarification"
        next_step = "Confirm the correct time column in the primary dataset."
    elif {
        "missing_target_column",
        "ambiguous_target_column",
        "target_not_numeric",
    } & primary_issue_codes:
        status = "needs_user_clarification"
        next_step = "Confirm the correct numeric target column in the primary dataset."
    elif combined_issues:
        status = "needs_standardization"
        if len(file_profiles) > 1:
            next_step = "Standardize the primary dataset and merge useful auxiliary files before forecasting."
        else:
            next_step = "Standardize and clean the dataset before forecasting."
    else:
        status = "ready"
        next_step = "Dataset looks forecast-ready. Proceed to forecasting workflow."

    result = {
        "goal": goal,
        "status": status,
        "next_step": next_step,
        "primary_dataset": {
            "path": primary_profile["path"],
            "detected_time_column": primary_profile["detected_time_column"],
            "detected_target_column": primary_profile["detected_target_column"],
            "frequency": primary_profile["frequency"],
        },
        "shared_join_keys": shared_join_keys,
        "files": file_profiles,
        "issues": combined_issues,
        "artifact_path": PROFILE_OUTPUT_PATH,
    }

    result_json = json.dumps(result)
    artifact_path = resolve_agent_path(session_id, PROFILE_OUTPUT_PATH)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if runtime is None:
        return result_json

    return Command(
        update={
            "latest_profile_result": result_json,
            "messages": [
                ToolMessage(
                    content=result_json,
                    tool_call_id=getattr(runtime, "tool_call_id", "profile_forecasting_data"),
                )
            ],
        },
    )


@tool(args_schema=ProfileForecastingDataInput)
def profile_forecasting_data(
    paths: list[str],
    time_column: str = "",
    target_column: str = "",
    goal: str = "",
    runtime: ToolRuntime | None = None,
) -> str | Command:
    """LangChain wrapper for the traced forecasting profile implementation."""
    return _profile_forecasting_data_impl(
        paths=paths,
        time_column=time_column,
        target_column=target_column,
        goal=goal,
        runtime=runtime,
    )
