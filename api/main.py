"""
FastAPI HTTP API for the Forecasting Platform agent.

Endpoints
    POST /run                              — form: ``query``, ``session_id`` (``thread_id`` for the graph).
    POST /resume                           — resume after an ``ask_user`` interrupt.
    POST /run/stream                       — JSON ``query`` / ``session_id``; SSE (``text/event-stream``) graph progress per node plus ``done`` matching ``/run``.
    POST /resume/stream                     — JSON ``resume_value`` / ``session_id``; same SSE semantics after ``interrupt``.
    POST /upload-data                      — multipart: CSV/Excel files → ``agent_filesystem/<session>/<filename>``.
    GET  /artifact/{session_id}/{path:path} — serve a static artifact (plot or output file)
                                              from the session workspace; read-only, sandbox-checked.

SSE contract (additive; Form routes unchanged): each frame follows the Server-Sent Events ``data`` line format (JSON payload, separated by blank line from the next frame).
``{"type":"node",...}`` — one LangGraph ``updates`` step per finished node (serial prep:
``ProfileSavedData`` → ``SummariseConversationalSummary`` → ``SelectPlannerSkills``, …).
``{"type":"done",...}`` — same fields ``get_api_response`` returns for ``/run``, plus keys ``type`` and ``session_id``.
``{"type":"error",...}`` — stream aborted; surfaced when the generator catches an exception after ``data`` has begun.

Loads ``.env`` from the project root for ``OPENAI_API_KEY`` and optional
Langfuse keys.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langfuse import observe, propagate_attributes

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from graph import AnalysisGraph
from observability.langfuse_handler import (
    build_request_metadata,
    get_langfuse_client,
    sse_stream_runnable_langfuse_pin,
)
from session_paths import (
    ensure_session_dirs,
    logical_input_file,
    resolve_agent_path,
    session_dir_for_paths,
    session_root,
)

ALLOWED_DATA_EXTENSIONS = {".csv", ".xlsx"}

# Extensions the ``/artifact`` endpoint will stream. Mirrors what the LLM is allowed to
# produce (csv/xlsx tabular outputs + png/svg plot outputs). Anything else is a 403.
ARTIFACT_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".png", ".svg"}

# Image extensions the API discovers under each ``run_<tool_call_id>`` folder.
# Kept tight: only the formats ``code_pipeline``'s plot patches actually emit.
IMAGE_EXTENSIONS = {".png", ".svg"}

_SSE_HEADERS = {
    "X-Accel-Buffering": "no",
}

app = FastAPI(
    title="Agentic Forecasting Platform",
    description=(
        "HTTP API for an agentic forecasting workspace. Send natural-language tasks; the agent "
        "reads workspace data, may pause for **Planner** clarification (interrupt / resume), and can generate "
        "and run analysis code under guardrails. Upload CSV or Excel into the session input area first when needed."
    ),
)
analysis_graph = AnalysisGraph()
langfuse = get_langfuse_client()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8050",
        "http://localhost:8050",
        "http://0.0.0.0:8050",
        "http://[::1]:8050",
    ],
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost|\[::1\]|0\.0\.0\.0)(:[1-9]\d*)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRunRequest(BaseModel):
    """JSON body for ``POST /run/stream``."""

    session_id: str = Field(default="default", description="Stable id used as LangGraph ``thread_id``.")
    query: str = Field(..., min_length=1, description="Natural-language agent instruction.")


class StreamResumeRequest(BaseModel):
    """JSON body for ``POST /resume/stream``."""

    session_id: str = Field(default="default", description="Same session id that paused on ``interrupt``.")
    resume_value: str = Field(..., min_length=1, description="User reply to ``ask_user``.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_run_id(raw: str) -> str:
    """
    Convert a LangChain ``tool_call_id`` into the same filesystem-safe folder id the
    forecasting / coding tools use when they create ``run_<id>/`` subfolders.

    Keeping the rule identical here is what lets the API line tool calls up with the
    on-disk plot folders without having to parse any tool result JSON.
    """
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw or ""))
    return safe.strip("_") or "unknown_run"


def _collect_current_turn_tool_call_ids(messages: list) -> list[str]:
    """
    Return the ``tool_call_id``s produced during the most recent agent turn.

    A "turn" starts at the most recent ``HumanMessage`` (the user's last input) and
    extends to the end of ``messages``. Within that slice we collect tool-call ids in
    chronological order from every ``ToolMessage``. Earlier tool calls (from previous
    turns) are intentionally ignored so the UI shows only this turn's plots.

    De-duplicates while preserving order; if a tool was somehow called twice with the
    same id (it shouldn't happen) we keep one entry.
    """
    last_human_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            last_human_idx = idx
            break

    turn_messages = messages[last_human_idx + 1:] if last_human_idx >= 0 else messages

    seen: set[str] = set()
    ordered: list[str] = []
    for msg in turn_messages:
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            if tool_call_id and tool_call_id not in seen:
                seen.add(tool_call_id)
                ordered.append(tool_call_id)
    return ordered


def _images_for_turn(session_id: str, tool_call_ids: list[str]) -> list[str]:
    """
    Walk every ``run_<sanitized_tool_call_id>/`` folder for the supplied tool calls and
    return the logical paths of any image artifacts (``.png`` / ``.svg``) inside.

    Only ``code_pipeline`` writes into these folders today (its plot patches redirect
    ``plt.savefig`` / ``Figure.savefig`` outputs there). Tools that do not emit images
    simply contribute an empty folder (or none at all) and are skipped silently.

    Returned paths use the logical ``agent_filesystem/<session>/run_<id>/<file>`` shape
    so the Dash UI can pass them straight to the ``/artifact/{session_id}/{path:path}``
    endpoint without any further translation.
    """
    if not tool_call_ids:
        return []

    sid = session_dir_for_paths(session_id)
    workspace = session_root(session_id)

    images: list[str] = []
    for tool_call_id in tool_call_ids:
        run_id = _sanitize_run_id(tool_call_id)
        run_dir = workspace / f"run_{run_id}"
        if not run_dir.is_dir():
            continue
        # Sort entries so the UI sees a stable order across requests; the underlying
        # filesystem order is not guaranteed.
        for entry in sorted(run_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            images.append(f"agent_filesystem/{sid}/run_{run_id}/{entry.name}")
    return images


def get_api_response(session_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the JSON body from a graph ``invoke()`` return value.

    When ``interrupt()`` runs inside the graph (e.g. ``ask_user``), v1 ``invoke``
    includes ``__interrupt__`` on that dict — no separate ``get_state()`` call.
    See https://docs.langchain.com/oss/python/langgraph/interrupts

    The response always includes an ``images`` list. It is computed from the current
    turn's tool-call ids (see ``_collect_current_turn_tool_call_ids``) by listing the
    matching ``run_<id>/`` folders. The UI consumes this list directly — it is no
    longer expected to scan ``last_tool_result`` for image paths.
    """
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        val = getattr(interrupts[0], "value", interrupts[0])
        if isinstance(val, dict):
            question = val.get("question", str(val))
        else:
            question = str(val)
        return {
            "session_id": session_id,
            "interrupted": True,
            "question": question,
            "summary": None,
            "last_tool_result": None,
            "images": [],
        }

    messages = result.get("messages", [])
    last_ai = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    last_tool_result = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "tool"),
        None,
    )
    tool_call_ids = _collect_current_turn_tool_call_ids(messages)
    images = _images_for_turn(session_id, tool_call_ids)

    return {
        "session_id": session_id,
        "interrupted": False,
        "summary": last_ai,
        "last_tool_result": last_tool_result,
        "images": images,
    }


def get_unique_upload_name(session_id: str, filename: str) -> tuple[str, bool]:
    """
    Return a session-local upload filename that will not overwrite an existing file.

    Duplicate names are preserved by appending ``_2``, ``_3``, ... before the suffix.
    """
    candidate = Path(filename).name or "data"
    stem = Path(candidate).stem or "data"
    suffix = Path(candidate).suffix
    renamed = False
    version = 1

    while True:
        logical_path = logical_input_file(session_id, candidate)
        if not resolve_agent_path(session_id, logical_path).exists():
            return candidate, renamed
        version += 1
        candidate = f"{stem}_{version}{suffix}"
        renamed = True


# ---------------------------------------------------------------------------
# Streaming / SSE helpers
# ---------------------------------------------------------------------------


def _sse(payload: dict[str, Any]) -> bytes:
    """Encode one Server-Sent Events ``data`` frame (newline-terminated UTF-8)."""

    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode("utf-8")


def _truncate_sse_preview(raw: Any, limit: int) -> str:
    """Prefer human-readable truncation for heterogeneous tool payloads."""

    body = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 1)] + "…"


def _sanitize_tool_calls_for_sse(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Project tool calls to names only (UI progress shows tool names, not arguments)."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            name = tc.get("name")
        else:
            name = getattr(tc, "name", None)
        out.append({"name": name})
    return out


_MAX_SSE_TODOS = 50
_MAX_SSE_TODO_CONTENT = 400


def _normalize_todos_for_sse(todos_raw: Any) -> tuple[list[dict[str, Any]], int]:
    """Project graph todo rows into SSE-safe dicts (``content``, ``status``, optional ``id``)."""
    if not isinstance(todos_raw, list):
        return [], 0
    normalized: list[dict[str, Any]] = []
    for item in todos_raw[:_MAX_SSE_TODOS]:
        if not isinstance(item, dict):
            continue
        ct = item.get("content")
        st = item.get("status")
        if ct is None and st is None:
            continue
        content_s = _truncate_sse_preview(ct if ct is not None else "", _MAX_SSE_TODO_CONTENT)
        status_s = _truncate_sse_preview(st if st is not None else "", 64)
        row: dict[str, Any] = {"content": content_s, "status": status_s}
        tid = item.get("id")
        if tid is not None:
            row["id"] = _truncate_sse_preview(str(tid), 128)
        normalized.append(row)
    return normalized, len(normalized)


def _snapshot_to_invoke_shape(graph: Any, session_id: str) -> Dict[str, Any]:
    """
    Recover invoke-shaped terminal dict from the compiled graph checkpoint.

    Copies ``interrupts`` from LangGraph snapshot state onto ``__interrupt__``, matching terminal ``invoke`` results.
    """
    snap = graph.get_state({"configurable": {"thread_id": session_id}})
    values_payload = getattr(snap, "values", None)
    merged: Dict[str, Any] = dict(values_payload or {})
    ints = getattr(snap, "interrupts", ()) or ()
    if ints:
        merged["__interrupt__"] = list(ints)
    return merged


# Omitted from SSE / UI progress (noisy join / bookkeeping nodes).
_SSE_SKIP_PROGRESS_NODES = frozenset({
    "ProfileSavedData_PostTools",
})


def stream_events_from_langgraph_chunk(session_id: str, update: Any) -> Iterator[Dict[str, Any]]:
    """Expand one ``graph.stream(..., stream_mode=\"updates\")`` chunk into SSE-ready payloads."""

    if not isinstance(update, dict) or not update:
        yield {"type": "debug", "session_id": session_id, "payload": update}
        return

    # ``updates`` can bundle multiple sibling nodes finishing the same tick (parallel fan-out).
    items = list(update.items())
    items.sort(key=lambda kv: kv[0])
    for node_name, payload in items:
        if node_name in _SSE_SKIP_PROGRESS_NODES:
            continue
        yield stream_event_single_node(session_id, node_name, payload)


def stream_event_single_node(session_id: str, node_name: str, payload: Any) -> Dict[str, Any]:
    """Map ``{node_name: partial_state_delta}`` to a compact client-facing envelope."""

    event: Dict[str, Any] = {
        "type": "node",
        "session_id": session_id,
        "node": node_name,
        "status": "completed",
    }

    # LangGraph emits this virtual key whenever ``interrupt()`` pauses execution.
    if node_name == "__interrupt__":
        interrupt_tuple = payload if isinstance(payload, (tuple, list)) else (payload,)
        first = interrupt_tuple[0] if interrupt_tuple else None
        value = getattr(first, "value", None) if first is not None else None
        question_preview = ""
        event["label"] = "Awaiting human input"
        if isinstance(value, dict):
            question_preview = str(value.get("question", value))
            if value.get("phase") == "planner":
                event["label"] = "Planning clarification"
        elif value is not None:
            question_preview = str(value)
        event["interrupt_preview"] = _truncate_sse_preview(question_preview or "(interrupt)", 400)
        return event

    if not isinstance(payload, dict):
        event["label"] = node_name.replace("_", " ").title()
        return event

    if node_name == "Orchestrator":
        event["label"] = "Orchestrator"
        msgs = payload.get("messages") or []
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                tool_calls = list(getattr(msg, "tool_calls", None) or [])
                event["had_tool_calls"] = bool(tool_calls)
                if tool_calls:
                    event["tool_calls"] = _sanitize_tool_calls_for_sse(tool_calls)
                break

    elif node_name == "RunTools":
        event["label"] = "Run tools"
        msgs = payload.get("messages") or []
        previews: list[dict[str, Any]] = []
        for msg in msgs:
            if isinstance(msg, ToolMessage):
                previews.append(
                    {
                        "name": getattr(msg, "name", None) or "tool",
                    }
                )
        event["tool_results"] = previews
        norm, n = _normalize_todos_for_sse(payload.get("todos"))
        if norm:
            event["todos"] = norm
            event["todo_count"] = n

    elif node_name == "Planner":
        event["label"] = "Plan session todos"
        norm, n = _normalize_todos_for_sse(payload.get("todos"))
        if norm:
            event["todos"] = norm
            event["todo_count"] = n

    elif node_name == "SelectPlannerSkills":
        planner_skills = payload.get("active_planner_skills") or []
        event["label"] = "Planner skills selected"
        event["active_planner_skills"] = planner_skills
        event["active_planner_skill_count"] = len(planner_skills)

    elif node_name == "SelectOrchestratorSkills":
        skills = payload.get("active_skills") or []
        event["label"] = "Orchestrator skills selected"
        event["active_skills"] = skills
        event["active_skill_count"] = len(skills)

    elif node_name == "ProfileSavedData":
        rows = payload.get("data_profile") or []
        event["label"] = "Profiling workspace inputs"
        event["profile_entries"] = len(rows)

    elif node_name == "ProfileSavedData_PostTools":
        rows = payload.get("data_profile") or []
        event["label"] = "Re-profile after tools"
        event["profile_entries"] = len(rows)

    elif node_name == "SummariseConversationalSummary":
        summary = payload.get("message_summary")
        preview = ""
        if isinstance(summary, str) and summary.strip():
            preview = summary.strip().split("\n")[0][:120]
        event["label"] = "Rolling context compaction"
        if preview:
            event["summary_preview"] = preview + ("…" if len(summary.strip()) > 120 else "")

    elif node_name == "BeginTurn":
        event["label"] = "Turn boundary"

    elif node_name == "FinalAnswer":
        event["label"] = "Assistant reply finalized"

    else:
        event["label"] = node_name.replace("_", " ").title()

    return event


def sse_event_lines_for_turn(
    analysis: AnalysisGraph,
    session_id: str,
    stream_updates: Iterator[Dict[str, Any]],
) -> Iterator[bytes]:
    """
    Drain one ``stream_*`` iterator, emit SSE payloads, append ``done`` via ``get_api_response``.

    The generator catches failures mid-flight — clients must treat trailing ``error`` payloads as authoritative.
    """

    try:
        for update in stream_updates:
            for envelope in stream_events_from_langgraph_chunk(session_id, update):
                yield _sse(envelope)
        merged_state = _snapshot_to_invoke_shape(analysis.graph, session_id)
        snapshot_body = get_api_response(session_id, merged_state)
        terminal: Dict[str, Any] = {"type": "done", "session_id": session_id}
        terminal.update(snapshot_body)
        yield _sse(terminal)
    except Exception as exc:
        yield _sse({"type": "error", "session_id": session_id, "error": str(exc)})


def _sse_bytes_stream_run(
    analysis: AnalysisGraph,
    session_id: str,
    query: str,
    langfuse_pin: dict[str, str],
) -> Iterator[bytes]:
    yield from sse_event_lines_for_turn(
        analysis,
        session_id,
        analysis.stream_graph(session_id, query, langfuse_pin=langfuse_pin),
    )


def _sse_bytes_stream_resume(
    analysis: AnalysisGraph,
    session_id: str,
    resume_value: str,
    langfuse_pin: dict[str, str],
) -> Iterator[bytes]:
    yield from sse_event_lines_for_turn(
        analysis,
        session_id,
        analysis.stream_resume(session_id, resume_value, langfuse_pin=langfuse_pin),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/run")
@observe(name="api.run", as_type="chain")
async def run(
    query: str = Form(...),
    session_id: str = Form("default"),
):
    """
    Run the analysis agent on a user query.

    Form fields
        query       — natural-language instruction.
        session_id  — ``thread_id`` for conversation memory.

    Returns
        JSON with ``session_id``, ``summary``, ``last_tool_result``, ``images``.
        ``images`` is a list of logical artifact paths
        (``agent_filesystem/<session>/run_<tool_call_id>/<file>.png|svg``) for
        every plot produced during this turn — discovered by listing the
        per-tool-call ``run_<id>`` folders, not by scanning tool result text.
        If the agent asks a clarifying question, ``interrupted`` is ``true``
        and ``question`` contains the text.
    """
    with propagate_attributes(session_id=session_id, tags=["api", "run"], metadata=build_request_metadata(endpoint="/run", interface="fastapi", query=query)):
        result = analysis_graph.run_graph(session_id, query)
        response = get_api_response(session_id, result)
        langfuse.update_current_span(
            output=response,
            metadata={
                "interrupted": response["interrupted"],
                "has_last_tool_result": response["last_tool_result"] is not None,
                "image_count": len(response.get("images") or []),
            },
        )
        return response


@app.post("/resume")
@observe(name="api.resume", as_type="chain")
async def resume(
    resume_value: str = Form(...),
    session_id: str = Form("default"),
):
    """
    Resume a paused agent after an **interrupt** (e.g. **Planner** clarification).

    Form fields
        resume_value — the user's answer to the clarifying question.
        session_id   — same session that was interrupted.

    Returns
        Same shape as ``/run``.
    """
    with propagate_attributes(session_id=session_id, tags=["api", "resume"], metadata=build_request_metadata(endpoint="/resume", interface="fastapi", query=resume_value)):
        result = analysis_graph.resume(session_id, resume_value)
        response = get_api_response(session_id, result)
        langfuse.update_current_span(
            output=response,
            metadata={
                "interrupted": response["interrupted"],
                "has_last_tool_result": response["last_tool_result"] is not None,
                "image_count": len(response.get("images") or []),
            },
        )
        return response


@app.post("/run/stream")
async def run_stream(request_body: StreamRunRequest) -> StreamingResponse:
    """
    SSE mirror of ``POST /run``: JSON body with ``session_id`` + ``query``.

    Streams ``type:node`` payloads for each LangGraph ``updates`` tick, ending with ``type:done`` whose
    fields match ``get_api_response`` / ``POST /run``.
    """

    sid = request_body.session_id
    query = request_body.query
    pin = sse_stream_runnable_langfuse_pin()
    return StreamingResponse(
        _sse_bytes_stream_run(analysis_graph, sid, query, pin),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/resume/stream")
async def resume_stream(request_body: StreamResumeRequest) -> StreamingResponse:
    """
    SSE mirror of ``POST /resume``: JSON body with ``session_id`` + ``resume_value``.
    """

    sid = request_body.session_id
    rv = request_body.resume_value
    pin = sse_stream_runnable_langfuse_pin()
    return StreamingResponse(
        _sse_bytes_stream_resume(analysis_graph, sid, rv, pin),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/upload-data")
@observe(name="api.upload_data", as_type="tool")
async def upload_data(
    files: list[UploadFile] = File(...),
    session_id: str = Form("default"),
):
    """Save uploaded CSV/Excel files to the current session workspace."""

    ensure_session_dirs(session_id)
    saved: list[str] = []
    renamed: list[dict[str, str]] = []

    for upload in files:
        name = Path(upload.filename or "").name or "data"
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_DATA_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Only .csv and .xlsx allowed. Got: {upload.filename or name}"
                    )
                },
            )
        stored_name, was_renamed = get_unique_upload_name(session_id, name)
        logical_path = logical_input_file(session_id, stored_name)
        dest = resolve_agent_path(session_id, logical_path)
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(logical_path)
        if was_renamed:
            renamed.append({"original_name": name, "stored_name": stored_name})

    response = {"saved": saved, "count": len(saved), "renamed": renamed}
    langfuse.update_current_span(output=response, metadata={"uploaded_count": len(saved), "session_id": session_id})
    return response


@app.get("/artifact/{session_id}/{path:path}")
async def artifact(session_id: str, path: str):
    """
    Stream a single artifact file from the session workspace.

    The Dash UI calls this for every plot path returned by ``code_pipeline`` (e.g.
    ``run_<run_id>/trend.png``) and for any other CSV / XLSX outputs the user wants to
    download. The endpoint is intentionally narrow: read-only, allowlisted extensions,
    and a hard symlink-resistant containment check against the session root.

    Path semantics
        ``session_id`` is the conversation key (matches ``thread_id``).
        ``path``       is everything after the session id, joined with ``/`` (FastAPI's
                       ``:path`` converter). Examples:
                          ``forecast.csv``
                          ``run_a1b2c3d4e5f6/trend.png``

    Failure modes
        400 — path escapes the session workspace (``..``, absolute, symlink out, …)
        403 — extension not in ``ARTIFACT_ALLOWED_EXTENSIONS``
        404 — file does not exist or is not a regular file
    """
    root = session_root(session_id).resolve()
    target = (root / path).resolve()

    # Containment check: ``relative_to`` raises if ``target`` is outside ``root``.
    # ``.resolve()`` above already followed any symlinks, so this defeats traversal.
    try:
        target.relative_to(root)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Path escapes session workspace."})

    if target.suffix.lower() not in ARTIFACT_ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=403, content={"error": f"Extension '{target.suffix}' is not served."})

    if not target.exists() or not target.is_file():
        return JSONResponse(status_code=404, content={"error": "Artifact not found."})

    return FileResponse(target)
