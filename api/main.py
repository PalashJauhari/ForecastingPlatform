"""
FastAPI HTTP API for the Forecasting Platform agent.

Endpoints
    POST /run                              — form: ``query``, ``session_id`` (``thread_id`` for the graph).
    POST /resume                           — resume after an ``ask_user`` interrupt.
    POST /upload-data                      — multipart: CSV/Excel files → ``agent_filesystem/<session>/<filename>``.
    GET  /artifact/{session_id}/{path:path} — serve a static artifact (plot or output file)
                                              from the session workspace; read-only, sandbox-checked.

Loads ``.env`` from the project root for ``OPENAI_API_KEY`` and optional
Langfuse keys.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langfuse import observe, propagate_attributes

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import AnalysisGraph
from observability.langfuse_handler import build_request_metadata, get_langfuse_client
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

app = FastAPI(
    title="GaussianBlurr — Forecasting Platform",
    description=(
        "HTTP API for the GaussianBlurr forecasting agent. Send natural-language tasks; the agent "
        "reads workspace data, may ask clarifying questions (interrupt / resume), and can generate "
        "and run analysis code under guardrails. Upload CSV or Excel into the session input area first when needed."
    ),
)
analysis_graph = AnalysisGraph()
langfuse = get_langfuse_client()


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
        JSON with ``session_id``, ``summary``, ``last_tool_result``, and ``images``.
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
        langfuse.update_current_span(output=response, metadata={"interrupted": response["interrupted"], "has_last_tool_result": response["last_tool_result"] is not None, "image_count": len(response.get("images") or [])})
        return response


@app.post("/resume")
@observe(name="api.resume", as_type="chain")
async def resume(
    resume_value: str = Form(...),
    session_id: str = Form("default"),
):
    """
    Resume a paused agent after an ``ask_user`` interrupt.

    Form fields
        resume_value — the user's answer to the clarifying question.
        session_id   — same session that was interrupted.

    Returns
        Same shape as ``/run``.
    """
    with propagate_attributes(session_id=session_id, tags=["api", "resume"], metadata=build_request_metadata(endpoint="/resume", interface="fastapi", query=resume_value)):
        result = analysis_graph.resume(session_id, resume_value)
        response = get_api_response(session_id, result)
        langfuse.update_current_span(output=response, metadata={"interrupted": response["interrupted"], "has_last_tool_result": response["last_tool_result"] is not None, "image_count": len(response.get("images") or [])})
        return response


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
