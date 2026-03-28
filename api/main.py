"""
FastAPI HTTP API for the Forecasting Platform agent.

Endpoints
    POST /run         — form: ``query``, ``session_id`` (``thread_id`` for the graph).
    POST /resume      — resume after an ``ask_user`` interrupt.
    POST /upload-data — multipart: one or more CSV/Excel files → ``agent_filesystem/input/<session>/``.

Loads ``.env`` from the project root for ``OPENAI_API_KEY`` and optional
Langfuse keys.
"""

from __future__ import annotations

import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage

from graph import AnalysisGraph
from observability.langfuse_handler import get_langfuse_callbacks

_cfg      = yaml.safe_load(open(_ROOT / "config.yaml"))
INPUT_DIR = _ROOT / _cfg["paths"]["input"]
AGENT_FS  = _ROOT / _cfg["paths"]["agent_filesystem"]

ALLOWED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(
    title="GaussianBlurr — Forecasting Platform",
    description="Agent API: natural-language queries with optional human-in-the-loop.",
)
analysis_graph = AnalysisGraph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def session_input_dir(session_id: str) -> Path:
    """``agent_filesystem/input/<sanitized_session_id>/`` (created if missing)."""
    safe = re.sub(r"[^\w\-]", "", session_id) or "default"
    d = INPUT_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_upload_basename(filename: str | None) -> str:
    """Single path segment, no traversal; safe for disk write."""
    base = Path(filename or "data").name
    if not base or base in {".", ".."}:
        base = "data"
    cleaned = re.sub(r"[^\w.\-]", "_", base).strip("._") or "data"
    return cleaned[:200]


def extract_response(result: Dict[str, Any]) -> tuple[str, Any]:
    """
    Pull the last ``AIMessage`` content and the last tool result from
    the graph result's message list.

    Returns
        ``(summary_text, last_tool_result)``
    """
    messages = result.get("messages", [])

    last_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai = msg.content or ""
            if last_ai:
                break

    last_tool_result = None
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "tool":
            last_tool_result = getattr(msg, "content", None)
            break

    return last_ai, last_tool_result


def check_interrupt(session_id: str) -> Dict[str, Any] | None:
    """
    If the graph is paused on ``interrupt()``, return API JSON for the client.

    LangGraph surfaces pending payloads on :class:`~langgraph.types.StateSnapshot`
    as ``snapshot.interrupts`` (tuple of ``Interrupt``).  The v1 ``invoke`` output
    also uses ``__interrupt__``; the same list may appear under ``values`` on
    the checkpoint, so we fall back to ``values['__interrupt__']`` when needed.
    See: https://docs.langchain.com/oss/python/langgraph/interrupts
    """
    snap = analysis_graph.get_state(session_id)
    if not snap:
        return None

    pending = tuple(getattr(snap, "interrupts", ()) or ())
    if not pending and isinstance(snap.values, dict):
        alt = snap.values.get("__interrupt__")
        if alt:
            pending = tuple(alt) if isinstance(alt, (list, tuple)) else (alt,)

    if not pending:
        return None

    first = pending[0]
    val = getattr(first, "value", first)
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
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/run")
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
        JSON with ``session_id``, ``summary``, ``last_tool_result``.
        If the agent asks a clarifying question, ``interrupted`` is ``true``
        and ``question`` contains the text.
    """
    result = analysis_graph.run_graph(
        session_id, query, config={"callbacks": get_langfuse_callbacks()},
    )

    interrupt_resp = check_interrupt(session_id)
    if interrupt_resp:
        return interrupt_resp

    summary, last_tool_result = extract_response(result)

    return {
        "session_id": session_id,
        "summary": summary or None,
        "last_tool_result": last_tool_result,
    }


@app.post("/resume")
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
    result = analysis_graph.resume(
        session_id, resume_value, config={"callbacks": get_langfuse_callbacks()},
    )

    interrupt_resp = check_interrupt(session_id)
    if interrupt_resp:
        return interrupt_resp

    summary, last_tool_result = extract_response(result)

    return {
        "session_id": session_id,
        "summary": summary or None,
        "last_tool_result": last_tool_result,
    }


@app.post("/upload-data")
async def upload_data(
    session_id: str = Form("default"),
    files: list[UploadFile] = File(...),
):
    """
    Save one or more CSV/Excel files under ``agent_filesystem/input/<session_id>/``.

    Form fields
        session_id — same id as ``/run`` so the agent sees files under that folder.
        files      — multipart file parts (repeat field name ``files`` for multiple).

    Returns
        JSON: ``session_id``, ``saved`` (list of ``{"path", "stored_as"}``), ``count``.
    """
    if not files:
        return JSONResponse(
            content={"error": "At least one file is required."},
            status_code=400,
        )

    session_d = session_input_dir(session_id)
    agent_root = AGENT_FS.resolve()
    saved: list[dict[str, str]] = []

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_DATA_EXTENSIONS:
            return JSONResponse(
                content={
                    "error": (
                        f"Only CSV and Excel files are allowed (.csv, .xlsx, .xls). "
                        f"Rejected: {upload.filename or '(no name)'} ({ext or 'no extension'})"
                    )
                },
                status_code=400,
            )
        base = safe_upload_basename(upload.filename)
        if Path(base).suffix.lower() not in ALLOWED_DATA_EXTENSIONS:
            base = f"{base}{ext}"
        dest = session_d / f"{uuid.uuid4().hex[:8]}_{base}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(upload.file, out)
        rel = dest.resolve().relative_to(agent_root)
        saved.append({
            "path": f"agent_filesystem/{rel.as_posix()}",
            "stored_as": dest.name,
        })

    return {
        "session_id": session_id,
        "saved": saved,
        "count": len(saved),
    }
