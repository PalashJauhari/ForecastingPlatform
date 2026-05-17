"""
Thin HTTP client for the GaussianBlurr FastAPI service (``api/main.py``).

All calls use ``requests`` with explicit timeouts and small, predictable payloads.
The Dash app imports this module so UI code stays free of transport details.

Endpoints (must match FastAPI ``Form`` / ``File`` field names):

- ``POST /upload-data`` — ``multipart/form-data``: form field ``session_id``,
  file field ``files`` (same name the server expects for ``UploadFile``).
- ``POST /run`` — ``application/x-www-form-urlencoded``-style body via
  ``data=``: ``query``, ``session_id``.
- ``POST /run/stream`` — JSON ``query`` / ``session_id``; streamed SSE frames (scripting helper :meth:`iter_run_stream`).
- ``POST /resume`` — ``data=``: ``resume_value``, ``session_id``.
- ``POST /resume/stream`` — JSON ``resume_value`` / ``session_id`` (:meth:`iter_resume_stream`).
- ``GET  /artifact/{session_id}/{path}`` — built via :meth:`GaussianBlurrApiClient.artifact_url`,
  used by the Dash UI's ``html.Img(src=...)`` to render plot artifacts inline.

Successful ``/run`` and ``/resume`` responses are JSON objects; this client
normalizes missing keys so callers can use ``.get()`` safely.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

# Environment variable for the API origin (no path, no trailing slash).
ENV_API_BASE = "GAUSSIANBLURR_API_URL"
DEFAULT_API_BASE = "http://127.0.0.1:8000"

# Wall-clock limits (seconds). Uploads are bounded I/O; agent runs include LLM + codegen.
UPLOAD_TIMEOUT = 120
AGENT_TIMEOUT = 600

_ALLOWED_UPLOAD_EXT = frozenset({".csv", ".xlsx"})


def default_base_url() -> str:
    """Resolve API origin from the environment with safe defaults."""
    return os.environ.get(ENV_API_BASE, DEFAULT_API_BASE).strip().rstrip("/")


def _mime_for_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return "text/csv"
    if ext == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def _error_message_from_response(resp: requests.Response) -> str:
    """
    Turn a non-2xx HTTP response into a single user-facing string.

    Handles typical FastAPI shapes: ``{"error": "..."}``, ``{"detail": ...}``,
    and plain text bodies.
    """
    try:
        body = resp.json()
        if isinstance(body, dict):
            if "error" in body:
                return str(body["error"])
            detail = body.get("detail")
            if detail is not None:
                if isinstance(detail, list):
                    parts = []
                    for item in detail:
                        if isinstance(item, dict):
                            parts.append(str(item.get("msg", item)))
                        else:
                            parts.append(str(item))
                    return "; ".join(parts) if parts else str(detail)
                return str(detail)
    except (ValueError, TypeError):
        pass
    text = (resp.text or "").strip()
    if text:
        return text[:500]
    return f"HTTP {resp.status_code}"


def _normalize_agent_json(data: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults so UI logic never assumes keys are present."""
    data.setdefault("session_id", None)
    data.setdefault("interrupted", False)
    data.setdefault("question", None)
    data.setdefault("summary", None)
    data.setdefault("last_tool_result", None)
    data.setdefault("codegen_requirement", None)
    return data


class GaussianBlurrApiClient:
    """
    Stateless wrapper around ``requests`` for the FastAPI GaussianBlurr endpoints.

    A shared :class:`requests.Session` keeps TCP keep-alive; responses default to JSON unless a method
    (e.g. :meth:`iter_run_stream`) sets ``Accept: text/event-stream`` per request.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base = (base_url or default_base_url()).strip().rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    @property
    def base_url(self) -> str:
        return self._base

    def artifact_url(self, session_id: str, relative_path: str) -> str:
        """
        Build a fully-qualified ``GET /artifact/...`` URL for an output file.

        Accepts either a logical path produced by ``code_pipeline``
        (e.g. ``"agent_filesystem/<session>/run_<run_id>/trend.png"``) or a
        bare session-relative path (e.g. ``"run_<run_id>/trend.png"``); the
        ``agent_filesystem/<session>/`` prefix is stripped if present so the
        result always lines up with the server route.
        """
        rel = (relative_path or "").lstrip("/")
        prefix = f"agent_filesystem/{session_id}/"
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
        return f"{self._base}/artifact/{session_id}/{rel}"

    def upload_data(self, file_bytes: bytes, filename: str, session_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        Save one CSV/XLSX into the session workspace.

        Parameters
            file_bytes   Raw file body.
            filename     Original name (basename is sent to the server).
            session_id   Same id the graph uses as ``thread_id``.

        Returns
            ``(logical_path, None)`` on success, where ``logical_path`` is the first
            entry in the API ``saved`` list (e.g. ``agent_filesystem/<session>/file.csv``).

            ``(None, error_message)`` on validation failure before the wire, transport
            error, HTTP error, or malformed success payload.
        """
        safe_name = Path(filename).name or "data"
        ext = Path(safe_name).suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXT:
            return None, f"Only .csv or .xlsx are allowed (got {ext or 'no extension'})."

        url = f"{self._base}/upload-data"
        try:
            resp = self._session.post(
                url,
                data={"session_id": session_id},
                files={"files": (safe_name, file_bytes, _mime_for_filename(safe_name))},
                timeout=UPLOAD_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            return None, "Cannot reach API — start uvicorn (port 8000) or set GAUSSIANBLURR_API_URL."
        except requests.exceptions.Timeout:
            return None, "Upload timed out."
        except OSError as exc:
            return None, str(exc)

        if resp.status_code == 200:
            try:
                payload = resp.json()
            except ValueError:
                return None, "Upload succeeded but response was not JSON."
            saved = payload.get("saved") or []
            if not saved:
                return None, "API returned no saved path."
            return str(saved[0]), None

        return None, _error_message_from_response(resp)

    def run(self, query: str, session_id: str) -> dict[str, Any]:
        """
        Run one orchestrator turn (``POST /run``).

        Returns a dict that always includes ``interrupted``, ``question``, ``summary``,
        ``last_tool_result``, ``codegen_requirement`` (dict or ``None`` when absent), and on failure ``error`` (str) instead of a successful payload.
        """
        url = f"{self._base}/run"
        try:
            resp = self._session.post(
                url,
                data={"query": query, "session_id": session_id},
                timeout=AGENT_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot reach API — start uvicorn or set GAUSSIANBLURR_API_URL."}
        except requests.exceptions.Timeout:
            return {"error": f"Request timed out after {AGENT_TIMEOUT}s. Try a smaller task."}
        except OSError as exc:
            return {"error": str(exc)}

        if resp.status_code != 200:
            return {"error": _error_message_from_response(resp)}
        try:
            data = resp.json()
        except ValueError:
            return {"error": (resp.text or "Invalid JSON from API")[:800]}
        if not isinstance(data, dict):
            return {"error": "Unexpected response shape from API."}
        return _normalize_agent_json(data)

    def resume(self, resume_value: str, session_id: str) -> dict[str, Any]:
        """
        Resume after ``ask_user`` (``POST /resume``).

        Same return contract as :meth:`run`.
        """
        url = f"{self._base}/resume"
        try:
            resp = self._session.post(
                url,
                data={"resume_value": resume_value, "session_id": session_id},
                timeout=AGENT_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot reach API — start uvicorn or set GAUSSIANBLURR_API_URL."}
        except requests.exceptions.Timeout:
            return {"error": f"Resume timed out after {AGENT_TIMEOUT}s."}
        except OSError as exc:
            return {"error": str(exc)}

        if resp.status_code != 200:
            return {"error": _error_message_from_response(resp)}
        try:
            data = resp.json()
        except ValueError:
            return {"error": (resp.text or "Invalid JSON from API")[:800]}
        if not isinstance(data, dict):
            return {"error": "Unexpected response shape from API."}
        return _normalize_agent_json(data)

    def iter_run_stream(self, query: str, session_id: str) -> Iterator[dict[str, Any]]:
        """
        Consume ``POST /run/stream`` (SSE), yielding decoded JSON payloads from each ``data:`` frame.

        Surface transport failures as ``ConnectionError``.
        """

        url = f"{self._base}/run/stream"
        try:
            with self._session.post(
                url,
                json={"session_id": session_id, "query": query},
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=AGENT_TIMEOUT,
            ) as response:
                response.raise_for_status()
                buffer = ""
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
                    if not chunk:
                        continue
                    buffer += chunk.decode("utf-8", errors="replace")
                    while True:
                        sep = buffer.find("\n\n")
                        if sep == -1:
                            break
                        frame = buffer[:sep]
                        buffer = buffer[sep + 2 :]
                        for raw_line in frame.split("\n"):
                            line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
                            if not line.startswith("data: "):
                                continue
                            payload = json.loads(line[6:])
                            yield payload
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError("Cannot reach API. Start uvicorn or set GAUSSIANBLURR_API_URL.") from exc

    def iter_resume_stream(self, resume_value: str, session_id: str) -> Iterator[dict[str, Any]]:
        """Same envelope contract as :meth:`iter_run_stream`, but resumes after ``interrupt``."""

        url = f"{self._base}/resume/stream"
        try:
            with self._session.post(
                url,
                json={"session_id": session_id, "resume_value": resume_value},
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=AGENT_TIMEOUT,
            ) as response:
                response.raise_for_status()
                buffer = ""
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
                    if not chunk:
                        continue
                    buffer += chunk.decode("utf-8", errors="replace")
                    while True:
                        sep = buffer.find("\n\n")
                        if sep == -1:
                            break
                        frame = buffer[:sep]
                        buffer = buffer[sep + 2 :]
                        for raw_line in frame.split("\n"):
                            line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
                            if not line.startswith("data: "):
                                continue
                            payload = json.loads(line[6:])
                            yield payload
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError("Cannot reach API. Start uvicorn or set GAUSSIANBLURR_API_URL.") from exc

