"""
GaussianBlurr — Plotly Dash web UI.

Layout, callbacks, and rendering live here. All HTTP traffic to the FastAPI
backend goes through :class:`ui.api_client.GaussianBlurrApiClient` so this file
stays focused on UX and state.

Run (from repository root)::

    export PYTHONPATH="$(pwd)"   # optional if the block below runs first
    python ui/dash_app.py
"""

from __future__ import annotations

import base64
import sys
import io
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from dash import ALL, Dash, Input, Output, State, callback, callback_context, dcc, html
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate

# Ensure ``import ui.*`` resolves when launching ``python ui/dash_app.py`` without PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_rs = str(_PROJECT_ROOT)
if _rs not in sys.path:
    sys.path.insert(0, _rs)

from ui.api_client import GaussianBlurrApiClient

# ---------------------------------------------------------------------------
# Backend client (one instance per server process)
# ---------------------------------------------------------------------------

api = GaussianBlurrApiClient()


# ---------------------------------------------------------------------------
# Browser session state (serialized in dcc.Store)
# ---------------------------------------------------------------------------


def _default_store() -> dict[str, Any]:
    """Initial Dash store: new graph session id and empty chat/files."""
    return {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "uploaded_files": [],
        "active_file": None,
        "awaiting_resume": False,
        "pending_question": "",
    }


def _file_type_label(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {".csv": "CSV", ".xlsx": "XLSX"}.get(ext, "FILE")


def _read_df_from_bytes(raw: bytes, name: str) -> pd.DataFrame | None:
    """Parse upload bytes locally for instant preview (does not hit the API)."""
    try:
        low = name.lower()
        bio = io.BytesIO(raw)
        if low.endswith(".csv"):
            return pd.read_csv(bio)
        if low.endswith(".xlsx"):
            return pd.read_excel(bio)
    except Exception:
        return None
    return None


def _output_paths_from_tool(last_tool: str) -> list[str]:
    """Best-effort extraction of saved artifact paths from a tool message string."""
    out: list[str] = []
    if not last_tool:
        return out
    try:
        tool_json = json.loads(last_tool)
        if isinstance(tool_json, dict):
            for v in tool_json.values():
                if isinstance(v, str) and v.startswith("agent_filesystem/"):
                    low = v.lower()
                    if low.endswith("pipeline_run.py"):
                        continue
                    if low.endswith((".csv", ".xlsx", ".png", ".pdf", ".svg", ".jpg", ".jpeg")):
                        out.append(v)
    except Exception:
        pass
    if "agent_filesystem/" in last_tool:
        found = re.findall(
            r"agent_filesystem/[^/\s]+/[\w./\-]+\.(?:csv|xlsx|png|pdf|svg|jpg|jpeg)",
            last_tool,
            flags=re.IGNORECASE,
        )
        out.extend(p for p in found if not p.lower().endswith("pipeline_run.py"))
    return list(dict.fromkeys(out))


def _logo_circle(size: int = 28) -> html.Div:
    """Small circular mark in the sidebar and next to assistant bubbles (no image asset)."""
    return html.Div(
        "G",
        style={
            "width": f"{size}px",
            "height": f"{size}px",
            "borderRadius": "50%",
            "background": "#1a1a1a",
            "color": "#fff",
            "fontSize": f"{max(11, size - 14)}px",
            "fontWeight": 600,
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "flexShrink": 0,
            "fontFamily": "Georgia, serif",
        },
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _build_layout() -> html.Div:
    # Outer shell: exactly one viewport tall (padding inside height) — avoids body vertical scroll.
    return html.Div(
        className="gb-shell",
        style={
            "height": "100vh",
            "maxHeight": "100vh",
            "padding": "8px",
            "boxSizing": "border-box",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
            "background": "#DCDCD8",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "color": "#1a1a1a",
        },
        children=[
            dcc.Store(id="ui-store", data=_default_store()),
            html.Div(
                className="gb-frame",
                style={
                    "display": "flex",
                    "flex": 1,
                    "minHeight": 0,
                    "border": "1px solid #BFBFBB",
                    "borderRadius": "12px",
                    "overflow": "hidden",
                    "boxShadow": "0 4px 24px rgba(0, 0, 0, 0.08)",
                    "background": "#FAFAF8",
                },
                children=[
                    html.Div(
                        style={
                            "width": "268px",
                            "flexShrink": 0,
                            "background": "#fff",
                            "borderRight": "1px solid #E8E8E6",
                            "padding": "14px 12px 18px",
                            "display": "flex",
                            "flexDirection": "column",
                            "height": "100%",
                            "boxSizing": "border-box",
                        },
                        children=[
                            html.Div(
                                style={"marginBottom": "16px"},
                                children=[
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                        children=[
                                            _logo_circle(30),
                                            html.Span(
                                                ["Gaussian", html.Span("Blurr", style={"fontWeight": 400, "opacity": 0.45})],
                                                style={"fontSize": "15px", "fontWeight": 600, "letterSpacing": "-0.02em"},
                                            ),
                                        ],
                                    ),
                                    html.P(
                                        "Analyse your data with AI",
                                        style={
                                            "fontSize": "11px",
                                            "fontStyle": "italic",
                                            "color": "#9A9A97",
                                            "margin": "6px 0 0 0",
                                            "paddingLeft": "38px",
                                            "lineHeight": 1.35,
                                            "letterSpacing": "0.01em",
                                        },
                                    ),
                                ],
                            ),
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "gap": "8px",
                                        "fontSize": "14px",
                                        "fontWeight": 500,
                                        "color": "#333",
                                    },
                                    children=[
                                        html.Span("+", style={"fontSize": "18px", "fontWeight": 400, "lineHeight": 1}),
                                        "Upload CSV / XLSX",
                                    ],
                                ),
                                style={
                                    "border": "none",
                                    "borderRadius": "10px",
                                    "padding": "11px 12px",
                                    "textAlign": "center",
                                    "cursor": "pointer",
                                    "background": "#EDEDEB",
                                    "marginBottom": "8px",
                                },
                                multiple=True,
                            ),
                            html.Div(
                                id="upload-status",
                                style={"fontSize": "11px", "marginTop": "4px", "color": "#3B6D11", "minHeight": "16px"},
                            ),
                            html.P(
                                "LOADED FILES",
                                style={
                                    "fontSize": "10px",
                                    "letterSpacing": "0.06em",
                                    "color": "#9A9A97",
                                    "margin": "14px 0 8px",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Div(id="file-list", style={"flex": 1, "overflowY": "auto", "minHeight": "0"}),
                            html.Div(
                                id="session-line",
                                style={
                                    "fontSize": "10px",
                                    "fontFamily": "monospace",
                                    "color": "#C4C4C1",
                                    "marginTop": "8px",
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Button(
                                "+ New session",
                                id="btn-new-session",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "marginTop": "6px",
                                    "padding": "11px 12px",
                                    "borderRadius": "10px",
                                    "border": "1px solid #DCDCD9",
                                    "background": "#fff",
                                    "cursor": "pointer",
                                    "fontSize": "14px",
                                    "fontWeight": 500,
                                    "color": "#333",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "flex": 1,
                            "display": "flex",
                            "flexDirection": "column",
                            "minWidth": 0,
                            "minHeight": 0,
                            "height": "100%",
                            "background": "#FAFAF8",
                        },
                        children=[
                            html.Div(id="interrupt-banner", style={"flexShrink": 0, "padding": "10px 16px 0"}),
                            html.Div(id="preview-block", style={"flexShrink": 0, "padding": "0 16px"}),
                            html.Div(
                                id="chat-area",
                                style={
                                    "flex": 1,
                                    "overflowY": "auto",
                                    "padding": "10px 16px 14px",
                                    "minHeight": 0,
                                },
                            ),
                            html.Div(
                                style={
                                    "flexShrink": 0,
                                    "padding": "10px 16px 18px",
                                    "background": "#FAFAF8",
                                    "borderTop": "1px solid #ECECE9",
                                    "boxSizing": "border-box",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "10px",
                                            "width": "100%",
                                            "maxWidth": "100%",
                                            "boxSizing": "border-box",
                                        },
                                        children=[
                                            dcc.Input(
                                                id="chat-input",
                                                type="text",
                                                placeholder="Upload a file and ask anything...",
                                                debounce=False,
                                                n_submit=0,
                                                style={
                                                    "flex": 1,
                                                    "minWidth": 0,
                                                    "width": "100%",
                                                    "minHeight": "46px",
                                                    "padding": "11px 14px",
                                                    "borderRadius": "10px",
                                                    "border": "1px solid #DCDCD9",
                                                    "fontSize": "15px",
                                                    "background": "#fff",
                                                    "outline": "none",
                                                    "boxSizing": "border-box",
                                                },
                                            ),
                                            html.Button(
                                                "Send",
                                                id="btn-send",
                                                n_clicks=0,
                                                style={
                                                    "minHeight": "46px",
                                                    "padding": "0 20px",
                                                    "borderRadius": "10px",
                                                    "border": "none",
                                                    "background": "#2C2C2A",
                                                    "color": "#fff",
                                                    "cursor": "pointer",
                                                    "fontWeight": 600,
                                                    "fontSize": "15px",
                                                    "flexShrink": 0,
                                                    "boxSizing": "border-box",
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)
app.title = "GaussianBlurr"
app.layout = _build_layout


# ---------------------------------------------------------------------------
# Callbacks — uploads and graph session
# ---------------------------------------------------------------------------


@callback(
    Output("ui-store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("ui-store", "data"),
    prevent_initial_call=True,
)
def on_upload(contents_list, names_list, store):
    """Decode browser uploads, POST each file to ``/upload-data``, merge into store."""
    if not contents_list or not names_list or not store:
        raise PreventUpdate
    if not isinstance(contents_list, list):
        contents_list = [contents_list]
        names_list = [names_list]
    store = dict(store)
    files = list(store.get("uploaded_files") or [])
    status_parts: list[Any] = []
    for contents, name in zip(contents_list, names_list):
        if not contents or not name:
            continue
        try:
            _, b64 = contents.split(",", 1)
            raw = base64.b64decode(b64)
        except Exception:
            status_parts.append(html.Span(f"{name}: decode error. ", style={"color": "#B42318"}))
            continue
        saved, err = api.upload_data(raw, name, store["session_id"])
        if err or not saved:
            status_parts.append(
                html.Span(f"{Path(name).name}: {err or 'upload failed'}. ", style={"color": "#B42318"}),
            )
            continue
        if any(f.get("path") == saved for f in files):
            status_parts.append(html.Span(f"{Path(saved).name}: already in list. ", style={"color": "#8A8A87"}))
            continue
        display_name = Path(saved).name
        df = _read_df_from_bytes(raw, name)
        rows = len(df) if df is not None else 0
        cols = len(df.columns) if df is not None else 0
        preview = df.head(5).to_dict("records") if df is not None else []
        columns = [{"name": c, "id": c} for c in (df.columns.astype(str).tolist() if df is not None else [])]
        files.append(
            {
                "name": display_name,
                "path": saved,
                "rows": rows,
                "cols": cols,
                "preview_records": preview,
                "preview_columns": columns,
            }
        )
        if store.get("active_file") is None:
            store["active_file"] = len(files) - 1
        status_parts.append(html.Span(f"{display_name} ready. ", style={"color": "#3B6D11"}))
    store["uploaded_files"] = files
    status_el = html.Div(status_parts) if status_parts else ""
    return store, status_el


@callback(
    Output("ui-store", "data", allow_duplicate=True),
    Input("btn-new-session", "n_clicks"),
    State("ui-store", "data"),
    prevent_initial_call=True,
)
def on_new_session(n, _store):
    """Reset browser state (server-side checkpoints for the old id are left as-is)."""
    if not n:
        raise PreventUpdate
    return _default_store()


@callback(
    Output("ui-store", "data", allow_duplicate=True),
    Input({"type": "view-file", "index": ALL}, "n_clicks"),
    State("ui-store", "data"),
    prevent_initial_call=True,
)
def on_view_file(_n_clicks, store):
    """Mark which uploaded file is active (drives preview + optional path hint on send)."""
    if not store:
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".n_clicks")[0]
    try:
        btn_id = json.loads(trig)
        idx = int(btn_id["index"])
    except Exception:
        raise PreventUpdate
    store = dict(store)
    file_list = store.get("uploaded_files") or []
    if 0 <= idx < len(file_list):
        store["active_file"] = idx
    return store


# ---------------------------------------------------------------------------
# Callbacks — chat (orchestrator / resume)
# ---------------------------------------------------------------------------


@callback(
    Output("ui-store", "data", allow_duplicate=True),
    Output("chat-input", "value"),
    Input("btn-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("ui-store", "data"),
    prevent_initial_call=True,
)
def on_send(_n_clicks, _n_submit, text, store):
    """
    Append the user line, then call ``/run`` or ``/resume`` and append the assistant turn.

    When the graph is waiting on ``ask_user``, the same text box submits to ``/resume``.
    If ``/resume`` fails, we restore the interrupt state so the user can retry.
    """
    if not callback_context.triggered:
        raise PreventUpdate
    if not store or not text or not str(text).strip():
        raise PreventUpdate
    prompt = str(text).strip()
    store = dict(store)
    messages = list(store.get("messages") or [])

    agent_query = prompt
    if store.get("active_file") is not None and not store.get("awaiting_resume"):
        af = store["uploaded_files"][store["active_file"]]
        if af["name"] not in prompt and af["path"] not in prompt:
            agent_query = f"[Active file: {af['path']}]\n{prompt}"

    messages.append({"role": "user", "content": prompt})

    was_resume = bool(store.get("awaiting_resume"))
    pending_before = store.get("pending_question") or ""

    if was_resume:
        data = api.resume(prompt, store["session_id"])
    else:
        data = api.run(agent_query, store["session_id"])

    if data.get("interrupted"):
        store["awaiting_resume"] = True
        q = data.get("question") or "Please clarify."
        store["pending_question"] = q
        messages.append(
            {
                "role": "assistant",
                "content": f"I need a bit more information before I proceed:\n\n**{q}**",
            }
        )
        store["messages"] = messages
        return store, ""

    if data.get("error"):
        if was_resume:
            store["awaiting_resume"] = True
            store["pending_question"] = pending_before
        else:
            store["awaiting_resume"] = False
            store["pending_question"] = ""
        messages.append({"role": "assistant", "content": f"Something went wrong: {data['error']}"})
        store["messages"] = messages
        return store, ""

    store["awaiting_resume"] = False
    store["pending_question"] = ""
    summary = data.get("summary") or ""
    last_tool = data.get("last_tool_result") or ""
    outs = _output_paths_from_tool(last_tool)
    reply = summary if summary else "Done."
    messages.append({"role": "assistant", "content": reply, "output_files": outs})
    store["messages"] = messages
    return store, ""


# ---------------------------------------------------------------------------
# Render — derive HTML from store (no side effects)
# ---------------------------------------------------------------------------


def _meta_row(*, is_user: bool) -> html.Div:
    """Small footer inside a bubble (timestamp + read receipt for user)."""
    checks = html.Span("✓✓", style={"color": "#4CAF50", "fontSize": "12px", "marginLeft": "6px"}) if is_user else None
    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "flex-end",
            "alignItems": "center",
            "marginTop": "8px",
            "fontSize": "11px",
            "color": "#9A9A97",
        },
        children=[html.Span("Just now"), checks] if checks else [html.Span("Just now")],
    )


@callback(
    Output("session-line", "children"),
    Output("file-list", "children"),
    Output("preview-block", "children"),
    Output("chat-area", "children"),
    Output("interrupt-banner", "children"),
    Output("chat-input", "placeholder"),
    Input("ui-store", "data"),
)
def render_all(store):
    """Pure projection of ``ui-store`` into sidebar + main column components."""
    if not store:
        store = _default_store()
    sid = store.get("session_id", "")
    session_line = sid[:14] + "…" if len(sid) > 14 else sid

    files = store.get("uploaded_files") or []
    active = store.get("active_file")

    file_children: list = []
    if not files:
        file_children = [
            html.P(
                "No files yet.",
                style={"fontSize": "12px", "color": "#BDBDBA", "margin": "8px 0"},
            )
        ]
    else:
        for i, f in enumerate(files):
            is_act = active == i
            raw_name = f["name"]
            name_display = raw_name if len(raw_name) <= 22 else raw_name[:19] + "…"
            file_children.append(
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                        "padding": "8px 6px",
                        "borderRadius": "8px",
                        "marginBottom": "4px",
                        "background": "#F3F3F1" if is_act else "transparent",
                    },
                    children=[
                        html.Span("📄", style={"fontSize": "16px", "opacity": 0.85}),
                        html.Div(
                            name_display,
                            style={
                                "flex": 1,
                                "minWidth": 0,
                                "fontSize": "13px",
                                "fontWeight": 500,
                                "color": "#333",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                            },
                            title=raw_name,
                        ),
                        html.Button(
                            "···",
                            id={"type": "view-file", "index": i},
                            n_clicks=0,
                            title="Select file",
                            style={
                                "border": "none",
                                "background": "transparent",
                                "cursor": "pointer",
                                "fontSize": "18px",
                                "lineHeight": 1,
                                "color": "#9A9A97",
                                "padding": "0 4px",
                            },
                        ),
                    ],
                )
            )

    preview = html.Div()
    if active is not None and 0 <= active < len(files):
        af = files[active]
        recs = af.get("preview_records") or []
        cols = list(af.get("preview_columns") or [])
        if recs and not cols:
            cols = [{"name": str(k), "id": str(k)} for k in recs[0].keys()]
        if recs:
            preview = html.Div(
                style={
                    "marginTop": "6px",
                    "marginBottom": "6px",
                    "padding": "10px 12px",
                    "background": "#fff",
                    "border": "1px solid #E8E8E6",
                    "borderRadius": "12px",
                    "width": "100%",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                },
                children=[
                    html.Div(
                        style={
                            "fontSize": "12px",
                            "color": "#666",
                            "marginBottom": "8px",
                            "display": "flex",
                            "justifyContent": "space-between",
                            "flexWrap": "wrap",
                            "gap": "8px",
                        },
                        children=[
                            html.Span([html.Strong(af["name"]), f" · {af['rows']:,} rows · {af['cols']} cols"]),
                        ],
                    ),
                    DataTable(
                        data=recs,
                        columns=cols,
                        style_table={"overflowX": "auto", "maxHeight": "180px"},
                        style_cell={"fontSize": "12px", "padding": "6px 8px", "textAlign": "left", "border": "1px solid #f0f0ee"},
                        style_header={"fontWeight": 600, "background": "#F6F6F4", "border": "1px solid #E8E8E6"},
                    ),
                ],
            )

    msgs = store.get("messages") or []
    if not msgs:
        chat = html.Div(style={"minHeight": "120px"})
    else:
        blocks: list = []
        for m in msgs:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            extras = []
            for fp in m.get("output_files") or []:
                extras.append(
                    html.P(
                        f"Saved: {fp}",
                        style={"fontSize": "11px", "color": "#666", "margin": "8px 0 0"},
                    )
                )
            if role == "user":
                bubble = html.Div(
                    style={
                        "maxWidth": "560px",
                        "marginLeft": "auto",
                        "padding": "11px 14px 8px",
                        "borderRadius": "14px",
                        "background": "#E4F1E8",
                        "color": "#1a1a1a",
                        "fontSize": "14px",
                        "lineHeight": 1.5,
                        "boxShadow": "0 1px 0 rgba(0,0,0,0.04)",
                    },
                    children=[
                        html.Div(content, style={"textAlign": "left", "whiteSpace": "pre-wrap"}),
                        _meta_row(is_user=True),
                    ],
                )
                blocks.append(
                    html.Div(
                        style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"},
                        children=[bubble],
                    )
                )
            else:
                bubble = html.Div(
                    style={
                        "maxWidth": "640px",
                        "padding": "11px 14px 8px",
                        "borderRadius": "14px",
                        "background": "#fff",
                        "border": "1px solid #E8E8E6",
                        "color": "#1a1a1a",
                        "fontSize": "14px",
                        "lineHeight": 1.5,
                    },
                    children=[
                        dcc.Markdown(content, dangerously_allow_html=False, className="gb-md"),
                        *extras,
                        _meta_row(is_user=False),
                    ],
                )
                row = html.Div(
                    style={"display": "flex", "gap": "10px", "alignItems": "flex-start", "marginBottom": "12px"},
                    children=[_logo_circle(32), bubble],
                )
                blocks.append(row)
        chat = html.Div(
            style={"width": "100%", "maxWidth": "100%", "boxSizing": "border-box"},
            children=blocks,
        )

    if store.get("awaiting_resume") and store.get("pending_question"):
        banner = html.Div(
            style={
                "background": "#F0F4FF",
                "border": "1px solid #B5D4F4",
                "borderRadius": "10px",
                "padding": "11px 14px",
                "fontSize": "14px",
                "color": "#185FA5",
                "width": "100%",
                "maxWidth": "100%",
                "boxSizing": "border-box",
            },
            children=[
                html.Strong("Clarification needed: "),
                html.Span(store["pending_question"]),
            ],
        )
    else:
        banner = html.Div()

    af_hint = ""
    if active is not None and 0 <= active < len(files):
        af_hint = files[active]["name"]
    if store.get("awaiting_resume"):
        ph = "Answer the question above..."
    elif af_hint:
        ph = f"Ask about {af_hint}..."
    else:
        ph = "Upload a file and ask anything..."

    return session_line, file_children, preview, chat, banner, ph


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False)
