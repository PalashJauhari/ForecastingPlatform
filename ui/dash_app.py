"""
GaussianBlurr — Plotly Dash web UI.

Layout and rendering live here. Chat turns ``POST`` to FastAPI SSE endpoints ``/run/stream`` and
``/resume/stream`` from the browser (see ``assets/gb_stream_ui.js``); uploads still use :class:`~ui.api_client.GaussianBlurrApiClient`.

Run (from repository root)::

    export PYTHONPATH="$(pwd)"   # optional if the block below runs first
    python ui/dash_app.py
"""

from __future__ import annotations

import base64
import sys
import io
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from dash import ALL, ClientsideFunction, Dash, Input, Output, State, callback, callback_context, dcc, html
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
        "awaiting_resume": False,
        "pending_question": "",
        "agent_thinking": False,
    }


def _upload_data_component() -> dcc.Upload:
    """CSV/XLSX upload control (remounted on new session so the browser clears stale file picks)."""
    return dcc.Upload(
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
            "padding": "10px 11px",
            "textAlign": "center",
            "cursor": "pointer",
            "background": "#EDEDEB",
            "marginBottom": "8px",
        },
        multiple=True,
    )


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


_LOGO_FILENAME = "gaussianblurr_favicon.png"


def _favicon_logo_img(size: int = 30) -> html.Img:
    """Sidebar + assistant row mark from ``assets/gaussianblurr_favicon.png`` (same as browser tab icon)."""
    px = f"{size}px"
    return html.Img(
        src=f"/assets/{_LOGO_FILENAME}",
        alt="GaussianBlurr",
        width=size,
        height=size,
        style={
            "width": px,
            "height": px,
            "borderRadius": "8px",
            "objectFit": "cover",
            "flexShrink": 0,
            "display": "block",
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
            "padding": "6px",
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
            dcc.Store(id="upload-gen", data=0),
            dcc.Store(id="api-base-url", data=api.base_url),
            html.Div(id="gb-clientside-dummy", style={"display": "none"}),
            dcc.Store(id="preview-dlg", data={"open": False, "i": None}),
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
                            "padding": "12px 10px 14px",
                            "display": "flex",
                            "flexDirection": "column",
                            "height": "100%",
                            "boxSizing": "border-box",
                        },
                        children=[
                            html.Div(
                                style={"marginBottom": "14px"},
                                children=[
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                        children=[
                                            _favicon_logo_img(30),
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
                                            "paddingLeft": "36px",
                                            "lineHeight": 1.35,
                                            "letterSpacing": "0.01em",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(id="upload-wrapper"),
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
                                    "margin": "12px 0 6px",
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
                                type="button",
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
                            html.Div(id="interrupt-banner", style={"flexShrink": 0, "padding": "8px 12px 0"}),
                            html.Div(
                                id="chat-area",
                                style={
                                    "flex": 1,
                                    "overflowY": "auto",
                                    "padding": "8px 12px 12px",
                                    "minHeight": 0,
                                },
                            ),
                            html.Div(
                                style={
                                    "flexShrink": 0,
                                    "padding": "8px 12px 14px",
                                    "background": "#FAFAF8",
                                    "borderTop": "1px solid #ECECE9",
                                    "boxSizing": "border-box",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "8px",
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
                                                    "minHeight": "44px",
                                                    "padding": "10px 12px",
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
                                                type="button",
                                                n_clicks=0,
                                                style={
                                                    "minHeight": "44px",
                                                    "padding": "0 18px",
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
                    html.Div(
                        className="gb-progress-col",
                        children=[
                            html.Div(className="gb-progress-head", children=[html.H3("Progress")]),
                            html.Div(id="gb-stream-progress"),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="preview-overlay",
                className="gb-preview-overlay",
                children=[
                    html.Div(
                        style={
                            "width": "100%",
                            "maxWidth": "min(920px, 96vw)",
                            "maxHeight": "88vh",
                            "overflow": "auto",
                            "background": "#FAFAF8",
                            "borderRadius": "14px",
                            "border": "1px solid #DCDCD9",
                            "boxShadow": "0 16px 48px rgba(0, 0, 0, 0.18)",
                            "padding": "16px 18px 18px",
                            "boxSizing": "border-box",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                    "gap": "12px",
                                    "marginBottom": "12px",
                                },
                                children=[
                                    html.Span("Preview", style={"fontSize": "15px", "fontWeight": 600}),
                                    html.Button(
                                        "×",
                                        id="preview-close",
                                        type="button",
                                        n_clicks=0,
                                        title="Close",
                                        style={
                                            "border": "none",
                                            "background": "transparent",
                                            "cursor": "pointer",
                                            "fontSize": "22px",
                                            "lineHeight": 1,
                                            "color": "#666",
                                            "padding": "0 4px",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(id="preview-modal-body"),
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
_tab_icon = app.get_asset_url(_LOGO_FILENAME)
_favicon_link = f'<link rel="icon" type="image/png" href="{_tab_icon}" sizes="any" />'
if "{%favicon%}" in app.index_string:
    app.index_string = app.index_string.replace("{%favicon%}", _favicon_link)
else:
    _head_i = app.index_string.find("</head>")
    if _head_i != -1:
        app.index_string = app.index_string[:_head_i] + _favicon_link + "\n    " + app.index_string[_head_i:]
app.layout = _build_layout


# ---------------------------------------------------------------------------
# Callbacks — uploads and graph session
# ---------------------------------------------------------------------------


@callback(
    Output("upload-wrapper", "children"),
    Input("upload-gen", "data"),
    prevent_initial_call=False,
)
def _mount_upload_widget(_upload_gen: int):
    return [_upload_data_component()]


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
        status_parts.append(html.Span(f"{display_name} ready. ", style={"color": "#3B6D11"}))
    store["uploaded_files"] = files
    status_el = html.Div(status_parts) if status_parts else ""
    return store, status_el


@callback(
    Output("ui-store", "data", allow_duplicate=True),
    Output("upload-gen", "data"),
    Output("upload-status", "children", allow_duplicate=True),
    Output("chat-input", "value", allow_duplicate=True),
    Output("preview-dlg", "data", allow_duplicate=True),
    Input("btn-new-session", "n_clicks"),
    State("upload-gen", "data"),
    prevent_initial_call=True,
)
def on_new_session(n, upload_gen):
    """New graph session id, empty chat/files, and remount upload so file picks do not carry over."""
    if not n:
        raise PreventUpdate
    return _default_store(), int(upload_gen or 0) + 1, "", "", {"open": False, "i": None}


@callback(
    Output("preview-dlg", "data"),
    Input({"type": "preview-file", "index": ALL}, "n_clicks"),
    State("ui-store", "data"),
    prevent_initial_call=True,
)
def open_file_preview(_n_clicks, store):
    """Open the preview dialog for the clicked file (does not change what is sent to ``/run``)."""
    if not store:
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    t0 = ctx.triggered[0]
    prop_id = str(t0.get("prop_id") or "")
    if not prop_id.endswith("n_clicks"):
        raise PreventUpdate
    # Re-renders reset ``n_clicks`` to 0 and can spuriously fire this callback — require a real click.
    n = t0.get("value")
    if n is None or int(n) < 1:
        raise PreventUpdate
    tid = ctx.triggered_id
    if not isinstance(tid, dict) or tid.get("type") != "preview-file":
        raise PreventUpdate
    idx = int(tid["index"])
    files = store.get("uploaded_files") or []
    if not (0 <= idx < len(files)):
        raise PreventUpdate
    return {"open": True, "i": idx}


@callback(
    Output("preview-dlg", "data", allow_duplicate=True),
    Input("preview-close", "n_clicks"),
    prevent_initial_call=True,
)
def close_file_preview(_n):
    if not _n:
        raise PreventUpdate
    return {"open": False, "i": None}


@callback(
    Output("preview-overlay", "style"),
    Output("preview-modal-body", "children"),
    Input("preview-dlg", "data"),
    State("ui-store", "data"),
    prevent_initial_call=False,
)
def sync_preview_modal(dlg, store):
    """Show or hide the preview overlay and fill the table from ``ui-store``."""
    base_style: dict[str, Any] = {
        "position": "fixed",
        "inset": "0",
        "zIndex": "2000",
        "backgroundColor": "rgba(0, 0, 0, 0.45)",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "20px",
        "boxSizing": "border-box",
    }
    if not dlg or not dlg.get("open") or dlg.get("i") is None:
        return {**base_style, "display": "none"}, []
    if not store:
        return {**base_style, "display": "none"}, []
    files = store.get("uploaded_files") or []
    idx = int(dlg["i"])
    if not (0 <= idx < len(files)):
        return {**base_style, "display": "none"}, []
    af = files[idx]
    recs = af.get("preview_records") or []
    cols = list(af.get("preview_columns") or [])
    if recs and not cols:
        cols = [{"name": str(k), "id": str(k)} for k in recs[0].keys()]
    if not recs:
        inner = html.P("No preview rows available.", style={"color": "#888", "margin": "8px 0"})
    else:
        inner = html.Div(
            children=[
                html.Div(
                    style={"fontSize": "12px", "color": "#555", "marginBottom": "10px"},
                    children=[
                        html.Strong(af["name"]),
                        html.Span(f" · {af['rows']:,} rows · {af['cols']} cols"),
                    ],
                ),
                DataTable(
                    id="preview-dl-table",
                    data=recs,
                    columns=cols,
                    style_table={"overflowX": "auto", "maxHeight": "min(52vh, 420px)"},
                    style_cell={"fontSize": "12px", "padding": "6px 8px", "textAlign": "left", "border": "1px solid #f0f0ee"},
                    style_header={"fontWeight": 600, "background": "#F6F6F4", "border": "1px solid #E8E8E6"},
                ),
            ],
        )
    return {**base_style, "display": "flex"}, [inner]


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Callbacks — chat (streaming SSE clientside)
# ---------------------------------------------------------------------------

app.clientside_callback(
    ClientsideFunction(namespace="gb_stream_ui", function_name="clear_stream_progress"),
    Output("gb-clientside-dummy", "children"),
    Input("upload-gen", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    ClientsideFunction(namespace="gb_stream_ui", function_name="submit_message_stream"),
    Output("ui-store", "data", allow_duplicate=True),
    Output("chat-input", "value", allow_duplicate=True),
    Input("btn-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("ui-store", "data"),
    State("api-base-url", "data"),
    prevent_initial_call=True,
)


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
            raw_name = f["name"]
            name_display = raw_name if len(raw_name) <= 22 else raw_name[:19] + "…"
            file_children.append(
                html.Button(
                    [
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
                                "textAlign": "left",
                            },
                            title=raw_name,
                        ),
                        html.Span(
                            "Preview",
                            style={
                                "fontSize": "10px",
                                "letterSpacing": "0.06em",
                                "textTransform": "uppercase",
                                "color": "#9A9A97",
                                "fontWeight": 600,
                                "flexShrink": 0,
                            },
                        ),
                    ],
                    id={"type": "preview-file", "index": i},
                    type="button",
                    n_clicks=0,
                    title="Click to preview in a dialog",
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                        "width": "100%",
                        "padding": "8px 8px",
                        "borderRadius": "8px",
                        "marginBottom": "4px",
                        "border": "1px solid #E8E8E6",
                        "background": "#fff",
                        "cursor": "pointer",
                        "boxSizing": "border-box",
                    },
                )
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
            # Render image artifacts attached to this assistant turn. ``output_images``
            # is the list returned by the API — logical paths to .png / .svg files
            # under ``agent_filesystem/<session>/run_<tool_call_id>/`` that were
            # produced during the same turn that generated this reply. Each one is
            # streamed back through the ``/artifact/{session_id}/{path}`` endpoint.
            for fp in m.get("output_images") or []:
                src = api.artifact_url(sid, fp)
                extras.append(
                    html.A(
                        html.Img(
                            src=src,
                            alt=fp.rsplit("/", 1)[-1],
                            style={
                                "display": "block",
                                "maxWidth": "100%",
                                "maxHeight": "420px",
                                "borderRadius": "8px",
                                "border": "1px solid #E8E8E6",
                                "marginTop": "8px",
                                "background": "#fff",
                            },
                        ),
                        href=src,
                        target="_blank",
                        title="Open full size in a new tab",
                        style={"textDecoration": "none"},
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
                        style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "10px"},
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
                    style={"display": "flex", "gap": "8px", "alignItems": "flex-start", "marginBottom": "10px"},
                    children=[_favicon_logo_img(32), bubble],
                )
                blocks.append(row)
        chat_children: list = list(blocks)
        chat = html.Div(
            style={"width": "100%", "maxWidth": "100%", "boxSizing": "border-box"},
            children=chat_children,
        )

    if store.get("awaiting_resume") and store.get("pending_question"):
        banner = html.Div(
            style={
                "background": "#F0F4FF",
                "border": "1px solid #B5D4F4",
                "borderRadius": "10px",
                "padding": "9px 12px",
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

    if store.get("awaiting_resume"):
        ph = "Answer the question above..."
    elif files:
        ph = "Ask about your data..."
    else:
        ph = "Upload a file and ask anything..."

    return session_line, file_children, chat, banner, ph


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False)
