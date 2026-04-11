"""
GaussianBlurr — redesigned Streamlit UI.

Changes from original:
  - Sidebar: file uploader + loaded-files list with preview on click
  - Main area: data preview table (real columns + rows from backend) + chat
  - Interrupt / resume fully wired up
  - Session state tracks uploaded files, messages, active file
"""

import io
import json
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GaussianBlurr",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* tighten sidebar padding */
section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }
section[data-testid="stSidebar"] { background: #fafaf9; }

/* remove top padding from main block */
.block-container { padding-top: 1rem !important; }

/* file uploader compact */
[data-testid="stFileUploader"] section {
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
}

/* chat message bubbles */
[data-testid="stChatMessage"] {
    padding: 0.4rem 0.6rem;
    border-radius: 10px;
}

/* metric label size */
[data-testid="stMetricLabel"] p { font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { font-size: 1.1rem !important; }

/* dataframe header */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* divider spacing */
hr { margin: 0.5rem 0 !important; }

/* caption */
.file-caption { font-size: 0.7rem; color: #888; margin-top: 2px; }

/* active file pill */
.active-pill {
    display: inline-block;
    background: #EAF3DE;
    color: #3B6D11;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid #C0DD97;
    margin-left: 6px;
    vertical-align: middle;
}

/* question banner */
.question-banner {
    background: #F0F4FF;
    border: 1px solid #B5D4F4;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #185FA5;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "uploaded_files": [],   # list of {"name": str, "path": str, "rows": int, "cols": int, "df_head": df}
        "active_file": None,    # index into uploaded_files
        "awaiting_resume": False,
        "pending_question": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── helpers ───────────────────────────────────────────────────────────────────

def new_session():
    for k in ["session_id", "messages", "uploaded_files", "active_file",
              "awaiting_resume", "pending_question"]:
        del st.session_state[k]
    init_state()
    st.rerun()


def post_upload(file_obj, session_id: str) -> Optional[str]:
    """POST a file to /upload-data. Returns the saved path or None on error."""
    try:
        resp = requests.post(
            f"{API_URL}/upload-data",
            data={"session_id": session_id},
            files={"files": (file_obj.name, file_obj.getvalue(), file_obj.type or "application/octet-stream")},
            timeout=30,
        )
        if resp.status_code == 200:
            saved = resp.json().get("saved", [])
            return saved[0] if saved else None
        st.error(f"Upload failed: {resp.text}")
    except Exception as e:
        st.error(f"Upload error: {e}")
    return None


def read_uploaded_file(file_obj) -> Optional[pd.DataFrame]:
    """Read an uploaded file directly into a DataFrame for instant preview."""
    try:
        file_obj.seek(0)
        name = file_obj.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file_obj)
        if name.endswith(".xlsx"):
            return pd.read_excel(file_obj)
    except Exception:
        pass
    return None


def call_agent(query: str) -> dict:
    """POST /run and return the JSON response dict."""
    try:
        resp = requests.post(
            f"{API_URL}/run",
            data={"query": query, "session_id": st.session_state.session_id},
            timeout=180,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out (3 min). Try a simpler query."}
    except Exception as e:
        return {"error": str(e)}


def call_resume(answer: str) -> dict:
    """POST /resume to continue an interrupted graph."""
    try:
        resp = requests.post(
            f"{API_URL}/resume",
            data={"resume_value": answer, "session_id": st.session_state.session_id},
            timeout=180,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": str(e)}


def file_type_label(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {".csv": "CSV", ".xlsx": "XLSX",
            ".png": "PNG", ".jpg": "JPG", ".pdf": "PDF",
            ".py": "PY"}.get(ext, "FILE")


def file_type_color(label: str) -> str:
    return {"CSV": "#EAF3DE", "XLSX": "#E6F1FB",
            "PNG": "#FAEEDA", "JPG": "#FAEEDA", "PDF": "#FCEBEB",
            "PY": "#EEEDFE"}.get(label, "#F1EFE8")


def file_type_text_color(label: str) -> str:
    return {"CSV": "#3B6D11", "XLSX": "#185FA5",
            "PNG": "#633806", "JPG": "#633806", "PDF": "#791F1F",
            "PY": "#3C3489"}.get(label, "#5F5E5A")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Logo
    st.markdown("""
    <div style="display:flex;align-items:center;gap:9px;padding:14px 0 12px">
      <div style="width:26px;height:26px;border-radius:7px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <svg width="14" height="10" viewBox="0 0 20 14" fill="none">
          <path d="M1 13 C4 13 4 1 7 1 C10 1 10 13 13 13 C16 13 16 1 19 1"
                stroke="white" stroke-width="2" stroke-linecap="round" fill="none"/>
        </svg>
      </div>
      <span style="font-size:14px;font-weight:500;color:inherit;letter-spacing:-0.2px">
        Gaussian<span style="font-weight:400;opacity:0.5">Blurr</span>
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.caption("UPLOAD DATA")
    uploaded = st.file_uploader(
        "Upload",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader_widget",
    )

    if uploaded:
        for uf in uploaded:
            already = any(f["name"] == uf.name for f in st.session_state.uploaded_files)
            if not already:
                with st.spinner(f"Uploading {uf.name}…"):
                    saved_path = post_upload(uf, st.session_state.session_id)
                if saved_path:
                    df_head = read_uploaded_file(uf)
                    rows = len(df_head) if df_head is not None else 0
                    cols = len(df_head.columns) if df_head is not None else 0
                    st.session_state.uploaded_files.append({
                        "name": uf.name,
                        "path": saved_path,
                        "rows": rows,
                        "cols": cols,
                        "df_head": df_head,
                    })
                    if st.session_state.active_file is None:
                        st.session_state.active_file = 0
                    st.success(f"{uf.name} ready")
                    st.rerun()

    st.divider()

    # ── Loaded files ──────────────────────────────────────────────────────────
    st.caption("LOADED FILES")

    if not st.session_state.uploaded_files:
        st.markdown(
            "<p style='font-size:0.75rem;color:#aaa;padding:4px 0'>No files yet — upload a CSV or Excel file above.</p>",
            unsafe_allow_html=True,
        )
    else:
        for i, f in enumerate(st.session_state.uploaded_files):
            is_active = st.session_state.active_file == i
            label = file_type_label(f["name"])
            bg = file_type_color(label)
            fg = file_type_text_color(label)
            border = "2px solid #378ADD" if is_active else "1px solid transparent"
            col_icon, col_info, col_btn = st.columns([1, 4, 2])
            with col_icon:
                st.markdown(
                    f"<div style='width:26px;height:26px;border-radius:5px;background:{bg};"
                    f"color:{fg};font-size:9px;font-weight:500;display:flex;align-items:center;"
                    f"justify-content:center;margin-top:4px'>{label}</div>",
                    unsafe_allow_html=True,
                )
            with col_info:
                name_display = f["name"] if len(f["name"]) <= 18 else f["name"][:15] + "…"
                st.markdown(
                    f"<p style='font-size:0.75rem;font-weight:500;margin:4px 0 0;line-height:1.3'>{name_display}</p>"
                    f"<p style='font-size:0.68rem;color:#888;margin:0'>{f['rows']:,} rows · {f['cols']} cols</p>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("View", key=f"view_file_{i}", use_container_width=True):
                    st.session_state.active_file = i
                    st.rerun()

        st.markdown("")

    st.divider()

    # ── Session ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.68rem;font-family:monospace;color:#aaa;margin-bottom:6px'>"
        f"● {st.session_state.session_id[:12]}…</p>",
        unsafe_allow_html=True,
    )
    if st.button("+ New session", use_container_width=True):
        new_session()


# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# Header row
hcol1, hcol2 = st.columns([6, 1])
with hcol1:
    if st.session_state.active_file is not None:
        af = st.session_state.uploaded_files[st.session_state.active_file]
        st.markdown(
            f"<h3 style='font-size:1rem;font-weight:500;margin:0'>{af['name']}"
            f"<span class='active-pill'>active</span></h3>"
            f"<p style='font-size:0.72rem;color:#888;margin:2px 0 0'>{af['path']} · "
            f"{af['rows']:,} rows · {af['cols']} columns</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h3 style='font-size:1rem;font-weight:500;margin:0'>GaussianBlurr</h3>"
            "<p style='font-size:0.72rem;color:#888;margin:2px 0 0'>Upload a file to get started</p>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Data preview ──────────────────────────────────────────────────────────────
if st.session_state.active_file is not None:
    af = st.session_state.uploaded_files[st.session_state.active_file]
    df_head = af.get("df_head")

    if df_head is not None and not df_head.empty:
        with st.expander("Data preview", expanded=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{af['rows']:,}")
            m2.metric("Columns", af["cols"])
            m3.metric("File", file_type_label(af["name"]))
            st.dataframe(
                df_head.head(5),
                use_container_width=True,
                height=180,
            )

    st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        "<div style='text-align:center;padding:2rem 0;color:#aaa'>"
        "<p style='font-size:0.85rem'>Ask anything about your data</p>"
        "<p style='font-size:0.75rem'>GaussianBlurr can read, analyse, generate code, and forecast</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("content", "")
        with st.chat_message(role, avatar="📈" if role == "assistant" else "👤"):
            st.markdown(content)
            # Show result card if present
            if msg.get("result_card"):
                with st.container():
                    for k, v in msg["result_card"].items():
                        st.caption(f"**{k}** — {v}")
            # Show download hint for output files
            if msg.get("output_files"):
                for fp in msg["output_files"]:
                    st.caption(f"📄 Saved: `{fp}`")

# ── Interrupted — show clarifying question banner ────────────────────────────
if st.session_state.awaiting_resume and st.session_state.pending_question:
    st.markdown(
        f"<div class='question-banner'>🤔 <strong>GaussianBlurr needs clarification:</strong><br>{st.session_state.pending_question}</div>",
        unsafe_allow_html=True,
    )

# ── Chat input ────────────────────────────────────────────────────────────────
active_file_hint = ""
if st.session_state.active_file is not None:
    active_file_hint = st.session_state.uploaded_files[st.session_state.active_file]["name"]
placeholder = (
    f"Answer the question above…" if st.session_state.awaiting_resume
    else f"Ask about {active_file_hint}…" if active_file_hint
    else "Upload a file and ask anything…"
)

user_input = st.chat_input(placeholder)

if user_input and user_input.strip():
    prompt = user_input.strip()

    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # If active file context, prepend path hint so the agent knows which file
    agent_query = prompt
    if st.session_state.active_file is not None and not st.session_state.awaiting_resume:
        af = st.session_state.uploaded_files[st.session_state.active_file]
        # Only prepend if user hasn't mentioned the file already
        if af["name"] not in prompt and af["path"] not in prompt:
            agent_query = f"[Active file: {af['path']}]\n{prompt}"

    with st.chat_message("assistant", avatar="📈"):
        with st.spinner("Thinking…"):
            if st.session_state.awaiting_resume:
                data = call_resume(prompt)
                st.session_state.awaiting_resume = False
                st.session_state.pending_question = ""
            else:
                data = call_agent(agent_query)

    # Handle interrupt
    if data.get("interrupted"):
        st.session_state.awaiting_resume = True
        question = data.get("question", "Please clarify.")
        st.session_state.pending_question = question
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I need a bit more information before I proceed:\n\n**{question}**",
        })
        st.rerun()

    # Handle error
    if data.get("error"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Something went wrong: {data['error']}",
        })
        st.rerun()

    # Normal response
    summary = data.get("summary") or ""
    last_tool = data.get("last_tool_result") or ""

    # Extract saved data/plot paths (flat: agent_filesystem/<session>/<file>)
    output_files_mentioned = []
    try:
        tool_json = json.loads(last_tool) if last_tool else {}
        if isinstance(tool_json, dict):
            for v in tool_json.values():
                if isinstance(v, str) and v.startswith("agent_filesystem/"):
                    low = v.lower()
                    if low.endswith("pipeline_run.py"):
                        continue
                    if low.endswith((".csv", ".xlsx", ".png", ".pdf", ".svg", ".jpg", ".jpeg")):
                        output_files_mentioned.append(v)
    except Exception:
        pass

    if last_tool and "agent_filesystem/" in last_tool:
        import re
        found = re.findall(
            r"agent_filesystem/[^/\s]+/[\w./\-]+\.(?:csv|xlsx|png|pdf|svg|jpg|jpeg)",
            last_tool,
            flags=re.IGNORECASE,
        )
        output_files_mentioned.extend(
            p for p in found if not p.lower().endswith("pipeline_run.py")
        )
    output_files_mentioned = list(dict.fromkeys(output_files_mentioned))  # dedupe

    reply_content = summary if summary else ("Done." if not data.get("error") else data["error"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply_content,
        "output_files": output_files_mentioned,
    })

    st.rerun()
