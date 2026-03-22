"""
Streamlit front-end for GaussianBlurr.

Calls the FastAPI ``POST /run`` endpoint (default ``http://localhost:8000``). Manages
``session_id``, CSV upload state, and chat history in ``st.session_state``. Layout uses
custom HTML/CSS for top bar, upload strip, scrollable messages, and input bar.
"""
import html
import uuid
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# Favicon path inside ui/ (same folder as this file)
_UI_DIR = Path(__file__).resolve().parent
_FAVICON = _UI_DIR / "gaussianblurr_favicon.png"

st.set_page_config(
    page_title="GaussianBlurr",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Reference-style CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f5f4f1 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Streamlit app wrapper — full height so content can fill */
section[data-testid="stAppViewContainer"],
section.main,
div[data-testid="stAppViewContainer"],
div.stApp,
.stApp > div {
    height: 100% !important;
    min-height: 100vh !important;
}

#MainMenu, footer, header { visibility: hidden; }
/* Slight left/right margin; full height flex column */
.block-container {
    padding: 0 24px !important;
    max-width: 100% !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
}
/* Middle (messages/empty) fills remaining height and scrolls */
.block-container > div:has(.fp-messages),
.block-container > div:has(.fp-empty) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
}
.fp-messages, .fp-empty { flex: 1 1 auto !important; min-height: 0 !important; }

/* Top bar — full width, no shrink */
.fp-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 24px; background: #fff;
    border-bottom: 1px solid #ece9e3;
    position: sticky; top: 0; z-index: 100;
    width: 100%; flex-shrink: 0;
}
.fp-logo { display: flex; align-items: center; gap: 10px; }
.fp-logo-mark {
    width: 30px; height: 30px; border-radius: 8px; background: #1a1a1a;
    display: flex; align-items: center; justify-content: center;
}
.fp-logo-name { font-size: 15px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.4px; }
.fp-logo-name span { color: #888; font-weight: 400; }
.fp-badge {
    font-size: 10px; font-weight: 500; letter-spacing: 0.4px;
    background: #f0efeb; color: #999; border-radius: 5px;
    padding: 2px 7px; border: 1px solid #e2e0db;
}
.fp-session {
    font-family: 'DM Mono', monospace; font-size: 11px; color: #aaa;
    display: flex; align-items: center; gap: 6px;
}
.fp-dot { width: 6px; height: 6px; border-radius: 50%; background: #4ade80; display: inline-block; }
.fp-btn-new {
    font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 500;
    padding: 6px 13px; border-radius: 7px;
    background: #fff; border: 1px solid #ddd; color: #555; cursor: pointer;
}
.fp-btn-new:hover { background: #f5f4f1; }

/* Upload strip — full width */
.fp-upload { margin: 14px 16px 0; padding: 11px 15px; border-radius: 10px; display: flex; align-items: center; gap: 12px; transition: all 0.18s; width: 100%; box-sizing: border-box; }
.fp-upload.idle { background: #fff; border: 1px dashed #d5d2cb; cursor: pointer; }
.fp-upload.idle:hover { border-color: #bbb; background: #fafaf8; }
.fp-upload.done { background: #f5fbf7; border: 1px solid #b6dfc8; }
.fp-upload-icon { width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.fp-upload-icon.idle { background: #f0efeb; border: 1px solid #e2e0db; }
.fp-upload-icon.done { background: #e8f5ee; border: 1px solid #b6dfc8; }
.fp-upload-title { font-size: 13px; font-weight: 500; }
.fp-upload-title.idle { color: #444; }
.fp-upload-title.done { color: #2d7a52; }
.fp-upload-sub { font-size: 11px; color: #aaa; margin-top: 2px; }

/* Messages — flex to fill space, scroll when needed */
.fp-messages { padding: 20px 16px 8px; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; flex: 1; }
.fp-msg { display: flex; gap: 10px; }
.fp-msg.user { justify-content: flex-end; }
.fp-avatar {
    width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600;
}
.fp-avatar.ai { background: #1a1a1a; }
.fp-avatar.user { background: #e8e6e1; color: #888; }
.fp-msg-name { font-size: 11px; color: #bbb; margin-bottom: 5px; font-weight: 500; }
.fp-bubble-ai { font-size: 13.5px; line-height: 1.7; color: #333; }
.fp-bubble-user {
    background: #fff; border: 1px solid #e8e6e1;
    border-radius: 10px; padding: 9px 13px;
    font-size: 13.5px; line-height: 1.65; color: #444;
    display: inline-block; max-width: 88%;
}
.fp-data-card {
    margin-top: 10px; background: #f8f7f4;
    border: 1px solid #e8e6e1; border-radius: 9px; padding: 12px 14px;
    font-family: 'DM Mono', monospace; font-size: 12px; color: #555; line-height: 1.9;
}
.fp-data-card .card-label { font-family: 'DM Sans', sans-serif; font-size: 11px; color: #aaa; margin-bottom: 6px; }
.fp-data-row { display: flex; justify-content: space-between; gap: 16px; }
.fp-data-row .k { color: #888; }
.fp-data-row .v { color: #1a1a1a; font-weight: 500; }
.fp-clarification { margin-top: 8px; padding: 8px 12px; background: #f0efeb; border-radius: 8px; font-size: 12px; color: #555; border: 1px solid #e2e0db; }
.fp-summary { margin-top: 8px; padding: 8px 12px; background: #e8f5ee; border-radius: 8px; font-size: 12px; color: #2d7a52; border: 1px solid #b6dfc8; }

/* Input bar — stays at bottom, no shrink */
.fp-inputbar { flex-shrink: 0; z-index: 100; padding: 10px 16px 16px; background: #fff; border-top: 1px solid #ece9e3; }
.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important; font-size: 13.5px !important;
    background: #f8f7f4 !important; border: 1px solid #ddd !important;
    border-radius: 11px !important; padding: 10px 14px !important;
    color: #333 !important; box-shadow: none !important;
}
.stTextInput > div > div > input:focus { border-color: #aaa !important; background: #fff !important; }
.stTextInput > label { display: none !important; }
.stButton > button {
    font-family: 'DM Sans', sans-serif !important; font-size: 12px !important; font-weight: 500 !important;
    background: #1a1a1a !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    padding: 8px 18px !important; transition: all 0.15s !important;
}
.stButton > button:hover { background: #333 !important; }

/* File uploader hidden — we use custom strip */
.stFileUploader { display: none; }
.fp-divider { height: 1px; background: #ece9e3; margin: 0 16px; }

/* Empty state — fills remaining height, content centered */
.fp-empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 8px; padding: 60px 24px; text-align: center;
    flex: 1; min-height: 200px;
}
.fp-empty-icon {
    width: 48px; height: 48px; border-radius: 13px;
    background: #f0efeb; border: 1px solid #e2e0db;
    display: flex; align-items: center; justify-content: center; margin-bottom: 4px;
}
.fp-empty h3 { font-size: 14px; font-weight: 500; color: #555; }
.fp-empty p { font-size: 12.5px; color: #aaa; max-width: 240px; line-height: 1.5; }
</style>
<script>
(function() {
  function setHeight() {
    var vh = window.innerHeight + 'px';
    document.documentElement.style.setProperty('--app-vh', vh);
    document.body.style.minHeight = vh;
    var app = document.querySelector('[data-testid="stAppViewContainer"]');
    var bc = document.querySelector('.block-container');
    if (app) { app.style.minHeight = vh; app.style.height = '100%'; }
    if (bc) { bc.style.minHeight = vh; bc.style.height = '100%'; }
  }
  setHeight();
  window.addEventListener('resize', setHeight);
  setTimeout(setHeight, 100);
  setTimeout(setHeight, 500);
})();
</script>
""", unsafe_allow_html=True)

# Logo SVG (Gaussian curve — same as reference)
LOGO_SVG = """
<svg width="18" height="12" viewBox="0 0 20 14" fill="none">
  <path d="M1 13 C4 13 4 1 7 1 C10 1 10 13 13 13 C16 13 16 1 19 1"
        stroke="white" stroke-width="2" stroke-linecap="round" fill="none"/>
</svg>
"""


def _ensure_session_id() -> None:
    """Assign a UUID ``session_id`` if the current session has none (first load)."""
    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())


def _new_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.csv_path = None
    st.session_state.messages = []
    st.session_state.uploaded_file = None
    st.rerun()


_ensure_session_id()
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False


# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fp-topbar">
  <div class="fp-logo">
    <div class="fp-logo-mark">{LOGO_SVG}</div>
    <span class="fp-logo-name">Gaussian<span>Blurr</span></span>
    <span class="fp-badge">ML Platform</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <div class="fp-session">
      <span class="fp-dot"></span>
      {st.session_state.session_id[:8]}...
    </div>
    <a href="#" class="fp-btn-new" style="text-decoration:none;color:inherit;" id="fp-new-session-link">+ New session</a>
  </div>
</div>
""", unsafe_allow_html=True)
# New session: link reloads page (new state/session_id); sidebar button clears and reruns
st.markdown("""
<script>
document.getElementById('fp-new-session-link').onclick = function() {
  window.location.href = window.location.pathname + window.location.search;
  return false;
};
</script>
""", unsafe_allow_html=True)
with st.sidebar:
    if st.button("+ New session", key="sidebar_new_session"):
        _new_session()


# ── Upload strip ─────────────────────────────────────────────────────────────
has_data = bool(st.session_state.uploaded_file or st.session_state.csv_path)
uploaded = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed", key="uploader")
if uploaded is not None and not has_data:
    st.session_state.uploaded_file = (uploaded.name, uploaded.getvalue())
    st.session_state.csv_path = None
    has_data = True

if not has_data:
    st.markdown("""
    <div class="fp-upload idle">
      <div class="fp-upload-icon idle">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 11V4M5.5 6.5L8 4l2.5 2.5" stroke="#888" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M2.5 12v.5a1 1 0 001 1h9a1 1 0 001-1V12" stroke="#888" stroke-width="1.3" stroke-linecap="round"/>
        </svg>
      </div>
      <div style="flex:1">
        <div class="fp-upload-title idle">+ Attach data file</div>
        <div class="fp-upload-sub">CSV — then send a message below to run</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    name = st.session_state.uploaded_file[0] if st.session_state.uploaded_file else (st.session_state.csv_path or "data").split("/")[-1]
    size = ""
    if st.session_state.uploaded_file:
        size = f"{round(len(st.session_state.uploaded_file[1]) / 1024, 1)} KB · ready"
    else:
        size = "attached"
    st.markdown(f"""
    <div class="fp-upload done">
      <div class="fp-upload-icon done">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M3 8.5l3 3 7-7" stroke="#2d7a52" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div style="flex:1">
        <div class="fp-upload-title done">{html.escape(name)}</div>
        <div class="fp-upload-sub">{html.escape(size)}</div>
      </div>
      <span style="font-size:12px;font-weight:500;padding:7px 15px;border-radius:7px;background:#2d7a52;color:#fff;">Attached ✓</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)


# ── Messages ─────────────────────────────────────────────────────────────────
def _render_messages() -> None:
    """Render chat history: empty-state placeholder or assistant/user bubbles and optional metadata."""
    if not st.session_state.messages:
        st.markdown("""
        <div class="fp-empty">
          <div class="fp-empty-icon">
            <svg width="22" height="16" viewBox="0 0 20 14" fill="none">
              <path d="M1 13 C4 13 4 1 7 1 C10 1 10 13 13 13 C16 13 16 1 19 1"
                    stroke="#bbb" stroke-width="1.8" stroke-linecap="round" fill="none"/>
            </svg>
          </div>
          <h3>Start with your data</h3>
          <p>Upload a file above, then ask anything — forecasts, trends, summaries.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown('<div class="fp-messages">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("content") or ""
        content_escaped = html.escape(content).replace("\n", "<br>")
        if role == "assistant":
            parts = [f'<div class="fp-bubble-ai">{content_escaped}</div>']
            # Avoid duplicate: show clarification/summary only if different from main content
            if msg.get("clarification") and msg.get("clarification") != msg.get("content"):
                parts.append(f'<div class="fp-clarification">{html.escape(msg["clarification"])}</div>')
            if msg.get("summary") and msg.get("summary") != msg.get("content"):
                parts.append(f'<div class="fp-summary">{html.escape(msg["summary"])}</div>')
            cols = msg.get("cols") or {}
            if msg.get("plot_saved_path") or cols.get("x_col") or cols.get("y_col"):
                rows = []
                if cols.get("x_col"):
                    rows.append(f'<div class="fp-data-row"><span class="k">X</span><span class="v">{html.escape(str(cols["x_col"]))}</span></div>')
                if cols.get("y_col"):
                    rows.append(f'<div class="fp-data-row"><span class="k">Y</span><span class="v">{html.escape(str(cols["y_col"]))}</span></div>')
                if cols.get("color_col"):
                    rows.append(f'<div class="fp-data-row"><span class="k">Color</span><span class="v">{html.escape(str(cols["color_col"]))}</span></div>')
                if msg.get("plot_saved_path"):
                    rows.append(f'<div class="fp-data-row"><span class="k">Saved</span><span class="v">{html.escape(msg["plot_saved_path"])}</span></div>')
                if rows:
                    parts.append(f'<div class="fp-data-card"><div class="card-label">Plot</div>{"".join(rows)}</div>')
            block = f"""
            <div class="fp-msg">
              <div class="fp-avatar ai">{LOGO_SVG}</div>
              <div>
                <div class="fp-msg-name">GaussianBlurr</div>
                {"".join(parts)}
              </div>
            </div>
            """
        else:
            block = f"""
            <div class="fp-msg user">
              <div style="display:flex;flex-direction:column;align-items:flex-end">
                <div class="fp-msg-name">You</div>
                <div class="fp-bubble-user">{content_escaped}</div>
              </div>
              <div class="fp-avatar user">U</div>
            </div>
            """
        st.markdown(block, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


_render_messages()


# ── Input bar ──────────────────────────────────────────────────────────────────
st.markdown('<div class="fp-inputbar">', unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Message",
        placeholder="Ask GaussianBlurr anything about your data...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send ↗", disabled=st.session_state.is_running)

if send and user_input and user_input.strip():
    if not st.session_state.uploaded_file and not st.session_state.csv_path:
        st.warning("Attach a CSV using the area above, then send your message.")
        st.stop()
    prompt = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder for assistant message (we'll replace with real response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": "",
        "clarification": None,
        "summary": None,
        "plot_saved_path": None,
        "cols": {},
    })
    idx = len(st.session_state.messages) - 1
    resp = None
    st.session_state.is_running = True
    with st.spinner("Thinking..."):
        try:
            if st.session_state.uploaded_file:
                name, data = st.session_state.uploaded_file
                resp = requests.post(
                    f"{API_URL}/run",
                    files={"file": (name, data, "text/csv")},
                    data={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=120,
                )
            else:
                resp = requests.post(
                    f"{API_URL}/run",
                    data={
                        "query": prompt,
                        "csv_path": st.session_state.csv_path,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )
        except requests.exceptions.Timeout:
            st.session_state.messages[idx] = {
                "role": "assistant",
                "content": "Request timed out (2 min). The agent may still be running; try again or use a simpler query.",
                "clarification": None,
                "summary": None,
                "plot_saved_path": None,
                "cols": {},
            }
            st.session_state.is_running = False
            st.rerun()
        except Exception as e:
            st.session_state.messages[idx] = {
                "role": "assistant",
                "content": str(e),
                "clarification": None,
                "summary": None,
                "plot_saved_path": None,
                "cols": {},
            }
            st.session_state.is_running = False
            st.rerun()
        finally:
            st.session_state.is_running = False

    if resp is None:
        st.rerun()
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text
        st.session_state.messages[idx] = {
            "role": "assistant",
            "content": err or "Request failed.",
            "clarification": None,
            "summary": None,
            "plot_saved_path": None,
            "cols": {},
        }
        st.rerun()

    data = resp.json()
    st.session_state.csv_path = data.get("csv_path") or st.session_state.csv_path
    # Only clear uploaded file after successful response and csv_path confirmed
    if resp.status_code == 200 and data.get("csv_path"):
        st.session_state.uploaded_file = None
    content = data.get("summary") or data.get("clarification_question") or ""
    if data.get("last_tool_result") and isinstance(data.get("last_tool_result"), str) and data["last_tool_result"].endswith(".html"):
        content = content or "Plot saved."
    if not content:
        content = "Done."
    tool_output = data.get("tool_output") or {}
    last_result = data.get("last_tool_result")
    plot_saved_path = last_result if isinstance(last_result, str) and last_result.endswith(".html") else None
    cols = {"x_col": tool_output.get("x_col"), "y_col": tool_output.get("y_col"), "color_col": tool_output.get("color_col")}
    st.session_state.messages[idx] = {
        "role": "assistant",
        "content": content,
        "clarification": data.get("clarification_question"),
        "summary": data.get("summary"),
        "plot_saved_path": plot_saved_path,
        "cols": cols,
    }
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
