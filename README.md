# Forecasting Platform

Upload a CSV, ask to **visualise** (or other queries) → the graph orchestrates tools (e.g. Plotly plot), summarises, or asks for clarification. One **session** per conversation; data and plots are stored under `data/<session_id>/`. The UI is styled as **GaussianBlurr** (full-height layout, slight side margins, chat-style interface).

## Run

**1. Install and configure**

```bash
pip install -r requirements.txt
```

Create or edit **`.env`** in the project root:

- `OPENAI_API_KEY=your-key` (required for the orchestration LLM)
- Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` for observability

The API loads `.env` on startup. `.env` is in `.gitignore`; do not commit it.

**2. Start the app**

From the **project root**:

```bash
./launch.sh
```

This starts the API (uvicorn on http://127.0.0.1:8000) and the Streamlit UI. The script sets `PYTHONPATH` to the project root and does not use `--reload` so imports resolve correctly.

**Alternative — separate terminals**

```bash
# Terminal 1 — API
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
streamlit run ui/app.py
```

Run both from the project root so the GaussianBlurr favicon (`gaussianblurr_favicon.png`) and paths resolve.

**3. Use the UI**

Open the Streamlit URL (e.g. http://localhost:8501). Use **+ Attach data file** to upload a CSV, then type a message (e.g. “Visualise sales by date” or “Summarise this dataset”). The app shows the summary, clarification questions, or plot info (saved path and columns). Use **+ New session** to start a fresh conversation (new session ID, cleared chat and data).

## Sessions

- Each conversation uses a unique **session ID** (created in the UI). The API keeps one `ForecastGraph` per `session_id`, so conversation and tool history are isolated per session.
- Uploaded files: `data/<session_id>/source_files/`
- Generated plots: `data/<session_id>/plots/`

## Structure

- **graph/** — `ForecastGraph` builds a LangGraph: **orchestration** → (tool | **summarisation** | **clarification**). The orchestration LLM decides whether to call a tool, summarise, or ask for clarification. `graph/tool_registry.yaml` describes tools for the LLM. `run_graph(session_id, user_query)` is the main entry; `csv_path` is set via `set_csv_path()` before each run.
- **memory/** — `ConversationalMemory` (bounded by `n_keep`) and `ToolMemory` store conversation turns and tool calls per graph instance (per session).
- **prompts/** — `ORCHESTRATION_PROMPT`, `SUMMARISATION_PROMPT` in `graph_prompts.py`.
- **tools/** — e.g. `plot_tool.py`: `plot_data(csv_path, x_col, y_col, color_col)` writes Plotly HTML under `data/<session_id>/plots/`. Session ID is injected by the graph; not passed by the LLM.
- **observability/** — Langfuse callback: when `LANGFUSE_*` env vars are set, graph runs are traced.
- **api/main.py** — Loads `.env`, then `POST /run`: accepts `query`, `file` (or `csv_path`), `session_id`; resolves CSV path, sets it on the graph, calls `graph.run_graph(session_id, query)`; returns `session_id`, `csv_path`, `summary`, `clarification_question`, `tool_output`, `last_tool_result`.
- **ui/app.py** — Streamlit UI (GaussianBlurr-style): topbar with session ID and New session, attach CSV strip, full-height message area with slight left/right margin, sticky input at bottom. Displays summary, clarification, and plot path/columns from `tool_output` and `last_tool_result`.

## API

- **`POST /run`** — Form: `query` (required), `file` (CSV upload, e.g. first message) or `csv_path` (follow-up), `session_id` (default `"default"`). Returns JSON: `session_id`, `csv_path`, `summary`, `clarification_question`, `tool_output`, `last_tool_result`. Plots are saved on disk; the response includes the path in `last_tool_result` and column info in `tool_output` when the plot tool ran.

Add more tools via `graph/tool_registry.yaml` and the graph’s tool registry in `graph/graph.py`.
