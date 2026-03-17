# GaussianBlurr — Forecasting Platform

Upload a CSV, then ask to **visualise**, **analyse**, or **forecast** → the agent orchestrates tools, summarises, or asks for clarification. One **session** per conversation; data and artifacts are stored under `data/<session_id>/`.

## Architecture

Built with **LangChain `create_agent`** — a graph-based agent loop with a composable middleware stack:

| # | Middleware | Purpose |
|---|-----------|---------|
| 1 | **SessionContextMiddleware** (custom) | Sets `session_id`, `csv_path`, `data_dir` as context vars so tools don't need session args |
| 2 | **LLMToolSelectorMiddleware** | Picks relevant tools per query; filesystem tools always included |
| 3 | **ToolCallLimitMiddleware** | Caps tool calls per run (default 15) to prevent runaway loops |
| 4 | **SummarizationMiddleware** | Condenses older history when tokens approach the limit |
| 5 | **ContextEditingMiddleware** | Clears old tool results, keeps the 3 most recent |
| 6 | **FilesystemMiddleware** (DeepAgents) | Gives the agent `ls`, `read_file`, `write_file`, `edit_file` tools for session-scoped storage |
| 7 | **ShellToolMiddleware** | Sandbox code execution (Docker preferred, host fallback) |
| 8 | **CodeSafetyMiddleware** (custom) | Inspects Python code before sandbox execution; only allows pandas/numpy/sklearn/plotly/matplotlib; auto-rewrites or rejects unsafe code |

**State**: `AgentState` (messages) + `session_id`, `csv_path`, `data_dir`.
**Persistence**: `InMemorySaver` checkpointer, `thread_id = session_id`.
**Observability**: Langfuse callbacks at graph + tool level.

## Run

**1. Install and configure**

```bash
pip install -r requirements.txt
```

Create or edit **`.env`** in the project root:

- `OPENAI_API_KEY=your-key` (required)
- `OPENAI_MODEL=gpt-4o-mini` (optional, default `gpt-4o-mini`)
- Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` for observability

**2. Start the app**

```bash
./launch.sh
```

This starts the API (uvicorn on http://127.0.0.1:8000) and the Streamlit UI. The script sets `PYTHONPATH` to the project root.

**Alternative — separate terminals**

```bash
# Terminal 1 — API
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
streamlit run ui/app.py
```

**3. Use the UI**

Open http://localhost:8501. Attach a CSV, then type a message. The agent decides whether to call tools (plot, analyse), ask for clarification, or respond directly. Use **+ New session** for a fresh conversation.

## Sessions

- **`thread_id = session_id`**: each conversation has its own checkpoint history (memory, tool results).
- Uploaded files: `data/<session_id>/source_files/`
- Generated plots: `data/<session_id>/plots/`
- Sandbox workspace: `data/<session_id>/filesystem/`

## Structure

```
graph/
  graph.py              — ForecastGraph: create_agent + middleware + run_graph()
  __init__.py
  middleware/
    session_context.py  — SessionContextMiddleware + context vars
    code_safety.py      — CodeSafetyMiddleware (AST-based Python safety check)
prompts/
  graph_prompts.py      — SYSTEM_PROMPT for the agent
tools/
  plot_tool.py          — plot_data tool (Plotly scatter, reads session context)
api/
  main.py               — FastAPI: POST /run → invoke agent → return output
ui/
  app.py                — Streamlit UI (GaussianBlurr style)
observability/
  langfuse_handler.py   — Langfuse callback setup
```

## API

- **`POST /run`** — Form: `query` (required), `file` (CSV, first message) or `csv_path` (follow-up), `session_id` (default `"default"`). Returns JSON: `session_id`, `csv_path`, `summary`, `clarification_question`, `tool_output`, `last_tool_result`.

## Code safety

The `CodeSafetyMiddleware` inspects Python code sent to the shell sandbox:

- **Allowed**: `pandas`, `numpy`, `sklearn`, `plotly`, `matplotlib`, `math`, `statistics`, `json`, `csv`, `io`, `datetime`, `collections`, `itertools`, `functools`, `re`, `typing`.
- **Blocked**: `os`, `sys`, `subprocess`, `shutil`, `socket`, `http`, `requests`, network libraries, `exec`, `eval`, `open`, file deletion methods.
- If unsafe: attempts auto-rewrite (strips blocked imports/calls). If still unsafe: rejects with explanation.

## Extending

Add new tools: define a `@tool` function in `tools/`, import it in `graph/graph.py`, add to the `tools` list in `_build_agent()`. The agent will discover it automatically via its tool-calling capability. Use context vars from `graph.middleware.session_context` if the tool needs session-scoped paths.
