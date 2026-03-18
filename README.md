# GaussianBlurr — Forecasting Platform

Upload a CSV, then **explore**, **clean**, and **analyse** your data. The orchestrator agent uses dedicated tools (peek, clean, check_ready, ask) and can ask for clarification. One **session** per conversation; data is stored under `data/<session_id>/`.

## Architecture

Built with **LangChain `create_agent`** — a single agent loop with middleware and session-scoped tools:

| # | Middleware | Purpose |
|---|-----------|---------|
| 1 | **SessionContextMiddleware** (custom) | Sets `session_id`, `csv_path`, `data_dir` as context vars so tools don't need session args |
| 2 | **LLMToolSelectorMiddleware** | Picks relevant tools per query; `peek_csv`, `clean_csv`, `check_ready` always included |
| 3 | **ToolCallLimitMiddleware** | Caps tool calls per run (default 15) |
| 4 | **SummarizationMiddleware** | Condenses older history when tokens grow |
| 5 | **ContextEditingMiddleware** | Clears old tool results, keeps recent ones (excludes `write_file`) |
| 6 | **FilesystemMiddleware** (DeepAgents) | `ls`, `read_file`, `write_file`, `edit_file` for session-scoped notes and outputs |

**Tools** (no shell/sandbox in the graph):

- **peek_csv** — Inspect session CSV (schema, dtypes, nulls, sample). No args; call first on new file.
- **clean_csv** — Clean the CSV via an internal LLM; writes to `processed/clean.csv`. Optional `user_instructions`. Returns JSON with `cleaned`, `path`, `error`, and on success a **snapshot** (columns, dtypes, sample_rows, total_rows).
- **check_ready** — Validate `processed/clean.csv` for forecasting (date column, numeric target, no nulls in key cols). No args.
- **ask_csv** — Answer a natural-language question about the data; internal LLM generates and runs pandas code. Argument: `query`.

Code generation in `clean_csv` and `ask_csv` uses **tools/safety.py** (AST-based): only pandas, numpy, sklearn, scipy, plotly, matplotlib; no `open`, `exec`, `eval`, or file-deletion calls.

**State**: `AgentState` (messages) + `session_id`, `csv_path`, `data_dir`.  
**Persistence**: `InMemorySaver` checkpointer, `thread_id = session_id`.  
**Observability**: Langfuse callbacks at graph level (when configured).

## Run

**1. Install and configure**

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`) — orchestrator model
- `CODING_MODEL` (optional, default `gpt-4o-mini`) — model used by `clean_csv` and `ask_csv` for code generation
- Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` for observability

**2. Start the app**

```bash
./launch.sh
```

Starts the API (http://127.0.0.1:8000) and the Streamlit UI. The script sets `PYTHONPATH` to the project root.

**Alternative — separate terminals**

```bash
# Terminal 1 — API
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
streamlit run ui/app.py
```

**3. Use the UI**

Open http://localhost:8501. Attach a CSV (first message), then ask to peek, clean, check readiness, or ask questions about the data. Use **+ New session** for a fresh conversation.

## Sessions

- **`thread_id = session_id`**: each conversation has its own checkpoint history.
- Uploaded files: `data/<session_id>/source_files/`
- Cleaned output: `data/<session_id>/processed/clean.csv` (canonical dataset for `ask_csv` and downstream use)
- Filesystem (DeepAgents): session-scoped `ls` / read / write / edit

## Structure

```
graph/
  graph.py              — AnalysisGraph: create_agent + middleware + run_graph()
  __init__.py
  middleware/
    session_context.py  — SessionContextMiddleware + context vars
prompts/
  graph_prompts.py      — SYSTEM_PROMPT for the orchestrator
  cleaning_prompt.py    — CLEANING_CODE_GENERATOR_PROMPT (clean_csv internal LLM)
  ask_csv_prompt.py     — ASK_CSV_CODE_GENERATOR_PROMPT (ask_csv internal LLM)
tools/
  peek_csv.py           — Peek session CSV (no args)
  clean_csv.py          — Clean CSV → processed/clean.csv; optional user_instructions; returns snapshot
  check_ready.py        — Validate clean.csv for forecasting (no args)
  ask_csv.py            — Natural-language query → pandas snippet run; argument: query
  safety.py             — AST-based safety check for generated code (used by clean_csv, ask_csv)
api/
  main.py               — FastAPI: POST /run → invoke agent → return output
ui/
  app.py                — Streamlit UI (GaussianBlurr style)
observability/
  langfuse_handler.py   — Langfuse callback setup
```

## API

- **`POST /run`** — Form: `query` (required), `file` (CSV, first message) or `csv_path` (follow-up), `session_id` (default `"default"`). Returns JSON: `session_id`, `csv_path`, `summary`, `clarification_question`, `tool_output`, `last_tool_result`. Tool results may be JSON (e.g. `clean_csv` with `cleaned`, `path`, `snapshot`) or plain text (`ask_csv` stdout).

## Code safety

Generated code in `clean_csv` and `ask_csv` is checked by **tools/safety.py** before execution:

- **Allowed**: `pandas`, `numpy`, `sklearn`, `scipy`, `plotly`, `matplotlib`.
- **Blocked**: `open`, `exec`, `eval`, `compile`, `__import__`, and attribute calls like `system`, `popen`, `remove`, `unlink`, `rmdir`, `rmtree`, `chmod`, `chown`.
- Code runs in a subprocess with a fixed preamble (imports + `input_path`/`output_path` or `df`); no shell middleware.

## Extending

- **New tools**: add a `@tool` (and optional Pydantic `args_schema`) in `tools/`, import in `graph/graph.py`, and add to the `tools` list in `_build_agent()`. Use `session_id_var`, `csv_path_var`, `data_dir_var` from `graph.middleware.session_context` for session-scoped paths.
- **Code-generation model**: set `CODING_MODEL` in `.env` to use a different model for `clean_csv` and `ask_csv` (e.g. `gpt-4o`).
