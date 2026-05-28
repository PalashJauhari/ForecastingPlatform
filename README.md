# GaussianBlurr — Agentic Forecasting Platform

Upload tabular data, ask questions in plain language, and get forecasts, analysis, and charts back in a streaming chat UI.

## Why this project

- **Agentic workflow** — The platform plans tasks, runs tools, and tracks progress automatically instead of relying on a single static prompt.
- **Session workspace** — Each conversation has its own folder for uploads and generated CSV/XLSX outputs.
- **Deterministic forecasting** — Built-in SARIMA, Prophet, and Holt-Winters tools for validated model fits when you want structured forecasts without custom code.
- **Safe custom analysis** — When code is needed, it is generated, scanned, reviewed, and executed in an isolated cloud sandbox (E2B).
- **Streaming UI** — Dash chat with live progress and inline plot rendering.

## Features

- Natural-language Q&A over uploaded CSV/XLSX files
- Automatic data profiling (columns, samples, basic stats)
- Todo planning at the start of each user turn
- Custom Python analysis and charts via the coding tool
- SARIMA / Prophet / Holt-Winters forecasting with interpretable JSON results
- FastAPI backend with SSE streaming; Plotly Dash frontend

## How it works

```text
Dash UI  →  FastAPI  →  LangGraph agent  →  tools (forecast / code / todos)
                              ↓
                    artifacts in session workspace  →  plots inline in chat
```

On each new message the agent profiles your files, summarizes long context, builds a todo list, then orchestrates tools until the task is done.

## Quickstart

1. **Clone and install**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**

   - Copy `.env.example` to `.env` at the repo root (OpenAI key, optional Postgres checkpoint URL, optional Langfuse keys when tracing is enabled).
   - Copy sub-agent templates:
     - `sub_agents/planner_sub_agent/.env.example` → `.env`
     - `sub_agents/coding_sub_agent/.env.example` → `.env`
   - Set `OPENAI_API_KEY` in all three; set `E2B_API_KEY` and `E2B_TEMPLATE_NAME` for cloud code execution (run `e2b/build_e2b_template.py` first).

3. **Run**

   ```bash
   ./launch.sh
   ```

   - API: http://127.0.0.1:8000  
   - UI: http://127.0.0.1:8501  

4. Upload a CSV in the sidebar, ask a forecasting or analysis question, and watch progress stream in the chat.

## Example workflow

1. Upload `sales.csv`.
2. Ask: *“Forecast the next 12 months of revenue and show a trend chart.”*
3. The agent profiles the file, plans steps, may run Prophet or SARIMA for the table (forecast + fitted CSVs at session root), and uses the coding tool for charts from those outputs.
4. Tables stay in your session workspace; plots appear inline in the assistant message.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload-data` | Upload CSV/XLSX into a session |
| POST | `/run` | Run agent (blocking JSON) |
| POST | `/run/stream` | Run agent (SSE progress + done payload) |
| POST | `/resume` | Resume after interrupt |
| POST | `/resume/stream` | Resume with SSE |
| GET | `/artifact/{session_id}/{path}` | Serve session files and plot images |

## Code execution and safety

Custom analysis runs through a dedicated coding pipeline:

1. Code is generated for your requirements and declared input/output filenames.
2. Static rules (Semgrep) and LLM judges check safety and file allowlists.
3. Approved scripts run in a fresh E2B cloud sandbox per `E2BExecute` attempt (custom template with data-science libraries, no internet); gate failures retry CodeGen until `CODING_RECURSION_LIMIT` (see coding sub-agent `.env`) is reached.
4. CSV/XLSX outputs land at the session root; PNG/SVG plots under `run_<tool_call_id>/` for inline display.

### E2B template (one-time setup)

Build the coding sandbox image manually (not on app startup). From the repo root with your venv active:

```bash
pip install 'e2b>=2.3.0'
python sub_agents/coding_sub_agent/e2b/build_e2b_template.py --write-env
```

Set `E2B_API_KEY` in `sub_agents/coding_sub_agent/.env` before building. After build, ensure `E2B_TEMPLATE_NAME` (and optional `E2B_TEMPLATE_ID`) are in that `.env`. Requires `e2b-code-interpreter>=2.7.0` (supports `lifecycle` on `Sandbox.create`). Runtime uses `allow_internet_access=False` and a 2-minute sandbox timeout (`E2B_SANDBOX_TIMEOUT_SECONDS=120`). By default `E2B_KILL_SANDBOX=false` so sandboxes stay visible in the E2B dashboard for debugging; set `true` in production to `kill()` after each run.

Guardrails include basename-only paths, allowed extensions, and blocked OS/network/subprocess patterns.

## Forecasting tools

All three tools inherit **`ForecastingModel`** (`tools/forecasting/base.py`) and share the same pipeline: validate → fit → fitted/residuals → forecast → save tables → LLM interpretation → unified JSON.

- **SARIMA tool** (`sarima_tool`) — Auto/manual ARIMA order, SARIMAX fit, 95% prediction intervals, Ljung-Box/Jarque-Bera diagnostics. Strict data rules (no missing target values). Writes **forecast + fitted** CSV/XLSX at session root.
- **Prophet tool** (`prophet_tool`) — Trend + weekly/monthly/yearly seasonality, changepoints, MAD residual checks. Tolerates missing target rows. Writes **forecast + fitted + decomposition** CSV/XLSX at session root.
- **Holt-Winters tool** (`holt_winters_tool`) — Exponential smoothing with additive/multiplicative trend and seasonality, 95% intervals, Ljung-Box/Jarque-Bera diagnostics. Writes **forecast + fitted + decomposition** CSV/XLSX at session root.
- **Charts** — No forecasting tool produces images. Use **`coding_tool`** on the saved forecast/fitted files (plots land under `run_<tool_call_id>/`).

**Unified JSON response:** `status`, `model_type`, `frequency`, `model`, `fit_quality`, `residual_diagnostics`, output file paths, previews, `llm_interpretation`, `warnings`. See each tool’s docstring for data format, arguments, and example calls.

**Input requirements:** regular date frequency (inferred from data), basename-only file references, minimum 10 observations. Upload CSV/XLSX to the session before calling a forecasting tool.

## Configuration

**Main app** (`config.yaml` + root `.env`): orchestrator model, context limits, optional Postgres checkpoints.

**Sub-agents** (separate `.env` files): planner model and limits; coding/judge models, E2B keys, retry limits. See each folder’s `.env.example`.

Runtime session data lives in `agent_filesystem/` (gitignored). Secrets belong in `.env` files only.

## Observability

Set `LANGFUSE_TRACING_ENABLED=true` (plus Langfuse keys in root `.env`) for one trace per `POST /run`, `POST /run/stream`, or `POST /resume`. The graph layer owns tracing: nested node spans, planner sub-graph spans, forecasting tool pipelines, and coding sub-agent spans (with explicit trace linking for nested `coding_tool` invokes). LLM generations record token counts only; SSE and the UI stay trace-free.

Notes:

- Planner `ask_user` interrupts produce a separate `resume` root span when you call `POST /resume` — expected for one logical turn.
- Coding spans store full generated code and stderr when tracing is on; artifacts also remain on disk.
- Parallel tool calls (`max_concurrency: 2`) may flatten nesting in Langfuse when multiple tools run in one orchestrator step.

## Project structure

```text
api/                 FastAPI service and SSE streaming
graph/               Main LangGraph agent
sub_agents/          Planner and coding sub-graphs (AnalysisGraph-style classes)
  planner_sub_agent/graph.py              PlannerGraph — mounted on main graph
  planner_sub_agent/tools/                write_todo, ask_user (planner clarification)
  coding_sub_agent/graph.py               CodingGraph — invoked by coding_tool
  coding_sub_agent/e2b/                   E2B template build + sandbox requirements
  coding_sub_agent/code_scan/             Semgrep rules + static scan for generated code
tools/               Main-graph @tool wrappers only
  coding_tools/coding_tool.py             Invokes coding sub-agent pipeline
  planning/update_todo.py                   Orchestrator todo status patches
  forecasting/                            ForecastingModel + SarimaModel / ProphetModel / HoltWintersModel + @tool wrappers
    base.py                                 Shared pipeline base class
    sarima_model.py                         SarimaModel
    prophet_model.py                        ProphetModel
    holt_winters_model.py                   HoltWintersModel
    sarima_tool.py                          LangChain sarima_tool wrapper
    prophet_tool.py                         LangChain prophet_tool wrapper
    holt_winters_tool.py                    LangChain holt_winters_tool wrapper
ui/                  Dash chat application
prompts/             Orchestrator system prompt
middleware/          LLM clients, context editing
session_paths.py     Session filesystem layout
config.yaml          Main platform settings
```

**Tool placement:** main-graph tools live under `tools/` and register on `graph/graph.py`. Sub-agent internal tools (planner `write_todo`, `ask_user`) live under `sub_agents/planner_sub_agent/tools/` only.

**Graph classes:** `AnalysisGraph` (main), `PlannerGraph` (mounted subgraph, no checkpointer — inherits parent for interrupts), `CodingGraph` (tool-invoked pipeline). Planner may pause for clarification via `ask_user` before `write_todo`; resume with `POST /resume` on the same session.

## For contributors

- **Prompts** (`prompts/`, `sub_agents/*/prompts.py`) hold LLM instructions; **code comments** explain graph/tool wiring and invariants — do not duplicate prompt text in comments.
- **Forecasting** — inherit `ForecastingModel`; use snake_case public methods; add `# Brief:` stage comments; tool docstrings are the orchestrator contract.
- Main graph: `graph/graph.py`
- Sub-agents: `sub_agents/planner_sub_agent/`, `sub_agents/coding_sub_agent/`
- Regenerate topology diagrams: `python scripts/generate_artifact_plot.py` → `artifact/*.png`
