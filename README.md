# GaussianBlurr — Agentic Forecasting Platform

Upload a spreadsheet, ask a question in everyday language, and get forecasts, analysis, and charts back in a chat-style interface—with live progress as the system works.

---

## What you can do (no jargon)

- **Upload** CSV or Excel files for one conversation (a “session”).
- **Ask** things like *“Forecast next year’s sales”* or *“Plot revenue by month.”*
- **Get** written answers, forecast tables (CSV), and charts shown inline in the chat.
- **Stay in control** — the assistant can ask a clarifying question before continuing; you answer in the same chat and it resumes.

Behind the scenes, an AI planner breaks your request into steps, specialized tools run forecasts or custom analysis, and any generated code runs in a **locked-down cloud sandbox** (no open internet) before results are saved to your session folder.

---

## 30-second overview

GaussianBlurr is a full-stack **agentic data-analysis platform**—not a static notebook. It wires together:

- A **LangGraph** orchestration layer (multi-step AI workflow),
- **Deterministic forecasting** (SARIMA, Prophet, Holt-Winters),
- **Sandboxed Python** when you need custom analysis or charts,
- **Session-scoped files** (uploads and outputs per chat),
- A **streaming Dash UI** fed by a FastAPI backend.

It is built to show solid product UX, safe automation, and optional operational tracing (Langfuse).

---

## Example: one request end to end

1. Upload `sales.csv` in the sidebar.
2. Ask: *“Forecast the next 12 months of revenue and show a trend chart.”*
3. The **planner** writes a short task checklist; the **orchestrator** picks forecasting and/or coding tools.
4. You see progress stream in chat; forecast CSVs land in your session; the chart appears in the reply.

---

## Why this project

| For everyone | For builders |
|--------------|--------------|
| Plans and tracks tasks automatically instead of one long prompt | LangGraph main graph + planner and coding sub-graphs |
| One folder per chat for uploads and outputs | Basename-only file contracts and path containment |
| Trusted forecast models when you want structure | Shared `ForecastingModel` pipeline + three `@tool` wrappers |
| Custom code only after scans and reviews, then E2B sandbox | Semgrep + LLM judges + optional Langfuse spans |
| Live progress in the UI | FastAPI SSE (`/run/stream`) + Dash client |

---

## Features

- Natural-language Q&A over uploaded CSV/XLSX
- Automatic **data profiling** (columns, samples, basic stats) at the start of each turn—and again after tools run
- **Todo planning** at the start of each user message
- Custom Python analysis and charts via the **coding tool**
- SARIMA / Prophet / Holt-Winters forecasting with structured JSON for the orchestrator
- FastAPI backend with SSE streaming; Plotly Dash frontend

---

## How it works

**Simple picture**

```text
You (browser)  →  Dash UI  →  FastAPI  →  AI agent  →  tools
                                              ↓
                                    files & charts in your session folder
```

**Each new message**

1. Profile files in the session workspace.  
2. **Planner** replaces the todo list for this turn (and may **ask you** a question—execution pauses until you resume).  
3. **Orchestrator** calls tools (forecast, code, update todo status) until work is done.  
4. **Final answer** node writes the user-facing reply.

Technical readers: main graph is `AnalysisGraph` in `graph/graph.py` — `ProfileSavedData` → `Planner` → `Orchestrator` ↔ `RunTools` → `FinalAnswer`.

---

## Quickstart

### 1. Clone and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

| Location | What to set |
|----------|-------------|
| Repo root `.env` (from `.env.example`) | All runtime config: `OPENAI_API_KEY`, `MAIN_*`, `PLANNER_*`, `CODING_*`, optional `DATABASE_URL`, optional Langfuse keys |

Use one `.env` at repo root. Secrets stay in `.env` only (never commit them).

For **custom Python analysis**, build the E2B sandbox template once (see [E2B template](#e2b-template-one-time-setup) below).

### 3. Run

```bash
./start.sh
```

- **API:** http://127.0.0.1:8000  
- **UI:** http://127.0.0.1:8501  

### 4. Use it

Upload a CSV in the sidebar, ask a forecasting or analysis question, and watch progress in the chat.

---

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload-data` | Upload CSV/XLSX into a session |
| POST | `/run` | Run agent (blocking JSON) |
| POST | `/run/stream` | Run agent (SSE progress + final `done` payload) |
| POST | `/resume` | Continue after a clarification interrupt |
| POST | `/resume/stream` | Same as resume, with SSE |
| GET | `/artifact/{session_id}/{path}` | Download a session file or plot image |

**Session id** — passed as `session_id` (same value LangGraph uses as `thread_id`). The Dash UI generates one per browser session.

---

## Code execution and safety

When the orchestrator needs **custom Python** (charts, transforms, etc.), it calls the **coding tool**, which runs an internal pipeline:

```text
Generate code → Semgrep scan → Safety judge → IO allowlist judge → Run in E2B sandbox
         ↑______________________________________________|
              retry up to CODING_MAX_CODEGEN_ATTEMPTS (default 3)
```

**In plain terms**

1. The system writes Python for your task and the file names it may read or write.  
2. Automated checks (pattern rules + AI reviewers) must pass before anything runs.  
3. Approved scripts run in a **fresh, internet-off cloud sandbox**; outputs are copied back to your session.  
4. If something fails (unsafe code, wrong files, runtime error), it **tries again** with feedback—up to a configured limit—then reports failure clearly.

**Where files go**

- Tables (CSV/XLSX) → session root (same folder as uploads).  
- Plots (PNG/SVG) → `run_<tool_call_id>/` so the UI can show them inline.

**Coding models and retries** (root `.env`)

| Variable | Meaning |
|----------|---------|
| `CODING_MODEL` | Model for code generation on attempts 1 … (MAX − 1) |
| `CODING_MODEL_LAST_ATTEMPT` | Optional stronger model **only** on the last permitted attempt (when attempt count equals `CODING_MAX_CODEGEN_ATTEMPTS`). Leave empty to always use `CODING_MODEL`. |
| `CODING_MAX_CODEGEN_ATTEMPTS` | How many times the pipeline may **regenerate** code after a gate or sandbox failure (default `3`) |
| `CODING_CODE_JUDGE_MODEL` / `CODING_IO_JUDGE_MODEL` | Models for safety and file-allowlist review |

Example: with `CODING_MAX_CODEGEN_ATTEMPTS=3`, attempts 1–2 use `CODING_MODEL`; attempt 3 can use `CODING_MODEL_LAST_ATTEMPT` (e.g. a larger model) if you set it.

Guardrails include **basename-only** paths (no folder prefixes in tool args), allowed extensions, blocked dangerous patterns, and session path containment.

### E2B template (one-time setup)

Build the sandbox image manually (not on app startup). From the repo root with your venv active:

```bash
pip install 'e2b>=2.3.0'
python sub_agents/coding_sub_agent/e2b/build_e2b_template.py --write-env
```

Set `CODING_E2B_API_KEY` in root `.env` before building. After build, ensure `CODING_E2B_TEMPLATE_NAME` (and optional `CODING_E2B_TEMPLATE_ID`) are in root `.env`.

Runtime defaults: `allow_internet_access=False`, sandbox timeout 120s (`CODING_E2B_SANDBOX_TIMEOUT_SECONDS`). By default `CODING_E2B_KILL_SANDBOX=false` so runs stay visible in the E2B dashboard for debugging; set `true` in production to destroy sandboxes after each run.

Requires `e2b-code-interpreter>=2.7.0` (supports `lifecycle` on `Sandbox.create`).

---

## Forecasting tools

All three tools share the same pipeline: validate → fit → fitted/residuals → forecast → save CSV/PNG → lean JSON (`ForecastingModel` in `tools/forecasting/base.py`).

| Tool | Best for |
|------|----------|
| **SARIMA** (`sarima_tool`) | Classical ARIMA/SARIMAX; strict (no missing target values); diagnostics (Ljung-Box, Jarque-Bera) |
| **Prophet** (`prophet_tool`) | Trend + seasonality; tolerates missing targets; decomposition outputs |
| **Holt-Winters** (`holt_winters_tool`) | Exponential smoothing; trend/seasonality options; decomposition |

**`experiment_name`** (required) — unique label for one run. Artifacts look like `{experiment_name}_fitted.csv`, `_forecast.csv`, optional `_decomposition.csv`, plus PNG plots. JSON returned to the agent uses **filenames only**, not full disk paths.

**Response shape:** `status`, `model_type`, `experiment_name`, `frequency`, `warnings`, and `pipeline` (per-stage status, metrics, previews, plot basenames). An LLM interpretation helper exists on the base class but is not wired into the live pipeline yet.

**Data expectations:** regular date frequency (inferred), basename-only `file_name`, at least 10 rows. Upload data to the session before calling a tool.

---

## Configuration

**Main app + sub-agents** — root `.env` only: orchestrator model, graph limits, planner/coding models, E2B keys, optional Postgres checkpointing (`MAIN_CHECKPOINTER_USE_NEON` + `DATABASE_URL`).

**Runtime data** — `agent_filesystem/` (gitignored): one directory per session id.

---

## Observability

Set `LANGFUSE_TRACING_ENABLED=true` and Langfuse keys in the root `.env` for traces on `POST /run`, `/run/stream`, `/resume`, and `/resume/stream`.

- One root trace per API call; nested spans for graph nodes, planner, forecasting pipelines, and coding sub-agent (linked under `coding_tool`).  
- Planner `ask_user` → a **resume** call gets its own root span (expected for one logical turn).  
- Coding traces may include generated code and stderr; artifacts remain on disk either way.  
- Parallel tools (`max_concurrency: 2`) can flatten nesting in Langfuse when multiple tools finish in one step.

The chat UI does not expose trace IDs—tracing is for operators and developers.

---

## Project structure

```text
api/                 FastAPI service and SSE streaming
graph/               Main LangGraph agent (AnalysisGraph)
sub_agents/
  planner_sub_agent/   PlannerGraph — todos, ask_user interrupt
  coding_sub_agent/    CodingGraph — codegen, scans, judges, E2B
    e2b/               Template build + sandbox requirements
    code_scan/         Semgrep rules + static scan
tools/               Main-graph tools only
  coding_tools/      coding_tool → CodingGraph
  planning/          update_todo (orchestrator)
  forecasting/       SARIMA / Prophet / Holt-Winters models + tools
ui/                  Dash chat application
prompts/             Orchestrator system prompt
middleware/          LLM clients, rate limiting
session_paths.py     Session filesystem layout
```

**Tool placement:** main-graph tools register on `graph/graph.py`. Planner-only tools (`write_todo`, `ask_user`) live under `sub_agents/planner_sub_agent/tools/`.

**Graph classes:** `AnalysisGraph` (main), `PlannerGraph` (mounted subgraph; interrupts use parent checkpointer), `CodingGraph` (invoked inside `coding_tool`). Resume after planner clarification: `POST /resume` with the same `session_id`.

---

## Glossary (quick)

| Term | Meaning |
|------|---------|
| **Session** | One chat’s workspace (uploads + generated files); identified by `session_id`. |
| **Planner** | Sub-agent that builds/replaces the todo list and may ask clarifying questions. |
| **Orchestrator** | Main agent that picks tools and drives work to completion. |
| **Coding tool** | Generates, reviews, and runs Python in E2B; retries on failure. |
| **SSE** | Server-Sent Events—live progress stream from API to browser. |
| **E2B** | Third-party isolated cloud environment where generated code runs. |
| **LangGraph** | Framework used to define the agent workflow as a graph of steps. |

---

## For contributors

- **Prompts** (`prompts/`, `sub_agents/*/prompts.py`) hold LLM instructions; **code comments** explain graph wiring—invariants, not duplicated prompt text.
- **Forecasting** — inherit `ForecastingModel`; snake_case public methods; tool docstrings are the orchestrator contract.
- **Entry points** — main graph: `graph/graph.py`; sub-agents: `sub_agents/planner_sub_agent/`, `sub_agents/coding_sub_agent/`.
- **Diagrams** — `python scripts/generate_artifact_plot.py` → `artifact/*.png`.
