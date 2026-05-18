# Agentic Forecasting Platform

An AI-powered forecasting and analysis workspace for uploaded tabular data. Users can upload CSV/XLSX files, ask natural-language questions, generate forecasts with SARIMA or Prophet, run guarded analysis code, and view generated tables and plots inside a Dash chat interface.

The project is built around a custom LangGraph agent rather than a single prompt wrapper. It includes planning, skill routing, session-scoped storage, dataset profiling, deterministic forecasting tools, sandboxed code execution, streaming API responses, and optional Langfuse observability.

## Why This Project Matters

Most forecasting demos stop at fitting a model. This project focuses on the surrounding platform work needed to make forecasting usable in an agentic application:

- A custom LangGraph `StateGraph` coordinates planning, tool use, forecasting, artifact generation, and final responses.
- Uploaded datasets are profiled automatically so the agent can reason about available files, columns, row counts, previews, and numeric statistics.
- Forecasting is handled by deterministic SARIMA and Prophet tools, while custom exploratory analysis and plotting go through a guarded code pipeline.
- Generated code is checked before execution with Semgrep and an LLM judge, then run in a constrained subprocess with controlled file access.
- API responses include explicit artifact and image paths, so the UI does not scrape model text to discover generated plots.

## Core Features

- **Natural-language forecasting workflow**: upload data, ask for forecasts, comparisons, plots, or analysis, and receive structured outputs.
- **Custom LangGraph agent**: planner, orchestrator, tool execution, todo completion gate, skill selection, and final-answer flow.
- **Forecasting tools**: SARIMA/SARIMAX via `statsmodels` and optional `pmdarima.auto_arima`; Prophet with trend, seasonality, residual, and component summaries.
- **Sandboxed analysis code**: LLM-generated Python is statically scanned, reviewed, saved as `pipeline_run.py`, and executed in a restricted subprocess.
- **Session workspace**: each chat session stores uploads, generated tables, scripts, and plot artifacts under `agent_filesystem/<session>/`.
- **Streaming API**: FastAPI exposes normal and SSE endpoints for run/resume flows.
- **Dash UI**: browser-based chat interface with file upload, interrupt/resume support, table previews, and inline image artifacts.
- **Observability**: optional Langfuse traces for graph nodes, tool calls, API requests, and model generations.
- **Config-driven models and limits**: model names, rate limits, graph recursion limit, sandbox timeout, paths, and checkpointing live in `config.yaml`.

## Tech Stack

| Area | Tools |
|------|-------|
| Agent orchestration | LangGraph, LangChain |
| LLM provider | OpenAI via `langchain-openai` |
| API | FastAPI, Uvicorn, SSE streaming |
| UI | Plotly Dash |
| Forecasting | Prophet, statsmodels, pmdarima |
| Data processing | pandas, NumPy, SciPy, scikit-learn |
| Code safety | Semgrep, subprocess sandbox, runtime file patches |
| Persistence | Session filesystem, optional Postgres/Neon checkpoints |
| Observability | Langfuse |

## Architecture

```text
User / Dash UI
    |
    v
FastAPI
    |
    v
LangGraph StateGraph
    |
    +--> Profile uploaded data
    +--> Select planner skills
    +--> Build/refresh todo plan
    +--> Select orchestration skills
    +--> Orchestrate tools
            |
            +--> SARIMA tool
            +--> Prophet tool
            +--> Sandboxed code pipeline
            +--> Todo updater
    |
    v
Final answer + generated artifacts
```

Graph flow:

```text
START -> BeginTurn -> ProfileSavedData -> SummariseConversationalSummary
      -> SelectPlannerSkills -> Planner -> SelectOrchestratorSkills -> Orchestrator
      -> RunTools -> ProfileSavedData_PostTools -> Orchestrator
      -> TodoCompletionGate -> FinalAnswer -> END
```

## Repository Layout

```text
api/                         FastAPI app, run/resume/upload/artifact endpoints
graph/                       LangGraph state machine and agent execution
middleware/                  LLM factory, rate limiting, context summarisation
tools/
  forecasting/               SARIMA and Prophet forecasting tools
  coding_tools/              Code generation, safety checks, sandbox runner
  file_management_tools/     Session data profiling
  planning/                  Todo update tool
skills/                      Planner and orchestrator skill playbooks
prompts/                     System prompts for graph, planner, codegen, judging
output_validation/           Pydantic schemas for structured model/tool outputs
ui/                          Dash app and API client
observability/               Langfuse helper utilities
config.yaml                  Models, graph limits, paths, checkpoint settings
requirements.txt             Python dependencies
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI key to `.env`:

```bash
OPENAI_API_KEY=your_key_here
```

Start the app:

```bash
./launch.sh
```

Default local URLs:

- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:8501`

Manual startup:

```bash
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --host 127.0.0.1 --port 8000
python ui/dash_app.py
```

## Example Workflow

1. Open the Dash UI.
2. Upload a `.csv` or `.xlsx` file with a date column and numeric target column.
3. Ask a question such as:

```text
Forecast monthly revenue for the next 6 months, explain the trend, and generate a chart.
```

4. The agent profiles the uploaded data, plans the task, chooses forecasting or analysis tools, saves output tables, and returns any generated plots in the chat.

## API Summary

| Endpoint | Purpose |
|----------|---------|
| `POST /run` | Run a natural-language agent request for a session |
| `POST /run/stream` | Stream graph progress and final response with SSE |
| `POST /resume` | Continue after a planner clarification interrupt |
| `POST /resume/stream` | Stream resume progress with SSE |
| `POST /upload-data` | Upload one or more `.csv` / `.xlsx` files |
| `GET /artifact/{session_id}/{path}` | Serve generated tables and plot artifacts |

Allowed upload types: `.csv`, `.xlsx`.

Allowed artifact types: `.csv`, `.xlsx`, `.png`, `.svg`.

## Code Execution Safety

The `code_pipeline` tool is designed for controlled analysis and plotting, not arbitrary execution.

Pipeline:

1. Generate structured Python code from the requested analysis task.
2. Run Semgrep rules against the generated source.
3. Run an LLM judge for an additional safety and task-fit review.
4. Save approved code to the session workspace as `pipeline_run.py`.
5. Execute it through `run_pipeline_sandboxed.py` with:
   - restricted file reads/writes,
   - controlled plot output directories,
   - credential-stripped environment variables,
   - wall-clock timeout,
   - memory limit where supported,
   - BLAS/OpenMP thread caps.

Plots are written only into the current tool-call folder:

```text
agent_filesystem/<session>/run_<tool_call_id>/<plot>.png
agent_filesystem/<session>/run_<tool_call_id>/<plot>.svg
```

The API discovers images by listing the current turn's tool-call folders and returns an explicit `images` list to the UI.

## Forecasting Tools

### SARIMA

The SARIMA tool validates a session CSV/XLSX file, checks the date and target columns, fits a SARIMAX model, produces forecasts with prediction intervals, and returns residual diagnostics and model interpretation.

It supports manual order selection or `pmdarima.auto_arima`-based order selection.

### Prophet

The Prophet tool validates a session CSV/XLSX file, infers regular frequency, fits a univariate Prophet model, and saves:

- future forecast table,
- fitted values and residuals,
- decomposition/component table.

It also returns fit-quality metrics, residual warnings, changepoint summaries, and structured interpretation.

## Configuration

Most runtime behavior is controlled from `config.yaml`:

```yaml
models:
  orchestrator: "gpt-5.4-mini"
  code_generation: "gpt-5.4-mini"
  code_judge: "gpt-5.4-mini"
  message_summarisation: "gpt-5.4-mini"

graph:
  recursion_limit: 100
  max_concurrency: 2

llm_rate_limit:
  enabled: true
  requests_per_second: 1.0
  check_every_n_seconds: 0.1
  max_bucket_size: 5.0

code_pipeline:
  timeout_seconds: 120

checkpointer:
  use_neon: false
```

Set `checkpointer.use_neon: true` and provide `DATABASE_URL` in `.env` to use Postgres checkpoints instead of the default in-memory checkpointer.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | Backend LLM calls |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing |
| `LANGFUSE_HOST` | No | Langfuse host |
| `LANGFUSE_BASE_URL` | No | Alternate Langfuse host variable |
| `DATABASE_URL` | Only for Postgres checkpoints | LangGraph checkpoint database |
| `GAUSSIANBLURR_API_URL` | No | Dash API base URL override |

The browser UI does not receive the OpenAI key. It only talks to the FastAPI backend.

## Notes for Reviewers

- Runtime session files are stored under `agent_filesystem/` and should remain gitignored.
- The app uses a session id as the LangGraph `thread_id`.
- Forecasting tools intentionally do not generate images directly; plotting is routed through the guarded `code_pipeline`.
- `data_profile` is held in graph state and refreshed before orchestration and after tool execution.
- Langfuse tracing is optional; the app still runs without Langfuse credentials.

## Resume Summary

Built an agentic forecasting platform with LangGraph, FastAPI, Dash, OpenAI APIs, Prophet, and SARIMA. The system supports natural-language dataset analysis, session-scoped file handling, automated data profiling, deterministic forecasting tools, sandboxed LLM-generated code execution, artifact serving, streaming responses, and optional observability through Langfuse.
