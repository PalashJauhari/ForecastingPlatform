# GaussianBlurr

An analysis assistant built with a custom **LangGraph `StateGraph`**, a single **`config.yaml`** for all configuration, and a **session workspace under `agent_filesystem/<session>/`**. Tabular uploads and outputs (`.csv`, `.xlsx`) and **`pipeline_run.py`** live at the **session root** so later questions can reuse them. **Plots are produced exclusively by `code_pipeline`**: every invocation allocates a per-tool-call **`run_<tool_call_id>/`** subfolder, and **matplotlib** saves (`plt.savefig` / `Figure.savefig`) write **`.png`** / **`.svg`** only into that folder. The API discovers images for a turn by scanning the matching `run_<tool_call_id>/` folders for the tool calls in that turn and returns an explicit **`images`** list — the UI never parses tool result text to find plots. Logical paths look like **`agent_filesystem/<session>/<filename>`** for tabular files and **`agent_filesystem/<session>/run_<tool_call_id>/<plot>.png`** for plots. Before each orchestrator turn, the **`data_profile`** graph node runs **`profiling_data.profile_session_workspace`**, which fills the **`data_profile`** state field with a **list** of per-file summaries (row counts, **`head`** previews, **`column_profiles`** per-column stats, and **`numeric_summary`** where applicable). **Profiling does not write a snapshot file**—only graph state and the orchestrator’s assembled context carry the profile. There is no list/read tool on the graph.

On each user turn after prep + optional summarisation, graph nodes **`SelectPlannerSkills`** and **`SelectOrchestratorSkills`** (see `skills/loader.py`) each run a structured LLM against **`skills/registry.py`** manifests, normalize **aliases**, and cap **`skills.max_orchestrator_selected`** / **`skills.max_planner_selected`** via **`models.identify_orchestrator_skills`** / **`models.identify_planner_skills`**; **`LoadPlannerSkills`** / **`LoadReasoningSkills`** hydrate **`approach.md`**. **`active_skills`** / **`active_planner_skills`** are stored on state so **`code_pipeline`** can load matching **`patterns.py`** snippets.

---

## Features

- **Custom StateGraph agent** — orchestrator node + `ToolNode` (parallel tool calling), no `create_agent` black box.
- **YAML-driven config** — orchestrator model, code-generation model, judge model, summarisation model, and middleware thresholds in `config.yaml`.
- **OpenAI rate limiting (server-side)** — optional shared [`InMemoryRateLimiter`](https://python.langchain.com/docs/integrations/chat/openai/#rate-limiting) from `middleware/llm_rate_limit.py`, attached inside **`middleware/llm_client.make_llm`** to every `ChatOpenAI` built through that factory (orchestrator, summarisation path, codegen, judge, skill router). Tuned under `llm_rate_limit` in `config.yaml`; set `enabled: false` to disable.
- **Four orchestrator-bound tools** — `code_pipeline`, `sarima_tool`, `prophet_tool`, **`update_todo`**. The orchestrator uses **`make_llm`** then **`llm.bind_tools(TOOLS)`** (`graph/graph.py`); codegen paths use **`make_llm(..., output_schema=...)`** without tools. **`Planner`** replaces **`todos`** on each new user message at the prep stage; **`update_todo`** patches statuses afterward.
- **Workspace profiling (`data_profile`)** — **ProfileSavedData** nodes run once per **prep** and again after **`RunTools`** (**ProfileSavedData_PostTools**) before **`MergeTools`** → **`Orchestrator`**. They update **`data_profile`** in state (**`profiling_data.profile_session_workspace`**).
- **Session planning in state** — The **`Planner`** node writes **`todos`** (`id`, `content`, `status`). **`update_todo`** updates one **`status`** via **`Command`**. **`TodoCompletionGate`** blocks **`FinalAnswer`** until todos are **`completed`** or empty. **`Orchestrator`** sees todos in context JSON.
- **Planner clarification** — **`interrupt()`** can pause inside the **Planner** (`phase: planner`). **`POST /resume`** continues with **`Command(resume=...)`**.
- **Running summarisation** — when the conversation exceeds a configurable token threshold, older messages are summarised into a running summary and truncated (via `RemoveMessage`), keeping the context window manageable.
- **Session-scoped storage** — logical paths and disk paths share the same tree under `./agent_filesystem/<session>/...` (resolved from the repo root), with **`run_<tool_call_id>/`** subfolders allocated per `code_pipeline` invocation for plot artifacts only.
- **Graph execution cap** — LangGraph’s `recursion_limit` in `config.yaml` bounds how many super-steps a single `/run` or `/resume` call can take, which is the main protection against runaway tool loops.
- **Code execution pipeline** (`tools/coding_tools/code_pipeline.py`) — LLM codegen → **Semgrep** (static) → **LLM judge** → save `agent_filesystem/<session>/pipeline_run.py` → subprocess runner. Plots land under **`run_<tool_call_id>/`** so the API can find them by inspecting the message-level `tool_call_id` rather than parsing the tool's JSON. Tool result JSON includes **`plots`**: logical paths under that folder (empty list if none), plus task echo and **`stdout`** / **`stderr`** / **`code_violation`** fields.
- **Plot artifacts in the UI** — FastAPI **`GET /artifact/{session_id}/{path}`** (`api/main.py`) serves allowlisted files from the session directory (containment-checked). Dash builds image URLs via **`GaussianBlurrApiClient.artifact_url`** and renders **png/svg/jpg/jpeg** inline in chat bubbles; other outputs stay as **`Saved: …`** text.
- **Two-layer code sandbox** —
  - **Layer 1 — Semgrep** (`tools/coding_tools/code_scan/semgrep_scan.py`): static scan on generated source before save, using `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` (tabular paths + allowed **`plt.savefig` / `Figure.savefig`** with bare **`.png`** / **`.svg`** literals only).
  - **Layer 2 — Runtime patch** (`tools/coding_tools/code_scan/runtime_patch_scan.py`): monkey-patches **`builtins.open`** (allowlisted extensions including `.png`/`.svg` for library internals), **`pd.read_csv` / `pd.read_excel` / `df.to_csv` / `df.to_excel`** (paths resolve under the **session** workspace), and **`plt.savefig` / `Figure.savefig`** (paths resolve under the **per-tool-call** `run_<tool_call_id>/` folder passed into the subprocess). Applied only while the generated script runs.
  - **Runner** (`tools/coding_tools/code_scan/run_pipeline_sandboxed.py`): `code_pipeline` always executes generated code through this script. **Required** argv: path to **`pipeline_run.py`**, **session workspace** (data read/write root), **run workspace** (plot write root). **Optional** 4th argv: path to a JSON file `{"input": [...], "output": [...]}` for runtime basename allowlists when the tool task lists are non-empty. It applies runtime patches, sets **200 MiB** virtual address limit (`RLIMIT_AS` where supported), caps **BLAS/OpenMP to one thread** (env vars before pandas loads), pins **CPU 0** on **Linux** (`sched_setaffinity`), then runs `pipeline_run.py` via `runpy.run_path`.
  - **Sanitized subprocess environment** — the child process that runs generated code receives a **filtered** copy of the parent’s environment: `OPENAI_API_KEY`, all `LANGFUSE_*` variables, and other listed provider credentials are **removed** before `subprocess.run` (see **`env_for_sandbox_subprocess`** in `code_pipeline.py`). The API process still has the full env for real LLM calls; the sandbox script cannot read those secrets via `os.environ`, in addition to Semgrep rules that discourage env access in source.
- **Parent timeout** — `code_pipeline` uses `subprocess.run(..., timeout=...)` from `config.yaml` (wall-clock kill of the child process).
- **Skills** (`skills/`) — Just-in-time expertise:
  - **`SelectPlannerSkills` / `SelectOrchestratorSkills`** (`skills/loader.py`) — two structured router calls (**`skills.max_*`** caps; **`skills.registry`** for catalogs + **`path_for_*_skill`**).
  - **`LoadReasoningSkills`** / **`LoadPlannerSkills`** — **`approach.md`** hydration.
  - **`LoadPatternSkills`** — inside **`code_pipeline`** codegen from **`patterns.py`**.

- **Sessions** — `thread_id = session_id`. Checkpoints use `InMemorySaver` by default, or Postgres when `checkpointer.use_neon: true` and `DATABASE_URL` is set (e.g. Neon; survives API restarts).
- **Observability** — optional **Langfuse** tracing: graph nodes and tools use `@observe` / nested spans; **`api/main.py`** wraps **`POST /run`** and **`POST /resume`** in **`propagate_attributes(...)`** so `session_id`, tags, and small request metadata attach to traces. **`observability/langfuse_handler.get_langfuse_client()`** uses the Langfuse SDK; if you set **`LANGFUSE_BASE_URL`** but not **`LANGFUSE_HOST`**, the client copies it into **`LANGFUSE_HOST`** for compatibility.

## Graph architecture

```
START → BeginTurn → ProfileSavedData → SummariseConversationalSummary
     → MergePrep → SelectPlannerSkills → Planner → SelectOrchestratorSkills → Orchestrator
     → [RunTools ‖ TodoCompletionGate → Orchestrator | FinalAnswer → END]

RunTools → ProfileSavedData_PostTools → MergeTools → Orchestrator
```

**State schema** (`graph/graph.py` — `AgentState`):

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `Annotated[list, add_messages]` | Full conversation — `HumanMessage`, `AIMessage`, `ToolMessage`. Supports `RemoveMessage` for truncation. |
| `message_summary` | `str` | Running summary of evicted messages. |
| `data_profile` | `list` | Refreshed by **`ProfileSavedData`** (prep) and **`ProfileSavedData_PostTools`** (after **`RunTools`**). Rendered in the **Session workspace** block. |
| `todos` | `list[dict]` | Optional. Each item: **`id`**, **`content`**, **`status`**. **Planner** replaces the full list on each new user message; **`update_todo`** patches one **`status`**. |
| `active_skills` | `list[str]` | Orchestrator skill ids (**`SelectOrchestratorSkills`**). |
| `active_planner_skills` | `list[str]` | Planner skill ids (**`SelectPlannerSkills`**). |
| `planner_skill_guidance` | `str` (optional) | Planner **`approach.md`** (**`SelectPlannerSkills`**). |
| `skill_guidance` | `str` (optional) | Orchestrator **`approach.md`** (**`SelectOrchestratorSkills`**). |

---

## Repository layout

```
config.yaml                                  # Models, middleware, graph, llm_rate_limit, checkpointer, paths, …
session_paths.py                             # Session id → disk root; logical agent_filesystem/… ↔ Path (cross-session safety)
agent_filesystem/                          # Runtime data (gitignored); one folder per session
  <session_id>/                              # Session root: uploads, CSV/XLSX outputs, pipeline_run.py
    run_<tool_call_id>/                      # Per code_pipeline invocation: plot outputs (.png / .svg) only
graph/
  graph.py                                   # StateGraph, AnalysisGraph
  __init__.py
skills/                                      # Skills library
  orchestrator_skills/                       # Execution-time playbooks (forecasting, …)
  planner_skills/                            # Planning / decomposition playbooks
  loader.py / registry.py                    # skill picks, manifests, path resolution (Langfuse‑traced)
middleware/
  llm_client.py                              # make_llm(...) — shared ChatOpenAI + rate_limiter + optional structured output
  context_editing.py                         # Truncate messages at safe turn boundaries
  (message summarisation lives in ``context_editing.py`` — ``truncate_and_summarize``)
  llm_rate_limit.py                          # Shared ``InMemoryRateLimiter`` (``config.yaml`` → ``llm_rate_limit``)
  __init__.py
tools/
  human_in_loop/
    ask_user.py                              # Not bound on orchestrator (legacy / future use)
  planning/
    update_todo.py                           # Patch one todo status by id
    write_todos.py                           # Legacy full-list tool (not bound on orchestrator)
  coding_tools/
    code_pipeline.py                         # codegen → Semgrep → judge → save → run_pipeline_sandboxed
    code_scan/
      run_pipeline_sandboxed.py              # argv: pipeline_run.py, session_workspace, run_workspace, [io_allowlist.json]
      semgrep_scan.py                        # Layer 1 — Semgrep
      llm_judge.py                           # Post-Semgrep LLM review
      runtime_patch_scan.py                  # Layer 2 — I/O monkey-patches
      codegen_scan_semgrep.yaml              # Semgrep rules
  file_management_tools/
    profiling_data.py                        # profile_session_workspace (graph profiling → data_profile)
prompts/
  graph_prompts.py                           # Orchestrator SYSTEM_PROMPT
  planner_prompt.py                          # Planner SYSTEM_PROMPT (structured plan / clarification)
  skill_identification_prompt.py             # Router persona and selection rules
  code_generation_prompt.py                  # Codegen system prompt
  code_judge_prompt.py                       # Judge system prompt
output_validation/
  code_generation.py                         # Pydantic: CodePipelineTask, codegen output, tool input schema
  judge_output.py                            # Pydantic: LLM judge
  skill_selection.py                         # Pydantic: OrchestratorSkillPick, PlannerSkillPick
  write_todos.py                             # Pydantic models for ``write_todos`` tool args
api/
  main.py                                    # POST /run, /resume, /upload-data; GET /artifact/… for plots & files
ui/
  __init__.py
  api_client.py                              # HTTP client (upload / run / resume + artifact_url for GET /artifact)
  dash_app.py                                # Plotly Dash UI (layout + callbacks)
  assets/                                    # Static CSS for Dash (e.g. markdown in chat)
observability/
  langfuse_handler.py
```

---

## Middleware (plain functions, called from orchestrator node)

| Function | Module | Role |
|----------|--------|------|
| `truncate_and_summarize` | `middleware/context_editing.py` | Truncation + running summary when token estimate exceeds threshold. |
| `make_llm` | `middleware/llm_client.py` | Builds `ChatOpenAI` with shared rate limiter and optional `with_structured_output(...)`. |
| `OPENAI_RATE_LIMITER` | `middleware/llm_rate_limit.py` | Optional `InMemoryRateLimiter` from `config.yaml`; consumed by `make_llm` (not wired on ad‑hoc clients that bypass the factory). |
| — | `code_pipeline` (tool) | Codegen → Semgrep → LLM judge → save → `run_pipeline_sandboxed.py` execution. |

**Graph nodes (non-middleware):** `BeginTurn`, `ProfileSavedData`, `SummariseConversationalSummary`, `MergePrep`, `SelectPlannerSkills`, `Planner`, `SelectOrchestratorSkills`, `Orchestrator`, `RunTools`, `ProfileSavedData_PostTools`, `MergeTools`, `TodoCompletionGate`, `FinalAnswer`.

---

## Tools

| Tool | Arguments | What it does |
|------|-----------|--------------|
| **code_pipeline** | Structured **`task`** (`requirements`, `input` basenames, `output` basenames), optional **`data_profile`**, **`previous_code_violation`** | Injects **`LoadPatternSkills(active_skills)`** into codegen. Pipeline: structured codegen → Semgrep → LLM judge (both always run when code is non-empty) → save `pipeline_run.py` → `run_pipeline_sandboxed.py` (session workspace + **`run_<tool_call_id>/`**, optional IO allowlist JSON). **Only image-producing tool**; API exposes plots via **`images`**. Tool JSON includes task echo, **`stdout`** / **`stderr`** / **`code_violation`** / **`plots`**, and **`code`** when safety or runtime issues require it. Retry with **`previous_code_violation`**. |
| **sarima_tool** | `file_name`, `date_column`, `target_column`, `horizon`, `seasonal_period`, `forecast_output_file`, optional `use_auto_arima` / `order` / `seasonal_order` | Deterministic ARIMA/SARIMA pipeline on a session CSV/XLSX: validate → time index → order selection (manual or **`pmdarima.auto_arima`** by AICc) → SARIMAX fit → residual diagnostics (Ljung-Box, Jarque-Bera) → forecast with 95% prediction intervals → save table at session root → four structured LLM interpretations (**`residual_analysis`**, **`fit_quality`**, **`forecast_summary`**, **`model_improvement_guidance`**). **Does not generate images** — follow up with `code_pipeline` if a chart is needed. Hard input/data errors (missing file, missing columns, non-regular spacing, missing/constant target) return a clean error JSON; bad residual diagnostics surface as warnings, not errors. **`fit_quality`** and **`residual_diagnostics`** include inline **`definitions`** so the orchestrator/LLM never has to guess what each metric means. |
| **prophet_tool** | `file_name`, `date_column`, `target_column`, `horizon`, `changepoint_prior_scale`, `seasonality_mode`, `weekly_seasonality`, `monthly_seasonality`, `yearly_seasonality`, `forecast_output_file`, `fitted_output_file`, `decomposition_output_file` | Deterministic univariate Prophet pipeline on a session CSV/XLSX: validate → rename to **`ds`**/**`y`** → infer freq → fit Prophet (caller-controlled `changepoint_prior_scale`, `seasonality_mode`, weekly/monthly/yearly flags; fixed `changepoint_range=0.8`, `seasonality_prior_scale=10.0`, `monthly_fourier_order=5`) → fit quality (MAE/RMSE/SMAPE) + MAD-based residual outlier diagnostics (`median ± 3 * robust_sigma`) + changepoint summary → save **forecast** (future + components), **fitted** (actual + fitted + residual + components), and **decomposition** (combined fitted+forecast components) tables at session root → five structured LLM interpretations (**`residual_analysis`**, **`fit_quality`**, **`forecast_summary`**, **`component_analysis`**, **`model_improvement_guidance`**). **Does not generate images** — follow up with `code_pipeline` if a chart is needed. Hard input/data errors return a clean error JSON; residual outliers surface as warnings, not errors. **`fit_quality`** and **`residual_diagnostics`** include inline **`definitions`**. |
| **update_todo** | `todo_id`, `status` | Patches one todo’s **`status`** by **`id`**; required bookkeeping before finishing when todos exist. |

---

## Configuration (`config.yaml`)

```yaml
models:
  orchestrator: "gpt-5.4-mini"       # Main agent + skill router; edit to taste
  code_generation: "gpt-5.4-mini"  # code_pipeline codegen
  code_judge: "gpt-5.4-mini"         # Post-Semgrep judge
  message_summarisation: "gpt-5.4-mini"

middleware:
  context_editing:
    keep_recent_messages: 10
  message_summarisation:
    token_threshold: 100000

graph:
  recursion_limit: 100               # Max LangGraph super-steps per invoke (run + resume)
  max_concurrency: 2

# Shared process-local token bucket for OpenAI calls made through make_llm (set enabled: false to disable).
llm_rate_limit:
  enabled: true
  requests_per_second: 1.0
  check_every_n_seconds: 0.1
  max_bucket_size: 5.0               # burst capacity (not “parallelism”; see LangChain docs)

code_pipeline:
  timeout_seconds: 120

paths:
  # On-disk: ./agent_filesystem/<session>/ (tabular I/O) and ./agent_filesystem/<session>/run_<tool_call_id>/ (plots per code_pipeline invocation)
  agent_filesystem: "agent_filesystem"

checkpointer:
  use_neon: false                    # true + DATABASE_URL → PostgresSaver (Neon, etc.)
```

Logical paths: **`agent_filesystem/<session-folder>/<file>`** for tabular artifacts at session root; **`agent_filesystem/<session-folder>/run_<tool_call_id>/<plot>.png`** (or `.svg`) for plot outputs from a single **`code_pipeline`** invocation.

**`llm_rate_limit`** — LangChain’s process-local token bucket for outbound OpenAI requests (not a Dash/browser limit). One `InMemoryRateLimiter` instance from `middleware/llm_rate_limit.py` is passed as `rate_limiter=` into each `ChatOpenAI` so concurrent tools and the orchestrator share the same cap. Parameters match LangChain: `requests_per_second`, `check_every_n_seconds`, `max_bucket_size`.

Semgrep rules live in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` — edit to tune blocking. Memory limit (200 MiB), BLAS thread caps, and Linux CPU affinity are defined in `tools/coding_tools/code_scan/run_pipeline_sandboxed.py`, not in YAML.

---

## Environment (`.env`)

- **`.env`** — your real secrets live here. It is listed in `.gitignore` (along with `*.env` / `.env.*`); **never commit it**.
- **`.env.example`** — safe template committed to the repo: same variable **names**, placeholder or empty values, and short comments. New setups: `cp .env.example .env`, then edit `.env`.

`python-dotenv` loads the project-root `.env` into the process environment at startup:

- **`api/main.py`** — loads before the graph is imported (FastAPI / `uvicorn`).
- **`graph/graph.py`** — loads at the top of the module so `OPENAI_API_KEY` is set **before** module-level `ChatOpenAI` clients are constructed (e.g. scripts or tests that import `graph` without going through `main`).

LangChain’s `ChatOpenAI` reads **`OPENAI_API_KEY` from the environment** by default (OpenAI SDK convention); you do not pass the key in code.

The Dash UI talks to the HTTP API only (via `ui/api_client.py`) and does **not** need an OpenAI key in the browser.

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (loaded into `os.environ` for backend LLM calls) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing |
| `LANGFUSE_HOST` | No | Langfuse API host (SDK default); use **`LANGFUSE_BASE_URL`** instead if you prefer—`get_langfuse_client()` maps it to `LANGFUSE_HOST` when the latter is unset |
| `DATABASE_URL` | When Postgres checkpoints are on | Postgres connection URI (e.g. Neon). Required only if `checkpointer.use_neon: true` in `config.yaml` (see **Checkpoints** below). |
| `GAUSSIANBLURR_API_URL` | No | Dash UI only: API origin (no trailing slash). Defaults to `http://127.0.0.1:8000`. |

---

## Checkpoints (InMemory vs Postgres / Neon)

LangGraph conversation state is keyed by **`session_id`** as **`thread_id`**.

| `config.yaml` | Behaviour |
|-----------------|------------|
| `checkpointer.use_neon: false` (default) | **`InMemorySaver`** — checkpoints exist only in the API process; they are **lost on restart**. |
| `checkpointer.use_neon: true` | **`PostgresSaver`** via **`psycopg`** — checkpoints are stored in Postgres; set **`DATABASE_URL`** in `.env` (see `.env.example`, typically `?sslmode=require` for Neon). On startup the API prints which backend is active. |

**Dependencies:** Postgres checkpointing uses **`langgraph-checkpoint-postgres`** and **`psycopg[binary]`** (listed in `requirements.txt`). After pulling the repo or enabling Neon, run **`pip install -r requirements.txt`** in the **same virtualenv** you use for `uvicorn`. If you see **`ModuleNotFoundError: No module named 'langgraph.checkpoint.postgres'`**, that package is missing from that environment.

**Schema:** the first successful startup with Neon runs **`checkpointer.setup()`** to create LangGraph’s checkpoint tables. If you change **`AgentState`** incompatibly, start new sessions (`session_id`) or migrate data as needed.

---

## Run

```bash
pip install -r requirements.txt
cp .env.example .env    # add OPENAI_API_KEY
./launch.sh
```

- **API:** `http://127.0.0.1:8000`
- **UI:** `http://127.0.0.1:8501` (Plotly Dash)

**Manual (two terminals):**

```bash
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --host 127.0.0.1 --port 8000
python ui/dash_app.py
```

### Dash UI and HTTP client

The browser UI is **`ui/dash_app.py`**. It does not call OpenAI directly; it only talks to **`api/main.py`** via **`ui/api_client.py`** (`GaussianBlurrApiClient`):

| Client method | HTTP | Form / file fields |
|---------------|------|---------------------|
| `upload_data(bytes, filename, session_id)` | `POST /upload-data` | `session_id`, multipart `files` |
| `run(query, session_id)` | `POST /run` | `query`, `session_id` |
| `resume(resume_value, session_id)` | `POST /resume` | `resume_value`, `session_id` |
| `artifact_url(session_id, relative_path)` | `GET /artifact/{session_id}/{path}` | Builds URL for browser `<img src>`; accepts logical `agent_filesystem/...` or session-relative `run_<tool_call_id>/file.png` |

- **Base URL:** defaults to `http://127.0.0.1:8000`. Override with **`GAUSSIANBLURR_API_URL`** (no trailing slash), e.g. another host or reverse proxy.
- **Timeouts:** uploads **120s**; agent **/run** and **/resume** **600s** (see `ui/api_client.py`).
- **Imports:** `dash_app.py` prepends the repository root to `sys.path` so `python ui/dash_app.py` works without setting `PYTHONPATH`; keeping `PYTHONPATH="$(pwd)"` is still recommended for other tooling.

---

## API

### `POST /run` — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `query` | Yes | User message |
| `session_id` | No | Defaults to `"default"` and is used as the LangGraph `thread_id` |

**Response (JSON):**

| Field | Description |
|-------|-------------|
| `session_id` | Echoed session id |
| `summary` | Last assistant reply |
| `last_tool_result` | Content of last tool message |
| `images` | Logical `agent_filesystem/<session>/run_<tool_call_id>/<file>.png\|svg` paths for every plot produced during this turn. The API discovers them by listing the `run_<tool_call_id>/` folders for the tool calls in the current turn — no scan of `last_tool_result` is required by the client. |
| `interrupted` | `true` if the agent is asking a clarifying question |
| `question` | The clarifying question (only when `interrupted` is `true`) |

### `POST /resume` — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `resume_value` | Yes | The user's answer to the clarifying question |
| `session_id` | Yes | Same session that was interrupted |

**Response:** same shape as `/run`.

### `POST /upload-data` — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `files` | Yes | One or more `.csv` or `.xlsx` uploads |
| `session_id` | No | Defaults to `"default"`; files are stored under `./agent_filesystem/<session-folder>/` and returned as logical `agent_filesystem/<session-folder>/<filename>` |

**Response (JSON):**

| Field | Description |
|-------|-------------|
| `saved` | List of logical `agent_filesystem/<session-folder>/<filename>` paths |
| `count` | Number of files saved |
| `renamed` | List of `{ "original_name", "stored_name" }` when a duplicate filename was avoided (`_2`, `_3`, … before the suffix); empty when no renames |

### `GET /artifact/{session_id}/{path}` — static file (plots & outputs)

Serves one file from **`./agent_filesystem/<session>/`** relative to **`path`** (FastAPI’s `:path` segment; may include slashes, e.g. **`run_call_xyz/trend.png`**). Extensions allowed: **`.csv`**, **`.xlsx`**, **`.png`**, **`.svg`**. Responses: **400** if the resolved path escapes the session root, **403** for other extensions, **404** if missing or not a regular file.

---

## Notes

- **`run_<tool_call_id>/` plots** — Each successful **`code_pipeline`** execution creates a new **`run_<tool_call_id>/`** directory; that tool call's plots live exclusively there. The API computes the response **`images`** list by enumerating these folders for the tool calls in the current turn, so the UI does not parse tool result text to find plots and does not have to diff the whole session folder between questions. **Forecasting tools (`sarima_tool`, `prophet_tool`) intentionally do not write images** — call `code_pipeline` for plots.
- **`data_profile` vs `code_pipeline.data_profile`** — Graph state **`data_profile`** is the auto-refreshed **list** of per-file profiles. The **`code_pipeline`** tool exposes a separate **`data_profile`** **string** parameter for optional codegen hints (often a short summary pasted from the Session workspace list).
- **`active_skills`** — Refreshed every orchestrator step; requirement planning and codegen both read it from graph state so skill routing stays consistent for the turn.
- **Graph recursion cap** — LangGraph stops a single execution when it reaches `graph.recursion_limit`, which is the hard guard against runaway loops in the orchestrator → tools cycle.
- **Ignored files** — `*.docx` is listed in `.gitignore` for local guides; `agent_filesystem/` is ignored as runtime data.
- **Production checkpoints:** see **Checkpoints (InMemory vs Postgres / Neon)** above. Changing **`AgentState`** incompatibly can break old checkpoints—use a new **`session_id`** after migrations.
- If `code_pipeline` is blocked by Semgrep or the LLM judge, read **`code_violation`** in the tool result, pass **`previous_code_violation`** on retry, and adjust rules in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` if needed.
- The Dash UI (`ui/dash_app.py`) supports upload, interrupt/resume, and per-session chat state; HTTP is centralized in **`ui/api_client.py`**.
