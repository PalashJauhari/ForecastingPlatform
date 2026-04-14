# GaussianBlurr

An analysis assistant built with a custom **LangGraph `StateGraph`**, a single **`config.yaml`** for all configuration, and a **flat `agent_filesystem/<session>/`** workspace (uploads, outputs, and `pipeline_run.py` share one folder per session). Logical paths look like **`agent_filesystem/<session>/<filename>`**. Before each orchestrator turn, **`profile_session_file`** runs **`profiling_data.profile_session_workspace`**, which fills **`data_profile`** in checkpointed state with a **list** of per-file summaries (row counts, column names, preview rows). **Profiling does not write a snapshot file**—only graph state and the orchestrator prompt carry the profile. There is no list/read tool on the graph.

---

## Features

- **Custom StateGraph agent** — orchestrator node + `ToolNode` (parallel tool calling), no `create_agent` black box.
- **YAML-driven config** — orchestrator model, code-generation model, judge model, summarisation model, and middleware thresholds in `config.yaml`.
- **OpenAI rate limiting (server-side)** — optional shared [`InMemoryRateLimiter`](https://python.langchain.com/docs/integrations/chat/openai/#rate-limiting) on every backend `ChatOpenAI` (orchestrator, summarisation, codegen, requirement planner, judge); tuned under `llm_rate_limit` in `config.yaml`.
- **Five bound tools** — `build_codegen_requirement`, `code_pipeline`, `ask_user`, `write_scratchpad`, `write_todos`.
- **Workspace profiling (`data_profile`)** — **`profile_session_file`** runs from **START** and again after **`tools`** (`tools` → `profile_session_file` → `orchestrator`). It only updates **`data_profile`** (not `todos` / `scratchpad` / `tool_call_count`). **`profiling_data.profile_session_workspace`** walks the whole session directory tree, collects every `.csv`/`.xlsx`, and returns **`build_session_data_profile`**’s list (one dict per path). Success entries include `file`, `row_count`, `columns` (names), and `head` (first five rows as dicts); failures are `{ "file", "error" }`. **Nothing is written to disk for profiling.** With no tabular files, **`data_profile` is `[]`**. In state it is always a **Python list**; the orchestrator passes **`json.dumps(..., indent=2)`** of that list into the **Session workspace** block of the HumanMessage (same idea in **`build_codegen_requirement`**).
- **Session planning in state** — `write_todos` replaces the full todo list (`content` + `status` per item); `write_scratchpad` appends a full **`note`** to state while the tool return shown in chat is a short **`summary`** (length-capped). Both merge via `Command`. The orchestrator sees **Current todo list** and **Scratchpad** as JSON each turn.
- **Requirement planning before codegen** — `build_codegen_requirement` reads the **`data_profile`** list (as JSON text in its planner prompt), messages, and **`message_summary`**, plus the orchestrator **`brief`**, before `code_pipeline`.
- **Human-in-the-loop** — `ask_user` tool pauses the graph via `interrupt()`. The API resumes with `Command(resume=...)` when the user replies.
- **Running summarisation** — when the conversation exceeds a configurable token threshold, older messages are summarised into a running summary and truncated (via `RemoveMessage`), keeping the context window manageable.
- **Session-scoped storage** — logical paths and disk paths share the same tree under `./agent_filesystem/<session>/...` (resolved from the repo root).
- **Per-session tool budget** — `middleware.tool_call_limit.check_tool_call_limit` runs at the start of each `orchestrator` step; `tool_call_count` in graph state accumulates each batch of tool calls the model requests (parallel calls count separately). When the counter is already ≥ `max_calls`, the orchestrator returns a final `AIMessage` with **no** `tool_calls` **without invoking the LLM** on that step, so routing goes straight to `END`.
- **Code execution pipeline** (`tools/coding_tools/code_pipeline.py`) — LLM codegen → **Semgrep** (static) → **LLM judge** → save `agent_filesystem/<session>/pipeline_run.py` → subprocess runner.
- **Two-layer code sandbox** —
  - **Layer 1 — Semgrep** (`tools/coding_tools/code_scan/semgrep_scan.py`): static scan on generated source before save, using `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml`.
  - **Layer 2 — Runtime patch** (`tools/coding_tools/code_scan/runtime_patch_scan.py`): monkey-patches `builtins.open`, `pd.read_csv`, `pd.read_excel`, `df.to_csv`, `df.to_excel` for path/extension rules under `agent_filesystem/`. Applied only while the generated script runs.
  - **Runner** (`tools/coding_tools/code_scan/run_pipeline_sandboxed.py`): `code_pipeline` always executes generated code through this script. It applies runtime patches, sets **200 MiB** virtual address limit (`RLIMIT_AS` where supported), caps **BLAS/OpenMP to one thread** via environment variables (before pandas loads), pins the process to **CPU 0** on **Linux** (`sched_setaffinity`), then runs `pipeline_run.py` via `runpy.run_path`.
  - **Sanitized subprocess environment** — the child process that runs generated code receives a **filtered** copy of the parent’s environment: `OPENAI_API_KEY`, all `LANGFUSE_*` variables, and other listed provider credentials are **removed** before `subprocess.run` (see `_env_for_sandbox_subprocess` in `code_pipeline.py`). The API process still has the full env for real LLM calls; the sandbox script cannot read those secrets via `os.environ`, in addition to Semgrep rules that discourage env access in source.
- **Parent timeout** — `code_pipeline` uses `subprocess.run(..., timeout=...)` from `config.yaml` (wall-clock kill of the child process).
- **Skills (optional, repo only)** — `skills/` and `loader.py` remain for future overlay guidance; the graph does **not** run `identify_skills` or inject `skill_context` into the orchestrator right now.
- **Sessions** — `thread_id = session_id`. Checkpoints use `InMemorySaver` by default, or Postgres when `checkpointer.use_neon: true` and `DATABASE_URL` is set (e.g. Neon; survives API restarts).
- **Observability** — optional **Langfuse** tracing: graph nodes and tools use `@observe` / nested spans; **`api/main.py`** wraps **`POST /run`** and **`POST /resume`** in **`propagate_attributes(...)`** so `session_id`, tags, and small request metadata attach to traces. **`observability/langfuse_handler.get_langfuse_client()`** uses the Langfuse SDK; if you set **`LANGFUSE_BASE_URL`** but not **`LANGFUSE_HOST`**, the client copies it into **`LANGFUSE_HOST`** for compatibility.

---

## Graph architecture

```
START → profile_session_file → orchestrator → [has tool calls?]
                                                                        ├─ YES → ToolNode → profile_session_file → orchestrator (loop)
                                                                        └─ NO  → END
```

**State schema** (`graph/graph.py` — `AgentState`):

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `Annotated[list, add_messages]` | Full conversation — `HumanMessage`, `AIMessage`, `ToolMessage`. Supports `RemoveMessage` for truncation. |
| `message_summary` | `str` | Running summary of evicted messages. |
| `data_profile` | `list` | Replaced every time **`profile_session_file`** runs. If the value is not a list (corrupt state), the orchestrator treats it as **`[]`**. Rendered for the LLM via **`json.dumps`** in the **Session workspace** block. |
| `todos` | `list[dict]` | Optional. Each item: `content` (str), `status` (`pending` \| `in_progress` \| `completed`). Replaced entirely on each `write_todos` call. |
| `scratchpad` | `Annotated[list[str], operator.add]` | Session notes, oldest→newest. Each `write_scratchpad` call supplies `[note]`; `operator.add` concatenates lists (list `+`). |
| `tool_call_count` | `int` | Cumulative orchestrator-requested tool calls for the session; each successful orchestrator step adds `len(tool_calls)` (after `ask_user`-in-batch rejection clears the batch). Compared against `middleware.tool_call_limit.max_calls` **before** the LLM runs. |

---

## Repository layout

```
config.yaml                                  # Models, middleware, graph, llm_rate_limit, checkpointer (use_neon), …
agent_filesystem/                          # Runtime data (gitignored); flat per session
  <session_id>/                              # Uploads, CSV/XLSX outputs, plots, pipeline_run.py
graph/
  graph.py                                   # StateGraph, AnalysisGraph
  __init__.py
skills/                                      # Curated reasoning guidance (approach.md per skill)
  loader.py                                  # Skill markdown assembly (not wired into graph currently)
  data_science_workflow/ … tabular_prep/ … metric_answering/ …
  visual_answering/ … one_shot_forecast/ …
middleware/
  context_editing.py                         # Truncate messages at safe turn boundaries
  (message summarisation lives in ``context_editing.py`` — ``truncate_and_summarize``)
  llm_rate_limit.py                          # Shared ``InMemoryRateLimiter`` for ``ChatOpenAI`` (``config.yaml`` → ``llm_rate_limit``)
  tool_call_limit.py                         # Per-session cap helper (used from orchestrator)
  __init__.py
tools/
  human_in_loop/
    ask_user.py                              # Tool — interrupt-based clarifying question
  planning/
    write_scratchpad.py                      # Write note → state scratchpad (append); ToolMessage uses summary
    write_todos.py                           # Replace session todo list
  coding_tools/
    build_codegen_requirement.py            # Requirement builder before code_pipeline
    code_pipeline.py                         # Codegen → Semgrep → judge → save → run
    code_scan/
      run_pipeline_sandboxed.py              # Runtime patches + RLIMIT + BLAS env + CPU pin (Linux)
      semgrep_scan.py                        # Layer 1 — Semgrep
      llm_judge.py                           # Post-Semgrep LLM review
      runtime_patch_scan.py                  # Layer 2 — I/O monkey-patches
      codegen_scan_semgrep.yaml              # Semgrep rules
  file_management_tools/
    profiling_data.py                        # profile_session_workspace (graph profiling → data_profile)
prompts/
  graph_prompts.py                           # Orchestrator SYSTEM_PROMPT
  build_codegen_requirement_prompt.py        # Requirement-builder system prompt
  skills_prompts.py                          # Skill identification prompt (unused by graph currently)
  code_generation_prompt.py                  # Codegen system prompt
  code_judge_prompt.py                       # Judge system prompt
output_validation/
  build_codegen_requirement.py               # Pydantic output validation for requirement builder
  code_generation.py                         # Pydantic output validation for code generation
  judge_output.py                            # Pydantic output validation for the LLM judge
  skill_selection.py                         # Pydantic for skill selection (unused by graph currently)
  scratchpad.py                              # Pydantic args for ``write_scratchpad``
  write_todos.py                             # Pydantic models for ``write_todos`` tool args
api/
  main.py                                    # POST /run, /resume, /upload-data
ui/
  __init__.py
  api_client.py                              # HTTP client for FastAPI (upload / run / resume)
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
| `OPENAI_RATE_LIMITER` | `middleware/llm_rate_limit.py` | Optional `InMemoryRateLimiter` built from `llm_rate_limit` in `config.yaml`; attached to every `ChatOpenAI` in the graph and coding tools. |
| `check_tool_call_limit` | `middleware/tool_call_limit.py` | Returns a stop `AIMessage` when session `tool_call_count` ≥ `max_calls` (checked at the top of `orchestrator`, before summarisation and the LLM). |
| — | `code_pipeline` (tool) | Codegen → Semgrep → LLM judge → save → `run_pipeline_sandboxed.py` execution. |

**Graph nodes (non-middleware):** `profile_session_file`, `orchestrator`, `tools`.

---

## Tools

| Tool | Arguments | What it does |
|------|-----------|--------------|
| **build_codegen_requirement** | `brief` | Reads **`data_profile`** and conversation state; returns a validated execution requirement before the orchestrator calls `code_pipeline` or clarifies with the user. |
| **code_pipeline** | `task`, `data_profile`, optional `previous_code_violation` | Orchestrator-supplied **`data_profile`** is a **string** argument: optional extra hints for codegen (summarize from **Session workspace** list in context). Pipeline: structured codegen → Semgrep → LLM judge → `pipeline_run.py` → `run_pipeline_sandboxed.py`. Returns JSON with `code_generation` and `execution`. On failure, retry with `previous_code_violation` set from `code_safety_evaluation.detail`. |
| **ask_user** | `question` | Pauses the graph via `interrupt()` and surfaces a clarifying question to the user. Must be the only tool call in the step. |
| **write_scratchpad** | `note`, `summary` | Appends `note` via `Command`; `summary` is the `ToolMessage` (≤2 lines, ≤400 chars). Full entries are JSON in **Scratchpad** each turn. |
| **write_todos** | `todos` | Replaces the session todo list via `Command`. `ToolMessage` confirms count only; full list is in state and in **Current todo list** JSON each turn (`[]` if empty). |

---

## Configuration (`config.yaml`)

```yaml
models:
  orchestrator: "gpt-4o-mini"       # Main agent LLM
  code_generation: "gpt-4o-mini"    # Codegen LLM inside code_pipeline
  code_judge: "gpt-4o-mini"         # Judge after Semgrep
  message_summarisation: "gpt-4o-mini"   # Summary LLM for context eviction
  # codegen_requirement: "gpt-4o-mini"  # optional; build_codegen_requirement defaults to orchestrator if omitted

middleware:
  context_editing:
    keep_recent_messages: 10         # Messages to keep after truncation
  message_summarisation:
    token_threshold: 100000          # Rough token estimate to trigger summarisation
  tool_call_limit:
    max_calls: 25                    # Session-wide cap on orchestrator-requested tool calls (parallel = multiple)

graph:
  recursion_limit: 100               # Max LangGraph super-steps per ``invoke`` (run + resume)
  max_concurrency: 2                 # Max parallel runnable work where LangGraph applies it

# Shared bucket for all ``ChatOpenAI`` clients (set ``enabled: false`` to disable).
llm_rate_limit:
  enabled: true
  requests_per_second: 1.0
  check_every_n_seconds: 0.1
  max_bucket_size: 2.0

code_pipeline:
  timeout_seconds: 120               # Wall-clock subprocess timeout for generated script

skills:
  max_skill_context_tokens: 3000     # Reserved for future skill overlays (graph does not load skills today)

paths:
  agent_filesystem: "agent_filesystem"   # logical prefix + ./agent_filesystem/ on disk
```

Logical paths: `agent_filesystem/<session-folder>/<file>` (flat folder on disk under `./agent_filesystem/<session-folder>/`).

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

---

## Notes

- **`data_profile` vs `code_pipeline.data_profile`** — Graph state **`data_profile`** is the auto-refreshed **list** of per-file profiles. The **`code_pipeline`** tool exposes a separate **`data_profile`** **string** parameter for optional codegen hints (often a short summary pasted from the Session workspace list).
- **Tool call cap** — per-session total via `middleware.tool_call_limit.max_calls` and `tool_call_count` state. The guard runs **before** the LLM on each orchestrator step; a single assistant step can still increment the counter by the full parallel batch size, so the running total can land **slightly above** `max_calls` if the model requests many tools in one turn when the counter was already near the limit. To reset the budget on every new user message instead, pass `tool_call_count: 0` in the dict passed to `graph.invoke` for that turn (overrides the checkpoint for that key).
- **Ignored files** — `*.docx` is listed in `.gitignore` for local guides; `agent_filesystem/` is ignored as runtime data.
- **Production checkpoints:** see **Checkpoints (InMemory vs Postgres / Neon)** above. Changing **`AgentState`** incompatibly can break old checkpoints—use a new **`session_id`** after migrations.
- If `code_pipeline` is blocked by Semgrep or the LLM judge, check `code_safety_evaluation.detail` in the tool result, pass `previous_code_violation` on retry, and adjust rules in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` if needed.
- The Dash UI (`ui/dash_app.py`) supports upload, interrupt/resume, and per-session chat state; HTTP is centralized in **`ui/api_client.py`**.
