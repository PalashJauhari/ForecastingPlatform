# GaussianBlurr

An analysis assistant built with a custom **LangGraph `StateGraph`**, a single **`config.yaml`** for all configuration, and a logical **`agent_filesystem/`** workspace for all data I/O and generated code. Before each orchestrator turn, **`profile_session_file`** runs **`profiling_data.profile_session_workspace`**, which fills **`data_profile`** in checkpointed state with a **list** of per-file summaries (head rows, dtypes, stats). **Profiling does not write a snapshot file**—only graph state and the orchestrator prompt carry the profile. There is no list/read tool on the graph.

---

## Features

- **Custom StateGraph agent** — orchestrator node + `ToolNode` (parallel tool calling), no `create_agent` black box.
- **YAML-driven config** — orchestrator model, code-generation model, judge model, summarisation model, and middleware thresholds in `config.yaml`.
- **Five bound tools** — `build_codegen_requirement`, `code_pipeline`, `ask_user`, `write_scratchpad`, `write_todos`. Optional modules `list_agent_filesystem_data` / `read_agent_filesystem_data` exist in the repo but are **not** bound to the orchestrator.
- **Workspace profiling (`data_profile`)** — **`profile_session_file`** runs from **START** and again after **`tools`** (`tools` → `profile_session_file` → `orchestrator`). It only updates **`data_profile`** (not `todos` / `scratchpad`). Implementation: **`profiling_data.profile_session_workspace`** discovers `.csv`/`.xlsx` under the session, then **`build_session_data_profile`** builds one dict per path. Success entries include `file`, `row_count`, `head` (5 rows as dicts), `time_columns`, and `columns[]` (`dtype`, `null_pct`, `numeric_pct`, `cardinality`, `possible_categorical`, `time_like`, `continuous_stats` with `min`/`max`/`mean`/`median` for numeric columns); failures are `{ "file", "error" }`. **Nothing is written to disk for profiling.** With no tabular files, **`data_profile` is `[]`**. In state it is always a **Python list**; the orchestrator passes **`json.dumps(..., indent=2)`** of that list into the **Session workspace** block of the HumanMessage (same idea in **`build_codegen_requirement`**).
- **Session planning in state** — `write_todos` replaces the full todo list (`content` + `status` per item); `write_scratchpad` appends a full **`note`** to state while the tool return shown in chat is a short **`summary`** (length-capped). Both merge via `Command`. The orchestrator sees **Current todo list** and **Scratchpad** as JSON each turn.
- **Requirement planning before codegen** — `build_codegen_requirement` reads the **`data_profile`** list (as JSON text in its planner prompt), messages, and **`message_summary`**, plus the orchestrator **`brief`**, before `code_pipeline`.
- **Human-in-the-loop** — `ask_user` tool pauses the graph via `interrupt()`. The API resumes with `Command(resume=...)` when the user replies.
- **Running summarisation** — when the conversation exceeds a configurable token threshold, older messages are summarised into a running summary and truncated (via `RemoveMessage`), keeping the context window manageable.
- **Session-scoped storage** — the model still uses logical `agent_filesystem/...` paths, while the backend resolves them into a private physical workspace for each `session_id`.
- **Code execution pipeline** (`tools/coding_tools/code_pipeline.py`) — LLM codegen → **Semgrep** (static) → **LLM judge** → save `agent_filesystem/code/pipeline_run.py` → subprocess runner.
- **Two-layer code sandbox** —
  - **Layer 1 — Semgrep** (`tools/coding_tools/code_scan/semgrep_scan.py`): static scan on generated source before save, using `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml`.
  - **Layer 2 — Runtime patch** (`tools/coding_tools/code_scan/runtime_patch_scan.py`): monkey-patches `builtins.open`, `pd.read_csv`, `pd.read_excel`, `df.to_csv`, `df.to_excel` for path/extension rules under `agent_filesystem/`. Applied only while the generated script runs.
  - **Runner** (`tools/coding_tools/code_scan/run_pipeline_sandboxed.py`): `code_pipeline` always executes generated code through this script. It applies runtime patches, sets **200 MiB** virtual address limit (`RLIMIT_AS` where supported), caps **BLAS/OpenMP to one thread** via environment variables (before pandas loads), pins the process to **CPU 0** on **Linux** (`sched_setaffinity`), then runs `pipeline_run.py` via `runpy.run_path`.
  - **Sanitized subprocess environment** — the child process that runs generated code receives a **filtered** copy of the parent’s environment: `OPENAI_API_KEY`, all `LANGFUSE_*` variables, and other listed provider credentials are **removed** before `subprocess.run` (see `_env_for_sandbox_subprocess` in `code_pipeline.py`). The API process still has the full env for real LLM calls; the sandbox script cannot read those secrets via `os.environ`, in addition to Semgrep rules that discourage env access in source.
- **Parent timeout** — `code_pipeline` uses `subprocess.run(..., timeout=...)` from `config.yaml` (wall-clock kill of the child process).
- **Skills (optional, repo only)** — `skills/` and `loader.py` remain for future overlay guidance; the graph does **not** run `identify_skills` or inject `skill_context` into the orchestrator right now.
- **Sessions** — `thread_id = session_id` with `InMemorySaver` (per-session checkpoints; lost on restart).
- **Observability** — optional Langfuse callbacks when `LANGFUSE_*` keys are set, propagated via `RunnableConfig`.

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

---

## Repository layout

```
config.yaml                                  # Models, middleware, paths, code_pipeline, skills budget
agent_sessions/
  <session_id>/
    input/                                   # Raw uploaded files (CSV, Excel)
    output/                                  # Agent-written results and plots
    processed/                               # Intermediate cleaned/transformed data
    scratchpad/                              # optional on-disk folder (agent notes use state + write_scratchpad)
    code/                                    # pipeline_run.py (generated; overwritten each run)
graph/
  graph.py                                   # StateGraph, AnalysisGraph
  __init__.py
skills/                                      # Curated reasoning guidance (approach.md per skill)
  loader.py                                  # Skill markdown assembly (not wired into graph currently)
  data_science_workflow/ … tabular_prep/ … metric_answering/ …
  visual_answering/ … one_shot_forecast/ …
middleware/
  context_editing.py                         # Truncate messages at safe turn boundaries
  summarization.py                           # Running summary of evicted messages
  tool_call_limit.py                         # Optional per-turn cap helper (not used by graph)
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
    profiling_data.py                        # profile_session_workspace + build_session_data_profile (graph profiling)
    list_agent_filesystem_data.py            # Standalone; not bound to the graph
    read_agent_filesystem_data.py            # Standalone; not bound to the graph
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
  app.py                                     # Streamlit UI
observability/
  langfuse_handler.py
```

---

## Middleware (plain functions, called from orchestrator node)

| Function | Module | Role |
|----------|--------|------|
| `truncate_and_summarize` | `middleware/context_editing.py` | Truncation + running summary when token estimate exceeds threshold. |
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
  summarization: "gpt-4o-mini"      # Summary LLM for context eviction
  # codegen_requirement: "gpt-4o-mini"  # optional; build_codegen_requirement defaults to orchestrator if omitted

middleware:
  context_editing:
    keep_recent_messages: 10         # Messages to keep after truncation
  summarization:
    token_threshold: 100000          # Rough token estimate to trigger summarisation

code_pipeline:
  timeout_seconds: 120               # Wall-clock subprocess timeout for generated script

skills:
  max_skill_context_tokens: 3000     # Reserved for future skill overlays (graph does not load skills today)

paths:
  agent_filesystem: "agent_filesystem"
  sessions_root: "agent_sessions"
  input:      "agent_filesystem/input"
  output:     "agent_filesystem/output"
  processed:  "agent_filesystem/processed"
  scratchpad: "agent_filesystem/scratchpad"
  code:       "agent_filesystem/code"
```

Semgrep rules live in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` — edit to tune blocking. Memory limit (200 MiB), BLAS thread caps, and Linux CPU affinity are defined in `tools/coding_tools/code_scan/run_pipeline_sandboxed.py`, not in YAML.

---

## Environment (`.env`)

- **`.env`** — your real secrets live here. It is listed in `.gitignore` (along with `*.env` / `.env.*`); **never commit it**.
- **`.env.example`** — safe template committed to the repo: same variable **names**, placeholder or empty values, and short comments. New setups: `cp .env.example .env`, then edit `.env`.

`python-dotenv` loads the project-root `.env` into the process environment at startup:

- **`api/main.py`** — loads before the graph is imported (FastAPI / `uvicorn`).
- **`graph/graph.py`** — loads at the top of the module so `OPENAI_API_KEY` is set **before** module-level `ChatOpenAI` clients are constructed (e.g. scripts or tests that import `graph` without going through `main`).

LangChain’s `ChatOpenAI` reads **`OPENAI_API_KEY` from the environment** by default (OpenAI SDK convention); you do not pass the key in code.

The Streamlit UI talks to the HTTP API only and does **not** need an OpenAI key in the browser.

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (loaded into `os.environ` for backend LLM calls) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing |

---

## Run

```bash
pip install -r requirements.txt
cp .env.example .env    # add OPENAI_API_KEY
./launch.sh
```

- **API:** `http://127.0.0.1:8000`
- **UI:** `http://localhost:8501`

**Manual (two terminals):**

```bash
export PYTHONPATH="$(pwd)"
uvicorn api.main:app --host 127.0.0.1 --port 8000
streamlit run ui/app.py
```

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
| `session_id` | No | Defaults to `"default"`; files are stored in that session's private physical workspace and returned as logical `agent_filesystem/input/...` paths |

**Response (JSON):**

| Field | Description |
|-------|-------------|
| `saved` | List of logical `agent_filesystem/input/...` paths |
| `count` | Number of files saved |

---

## Notes

- **`data_profile` vs `code_pipeline.data_profile`** — Graph state **`data_profile`** is the auto-refreshed **list** of per-file profiles. The **`code_pipeline`** tool exposes a separate **`data_profile`** **string** parameter for optional codegen hints (often a short summary pasted from the Session workspace list).
- **Per-turn tool cap** — not enforced in the graph right now; `middleware/tool_call_limit.py` remains if you want to wire a limit back in.
- **Production:** replace `InMemorySaver` with a persistent checkpointer (e.g. `PostgresSaver`) so conversations survive restarts. Changing **`AgentState`** (field names or types such as **`data_profile`**) can break old checkpoints—use a new **`session_id`** after migrations.
- If `code_pipeline` is blocked by Semgrep or the LLM judge, check `code_safety_evaluation.detail` in the tool result, pass `previous_code_violation` on retry, and adjust rules in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` if needed.
- The Streamlit UI (`ui/app.py`) supports upload, interrupt/resume, and per-session chat state.
