# GaussianBlurr

An analysis assistant built with a custom **LangGraph `StateGraph`**, a single **`config.yaml`** for all configuration, and an **`agent_filesystem/`** workspace for all data I/O and generated code.

---

## Features

- **Custom StateGraph agent** — orchestrator node + `ToolNode` (parallel tool calling), no `create_agent` black box.
- **YAML-driven config** — orchestrator model, code-generation model, judge model, summarisation model, and middleware thresholds in `config.yaml`.
- **Six tools** — `list_agent_filesystem_data`, `read_agent_filesystem_data`, `code_pipeline`, `ask_user`, `read_scratchpad`, `write_scratchpad`.
- **Human-in-the-loop** — `ask_user` tool pauses the graph via `interrupt()`. The API resumes with `Command(resume=...)` when the user replies.
- **Running summarisation** — when the conversation exceeds a configurable token threshold, older messages are summarised into a running summary and truncated (via `RemoveMessage`), keeping the context window manageable.
- **Auto-refreshed file listing** — a `refresh_data_schema` node runs before every orchestrator call, scanning `agent_filesystem/` so the LLM always sees the current file listing without a tool call.
- **Code execution pipeline** (`tools/coding_tools/code_pipeline.py`) — LLM codegen → **Semgrep** (static) → **LLM judge** → save `agent_filesystem/code/pipeline_run.py` → subprocess runner.
- **Two-layer code sandbox** —
  - **Layer 1 — Semgrep** (`tools/coding_tools/code_scan/semgrep_scan.py`): static scan on generated source before save, using `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml`.
  - **Layer 2 — Runtime patch** (`tools/coding_tools/code_scan/runtime_patch_scan.py`): monkey-patches `builtins.open`, `pd.read_csv`, `pd.read_excel`, `df.to_csv`, `df.to_excel` for path/extension rules under `agent_filesystem/`. Applied only while the generated script runs.
  - **Runner** (`tools/coding_tools/code_scan/run_pipeline_sandboxed.py`): `code_pipeline` always executes generated code through this script. It applies runtime patches, sets **200 MiB** virtual address limit (`RLIMIT_AS` where supported), caps **BLAS/OpenMP to one thread** via environment variables (before pandas loads), pins the process to **CPU 0** on **Linux** (`sched_setaffinity`), then runs `pipeline_run.py` via `runpy.run_path`.
- **Parent timeout** — `code_pipeline` uses `subprocess.run(..., timeout=...)` from `config.yaml` (wall-clock kill of the child process).
- **Skills layer** — `identify_skills` (one JSON LLM call on each new user turn) picks from five curated skills (`eda`, `data_processing`, `feature_engineering`, `modeling`, `visualisation`). `skills/loader.py` loads `approach.md` + reference `.py` patterns into `skill_context`, injected into the orchestrator context as “Skill guidance” (budget: `skills.max_skill_context_tokens`).
- **Sessions** — `thread_id = session_id` with `InMemorySaver` (per-session checkpoints; lost on restart).
- **Observability** — optional Langfuse callbacks when `LANGFUSE_*` keys are set, propagated via `RunnableConfig`.

---

## Graph architecture

```
START → refresh_data_schema → identify_skills → reset_tool_budget → orchestrator → [has tool calls?]
                                                                        ├─ YES → ToolNode → refresh_data_schema (loop)
                                                                        └─ NO  → END
```

**State schema** (`graph/graph.py` — `AgentState`):

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `Annotated[list, add_messages]` | Full conversation — `HumanMessage`, `AIMessage`, `ToolMessage`. Supports `RemoveMessage` for truncation. |
| `message_summary` | `str` | Running summary of evicted messages. |
| `number_of_tool_calls` | `int` | Total individual tool invocations in the current user turn (reset per invocation). |
| `data_schema` | `str` | Newline-separated file listing from `agent_filesystem/` (excluding `scratchpad/`). |
| `active_skills` | `list[str]` | Skill ids selected on the latest user turn (`identify_skills`). |
| `skill_context` | `str` | Loaded guidance text from `skills/` for those ids. |

---

## Repository layout

```
config.yaml                                  # Models, middleware, paths, code_pipeline, skills budget
agent_filesystem/
  input/                                     # Raw uploaded files (CSV, Excel)
  output/                                    # Agent-written results and plots
  processed/                                 # Intermediate cleaned/transformed data
  scratchpad/                                # scratchpad.md — working notes
  code/                                      # pipeline_run.py (generated; overwritten each run)
graph/
  graph.py                                   # StateGraph, identify_skills, AnalysisGraph
  __init__.py
skills/                                      # Curated DS guidance (approach.md + reference *.py per skill)
  loader.py                                  # Assembles skill_context for the orchestrator
  eda/ … data_processing/ … feature_engineering/ … modeling/ … visualisation/
middleware/
  context_editing.py                         # Truncate messages at safe turn boundaries
  summarization.py                           # Running summary of evicted messages
  tool_call_limit.py                         # Abort agent loop when budget exceeded
  __init__.py
tools/
  human_in_loop/
    ask_user.py                              # Tool — interrupt-based clarifying question
  coding_tools/
    code_pipeline.py                         # Codegen → Semgrep → judge → save → run
    code_scan/
      run_pipeline_sandboxed.py              # Runtime patches + RLIMIT + BLAS env + CPU pin (Linux)
      semgrep_scan.py                        # Layer 1 — Semgrep
      llm_judge.py                           # Post-Semgrep LLM review
      runtime_patch_scan.py                  # Layer 2 — I/O monkey-patches
      codegen_scan_semgrep.yaml              # Semgrep rules
  file_management_tools/
    list_agent_filesystem_data.py
    read_agent_filesystem_data.py
    read_scratchpad.py
    write_scratchpad.py
prompts/
  graph_prompts.py                           # Orchestrator SYSTEM_PROMPT
  skills_prompts.py                          # Skill identification (JSON) system prompt
  code_generation_prompt.py                  # Codegen system prompt
  code_judge_prompt.py                       # Judge system prompt
api/
  main.py                                    # POST /run, POST /resume
ui/
  app.py                                     # Streamlit UI
observability/
  langfuse_handler.py
```

---

## Middleware (plain functions, called from orchestrator node)

| Function | Module | Role |
|----------|--------|------|
| `check_tool_call_limit` | `middleware/tool_call_limit.py` | Returns early-stop `AIMessage` when `max_calls` exceeded. Parallel calls count individually. |
| `truncate_and_summarize` | `middleware/context_editing.py` | Truncation + running summary when token estimate exceeds threshold. |
| — | `code_pipeline` (tool) | Codegen → Semgrep → LLM judge → save → `run_pipeline_sandboxed.py` execution. |

**Graph nodes (non-middleware):** `refresh_data_schema`, `identify_skills` (JSON skill picker + `load_skills`), `reset_tool_budget`, `orchestrator`, `tools`.

---

## Tools

| Tool | Arguments | What it does |
|------|-----------|--------------|
| **list_agent_filesystem_data** | none | Scans `agent_filesystem/` recursively and returns all `.csv` and `.xlsx` files as `agent_filesystem/...` paths. |
| **read_agent_filesystem_data** | `path`, `n_rows` (optional, default 5, max 100) | Opens a `.csv` or `.xlsx` file and returns columns, preview rows, and total row count. Path must start with `agent_filesystem/`. |
| **code_pipeline** | `task`, `data_schema`, optional `previous_code_violation` | Codegen (JSON) → Semgrep → LLM judge → writes `agent_filesystem/code/pipeline_run.py` → runs it via `run_pipeline_sandboxed.py`. Returns JSON with `code_generation` and `execution`. On Semgrep/judge failure, pass `previous_code_violation` on retry with the prior `code_safety_evaluation.detail`. |
| **ask_user** | `question` | Pauses the graph via `interrupt()` and surfaces a clarifying question to the user. Must be the only tool call in the step. |
| **read_scratchpad** | none | Reads `agent_filesystem/scratchpad/scratchpad.md`. |
| **write_scratchpad** | `content` | Appends text to `agent_filesystem/scratchpad/scratchpad.md`. Cleared automatically on new sessions. |

---

## Configuration (`config.yaml`)

```yaml
models:
  orchestrator: "gpt-4o-mini"       # Main agent LLM
  code_generation: "gpt-4o-mini"    # Codegen LLM inside code_pipeline
  code_judge: "gpt-4o-mini"         # Judge after Semgrep
  summarization: "gpt-4o-mini"      # Summary LLM for context eviction

middleware:
  context_editing:
    keep_recent_messages: 10         # Messages to keep after truncation
  summarization:
    token_threshold: 100000          # Rough token estimate to trigger summarisation
  tool_call_limit:
    max_calls: 15                    # Max individual tool calls per user turn

code_pipeline:
  timeout_seconds: 120               # Wall-clock subprocess timeout for generated script

skills:
  max_skill_context_tokens: 3000     # Rough budget for assembled skill_context (~chars/4)

paths:
  agent_filesystem: "agent_filesystem"
  input:      "agent_filesystem/input"
  output:     "agent_filesystem/output"
  processed:  "agent_filesystem/processed"
  scratchpad: "agent_filesystem/scratchpad"
  code:       "agent_filesystem/code"
```

Semgrep rules live in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` — edit to tune blocking. Memory limit (200 MiB), BLAS thread caps, and Linux CPU affinity are defined in `tools/coding_tools/code_scan/run_pipeline_sandboxed.py`, not in YAML.

---

## Environment (`.env`)

Copy `.env.example` → `.env` and fill in your keys. **Never commit `.env` to git.**

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
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
| `file` | No | CSV/Excel upload (stored under `agent_filesystem/input/<session_id>/`) |
| `csv_path` | No | Path for follow-up when no new file |
| `session_id` | No | Defaults to `"default"` |

**Response (JSON):**

| Field | Description |
|-------|-------------|
| `session_id` | Echoed session id |
| `csv_path` | Resolved upload path |
| `summary` | Last assistant reply |
| `last_tool_result` | Content of last tool message |
| `interrupted` | `true` if the agent is asking a clarifying question |
| `question` | The clarifying question (only when `interrupted` is `true`) |

### `POST /resume` — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `resume_value` | Yes | The user's answer to the clarifying question |
| `session_id` | Yes | Same session that was interrupted |
| `csv_path` | No | Carried forward from the original `/run` call |

**Response:** same shape as `/run`.

---

## Notes

- **Production:** replace `InMemorySaver` with a persistent checkpointer (e.g. `PostgresSaver`) so conversations survive restarts.
- If `code_pipeline` is blocked by Semgrep or the LLM judge, check `code_safety_evaluation.detail` in the tool result, pass `previous_code_violation` on retry, and adjust rules in `tools/coding_tools/code_scan/codegen_scan_semgrep.yaml` if needed.
- The Streamlit UI (`ui/app.py`) has not yet been updated for interrupt/resume — it still calls only `POST /run`.
