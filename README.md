<div align="center">

# GaussianBlurr

### An agentic forecasting workspace for real-world data

Upload a spreadsheet, ask a question in plain English, and receive forecasts, analysis, downloadable data, and charts—without building a notebook first.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## What is GaussianBlurr?

GaussianBlurr turns conversational requests into complete forecasting and data-analysis workflows.

Instead of choosing a model, preparing scripts, and manually moving results between tools, you can upload a CSV or Excel file and describe the outcome you need:

> Forecast monthly revenue for the next 12 months, explain the result, and create a trend chart.

The platform profiles the data, plans the work when needed, selects the appropriate tools, streams progress to the browser, and returns the results in the same conversation.

## Highlights

- **Natural-language analysis** — ask questions without writing Python or SQL.
- **Built-in forecasting** — SARIMA, Prophet, and Holt-Winters workflows with validation, diagnostics, forecasts, and downloadable CSVs.
- **Adaptive planning** — simple requests run directly; multi-step requests are broken into a tracked plan.
- **Custom analysis and visualization** — generated Python can handle transformations and charts beyond the built-in forecasting tools.
- **Live progress** — follow planning, tool execution, and completion events as they happen.
- **Session-based workspace** — every conversation keeps its uploads and generated artifacts isolated.
- **Human-in-the-loop** — the planner can pause for clarification and resume from the same conversation.
- **Layered code safety** — generated code is statically checked, restricted to declared files, and executed in an isolated E2B sandbox without internet access.
- **Optional observability** — inspect application and agent traces with Langfuse.

## Example workflows

Upload a file such as `sales.csv`, then try:

```text
Forecast monthly revenue for the next 12 months. Choose a suitable model,
summarize the forecast, and show the result as a chart.
```

```text
Compare SARIMA, Prophet, and Holt-Winters for this time series.
Explain which result is strongest and why.
```

```text
Check the data quality, clean any obvious issues, analyze the trend and
seasonality, then produce a forecast and downloadable output files.
```

## Quick start

### Prerequisites

- Python 3.10 or newer
- An OpenAI API key
- An E2B account only if you want generated Python analysis and custom charts

### 1. Install

```bash
git clone https://github.com/PalashJauhari/ForecastingPlatform.git
cd ForecastingPlatform

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI key:

```dotenv
OPENAI_API_KEY="your-key"
```

All available settings are documented in [`.env.example`](.env.example). The `.env` file is ignored by Git and should never be committed.

### 3. Run

```bash
./start.sh
```

Then open:

- UI: http://127.0.0.1:8501
- API: http://127.0.0.1:8000

Stop both services with:

```bash
./kill.sh
```

## Enable custom Python analysis

Built-in forecasting works without E2B. Custom transformations and visualizations use an isolated E2B environment.

Add `CODING_E2B_API_KEY` to `.env`, then build the template once:

```bash
python sub_agents/coding_sub_agent/e2b/build_e2b_template.py --write-env
```

The command records the template configuration in your local `.env`. See the [E2B setup guide](sub_agents/coding_sub_agent/e2b/README.md) for details.

## How it works

```text
Upload data
    ↓
Profile the session workspace
    ↓
Decide whether a plan is needed
    ↓
Run forecasting, file, or coding tools
    ↓
Validate and save generated artifacts
    ↓
Stream the final answer, files, and charts to the UI
```

GaussianBlurr uses a main LangGraph workflow with two focused sub-agents:

- The **planner** converts larger requests into a sequence of outcomes and can ask for clarification.
- The **coding agent** generates Python for custom work, validates it, and executes approved code in a sandbox.

The orchestrator coordinates these capabilities and produces the final user-facing response.

## Safety by design

Custom code is never executed directly by the web application.

Before execution, the coding pipeline:

1. Confirms that requested inputs exist in the current session.
2. Restricts inputs and outputs to supported, directory-free filenames.
3. Scans generated Python with Semgrep.
4. Verifies that code accesses only the files declared by the tool call.
5. Executes approved code in a fresh E2B sandbox with internet access disabled.
6. Copies back only the declared output files.

Uploads, generated files, logs, and local environment variables are excluded from version control.

## Project structure

```text
ForecastingPlatform/
├── api/                 Backend application and streaming endpoints
├── graph/               Main agent workflow
├── middleware/          LLM clients, rate limiting, and context handling
├── observability/       Optional Langfuse tracing
├── output_validation/   Structured tool and agent schemas
├── prompts/             Main orchestration prompts
├── sub_agents/
│   ├── planner_sub_agent/
│   └── coding_sub_agent/
├── tools/
│   ├── forecasting/
│   ├── file_management_tools/
│   └── coding_tools/
├── ui/                  Plotly Dash chat interface
└── tests/unit/          Unit test suite
```

## Observability

Langfuse tracing is optional. To enable it, configure the following values in `.env`:

```dotenv
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_SECRET_KEY="..."
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

Traces cover API requests, graph nodes, planning, forecasting pipelines, and sandboxed coding runs.

## License

GaussianBlurr is available under the [MIT License](LICENSE).
