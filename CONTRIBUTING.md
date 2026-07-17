# Contributing

Thanks for helping improve GaussianBlurr / ForecastingPlatform.

## Prerequisites

- Python **3.10+** (3.12 recommended)
- A virtualenv and `pip install -r requirements.txt`
- Root `.env` copied from `.env.example` (at least `OPENAI_API_KEY` for full app runs)

## Local workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test          # or: pytest
./start.sh         # API :8000 + UI :8501
./kill.sh          # stop both
```

## Pull requests

1. Keep changes focused; match existing style and layout.
2. Add or update unit tests under `tests/unit/` when behavior changes.
3. Do not commit secrets, `.env`, or `agent_filesystem/` data.
4. Prefer clear commit messages that explain **why**.

## Where to change things

| Area | Location |
|------|----------|
| Main graph | `graph/graph.py` |
| Planner | `sub_agents/planner_sub_agent/` |
| Coding sandbox | `sub_agents/coding_sub_agent/` |
| Forecasting tools | `tools/forecasting/` (inherit `ForecastingUnivariateModel`) |
| API | `api/main.py` |
| Prompts | `prompts/`, `sub_agents/*/prompts.py` |

**Planner rule:** todo text describes outcomes only — never name orchestrator tools (`sarima_tool`, `coding_tool`, etc.).

## License

By contributing, you agree your contributions are licensed under the MIT License (see `LICENSE`).
