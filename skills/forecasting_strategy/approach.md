Use this skill for forecasting and time-series requests.

What to establish:
- What is the forecasting target?
- What is the time column?
- What horizon is requested?
- What is the grain or frequency?
- Is this a single series or multiple related series?

How to reason:
- Do not start forecasting until the time axis and target are clear enough.
- Preserve chronological integrity in every planning decision.
- Avoid leakage from future rows, future-derived features, or random splits.
- Think about evaluation before thinking about model complexity.
- Prefer baseline-first reasoning and explicit assumptions.
- If covariates are used, ask whether they are available at forecast time.

Tooling implications:
- Use `profile_forecasting_data` first when forecast readiness is unclear.
- If data needs custom preparation, use `build_codegen_requirement` to describe the required preprocessing and forecasting workflow before `code_pipeline`.
- Surface unresolved ambiguity about time column, target, join keys, or forecast definition instead of guessing.
