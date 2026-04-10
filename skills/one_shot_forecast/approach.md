Use this skill for quick forecasting requests where the user wants a practical projection, not a full model-selection or backtesting workflow.

What to establish first:
- what is the target
- what is the time column
- what horizon is being projected
- what is the frequency or grain
- is this one series or multiple related series

Core rules:
- Do not forecast until the time axis and target are clear enough.
- Preserve chronological integrity in every decision.
- Avoid random splits and future leakage.
- Prefer a quick trustworthy forecast over a complicated fragile one.
- Be explicit that this is a one-shot projection, not an exhaustive model search.

How to reason:
- Use **Session workspace** `data_profile` (time columns, dtypes, cardinality) early when forecast readiness is unclear; clarify with the user or **`code_pipeline`** as needed.
- Check whether the history length and frequency are adequate for a credible projection.
- If multiple files are involved, only merge covariates that are actually available at forecast time.
- Prefer baseline-first forecasting logic and plain assumptions.
- Surface missing horizon, target ambiguity, or unreliable data conditions instead of pretending confidence.

Answer shape:
- state the projection clearly
- mention the forecast horizon
- mention the main assumptions or caveats
- avoid overstating confidence when the data is thin or irregular

Tooling implications:
- Pair with `tabular_prep` if the forecasting dataset needs joining, aggregation, or cleanup before modeling.
- Use `build_codegen_requirement` to describe the one-shot forecast workflow before `code_pipeline` when execution is needed.
- Do not use this skill as a substitute for full backtesting or best-model selection.
- When using `build_codegen_requirement`, encode the forecast logic clearly:
  - target column
  - time column
  - frequency / grain
  - horizon
  - single-series vs multi-series setup
  - required preprocessing
  - allowed covariates only if available at forecast time
  - expected forecast output shape and saved artifacts
- Do not rely on downstream tools to rediscover the forecasting setup. This skill is responsible for framing the projection correctly and safely.
