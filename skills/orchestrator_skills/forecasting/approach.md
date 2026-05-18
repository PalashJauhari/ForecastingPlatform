# Skill: Forecasting & Time Series Hygiene

This skill covers the mental models for preparing time-stamped data and generating reliable predictions.

## Core Reasoning Principles

1.  **Chronological Integrity**:
    *   Time series data must be continuous. Check for missing dates/time-steps.
    *   If gaps exist, explicitly decide on an imputation strategy: `ffill`, `bfill`, `interpolation`, or `zero-filling`.

2.  **Stationarity and Seasonality**:
    *   Observe if the series has a trend (non-constant mean) or changing variance.
    *   Identify dominant seasonal patterns (Daily, Weekly, Monthly, Yearly).
    *   Acknowledge "Change Points" where the behavior of the series fundamentally shifted.

3.  **The "Future Leakage" Guardrail**:
    *   Ensure that any features (covariates) used to train the model are actually known at the time of the forecast.
    *   Never use "Total Sales" from the future to predict "Daily Sales" today.

4.  **Evaluation and baselines**:
    *   For holdout design, chronological backtests, metric choice (MAPE, WAPE, bias), and “beat the baseline” rules, follow the **evaluation-design** skill—avoid duplicating that playbook here.

## Tool mapping (forecasting)

- **`sarima_tool`** — Use for a structured SARIMA / auto-ARIMA path on a session CSV/XLSX when you want a validated fit, diagnostics, and a saved forecast table at session root; follow with **`code_pipeline`** if you need publication-style plots (this platform routes plots through codegen, not SARIMA’s JSON alone).
- **`prophet_tool`** — Use similarly for Prophet additive models when seasonality and holidays matter; interpret uncertainty intervals in user language; add **`code_pipeline`** for extra visuals if needed.
- **`code_pipeline`** — Use for custom feature engineering, multiple series, nonstandard seasonality, or any **image** output (matplotlib saves only under the tool’s run folder). Use **`detail_execution_requirement_first: true`** inside **`code_pipeline`** when complex codegen needs a staged execution brief first.
- **`ask_user`** — Use when horizon, seasonality, or file/column choice is ambiguous and blocks a defensible forecast.
