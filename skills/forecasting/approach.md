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

4.  **Validation over Accuracy**:
    *   Avoid simple random splits. Use **TimeSeriesSplit** or **Hold-out Backtesting** (testing on the most recent N days).
    *   Report multiple metrics: MAPE (accuracy), WAPE (volume-weighted accuracy), and Bias (over/under predicting).

5.  **Baseline First**:
    *   Always compare a complex model (Prophet, XGBoost) against a "Naive" baseline (e.g., "Next week = Last week").
    *   If the complex model doesn't beat the baseline, keep it simple.
