# Skill: Feature Engineering

This skill guides how to design predictive inputs that are useful, realistic, and safe from leakage.

## Core Reasoning Principles

1.  **Features Must Exist at Prediction Time**:
    *   Only use information that would actually be known when the prediction is made.
    *   Lag features, rolling statistics, and external regressors must be built from past or known-future information only.
    *   If a feature depends on future outcomes, it is leakage, not signal.

2.  **Match Features to the Problem Structure**:
    *   Time-based problems may need lags, trends, seasonality flags, and calendar features.
    *   Cross-sectional problems may need ratios, group summaries, encodings, or domain transforms.
    *   Feature design should follow the grain, target, and business logic of the task.

3.  **Prefer Signal Over Volume**:
    *   More features do not automatically improve the solution.
    *   Favor features with a clear causal or behavioral story over mechanically generated noise.
    *   Keep the set compact when the data is limited or the grain is sparse.

4.  **Engineer Features in a Stable Way**:
    *   Rolling windows, aggregations, and encodings should be computed consistently across train and evaluation periods.
    *   Be careful with rare categories, sparse groups, and unstable denominators.
    *   A feature that behaves differently across time or segments can mislead the model.

5.  **Feature Choices Should Be Explainable**:
    *   Every important feature should have a reason to exist.
    *   If a transformation changes scale or interpretation, carry that into evaluation and explanation.
    *   Feature engineering is successful when it improves predictive usefulness without hiding the underlying logic.

## Tool mapping (features)

- Implement feature builds in **`code_pipeline`** with explicit train vs scoring period logic; document lags and rolling windows in `task`.
- Frame features with explicit input/output basenames in **`code_pipeline`** **`task`** objects so leakage and join keys are pinned before codegen.
