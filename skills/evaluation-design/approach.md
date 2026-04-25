# Skill: Evaluation Design

This skill guides how to judge whether an analysis, forecast, or model is actually good enough to trust.

## Core Reasoning Principles

1.  **Define Success Before Modeling**:
    *   Identify what “good” means for this task: accuracy, ranking quality, stability, bias, interpretability, or business usefulness.
    *   Match evaluation to the user’s decision, not just to a standard metric.
    *   Do not treat model fitting as success.

2.  **Use the Right Validation Scheme**:
    *   For time-based problems, prefer chronological holdout or rolling backtests.
    *   For grouped problems, consider whether evaluation should be per segment as well as overall.
    *   Avoid random splits when they break the real-world deployment logic.

3.  **Always Compare Against a Baseline**:
    *   Every complex method should be compared with a simple baseline.
    *   If a complex model does not beat a defensible baseline, prefer the simpler approach.
    *   Baselines reveal whether the workflow is adding real value.

4.  **Choose Metrics That Match the Problem**:
    *   Use error metrics appropriate to the target scale, business cost, and aggregation level.
    *   Consider weighted metrics when large-volume segments matter more.
    *   Watch for bias, not just absolute error.

5.  **Evaluate Failure Modes, Not Just Averages**:
    *   Check whether performance breaks on recent periods, sparse segments, edge cases, or large-value groups.
    *   Report important caveats when evaluation is thin, noisy, or incomplete.
    *   A trustworthy result explains where it works, where it fails, and how confident we should be.
