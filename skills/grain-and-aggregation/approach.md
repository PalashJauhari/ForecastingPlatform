# Skill: Grain and Aggregation

This skill ensures the analysis respects the true unit of observation and the level at which results should be produced.

## Core Reasoning Principles

1.  **Define the Grain of Every Table**:
    *   Explicitly identify what one row means in each dataset.
    *   Grain may be per date, per product, per store-date, per customer-order, or another composite unit.
    *   Never join, aggregate, or model before the row meaning is clear.

2.  **Match Output Grain to User Intent**:
    *   Determine whether the user wants aggregate results or segment-level results.
    *   Ensure the requested deliverable aligns with the dataset grain and the chosen method.
    *   Do not accidentally answer a store-level question with a global aggregate, or vice versa.

3.  **Respect Aggregation Side Effects**:
    *   Aggregation can hide volatility, duplicates, outliers, and segment-specific behavior.
    *   Disaggregation can create sparse series, noisy metrics, and unstable comparisons.
    *   Choose the level that preserves the signal needed for the question.

4.  **Guard Against Grain Mismatch in Joins**:
    *   Before every merge, compare the grain of both inputs.
    *   A one-to-many join may be correct, but it must be intentional.
    *   If post-join row counts or metrics change unexpectedly, treat it as a grain problem first.

5.  **Model at the Right Level**:
    *   For forecasting or comparisons, decide whether to model pooled data, separate segments, or a hierarchy.
    *   Small segments may need aggregation; large heterogeneous groups may need separation.
    *   The correct grain is often the difference between insight and noise.
