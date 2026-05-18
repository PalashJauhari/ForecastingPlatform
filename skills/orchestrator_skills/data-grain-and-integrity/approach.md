# Skill: Data Grain & Integrity

Use this skill when you need a **clear row-level meaning**, **correct aggregation level**, **safe joins**, and **ongoing data-quality checks** (counts, keys, nulls, outliers).

## Part A — Grain, aggregation, and modeling level

1.  **Define the grain of every table**:
    *   Explicitly identify what one row means in each dataset.
    *   Grain may be per date, per product, per store-date, per customer-order, or another composite unit.
    *   Never join, aggregate, or model before the row meaning is clear.

2.  **Match output grain to user intent**:
    *   Determine whether the user wants aggregate results or segment-level results.
    *   Ensure the requested deliverable aligns with the dataset grain and the chosen method.
    *   Do not answer a store-level question with a global aggregate, or vice versa, unless intentional.

3.  **Respect aggregation side effects**:
    *   Aggregation can hide volatility, duplicates, outliers, and segment-specific behavior.
    *   Disaggregation can create sparse series, noisy metrics, and unstable comparisons.
    *   Choose the level that preserves the signal needed for the question.

4.  **Model at the right level**:
    *   For forecasting or comparisons, decide whether to model pooled data, separate segments, or a hierarchy.
    *   Small segments may need aggregation; large heterogeneous groups may need separation.

## Part B — Integrity, joins, and anomalies

1.  **Trust but verify (grain check)**:
    *   Never assume a join or pivot is correct.
    *   Re-verify grain before and after structural changes.
    *   If a dataset is “Daily Sales per Store,” ensure no duplicate (Store, Date) pairs unless documented.

2.  **Row count vigilance**:
    *   Compare row counts before and after every join.
    *   If a LEFT JOIN grows row count vs the primary table, suspect duplicate keys in the secondary.
    *   If a join returns zero rows, check key type, casing, and normalization.

3.  **Guard against grain mismatch in joins**:
    *   Before every merge, compare the grain of both inputs.
    *   One-to-many joins must be intentional; unexpected metric changes often trace to grain bugs.

4.  **Null and zero patterns**:
    *   Check nulls in join keys before merging.
    *   Distinguish zero (measured) from missing (NaN).
    *   In time series, watch for “dark periods” where values are all zero or null.

5.  **Anomaly awareness**:
    *   Use IQR or z-scores to flag outliers; before dropping, ask if they reflect errors vs real events.
    *   Never discard outliers without documenting why.

## Tool mapping (grain & integrity)

- Use **`code_pipeline`** when you need code to audit grain (groupby counts, duplicate checks, profile joins) or to reshape data; keep filenames basename-only in arguments.
- After **`sarima_tool`** or **`prophet_tool`**, sanity-check that the input file’s time grain matches the question (regular spacing, no silent duplicate dates).
