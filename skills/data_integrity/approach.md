# Skill: Data Integrity

This skill focuses on ensuring data quality, consistency, and reliability throughout the analysis pipeline.

## Core Reasoning Principles

1.  **Trust but Verify (The Grain Check)**:
    *   Never assume a join or a pivot is correct.
    *   Always verify the "Grain" (the meaning of one row) before and after every operation.
    *   If a dataset represents "Daily Sales per Store," ensure there are no duplicate (Store, Date) pairs.

2.  **Row Count Vigilance**:
    *   Compare `len(df)` before and after every join.
    *   If a LEFT JOIN results in more rows than the primary table, you have a duplicate key issue in the secondary table. Stop and investigate.
    *   If a join results in zero rows, your keys likely have type mismatches (e.g., `int` vs `str`) or casing issues.

3.  **Null and Zero Patterns**:
    *   Check for nulls in join keys before merging.
    *   Distinguish between "Zero" (a measured value of 0) and "Missing" (NaN).
    *   In time series, look for "Dark Periods" where all values are suddenly zero or null.

4.  **Anomaly Awareness**:
    *   Use IQR or Z-Scores to find outliers.
    *   Before deleting an outlier, ask: "Is this a data error (sensor noise) or a business event (promotion, holiday)?"
    *   Never ignore outliers without documenting why.
