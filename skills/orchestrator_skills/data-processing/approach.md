# Skill: Data Processing

This skill guides how to clean, reshape, combine, and prepare data in a way that preserves analytical meaning.

## Core Reasoning Principles

1.  **Process in a Defensible Order**:
    *   Start by understanding schema, grain, and key fields before transforming anything.
    *   Do structural fixes first, then type handling, then missing values, then joins, then aggregations or reshaping.
    *   A poor preprocessing order can silently distort the final result.

2.  **Every Transformation Must Have a Reason**:
    *   Do not clean, drop, fill, filter, or reshape data just because it is common practice.
    *   Tie each processing step to a clear analytical need or data issue.
    *   If a transformation changes meaning, document the tradeoff explicitly.

3.  **Preserve Meaning While Cleaning**:
    *   Distinguish between fixing format problems and changing the signal.
    *   Treat deduplication, null imputation, clipping, resampling, and filtering as analytical choices, not just technical chores.
    *   Prefer reversible or auditable processing when the impact is material.

4.  **Respect Keys, Time, and Categories**:
    *   Normalize types before joining or grouping.
    *   Handle timestamps, categorical values, and identifiers carefully so processing does not collapse distinct cases.
    *   Before wide-to-long or long-to-wide reshaping, confirm that the target structure matches the analytical goal.

5.  **Processing Should Make Downstream Steps Safer**:
    *   Good preprocessing reduces ambiguity for feature engineering, modeling, and interpretation.
    *   If processing choices can materially affect the conclusion, surface them as assumptions or caveats.
    *   The best data-processing plan makes later steps simpler, not more fragile.

## Tool mapping (processing)

- **`code_pipeline`** — Primary way to execute multi-step cleaning, reshaping, and joins with auditable code; pass a structured **`task`** (`requirements`, `input`, `output` basenames) for larger pipelines.
- Combine with the **data-grain-and-integrity** skill when joins or aggregation change row meaning—verify duplicates and counts before and after.
