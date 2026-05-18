# Skill: Data Readiness

This skill determines whether the available data is actually sufficient and usable for the requested analysis.

## Core Reasoning Principles

1.  **Check Task-to-Data Fit**:
    *   Verify that the available files can answer the requested question.
    *   Confirm the presence of required columns, keys, timestamps, and target variables.
    *   Do not proceed to modeling if the data cannot support the task yet.

2.  **Assess Minimum Viability**:
    *   For forecasting, check for enough history, a usable time column, and a measurable target.
    *   For joins or multi-file analysis, confirm shared keys and compatible grains.
    *   For evaluation tasks, verify that actual outcomes or holdout periods exist.

3.  **Look for Structural Problems First**:
    *   Detect missing timestamps, duplicate entities, impossible types, or heavily null critical fields.
    *   Separate “messy but workable” data from “blocked until clarified or repaired” data.
    *   If the structure is unclear, prioritize understanding over execution.

4.  **Decide Whether to Proceed, Explore, or Clarify**:
    *   If the data is ready, move forward confidently.
    *   If the data might be usable but key assumptions remain unresolved, do a cautious exploratory step.
    *   If the task is blocked by missing columns, ambiguous schemas, or insufficient history, ask the user directly.

5.  **Make Readiness Explicit**:
    *   State what is available, what is missing, and what assumptions are being made.
    *   Treat readiness as a gate, not a side note.
    *   A good workflow starts with confirming the data can carry the intended conclusion.

## Tool mapping (readiness)

- Rely on **`data_profile`** in orchestrator context and **`code_pipeline`** for exploratory profiling when the session files exist but structure is unclear.
- **`ask_user`** — Use when the task is blocked (missing columns, no time column, ambiguous file role) rather than guessing from thin data.
