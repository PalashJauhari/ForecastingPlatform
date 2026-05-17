# Skill: Interpretation

This skill helps turn analytical output into careful, defensible conclusions instead of overconfident claims.

## Core Reasoning Principles

1.  **Separate Result from Meaning**:
    *   A metric, coefficient, or forecast is not the conclusion by itself.
    *   Explain what the result implies for the user’s question and decision.
    *   Keep interpretation tied to the original problem framing.

2.  **Do Not Overclaim**:
    *   Correlation is not causation unless the design supports it.
    *   Good backtest performance does not guarantee future performance.
    *   If evidence is suggestive but not conclusive, say so clearly.

3.  **State Assumptions and Caveats**:
    *   Surface important assumptions behind joins, imputations, feature choices, and evaluation setup.
    *   Highlight limitations from short history, missing variables, sparse segments, or unstable behavior.
    *   Strong interpretation includes what the analysis cannot prove.

4.  **Focus on Material Findings**:
    *   Prioritize the findings most relevant to the user’s objective.
    *   Avoid drowning key conclusions in low-value detail.
    *   Summaries should lead with the strongest supported insight.

5.  **Translate Analysis into Action Carefully**:
    *   Recommendations should follow from evidence, not from model output alone.
    *   If the result supports multiple actions, explain the tradeoffs.
    *   Interpretation is complete only when the user can understand both the insight and its confidence level.

## Tool mapping (interpretation)

- Ground claims in **`sarima_tool`** / **`prophet_tool`** JSON fields (fit quality, residuals, intervals) and in **`code_pipeline`** outputs; do not invent metrics absent from tool results.
- When tool output is partial, say what is unknown rather than filling gaps with model narrative alone.
