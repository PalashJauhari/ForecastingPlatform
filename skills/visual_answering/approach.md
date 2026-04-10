Use this skill when the best answer is a chart, plot, or visual summary.

What this skill owns:
- deciding whether a visual is actually the right answer
- choosing the chart type that best matches the question
- deciding the needed aggregation and grouping before plotting
- keeping the visual interpretable and decision-oriented

Sample chart heuristics:
- use a line chart for change over time
- use bars for ranked or grouped comparisons
- use stacked or grouped bars only when composition is central to the question
- avoid clutter and unnecessary multi-chart output

How to reason:
- Start with the business question, not the plot type.
- Decide what comparison, trend, or distribution the visual should reveal.
- Make sure the dataframe is at the correct grain before plotting.
- If one chart answers the question cleanly, prefer one chart over a dashboard-style bundle.
- Titles, axes, legends, and labels should make the answer understandable without extra decoding.

Tooling implications:
- Pair with `tabular_prep` whenever aggregation, filtering, reshaping, or merging may be required first.
- If the user wants a visual and a short interpretation, produce both, but keep the interpretation tied to what the chart shows.
- When execution is needed, make the plotting requirement explicit before `code_pipeline`.
- If you use `build_codegen_requirement`, describe the visual precisely:
  - what dataframe should exist before plotting
  - what is on the x-axis and y-axis
  - what grouping, hue, stacking, or faceting is needed
  - what aggregation is required before plotting
  - what the chart is meant to answer
- Do not hand off "make a chart". Hand off a plotting plan that reflects your reasoning, because this skill is the part that understands the visual goal.
