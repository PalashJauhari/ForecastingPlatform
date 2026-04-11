Use this skill for direct analytical questions where the user wants a metric, grouped summary, comparison, or compact answer from the data.

Sample question types include, but are not limited to:
- what is the average revenue
- what is the total sales by region
- what is the ratio of cost to revenue
- which category has the highest margin
- compare this month vs last month

How to reason:
- Identify the exact metric first.
- Then identify the scope:
  - which file or files (read paths: **`agent_filesystem/<session-folder>/input/...`** or **`.../output/...`**; any saves → **`.../output/...` only**)
  - which filters
  - which grouping level
  - which time slice
- If a ratio or derived metric is requested, define numerator and denominator explicitly.
- If multiple valid interpretations exist, surface the single blocking ambiguity instead of guessing.
- Prefer answering directly and clearly over producing unnecessary artifacts.

Answer shape:
- lead with the metric or conclusion
- include the grouping or filter context that makes the number meaningful
- mention assumptions only when they affect correctness

Tooling implications:
- Pair with `tabular_prep` when filtering, grouping, or merging is likely required.
- Use direct inspection when the answer is small and obvious from the data preview.
- Use `build_codegen_requirement` only when the metric requires more substantial preparation or computation than direct inspection can support safely.
- If you do call `build_codegen_requirement`, state the metric in operational terms:
  - exact numerator and denominator for ratios
  - exact grouping level
  - exact filters or date windows
  - exact output columns or summary shape
- Do not assume the downstream step will infer your intended metric correctly from a loose brief. This skill should make the requirement unambiguous.
