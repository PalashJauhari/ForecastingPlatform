Use this skill when the answer depends on preparing one or more tables before the final metric, visual, or forecast can be produced.

What this skill owns:
- choosing the right file or files
- deciding whether a merge is needed
- aligning grain before comparison or joining
- filtering rows
- grouping and aggregating
- reshaping or deduplicating
- creating derived columns such as ratios, shares, growth, or averages

How to reason:
- Start from the final question, then work backward to the dataframe shape needed to answer it.
- When multiple files are present, identify the primary table and the exact join purpose before merging.
- Do not merge just because multiple files exist.
- Be explicit about join keys, join type, aggregation level, and dropped rows when they materially affect the answer.
- Align time grain and entity grain before computing metrics across files.
- Prefer the minimum viable transformation that makes the answer trustworthy.

Sample situations where this skill is especially relevant include:
- the user asks about sums, averages, ratios, comparisons, or grouped metrics
- the user asks for a chart and the data likely needs aggregation first
- the user asks for a forecast but the raw data is split across files or not yet forecast-ready

Tooling implications:
- Inspect available files first when the relevant inputs are not obvious.
- Read the specific file schemas you need before planning code.
- If the prep is simple and can be reasoned from inspection, keep it simple.
- If the prep is too complex for direct reasoning, use `build_codegen_requirement` to specify the exact transformation workflow before `code_pipeline`.
- When using `build_codegen_requirement`, spell out the prep logic you have already inferred:
  - which files are involved
  - which file is primary
  - whether a merge is needed and on what keys
  - what filters, aggregations, reshaping, or derived columns are required
  - what the final dataframe should look like before the downstream step
- Do not send a vague request like "prepare the data". Encode the actual transformation plan because this skill is the part of the system that understands it.
