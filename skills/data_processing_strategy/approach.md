Use this skill when the dataframe likely needs preparation before the user’s real task can be answered well.

When this skill applies:
- The task needs aggregation before analysis or plotting.
- The data needs reshaping, filtering, merging, or type fixes.
- The user wants a forecast-ready or analysis-ready table.
- The task depends on weekly/monthly rollups, deduplication, or column cleanup.

How to reason:
- Keep the final required dataframe shape in mind before writing any code.
- Make preprocessing serve the user’s actual goal, not become a side quest.
- Prefer the minimum transformations needed to make the task possible and trustworthy.
- Be explicit about joins, aggregation grain, missing values, type coercion, and dropped rows when they matter.

Tooling implications:
- Inspect first if the current dataframe shape is unclear.
- If custom preparation is required, use `build_codegen_requirement` to specify the dataframe transformations before `code_pipeline`.
- Do not hide meaningful preprocessing decisions inside vague execution requests.
