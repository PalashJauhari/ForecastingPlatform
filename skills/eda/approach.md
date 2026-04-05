PURPOSE: Use this skill whenever the task involves understanding a dataset for the first time — checking structure, spotting problems, understanding distributions and relationships. Fire this skill for queries containing words like: explore, understand, summarise, what columns, what's in, describe, check, overview, profile, analyse.

ORDERING: Always run EDA steps in this sequence:

Shape and schema — row count, column count, dtypes, memory usage
Null analysis — null counts and percentages per column, flag columns over 20% null
Cardinality scan — unique value counts for all columns, flag high-cardinality categoricals (>50 unique)
Numeric summary — describe() on numeric columns, note skewness, ranges, suspicious zeros
Categorical summary — value_counts() for low-cardinality categoricals
Correlation — correlation matrix on numerics, flag pairs above 0.85
Target analysis — if a target column exists, show its distribution and class balance

WHAT TO PRINT: Every EDA script must print structured, labelled output. Use section headers printed with separators so the output is readable. Never just dump a raw DataFrame.

COMMON PITFALLS TO AVOID:

Do not call .corr() on the whole DataFrame — select numeric columns first with select_dtypes(include='number')
Do not assume a column called 'id' or 'ID' is numeric — it may be a string identifier
Do not drop columns during EDA — observe and report only
Do not impute during EDA — report nulls, do not fix them
Date columns often arrive as object dtype — note this in output, do not silently parse

WHAT GOOD EDA OUTPUT LOOKS LIKE: Shape line, then a null summary table, then cardinality table, then numeric describe, then top value_counts for categoricals, then correlation matrix for numerics. Each section has a printed header. Concise, not exhaustive.
