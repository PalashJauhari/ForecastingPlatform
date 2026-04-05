PURPOSE: Use this skill when the task involves cleaning data — handling nulls, fixing dtypes, removing or capping outliers, deduplicating, standardising string formats. Fire this skill for queries containing: clean, process, fix, handle missing, remove duplicates, normalise, standardise, impute, outlier.

DECISION GUIDE — MISSING VALUES:

If null percentage > 60% for a column: consider dropping the column entirely, print a warning
If null percentage 20-60%: impute with median (numeric) or mode (categorical), or create an indicator column _was_null
If null percentage < 20%: median imputation for numeric, mode for categorical is usually fine
For time series data: use forward fill (ffill) first, then backfill remaining
NEVER use mean for skewed distributions — use median
NEVER impute using statistics computed on the full dataset when a train/test split exists — fit imputers on train only

DECISION GUIDE — OUTLIERS:

Use IQR method as default: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers
For capping (preferred over dropping): clip to the IQR fence values
For skewed distributions, consider log transform instead of clipping
Do not remove outliers blindly — print count of affected rows first
Never remove outliers from a target column without explicit instruction

DECISION GUIDE — DTYPES:

Columns that look like dates (match \d{4}[-/]\d{2}) should be parsed with pd.to_datetime(errors='coerce')
Columns that are numeric but stored as object: use pd.to_numeric(errors='coerce')
Integer columns with nulls will be float64 in pandas — this is expected behaviour, not a bug
Boolean columns stored as object ('True'/'False'): map with .map({'True': True, 'False': False})

ORDERING: Always process in this order — (1) drop exact duplicates, (2) fix dtypes, (3) handle missing values, (4) handle outliers. Do not reorder.

WHAT TO PRINT: Before and after row counts, null counts before/after imputation, how many outliers were capped, which columns had dtype changes.
