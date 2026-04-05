PURPOSE: Use this skill when the task involves creating new features from existing ones. Fire for queries containing: feature engineering, new features, encode, transform, extract, create columns, polynomial, interaction.

PRE-CONDITION: Data must be cleaned before feature engineering. Categoricals should have no nulls before encoding. Datetime columns must already be parsed to datetime dtype.

DECISION GUIDE — CATEGORICAL ENCODING:

Binary categoricals (2 unique values): map directly to 0/1
Ordinal categoricals (natural order exists): use OrdinalEncoder with explicit category order
Nominal, low cardinality (<= 10 unique): one-hot encode with pd.get_dummies(drop_first=True)
Nominal, medium cardinality (10-50 unique): frequency encoding (replace with value_counts normalised)
Nominal, high cardinality (> 50 unique): target encoding — but only when a target column is available and data is being split train/test. Use out-of-fold to avoid leakage. Without a target, use frequency encoding.

CRITICAL LEAKAGE RULE: Target encoding must be fit on training data only, then applied to test. Never fit on the full dataset.

DECISION GUIDE — DATETIME FEATURES:

Always extract: year, month, day_of_week, is_weekend
For hourly data: also extract hour, is_business_hour
For cyclical features (month, day_of_week, hour): consider sine/cosine encoding to capture cyclical nature
Drop the original datetime column after extraction unless instructed otherwise

DECISION GUIDE — NUMERIC TRANSFORMS:

Right-skewed distributions (skewness > 1): apply log1p transform
Scaling: StandardScaler for linear models and SVMs, no scaling needed for tree-based models
Never scale the target variable unless specifically requested

ANTI-PATTERNS:

Do not one-hot encode a column with 200 unique values — it creates 200 new sparse columns
Do not apply transforms blindly — check the distribution first
Do not encode the target column as if it is a feature
