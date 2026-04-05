import numpy as np
import pandas as pd


def numeric_summary(df):
    """
    Extended describe() for numeric columns.
    Adds skewness and flags columns with suspicious all-zero or near-constant values.
    """
    nums = df.select_dtypes(include="number")
    if nums.empty:
        return pd.DataFrame()
    desc = nums.describe().T
    desc["skewness"] = nums.skew().round(3)
    desc["near_constant"] = desc["std"] < 0.01
    return desc


def categorical_summary(df, max_cols=10):
    """
    Top 5 value counts for each low-cardinality categorical column.
    Skips columns with more than 50 unique values.
    """
    cats = df.select_dtypes(include="object")
    summaries = {}
    for col in list(cats.columns)[:max_cols]:
        if df[col].nunique() <= 50:
            summaries[col] = df[col].value_counts().head(5)
    return summaries
