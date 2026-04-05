import numpy as np
import pandas as pd


def cap_outliers_iqr(df, columns=None, factor=1.5):
    """
    Cap outliers at IQR fences for numeric columns.
    Prefer capping over dropping — preserves row count.
    Prints how many values were capped per column.
    columns: list of column names to process, or None for all numeric.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_capped > 0:
            print(f"{col}: capped {n_capped} outliers to [{lower:.2f}, {upper:.2f}]")
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df
