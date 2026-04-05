import numpy as np
import pandas as pd


def profile_dtypes(df):
    """
    Print a dtype profile: column name, inferred dtype, null count, null pct,
    and a flag if the column looks like a date stored as object.
    Returns a summary DataFrame for further use.
    """
    rows = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        dtype = str(df[col].dtype)
        looks_like_date = False
        if dtype == "object":
            sample = df[col].dropna().head(5).astype(str)
            looks_like_date = sample.str.match(r"\d{4}[-/]\d{2}").any()
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "null_count": null_count,
                "null_pct": round(null_pct, 1),
                "possible_date": looks_like_date,
            }
        )
    return pd.DataFrame(rows)
