import numpy as np
import pandas as pd


def impute_numeric(df, strategy="median"):
    """
    Impute missing values in numeric columns.
    strategy: 'median' (default, robust to skew) or 'mean'
    Adds a boolean _was_null indicator column for each imputed column.
    Returns modified DataFrame.
    """
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isna().any():
            df[col + "_was_null"] = df[col].isna().astype(int)
            fill_val = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
    return df


def impute_categorical(df):
    """
    Impute missing values in object/categorical columns with mode.
    If mode is ambiguous (multiple modes), uses the first.
    """
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
    return df
