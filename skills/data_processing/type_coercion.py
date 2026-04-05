import numpy as np
import pandas as pd


def coerce_dates(df, date_columns):
    """
    Parse date columns stored as strings.
    Uses errors='coerce' so unparseable values become NaT rather than raising.
    """
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def coerce_numeric(df, columns):
    """
    Force numeric columns stored as object to float.
    Uses errors='coerce' so strings become NaN rather than raising.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
