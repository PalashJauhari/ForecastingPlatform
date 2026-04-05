import numpy as np
import pandas as pd


def extract_datetime_features(df, date_col, drop_original=True):
    """
    Extract useful features from a datetime column.
    date_col must already be parsed as datetime dtype — call pd.to_datetime() first.
    Adds: year, month, day, day_of_week, is_weekend, quarter.
    drop_original: whether to drop the source datetime column after extraction.
    """
    df[date_col + "_year"] = df[date_col].dt.year
    df[date_col + "_month"] = df[date_col].dt.month
    df[date_col + "_day"] = df[date_col].dt.day
    df[date_col + "_day_of_week"] = df[date_col].dt.dayofweek
    df[date_col + "_is_weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df[date_col + "_quarter"] = df[date_col].dt.quarter
    if drop_original:
        df = df.drop(columns=[date_col])
    return df


def cyclical_encode(df, col, max_val):
    """
    Cyclical encoding using sine and cosine.
    Useful for month (max_val=12), day_of_week (max_val=7), hour (max_val=24).
    Captures the fact that month 12 and month 1 are adjacent.
    """
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df
