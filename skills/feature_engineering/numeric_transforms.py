import numpy as np
import pandas as pd


def log_transform_skewed(df, columns=None, skew_threshold=1.0):
    """
    Apply log1p to right-skewed numeric columns.
    log1p handles zeros safely (log(0) is undefined, log1p(0) = 0).
    If columns is None, auto-detects columns with skewness > skew_threshold.
    """
    if columns is None:
        nums = df.select_dtypes(include="number")
        columns = nums.columns[nums.skew() > skew_threshold].tolist()
    for col in columns:
        if (df[col] >= 0).all():
            df[col + "_log"] = np.log1p(df[col])
    return df


def standard_scale(df, columns):
    """
    StandardScaler pattern — fit on train, apply to both.
    This function is for reference only.
    In practice, use sklearn Pipeline to ensure scaler is fit on train data only.
    """
    means = df[columns].mean()
    stds = df[columns].std()
    df[columns] = (df[columns] - means) / stds
    return df
