import numpy as np
import pandas as pd


def frequency_encode(df, columns):
    """
    Replace category values with their frequency in the dataset.
    Safe for medium-to-high cardinality when no target is available.
    Returns modified DataFrame.
    """
    for col in columns:
        freq_map = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq_map)
    return df


def onehot_encode(df, columns):
    """
    One-hot encode low cardinality nominal columns.
    Uses drop_first=True to avoid dummy variable trap.
    Only use for columns with <= 10 unique values.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)


def target_encode_train_test(train_df, test_df, cat_col, target_col, smoothing=10):
    """
    Target encoding with smoothing fitted on train only, applied to test.
    Smoothing shrinks rare category means toward global mean.
    Never fit on full dataset — that causes target leakage.
    """
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(cat_col)[target_col].agg(["mean", "count"])
    agg["smooth"] = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    encode_map = agg["smooth"]
    train_df[cat_col + "_te"] = train_df[cat_col].map(encode_map).fillna(global_mean)
    test_df[cat_col + "_te"] = test_df[cat_col].map(encode_map).fillna(global_mean)
    return train_df, test_df
