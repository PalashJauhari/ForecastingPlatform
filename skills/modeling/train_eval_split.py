import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_classification(df, target_col, test_size=0.2, random_state=42):
    """
    Standard stratified train/test split for classification.
    Stratification ensures class balance in both splits.
    Prints class distribution in train and test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print("Train class dist:", y_train.value_counts(normalize=True).round(3).to_dict())
    print("Test class dist:", y_test.value_counts(normalize=True).round(3).to_dict())
    return X_train, X_test, y_train, y_test


def split_regression(df, target_col, test_size=0.2, random_state=42):
    """
    Standard train/test split for regression.
    No stratification needed for continuous targets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_timeseries(df, date_col, target_col, test_fraction=0.2):
    """
    Chronological split for time series — no shuffling.
    Earlier data for train, later data for test.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test
