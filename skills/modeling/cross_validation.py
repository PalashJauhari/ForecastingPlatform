import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate


def cv_classification(model, X, y, n_splits=5):
    """
    Stratified K-Fold cross-validation for classification.
    Stratification preserves class distribution across all folds.
    Prints mean and std of accuracy and F1 across folds.
    Use for small datasets where a single split wastes too much data.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(
        f"CV Accuracy: {results['test_accuracy'].mean():.4f} +/- {results['test_accuracy'].std():.4f}"
    )
    print(
        f"CV F1 Macro: {results['test_f1_macro'].mean():.4f} +/- {results['test_f1_macro'].std():.4f}"
    )
    return results


def cv_regression(model, X, y, n_splits=5):
    """
    K-Fold cross-validation for regression.
    Prints mean RMSE and R2 across folds.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_root_mean_squared_error", "r2"],
        return_train_score=True,
    )
    rmse = -results["test_neg_root_mean_squared_error"]
    print(f"CV RMSE: {rmse.mean():.4f} +/- {rmse.std():.4f}")
    print(f"CV R2:   {results['test_r2'].mean():.4f} +/- {results['test_r2'].std():.4f}")
    return results
