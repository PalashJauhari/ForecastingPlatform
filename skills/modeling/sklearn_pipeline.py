import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classification_pipeline(model=None):
    """
    Minimal sklearn Pipeline for classification.
    Scaler is included but tree models don't need it — swap model as needed.
    Pipeline ensures all transforms are fit on training data only.
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """
    Fit pipeline and print classification report for train and test.
    Large train/test gap = overfitting.
    """
    pipeline.fit(X_train, y_train)
    print("--- Train ---")
    print(classification_report(y_train, pipeline.predict(X_train)))
    print("--- Test ---")
    print(classification_report(y_test, pipeline.predict(X_test)))
    return pipeline


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """
    Fit pipeline and print RMSE and R2 for train and test.
    """
    pipeline.fit(X_train, y_train)
    for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        preds = pipeline.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        print(f"{name} RMSE: {rmse:.4f} | R2: {r2:.4f}")
    return pipeline
