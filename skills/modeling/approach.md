PURPOSE: Use this skill when the task involves training a model, evaluating performance, predicting, or comparing models. Fire for queries containing: train, model, predict, classify, regress, evaluate, accuracy, RMSE, F1, cross-validate, fit.

PRE-CONDITION: Data must be cleaned and features engineered before modeling. The script must split data before any fitting — no fitting on the full dataset.

DECISION GUIDE — SPLIT STRATEGY:

Default: train_test_split with test_size=0.2, random_state=42
Classification with class imbalance: always use stratify=y in train_test_split
Time series data: never shuffle. Use a chronological split — earlier dates for train, later for test
Small datasets (< 500 rows): use cross-validation instead of a single split

DECISION GUIDE — MODEL SELECTION:

Classification, tabular data: start with RandomForestClassifier or LogisticRegression
Regression, tabular data: start with RandomForestRegressor or Ridge
Always try at least two models and compare metrics
Report feature importances when using tree-based models

DECISION GUIDE — EVALUATION METRICS:

Binary classification: accuracy, precision, recall, F1, AUC-ROC
Multiclass classification: accuracy, macro F1
Regression: RMSE, MAE, R-squared
Always print both train and test metrics — a large gap indicates overfitting

CRITICAL RULES:

NEVER fit a scaler, encoder, or imputer on the full dataset before splitting
Use sklearn Pipeline to ensure all transforms are fit on train data only
Always set random_state for reproducibility
Print confusion matrix for classification tasks
