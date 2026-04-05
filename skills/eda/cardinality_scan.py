import pandas as pd


def scan_cardinality(df, high_threshold=50):
    """
    Return unique value counts for every column.
    Flags columns above high_threshold as high_cardinality.
    Useful for deciding encoding strategy before feature engineering.
    """
    rows = []
    for col in df.columns:
        n_unique = df[col].nunique()
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": n_unique,
                "high_cardinality": n_unique > high_threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False)
