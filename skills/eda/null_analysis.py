import numpy as np
import pandas as pd


def analyse_nulls(df):
    """
    Return a DataFrame showing null count and null percentage per column.
    Sorted by null_pct descending. Flags columns with >20% nulls as high_null.
    """
    null_counts = df.isna().sum()
    null_pcts = (null_counts / len(df) * 100).round(1)
    result = pd.DataFrame(
        {
            "null_count": null_counts,
            "null_pct": null_pcts,
            "high_null": null_pcts > 20,
        }
    )
    return result[result["null_count"] > 0].sort_values("null_pct", ascending=False)
