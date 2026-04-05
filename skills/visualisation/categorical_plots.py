import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_categorical_counts(df, columns, output_path):
    """
    Bar chart of value counts for each categorical column.
    Only shows top 15 categories per column to keep charts readable.
    output_path must start with agent_filesystem/output/
    """
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        counts = df[col].value_counts().head(15)
        ax.barh(counts.index.astype(str), counts.values, color="steelblue")
        ax.set_title(f"{col} (top {len(counts)})")
        ax.set_xlabel("Count")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
