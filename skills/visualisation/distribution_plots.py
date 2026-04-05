import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_numeric_distributions(df, output_path):
    """
    One subplot per numeric column showing histogram + KDE.
    Saves to output_path as a single figure.
    output_path must start with agent_filesystem/output/
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        print("No numeric columns found")
        return
    n = len(num_cols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="white", color="steelblue", density=True)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
