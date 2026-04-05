import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_correlation_heatmap(df, output_path):
    """
    Annotated correlation heatmap for all numeric columns.
    cmap='coolwarm': blue = negative correlation, red = positive.
    output_path must start with agent_filesystem/output/
    """
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        print("Need at least 2 numeric columns for correlation heatmap")
        return
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    im = ax.imshow(corr.where(~mask), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
