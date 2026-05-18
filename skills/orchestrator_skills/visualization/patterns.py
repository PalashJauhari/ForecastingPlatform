"""Vetted code patterns for professional Data Visualization."""

import matplotlib.pyplot as plt
import seaborn as sns

# Pattern: Professional Line Chart with Highlight
def PlotTrend(df, x, y, title, highlight_start=None):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=x, y=y, color='#2c3e50', linewidth=2)
    
    if highlight_start:
        plt.axvspan(highlight_start, df[x].max(), color='#ecf0f1', alpha=0.5, label='Forecast Period')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(y.replace('_', ' ').title())
    plt.xlabel(x.replace('_', ' ').title())
    sns.despine()
    plt.tight_layout()
    return plt.gcf()

# Pattern: Distribution with Outlier Marking
def PlotDistribution(df, value_col):
    plt.figure(figsize=(10, 4))
    sns.histplot(df[value_col], kde=True, color='#3498db')
    
    # Mark mean/median
    plt.vlines(df[value_col].mean(), 0, plt.ylim()[1], color='red', linestyle='--', label='Mean')
    plt.vlines(df[value_col].median(), 0, plt.ylim()[1], color='green', linestyle='-', label='Median')
    
    plt.title(f'Distribution of {value_col}', fontsize=14)
    plt.legend()
    sns.despine()
    return plt.gcf()
