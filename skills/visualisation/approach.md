PURPOSE: Use this skill when the task involves creating charts, plots, or visual analysis. Fire for queries containing: plot, chart, visualise, graph, distribution, show, histogram, heatmap, bar chart, line chart, scatter.

SANDBOX RULE: All plots must be saved with plt.savefig('agent_filesystem/output/...') — not plt.show(). Always call plt.close() after savefig to free memory and prevent figure bleed between plots.

DECISION GUIDE — CHART TYPE SELECTION:

Distribution of one numeric column: histogram with KDE overlay (histplot with kde=True)
Distribution comparison across categories: boxplot or violin plot
Correlation between two numerics: scatter plot
Correlation matrix across all numerics: heatmap with annotated values
Counts of a categorical: bar chart (countplot)
Trend over time: line chart with date on x-axis
Target vs feature relationship: scatter (numeric) or grouped box (categorical)

GOOD CHART HABITS:

Always set a title, x-label, and y-label — a plot without labels is useless
Use figsize=(10, 6) as default — wide enough to be readable
For correlation heatmaps: use cmap='coolwarm', annot=True, fmt='.2f'
For many categories on x-axis: rotate x-tick labels 45 degrees
Print a confirmation message after each save: 'Saved: agent_filesystem/output/...'

ANTI-PATTERNS:

Do not create pie charts for more than 5 categories — use bar charts
Do not use default matplotlib colours for multiple series — set a palette
Do not forget plt.tight_layout() before savefig — prevents label clipping
