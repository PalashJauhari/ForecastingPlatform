"""
Single tool: plot CSV with Plotly. X axis, Y axis, optional color column.
Plots are saved under data/<session_id>/plots/.
"""
import re
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
import plotly.express as px

DATA_DIR = Path("./data")


def _session_plot_dir(session_id: str) -> Path:
    """Return data/session_id/plots; session_id sanitized for path safety."""
    safe = re.sub(r"[^\w\-]", "", session_id or "default") or "default"
    d = DATA_DIR / safe / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


@tool
def plot_data(csv_path: str, x_col: str, y_col: str, color_col: str = "", **kwargs) -> dict:
    """
    Plot the CSV: x_col on X axis, y_col on Y axis. Optionally use color_col for point color.
    Returns dict with plot_saved (bool) and location (path to saved HTML, or empty if failed).
    """
    session_id = kwargs.get("session_id", "default")  # Injected by graph; not in tool signature
    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        return {"plot_saved": False, "location": ""}
    color_col = (color_col or "").strip() or None
    if color_col and color_col not in df.columns:
        color_col = None

    if color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)

    fig.update_layout(
        title=f"{y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    plot_dir = _session_plot_dir(session_id)
    out_path = plot_dir / (Path(csv_path).stem + "_plot.html")
    fig.write_html(str(out_path))
    return {"plot_saved": True, "location": str(out_path)}
