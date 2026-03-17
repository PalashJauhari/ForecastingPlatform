"""
Plot CSV with Plotly. X axis, Y axis, optional color column.
Plots are saved under data/<session_id>/plots/.

Reads session_id and csv_path from session context vars
(set by SessionContextMiddleware) so they don't appear in the tool signature.
"""
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
from langchain_core.tools import tool

from graph.middleware.session_context import session_id_var, csv_path_var

DATA_DIR = Path("./data")


def _session_plot_dir(session_id: str) -> Path:
    safe = re.sub(r"[^\w\-]", "", session_id or "default") or "default"
    d = DATA_DIR / safe / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


@tool
def plot_data(x_col: str, y_col: str, color_col: str = "") -> dict:
    """
    Create a Plotly scatter plot from the session CSV.
    x_col: column for X axis.
    y_col: column for Y axis.
    color_col: optional column for point color (omit or "" for none).
    Returns dict with plot_saved (bool) and location (path to saved HTML).
    """
    session_id = session_id_var.get()
    csv_path = csv_path_var.get()

    if not csv_path:
        return {"plot_saved": False, "location": "", "error": "No CSV loaded for this session."}

    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        return {
            "plot_saved": False,
            "location": "",
            "error": f"Column(s) not found. Available: {list(df.columns)}",
        }

    color_col = (color_col or "").strip() or None
    if color_col and color_col not in df.columns:
        color_col = None

    if color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)

    fig.update_layout(title=f"{y_col} vs {x_col}", xaxis_title=x_col, yaxis_title=y_col)

    plot_dir = _session_plot_dir(session_id)
    out_path = plot_dir / (Path(csv_path).stem + "_plot.html")
    fig.write_html(str(out_path))
    return {"plot_saved": True, "location": str(out_path)}
