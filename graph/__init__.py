"""
Graph package: exports the main agent entry point :class:`AnalysisGraph`.

Example::

    from graph import AnalysisGraph
    g = AnalysisGraph()
    g.run_graph(session_id=\"abc\", user_query=\"Forecast sales\")
"""

from .graph import AnalysisGraph

__all__ = ["AnalysisGraph"]
