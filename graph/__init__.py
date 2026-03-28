"""
Graph package: exports the main agent entry point :class:`AnalysisGraph`.

Example::

    from graph import AnalysisGraph
    g = AnalysisGraph()
    result = g.run_graph(session_id="abc", user_query="Forecast sales")
    state  = g.get_state(session_id="abc")
"""

from .graph import AgentState, AnalysisGraph

__all__ = ["AgentState", "AnalysisGraph"]
