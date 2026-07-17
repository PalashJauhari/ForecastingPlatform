"""
Graph package: exports the main agent entry point :class:`AnalysisGraph`.

Example::

    from graph import AnalysisGraph
    g = await AnalysisGraph.acreate()
    result = await g.run_graph(session_id="abc", user_query="Forecast sales")
    state  = await g.get_state(session_id="abc")
"""

from graph.graph import AgentState, AnalysisGraph

__all__ = ["AgentState", "AnalysisGraph"]
