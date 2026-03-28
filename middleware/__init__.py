"""
Custom middleware functions for the Forecasting Platform agent.

These are plain functions called from the orchestrator node — not
LangChain ``AgentMiddleware`` classes. Keeps the graph transparent:
every middleware step is visible in the node logic.

Modules
    context_editing  — token-aware truncation + running summarisation.
    tool_call_limit  — abort the agent loop when the budget is exceeded.
"""
