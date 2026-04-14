"""
Custom middleware functions for the Forecasting Platform agent.

These are plain functions called from the orchestrator node — not
LangChain ``AgentMiddleware`` classes. Keeps the graph transparent:
every middleware step is visible in the node logic.

Modules
    context_editing  — token-aware truncation + running summarisation.
    llm_rate_limit   — shared ``InMemoryRateLimiter`` for ``ChatOpenAI`` (from ``config.yaml``).
    tool_call_limit  — helper module (not wired into the graph currently).
"""
