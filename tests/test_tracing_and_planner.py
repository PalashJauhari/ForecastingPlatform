import unittest

from langchain_core.messages import AIMessage, ToolMessage
from langfuse.types import TraceContext
from langgraph.graph import END

from observability.langfuse_handler import add_trace_context_to_config, trace_context_from_runnable_config
from sub_agents.planner_sub_agent.graph import route_after_planner, route_after_planner_tools


class TracingAndPlannerTest(unittest.TestCase):
    """Focused regression coverage for trace context propagation and planner routing."""

    def test_trace_context_round_trips_through_runnable_config(self):
        """Graph configs carry the Langfuse parent span across LangGraph node boundaries."""
        config = {"configurable": {"thread_id": "session-1"}, "recursion_limit": 10}
        trace_context = TraceContext(trace_id="trace-1", parent_span_id="span-1")

        updated = add_trace_context_to_config(config, trace_context)
        restored = trace_context_from_runnable_config(updated)

        self.assertIsNotNone(restored)
        self.assertEqual(restored["trace_id"], "trace-1")
        self.assertEqual(restored["parent_span_id"], "span-1")
        self.assertEqual(updated["configurable"]["thread_id"], "session-1")

    def test_missing_trace_context_returns_none(self):
        """Tracing helpers are no-ops when a run is not carrying Langfuse context."""
        self.assertIsNone(trace_context_from_runnable_config({"configurable": {"thread_id": "session-1"}}))
        self.assertEqual(
            add_trace_context_to_config({"configurable": {"thread_id": "session-1"}}, None),
            {"configurable": {"thread_id": "session-1"}},
        )

    def test_planner_routes_to_tools_when_llm_requests_tool_call(self):
        """Planner still executes tools when the LLM asks for clarification or writes todos."""
        state = {"messages": [AIMessage(content="", tool_calls=[{"name": "write_todo", "args": {}, "id": "call-1"}])]}

        self.assertEqual(route_after_planner(state), "RunTools")

    def test_planner_ends_after_write_todo(self):
        """A successful write_todo is the end of planning for the current turn."""
        state = {
            "messages": [
                ToolMessage(content="Todos updated for this turn.", tool_call_id="call-1", name="write_todo")
            ]
        }

        self.assertEqual(route_after_planner_tools(state), END)

    def test_planner_continues_after_ask_user(self):
        """After a clarification answer returns, planner must produce the final todo list."""
        state = {"messages": [ToolMessage(content="monthly revenue", tool_call_id="call-1", name="ask_user")]}

        self.assertEqual(route_after_planner_tools(state), "PlannerOrchestrator")


if __name__ == "__main__":
    unittest.main()
