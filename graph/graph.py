"""
Forecast graph: orchestration → tool | summarisation | clarification.

Entry point is orchestration_node (LLM decides next step). Then route to tool_node,
summarisation_node, or clarification_node. Uses ConversationalMemory and ToolMemory.
"""
import json
import os
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any, List
import yaml
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tools.plot_tool import plot_data
from prompts.graph_prompts import ORCHESTRATION_PROMPT, SUMMARISATION_PROMPT
from memory.conversation import ConversationalMemory
from memory.tool_memory import ToolMemory


class ForecastGraphState(TypedDict):
    """State passed through the graph. Keys from orchestration JSON plus session/csv/query."""

    session_id: Optional[str]
    csv_path: str
    user_query: str
    # Set by orchestration_node from LLM JSON:
    should_proceed: Optional[bool]  # True → route to tool_node
    is_summarisation: Optional[bool]  # True → route to summarisation_node, use final_response
    is_clarification: Optional[bool]  # True → route to clarification_node
    tool_name: Optional[str]  # Name of tool to run (must exist in self.tools)
    tool_parameter: Optional[Dict[str, Any]]  # Kwargs passed to the tool
    clarification_question: Optional[str]  # Question to ask the user
    final_response: Optional[str]  # Orchestration/summarisation output text
    last_tool_result: Optional[Any]  # Return value of the last tool run (generic, for UI/API)


class ForecastGraph:
    """
    Builds a LangGraph: orchestration → (tool | summarisation | clarification).
    Uses one LLM client, conversational memory, tool memory, and a tool registry.
    """

    def __init__(self, session_id: str = "default"):
        """Create LLM client, conversational memory, tool memory, and load tools from registry."""
        self.session_id = session_id
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self.conversational_memory = ConversationalMemory()
        self.tool_memory = ToolMemory()
        self._csv_path: Optional[str] = None
        self._init_tools()
        self.available_tools = self.get_tool_description()
        self._compiled_graph = self.build_graph()

    def set_csv_path(self, path: str) -> None:
        """Set the CSV path for this session (call before run_graph when file is uploaded or path is known)."""
        self._csv_path = path

    def _init_tools(self) -> None:
        """
        Load tool_registry.yaml and set self.tools to a map of tool name → callable.
        Only names listed in the registry are added; extend the loop when adding new tools.
        """
        self.tools: Dict[str, Any] = {}
        registry_path = Path(__file__).resolve().parent / "tool_registry.yaml"
        if not registry_path.exists():
            return
        with open(registry_path) as f:
            data = yaml.safe_load(f) or {}
        for name in data.get("tools") or {}:
            if name == "plot_data":
                self.tools["plot_data"] = plot_data
            # Register more tools here when added to tool_registry.yaml

    def get_tool_description(self) -> str:
        """
        Read tool_registry.yaml and return a single string describing all tools:
        name, description, required/optional params, and example calls.
        Used to fill {available_tools} in ORCHESTRATION_PROMPT.
        """
        registry_path = Path(__file__).resolve().parent / "tool_registry.yaml"
        if not registry_path.exists():
            return "No tools registered."
        with open(registry_path) as f:
            data = yaml.safe_load(f) or {}
        tools = data.get("tools") or {}
        lines: List[str] = []
        for name, info in tools.items():
            if not isinstance(info, dict):
                continue
            desc = (info.get("description") or "").strip()
            req = info.get("parameters_required") or []
            opt = info.get("parameters_optional") or []
            examples = info.get("examples") or []
            lines.append(f"- {name}: {desc}")
            if req:
                lines.append(f"  Required: {', '.join(str(p).split()[0] for p in req)}")
            if opt:
                lines.append(f"  Optional: {', '.join(str(p).split()[0] for p in opt)}")
            for ex in examples[:2]:
                lines.append(f"  Example: {ex}")
        return "\n".join(lines) if lines else "No tools registered."

    def orchestration_node(self, state: ForecastGraphState) -> dict:
        """
        First node: call LLM with user_input, conversation_memory, tool_history, and available_tools.
        Parse JSON response and update state (should_proceed, is_summarisation, is_clarification,
        tool_name, tool_parameter, clarification_question, final_response). Does not add new keys.
        """
        user_input = state.get("user_query") or ""
        conversation_memory = self.conversational_memory.get_conversation() or "(none)"
        tool_history = self.tool_memory.get_tool_history() or "(none)"
        prompt = ORCHESTRATION_PROMPT.format(
            user_input=user_input,
            conversation_memory=conversation_memory,
            tool_history=tool_history,
            available_tools=self.available_tools,
        )

        msg = self.llm.invoke(prompt)
        content = msg.content if hasattr(msg, "content") else str(msg)
        try:
            out = json.loads(content)
        except json.JSONDecodeError:
            out = {"should_proceed": False, "is_summarisation": False, "clarification_question": "Could not parse response."}

        # Update only orchestration-derived keys; rest of state unchanged
        state["should_proceed"] = out.get("should_proceed", False)
        state["is_summarisation"] = out.get("is_summarisation", False)
        state["is_clarification"] = not state["should_proceed"] and not state["is_summarisation"]
        state["tool_name"] = out.get("tool_name")
        state["tool_parameter"] = out.get("tool_parameter")
        state["clarification_question"] = out.get("clarification_question")
        state["final_response"] = out.get("final_response")
        
        return state
    
    def route_after_orchestration(self, state: ForecastGraphState) -> str:
        """
        Conditional routing after orchestration_node. Returns node name: "tool", "summarisation", or "clarification".
        """
        if state.get("should_proceed"):
            return "tool"
        if state.get("is_summarisation"):
            return "summarisation"
        if state.get("is_clarification"):
            return "clarification"
        return "clarification"

    def tool_node(self, state: ForecastGraphState) -> dict:
        """
        Run when route is tool. Looks up tool_name in self.tools; raises ValueError if missing.
        Injects session_id from state into params so tools (e.g. plot_data) can write under data/session_id/.
        Executes the underlying Python function with tool_parameter as kwargs, then logs
        tool_name, params, and result to tool_memory (AIMessage + ToolMessage).
        """
        tool_name = state.get("tool_name")
        params = dict(state.get("tool_parameter") or {})
        params["session_id"] = state.get("session_id") or "default"
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}")
        tool = self.tools[tool_name]
        tool_callable = getattr(tool, "func", tool)
        tool_response = tool_callable(**params)
        tool_response_str = str(tool_response) if tool_response is not None else ""
        self.tool_memory.save_tool_responses(tool_name, params, tool_response_str)
        
        return state

    def summarisation_node(self, state: ForecastGraphState) -> dict:
        """
        Run when route is summarisation. Calls LLM with user_input, conversation_memory, tool_history.
        Expects JSON with "final_response"; saves that to conversational memory and sets state["final_response"].
        """
        user_input = state.get("user_query") or ""
        conversation_memory = self.conversational_memory.get_conversation() or "(none)"
        tool_history = self.tool_memory.get_tool_history() or "(none)"
        prompt = SUMMARISATION_PROMPT.format(
            user_input=user_input,
            conversation_memory=conversation_memory,
            tool_history=tool_history,
        )
        msg = self.llm.invoke(prompt)
        content = msg.content if hasattr(msg, "content") else str(msg)
        try:
            out = json.loads(content)
            final_response = out.get("final_response") or "Done."
        except json.JSONDecodeError:
            final_response = content.strip() if content else "Done."
        self.conversational_memory.save_conversation(user_input, final_response)
        state["final_response"] = final_response
        return state

    def clarification_node(self, state: ForecastGraphState) -> dict:
        """
        Run when route is clarification. Saves clarification_question (and user query) to conversational
        memory and returns state unchanged.
        """
        query = state.get("user_query") or ""
        q = state.get("clarification_question") or "?"
        self.conversational_memory.save_conversation(query, q)
        return state

    def build_graph(self):
        """
        Build the StateGraph: add orchestration, summarisation, clarification, tool nodes;
        set entry point to orchestration; add conditional edges from orchestration; wire all to END.
        Returns the compiled graph.
        """
        builder = StateGraph(ForecastGraphState)
        builder.add_node("orchestration", self.orchestration_node)
        builder.add_node("summarisation", self.summarisation_node)
        builder.add_node("clarification", self.clarification_node)
        builder.add_node("tool", self.tool_node)
        builder.set_entry_point("orchestration")
        builder.add_conditional_edges(
            "orchestration",
            self.route_after_orchestration,
            {"tool": "tool", "summarisation": "summarisation", "clarification": "clarification"},
        )
        builder.add_edge("summarisation", END)
        builder.add_edge("clarification", END)
        builder.add_edge("tool", "orchestration")  # loop back so LLM can call more tools or summarise
        return builder.compile()

    def run_graph(
        self,
        session_id: str,
        user_query: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> dict:

        """
        Invoke the graph. Inputs: session_id, user_query. csv_path must be set via set_csv_path() first.
        Builds ForecastGraphState inside; returns the final state from the run. Clears tool memory after.
        """
        initial_state: ForecastGraphState = {
            "session_id": session_id,
            "csv_path": self._csv_path or "",
            "user_query": user_query,
            "should_proceed": True,
            "is_summarisation": False,
            "is_clarification": False,
            "tool_name": "",
            "tool_parameter": {},
            "clarification_question": "",
            "final_response": "",
            "last_tool_result": "",
        }

        output = self._compiled_graph.invoke(initial_state, config=config or {})

        # clear tool history
        self.tool_memory.clear()


        return output
