"""
Tool memory: store tool calls (AIMessage template: tool_name, tool_parameter) and tool responses (ToolMessage template).
"""
import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, ToolMessage


class ToolMemory:
    """Stores tool-call history as AIMessage (tool_name + tool_parameter) and ToolMessage (tool_response)."""

    def __init__(self):
        self._messages: List[tuple] = []  # list of (AIMessage, ToolMessage) pairs

    def get_tool_history(self) -> str:
        """Return the tool-call history as a string. Empty if none."""
        lines = []
        for ai_msg, tool_msg in self._messages:
            lines.append(f"AIMessage: {ai_msg.content}")
            lines.append(f"ToolMessage: {tool_msg.content}")
        return "\n".join(lines).strip()

    def save_tool_responses(self, tool_name: str, tool_parameter: Dict[str, Any], tool_response: str) -> None:
        """Append one tool call: AIMessage template (tool_name, tool_parameter) and ToolMessage template (tool_response)."""
        ai_content = json.dumps({"tool_name": tool_name, "tool_parameter": tool_parameter})
        ai_msg = AIMessage(content=ai_content, metadata={"source": "orchestration agent"})
        tool_msg = ToolMessage(content=tool_response, tool_call_id="", metadata={"source": tool_name})
        self._messages.append((ai_msg, tool_msg))

    def clear(self) -> None:
        """Clear all tool-call history."""
        self._messages.clear()
