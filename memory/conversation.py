"""
Conversational memory: user_input in HumanMessage template, ai_output in AIMessage template.
Supports keeping a bounded number of turns via n_keep.
"""
from typing import List

from langchain_core.messages import AIMessage, HumanMessage


def _format_turns(messages: List[tuple]) -> str:
    """Format a list of (HumanMessage, AIMessage) pairs as a single string."""
    lines = []
    for human_msg, ai_msg in messages:
        lines.append(f"Human: {human_msg.content}")
        lines.append(f"AI: {ai_msg.content}")
    return "\n".join(lines).strip()


class ConversationalMemory:
    """Stores conversation turns as HumanMessage (user_input) and AIMessage (ai_output) pairs.
    Optionally caps the number of turns kept via n_keep.
    """

    def __init__(self, n_keep: int = 50):
        """
        Args:
            n_keep: Maximum number of conversation (Human, AI) pairs to keep.
                    Default 50. Use 0 for no limit.
        """
        self._messages: List[tuple] = []  # list of (HumanMessage, AIMessage) pairs
        self._n_keep = n_keep or 0  # 0 means no limit

    def get_conversation(self) -> str:
        """Return the conversation history as a string. Empty if none."""
        return _format_turns(self._messages)

    def save_conversation(self, user_input: str, ai_output: str) -> None:
        """Append one turn and trim to n_keep if set (drop oldest)."""
        human_msg = HumanMessage(content=user_input, metadata={"source": "user_input"})
        ai_msg = AIMessage(content=ai_output, metadata={"source": "final_response"})
        self._messages.append((human_msg, ai_msg))
        if self._n_keep > 0 and len(self._messages) > self._n_keep:
            self._messages = self._messages[-self._n_keep :]

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
