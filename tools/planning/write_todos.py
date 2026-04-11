"""
LangGraph tool: replace session ``todos`` in state and append a ``ToolMessage``.

Returns a ``Command`` so ``ToolNode`` merges ``todos`` and messages into graph state.
"""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langfuse import observe
from langgraph.types import Command

from observability.langfuse_handler import get_langfuse_client
from output_validation.write_todos import TodoItem, WriteTodosInput

langfuse = get_langfuse_client()


@observe(name="tool.write_todos", as_type="tool")
def _write_todos_impl(todos: list[TodoItem], runtime: ToolRuntime) -> Command:
    """
    Replace the graph state's todo list and surface the result as a tool message.

    The orchestrator supplies the full ``todos`` array (validated). Each item has
    ``content`` and ``status`` in ``pending`` | ``in_progress`` | ``completed``.
    """
    validated = WriteTodosInput(todos=todos)
    payload = {"todos": [item.model_dump() for item in validated.todos]}
    n = len(validated.todos)
    if n == 0:
        tool_content = "Todo list cleared (0 items)."
    else:
        tool_content = f"Todos updated ({n} item{'s' if n != 1 else ''})."

    tool_call_id = runtime.tool_call_id
    with langfuse.start_as_current_observation(name="tool.write_todos.state_update", as_type="span", input={"todos": payload["todos"]}) as gen:
        gen.update(
            output={"todos": payload["todos"], "tool_message": tool_content},
            metadata={"todo_count": n},
        )

    return Command(
        update={
            "todos": payload["todos"],
            "messages": [
                ToolMessage(
                    content=tool_content,
                    tool_call_id=tool_call_id,
                )
            ],
        },
    )


@tool(args_schema=WriteTodosInput)
def write_todos(todos: list[TodoItem], runtime: ToolRuntime) -> Command:
    """Maintain a structured todo list for the **current multi-step analysis session**.

    For complex, multi-step analysis, call this tool with a full `todos` argument: a list of objects with `content` (str) and `status` (`pending`, `in_progress`, or `completed`). The session stores the latest list; each orchestrator turn includes **Current todo list** as JSON (including `[]` when empty). Use it to plan, show progress, and revise the plan as you learn from tools—not for trivial one-off questions.

    ## When to use
    - The user's request needs **3+ concrete steps** (e.g. profile → clean → model → save), or multiple deliverables.
    - Work spans several tool calls and you want **clear progress** for yourself and the user.
    - The plan may **change** as you learn from data (add, reorder, or drop tasks).

    ## When not to use
    - Single-step or trivial tasks (one or two quick tool calls).
    - Purely conversational answers with no real workspace work.

    ## How to use
    1. Pass the **full** list each time — this tool **replaces** the stored list; it does not merge patches.
    2. Use **exactly** these statuses: `pending`, `in_progress`, `completed`.
    3. Mark a task `in_progress` **before** you start it; mark `completed` **right after** it is fully done (do not batch many completions).
    4. Prefer **specific** task text (e.g. "Run code_pipeline on sales.csv") over vague steps.
    5. Keep at least one task `in_progress` while work remains (unless everything is `completed`).
    6. Do **not** call this tool in parallel with itself in the same step — one `write_todos` call per model turn.

    ## Output
    The tool result is a short confirmation with the number of todos. The full list remains in session state and appears under **Current todo list** on your next orchestrator turn.
    """
    return _write_todos_impl(todos=todos, runtime=runtime)
