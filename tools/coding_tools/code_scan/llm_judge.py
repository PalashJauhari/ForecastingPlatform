"""
Layer between Semgrep and disk: an LLM reviews generated Python against the user task.

Used by ``code_pipeline`` as Step 2c (after static scan, before save/run).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)

from prompts.code_judge_prompt import CODE_JUDGE_SYSTEM_PROMPT

langfuse = get_langfuse_client()


class JudgeOutput(BaseModel):
    """Structured accept/reject response from the judge model."""

    passed: bool = Field(description="Whether the generated code is safe and on-spec.")
    detail: str = Field(default="", description="Short rejection reason when passed is false.")


def run_llm_judge(*, code: str, task: str, model_name: str) -> tuple[bool, str]:
    """
    Ask the judge model to accept or reject the script.

    Returns
        ``(True, "")`` if the judge approves.
        ``(False, detail)`` if rejected or the response is invalid; ``detail`` is safe to surface in
        ``code_safety_evaluation.detail``.
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    ).with_structured_output(JudgeOutput)
    # Fenced block helps the model locate the script; task is the same string codegen saw in ``code_pipeline``.
    human = (
        "## User task\n"
        + task.strip()
        + "\n\n## Generated Python\n```python\n"
        + code
        + "\n```\n"
    )
    prompt_messages = [SystemMessage(content=CODE_JUDGE_SYSTEM_PROMPT), HumanMessage(content=human)]
    with langfuse.start_as_current_observation(name="code_pipeline.llm_judge", as_type="generation", model=model_name, input=[serialize_message(message) for message in prompt_messages]) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            generation.update(output={"error": str(e)})
            return False, f"Judge model failed to produce valid structured output: {e}"
        generation.update(output=resp.model_dump())

    if resp.passed:
        return True, ""

    detail = resp.detail.strip()
    if not detail:
        detail = "LLM judge rejected the code (no reason provided)."
    return False, detail
