"""
Layer between Semgrep and disk: an LLM reviews generated Python against the user task.

Used by ``code_pipeline`` as Step 2c (after static scan, before save/run).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)

from output_validation.judge_output import JudgeOutput
from prompts.code_judge_prompt import CODE_JUDGE_SYSTEM_PROMPT

langfuse = get_langfuse_client()


def run_llm_judge(
    *,
    code: str,
    task: str,
    model_name: str,
) -> tuple[bool, str]:
    """
    Ask the judge model to accept or reject the script.

    Returns
        ``(True, "")`` if the judge approves.
        ``(False, detail)`` if rejected or the response is invalid; ``detail`` is safe to surface in
        ``code_safety_evaluation.detail``.
    """
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(JudgeOutput)
    # Policy in ``CODE_JUDGE_SYSTEM_PROMPT``; same task string codegen saw, then script only in the user turn.
    system_content = CODE_JUDGE_SYSTEM_PROMPT + "\n\n" + task.strip()
    human_content = f"```python\n{code}\n```"
    prompt_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]
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
