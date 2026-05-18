"""
Layer between Semgrep and disk: an LLM reviews generated Python against the user task.

Used by ``code_pipeline`` after static scan, before save/run.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)
from output_validation.code_generation import CodePipelineTask
from output_validation.judge_output import JudgeOutput
from prompts.code_judge_prompt import CODE_JUDGE_SYSTEM_PROMPT

from .safety_check import SafetyCheckResult

langfuse = get_langfuse_client()


def run_llm_judge(
    *,
    code: str,
    task: CodePipelineTask,
    model_name: str,
) -> SafetyCheckResult:
    """Ask the judge model to accept or reject the script against *task*."""
    llm = make_llm(model=model_name, temperature=0, output_schema=JudgeOutput)
    task_spec = json.dumps(task.model_dump(), ensure_ascii=False, indent=2)
    system_content = CODE_JUDGE_SYSTEM_PROMPT + "\n\n## Task (structured)\n" + task_spec.strip()
    human_content = f"```python\n{code}\n```"
    prompt_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]
    with langfuse.start_as_current_observation(
        name="code_pipeline.llm_judge",
        as_type="generation",
        model=model_name,
        input=[serialize_message(message) for message in prompt_messages],
    ) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            generation.update(output={"error": str(e)})
            return SafetyCheckResult(
                passed=False,
                source="judge",
                detail=f"Judge model failed to produce valid structured output: {e}",
            )
        generation.update(output=resp.model_dump())

    if resp.passed:
        return SafetyCheckResult(passed=True, source="judge")

    detail = resp.detail.strip()
    if not detail:
        detail = "LLM judge rejected the code (no reason provided)."
    return SafetyCheckResult(passed=False, source="judge", detail=detail)
