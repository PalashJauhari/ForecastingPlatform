"""
Layer between Semgrep and disk: an LLM reviews generated Python against the user task.

Used by ``code_pipeline`` as Step 2c (after static scan, before save/run).
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from observability.langfuse_handler import (
    extract_usage_details,
    get_langfuse_client,
    serialize_message,
)

from prompts.code_judge_prompt import CODE_JUDGE_SYSTEM_PROMPT

langfuse = get_langfuse_client()


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
        model_kwargs={"response_format": {"type": "json_object"}},
    )
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
        resp = llm.invoke(prompt_messages)
        generation.update(output=serialize_message(resp), usage_details=extract_usage_details(resp))
    raw = resp.content if hasattr(resp, "content") else str(resp)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, "Judge model returned invalid JSON."

    # Contract: ``{"passed": true}`` or ``{"passed": false, "detail": "..."}`` (see CODE_JUDGE_SYSTEM_PROMPT).
    if data.get("passed") is True:
        return True, ""

    detail = (data.get("detail") or "").strip()
    if not detail:
        detail = "LLM judge rejected the code (no reason provided)."
    return False, detail
