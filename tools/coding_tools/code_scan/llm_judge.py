"""
Layer between Semgrep and disk: an LLM reviews generated Python against the user task.

Used by ``code_pipeline`` as Step 2c (after static scan, before save/run).
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prompts.code_judge_prompt import CODE_JUDGE_SYSTEM_PROMPT


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
    resp = llm.invoke([SystemMessage(content=CODE_JUDGE_SYSTEM_PROMPT), HumanMessage(content=human)])
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
