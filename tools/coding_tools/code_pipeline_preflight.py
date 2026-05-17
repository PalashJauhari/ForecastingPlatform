"""
Codegen **preflight** planner: expands an orchestrator task into ``CodegenPreflightOutput``.

Invoked internally from ``code_pipeline`` when ``detail_execution_requirement_first`` is true.
Reads graph state from ``ToolRuntime`` (messages, profiling, skills, summary).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage
from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)
from output_validation.build_codegen_requirement import CodegenPreflightOutput
from prompts.build_codegen_requirement_prompt import BUILD_CODEGEN_REQUIREMENT_SYSTEM_PROMPT
from skills.loader import LoadReasoningSkills

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
PLANNING_MODEL = cfg["models"].get("codegen_requirement", cfg["models"]["orchestrator"])
langfuse = get_langfuse_client()


def run_codegen_preflight(*, orchestrator_task: str, runtime: ToolRuntime) -> CodegenPreflightOutput:
    """
    Build a structured execution requirement from graph state plus ``orchestrator_task`` (typically the visible ``task`` arg).
    """
    state = runtime.state or {}
    messages = list(state.get("messages", []))
    data_profile = state.get("data_profile", [])
    data_profile_text = json.dumps(data_profile, indent=2, default=str)
    message_summary = state.get("message_summary", "")
    active_skills = state.get("active_skills", [])
    skill_guidance = LoadReasoningSkills(active_skills)

    human_payload = (
        "## Files\n"
        "In **every** field of your JSON and in `detailed_requirement`, use **filename only** "
        "(e.g. `sales.csv`). Never write `agent_filesystem/`, session ids, slashes, or paths. "
        "**Plot/chart outputs** in requirements must use **`.png` or `.svg` only** (codegen rejects other image extensions).\n\n"
        "## Orchestrator task (preflight brief)\n"
        f"{orchestrator_task.strip()}\n\n"
        "## Session workspace (data_profile list from graph state)\n"
        f"{data_profile_text}\n\n"
        "## Expert Guidance (Reasoning Skills)\n"
        f"{skill_guidance or '(none)'}\n\n"
        "## Conversation Summary\n"
        f"{message_summary or '(none)'}\n\n"
    )

    prompt_messages = [
        SystemMessage(content=BUILD_CODEGEN_REQUIREMENT_SYSTEM_PROMPT),
        HumanMessage(content=human_payload),
        *messages,
    ]

    llm = make_llm(
        model=PLANNING_MODEL,
        temperature=0,
        output_schema=CodegenPreflightOutput,
    )

    with langfuse.start_as_current_observation(
        name="code_pipeline.preflight.llm",
        as_type="generation",
        model=PLANNING_MODEL,
        input=[serialize_message(message) for message in prompt_messages],
    ) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            fallback = CodegenPreflightOutput(
                detailed_requirement=(
                    "1. Objective\n"
                    "Execute the orchestrator task using workspace data; planner output validation failed "
                    f"({type(e).__name__}).\n\n"
                    "2. Available Data Context\n"
                    "Use filenames from the Session workspace data_profile.\n\n"
                    "3. Known Findings From Profiling\n"
                    "Treat profiling in context as authoritative where present.\n\n"
                    "4. Required Preprocessing\n"
                    "Apply standard cleaning only as implied by columns and dtypes.\n\n"
                    "5. Forecasting / Evaluation Constraints\n"
                    "Preserve chronological order and avoid leakage.\n\n"
                    "6. Output Expectations\n"
                    "Follow filenames and artifacts named in the original orchestrator task; tabular `.csv`/`.xlsx`; "
                    "plots `.png`/`.svg` only.\n\n"
                    "7. Unresolved Ambiguities\n"
                    "Assume reasonable defaults and state them explicitly in assumptions in the next codegen step if needed."
                ),
                dataset_paths=[],
                assumptions=[
                    "Preflight planner could not validate structured JSON; codegen received a conservative fallback requirement.",
                    str(e)[:240],
                ],
            )
            generation.update(output={"error": str(e)})
            langfuse.update_current_span(metadata={"status": "invalid_structured_output"})
            return fallback

        generation.update(output=resp.model_dump())

    langfuse.update_current_span(
        metadata={
            "dataset_path_count": len(resp.dataset_paths),
            "assumption_count": len(resp.assumptions),
        },
    )
    return resp
