"""
``build_codegen_requirement`` tool implementation.

This tool reads graph state, recent conversation context, available files, and
the latest profiling result to produce a precise requirement that the
orchestrator can review before deciding whether to call ``code_pipeline``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import observe
from pydantic import BaseModel, Field

from observability.langfuse_handler import (
    get_langfuse_client,
    serialize_message,
)
from output_validation.build_codegen_requirement import BuildCodegenRequirementOutput
from prompts.build_codegen_requirement_prompt import BUILD_CODEGEN_REQUIREMENT_SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
PLANNING_MODEL = cfg["models"].get("codegen_requirement", cfg["models"]["orchestrator"])
langfuse = get_langfuse_client()


class BuildCodegenRequirementInput(BaseModel):
    """Arguments exposed to the orchestrator for ``build_codegen_requirement``."""

    brief: str = Field(
        description=(
            "Short orchestrator brief describing what code may need to accomplish. "
            "Summarize the user's objective and any important choices already made."
        ),
    )


@observe(name="tool.build_codegen_requirement", as_type="tool")
def _build_codegen_requirement_impl(
    brief: str,
    runtime: ToolRuntime | None = None,
) -> str:
    """
    Build a detailed execution requirement for potential code generation.

    The tool reads the current graph state so the orchestrator does not need to
    pass long context as visible tool arguments.
    """
    state = runtime.state if runtime is not None else {}
    messages = list(state.get("messages", []))
    data_schema = state.get("data_schema", "")
    message_summary = state.get("message_summary", "")
    latest_profile_result = state.get("latest_profile_result", "")

    human_payload = (
        "## Orchestrator Brief\n"
        f"{brief.strip()}\n\n"
        "## Available Data Files\n"
        f"{data_schema or '(no available data files)'}\n\n"
        "## Latest Profiling Result\n"
        f"{latest_profile_result or '(no profiling result available)'}\n\n"
        "## Conversation Summary\n"
        f"{message_summary or '(no summary available)'}"
    )

    prompt_messages = [
        SystemMessage(content=BUILD_CODEGEN_REQUIREMENT_SYSTEM_PROMPT),
        HumanMessage(content=human_payload),
        *messages,
    ]

    llm = ChatOpenAI(model=PLANNING_MODEL, temperature=0).with_structured_output(BuildCodegenRequirementOutput)

    with langfuse.start_as_current_observation(name="build_codegen_requirement.llm", as_type="generation", model=PLANNING_MODEL, input=[serialize_message(message) for message in prompt_messages]) as generation:
        try:
            resp = llm.invoke(prompt_messages)
        except Exception as e:
            fallback = BuildCodegenRequirementOutput(
                detailed_requirement=(
                    "1. Objective\n"
                    "Build a safe execution requirement could not be completed from the current context.\n\n"
                    "2. Available Data Context\n"
                    "Use the available workspace files and latest profiling result to restate the user's goal before code generation.\n\n"
                    "3. Known Findings From Profiling\n"
                    "The planner failed to produce a validated requirement from the current context.\n\n"
                    "4. Required Preprocessing\n"
                    "No preprocessing steps have been confirmed yet.\n\n"
                    "5. Forecasting / Evaluation Constraints\n"
                    "Preserve chronological integrity and avoid future-data leakage.\n\n"
                    "6. Output Expectations\n"
                    "Do not execute code until the blocking ambiguity is resolved.\n\n"
                    "7. Unresolved Ambiguities\n"
                    "A validated code-generation requirement could not be produced from the current context."
                ),
                dataset_paths=[],
                assumptions=[],
                needs_clarification=True,
                clarification_question=(
                    "I could not build a reliable execution requirement. What exact data, columns, or outputs should the code use?"
                ),
            )
            generation.update(output={"error": str(e)})
            langfuse.update_current_span(metadata={"status": "invalid_structured_output"})
            return fallback.model_dump_json()

        generation.update(output=resp.model_dump())

    langfuse.update_current_span(metadata={"needs_clarification": str(resp.needs_clarification).lower(), "dataset_path_count": len(resp.dataset_paths), "assumption_count": len(resp.assumptions)})
    return resp.model_dump_json()


@tool(args_schema=BuildCodegenRequirementInput)
def build_codegen_requirement(
    brief: str,
    runtime: ToolRuntime | None = None,
) -> str:
    """Draft a validated execution requirement before potential code generation."""
    return _build_codegen_requirement_impl(brief=brief, runtime=runtime)
