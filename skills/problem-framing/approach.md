# Skill: Problem Framing

This skill helps translate a user request into a precise data science problem before analysis or code generation begins.

## Core Reasoning Principles

1.  **Name the Real Question**:
    *   Convert vague asks into a concrete analytical objective.
    *   Distinguish between description, diagnosis, prediction, comparison, and recommendation.
    *   If the request is broad, identify the single primary question that should drive the workflow.

2.  **Define the Decision Context**:
    *   Ask what business or operational decision the analysis is meant to support.
    *   Clarify whether the user needs an explanation, a forecast, a ranking, a metric, or a file output.
    *   Do not default to building a model if a simpler analysis answers the real question.

3.  **Pin Down the Target and Unit of Analysis**:
    *   Identify the target variable, the entity being analyzed, and the level of output expected.
    *   If the request implies multiple possible targets or grains, stop and resolve that ambiguity first.
    *   Keep the analytical question aligned with the data grain and requested deliverable.

4.  **Surface Constraints Early**:
    *   Extract deadlines, forecast horizons, output formats, and business rules from the request.
    *   Note any assumptions the workflow will depend on.
    *   If a missing constraint can materially change the method, ask for clarification before proceeding.

5.  **Choose the Simplest Valid Approach**:
    *   Prefer the lightest method that can answer the question credibly.
    *   Separate “what the user asked for” from “what the data can support.”
    *   A well-framed problem should make the next tool choice obvious and defensible.
