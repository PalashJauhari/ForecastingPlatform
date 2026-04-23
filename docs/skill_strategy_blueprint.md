# GaussianBlurr Skill Strategy

This document outlines a practical strategy for evolving GaussianBlurr into a more specialized data-science agent using a **staged skill system**.

The key idea remains the same: avoid one giant prompt, and inject focused expertise only when the task calls for it. The change is in **how aggressively** we build that system. Rather than launching a full three-layer skill stack at once, we roll it out in phases and prove value at each step.

---

## 1. Core Direction

The goal is **specialization via just-in-time scaffolding**.

Instead of teaching the agent everything in one prompt, we maintain a library of narrow domain skills and load only the skills that matter for the current request. This keeps the base agent lean while allowing it to behave more like a disciplined data scientist when the task requires it.

Examples:

- A KPI question should load metric reasoning, not forecasting guidance.
- A chart request should load visualization reasoning, not join-heavy prep logic unless needed.
- A quick forecast should load forecast reasoning and only the prep guidance required to make the series usable.

---

## 2. Recommended Rollout Order

The skills system should be built in **three stages**.

### Stage 1: Reasoning Skills First (`approach.md`)

**Target:** The orchestrator / planning phase.

**Purpose:** Improve task framing, decomposition, and tool selection before any code is generated.

This is the first rollout because it is the cheapest, lowest-risk, and most aligned with the current repo. GaussianBlurr already has reasoning-oriented skill files such as:

- `data_science_workflow`
- `tabular_prep`
- `metric_answering`
- `visual_answering`
- `one_shot_forecast`

The first milestone is simply:

1. identify which reasoning skills are relevant for a user turn
2. inject those `approach.md` files into the orchestrator context
3. measure whether routing improves planning quality and task completion

This gives the system sharper planning without adding a second code architecture.

### Stage 2: Deterministic Validators Second

**Target:** Concrete verification points around planning, code generation, and execution.

**Purpose:** Catch a small number of high-value failure modes that already matter in practice.

Do **not** start with a broad `validator.yaml` layer for every skill. A validator is only useful if it maps cleanly to a real check in code or in an enforceable judge policy.

Instead, add a few targeted validators for issues such as:

- bad joins or obvious grain mismatches
- forecast leakage / use of future-only covariates
- missing required output columns
- invalid file usage or unsupported write targets

These checks should be narrow, explicit, and tied to known failure patterns rather than aspirational "laws of data science."

### Stage 3: Small Reusable Execution Helpers Last

**Target:** Repeated implementation patterns that remain error-prone after stages 1 and 2.

**Purpose:** Reduce repeated mistakes in code generation without creating a shadow library of prompt-injected code.

This stage should be intentionally conservative.

Avoid a large `patterns.py` layer that attempts to encode all best practices as prompt context. That tends to create context bloat, version drift, and a second unofficial codebase that is hard to trust.

If reusable execution support is needed, keep it small:

- short templates
- tiny helper snippets
- narrow task-specific examples

Only add these when there is clear evidence that the same implementation mistake keeps recurring.

---

## 3. Skill Lifecycle

The lifecycle should also be staged.

### Phase A: Skill Identification

The skill router looks at:

- the user query
- the current `data_profile`
- optionally the current task mode inferred by the orchestrator

Example:

- **User query:** "Join sales and returns and forecast the next 3 months."
- **Matched reasoning skills:** `['tabular_prep', 'one_shot_forecast']`

### Phase B: Orchestrator Injection

The system loads the matched `approach.md` files and injects them into the orchestrator context.

**Result:** The orchestrator plans the task with better domain judgment, for example:

- checking grain before merging
- preserving chronological integrity
- clarifying target/horizon when forecasting details are ambiguous

### Phase C: Existing Code Pipeline Runs

The orchestrator continues to use the current execution path:

- requirement planning
- code generation
- Semgrep/static checks
- LLM judge
- sandboxed execution

At this stage, the main gain comes from **better planning before codegen**, not from a new execution-layer skill system.

### Phase D: Targeted Validators

Once Stage 1 is working, add a few deterministic validators at the right enforcement points.

Examples:

- join validator before accepting merge logic
- leakage validator before accepting a forecast workflow
- output validator after execution or before save

These validators should be treated as productized checks, not just text files that hope the model behaves.

### Phase E: Optional Execution Helpers

Only after repeated implementation patterns are proven valuable should the system load tiny reusable helpers for execution.

Even then, helpers should remain small and narrowly scoped.

---

## 4. Concrete Example: Join Skill Under the New Plan

Here is how a **join-oriented skill** should evolve under this staged strategy.

### Step 1: Reasoning skill first

`approach.md`

> "When joining datasets, identify the primary table first. Check whether the join key is unique on each side, confirm the intended grain, and prefer preserving the primary table unless the task explicitly calls for a different join."

This helps the orchestrator decide whether a join is needed and how it should be framed.

### Step 2: Add a validator only if the failure is common

Example validator behavior:

- block or warn when a supposed one-to-one join has duplicate keys
- flag suspicious row growth when the task did not imply a one-to-many relationship
- require the join key and join type to be explicit in the generated requirement

The important point is that this logic must run as a real check, not just sit in YAML as documentation.

### Step 3: Add a helper only if repetition justifies it

If the agent repeatedly writes bad merge code even after better planning and validation, add a **small merge helper/template**.

Do not begin with the helper layer.

---

## 5. Why This Order Is Better

1. **Fits the current repo**
   GaussianBlurr already has reasoning-skill assets and a real execution safety stack. Stage 1 extends what exists instead of introducing a second architecture immediately.

2. **Improves quality at the cheapest point**
   Better planning often fixes problems before they become code-generation or validation problems.

3. **Avoids fake safety**
   A broad `validator.yaml` design sounds rigorous, but many data-science rules are context-sensitive. A few real validators are more valuable than a large catalog of unenforced rules.

4. **Avoids context bloat**
   Large prompt-injected `patterns.py` files can become expensive, stale, and difficult to maintain.

5. **Creates a measurable roadmap**
   Each stage can be evaluated independently:
   - Does reasoning-skill routing improve planning?
   - Do validators reduce real failures?
   - Do helpers reduce repeated code mistakes?

---

## 6. Final Recommendation

The recommended blueprint for GaussianBlurr is:

1. **Wire the existing reasoning skills into the orchestrator and prove routing helps.**
2. **Add a tiny number of high-value deterministic validators for concrete failure modes.**
3. **Only then add small reusable execution helpers where repeated mistakes justify them.**

This preserves the core vision of modular specialization, but grounds it in a rollout path that is realistic for the current codebase and less likely to create complexity without reliability.
