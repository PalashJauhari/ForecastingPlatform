# Skills library (`skills/`)

Domain playbooks for the forecasting agent. Each skill is a folder with:

| File | Purpose |
|------|---------|
| **`skill.yaml`** | **Routing metadata** — `id` (must match folder name), `name`, `description`, optional `tags`, optional `aliases` for legacy ids after merges. |
| **`approach.md`** | Full reasoning text injected into orchestrator / requirement planning when the skill is selected. |
| **`patterns.py`** | Optional; vetted snippets injected into **`code_pipeline`** codegen. |

## Naming convention

- **Folder name = skill id** — kebab-case (e.g. `data-grain-and-integrity`, `evaluation-design`).
- The **`id`** field in `skill.yaml` must equal the folder name or the manifest is ignored.

## When to extend vs add a skill

- **Extend** when the mental model is unchanged (e.g. new metric or checklist under **evaluation-design**, new plot type under **visualization**).
- **Add a new folder** when the workflow is a new concern (e.g. causal inference, experimentation).
- After **merging** skills, add **`aliases`** in `skill.yaml` so old checkpoint `active_skills` ids still resolve via **`resolve_skill_dir`**.

## Router

- **`IdentifySkills`** ([`loader.py`](loader.py)) reads all `skill.yaml` files via [`registry.py`](registry.py), sends **ids + descriptions** to the router model, then normalizes aliases and caps count (**`config.yaml` → `skills.max_selected`**).

## Current ids

Run `python -c "from skills.registry import catalog_for_prompt; print(catalog_for_prompt())"` to print the live catalog.
