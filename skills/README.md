# Skills library (`skills/`)

Domain playbooks live under **`orchestrator_skills/`** (execution-time reasoning; optional **`patterns.py`** reference snippets) and **`planner_skills/`** (task decomposition before execution). Each skill folder contains:

| File | Purpose |
|------|---------|
| **`skill.yaml`** | **Routing metadata** — `id` (must match folder name), `name`, `description`, optional **`aliases`** for legacy ids after merges. |
| **`approach.md`** | Full reasoning text injected when the skill is selected. |
| **`patterns.py`** | Optional; vetted code examples (repository reference; not auto-injected). |

## Naming convention

- **Folder name = skill id** — kebab-case (e.g. `data-grain-and-integrity`, `evaluation-design`).
- The **`id`** field in `skill.yaml` must equal the folder name or the manifest is ignored.

## When to extend vs add a skill

- **Extend** when the mental model is unchanged (e.g. new metric or checklist under **evaluation-design**, new plot type under **visualization**).
- **Add a new folder** when the workflow is a new concern (e.g. causal inference, experimentation).
- After **merging** skills, add **`aliases`** in `skill.yaml` so old **`active_skills`** ids still resolve via **`path_for_orchestrator_skill`** / **`path_for_planner_skill`**.

## Router

- **`invoke_planner_skill_pick`** / **`invoke_orchestrator_skill_pick`** ([`loader.py`](loader.py)) read manifests via [`registry.py`](registry.py), run one structured LLM per call, normalize aliases, apply caps from **`config.yaml`** (`skills.max_orchestrator_selected`, `skills.max_planner_selected`) and router models **`models.identify_orchestrator_skills`**, **`models.identify_planner_skills`**.

## Catalogs (REPL)

```bash
python3 -c "from skills.registry import orchestrator_catalog_text, planner_catalog_text; print(orchestrator_catalog_text()); print('---'); print(planner_catalog_text())"
```
