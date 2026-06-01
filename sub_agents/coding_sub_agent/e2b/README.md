# E2B (coding sub-agent)

Cloud sandbox template and build tooling for `E2BExecute` in `graph.py`.

| File | Purpose |
|------|---------|
| `requirements-sandbox.txt` | Python packages baked into the template image |
| `template_def.py` | E2B `Template` definition (used only at build time) |
| `build_e2b_template.py` | One-time CLI to build/replace the template |

## Build template

From repo root (venv active, `CODING_E2B_API_KEY` in root `.env`):

```bash
pip install 'e2b>=2.3.0'
python sub_agents/coding_sub_agent/e2b/build_e2b_template.py --write-env
```

Runtime needs `e2b-code-interpreter>=2.7.0` on the host and `CODING_E2B_TEMPLATE_NAME` in root `.env`. See the root [README](../../../README.md) **E2B template** section for timeouts and `CODING_E2B_KILL_SANDBOX`.

## Network isolation

Sandboxes are created with **`allow_internet_access=False`** in `E2BExecute` (`graph.py`). This is a runtime flag at `Sandbox.create()` — not configured in the template image. Do not change it without security review.
