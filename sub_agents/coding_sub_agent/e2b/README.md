# E2B (coding sub-agent)

Cloud sandbox template and build tooling for `E2BExecute` in `graph.py`.

| File | Purpose |
|------|---------|
| `requirements-sandbox.txt` | Python packages baked into the template image |
| `template_def.py` | E2B `Template` definition (used only at build time) |
| `build_e2b_template.py` | One-time CLI to build/replace the template |

## Build template

From repo root (venv active, `E2B_API_KEY` in `../.env`):

```bash
pip install 'e2b>=2.3.0'
python sub_agents/coding_sub_agent/e2b/build_e2b_template.py --write-env
```

Runtime needs `e2b-code-interpreter>=2.7.0` on the host and `E2B_TEMPLATE_NAME` in `sub_agents/coding_sub_agent/.env`. See the root [README](../../../README.md) **E2B template** section for timeouts and `E2B_KILL_SANDBOX`.
