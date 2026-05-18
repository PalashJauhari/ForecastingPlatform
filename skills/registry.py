"""Discover skills under ``orchestrator_skills/`` and ``planner_skills/`` via ``skill.yaml`` manifests."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

SKILLS_ROOT = Path(__file__).resolve().parent

_ORCH_ROOT = SKILLS_ROOT / "orchestrator_skills"
_PLAN_ROOT = SKILLS_ROOT / "planner_skills"

_Branch = Literal["orchestrator", "planner"]


@dataclass(frozen=True)
class SkillManifest:
    id: str
    name: str
    description: str
    aliases: tuple[str, ...]
    branch: _Branch


def _read_manifest(skill_dir: Path, branch: _Branch) -> SkillManifest | None:
    meta_path = skill_dir / "skill.yaml"
    if not meta_path.is_file():
        return None
    raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    sid = str(raw.get("id") or skill_dir.name).strip()
    if sid != skill_dir.name:
        return None
    name = str(raw.get("name") or sid).strip()
    desc = str(raw.get("description") or "").strip()
    if not desc:
        return None
    aliases_raw = raw.get("aliases") or []
    if not isinstance(aliases_raw, list):
        aliases_raw = []
    aliases = tuple(str(a).strip() for a in aliases_raw if str(a).strip())
    return SkillManifest(id=sid, name=name, description=desc, aliases=aliases, branch=branch)


def _manifest_list(root: Path, branch: _Branch) -> list[SkillManifest]:
    if not root.is_dir():
        return []
    out: list[SkillManifest] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        m = _read_manifest(child, branch)
        if m is not None:
            out.append(m)
    return out


@lru_cache(maxsize=1)
def load_skill_manifests() -> tuple[SkillManifest, ...]:
    combined = tuple(_manifest_list(_ORCH_ROOT, "orchestrator") + _manifest_list(_PLAN_ROOT, "planner"))
    ids = [m.id for m in combined]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise RuntimeError(f"Duplicate skill ids across dirs (must be unique): {sorted(dupes)}")
    return combined


def _by_branch(branch: _Branch) -> tuple[SkillManifest, ...]:
    return tuple(m for m in load_skill_manifests() if m.branch == branch)


def orchestrator_catalog_text() -> str:
    return _catalog_lines(_by_branch("orchestrator"))


def planner_catalog_text() -> str:
    return _catalog_lines(_by_branch("planner"))


def _catalog_lines(manifests: tuple[SkillManifest, ...]) -> str:
    lines: list[str] = []
    for man in manifests:
        lines.append(f"- **id:** `{man.id}`")
        lines.append(f"  **name:** {man.name}")
        lines.append(f"  **description:** {man.description}")
        lines.append("")
    return "\n".join(lines).rstrip()


def allowed_orchestrator_skill_ids() -> frozenset[str]:
    return frozenset(m.id for m in _by_branch("orchestrator"))


def allowed_planner_skill_ids() -> frozenset[str]:
    return frozenset(m.id for m in _by_branch("planner"))


@lru_cache(maxsize=1)
def _alias_canonical_pairs() -> dict[str, str]:
    m: dict[str, str] = {}
    for man in load_skill_manifests():
        for a in man.aliases:
            m[a] = man.id
    return m


def normalize_skill_pick(raw: list[str], *, allowed_ids: frozenset[str], max_n: int) -> list[str]:
    """Map aliases → canonical ids, preserve order, dedupe, cap length."""
    aliases = _alias_canonical_pairs()
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        sid = (item or "").strip()
        if not sid:
            continue
        canon = aliases.get(sid, sid)
        if canon not in allowed_ids or canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
        if len(out) >= max_n:
            break
    return out


def path_for_orchestrator_skill(skill_ref: str) -> Path | None:
    aliases = _alias_canonical_pairs()
    sid = aliases.get(skill_ref.strip(), skill_ref.strip())
    path = _ORCH_ROOT / sid
    if path.is_dir() and (path / "approach.md").is_file():
        return path
    return None


def path_for_planner_skill(skill_ref: str) -> Path | None:
    aliases = _alias_canonical_pairs()
    sid = aliases.get(skill_ref.strip(), skill_ref.strip())
    path = _PLAN_ROOT / sid
    if path.is_dir() and (path / "approach.md").is_file():
        return path
    return None


def clear_registry_cache() -> None:
    load_skill_manifests.cache_clear()
    _alias_canonical_pairs.cache_clear()
