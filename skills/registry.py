"""Discover skills on disk via ``skill.yaml`` manifests and resolve canonical ids / aliases."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml

SKILLS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SkillManifest:
    id: str
    name: str
    description: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...]


def _read_manifest(skill_dir: Path) -> SkillManifest | None:
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
    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    aliases = raw.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    return SkillManifest(
        id=sid,
        name=name,
        description=desc,
        tags=tuple(str(t) for t in tags),
        aliases=tuple(str(a).strip() for a in aliases if str(a).strip()),
    )


@lru_cache(maxsize=1)
def load_skill_manifests() -> tuple[SkillManifest, ...]:
    manifests: list[SkillManifest] = []
    for child in sorted(SKILLS_ROOT.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        m = _read_manifest(child)
        if m is not None:
            manifests.append(m)
    return tuple(manifests)


@lru_cache(maxsize=1)
def alias_to_canonical_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for man in load_skill_manifests():
        for a in man.aliases:
            m[a] = man.id
    return m


def catalog_for_prompt() -> str:
    """Human-readable block for the skill-router LLM (ids + names + descriptions)."""
    lines: list[str] = []
    for man in load_skill_manifests():
        lines.append(f"- **id:** `{man.id}`")
        lines.append(f"  **name:** {man.name}")
        lines.append(f"  **description:** {man.description}")
        if man.tags:
            lines.append(f"  **tags:** {', '.join(man.tags)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def canonical_skill_ids() -> frozenset[str]:
    return frozenset(m.id for m in load_skill_manifests())


def normalize_selected_skills(selected: list[str], max_skills: int) -> list[str]:
    """
    Map aliases to canonical folder ids, drop unknowns, dedupe, preserve order, cap length.
    """
    aliases = alias_to_canonical_map()
    valid = canonical_skill_ids()
    seen: set[str] = set()
    out: list[str] = []
    for raw in selected:
        sid = (raw or "").strip()
        if not sid:
            continue
        canon = aliases.get(sid, sid)
        if canon not in valid or canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
        if len(out) >= max_skills:
            break
    return out


def resolve_skill_dir(skill_ref: str) -> Path | None:
    """Return the skill directory for a canonical id or legacy alias."""
    aliases = alias_to_canonical_map()
    sid = aliases.get(skill_ref.strip(), skill_ref.strip())
    path = SKILLS_ROOT / sid
    if path.is_dir() and (path / "approach.md").is_file():
        return path
    return None


def clear_registry_cache() -> None:
    """For tests / hot reload."""
    load_skill_manifests.cache_clear()
    alias_to_canonical_map.cache_clear()
