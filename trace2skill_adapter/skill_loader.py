"""Load a validated, versioned medical skill for runtime prompt injection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .validator import validate_skill_dir


_REFERENCE_RE = re.compile(r"\]\((references/[A-Za-z0-9_.-]+\.md)\)")


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    version: str
    content: str
    source_dir: Path


def _strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text.strip()
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1:]).strip()
    return text.strip()


def load_medical_skill(skill_dir: str | Path, max_chars: int = 16000) -> LoadedSkill:
    skill_path = Path(skill_dir).resolve()
    errors = validate_skill_dir(skill_path)
    if errors:
        raise ValueError("Invalid medical skill: " + "; ".join(errors))

    skill_md = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    parts = [_strip_frontmatter(skill_md)]
    for relative_path in sorted(set(_REFERENCE_RE.findall(skill_md))):
        reference = skill_path / relative_path
        parts.append(f"\n\n## Reference: {relative_path}\n\n{reference.read_text(encoding='utf-8').strip()}")

    content = "".join(parts).strip()
    if len(content) > max_chars:
        raise ValueError(
            f"Medical skill content has {len(content)} characters, exceeding {max_chars}"
        )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return LoadedSkill(
        name=skill_path.name,
        version=digest,
        content=content,
        source_dir=skill_path,
    )


def inject_skill(base_prompt: str, role: str, skill: LoadedSkill | None) -> str:
    if skill is None:
        return base_prompt
    return f"""{base_prompt.rstrip()}

以下是当前已审核的医疗多智能体工作流技能。它约束你的工作方式，不替代检索证据，
也不允许你据此虚构患者事实或医学知识。

<medical_skill name="{skill.name}" version="{skill.version}" role="{role}">
{skill.content}
</medical_skill>"""
