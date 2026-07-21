"""Validate the local SKILL.md subset used by the medical application."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REFERENCE_RE = re.compile(r"\]\((references/[A-Za-z0-9_.-]+\.md)\)")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["SKILL.md must start with YAML frontmatter"]

    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, ["SKILL.md frontmatter is not closed"]

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            errors.append(f"Invalid frontmatter line: {line}")
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in metadata:
            errors.append(f"Duplicate frontmatter field: {key}")
        metadata[key] = value
    return metadata, errors


def validate_skill_dir(skill_dir: str | Path) -> list[str]:
    skill_path = Path(skill_dir).resolve()
    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        return [f"SKILL.md not found in {skill_path}"]

    text = skill_md.read_text(encoding="utf-8")
    metadata, errors = _parse_frontmatter(text)
    allowed_fields = {"name", "description"}
    extra_fields = sorted(set(metadata) - allowed_fields)
    if extra_fields:
        errors.append("Unsupported frontmatter fields: " + ", ".join(extra_fields))

    name = metadata.get("name", "")
    description = metadata.get("description", "")
    if not name:
        errors.append("Missing frontmatter field: name")
    elif not _NAME_RE.fullmatch(name):
        errors.append("Skill name must use lowercase letters, digits, and hyphens")
    elif skill_path.name != name:
        errors.append(f"Skill directory '{skill_path.name}' must match name '{name}'")

    if not description:
        errors.append("Missing frontmatter field: description")
    elif "TODO" in description:
        errors.append("Skill description still contains TODO")

    if "TODO" in text:
        errors.append("SKILL.md still contains TODO")
    if len(text.splitlines()) > 500:
        errors.append("SKILL.md exceeds the 500-line limit")

    for relative_path in sorted(set(_REFERENCE_RE.findall(text))):
        reference = (skill_path / relative_path).resolve()
        try:
            reference.relative_to(skill_path)
        except ValueError:
            errors.append(f"Reference escapes skill directory: {relative_path}")
            continue
        if not reference.is_file():
            errors.append(f"Missing referenced file: {relative_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a medical skill directory")
    parser.add_argument("skill_dir")
    args = parser.parse_args()
    errors = validate_skill_dir(args.skill_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Skill is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
