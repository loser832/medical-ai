"""Promote a validated candidate skill after explicit human approval."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from config import MEDICAL_SKILL_DIR, PROJECT_ROOT, TRACE2SKILL_ARTIFACT_DIR
from .validator import validate_skill_dir


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a reviewed medical skill candidate")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Confirm that a human reviewed the candidate and its evaluation results",
    )
    args = parser.parse_args()
    if not args.approve:
        raise RuntimeError("Promotion requires the explicit --approve flag")

    candidate = Path(args.candidate_dir).resolve()
    artifact_root = TRACE2SKILL_ARTIFACT_DIR.resolve()
    candidates_root = artifact_root / "candidates"
    production = MEDICAL_SKILL_DIR.resolve()
    if not _inside(candidate, candidates_root):
        raise ValueError(f"Candidate must be inside {candidates_root}")
    production_root = (PROJECT_ROOT / "skills").resolve()
    if not _inside(production, production_root):
        raise ValueError(f"Production skill must be inside {production_root}")
    errors = validate_skill_dir(candidate)
    if errors:
        raise ValueError("Candidate is invalid: " + "; ".join(errors))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = artifact_root / "promoted_backups" / stamp / production.name
    backup.parent.mkdir(parents=True, exist_ok=False)
    if production.exists():
        shutil.copytree(production, backup)

    staged = production.parent / f".{production.name}.staged-{stamp}"
    if staged.exists():
        raise FileExistsError(staged)
    shutil.copytree(candidate, staged)
    if production.exists():
        shutil.rmtree(production)
    staged.replace(production)

    promotion_record = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "candidate": str(candidate),
        "production": str(production),
        "previous_skill_backup": str(backup),
    }
    (backup.parent / "promotion.json").write_text(
        json.dumps(promotion_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Promoted candidate to {production}")
    print(f"Previous skill backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
