"""Run pinned Trace2Skill against an isolated candidate skill copy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    LLM_ENABLE_THINKING,
    MEDICAL_SKILL_DIR,
    MODEL_NAME,
    SERVE_URL,
    TRACE2SKILL_ARTIFACT_DIR,
    TRACE2SKILL_REPO,
)
from .validator import validate_skill_dir


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_records(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty list")
    for index, record in enumerate(data):
        if not isinstance(record, dict) or not record.get("instance_id"):
            raise ValueError(f"Record {index} is missing instance_id")
        if not isinstance(record.get("items"), list) or not record["items"]:
            raise ValueError(f"Record {index} is missing items")
    return len(data)


def _resolve_evolution_python(repo: Path, requested: str | None) -> Path:
    if requested:
        candidate = Path(requested).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Trace2Skill Python not found: {candidate}")
        return candidate
    candidates = [
        repo / ".venv" / "Scripts" / "python.exe",
        repo / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evolve an isolated medical skill candidate; dry-run is the default"
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--skill-dir", default=str(MEDICAL_SKILL_DIR))
    parser.add_argument("--trace2skill-repo", default=str(TRACE2SKILL_REPO))
    parser.add_argument(
        "--trace2skill-python",
        default=None,
        help="Python executable with Trace2Skill dependencies; auto-detects its .venv",
    )
    parser.add_argument("--artifact-dir", default=str(TRACE2SKILL_ARTIFACT_DIR))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=SERVE_URL)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--apply-to-candidate",
        action="store_true",
        help="Write evolved files to the isolated candidate copy, never the live skill",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json).resolve()
    skill_dir = Path(args.skill_dir).resolve()
    repo = Path(args.trace2skill_repo).resolve()
    artifact_root = Path(args.artifact_dir).resolve()
    evolution_python = _resolve_evolution_python(repo, args.trace2skill_python)

    if not input_json.is_file():
        raise FileNotFoundError(input_json)
    record_count = _validate_records(input_json)
    skill_errors = validate_skill_dir(skill_dir)
    if skill_errors:
        raise ValueError("Live skill is invalid: " + "; ".join(skill_errors))
    runner = repo / "skill_evolver" / "run_parallel_skill_evolution.py"
    validator = repo / "skills" / "skill-creator" / "scripts" / "quick_validate.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Trace2Skill runner not found: {runner}")
    if args.apply_to_candidate and not validator.is_file():
        raise FileNotFoundError(f"Trace2Skill validator shim not found: {validator}")

    child_env = os.environ.copy()
    if not child_env.get("OPENAI_API_KEY") and child_env.get("MODAGENT_API_KEY"):
        child_env["OPENAI_API_KEY"] = child_env["MODAGENT_API_KEY"]
    if not child_env.get("OPENAI_API_KEY"):
        raise RuntimeError("Set MODAGENT_API_KEY or OPENAI_API_KEY before running evolution")
    child_env["OPENAI_BASE_URL"] = args.base_url

    run_root = artifact_root / "candidates" / _timestamp()
    candidate_dir = run_root / skill_dir.name
    intermediates_dir = run_root / "intermediates"
    parse_failure_dir = run_root / "parse_failures"
    run_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(skill_dir, candidate_dir)

    command = [
        str(evolution_python),
        "-m",
        "skill_evolver.run_parallel_skill_evolution",
        "--input-json",
        str(input_json),
        "--skill-dir",
        str(candidate_dir),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--max-workers",
        str(args.max_workers),
        "--input-mode",
        "records",
        "--prompt",
        "generic",
        "--save-intermediates",
        "--intermediates-dir",
        str(intermediates_dir),
        "--parse-failure-dir",
        str(parse_failure_dir),
    ]
    if not args.apply_to_candidate:
        command.append("--dry-run")

    generation_config = {
        "extra_body": {"enable_thinking": bool(LLM_ENABLE_THINKING)},
    }
    command.extend(["--generation-config", json.dumps(generation_config)])

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply-to-candidate" if args.apply_to_candidate else "dry-run",
        "record_count": record_count,
        "source_skill": str(skill_dir),
        "candidate_skill": str(candidate_dir),
        "trace2skill_repo": str(repo),
        "trace2skill_python": str(evolution_python),
        "model": args.model,
        "base_url": args.base_url,
        "command": command,
    }
    (run_root / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Candidate skill: {candidate_dir}")
    print("Mode:", metadata["mode"])
    completed = subprocess.run(command, cwd=repo, env=child_env, check=False)
    if completed.returncode != 0:
        print(f"Trace2Skill failed with exit code {completed.returncode}")
        return completed.returncode

    candidate_errors = validate_skill_dir(candidate_dir)
    if candidate_errors:
        print("Candidate validation failed:")
        for error in candidate_errors:
            print(f"- {error}")
        return 1
    print(f"Candidate validation passed: {candidate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
