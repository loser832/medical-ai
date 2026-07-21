"""Convert manually reviewed medical cases into Trace2Skill record JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_item(item_type: str, item: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "type": item_type,
        "title": str(item.get("title", "")).strip(),
        "description": str(item.get("description", "")).strip(),
        "content": str(item.get("content", "")).strip(),
    }
    missing = [key for key in ("title", "description", "content") if not normalized[key]]
    if missing:
        raise ValueError(f"{item_type} item is missing: {', '.join(missing)}")
    return normalized


def normalize_reviewed_case(case: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(case.get("instance_id", "")).strip()
    if not instance_id:
        raise ValueError("Reviewed case is missing instance_id")

    items: list[dict[str, Any]] = []
    if "items" in case:
        raw_items = case["items"]
        if not isinstance(raw_items, list):
            raise ValueError(f"{instance_id}: items must be a list")
        for item in raw_items:
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type not in {"failure_cause", "failure_memory"}:
                raise ValueError(f"{instance_id}: unsupported item type {item_type!r}")
            items.append(_normalize_item(item_type, item))
    else:
        for item in case.get("failure_causes", []):
            items.append(_normalize_item("failure_cause", item))
        for item in case.get("failure_memories", []):
            items.append(_normalize_item("failure_memory", item))

    if not items:
        raise ValueError(f"{instance_id}: no failure analysis items")
    if not any(item["type"] == "failure_cause" for item in items):
        raise ValueError(f"{instance_id}: at least one failure_cause is required")
    return {"instance_id": instance_id, "items": items}


def load_reviewed_cases(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            record = normalize_reviewed_case(case)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
        if record["instance_id"] in seen_ids:
            raise ValueError(f"{source}:{line_number}: duplicate instance_id")
        seen_ids.add(record["instance_id"])
        records.append(record)
    if not records:
        raise ValueError(f"No reviewed cases found in {source}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare manually reviewed medical failure records for Trace2Skill"
    )
    parser.add_argument("--input", required=True, help="Reviewed JSONL file")
    parser.add_argument("--output", required=True, help="Output parsed_error_records.json")
    args = parser.parse_args()

    records = load_reviewed_cases(args.input)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} reviewed records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
