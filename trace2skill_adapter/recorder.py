"""Thread-safe, privacy-conscious trace capture for offline skill evolution."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:(?:\+?86)[ -]?)?1[3-9]\d{9}(?!\d)")
_PRC_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(text: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return _PRC_ID_RE.sub("[REDACTED_PRC_ID]", text)


class TraceRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        session_id: str,
        question: str,
        model: str,
        capture_content: bool = False,
    ):
        self.trace_id = uuid.uuid4().hex
        self.output_dir = Path(output_dir).resolve()
        self.capture_content = capture_content
        self._lock = threading.Lock()
        self._finalized = False
        self._record: dict[str, Any] = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "session_hash": hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16],
            "model": model,
            "capture_mode": "redacted_content" if capture_content else "hash_only",
            "started_at": _utc_now(),
            "question": self._protect(question),
            "context": {},
            "events": [],
            "agent_calls": [],
        }

    def _protect(self, value: Any) -> Any:
        if value is None:
            return None
        text = _redact(str(value))
        if self.capture_content:
            return text
        return {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chars": len(text),
        }

    def set_context(self, **values: Any) -> None:
        with self._lock:
            self._record["context"].update(values)

    def record_event(
        self,
        event_type: str,
        content: str,
        agent_name: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        if event_type == "incremental":
            return
        event = {
            "timestamp": _utc_now(),
            "type": event_type,
            "agent": agent_name,
            "content": self._protect(content),
            "details": [self._protect(item) for item in (details or [])],
        }
        with self._lock:
            self._record["events"].append(event)

    def record_agent_call(
        self,
        role: str,
        prompt: str,
        response: str | None,
        latency_ms: int,
        status: str,
        error: str | None = None,
    ) -> None:
        call = {
            "timestamp": _utc_now(),
            "role": role,
            "prompt": self._protect(prompt),
            "response": self._protect(response),
            "latency_ms": latency_ms,
            "status": status,
            "error": self._protect(error),
        }
        with self._lock:
            self._record["agent_calls"].append(call)

    def finalize(
        self,
        outcome: str,
        final_answer: str | None = None,
        error: str | None = None,
    ) -> Path | None:
        with self._lock:
            if self._finalized:
                return None
            self._finalized = True
            self._record["finished_at"] = _utc_now()
            self._record["outcome"] = outcome
            self._record["final_answer"] = self._protect(final_answer)
            self._record["error"] = self._protect(error)
            payload = json.dumps(self._record, ensure_ascii=False, indent=2)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"trace_{self.trace_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
        return target


def create_trace_recorder(
    *,
    enabled: bool,
    output_dir: str | Path,
    session_id: str,
    question: str,
    model: str,
    capture_content: bool,
) -> TraceRecorder | None:
    if not enabled:
        return None
    return TraceRecorder(
        output_dir=output_dir,
        session_id=session_id,
        question=question,
        model=model,
        capture_content=capture_content,
    )
