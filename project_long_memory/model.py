from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 2
EVENT_KINDS = {"core", "fact", "episode", "procedure", "artifact_ref"}
SCOPES = {"global", "project", "agent", "session", "shared"}
VISIBILITIES = {"private", "local", "shared"}
FACT_STATUSES = {"active", "superseded", "retracted", "expired"}
NAMESPACE = uuid.UUID("f10d42a8-b6cb-4f06-a034-a17f29009ee1")


class PLMError(RuntimeError):
    """User-facing PLM failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_time(value: Optional[str]) -> str:
    if not value:
        return utc_now()
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        try:
            parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S %z")
        except ValueError as exc:
            raise PLMError("invalid timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def memory_base() -> Path:
    configured = os.environ.get("PROJECT_LONG_MEMORY_DIR")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".codex" / "project-memory"


def v2_root() -> Path:
    return memory_base() / "v2"


def canonical_root(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def project_id_for_root(path: Path) -> str:
    canonical = str(canonical_root(path))
    return str(uuid.uuid5(NAMESPACE, "project:" + canonical))


def display_slug(path: Path) -> str:
    name = canonical_root(path).name or "root"
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "-", name).strip("-._")
    return (slug[:60] or "project") + "--" + project_id_for_root(path).replace("-", "")[:12]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_event_id(source_path: str, source_hash: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "legacy:" + source_path + ":" + source_hash))


@dataclass
class Event:
    metadata: Dict[str, Any]
    body: str
    path: Optional[Path] = None

    @property
    def event_id(self) -> str:
        return str(self.metadata["id"])

    @property
    def project_id(self) -> str:
        return str(self.metadata["project_id"])


@dataclass
class SearchResult:
    ref_id: str
    record_type: str
    title: str
    body: str
    project_id: str
    score: float
    reasons: List[str]
    source_path: str = ""
    status: str = ""
    valid_from: str = ""
    valid_to: str = ""
    source_event_id: str = ""
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "record_type": self.record_type,
            "title": self.title,
            "body": self.body,
            "project_id": self.project_id,
            "score": round(self.score, 8),
            "reasons": self.reasons,
            "source_path": self.source_path,
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_event_id": self.source_event_id,
            "evidence_spans": self.evidence_spans,
        }


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
