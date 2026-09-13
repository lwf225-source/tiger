from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from .model import EVENT_KINDS, FACT_STATUSES, SCHEMA_VERSION, SCOPES, VISIBILITIES, Event, PLMError, json_compact, v2_root
from .security import assert_safe_path, ensure_private_dir


COMMON_FIELDS = {
    "schema_version", "id", "project_id", "project_root", "scope", "scope_id",
    "kind", "title", "tags", "observed_at", "recorded_at", "source",
    "source_hash", "source_legacy_path", "visibility", "created_by",
    "idempotency_key", "operation", "target_id", "entities", "batch_id", "conditions",
}
FACT_FIELDS = {
    "fact_id", "fact_key", "value", "status", "valid_from", "valid_to",
    "retracted_at", "supersedes", "source_event_id", "confidence",
    "verified_at", "expires_at",
    "evidence_kind", "evidence",
}
ALLOWED_FIELDS = COMMON_FIELDS | FACT_FIELDS
REQUIRED_FIELDS = {
    "schema_version", "id", "project_id", "project_root", "scope", "scope_id",
    "kind", "title", "tags", "observed_at", "recorded_at", "source",
    "source_hash", "visibility", "created_by", "idempotency_key",
}


def validate_metadata(meta: Dict[str, Any]) -> None:
    unknown = set(meta) - ALLOWED_FIELDS
    missing = REQUIRED_FIELDS - set(meta)
    if unknown:
        raise PLMError("unknown event metadata fields: %s" % ", ".join(sorted(unknown)))
    if missing:
        raise PLMError("missing event metadata fields: %s" % ", ".join(sorted(missing)))
    if meta["schema_version"] != SCHEMA_VERSION:
        raise PLMError("unsupported schema_version")
    if meta["kind"] not in EVENT_KINDS:
        raise PLMError("invalid memory kind")
    if meta["scope"] not in SCOPES:
        raise PLMError("invalid scope")
    if meta["scope"] in {"session", "agent"} and (not meta["scope_id"] or meta["scope_id"] == meta["scope"]):
        raise PLMError("private scope requires an explicit scope_id")
    for field in ("id", "project_id", "batch_id"):
        if meta.get(field):
            import uuid
            try:
                uuid.UUID(str(meta[field]))
            except (ValueError, TypeError):
                raise PLMError("invalid identity field: " + field)
    if meta["visibility"] not in VISIBILITIES:
        raise PLMError("invalid visibility")
    if not isinstance(meta["tags"], list) or not all(isinstance(tag, str) for tag in meta["tags"]):
        raise PLMError("tags must be a list of strings")
    if "entities" in meta and not isinstance(meta["entities"], list):
        raise PLMError("entities must be a list")
    if meta["kind"] == "fact":
        for field in ("fact_id", "fact_key", "value", "status"):
            if field not in meta:
                raise PLMError("fact event missing %s" % field)
        if meta["status"] not in FACT_STATUSES:
            raise PLMError("invalid fact status")


def serialize_event(meta: Dict[str, Any], body: str) -> str:
    validate_metadata(meta)
    lines = ["---"]
    for key in sorted(meta):
        lines.append("%s: %s" % (key, json_compact(meta[key])))
    lines.extend(["---", "", body.rstrip(), ""])
    return "\n".join(lines)


def parse_event_text(text: str) -> Event:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "---":
        raise PLMError("not a PLM v2 event")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise PLMError("unterminated event frontmatter") from exc
    meta: Dict[str, Any] = {}
    for line in lines[1:end]:
        if ": " not in line:
            raise PLMError("invalid event frontmatter line")
        key, raw = line.split(": ", 1)
        if key in meta:
            raise PLMError("duplicate event metadata field")
        try:
            meta[key] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PLMError("invalid JSON metadata value for %s" % key) from exc
    validate_metadata(meta)
    return Event(meta, "\n".join(lines[end + 1:]).strip())


def parse_event(path: Path) -> Event:
    if path.is_symlink():
        raise PLMError("symlinked event rejected")
    event = parse_event_text(path.read_text(encoding="utf-8"))
    event.path = path
    return event


def parse_legacy_note(path: Path) -> Tuple[Dict[str, str], str]:
    if path.is_symlink():
        raise PLMError("symlinked legacy note rejected")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "---":
        raise PLMError("invalid legacy note")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise PLMError("unterminated legacy frontmatter") from exc
    meta: Dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    for field in ("title", "time", "project_root", "tags"):
        if field not in meta:
            raise PLMError("legacy note missing %s" % field)
    return meta, "\n".join(lines[end + 1:]).strip()


@contextmanager
def project_lock(project_id: str) -> Iterator[None]:
    lock_dir = v2_root() / "locks"
    ensure_private_dir(lock_dir)
    path = lock_dir / (project_id + ".lock")
    assert_safe_path(path, v2_root())
    if path.exists() and path.is_symlink():
        raise PLMError("symlinked project lock rejected")
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def maintenance_lock(exclusive: bool = False) -> Iterator[None]:
    """Coordinate normal mutations with maintenance operations such as rebuild."""
    lock_dir = v2_root() / "locks"
    ensure_private_dir(lock_dir)
    path = lock_dir / "maintenance.lock"
    assert_safe_path(path, v2_root())
    if path.exists() and path.is_symlink():
        raise PLMError("symlinked maintenance lock rejected")
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write(path: Path, text: str) -> None:
    assert_safe_path(path, v2_root())
    ensure_private_dir(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, str(path))
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def event_relative_path(project_folder: str, event_id: str, recorded_at: str) -> Path:
    dt = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    stamp = dt.strftime("%Y%m%dT%H%M%SZ")
    return Path("projects") / project_folder / "events" / dt.strftime("%Y") / dt.strftime("%m") / (stamp + "_" + event_id + ".md")
