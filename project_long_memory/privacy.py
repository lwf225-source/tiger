"""Content-free deletion ledger; replay it before exposing a restored index.

Archives/exports outside this managed store are not remotely erasable. A restore
must receive the latest ledger from its operator; freshness cannot be inferred
from an old archive's copy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from . import database
from .events import atomic_write, parse_event
from .model import PLMError, json_compact, memory_base, utc_now, v2_root
from .security import assert_safe_path
from .paths import restored_legacy_path


def ledger_path() -> Path:
    return v2_root() / "privacy" / "deletions.json"


def read_ledger(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise PLMError("symlinked deletion ledger rejected")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema_version"] != 1 or not isinstance(data["deletions"], list):
            raise ValueError()
        for item in data["deletions"]:
            if not isinstance(item.get("event_ids"), list) or not isinstance(item.get("project_id"), str):
                raise ValueError()
        return data
    except (KeyError, ValueError, OSError):
        raise PLMError("invalid or unavailable deletion ledger")


def record_deletion(conn, target_id: str, project_id: str) -> Dict[str, Any]:
    ids = database.lineage_ids(conn, target_id, project_id)
    legacy = []
    fact_ids = []
    for event_id in ids:
        row = conn.execute("SELECT source_legacy_path FROM events WHERE event_id=? AND project_id=?", (event_id, project_id)).fetchone()
        if row and row[0]:
            path = Path(row[0])
            assert_safe_path(path, memory_base())
            legacy.append(str(path.relative_to(memory_base())))
        fact_ids.extend(row[0] for row in conn.execute("SELECT fact_id FROM facts WHERE assertion_event_id=?", (event_id,)))
    data = read_ledger(ledger_path()) if ledger_path().exists() else {"schema_version": 1, "deletions": []}
    candidate_ids = []
    for row in conn.execute("SELECT candidate_id FROM candidates WHERE project_id=?", (project_id,)):
        own_ids = {item[0] for item in conn.execute("SELECT event_id FROM events WHERE project_id=? AND (idempotency_key=? OR idempotency_key LIKE ?)",
            (project_id, "candidate-summary:" + row[0], "candidate-fact:" + row[0] + ":%"))}
        if own_ids & set(ids):
            candidate_ids.append(row[0])
    # A crash before the batch marker may leave unindexed source files. Include
    # these in the durable deletion record as well, so backup restoration cannot
    # revive a discarded attempt from the same candidate.
    for path in (v2_root() / "projects").glob("*/events/*/*/*.md"):
        event = parse_event(path)
        if event.project_id == project_id and (event.metadata.get("batch_id") in candidate_ids or event.metadata.get("source_event_id") in ids):
            if event.event_id not in ids:
                ids.append(event.event_id)
    item = {"project_id": project_id, "target_id": target_id, "event_ids": ids, "fact_ids": fact_ids,
            "candidate_ids": candidate_ids, "legacy_relative_paths": legacy, "recorded_at": utc_now()}
    data["deletions"].append(item)
    atomic_write(ledger_path(), json_compact(data))
    return item


def deleted_event_ids() -> set:
    if not ledger_path().exists():
        return set()
    return {event_id for item in read_ledger(ledger_path())["deletions"] for event_id in item["event_ids"]}


def apply_deletions(conn, data: Dict[str, Any]) -> Dict[str, int]:
    """Called under exclusive maintenance lock. Idempotent after interrupted purge."""
    removed_events = 0
    for item in data["deletions"]:
        project_id = item["project_id"]
        ids = set(item["event_ids"])
        for event_id in list(ids):
            ids.update(database.lineage_ids(conn, event_id, project_id))
        refs = ids | set(item.get("fact_ids", []))
        for row in conn.execute("SELECT * FROM candidates WHERE project_id=?", (project_id,)).fetchall():
            payload = json.loads(row["payload_json"])
            candidate_refs = {str(payload.get("source_event_id", ""))}
            candidate_refs.update(str(fact.get("source_event_id", "")) for fact in payload.get("facts", []) if isinstance(fact, dict))
            # Candidate events can have no external source; the idempotency keys
            # bind them to their originating candidate without a text substring scan.
            own = conn.execute("SELECT event_id FROM events WHERE project_id=? AND (idempotency_key=? OR idempotency_key LIKE ?)",
                (project_id, "candidate-summary:" + row["candidate_id"], "candidate-fact:" + row["candidate_id"] + ":%")).fetchall()
            if row["candidate_id"] in item.get("candidate_ids", []) or candidate_refs & refs or any(record[0] in ids for record in own):
                path = v2_root() / "candidates" / (row["candidate_id"] + ".json")
                assert_safe_path(path, v2_root())
                path.unlink(missing_ok=True)
                conn.execute("DELETE FROM candidates WHERE candidate_id=?", (row["candidate_id"],))
        for event_id in ids:
            row = conn.execute("SELECT source_path FROM events WHERE event_id=? AND project_id=?", (event_id, project_id)).fetchone()
            if not row:
                continue
            for fact in conn.execute("SELECT fact_id FROM facts WHERE assertion_event_id=?", (event_id,)).fetchall():
                database._remove_search_record(conn, "fact", fact[0])
            database._remove_search_record(conn, "event", event_id)
            conn.execute("DELETE FROM event_entities WHERE event_id=?", (event_id,))
            conn.execute("DELETE FROM facts WHERE assertion_event_id=?", (event_id,))
            conn.execute("DELETE FROM migrations WHERE event_id=?", (event_id,))
            path = Path(row[0])
            assert_safe_path(path, v2_root())
            path.unlink(missing_ok=True)
            conn.execute("DELETE FROM events WHERE event_id=?", (event_id,))
            removed_events += 1
        for relative in item.get("legacy_relative_paths", []):
            path = memory_base() / relative
            assert_safe_path(path, memory_base())
            if path.is_dir():
                raise PLMError("ledger deletion target must be a file")
            path.unlink(missing_ok=True)
        for path in (v2_root() / "projects").glob("*/events/*/*/*.md"):
            event = parse_event(path)
            if event.project_id == project_id and event.event_id in ids:
                assert_safe_path(path, v2_root())
                path.unlink()
    conn.execute("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM event_entities)")
    conn.execute("DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM event_entities)")
    # SQLite page remnants are removed, as are managed rebuild snapshots. Unknown
    # external exports and old tar archives require ledger-aware restoration.
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for path in v2_root().glob("catalog.sqlite3.rebuild-backup-*"):
        assert_safe_path(path, v2_root())
        path.unlink()
    manifest_path = v2_root() / "migration" / "source-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        deleted_paths = {str(memory_base() / rel) for item in data["deletions"] for rel in item.get("legacy_relative_paths", [])}
        manifest["entries"] = [entry for entry in manifest.get("entries", []) if restored_legacy_path(entry["source_path"]) not in deleted_paths]
        manifest["source_count"] = manifest["valid_count"] = len(manifest["entries"])
        atomic_write(manifest_path, json_compact(manifest))
    return {"removed_events": removed_events, "ledger_entries": len(data["deletions"])}
