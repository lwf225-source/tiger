from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import Event, PLMError, json_compact, utc_now, v2_root
from .security import ensure_private_dir
from .vector import encode_vector
from .paths import restored_legacy_path


DATABASE_VERSION = 5

# Platform session identifiers are private routing metadata.  AML derives a
# one-way digest from them for session-level RRF, but that digest must never
# become a lexical/semantic retrieval feature.
INTERNAL_RETRIEVAL_TAG_PREFIXES = ("aml-session:",)


def searchable_tags(tags: Iterable[object]) -> List[str]:
    """Return public retrieval tags, excluding internal routing metadata."""
    return [str(tag) for tag in tags
            if not str(tag).lower().startswith(INTERNAL_RETRIEVAL_TAG_PREFIXES)]


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    canonical_root TEXT NOT NULL,
    display_name TEXT NOT NULL,
    folder_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS project_aliases (
    alias_path TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_legacy_path TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL UNIQUE,
    visibility TEXT NOT NULL,
    created_by TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    operation TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS events_idempotency
ON events(project_id, idempotency_key) WHERE idempotency_key <> '';
CREATE INDEX IF NOT EXISTS events_project_time ON events(project_id, recorded_at DESC);
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    fact_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    status TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    retracted_at TEXT NOT NULL,
    supersedes TEXT NOT NULL,
    assertion_event_id TEXT NOT NULL REFERENCES events(event_id),
    source_event_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    verified_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_current ON facts(project_id, fact_key, status, recorded_at DESC);
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, entity_type, canonical_name)
);
CREATE TABLE IF NOT EXISTS entity_aliases (
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    PRIMARY KEY(project_id, alias)
);
CREATE TABLE IF NOT EXISTS event_entities (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    relation TEXT NOT NULL,
    PRIMARY KEY(event_id, entity_id, relation)
);
CREATE TABLE IF NOT EXISTS vectors (
    record_type TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY(record_type, ref_id)
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    reason TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS migrations (
    source_path TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    event_id TEXT NOT NULL,
    migrated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    project_id TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    legacy_refs_json TEXT NOT NULL,
    v2_refs_json TEXT NOT NULL,
    overlap REAL NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    project_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    client TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    success INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_runs_created ON usage_runs(created_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    ref_id UNINDEXED,
    project_id UNINDEXED,
    record_type UNINDEXED,
    title,
    tags,
    body,
    tokenize='trigram'
);
"""


def db_path() -> Path:
    return v2_root() / "catalog.sqlite3"


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    target = path or db_path()
    ensure_private_dir(target.parent)
    conn = sqlite3.connect(str(target), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Switching a fresh catalog into WAL needs an exclusive moment; racing
    # connections (threaded servers, parallel CLI processes) can hold the
    # database right then, and the busy handler does not cover the journal
    # mode transition. Retry briefly before giving up.
    for attempt in range(100):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == 99:
                conn.close()
                raise
            time.sleep(0.05)
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if "source_event_id" not in columns:
        try:
            conn.execute("ALTER TABLE events ADD COLUMN source_event_id TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            # On a fresh database, concurrent connect() calls (multi-threaded
            # servers, parallel CLI processes) can both observe the column as
            # missing; one wins the ALTER and the other must re-verify instead
            # of failing the whole connection.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            if "source_event_id" not in columns:
                raise
    conn.execute("CREATE INDEX IF NOT EXISTS events_source ON events(source_event_id)")
    conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('database_version',?)", (str(DATABASE_VERSION),))
    conn.commit()
    try:
        os.chmod(str(target), 0o600)
    except OSError:
        pass
    return conn


def connect_readonly(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the derived search index without creating or mutating anything."""
    target = (path or db_path()).expanduser().resolve(strict=False)
    if not target.is_file():
        raise PLMError("memory index does not exist: %s" % target)
    conn = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def register_project(conn: sqlite3.Connection, project_id: str, canonical_root: str, display_name: str, folder_name: str) -> None:
    now = utc_now()
    conn.execute(
        "INSERT OR IGNORE INTO projects(project_id,canonical_root,display_name,folder_name,created_at) VALUES(?,?,?,?,?)",
        (project_id, canonical_root, display_name, folder_name, now),
    )
    row = conn.execute("SELECT project_id FROM projects WHERE project_id=?", (project_id,)).fetchone()
    if not row:
        raise PLMError("could not register project")
    conn.execute(
        "INSERT INTO project_aliases(alias_path,project_id,active,created_at) VALUES(?,?,1,?) "
        "ON CONFLICT(alias_path) DO UPDATE SET project_id=excluded.project_id, active=1",
        (canonical_root, project_id, now),
    )


def project_for_alias(conn: sqlite3.Connection, path: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT p.* FROM project_aliases a JOIN projects p ON p.project_id=a.project_id "
        "WHERE a.alias_path=? AND a.active=1",
        (path,),
    ).fetchone()


def event_for_idempotency(conn: sqlite3.Connection, project_id: str, key: str) -> Optional[sqlite3.Row]:
    if not key:
        return None
    return conn.execute("SELECT * FROM events WHERE project_id=? AND idempotency_key=?", (project_id, key)).fetchone()


def _remove_search_record(conn: sqlite3.Connection, record_type: str, ref_id: str) -> None:
    conn.execute("DELETE FROM memory_fts WHERE record_type=? AND ref_id=?", (record_type, ref_id))
    conn.execute("DELETE FROM vectors WHERE record_type=? AND ref_id=?", (record_type, ref_id))


def _index_search_record(
    conn: sqlite3.Connection,
    record_type: str,
    ref_id: str,
    project_id: str,
    title: str,
    tags: str,
    body: str,
) -> None:
    _remove_search_record(conn, record_type, ref_id)
    conn.execute(
        "INSERT INTO memory_fts(ref_id,project_id,record_type,title,tags,body) VALUES(?,?,?,?,?,?)",
        (ref_id, project_id, record_type, title, tags, body),
    )
    conn.execute(
        "INSERT OR REPLACE INTO vectors(record_type,ref_id,project_id,vector) VALUES(?,?,?,?)",
        (record_type, ref_id, project_id, encode_vector(" ".join((title, tags, body)))),
    )


def index_event(conn: sqlite3.Connection, event: Event) -> None:
    meta = event.metadata
    path = str(event.path or "")
    tags = meta.get("tags", [])
    operation = str(meta.get("operation", ""))
    target_id = str(meta.get("target_id", ""))
    conn.execute(
        "INSERT OR REPLACE INTO events(event_id,project_id,scope,scope_id,kind,title,tags_json,body,status,"
        "observed_at,recorded_at,source,source_hash,source_legacy_path,source_path,visibility,created_by,idempotency_key,operation,target_id) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            meta["id"], meta["project_id"], meta["scope"], meta["scope_id"], meta["kind"],
            meta["title"], json_compact(tags), event.body, "active", meta["observed_at"],
            meta["recorded_at"], meta["source"], meta["source_hash"], restored_legacy_path(str(meta.get("source_legacy_path", ""))), path,
            meta["visibility"], meta["created_by"], meta["idempotency_key"], operation, target_id,
        ),
    )
    conn.execute("UPDATE events SET source_event_id=? WHERE event_id=?", (str(meta.get("source_event_id", "")), meta["id"]))
    if operation == "retract" and target_id:
        retract_lineage(conn, target_id, meta["project_id"], meta["recorded_at"])
        return
    if meta["kind"] != "fact":
        _index_search_record(conn, "event", meta["id"], meta["project_id"], meta["title"], " ".join(searchable_tags(tags)), event.body)
        if meta.get("entities"):
            add_entities(conn, meta["project_id"], meta["id"], meta["entities"])
        return

    fact_id = str(meta["fact_id"])
    supersedes = str(meta.get("supersedes") or "")
    valid_from = str(meta.get("valid_from") or meta["observed_at"])
    if supersedes:
        conn.execute(
            "UPDATE facts SET status='superseded', valid_to=? WHERE fact_id=? AND project_id=?",
            (valid_from, supersedes, meta["project_id"]),
        )
    conn.execute(
        "INSERT OR REPLACE INTO facts(fact_id,project_id,fact_key,value_json,status,valid_from,valid_to,recorded_at,"
        "retracted_at,supersedes,assertion_event_id,source_event_id,confidence,verified_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fact_id, meta["project_id"], meta["fact_key"], json_compact(meta["value"]), meta["status"],
            valid_from, str(meta.get("valid_to") or ""), meta["recorded_at"],
            str(meta.get("retracted_at") or ""), supersedes, meta["id"], str(meta.get("source_event_id") or meta["id"]),
            float(meta.get("confidence", 0.8)), str(meta.get("verified_at") or ""),
            str(meta.get("expires_at") or ""),
        ),
    )
    if meta["status"] == "active":
        value_text = meta["value"] if isinstance(meta["value"], str) else json_compact(meta["value"])
        if meta.get("conditions") is not None:
            value_text += "\nConditions: " + json_compact(meta["conditions"])
        _index_search_record(conn, "fact", fact_id, meta["project_id"], meta["fact_key"], " ".join(searchable_tags(tags)), value_text)
    if meta.get("entities"):
        add_entities(conn, meta["project_id"], meta["id"], meta["entities"])


def lineage_ids(conn: sqlite3.Connection, target_id: str, project_id: str) -> List[str]:
    """Return source and all derived assertions, never another project."""
    row = conn.execute("SELECT assertion_event_id FROM facts WHERE fact_id=? AND project_id=?", (target_id, project_id)).fetchone()
    pending = [row[0] if row else target_id]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        candidate = conn.execute("SELECT idempotency_key FROM events WHERE event_id=? AND project_id=?", (current, project_id)).fetchone()
        if candidate and candidate[0].startswith(("candidate-summary:", "candidate-fact:")):
            candidate_id = candidate[0].split(":")[1]
            # Until per-span dependencies are represented, a candidate is one
            # conservative privacy unit: its summary may repeat any sibling Fact.
            pending.extend(row[0] for row in conn.execute(
                "SELECT event_id FROM events WHERE project_id=? AND (idempotency_key=? OR idempotency_key LIKE ?)",
                (project_id, "candidate-summary:" + candidate_id, "candidate-fact:" + candidate_id + ":%")))
        pending.extend(row[0] for row in conn.execute(
            "SELECT event_id FROM events WHERE project_id=? AND source_event_id=? "
            "UNION SELECT assertion_event_id FROM facts WHERE project_id=? AND source_event_id=?",
            (project_id, current, project_id, current)))
    return sorted(seen)


def retract_lineage(conn: sqlite3.Connection, target_id: str, project_id: str, recorded_at: str) -> None:
    for event_id in lineage_ids(conn, target_id, project_id):
        conn.execute("UPDATE events SET status='retracted' WHERE event_id=? AND project_id=?", (event_id, project_id))
        _remove_search_record(conn, "event", event_id)
        for row in conn.execute("SELECT fact_id FROM facts WHERE assertion_event_id=?", (event_id,)).fetchall():
            conn.execute("UPDATE facts SET status='retracted', retracted_at=? WHERE fact_id=?", (recorded_at, row[0]))
            _remove_search_record(conn, "fact", row[0])
        conn.execute("DELETE FROM event_entities WHERE event_id=?", (event_id,))
    conn.execute("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM event_entities)")
    conn.execute("DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM event_entities)")


def invalidate_superseded_derivations(conn, fact_id: str, project_id: str, recorded_at: str) -> None:
    row = conn.execute("SELECT assertion_event_id FROM facts WHERE fact_id=? AND project_id=?", (fact_id, project_id)).fetchone()
    if row:
        for event_id in lineage_ids(conn, row[0], project_id):
            if event_id != row[0]:
                retract_lineage(conn, event_id, project_id, recorded_at)


def add_entities(conn: sqlite3.Connection, project_id: str, event_id: str, entities: Sequence[Dict[str, Any]]) -> None:
    import uuid
    for item in entities:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        entity_type = str(item.get("type", "concept")).strip() or "concept"
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE project_id=? AND entity_type=? AND canonical_name=?",
            (project_id, entity_type, name),
        ).fetchone()
        entity_id = row["entity_id"] if row else str(uuid.uuid4())
        conn.execute(
            "INSERT OR IGNORE INTO entities(entity_id,project_id,entity_type,canonical_name,created_at) VALUES(?,?,?,?,?)",
            (entity_id, project_id, entity_type, name, utc_now()),
        )
        aliases = set(str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip())
        aliases.add(name)
        for alias in aliases:
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases(project_id,alias,entity_id) VALUES(?,?,?)",
                (project_id, alias.lower(), entity_id),
            )
        conn.execute(
            "INSERT OR IGNORE INTO event_entities(event_id,entity_id,relation) VALUES(?,?,?)",
            (event_id, entity_id, str(item.get("relation", "mentions"))),
        )


def integrity(conn: sqlite3.Connection) -> Dict[str, Any]:
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {}
    existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for table in ("projects", "project_aliases", "events", "facts", "entities", "candidates", "migrations", "shadow_runs", "usage_runs"):
        counts[table] = conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0] if table in existing else 0
    counts["fts"] = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
    counts["vectors"] = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    return {"integrity": result, "counts": counts}
