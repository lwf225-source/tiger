from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import database
from .evidence import fact_claim, verify_evidence
from .paths import restored_legacy_path
from .context_pack import pack_context
from .database import add_entities, connect, connect_readonly, event_for_idempotency, index_event, integrity, project_for_alias, register_project
from .events import atomic_write, event_relative_path, maintenance_lock, parse_event, parse_event_text, parse_legacy_note, project_lock, serialize_event
from .model import (
    EVENT_KINDS, FACT_STATUSES, PLMError, SCHEMA_VERSION, SCOPES, Event, SearchResult,
    canonical_root, display_slug, json_compact, memory_base, parse_time,
    project_id_for_root, sha256_bytes, stable_event_id, utc_now, v2_root,
)
from .search import _eligible_records, search, startup_context
from .passages import expand_passage, locate_passage
from .security import SecretDetected, assert_no_secrets, assert_safe_path, ensure_private_dir
from .vector import cosine_blob, encode_vector


INDICATOR_FILES = (
    "AGENTS.md", "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
    "pnpm-workspace.yaml", "yarn.lock", "vite.config.js", "vite.config.ts",
)
HIGH_RISK_FACT = re.compile(r"发布|上线|部署|验收|付款|支付|权限|账号|publish|release|deploy|accept|payment|permission", re.I)
STRONG_EVIDENCE = {"user_confirmation", "git_commit", "command_output", "file_hash", "test_result"}
# Opt-in consolidate-time near-duplicate detection (batch 9). Off by default
# (threshold 0.0); the reference threshold mirrors LLLMemoryAgent (AML #9,
# cosine > 0.85). With the zero-dependency lexical vectors 0.85 only catches
# near-verbatim duplicates — that conservatism is deliberate; an embedding
# provider can be plugged in for semantic near-duplicates.
DEDUP_SIMILARITY_THRESHOLD = 0.85
DEDUP_ACTIONS = ("review", "skip", "supersede")
EXTRA_FIELDS = {
    "operation", "target_id", "entities", "fact_id", "fact_key", "value", "status",
    "valid_from", "valid_to", "retracted_at", "supersedes", "source_event_id",
    "confidence", "verified_at", "expires_at", "evidence_kind", "evidence", "conditions",
}
_WRITE_BATCH = ContextVar("plm_write_batch", default=None)


def batch_committed(event: Event) -> bool:
    batch_id = event.metadata.get("batch_id")
    if not batch_id:
        return True
    path = v2_root() / "batches" / (batch_id + ".json")
    assert_safe_path(path, v2_root())
    if not path.is_file():
        return False
    marker = json.loads(path.read_text(encoding="utf-8"))
    return event.event_id in marker.get("event_ids", []) and marker.get("project_id") == event.project_id


@contextmanager
def event_batch(root: Path, batch_id: str):
    """Prevalidate on a private SQL transaction; a durable marker commits source files.

    Readers see either the entire SQL batch or none. After a process crash rebuild
    replays only batches with a complete commit marker. No partial candidate is active.
    """
    with maintenance_lock():
        conn = connect()
        project = ensure_project(conn, find_project_root(root))
        conn.commit()
        with project_lock(project["project_id"]):
            if (v2_root() / "batches" / (batch_id + ".json")).exists():
                conn.close()
                raise PLMError("batch already source-committed; rebuild-index required")
            batch = {"conn": conn, "project": project, "id": batch_id, "events": {}}
            token = _WRITE_BATCH.set(batch)
            committed = False
            written = []
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield batch
                for event in batch["events"].values():
                    atomic_write(event.path, serialize_event(event.metadata, event.body))
                    written.append(event.path)
                marker_path = v2_root() / "batches" / (batch_id + ".json")
                if batch["events"]:
                    atomic_write(marker_path, json_compact({
                        "project_id": project["project_id"], "event_ids": list(batch["events"]),
                        "committed_at": utc_now(), "candidate": batch.get("candidate"),
                    }))
                committed = True
                conn.commit()
            except BaseException:
                conn.rollback()
                if not committed:
                    for path in written:
                        path.unlink(missing_ok=True)
                # A durable commit marker after a SQL failure requires rebuild,
                # never a misleading rejected state.
                if committed:
                    raise PLMError("batch source committed; rebuild-index required")
                raise
            finally:
                _WRITE_BATCH.reset(token)
                conn.close()


def _index_validated_write(conn, event, project, batch=None):
    meta = event.metadata
    if meta["kind"] == "fact":
        claim = fact_claim(meta["fact_key"], meta["value"])
        if HIGH_RISK_FACT.search(claim + meta["title"]):
            checked = verify_evidence(meta.get("evidence_kind", ""), meta.get("evidence", {}), Path(project["canonical_root"]), claim)
            if not checked["high_risk_eligible"]:
                raise PLMError("high-risk fact requires verifiable, claim-bound evidence")
    source_id = meta.get("source_event_id")
    if source_id:
        source = conn.execute("SELECT scope,scope_id,status FROM events WHERE event_id=? AND project_id=?", (source_id, event.project_id)).fetchone()
        if not source or source["status"] != "active":
            raise PLMError("source event is missing, retracted or in another project")
        if (source["scope"], source["scope_id"]) != (meta["scope"], meta["scope_id"]):
            raise PLMError("derived memory must retain source scope")
        source_fact = conn.execute("SELECT status,valid_from,valid_to FROM facts WHERE assertion_event_id=?", (source_id,)).fetchone()
        now = utc_now()
        if source_fact and (source_fact["status"] not in {"active", "superseded"} or source_fact["valid_from"] > now or (source_fact["valid_to"] and source_fact["valid_to"] <= now)):
            raise PLMError("derived memory source fact is not currently valid")
    if meta["kind"] == "fact" and meta.get("status") == "active":
        params = (event.project_id, meta["fact_key"], meta["scope"], meta["scope_id"])
        if meta.get("supersedes"):
            old = conn.execute(
                "SELECT f.fact_key,f.status,f.valid_from,e.scope,e.scope_id FROM facts f JOIN events e ON e.event_id=f.assertion_event_id WHERE f.fact_id=? AND f.project_id=?",
                (meta["supersedes"], event.project_id)).fetchone()
            if not old or old["fact_key"] != meta["fact_key"] or old["status"] != "active" or (old["scope"], old["scope_id"]) != params[2:]:
                raise PLMError("superseded fact must be active with the same key and scope")
            if meta["valid_from"] < old["valid_from"]:
                raise PLMError("supersession cannot precede the previous valid_from")
        elif conn.execute(
            "SELECT 1 FROM facts f JOIN events e ON e.event_id=f.assertion_event_id WHERE f.project_id=? AND f.fact_key=? AND e.scope=? AND e.scope_id=? AND f.status='active' LIMIT 1", params).fetchone():
            raise PLMError("active fact already exists in this scope; supersede it explicitly")
    index_event(conn, event)
    if meta.get("supersedes") and meta.get("valid_from", "") <= utc_now():
        database.invalidate_superseded_derivations(conn, meta["supersedes"], event.project_id, meta["recorded_at"])
    if batch is not None:
        batch["events"][event.event_id] = event


def find_project_root(cwd: Path) -> Path:
    cwd = canonical_root(cwd)
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if result.stdout.strip():
            return canonical_root(Path(result.stdout.strip()))
    except Exception:
        pass
    home = Path.home().resolve()
    for path in [cwd, *cwd.parents]:
        if path == home.parent:
            break
        if any((path / marker).exists() for marker in INDICATOR_FILES):
            return path
    return cwd


def config_path() -> Path:
    return v2_root() / "config.json"


def read_config() -> Dict[str, Any]:
    defaults = {
        "schema_version": 2,
        "engine": "v1",
        "write_mode": "v1",
        "auto_consolidate": True,
        "token_budget": 1800,
        "installed_version": "",
        "cutover_at": "",
        "skill_backup": "",
        "last_install_backup": "",
    }
    path = config_path()
    if not path.exists():
        return defaults
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PLMError("invalid v2 config: %s" % exc) from exc
    defaults.update(loaded)
    if defaults["engine"] not in {"v1", "shadow", "v2"}:
        raise PLMError("invalid engine in config")
    return defaults


def write_config(changes: Dict[str, Any]) -> Dict[str, Any]:
    config = read_config()
    config.update(changes)
    if config["engine"] not in {"v1", "shadow", "v2"}:
        raise PLMError("invalid engine")
    atomic_write(config_path(), json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return config


def ensure_project(conn: sqlite3.Connection, root: Path) -> Dict[str, str]:
    root = canonical_root(root)
    existing = project_for_alias(conn, str(root))
    if existing:
        return dict(existing)
    project_id = project_id_for_root(root)
    folder = display_slug(root)
    register_project(conn, project_id, str(root), root.name or "root", folder)
    manifest_path = v2_root() / "projects" / folder / "manifest.json"
    if not manifest_path.exists():
        atomic_write(
            manifest_path,
            json.dumps(
                {
                    "schema_version": 2,
                    "project_id": project_id,
                    "canonical_root": str(root),
                    "aliases": [str(root)],
                    "display_name": root.name or "root",
                    "created_at": utc_now(),
                },
                ensure_ascii=False, indent=2, sort_keys=True,
            ) + "\n",
        )
    return {
        "project_id": project_id,
        "canonical_root": str(root),
        "display_name": root.name or "root",
        "folder_name": folder,
    }


def _tags(tags: Any) -> List[str]:
    if isinstance(tags, str):
        values = tags.split(",")
    elif isinstance(tags, list):
        values = tags
    else:
        raise PLMError("tags must be comma-separated text or a list")
    result: List[str] = []
    seen = set()
    for value in values:
        tag = str(value).strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            result.append(tag)
    return result


def write_event(
    cwd: Path,
    title: str,
    content: str,
    tags: Any = None,
    kind: str = "episode",
    scope: str = "project",
    scope_id: str = "",
    observed_at: Optional[str] = None,
    source: str = "agent",
    visibility: str = "local",
    created_by: str = "codex",
    idempotency_key: str = "",
    event_id: str = "",
    recorded_at: Optional[str] = None,
    source_hash: str = "",
    source_legacy_path: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Event:
    if kind not in EVENT_KINDS:
        raise PLMError("invalid kind")
    title = title.strip()
    content = content.strip()
    if not title or not content:
        raise PLMError("title and content are required")
    tag_list = _tags(tags or [])
    assert_no_secrets([title, content, " ".join(tag_list), source])
    root = find_project_root(cwd)
    batch = _WRITE_BATCH.get()
    conn = batch["conn"] if batch else connect()
    project = batch["project"] if batch else ensure_project(conn, root)
    if batch and str(root) != project["canonical_root"]:
        raise PLMError("batch cannot cross project boundary")
    if not batch:
        conn.commit()
    if idempotency_key:
        existing = event_for_idempotency(conn, project["project_id"], idempotency_key)
        if existing:
            event = batch["events"].get(existing["event_id"]) if batch else None
            event = event or parse_event(Path(existing["source_path"]))
            if not batch:
                conn.close()
            return event
    if not batch:
        conn.close()

    recorded = parse_time(recorded_at)
    observed = parse_time(observed_at or recorded)
    event_id = event_id or str(uuid.uuid4())
    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": event_id,
        "project_id": project["project_id"],
        "project_root": project["canonical_root"],
        "scope": scope,
        "scope_id": scope_id or (project["project_id"] if scope == "project" else scope),
        "kind": kind,
        "title": title,
        "tags": tag_list,
        "observed_at": observed,
        "recorded_at": recorded,
        "source": source,
        "source_hash": source_hash or sha256_bytes(content.encode("utf-8")),
        "visibility": visibility,
        "created_by": created_by,
        "idempotency_key": idempotency_key,
    }
    if source_legacy_path:
        meta["source_legacy_path"] = source_legacy_path
    if batch:
        meta["batch_id"] = batch["id"]
    if extra:
        unknown_extra = set(extra) - EXTRA_FIELDS
        if unknown_extra:
            raise PLMError("protected or unknown event fields: %s" % ", ".join(sorted(unknown_extra)))
        meta.update(extra)
    assert_no_secrets([content, json_compact(meta)])
    relative = event_relative_path(project["folder_name"], event_id, recorded)
    path = v2_root() / relative
    event = Event(meta, content, path)

    serialized = serialize_event(meta, content)
    if batch:
        _index_validated_write(batch["conn"], event, project, batch)
        return event
    with maintenance_lock():
        with project_lock(project["project_id"]):
            conn = connect()
            if idempotency_key:
                existing = event_for_idempotency(conn, project["project_id"], idempotency_key)
                if existing:
                    conn.close()
                    return parse_event(Path(existing["source_path"]))
            existing_id = conn.execute("SELECT source_path FROM events WHERE event_id=?", (event_id,)).fetchone()
            if existing_id:
                existing_event = parse_event(Path(existing_id["source_path"]))
                conn.close()
                if existing_event.metadata == meta and existing_event.body == content:
                    return existing_event
                raise PLMError("event id already exists with different content")
            if path.exists():
                existing_event = parse_event(path)
                if existing_event.metadata != meta or existing_event.body != content:
                    conn.close()
                    raise PLMError("event path already exists with different content")
            duplicate_paths = [
                candidate for candidate in (v2_root() / "projects" / project["folder_name"] / "events").glob("*/*/*_" + event_id + ".md")
                if candidate != path
            ]
            if duplicate_paths:
                conn.close()
                raise PLMError("event id already exists at another source path")
            try:
                conn.execute("BEGIN IMMEDIATE")
                _index_validated_write(conn, event, project)
                if not path.exists():
                    atomic_write(path, serialized)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    if source != "legacy_import":
        try:
            record_usage(project["project_id"], "write", created_by, "", 1, True)
        except Exception:
            pass
    return event


def write_fact(
    cwd: Path,
    fact_key: str,
    value: Any,
    title: str = "",
    tags: Any = None,
    supersedes: str = "",
    status: str = "active",
    valid_from: Optional[str] = None,
    valid_to: str = "",
    confidence: float = 0.8,
    verified_at: str = "",
    expires_at: str = "",
    source_event_id: str = "",
    source: str = "agent",
    idempotency_key: str = "",
    entities: Optional[Sequence[Dict[str, Any]]] = None,
    scope: str = "project",
    scope_id: str = "",
    observed_at: Optional[str] = None,
    evidence_kind: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    conditions: Any = None,
) -> Event:
    if status not in FACT_STATUSES:
        raise PLMError("invalid fact status")
    if not fact_key.strip():
        raise PLMError("fact_key is required")
    fact_id = str(uuid.uuid4())
    value_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    high_risk_text = " ".join((fact_key, title, value_text))
    if HIGH_RISK_FACT.search(high_risk_text):
        checked = verify_evidence(evidence_kind, evidence or {}, find_project_root(cwd), fact_claim(fact_key.strip(), value))
        if not checked["high_risk_eligible"]:
            raise PLMError("high-risk fact requires verifiable, claim-bound evidence")
    extra = {
        "fact_id": fact_id,
        "fact_key": fact_key.strip(),
        "value": value,
        "status": status,
        "valid_from": parse_time(valid_from),
        "valid_to": parse_time(valid_to) if valid_to else "",
        "retracted_at": "",
        "supersedes": supersedes,
        "source_event_id": source_event_id,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "verified_at": parse_time(verified_at) if verified_at else "",
        "expires_at": parse_time(expires_at) if expires_at else "",
        "evidence_kind": evidence_kind,
        "evidence": evidence or {},
    }
    if entities:
        extra["entities"] = list(entities)
    if conditions is not None:
        extra["conditions"] = conditions
        value_text += "\nConditions: " + json.dumps(conditions, ensure_ascii=False, sort_keys=True)
    if extra["valid_to"] and extra["valid_to"] <= extra["valid_from"]:
        raise PLMError("valid_to must be after valid_from")
    return write_event(
        cwd, title or fact_key, value_text, tags=tags, kind="fact", scope=scope,
        scope_id=scope_id, observed_at=observed_at, source=source,
        idempotency_key=idempotency_key, extra=extra,
    )


def collect_agent_files(cwd: Path) -> List[Path]:
    cwd = canonical_root(cwd)
    home = Path.home().resolve()
    files = []
    for path in reversed([cwd, *cwd.parents]):
        if home not in [path, *path.parents]:
            continue
        candidate = path / "AGENTS.md"
        if candidate.is_file():
            files.append(candidate)
    return files


def context_data(
    cwd: Path, query: str, limit: int = 6, view: str = "current",
    scope: str = "project", scope_id: str = "", retrieval_profile: str = "lexical",
    adjacent: int = 0, recency_boost: bool = False,
    embedding_provider: Any = None, reranker: Any = None,
    expand_entities: int = 0, mmr: bool = False,
    abstain_threshold: float = 0.0, diagnostics: Optional[Dict[str, Any]] = None,
    valid_at: str = "", known_at: str = "", soft_supersede: bool = False,
    temporal_intent: bool = False,
    window_reserve: bool = False, coverage_rerank: bool = False,
    session_views: bool = False,
    session_rrf: bool = False,
    query_provider: Any = None,
) -> Tuple[Dict[str, str], List[SearchResult]]:
    if retrieval_profile not in {"lexical", "passages"}:
        raise PLMError("invalid retrieval profile")
    root = find_project_root(cwd)
    if scope in {"agent", "session"} and not scope_id:
        raise PLMError("scope-id is required for agent and session scope")
    project = {
        "project_id": project_id_for_root(root),
        "canonical_root": str(root),
        "display_name": root.name or "root",
        "folder_name": display_slug(root),
    }
    if not database.db_path().is_file():
        return project, []
    conn = connect_readonly()
    try:
        existing = project_for_alias(conn, str(root))
        if existing:
            project = dict(existing)
        else:
            existing = conn.execute("SELECT * FROM projects WHERE project_id=?", (project["project_id"],)).fetchone()
            if not existing:
                return project, []
            project = dict(existing)
        if query.strip():
            results = search(conn, project["project_id"], query, limit, view, scope, scope_id,
                             profile=retrieval_profile, adjacent=adjacent, recency_boost=recency_boost,
                             embedding_provider=embedding_provider, reranker=reranker,
                             expand_entities=expand_entities, mmr=mmr,
                             abstain_threshold=abstain_threshold, diagnostics=diagnostics,
                             valid_at=valid_at, known_at=known_at, soft_supersede=soft_supersede,
                             temporal_intent=temporal_intent,
                             window_reserve=window_reserve, coverage_rerank=coverage_rerank,
                             session_views=session_views,
                             session_rrf=session_rrf,
                             query_provider=query_provider)
        else:
            results = startup_context(conn, project["project_id"], limit, view, scope, scope_id,
                                      valid_at=valid_at, known_at=known_at, soft_supersede=soft_supersede)
        return project, results
    finally:
        conn.close()


def source_event_data(cwd: Path, event_id: str, scope: str, scope_id: str) -> Optional[Dict[str, str]]:
    """Return one active original event for a retrieval projection.

    It uses the indexed identity rather than a caller-provided path and never
    returns a fact assertion event, so AML can return source messages only.
    """
    if not event_id or not database.db_path().is_file():
        return None
    root = find_project_root(cwd)
    conn = connect_readonly()
    try:
        project = project_for_alias(conn, str(root))
        if not project:
            return None
        row = conn.execute(
            "SELECT event_id,body,observed_at FROM events WHERE project_id=? AND event_id=? "
            "AND scope=? AND scope_id=? AND kind<>'fact' AND status='active' AND operation=''",
            (project["project_id"], event_id, scope, scope_id),
        ).fetchone()
        if not row:
            return None
        return {"event_id": row["event_id"], "body": row["body"], "observed_at": row["observed_at"]}
    finally:
        conn.close()


def render_context(
    cwd: Path,
    query: str,
    limit: int = 6,
    view: str = "current",
    token_budget: int = 1800,
    explain: bool = False,
    scope: str = "project",
    scope_id: str = "",
    retrieval_profile: str = "lexical",
    adjacent: int = 0,
    recency_boost: bool = False,
    excerpt_fallback: bool = False,
    expand_entities: int = 0,
    mmr: bool = False,
    abstain_threshold: float = 0.0,
    valid_at: str = "",
    known_at: str = "",
    soft_supersede: bool = False,
    temporal_intent: bool = False,
    window_reserve: bool = False,
    coverage_rerank: bool = False,
) -> str:
    started = time.perf_counter()
    project, results = context_data(cwd, query, limit, view, scope, scope_id, retrieval_profile,
                                    adjacent=adjacent, recency_boost=recency_boost,
                                    expand_entities=expand_entities, mmr=mmr,
                                    abstain_threshold=abstain_threshold,
                                    valid_at=valid_at, known_at=known_at, soft_supersede=soft_supersede,
                                    temporal_intent=temporal_intent,
                                    window_reserve=window_reserve, coverage_rerank=coverage_rerank)
    rule_files = collect_agent_files(cwd)
    rendered = pack_context(dict(project, view=view, scope=scope), cwd, query, results,
        [(path, path.read_text(encoding="utf-8", errors="replace")) for path in rule_files],
        token_budget=token_budget, explain=explain, excerpt_fallback=excerpt_fallback)
    try:
        record_usage(
            project["project_id"], "context", "plm", query, len(results), True,
            (time.perf_counter() - started) * 1000.0,
        )
    except Exception:
        pass
    return rendered


def _read_evidence_event(path: Path) -> Event:
    """Read a private regular v2 Event through non-following file descriptors."""
    root = v2_root()
    if not path.is_absolute() or ".." in path.parts:
        raise PLMError("unsafe event path")
    assert_safe_path(path, root)
    parts = path.relative_to(root).parts
    if not parts:
        raise PLMError("invalid event path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(str(root), directory_flags)
    try:
        for part in parts[:-1]:
            following = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = following
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = os.fstat(descriptor)
            maximum = 32 * 1024 * 1024
            if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o077
                    or before.st_uid != os.getuid() or before.st_size > maximum):
                raise PLMError("unsafe event file")
            chunks = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            if (len(raw) > maximum or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise PLMError("event changed during read")
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
    event = parse_event_text(raw.decode("utf-8"))
    event.path = path
    return event


def evidence_data(
    cwd: Path, event_id: str, body_sha256: str, start: int, end: int,
    scope: str = "project", scope_id: str = "", before_chars: int = 120,
    after_chars: int = 120, max_chars: int = 1800,
) -> Dict[str, Any]:
    """Expand an eligible current source span without writing usage or caches.

    Identity, lifecycle and scope are checked before reading a source file.
    Coordinates refer to the canonical Event body, not frontmatter/file bytes.
    Failed requests deliberately expose neither source text nor filesystem paths.
    """
    conn = None
    try:
        if scope not in SCOPES or (scope in {"agent", "session"} and not scope_id):
            raise PLMError("invalid scope")
        if not isinstance(event_id, str) or str(uuid.UUID(event_id)) != event_id:
            raise PLMError("invalid event identity")
        if not isinstance(body_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", body_sha256):
            raise PLMError("invalid body hash")
        root = find_project_root(cwd)
        conn = connect_readonly()
        conn.execute("BEGIN")
        project = project_for_alias(conn, str(root))
        if not project:
            project = conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id_for_root(root),)).fetchone()
        if not project:
            raise PLMError("project unavailable")
        project_id = project["project_id"]
        scope_id = scope_id or (project_id if scope == "project" else scope)
        eligible = _eligible_records(conn, project_id, scope, scope_id, "current", utc_now())
        matched = next((record for record in eligible.values() if record["assertion_event_id"] == event_id), None)
        if matched is None:
            raise PLMError("event unavailable")
        # Display paths can point at v1 migration inputs; evidence never reads
        # those. Fetch the actual immutable v2 source path by the eligible ID.
        row = conn.execute("SELECT * FROM events WHERE event_id=? AND project_id=? AND scope=? AND scope_id=?",
                           (event_id, project_id, scope, scope_id)).fetchone()
        if not row:
            raise PLMError("event unavailable")
        path = Path(row["source_path"])
        event = _read_evidence_event(path)
        metadata = event.metadata
        if (event.event_id != event_id or event.project_id != project_id
                or metadata["scope"] != scope or metadata["scope_id"] != scope_id
                or metadata["kind"] != row["kind"] or metadata.get("operation", "")
                or any(metadata[field] != row[field] for field in ("recorded_at", "observed_at", "source_hash", "title"))
                or metadata.get("source_event_id", "") != (row["source_event_id"] if "source_event_id" in row.keys() else "")
                or event.body != matched["body"]):
            raise PLMError("event provenance mismatch")
        expected_path = v2_root() / event_relative_path(project["folder_name"], event_id, metadata["recorded_at"])
        if path != expected_path or sha256_bytes(event.body.encode("utf-8")) != body_sha256.lower():
            raise PLMError("event provenance mismatch")
        assert_no_secrets([event.body, json_compact(metadata)])
        located = locate_passage(event.body, event_id, start, end)
        return expand_passage(event.body, located, before_chars, after_chars, max_chars)
    except (PLMError, ValueError, TypeError, KeyError, OSError, sqlite3.Error):
        raise PLMError("evidence unavailable: request or provenance validation failed") from None
    finally:
        if conn is not None:
            conn.close()


def legacy_note_paths() -> List[Path]:
    root = memory_base() / "projects"
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/notes/*.md") if path.is_file() and not path.is_symlink())


def migration_plan() -> Dict[str, Any]:
    notes = legacy_note_paths()
    roots: Counter[str] = Counter()
    buckets: Dict[str, Counter[str]] = defaultdict(Counter)
    entries = []
    errors = []
    for path in notes:
        try:
            meta, _ = parse_legacy_note(path)
            root = str(canonical_root(Path(meta["project_root"])))
            raw_hash = sha256_bytes(path.read_bytes())
            roots[root] += 1
            buckets[path.parent.parent.name][root] += 1
            entries.append({"source_path": str(path), "source_hash": raw_hash, "project_root": root})
        except Exception as exc:
            errors.append({"source_path": str(path), "error": str(exc)})
    collisions = {
        bucket: dict(counts) for bucket, counts in buckets.items() if len(counts) > 1
    }
    return {
        "generated_at": utc_now(),
        "source_count": len(notes),
        "valid_count": len(entries),
        "error_count": len(errors),
        "project_count": len(roots),
        "projects": dict(sorted(roots.items())),
        "collision_buckets": collisions,
        "entries": entries,
        "errors": errors,
    }


def migrate_legacy(dry_run: bool = False) -> Dict[str, Any]:
    plan = migration_plan()
    if dry_run:
        return plan
    ensure_private_dir(v2_root() / "migration")
    manifest_path = v2_root() / "migration" / "source-manifest.json"
    atomic_write(manifest_path, json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    migrated = 0
    skipped = 0
    failures = []
    for item in plan["entries"]:
        path = Path(item["source_path"])
        try:
            result = migrate_legacy_path(path)
            if result["status"] == "skipped":
                skipped += 1
            else:
                migrated += 1
        except Exception as exc:
            failures.append({"source_path": str(path), "error": str(exc)})
    plan.update({"migrated": migrated, "skipped": skipped, "failures": failures, "failure_count": len(failures)})
    report_path = v2_root() / "migration" / "last-report.json"
    atomic_write(report_path, json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return plan


def migrate_legacy_path(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise PLMError("legacy note path is not a regular file")
    source_hash = sha256_bytes(path.read_bytes())
    conn = connect()
    existing = conn.execute("SELECT event_id,source_hash FROM migrations WHERE source_path=?", (str(path),)).fetchone()
    conn.close()
    if existing and existing["source_hash"] == source_hash:
        return {"status": "skipped", "event_id": existing["event_id"], "source_path": str(path)}
    legacy, body = parse_legacy_note(path)
    event_id = stable_event_id(str(path), source_hash)
    event = write_event(
        Path(legacy["project_root"]), legacy["title"], body,
        tags=legacy.get("tags", ""), kind="episode", observed_at=legacy["time"],
        recorded_at=legacy["time"], source="legacy_import", created_by="migration",
        idempotency_key="legacy:" + source_hash, event_id=event_id,
        source_hash=source_hash, source_legacy_path=str(path),
    )
    conn = connect()
    conn.execute(
        "INSERT OR REPLACE INTO migrations(source_path,source_hash,event_id,migrated_at) VALUES(?,?,?,?)",
        (str(path), source_hash, event.event_id, utc_now()),
    )
    conn.commit()
    conn.close()
    return {"status": "migrated", "event_id": event.event_id, "source_path": str(path)}


def rebuild_index() -> Dict[str, Any]:
    path = database.db_path()
    backup = ""
    ensure_private_dir(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=".catalog-rebuild-", suffix=".sqlite3", dir=str(path.parent))
    os.close(fd)
    os.unlink(temp_name)
    temp_path = Path(temp_name)
    errors: List[Dict[str, str]] = []
    indexed = 0
    try:
        with maintenance_lock(exclusive=True):
            usage_rows: List[Dict[str, Any]] = []
            if path.is_file():
                previous = connect(path)
                usage_rows = [dict(row) for row in previous.execute(
                    "SELECT created_at,project_id,operation,client,query_hash,result_count,success,latency_ms FROM usage_runs ORDER BY run_id"
                )]
                previous.close()
            conn = connect(temp_path)
            try:
                for manifest_path in sorted((v2_root() / "projects").glob("*/manifest.json")):
                    try:
                        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        register_project(
                            conn, manifest_data["project_id"], manifest_data["canonical_root"],
                            manifest_data.get("display_name") or Path(manifest_data["canonical_root"]).name,
                            manifest_path.parent.name,
                        )
                        for alias in manifest_data.get("aliases", []):
                            conn.execute(
                                "INSERT OR REPLACE INTO project_aliases(alias_path,project_id,active,created_at) VALUES(?,?,1,?)",
                                (str(canonical_root(Path(alias))), manifest_data["project_id"], utc_now()),
                            )
                    except Exception as exc:
                        errors.append({"path": str(manifest_path), "error": str(exc)})

                parsed: List[Event] = []
                seen_ids: Dict[str, str] = {}
                disk_batches = set()
                for event_path in sorted((v2_root() / "projects").glob("*/events/*/*/*.md")):
                    try:
                        event = parse_event(event_path)
                        if event.metadata.get("batch_id"):
                            disk_batches.add(event.metadata["batch_id"])
                        if not batch_committed(event):
                            continue
                        previous = seen_ids.get(event.event_id)
                        if previous:
                            raise PLMError("duplicate event id also present at %s" % previous)
                        seen_ids[event.event_id] = str(event_path)
                        if not conn.execute("SELECT 1 FROM projects WHERE project_id=?", (event.project_id,)).fetchone():
                            raise PLMError("event project manifest is missing")
                        parsed.append(event)
                    except Exception as exc:
                        errors.append({"path": str(event_path), "error": str(exc)})

                assertions = sorted(
                    (event for event in parsed if not event.metadata.get("operation")),
                    key=lambda event: (event.metadata["recorded_at"], event.event_id),
                )
                operations = sorted(
                    (event for event in parsed if event.metadata.get("operation")),
                    key=lambda event: (event.metadata["recorded_at"], event.event_id),
                )
                from .privacy import deleted_event_ids
                deleted_ids = deleted_event_ids()
                for marker_path in (v2_root() / "batches").glob("*.json"):
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    missing = set(marker.get("event_ids", [])) - set(seen_ids) - deleted_ids
                    if missing:
                        errors.append({"path": str(marker_path), "error": "committed batch source files missing"})
                for event in [*assertions, *operations]:
                    try:
                        index_event(conn, event)
                        indexed += 1
                    except Exception as exc:
                        errors.append({"path": str(event.path), "error": str(exc)})

                for event in assertions:
                    meta = event.metadata
                    if meta.get("supersedes") and meta.get("valid_from", "") <= utc_now():
                        database.invalidate_superseded_derivations(conn, meta["supersedes"], event.project_id, meta["recorded_at"])

                manifest = v2_root() / "migration" / "source-manifest.json"
                if manifest.exists():
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    for item in data.get("entries", []):
                        event_id = stable_event_id(item["source_path"], item["source_hash"])
                        conn.execute(
                            "INSERT OR IGNORE INTO migrations(source_path,source_hash,event_id,migrated_at) VALUES(?,?,?,?)",
                            (restored_legacy_path(item["source_path"]), item["source_hash"], event_id, utc_now()),
                        )
                for candidate_path in sorted((v2_root() / "candidates").glob("*.json")):
                    try:
                        item = json.loads(candidate_path.read_text(encoding="utf-8"))
                        marker = v2_root() / "batches" / (item["candidate_id"] + ".json")
                        if item["state"] == "active" and (item.get("batch_id") or item["candidate_id"] in disk_batches) and not marker.is_file():
                            raise PLMError("active candidate commit marker missing")
                        if marker.is_file():
                            committed_candidate = json.loads(marker.read_text(encoding="utf-8")).get("candidate")
                            if committed_candidate:
                                item.update(committed_candidate)
                        conn.execute(
                            "INSERT OR REPLACE INTO candidates(candidate_id,project_id,payload_json,state,reason,idempotency_key,created_at,updated_at) "
                            "VALUES(?,?,?,?,?,?,?,?)",
                            (
                                item["candidate_id"], item["project_id"], item["payload_json"], item["state"],
                                item.get("reason", ""), item["idempotency_key"], item["created_at"], item["updated_at"],
                            ),
                        )
                    except Exception as exc:
                        errors.append({"path": str(candidate_path), "error": str(exc)})
                for usage in usage_rows:
                    conn.execute(
                        "INSERT INTO usage_runs(created_at,project_id,operation,client,query_hash,result_count,success,latency_ms) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        tuple(usage[field] for field in (
                            "created_at", "project_id", "operation", "client", "query_hash",
                            "result_count", "success", "latency_ms",
                        )),
                    )
                from .privacy import ledger_path, read_ledger, apply_deletions
                if ledger_path().is_file():
                    apply_deletions(conn, read_ledger(ledger_path()))
                conn.commit()
                report = integrity(conn)
                if errors or report["integrity"] != "ok":
                    report.update({"indexed": indexed, "errors": errors, "backup": "", "swapped": False})
                    return report
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA journal_mode=DELETE")
            finally:
                conn.close()

            if path.exists():
                current = connect(path)
                current.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                current.close()
                stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                backup_path = path.with_name(path.name + ".rebuild-backup-" + stamp)
                if not ledger_path().exists():
                    shutil.copy2(str(path), str(backup_path))
                    os.chmod(str(backup_path), 0o600)
                    backup = str(backup_path)
            os.replace(str(temp_path), str(path))
            os.chmod(str(path), 0o600)
            for suffix in ("-wal", "-shm"):
                aux = Path(str(path) + suffix)
                if aux.exists():
                    aux.unlink()
            report.update({"indexed": indexed, "errors": [], "backup": backup, "swapped": True})
            return report
    finally:
        for candidate in (temp_path, Path(str(temp_path) + "-wal"), Path(str(temp_path) + "-shm")):
            if candidate.exists():
                candidate.unlink()


def add_project_alias(cwd: Path, alias_path: Path) -> Dict[str, Any]:
    with maintenance_lock():
        conn = connect()
        project = ensure_project(conn, find_project_root(cwd))
        alias = str(canonical_root(alias_path))
        existing = project_for_alias(conn, alias)
        if existing and existing["project_id"] != project["project_id"]:
            conn.close()
            raise PLMError("alias is already assigned to another project")
        manifest_path = v2_root() / "projects" / project["folder_name"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        aliases = list(dict.fromkeys([*manifest.get("aliases", []), alias]))
        manifest["aliases"] = aliases
        atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        conn.execute(
            "INSERT OR REPLACE INTO project_aliases(alias_path,project_id,active,created_at) VALUES(?,?,1,?)",
            (alias, project["project_id"], utc_now()),
        )
        conn.commit()
        conn.close()
    return {"project_id": project["project_id"], "canonical_root": project["canonical_root"], "alias": alias}


def submit_candidate(cwd: Path, payload: Dict[str, Any], idempotency_key: str = "") -> Dict[str, Any]:
    serialized = json_compact(payload)
    assert_no_secrets([serialized])
    with maintenance_lock():
        conn = connect()
        project = ensure_project(conn, find_project_root(cwd))
        conn.commit()
        conn.close()
        with project_lock(project["project_id"]):
            return _submit_candidate_locked(project, serialized, idempotency_key)


def _submit_candidate_locked(project, serialized, idempotency_key):
    conn = connect()
    try:
        key = idempotency_key or hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        candidate_id = str(uuid.uuid4())
        now = utc_now()
        existing = conn.execute(
            "SELECT * FROM candidates WHERE project_id=? AND idempotency_key=?", (project["project_id"], key),
        ).fetchone()
        if existing:
            return dict(existing)
        record = {
            "candidate_id": candidate_id, "project_id": project["project_id"], "payload_json": serialized,
            "state": "pending", "reason": "", "idempotency_key": key, "created_at": now, "updated_at": now,
        }
        candidate_dir = v2_root() / "candidates"
        ensure_private_dir(candidate_dir)
        atomic_write(candidate_dir / (candidate_id + ".json"), json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        conn.execute(
            "INSERT INTO candidates(candidate_id,project_id,payload_json,state,reason,idempotency_key,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            tuple(record[field] for field in ("candidate_id", "project_id", "payload_json", "state", "reason", "idempotency_key", "created_at", "updated_at")),
        )
        conn.commit()
        return record
    finally:
        conn.close()


def _candidate_decision(payload: Dict[str, Any], root: Path) -> Tuple[str, str]:
    evidence_kind = str(payload.get("evidence_kind", ""))
    evidence = payload.get("evidence", {})
    source_event_id = str(payload.get("source_event_id", ""))
    facts = payload.get("facts", [])
    if not isinstance(facts, list) or any(not isinstance(fact, dict) or not fact.get("fact_key") for fact in facts):
        return "rejected", "invalid fact list"
    for fact in facts:
        claim = fact_claim(str(fact["fact_key"]), fact.get("value", ""))
        if HIGH_RISK_FACT.search(claim + str(fact.get("title", ""))):
            checked = verify_evidence(str(fact.get("evidence_kind", evidence_kind)), fact.get("evidence", evidence), root, claim)
            if not checked["high_risk_eligible"]:
                return "needs_confirmation", "high-risk fact requires claim-bound evidence"
    if not payload.get("summary") and not facts:
        return "rejected", "candidate has no summary or facts"
    if payload.get("kind", "episode") in {"procedure", "artifact_ref"} and not verify_evidence(evidence_kind, evidence, root)["valid"]:
        return "needs_confirmation", "procedure or artifact candidate requires verifiable evidence"
    return "validated", "deterministic policy passed"


def _dedup_probe_text(fact_key: str, value: Any, conditions: Any = None) -> str:
    """Comparison text for near-duplicate detection: key + value (+ conditions).

    Deliberately not the persisted search-index recipe (which mixes in tags and
    is frozen with retrieval); dedup compares claim content only, encoded fresh
    on both sides so the comparison is symmetric.
    """
    value_text = value if isinstance(value, str) else json_compact(value)
    text = fact_key.strip() + "\n" + value_text
    if conditions is not None:
        text += "\nConditions: " + json_compact(conditions)
    return text


def _cosine_dense(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(y * y for y in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def find_near_duplicate_fact(
    conn: sqlite3.Connection,
    project_id: str,
    scope: str,
    scope_id: str,
    fact_key: str,
    value: Any,
    threshold: float,
    conditions: Any = None,
    embedding_provider: Any = None,
) -> Optional[Dict[str, Any]]:
    """Best-scoring active Fact near duplicate in the same project+scope, or None.

    Only active Facts of the same project and exact scope are candidates, so a
    different scope never cross-fires. The lexical path is zero-dependency
    (512-dim character n-gram cosine); an embedding provider, when present,
    embeds both sides on the fly with the same cosine rule.
    """
    rows = conn.execute(
        "SELECT f.fact_id,f.fact_key,f.value_json FROM facts f "
        "JOIN events e ON e.event_id=f.assertion_event_id "
        "WHERE f.project_id=? AND f.status='active' AND e.scope=? AND e.scope_id=?",
        (project_id, scope, scope_id),
    ).fetchall()
    if not rows:
        return None
    probe_text = _dedup_probe_text(fact_key, value, conditions)
    texts = [probe_text] + [
        _dedup_probe_text(row["fact_key"], json.loads(row["value_json"])) for row in rows
    ]
    if embedding_provider is not None:
        embeddings = embedding_provider.embed(texts)
        scores = [_cosine_dense(embeddings[0], other) for other in embeddings[1:]]
    else:
        probe = encode_vector(probe_text)
        scores = [cosine_blob(probe, encode_vector(text)) for text in texts[1:]]
    best = max(range(len(rows)), key=lambda index: scores[index])
    if scores[best] < threshold:
        return None
    return {
        "fact_id": rows[best]["fact_id"],
        "fact_key": rows[best]["fact_key"],
        "similarity": round(scores[best], 6),
    }


def _dedup_precheck(
    payload: Dict[str, Any],
    project_id: str,
    threshold: float,
    action: str,
    embedding_provider: Any,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Scan a validated candidate's facts for near duplicates before any write.

    Returns (state_override, reason, matches). ``state_override`` is
    "needs_confirmation" when the conservative path must decide; "" means the
    caller proceeds to materialize (with ``matches`` driving skip/supersede).
    Facts carrying an explicit ``supersedes`` are left to the existing
    supersede validation and never scanned.
    """
    base_scope = str(payload.get("scope", "project"))
    base_scope_id = str(payload.get("scope_id", ""))
    matches: List[Dict[str, Any]] = []
    conn = connect()
    try:
        for position, fact in enumerate(payload.get("facts", [])):
            if fact.get("supersedes"):
                continue
            scope = str(fact.get("scope", base_scope))
            scope_id = str(fact.get("scope_id", base_scope_id)) or (project_id if scope == "project" else scope)
            match = find_near_duplicate_fact(
                conn, project_id, scope, scope_id, str(fact["fact_key"]), fact.get("value", ""),
                threshold, conditions=fact.get("conditions", payload.get("conditions")),
                embedding_provider=embedding_provider,
            )
            if match:
                matches.append(dict(match, position=position, candidate_fact_key=str(fact["fact_key"])))
    finally:
        conn.close()
    if not matches:
        return "", "", []
    detail = "; ".join(
        "fact[%d] %r ≈ active fact %s (%r), similarity %.4f" % (
            match["position"], match["candidate_fact_key"], match["fact_id"],
            match["fact_key"], match["similarity"],
        )
        for match in matches
    )
    if action == "skip":
        return "", "", matches
    if action == "supersede" and all(
        match["fact_key"] == match["candidate_fact_key"] for match in matches
    ):
        return "", "", matches
    if action == "supersede":
        return "needs_confirmation", (
            "near-duplicate with a different fact_key cannot auto-supersede "
            "(supersede requires the same key); human review required — " + detail
        ), matches
    return "needs_confirmation", "suspected near-duplicate of existing memory; human review required — " + detail, matches


def consolidate(
    limit: int = 20, candidate_id: str = "",
    dedup_similarity: float = 0.0, dedup_action: str = "review",
    embedding_provider: Any = None,
) -> Dict[str, Any]:
    """Materialize pending candidates; opt-in near-duplicate detection.

    ``dedup_similarity`` 0.0 (default) keeps the historical behavior exactly.
    When in (0, 1], each validated candidate's facts are compared against
    active Facts of the same project+scope before any write; ``dedup_action``
    selects the conservative default "review" (candidate goes to
    needs_confirmation for candidate-review), "skip" (do not write the
    duplicate fact, report the existing fact_id), or "supersede" (write the
    new value superseding the match — same fact_key only, otherwise it falls
    back to "review"). The high-risk evidence gate runs first in
    ``_candidate_decision`` and is never bypassed: a candidate that failed it
    is needs_confirmation/rejected before dedup is consulted.
    """
    dedup_similarity = float(dedup_similarity or 0.0)
    if dedup_similarity:
        if not 0.0 < dedup_similarity <= 1.0:
            raise PLMError("dedup similarity threshold must be in (0, 1]")
        if dedup_action not in DEDUP_ACTIONS:
            raise PLMError("invalid dedup action")
    conn = connect()
    if candidate_id:
        rows = conn.execute(
            "SELECT * FROM candidates WHERE state='pending' AND candidate_id=? LIMIT 1", (candidate_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM candidates WHERE state='pending' ORDER BY created_at LIMIT ?", (limit,)).fetchall()
    conn.close()
    counts = Counter()
    outputs = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        conn = connect()
        project = conn.execute("SELECT canonical_root FROM projects WHERE project_id=?", (row["project_id"],)).fetchone()
        conn.close()
        root = Path(project["canonical_root"]) if project else None
        state, reason = _candidate_decision(payload, root) if root else ("rejected", "project missing")
        dedup_matches: List[Dict[str, Any]] = []
        if state == "validated" and dedup_similarity > 0.0:
            override, dedup_reason, dedup_matches = _dedup_precheck(
                payload, row["project_id"], dedup_similarity, dedup_action, embedding_provider)
            if override:
                state, reason = override, dedup_reason
        try:
            if state == "validated":
                source_event_id = str(payload.get("source_event_id", ""))
                with event_batch(root, row["candidate_id"]) as batch:
                    current = batch["conn"].execute("SELECT state FROM candidates WHERE candidate_id=?", (row["candidate_id"],)).fetchone()
                    if current["state"] != "pending":
                        continue
                    scope = str(payload.get("scope", "project"))
                    scope_id = str(payload.get("scope_id", ""))
                    if payload.get("summary"):
                        extra = {key: payload[key] for key in ("entities", "source_event_id", "evidence_kind", "evidence", "conditions") if key in payload}
                        write_event(
                            root, str(payload.get("title") or "Agent memory candidate"), str(payload["summary"]),
                            tags=payload.get("tags", []), kind=str(payload.get("kind", "episode")),
                            source="agent_candidate", created_by="codex-agent",
                            scope=scope, scope_id=scope_id, observed_at=payload.get("observed_at"),
                            idempotency_key="candidate-summary:" + row["candidate_id"],
                            extra=extra,
                        )
                    dedup_by_position = {match["position"]: match for match in dedup_matches}
                    skipped_duplicates = 0
                    auto_superseded = 0
                    for position, fact in enumerate(payload.get("facts", [])):
                        match = dedup_by_position.get(position)
                        if match and dedup_action == "skip":
                            skipped_duplicates += 1
                            continue
                        supersedes = str(fact.get("supersedes", ""))
                        if match and dedup_action == "supersede" and not supersedes:
                            supersedes = match["fact_id"]
                            auto_superseded += 1
                        write_fact(
                            root, str(fact["fact_key"]), fact.get("value", ""),
                            title=str(fact.get("title") or fact["fact_key"]), tags=payload.get("tags", []),
                            supersedes=supersedes, confidence=float(fact.get("confidence", 0.8)),
                            verified_at=str(fact.get("verified_at", "")), source_event_id=str(fact.get("source_event_id", source_event_id)),
                            source="agent_candidate", idempotency_key="candidate-fact:%s:%d" % (row["candidate_id"], position),
                            entities=fact.get("entities", payload.get("entities", [])), evidence_kind=str(fact.get("evidence_kind", payload.get("evidence_kind", ""))),
                            evidence=fact.get("evidence", payload.get("evidence", {})),
                            scope=str(fact.get("scope", scope)), scope_id=str(fact.get("scope_id", scope_id)),
                            observed_at=fact.get("observed_at", payload.get("observed_at")),
                            valid_from=fact.get("valid_from"), valid_to=fact.get("valid_to", ""), expires_at=fact.get("expires_at", ""),
                            conditions=fact.get("conditions", payload.get("conditions")),
                        )
                    state = "active"
                    reason = "candidate materialized as immutable events"
                    if skipped_duplicates:
                        reason += "; %d near-duplicate fact(s) skipped, existing fact kept" % skipped_duplicates
                    if auto_superseded:
                        reason += "; %d near-duplicate fact(s) auto-superseded" % auto_superseded
                    batch["candidate"] = {"state": state, "reason": reason, "updated_at": utc_now()}
                    batch["conn"].execute("UPDATE candidates SET state=?,reason=?,updated_at=? WHERE candidate_id=?", (state, reason, utc_now(), row["candidate_id"]))
        except (PLMError, KeyError, TypeError, ValueError, sqlite3.Error, OSError):
            if (v2_root() / "batches" / (row["candidate_id"] + ".json")).exists():
                state, reason = "pending", "committed batch requires rebuild-index recovery"
            else:
                state, reason = "rejected", "candidate validation or atomic commit failed"
        state, reason = _finish_candidate(row, state, reason)
        counts[state] += 1
        output = {"candidate_id": row["candidate_id"], "state": state, "reason": reason}
        if dedup_similarity > 0.0:
            output["dedup"] = {
                "action": dedup_action,
                "threshold": dedup_similarity,
                "matches": dedup_matches,
            }
        outputs.append(output)
    return {"processed": len(rows), "states": dict(counts), "results": outputs}


def _finish_candidate(row, state, reason):
    with maintenance_lock():
        with project_lock(row["project_id"]):
            conn = connect()
            try:
                current = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (row["candidate_id"],)).fetchone()
                if current is None:
                    return "removed", "candidate was removed during processing"
                record = dict(current)
                if (v2_root() / "batches" / (row["candidate_id"] + ".json")).is_file():
                    record["batch_id"] = row["candidate_id"]
                # Do not overwrite a concurrent review or resurrect a purged
                # candidate from the stale snapshot captured by the queue reader.
                if current["state"] == row["state"] and current["updated_at"] == row["updated_at"]:
                    record.update(state=state, reason=reason, updated_at=utc_now())
                atomic_write(v2_root() / "candidates" / (row["candidate_id"] + ".json"), json_compact(record))
                conn.execute("UPDATE candidates SET state=?,reason=?,updated_at=? WHERE candidate_id=?", (record["state"], record["reason"], record["updated_at"], row["candidate_id"]))
                conn.commit()
                return record["state"], record["reason"]
            finally:
                conn.close()


def review_candidate(
    cwd: Path, candidate_id: str, decision: str,
    evidence_kind: str = "", evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if decision not in {"approve", "reject"}:
        raise PLMError("candidate decision must be approve or reject")
    conn = connect()
    project = ensure_project(conn, find_project_root(cwd))
    row = conn.execute(
        "SELECT * FROM candidates WHERE candidate_id=? AND project_id=?",
        (candidate_id, project["project_id"]),
    ).fetchone()
    if not row:
        conn.close()
        raise PLMError("candidate not found in current project")
    if row["state"] not in {"pending", "needs_confirmation"}:
        conn.close()
        raise PLMError("candidate is not awaiting review")
    payload = json.loads(row["payload_json"])
    if decision == "reject":
        state, reason = "rejected", "explicitly rejected"
    else:
        claims = [fact_claim(str(fact["fact_key"]), fact.get("value", "")) for fact in payload.get("facts", [])]
        if not all(verify_evidence(evidence_kind, evidence or {}, Path(project["canonical_root"]), claim)["valid"] for claim in (claims or [""])):
            conn.close()
            raise PLMError("approval requires verifiable evidence")
        payload["evidence_kind"] = evidence_kind
        payload["evidence"] = evidence or {}
        state, reason = "pending", "approved for deterministic consolidation"
    serialized = json_compact(payload)
    assert_no_secrets([serialized])
    updated = utc_now()
    record = dict(row)
    record.update({"payload_json": serialized, "state": state, "reason": reason, "updated_at": updated})
    candidate_path = v2_root() / "candidates" / (candidate_id + ".json")
    atomic_write(candidate_path, json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    conn.execute(
        "UPDATE candidates SET payload_json=?,state=?,reason=?,updated_at=? WHERE candidate_id=?",
        (serialized, state, reason, updated, candidate_id),
    )
    conn.commit()
    conn.close()
    if decision == "approve":
        return consolidate(1, candidate_id)
    return {"processed": 1, "states": {"rejected": 1}, "results": [{"candidate_id": candidate_id, "state": state, "reason": reason}]}


def fact_history(cwd: Path, fact_key: str = "", scope: str = "project", scope_id: str = "") -> List[Dict[str, Any]]:
    if scope in {"agent", "session"} and not scope_id:
        raise PLMError("scope-id is required for agent and session scope")
    conn = connect()
    project = ensure_project(conn, find_project_root(cwd))
    scope_id = scope_id or (project["project_id"] if scope == "project" else scope)
    params = [project["project_id"], scope, scope_id]
    clause = " AND f.fact_key=?" if fact_key else ""
    if fact_key:
        params.append(fact_key)
    rows = conn.execute("SELECT f.*,e.scope,e.scope_id FROM facts f JOIN events e ON e.event_id=f.assertion_event_id WHERE f.project_id=? AND e.scope=? AND e.scope_id=?" + clause + " ORDER BY f.recorded_at", params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def supersede_fact(
    cwd: Path, old_fact_id: str, value: Any, valid_from: str = "", confidence: float = 0.9,
    evidence_kind: str = "", evidence: Optional[Dict[str, Any]] = None,
) -> Event:
    conn = connect()
    current_project = ensure_project(conn, find_project_root(cwd))
    row = conn.execute(
        "SELECT f.*,e.scope,e.scope_id,e.source_path FROM facts f JOIN events e ON e.event_id=f.assertion_event_id WHERE f.fact_id=? AND f.project_id=?", (old_fact_id, current_project["project_id"]),
    ).fetchone()
    if not row:
        conn.close()
        raise PLMError("fact not found in current project")
    if row["status"] != "active":
        conn.close()
        raise PLMError("only an active fact can be superseded")
    project = conn.execute("SELECT canonical_root FROM projects WHERE project_id=?", (row["project_id"],)).fetchone()
    conn.close()
    previous_meta = parse_event(Path(row["source_path"])).metadata
    return write_fact(
        Path(project["canonical_root"]), row["fact_key"], value, supersedes=old_fact_id,
        valid_from=valid_from or utc_now(), confidence=confidence, source="explicit_supersede",
        idempotency_key="supersede:%s:%s" % (old_fact_id, hashlib.sha256(json_compact(value).encode("utf-8")).hexdigest()),
        evidence_kind=evidence_kind, evidence=evidence or {},
        scope=row["scope"], scope_id=row["scope_id"], conditions=previous_meta.get("conditions"),
    )


def forget(cwd: Path, target_id: str, purge: bool = False) -> Event:
    conn = connect()
    current_project = ensure_project(conn, find_project_root(cwd))
    row = conn.execute(
        "SELECT e.project_id,e.event_id,e.source_path,e.source_legacy_path FROM events e "
        "WHERE e.event_id=? AND e.project_id=? UNION ALL "
        "SELECT f.project_id,e.event_id,e.source_path,e.source_legacy_path FROM facts f "
        "JOIN events e ON e.event_id=f.assertion_event_id WHERE f.fact_id=? AND f.project_id=? LIMIT 1",
        (target_id, current_project["project_id"], target_id, current_project["project_id"]),
    ).fetchone()
    if not row:
        conn.close()
        raise PLMError("memory target not found in current project")
    project = conn.execute("SELECT canonical_root FROM projects WHERE project_id=?", (row["project_id"],)).fetchone()
    conn.close()
    event = write_event(
        Path(project["canonical_root"]), "Memory retraction", "Retracted memory target %s" % target_id,
        tags=["retraction"], kind="episode", source="explicit_forget", created_by="user_action",
        idempotency_key="retract:" + target_id,
        extra={"operation": "retract", "target_id": target_id},
    )
    if purge:
        from .privacy import record_deletion, read_ledger, ledger_path, apply_deletions
        with maintenance_lock(exclusive=True):
            conn = connect()
            try:
                record_deletion(conn, target_id, current_project["project_id"])
                apply_deletions(conn, read_ledger(ledger_path()))
            finally:
                conn.close()
    return event


def doctor() -> Dict[str, Any]:
    if not database.db_path().is_file():
        return {
            "engine": read_config()["engine"], "integrity": "missing", "counts": {},
            "disk_event_count": 0, "missing_event_files": [], "unindexed_event_files": [],
            "permission_issues": [], "semantic_issues": ["memory index is missing"],
            "candidate_file_count": 0, "healthy": False,
        }
    conn = connect_readonly()
    report = integrity(conn)
    db_paths = {row[0] for row in conn.execute("SELECT source_path FROM events")}
    semantic_issues: List[str] = []
    stale_event_indexes = conn.execute(
        "SELECT COUNT(*) FROM vectors v JOIN events e ON v.record_type='event' AND v.ref_id=e.event_id "
        "WHERE e.status<>'active' OR e.operation<>'' OR e.kind='fact'"
    ).fetchone()[0]
    stale_fact_indexes = conn.execute(
        "SELECT COUNT(*) FROM vectors v JOIN facts f ON v.record_type='fact' AND v.ref_id=f.fact_id "
        "WHERE f.status='retracted'",
    ).fetchone()[0]
    replay_mismatches = conn.execute(
        "SELECT COUNT(*) FROM events op JOIN events target ON target.event_id=op.target_id "
        "WHERE op.operation='retract' AND target.status='active'"
    ).fetchone()[0]
    duplicate_active_facts = conn.execute(
        "SELECT COUNT(*) FROM (SELECT f.project_id,f.fact_key,e.scope,e.scope_id FROM facts f JOIN events e ON e.event_id=f.assertion_event_id WHERE f.status='active' "
        "GROUP BY f.project_id,f.fact_key,e.scope,e.scope_id HAVING COUNT(*)>1)"
    ).fetchone()[0]
    if stale_event_indexes:
        semantic_issues.append("%d inactive or non-searchable events remain indexed" % stale_event_indexes)
    if stale_fact_indexes:
        semantic_issues.append("%d retracted facts remain indexed" % stale_fact_indexes)
    if replay_mismatches:
        semantic_issues.append("%d retractions have active targets" % replay_mismatches)
    if duplicate_active_facts:
        semantic_issues.append("%d fact keys have multiple active versions" % duplicate_active_facts)
    conn.close()
    all_disk_paths = list((v2_root() / "projects").glob("*/events/*/*/*.md"))
    disk_paths = {str(path) for path in all_disk_paths if batch_committed(parse_event(path))}
    permission_issues = []
    for path in [v2_root(), database.db_path()]:
        if path.exists():
            mode = path.stat().st_mode & 0o777
            expected = 0o700 if path.is_dir() else 0o600
            if mode != expected:
                permission_issues.append({"path": str(path), "mode": oct(mode), "expected": oct(expected)})
    config = read_config()
    candidate_files = list((v2_root() / "candidates").glob("*.json")) if (v2_root() / "candidates").exists() else []
    for candidate_file in candidate_files:
        candidate_record = json.loads(candidate_file.read_text(encoding="utf-8"))
        if candidate_record.get("state") == "active" and candidate_record.get("batch_id") and not (v2_root() / "batches" / (candidate_record["batch_id"] + ".json")).exists():
            semantic_issues.append("active candidate commit marker missing")
    report.update({
        "engine": config["engine"],
        "disk_event_count": len(disk_paths),
        "uncommitted_batch_files": len(all_disk_paths) - len(disk_paths),
        "missing_event_files": sorted(db_paths - disk_paths),
        "unindexed_event_files": sorted(disk_paths - db_paths),
        "permission_issues": permission_issues,
        "semantic_issues": semantic_issues,
        "candidate_file_count": len(candidate_files),
        "healthy": report["integrity"] == "ok" and not (db_paths - disk_paths) and not (disk_paths - db_paths) and not permission_issues and not semantic_issues,
    })
    return report


def record_usage(
    project_id: str, operation: str, client: str, query: str,
    result_count: int, success: bool, latency_ms: float = 0.0,
) -> None:
    """Record invocation metadata without storing query text or memory content."""
    conn = connect()
    conn.execute(
        "INSERT INTO usage_runs(created_at,project_id,operation,client,query_hash,result_count,success,latency_ms) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (
            utc_now(), project_id, operation, client or "unknown",
            hashlib.sha256(query.encode("utf-8")).hexdigest() if query else "",
            max(0, int(result_count)), 1 if success else 0, max(0.0, float(latency_ms)),
        ),
    )
    conn.commit()
    conn.close()


def usage_report(days: int = 7) -> Dict[str, Any]:
    if days < 1 or days > 3650:
        raise PLMError("usage days must be between 1 and 3650")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="microseconds").replace("+00:00", "Z")
    conn = connect_readonly()
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='usage_runs'").fetchone():
        conn.close()
        return {"days": days, "since": cutoff, "summary": [], "daily": []}
    summary = [dict(row) for row in conn.execute(
        "SELECT operation,client,COUNT(*) AS runs,SUM(success) AS successful,SUM(result_count) AS results,"
        "AVG(latency_ms) AS average_latency_ms "
        "FROM usage_runs WHERE created_at>=? GROUP BY operation,client ORDER BY runs DESC",
        (cutoff,),
    )]
    daily = [dict(row) for row in conn.execute(
        "SELECT substr(created_at,1,10) AS day,operation,COUNT(*) AS runs "
        "FROM usage_runs WHERE created_at>=? GROUP BY day,operation ORDER BY day DESC,operation",
        (cutoff,),
    )]
    conn.close()
    return {"days": days, "since": cutoff, "summary": summary, "daily": daily}


def record_shadow(project_id: str, query: str, legacy_refs: Sequence[str], v2_refs: Sequence[str], latency_ms: float) -> None:
    legacy = list(legacy_refs)
    v2 = list(v2_refs)
    denominator = max(1, min(len(legacy), len(v2)))
    overlap = len(set(legacy) & set(v2)) / denominator
    conn = connect()
    conn.execute(
        "INSERT INTO shadow_runs(created_at,project_id,query_hash,legacy_refs_json,v2_refs_json,overlap,latency_ms) VALUES(?,?,?,?,?,?,?)",
        (utc_now(), project_id, hashlib.sha256(query.encode("utf-8")).hexdigest(), json_compact(legacy), json_compact(v2), overlap, latency_ms),
    )
    conn.commit()
    conn.close()


def shadow_report() -> Dict[str, Any]:
    conn = connect()
    row = conn.execute(
        "SELECT COUNT(*) AS runs,COALESCE(AVG(overlap),0) AS average_overlap,COALESCE(AVG(latency_ms),0) AS average_latency_ms,"
        "COALESCE(MAX(latency_ms),0) AS max_latency_ms FROM shadow_runs"
    ).fetchone()
    per_project = [dict(item) for item in conn.execute(
        "SELECT project_id,COUNT(*) AS runs,AVG(overlap) AS average_overlap,AVG(latency_ms) AS average_latency_ms "
        "FROM shadow_runs GROUP BY project_id ORDER BY runs DESC"
    )]
    conn.close()
    return {"summary": dict(row), "projects": per_project}
