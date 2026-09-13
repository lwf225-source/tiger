from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .database import connect
from .events import atomic_write, maintenance_lock, parse_event
from .model import PLMError, canonical_root, memory_base, sha256_bytes, utc_now, v2_root
from .search import search
from .security import ensure_private_dir
from .service import doctor, migration_plan, read_config, record_shadow, shadow_report, write_config


def repository_root() -> Path:
    package = Path(__file__).resolve().parent
    source = package.parent
    if (source / "skill" / "SKILL.md").is_file():
        return source
    installed = package.parent.parent
    if (installed / "SKILL.md").is_file() and (installed / "scripts").is_dir():
        return installed
    raise PLMError("could not locate the PLM source or installed bundle root")


def code_fingerprint() -> str:
    root = repository_root()
    candidates: List[Path] = []
    for relative in ("project_long_memory", "lib/project_long_memory", "scripts"):
        folder = root / relative
        if folder.is_dir():
            candidates.extend(folder.rglob("*.py"))
    skill_file = root / "skill" / "SKILL.md" if (root / "skill" / "SKILL.md").is_file() else root / "SKILL.md"
    if skill_file.is_file():
        candidates.append(skill_file)
    digest = __import__("hashlib").sha256()
    for path in sorted(set(candidates), key=lambda item: str(item.relative_to(root))):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_bundle_tests(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repository_root()
    tests = root / "tests"
    if not tests.is_dir():
        return {"passed": False, "returncode": 2, "summary": "installed bundle has no tests"}
    env = dict(os.environ)
    import_path = root / "lib" if (root / "lib" / "project_long_memory").is_dir() else root
    env["PYTHONPATH"] = str(import_path)
    env["PLM_INSTALL_VALIDATION"] = "1"
    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", str(tests), "-v"],
        cwd=str(root), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return {"passed": result.returncode == 0, "returncode": result.returncode, "summary": output[-4000:]}


def backup_memory(destination: Optional[Path] = None) -> Dict[str, Any]:
    backup_dir = destination or (memory_base() / "backups")
    ensure_private_dir(backup_dir)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    archive = backup_dir / ("project-memory-" + stamp + ".tar.gz")
    with maintenance_lock(exclusive=True):
        from .privacy import ledger_path, read_ledger
        if not ledger_path().exists():
            atomic_write(ledger_path(), json.dumps({"schema_version": 1, "deletions": []}))
        else:
            read_ledger(ledger_path())
        atomic_write(v2_root() / "backup-metadata.json", json.dumps({
            "schema_version": 1, "source_memory_root": str(memory_base()), "created_at": utc_now(),
        }))
        if (v2_root() / "catalog.sqlite3").is_file():
            conn = connect()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        with tarfile.open(str(archive), "w:gz") as bundle:
            for name in ("global.md", "projects", "v2"):
                path = memory_base() / name
                if not path.exists():
                    continue
                if name == "v2":
                    def exclude_backups(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
                        return None if ".rebuild-backup-" in info.name or "/backups/" in info.name else info
                    bundle.add(str(path), arcname=name, filter=exclude_backups)
                else:
                    bundle.add(str(path), arcname=name)
    os.chmod(str(archive), 0o600)
    return {"archive": str(archive), "size": archive.stat().st_size, "created_at": utc_now(),
            "deletion_ledger": str(ledger_path()), "requires_latest_deletion_ledger_for_restore": True,
            "archives_retain_historical_data": True}


def _checked_archive_members(bundle: tarfile.TarFile) -> List[tarfile.TarInfo]:
    """Validate the complete archive before creating any member on disk."""
    checked = []
    names = set()
    total_size = 0
    for member in bundle:
        path = PurePosixPath(member.name)
        if (not member.name or "\\" in member.name or "\x00" in member.name or path.is_absolute()
                or ".." in path.parts or not path.parts
                or path.parts[0] not in {"v2", "projects", "global.md"}
                or not (member.isfile() or member.isdir()) or member.issparse()):
            raise PLMError("unsafe backup archive member")
        normalized = str(path)
        if normalized in names:
            raise PLMError("duplicate backup archive member")
        names.add(normalized)
        if member.size < 0 or member.size > 128 * 1024 * 1024:
            raise PLMError("backup archive member exceeds size limit")
        total_size += member.size
        if total_size > 2 * 1024 * 1024 * 1024 or len(checked) >= 100000:
            raise PLMError("backup archive exceeds restore limit")
        checked.append(member)
    return checked


def _extract_checked(bundle: tarfile.TarFile, members: List[tarfile.TarInfo], root: Path) -> None:
    for member in members:
        path = root.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            ensure_private_dir(path)
            continue
        ensure_private_dir(path.parent)
        source = bundle.extractfile(member)
        if source is None:
            raise PLMError("backup archive file is unavailable")
        # No extract/extractall: never apply archive modes, ownership or links.
        with source, path.open("xb") as destination:
            remaining = member.size
            while remaining:
                block = source.read(min(65536, remaining))
                if not block:
                    raise PLMError("backup archive file is truncated")
                destination.write(block)
                remaining -= len(block)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(str(path), 0o600)


def _merge_deletion_ledgers(*ledgers: Dict[str, Any]) -> Dict[str, Any]:
    entries = []
    seen = set()
    for ledger in ledgers:
        for item in ledger["deletions"]:
            # Deletions may address only ordinary files inside the restored
            # legacy notes collection. Never accept arbitrary operator paths.
            for relative in item.get("legacy_relative_paths", []):
                if not isinstance(relative, str):
                    raise PLMError("unsafe deletion ledger path")
                path = PurePosixPath(relative)
                if (path.is_absolute() or ".." in path.parts or "\\" in relative
                        or len(path.parts) != 4 or path.parts[0] != "projects"
                        or path.parts[2] != "notes" or path.suffix != ".md"):
                    raise PLMError("unsafe deletion ledger path")
            identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if identity not in seen:
                seen.add(identity)
                entries.append(item)
    return {"schema_version": 1, "deletions": entries}


def _restored_legacy_path(value: str, origins: Sequence[str], restored_root: Path) -> str:
    path = Path(value)
    for origin in origins:
        try:
            relative = path.relative_to(Path(origin))
        except ValueError:
            continue
        if ".." in relative.parts or not relative.parts or relative.parts[0] != "projects":
            raise PLMError("unsafe legacy path in backup")
        return str(restored_root / relative)
    return value


def _archive_legacy_origins(staging_root: Path) -> List[str]:
    """Infer old backup roots only from matching files already inside the archive."""
    origins = []
    for path in (staging_root / "v2" / "projects").glob("*/events/*/*/*.md"):
        event = parse_event(path)
        value = event.metadata.get("source_legacy_path", "")
        if not value:
            continue
        source = Path(value)
        parts = source.parts
        if (not source.is_absolute() or ".." in parts or len(parts) < 5
                or parts[-4] != "projects" or parts[-2] != "notes" or source.suffix != ".md"):
            raise PLMError("unsafe legacy provenance in backup")
        local_copy = staging_root.joinpath(*parts[-4:])
        if not local_copy.is_file():
            raise PLMError("historical backup has unresolved legacy provenance")
        origin = str(source.parents[3])
        if origin not in origins:
            origins.append(origin)
    return origins


def _rebase_restore_index(staging_root: Path, destination: Path, origins: Sequence[str]) -> None:
    """Adjust derived paths only; immutable Markdown provenance is preserved."""
    conn = connect()
    try:
        for row in conn.execute("SELECT event_id,source_path,source_legacy_path FROM events").fetchall():
            source_path = Path(row["source_path"])
            try:
                relative = source_path.relative_to(staging_root)
            except ValueError:
                raise PLMError("restored index contains an external event path") from None
            conn.execute("UPDATE events SET source_path=?,source_legacy_path=? WHERE event_id=?", (
                str(destination / relative), _restored_legacy_path(row["source_legacy_path"], origins, destination), row["event_id"],
            ))
        for row in conn.execute("SELECT source_path FROM migrations").fetchall():
            conn.execute("UPDATE migrations SET source_path=? WHERE source_path=?", (
                _restored_legacy_path(row[0], origins, destination), row[0],
            ))
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def restore_backup(archive: Path, destination: Path, deletion_ledger: Path) -> Dict[str, Any]:
    """CLI/single-process restore to a fresh store; never changes the active store.

    The operator must explicitly supply the latest external deletion ledger.
    Freshness cannot be established from the archive's own historical copy.
    This function temporarily changes PROJECT_LONG_MEMORY_DIR and must not run
    concurrently with other in-process memory operations.
    """
    from .privacy import read_ledger
    from .service import rebuild_index

    if deletion_ledger is None:
        raise PLMError("restore requires an explicit latest deletion ledger")
    ledger_source = Path(deletion_ledger).expanduser().absolute()
    latest = read_ledger(ledger_source)
    destination = Path(destination).expanduser().absolute()
    if destination.is_symlink():
        raise PLMError("restore destination must not be a symlink")
    destination = destination.parent.resolve(strict=True) / destination.name
    active_root = memory_base()
    if (destination == active_root or destination in active_root.parents
            or active_root in destination.parents):
        raise PLMError("restore destination must be separate from the active memory store")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise PLMError("restore destination must be new or empty")
    archive = Path(archive).expanduser().absolute()
    if archive.is_symlink() or not archive.is_file():
        raise PLMError("backup archive must be a regular local file")
    original_environment = os.environ.get("PROJECT_LONG_MEMORY_DIR")
    staging = Path(tempfile.mkdtemp(prefix=".plm-restore-", dir=str(destination.parent)))
    published = False
    try:
        with tarfile.open(str(archive), "r:*") as bundle:
            members = _checked_archive_members(bundle)
            _extract_checked(bundle, members, staging)
        if not (staging / "v2").is_dir():
            raise PLMError("backup archive has no v2 store")
        archived_ledger_path = staging / "v2" / "privacy" / "deletions.json"
        archived = read_ledger(archived_ledger_path) if archived_ledger_path.is_file() else {"schema_version": 1, "deletions": []}
        merged = _merge_deletion_ledgers(archived, latest)
        origins = []
        for metadata_name in ("backup-metadata.json", "restore-origin.json"):
            metadata_path = staging / "v2" / metadata_name
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                values = [metadata.get("source_memory_root", ""), *metadata.get("source_memory_roots", [])]
                for value in values:
                    if isinstance(value, str) and value and Path(value).is_absolute() and value not in origins:
                        origins.append(value)
        if not origins:
            origins = _archive_legacy_origins(staging)
        os.environ["PROJECT_LONG_MEMORY_DIR"] = str(staging)
        atomic_write(archived_ledger_path, json.dumps(merged, ensure_ascii=False, sort_keys=True))
        atomic_write(staging / "v2" / "restore-origin.json", json.dumps({
            "schema_version": 1, "source_memory_root": origins[0] if origins else "",
            "source_memory_roots": origins, "restored_at": utc_now(),
        }))
        # Only discard SQLite files in our isolated copy. Replaying the archive's
        # absolute-path database could otherwise act on the live source store.
        for path in (staging / "v2").iterdir():
            if path.name == "catalog.sqlite3" or path.name.startswith(("catalog.sqlite3-", "catalog.sqlite3.rebuild-backup-")):
                if not path.is_file():
                    raise PLMError("invalid archived index file")
                path.unlink()
        rebuilt = rebuild_index()
        if not rebuilt.get("swapped") or rebuilt.get("errors"):
            raise PLMError("restored sources failed index rebuild")
        checked = doctor()
        if not checked.get("healthy"):
            raise PLMError("restored store failed health checks")
        _rebase_restore_index(staging, destination, [str(staging), *origins])
        if destination.exists() and (destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())):
            raise PLMError("restore destination changed during validation")
        os.replace(str(staging), str(destination))
        published = True
        os.environ["PROJECT_LONG_MEMORY_DIR"] = str(destination)
        checked = doctor()
        if not checked.get("healthy"):
            # Withdraw only the directory created by this restore operation.
            os.replace(str(destination), str(staging))
            published = False
            raise PLMError("restored store failed final path validation")
        return {"destination": str(destination), "healthy": True, "counts": checked["counts"],
                "ledger_entries": len(merged["deletions"]), "restored_at": utc_now(),
                "active_store_changed": False, "ledger_freshness": "operator_supplied",
                "archives_retain_historical_data": True}
    except (tarfile.TarError, OSError, ValueError, TypeError, KeyError):
        raise PLMError("backup restoration failed safely") from None
    finally:
        if original_environment is None:
            os.environ.pop("PROJECT_LONG_MEMORY_DIR", None)
        else:
            os.environ["PROJECT_LONG_MEMORY_DIR"] = original_environment
        if not published and staging.exists():
            shutil.rmtree(str(staging))


def _copy_skill_bundle(destination: Path) -> None:
    repo = repository_root()
    ensure_private_dir(destination)
    skill_root = repo / "skill" if (repo / "skill" / "SKILL.md").is_file() else repo
    package_root = repo / "project_long_memory" if (repo / "project_long_memory").is_dir() else repo / "lib" / "project_long_memory"
    shutil.copy2(str(skill_root / "SKILL.md"), str(destination / "SKILL.md"))
    shutil.copytree(str(skill_root / "agents"), str(destination / "agents"))
    shutil.copytree(str(skill_root / "legacy"), str(destination / "legacy"))
    shutil.copytree(str(repo / "scripts"), str(destination / "scripts"))
    shutil.copytree(str(package_root), str(destination / "lib" / "project_long_memory"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (repo / "tests").is_dir():
        shutil.copytree(str(repo / "tests"), str(destination / "tests"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (repo / "evaluation").is_dir():
        shutil.copytree(str(repo / "evaluation"), str(destination / "evaluation"), ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "reports", "models", "*.safetensors", "*.onnx", "*.bin",
        ))
    code_examples = repo / "examples" / "code_memory"
    if code_examples.is_dir():
        shutil.copytree(str(code_examples), str(destination / "examples" / "code_memory"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "VERSION").write_text(__version__ + "\n", encoding="utf-8")


def install_skill(target: Optional[Path] = None) -> Dict[str, Any]:
    previous_config = read_config()
    target = target or Path(os.environ.get("PLM_SKILL_DIR", str(Path.home() / ".codex" / "skills" / "project-long-memory")))
    target = target.expanduser().absolute()
    if target.is_symlink():
        raise PLMError("refusing to replace symlinked skill directory")
    ensure_private_dir(target.parent)
    temp = Path(tempfile.mkdtemp(prefix=".project-long-memory-v2-", dir=str(target.parent)))
    backup_root = Path(os.environ.get("PLM_SKILL_BACKUP_DIR", str(Path.home() / ".codex" / "skill-backups" / "project-long-memory")))
    ensure_private_dir(backup_root)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup = backup_root / stamp
    try:
        _copy_skill_bundle(temp)
        compile_result = subprocess.run(
            ["python3", "-m", "compileall", "-q", str(temp)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if compile_result.returncode:
            raise PLMError("installed bundle did not compile: " + compile_result.stderr.strip())
        if not os.environ.get("PLM_INSTALL_VALIDATION"):
            tests = run_bundle_tests(temp)
            if not tests["passed"]:
                raise PLMError("installed bundle tests failed: " + tests["summary"])
        if target.exists():
            os.replace(str(target), str(backup))
        try:
            os.replace(str(temp), str(target))
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(str(backup), str(target))
            raise
    finally:
        if temp.exists():
            shutil.rmtree(str(temp))
    original_backup = previous_config.get("skill_backup")
    if not original_backup or not Path(original_backup).exists():
        original_backup = str(backup) if backup.exists() else ""
    try:
        config = write_config({
            "engine": "shadow",
            "write_mode": "v1",
            "installed_version": __version__,
            "skill_backup": original_backup,
            "last_install_backup": str(backup) if backup.exists() else "",
            "cutover_at": "",
        })
    except Exception:
        failed_install = backup_root / (stamp + "-config-failed")
        if target.exists():
            os.replace(str(target), str(failed_install))
        if backup.exists():
            os.replace(str(backup), str(target))
        raise
    return {"target": str(target), "backup": str(backup) if backup.exists() else "", "version": __version__, "engine": config["engine"]}


def _legacy_slug(root: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", root.strip(os.sep)) or "root"


def export_v1(destination: Path, since: str = "") -> Dict[str, Any]:
    destination = destination.expanduser().absolute()
    if destination.is_symlink():
        raise PLMError("refusing to export through a symlinked destination")
    ensure_private_dir(destination)
    conn = connect()
    if since:
        rows = conn.execute(
            "SELECT e.*,p.canonical_root FROM events e JOIN projects p ON p.project_id=e.project_id "
            "WHERE e.recorded_at>=? AND e.source<>'legacy_import' AND e.status='active' AND e.operation='' ORDER BY e.recorded_at",
            (since,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT e.*,p.canonical_root FROM events e JOIN projects p ON p.project_id=e.project_id "
            "WHERE e.source<>'legacy_import' AND e.status='active' AND e.operation='' ORDER BY e.recorded_at"
        ).fetchall()
    conn.close()
    exported = 0
    for row in rows:
        project_dir = destination / "projects" / _legacy_slug(row["canonical_root"]) / "notes"
        ensure_private_dir(project_dir)
        date = row["recorded_at"][:10]
        filename = date + "-" + re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "-", row["title"]).strip("-._")[:80] + "-" + row["event_id"][:8] + ".md"
        text = "\n".join([
            "---", "title: " + row["title"], "time: " + row["recorded_at"],
            "project_root: " + row["canonical_root"], "tags: " + ",".join(json.loads(row["tags_json"])), "---", "", row["body"], "",
        ])
        path = project_dir / filename
        path.write_text(text, encoding="utf-8")
        os.chmod(str(path), 0o600)
        exported += 1
    return {"destination": str(destination), "exported": exported, "since": since}


def rollback_skill(target: Optional[Path] = None, materialize_v1: bool = False) -> Dict[str, Any]:
    config = read_config()
    backup = Path(config.get("skill_backup", "")) if config.get("skill_backup") else None
    if not backup or not backup.exists():
        raise PLMError("no installed skill backup is available")
    target = target or Path(os.environ.get("PLM_SKILL_DIR", str(Path.home() / ".codex" / "skills" / "project-long-memory")))
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    recovery = memory_base() / "rollback-export" / stamp
    exported = export_v1(recovery, config.get("cutover_at", ""))
    failed_copy = backup.parent / ("v2-rolled-back-" + stamp)
    if target.exists():
        os.replace(str(target), str(failed_copy))
    os.replace(str(backup), str(target))
    write_config({"engine": "v1", "write_mode": "v1", "cutover_at": ""})
    materialized = 0
    if materialize_v1:
        source_projects = recovery / "projects"
        destination_projects = memory_base() / "projects"
        for path in source_projects.glob("*/notes/*.md"):
            destination = destination_projects / path.parent.parent.name / "notes" / path.name
            ensure_private_dir(destination.parent)
            shutil.copy2(str(path), str(destination))
            materialized += 1
    return {
        "target": str(target), "restored_from": str(backup), "v2_bundle": str(failed_copy),
        "recovery_export": exported, "materialized": materialized,
    }


def _legacy_search_paths(project_root: str, query: str, limit: int = 5) -> List[str]:
    directory = memory_base() / "projects" / _legacy_slug(project_root) / "notes"
    keywords = set(re.findall(r"[\w\u4e00-\u9fff.-]{2,}", (query + " " + project_root).lower()))
    scored = []
    for path in directory.glob("*.md") if directory.exists() else []:
        text = path.read_text(encoding="utf-8", errors="replace")[:20000].lower()
        score = sum(text.count(word) for word in keywords)
        if score:
            scored.append((score, path.stat().st_mtime, str(path)))
    scored.sort(reverse=True)
    return [path for _, _, path in scored[:limit]]


def run_retrieval_benchmark(sample_limit: int = 80) -> Dict[str, Any]:
    conn = connect()
    rows = conn.execute(
        "SELECT e.event_id,e.project_id,e.title,e.source_legacy_path,p.canonical_root "
        "FROM events e JOIN projects p ON p.project_id=e.project_id "
        "WHERE e.source='legacy_import' AND e.source_legacy_path<>'' ORDER BY e.event_id LIMIT ?",
        (sample_limit,),
    ).fetchall()
    hits = 0
    cross_project = 0
    latencies = []
    for row in rows:
        query = row["title"]
        start = time.perf_counter()
        results = search(conn, row["project_id"], query, 5, "current")
        latency = (time.perf_counter() - start) * 1000.0
        latencies.append(latency)
        refs = [result.source_path for result in results]
        if row["source_legacy_path"] in refs:
            hits += 1
        cross_project += sum(1 for result in results if result.project_id != row["project_id"])
        legacy_refs = _legacy_search_paths(row["canonical_root"], query, 5)
        record_shadow(row["project_id"], query, legacy_refs, refs, latency)
    conn.close()
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    return {
        "queries": len(rows),
        "recall_at_5": (hits / len(rows)) if rows else 0.0,
        "cross_project_results": cross_project,
        "p95_latency_ms": p95,
    }


def verify_acceptance(tests_passed: bool = False) -> Dict[str, Any]:
    health = doctor()
    tests = run_bundle_tests()
    plan = migration_plan()
    conn = connect()
    migration_rows = conn.execute("SELECT source_path,source_hash FROM migrations").fetchall()
    conn.close()
    migrated = len(migration_rows)
    indexed_hashes = {row["source_path"]: row["source_hash"] for row in migration_rows}
    source_hashes = {item["source_path"]: item["source_hash"] for item in plan["entries"]}
    retrieval = run_retrieval_benchmark()
    collision = plan["collision_buckets"].get("Users_tiger_Documents_", {})
    gates = {
        "unit_tests": bool(tests["passed"]),
        "database_integrity": health["integrity"] == "ok",
        "doctor_healthy": bool(health["healthy"]),
        "migration_parity": migrated == plan["valid_count"] and plan["error_count"] == 0,
        "migration_hashes": indexed_hashes == source_hashes,
        "collision_split": not collision or (len(collision) >= 9 and sum(collision.values()) >= 280),
        "cross_project_isolation": retrieval["cross_project_results"] == 0,
        "recall_at_5": retrieval["recall_at_5"] >= 0.90,
        "p95_latency": retrieval["p95_latency_ms"] < 200.0,
    }
    report = {
        "created_at": utc_now(), "passed": all(gates.values()), "gates": gates,
        "health": health, "migration": {"source": plan["valid_count"], "indexed": migrated},
        "retrieval": retrieval, "shadow": shadow_report(),
        "tests": tests, "installed_version": __version__, "code_fingerprint": code_fingerprint(),
        "event_count": health.get("counts", {}).get("events", 0),
    }
    report_dir = v2_root() / "reports"
    ensure_private_dir(report_dir)
    path = report_dir / "acceptance.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(str(path), 0o600)
    report["report_path"] = str(path)
    return report


def cutover() -> Dict[str, Any]:
    path = v2_root() / "reports" / "acceptance.json"
    if not path.exists():
        raise PLMError("acceptance report missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise PLMError("acceptance gates have not passed; remaining in shadow mode")
    health = doctor()
    if report.get("installed_version") != __version__ or report.get("code_fingerprint") != code_fingerprint():
        raise PLMError("acceptance report does not match the installed code")
    if report.get("event_count") != health.get("counts", {}).get("events") or not health.get("healthy"):
        raise PLMError("acceptance report is stale for the current memory data")
    config = write_config({"engine": "v2", "write_mode": "v2", "cutover_at": utc_now()})
    return {"engine": config["engine"], "write_mode": config["write_mode"], "cutover_at": config["cutover_at"]}
