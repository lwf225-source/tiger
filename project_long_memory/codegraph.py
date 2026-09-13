"""Optional local CodeGraph asset. Code is retrieved on demand, never ingested.

The CLI boundary keeps the memory kernel dependency-free. A binding describes
the exact Git working tree successfully indexed by PLM. Queries refuse stale
or unbound indexes; only explicit init/sync commands change the graph.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from .model import PLMError, utc_now
from .security import assert_no_secrets

ACTIONS = ("explore", "query", "callers", "callees", "impact", "files")
MAX_SCAN_BYTES = 128 * 1024 * 1024


def _run(argv, cwd, timeout=30, max_bytes=24000):
    # No shell, daemon, auto-sync or telemetry. The installed binary is explicit.
    env = dict(os.environ, NO_COLOR="1", CODEGRAPH_NO_DAEMON="1",
               DO_NOT_TRACK="1", CODEGRAPH_NO_DOWNLOAD="1", CODEGRAPH_NO_UPDATE_CHECK="1")
    # Caller Git environment must not silently select a different repository.
    for name in list(env):
        if name.startswith("GIT_"):
            del env[name]
    with tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=output,
                                    stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError:
            raise PLMError("codegraph executable unavailable") from None
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise PLMError("codegraph command timed out") from None
        output.seek(0)
        raw = output.read(max_bytes + 1)
        if proc.returncode:
            # Do not leak command stderr (may include credentials or source).
            raise PLMError("codegraph command failed (exit %d)" % proc.returncode)
    text = raw[:max_bytes].decode("utf-8", errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    return text, len(raw) > max_bytes


def _git(root, *args):
    text, truncated = _run(["git", "--no-optional-locks", *args], root,
                           timeout=15, max_bytes=4 * 1024 * 1024)
    if truncated:
        raise PLMError("repository listing exceeds codegraph snapshot budget")
    return text


def _root(cwd):
    cwd = Path(cwd).expanduser().resolve()
    root = Path(_git(cwd, "rev-parse", "--show-toplevel").strip()).resolve()
    if root == Path.home().resolve() or root == Path(root.anchor):
        raise PLMError("codegraph requires a project repository")
    if (root / ".codegraph").is_symlink():
        raise PLMError("codegraph index must be local to the repository")
    return root


def _snapshot(root):
    head = _git(root, "rev-parse", "HEAD").strip()
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    names = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    digest = hashlib.sha256()
    total = 0
    for name in sorted(set(names.split("\0"))):
        if not name or name.startswith(".codegraph/") or name == ".DS_Store":
            continue
        path = root / name
        digest.update(name.encode("utf-8") + b"\0")
        if path.is_symlink():
            digest.update(b"symlink:" + os.readlink(path).encode("utf-8"))
        elif path.is_file():
            total += path.stat().st_size
            if total > MAX_SCAN_BYTES:
                raise PLMError("repository exceeds codegraph snapshot budget; exclude generated assets")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing-or-directory")
        digest.update(b"\0")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal",
                 "--", ".", ":(exclude).codegraph", ":(exclude).DS_Store").strip()
    return {"head": head, "branch": branch, "dirty": bool(dirty),
            "files_sha256": digest.hexdigest()}


def _binding(root):
    path = root / ".codegraph" / "plm-binding.json"
    if path.is_symlink():
        raise PLMError("invalid codegraph binding")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("project_root") != str(root):
        return None
    return data


def _index_health(root):
    executable = shutil.which("codegraph")
    if not executable:
        raise PLMError("codegraph CLI is not installed")
    text, truncated = _run([executable, "status", "--json", str(root)], root)
    try:
        data = json.loads(text)
        index = data.get("index", {})
        if (truncated or not data.get("initialized") or not data.get("lastIndexed")
                or Path(data["projectPath"]).resolve() != root
                or data.get("worktreeMismatch") or index.get("state") != "complete"
                or index.get("reindexRecommended")
                or any(data.get("pendingChanges", {}).values())):
            raise ValueError("incomplete index")
        return {"last_indexed": data["lastIndexed"], "files": data["fileCount"],
                "nodes": data["nodeCount"], "edges": data["edgeCount"],
                "engine_version": data["version"]}
    except (ValueError, KeyError, TypeError, AttributeError):
        raise PLMError("codegraph index incomplete, incompatible or pending sync") from None


def status(cwd):
    root = _root(cwd)
    binding = _binding(root)
    current = _snapshot(root)
    state = "unbound"
    if (root / ".codegraph" / "plm-sync.lock").exists():
        state = "syncing"
    elif binding:
        state = "ready" if binding.get("snapshot") == current else "stale"
    health = None
    if state == "ready":
        try:
            health = _index_health(root)
            if health != binding.get("index_health"):
                state = "stale"
        except PLMError:
            state = "unavailable"
    return {"schema_version": 1, "source_kind": "codegraph",
            "project_root": str(root), "status": state,
            "snapshot": current, "indexed_snapshot": binding.get("snapshot") if binding else None,
            "indexed_at": binding.get("indexed_at") if binding else None,
            "index_health": health,
            "cli_available": shutil.which("codegraph") is not None}


def sync(cwd, initialize=False, timeout=180):
    root = _root(cwd)
    executable = shutil.which("codegraph")
    if not executable:
        raise PLMError("codegraph CLI is not installed")
    graph = root / ".codegraph"
    exists = (graph / "codegraph.db").is_file()
    if not exists and not initialize:
        raise PLMError("codegraph index absent; run plm code init explicitly")
    if not 1 <= timeout <= 600:
        raise PLMError("codegraph sync timeout must be between 1 and 600 seconds")
    # The lock covers initial construction as well as subsequent syncs.
    before = _snapshot(root)
    graph.mkdir(exist_ok=True)
    lock = graph / "plm-sync.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PLMError("codegraph sync already running; inspect plm-sync.lock if interrupted") from None
    try:
        os.close(fd)
        # A failed rebuild must not leave an old "ready" receipt behind.
        (graph / "plm-binding.json").unlink(missing_ok=True)
        if exists:
            _run([executable, "sync", str(root)], root, timeout=timeout)
        else:
            _run([executable, "init", str(root)], root, timeout=timeout)
        after = _snapshot(root)
        if before != after:
            raise PLMError("repository changed while indexing; run plm code sync again")
        health = _index_health(root)
        version, _ = _run([executable, "version"], root)
        data = {"schema_version": 1, "project_root": str(root), "snapshot": after,
                "indexed_at": utc_now(), "cli_version": version.strip(), "index_health": health}
        target = graph / "plm-binding.json"
        if target.is_symlink():
            raise PLMError("invalid codegraph binding")
        fd, temporary = tempfile.mkstemp(dir=str(graph), prefix=".plm-binding-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    finally:
        lock.unlink(missing_ok=True)
    return status(root)


def query(cwd, text="", action="explore", max_bytes=16000, timeout=30):
    if action not in ACTIONS:
        raise PLMError("unsupported codegraph query action")
    if action != "files" and (not text.strip() or len(text) > 4000):
        raise PLMError("codegraph query must contain 1 to 4000 characters")
    if not 256 <= max_bytes <= 64000 or not 1 <= timeout <= 60:
        raise PLMError("invalid codegraph query budget")
    result = {"source_kind": "codegraph", "action": action, "text": "",
              "status": "unavailable", "truncated": False}
    try:
        result.update(status(cwd))
        if result["status"] != "ready":
            result["hint"] = "Run plm code init (unbound) or plm code sync (stale)."
            return result
        executable = shutil.which("codegraph")
        if not executable:
            raise PLMError("codegraph CLI is not installed")
        root = Path(result["project_root"])
        argv = [executable, action]
        if action == "explore":
            argv += ["--max-files", "6"]
        if action != "files":
            argv += ["--", text]
        content, truncated = _run(argv, root, timeout, max_bytes)
        # A concurrent edit or sync invalidates the result just obtained.
        after = status(root)
        if after["status"] != "ready" or after["snapshot"] != result["snapshot"]:
            result.update(status="stale", hint="Repository changed during query; sync and retry.")
            return result
        assert_no_secrets([content])
        result.update(text=content, truncated=truncated,
                      provenance="CodeGraph source and locations; inspect snapshot before reuse.")
    except (PLMError, OSError, ValueError) as exc:
        result.update(status="unavailable", reason=type(exc).__name__,
                      hint="Check plm code status and CLI availability; memory retrieval is unaffected.")
    return result
