"""Bounded, local evidence checks; these do not attest a user or remote service.

A matching hash proves which local bytes were inspected. A claim-bound receipt
proves that those bytes explicitly record the claim, not that its author was a
real user or that the recorded external operation actually happened.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .model import project_id_for_root
from .security import SecretDetected, assert_no_secrets


MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_CLAIM_CHARS = 8192
HIGH_RISK_CLAIM = re.compile(
    r"发布|上线|部署|验收|付款|支付|权限|账号|publish|release|deploy|accept|payment|permission|account", re.I,
)
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_KINDS = {"user_confirmation", "git_commit", "file_hash", "command_output", "test_result"}
_OPERATIONS = (
    ("deployment", re.compile(r"deploy|部署", re.I)),
    ("release", re.compile(r"release|publish|发布|上线", re.I)),
    ("acceptance", re.compile(r"accept|验收", re.I)),
    ("payment", re.compile(r"payment|付款|支付", re.I)),
    ("permission", re.compile(r"permission|权限", re.I)),
    ("account", re.compile(r"account|账号", re.I)),
)
_STATES = {
    "deployment": {"deployed": "deployed", "已部署": "deployed", "not_deployed": "not_deployed", "未部署": "not_deployed", "failed": "failed", "部署失败": "failed"},
    "release": {"released": "released", "published": "released", "已发布": "released", "已上线": "released", "unreleased": "unreleased", "未发布": "unreleased", "未上线": "unreleased", "blocked": "blocked"},
    "acceptance": {"passed": "passed", "已验收": "passed", "验收通过": "passed", "failed": "failed", "验收失败": "failed", "pending": "pending", "未验收": "pending", "待验收": "pending"},
    "payment": {"paid": "paid", "已付款": "paid", "已支付": "paid", "unpaid": "unpaid", "未付款": "unpaid", "未支付": "unpaid", "pending": "pending", "failed": "failed", "支付失败": "failed"},
    "permission": {"granted": "granted", "已授权": "granted", "denied": "denied", "拒绝": "denied", "revoked": "revoked", "已撤销": "revoked"},
    "account": {"active": "active", "已启用": "active", "disabled": "disabled", "已停用": "disabled", "locked": "locked", "已锁定": "locked"},
}


def fact_claim(fact_key: str, value: Any) -> str:
    """Canonical binding includes the fact key as well as its JSON value."""
    return json.dumps({"fact_key": fact_key.strip(), "value": value}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def is_high_risk_claim(claim: str) -> bool:
    return bool(HIGH_RISK_CLAIM.search(claim))


def _result(valid: bool = False, reason: str = "invalid_evidence", level: str = "none",
            source_verified: bool = False, supports_claim: bool = False,
            high_risk_eligible: bool = False) -> Dict[str, Any]:
    return {
        "valid": valid, "reason": reason, "level": level,
        "source_verified": source_verified, "supports_claim": supports_claim,
        "high_risk_eligible": high_risk_eligible,
        "identity_verified": False, "external_state_verified": False,
    }


def _path_parts(value: Any, root: Path, supplied_root: Path) -> Tuple[str, ...]:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        raise ValueError("invalid_path")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("unsafe_path")
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            try:
                path = path.relative_to(supplied_root)
            except ValueError:
                raise ValueError("unsafe_path") from None
    if not path.parts:
        raise ValueError("invalid_path")
    return path.parts


def _read_local(value: Any, expected: Any, root: Path, supplied_root: Path) -> bytes:
    if not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise ValueError("invalid_hash")
    parts = _path_parts(value, root, supplied_root)
    # Walk with dir_fd so a replaced/symlinked intermediate directory cannot
    # redirect the read. O_NONBLOCK avoids hanging on a supplied FIFO.
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(str(root), directory_flags)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("not_regular_file")
            if before.st_size > MAX_EVIDENCE_BYTES:
                raise ValueError("evidence_too_large")
            chunks = []
            remaining = MAX_EVIDENCE_BYTES + 1
            while remaining:
                chunk = os.read(file_descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(file_descriptor)
            if len(data) > MAX_EVIDENCE_BYTES:
                raise ValueError("evidence_too_large")
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("evidence_changed_during_read")
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)
    if hashlib.sha256(data).hexdigest() != expected.lower():
        raise ValueError("hash_mismatch")
    assert_no_secrets([data.decode("utf-8", errors="replace")])
    return data


def _git(root: Path, *arguments: str) -> bytes:
    # Disable inherited Git execution context and lazy object fetching. No
    # contributor-provided shell command or external diff/filter is executed.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                        "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1",
                        "GIT_NO_REPLACE_OBJECTS": "1"})
    process = subprocess.run(
        ["git", "--no-pager", "-C", str(root), *arguments],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=environment, timeout=5, check=False,
    )
    if process.returncode:
        raise ValueError("git_object_unavailable")
    return process.stdout


def _verify_git(evidence: Dict[str, Any], root: Path, supplied_root: Path) -> None:
    commit = evidence.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise ValueError("invalid_commit")
    top = _git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(top).resolve() != root:
        raise ValueError("git_project_mismatch")
    if _git(root, "cat-file", "-t", commit).strip() != b"commit":
        raise ValueError("not_a_commit")
    if "path" in evidence or "sha256" in evidence:
        path = "/".join(_path_parts(evidence.get("path"), root, supplied_root))
        expected = evidence.get("sha256")
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise ValueError("invalid_hash")
        # Literal pathspec prevents a user path being interpreted as glob magic.
        listing = _git(root, "ls-tree", "-z", commit, "--", ":(literal)" + path)
        entries = [entry for entry in listing.split(b"\x00") if entry]
        if len(entries) != 1:
            raise ValueError("git_file_unavailable")
        header, actual_path = entries[0].split(b"\t", 1)
        mode, object_type, object_id = header.split(b" ")
        if mode not in {b"100644", b"100755"} or object_type != b"blob" or actual_path.decode("utf-8") != path:
            raise ValueError("git_file_unavailable")
        blob = object_id.decode("ascii")
        size = int(_git(root, "cat-file", "-s", blob).strip())
        if size > MAX_EVIDENCE_BYTES:
            raise ValueError("evidence_too_large")
        data = _git(root, "cat-file", "blob", blob)
        if len(data) != size or hashlib.sha256(data).hexdigest() != expected.lower():
            raise ValueError("hash_mismatch")
        assert_no_secrets([data.decode("utf-8", errors="replace")])


def _receipt(data: bytes) -> Optional[Dict[str, Any]]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_receipt_field")
            result[key] = value
        return result
    try:
        receipt = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        return None
    return receipt


def _bound(receipt: Dict[str, Any], evidence: Dict[str, Any], claim: str) -> bool:
    return bool(claim and evidence.get("asserted_claim") == claim and receipt.get("asserted_claim") == claim)


def _tool_supports(receipt: Dict[str, Any], evidence: Dict[str, Any], claim: str) -> bool:
    if receipt.get("receipt_type") != "tool_result" or not _bound(receipt, evidence, claim):
        return False
    try:
        assertion = json.loads(claim)
        if not isinstance(assertion, dict) or set(assertion) != {"fact_key", "value"}:
            return False
        key, value = assertion["fact_key"], assertion["value"]
        if not isinstance(key, str) or not isinstance(value, str) or not key.endswith((".status", ".state")):
            return False
        if receipt.get("assertion") != assertion or fact_claim(key, value) != claim:
            return False
        operations = [operation for operation, pattern in _OPERATIONS if pattern.search(key)]
        if len(operations) != 1:
            return False
        operation = operations[0]
        canonical_state = _STATES[operation].get(value.lower())
        if not canonical_state or receipt.get("operation") != operation or receipt.get("status") != canonical_state:
            return False
        for field in ("tool", "subject"):
            if not isinstance(receipt.get(field), str) or not receipt[field].strip() or len(receipt[field]) > 512:
                return False
        observed_at = receipt.get("observed_at")
        if not isinstance(observed_at, str):
            return False
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        return observed.tzinfo is not None
    except (ValueError, TypeError, KeyError, RecursionError):
        return False


def _check_source_event(source_event: Any, root: Path) -> None:
    if source_event is None:
        return
    metadata = source_event.metadata if hasattr(source_event, "metadata") else source_event
    if not isinstance(metadata, dict):
        try:
            metadata = dict(metadata)
        except (TypeError, ValueError):
            raise ValueError("invalid_source_event") from None
    if metadata.get("status") in {"retracted", "expired"} or metadata.get("operation") in {"retract", "forget", "purge"}:
        raise ValueError("inactive_source_event")
    project_root = metadata.get("project_root")
    if project_root:
        if Path(project_root).resolve() != root:
            raise ValueError("source_project_mismatch")
    elif metadata.get("project_id") != project_id_for_root(root):
        raise ValueError("source_project_mismatch")


def verify_evidence(kind: str, evidence: Any, project_root: Path, claim: str = "", source_event: Any = None) -> Dict[str, Any]:
    """Verify local provenance; return static reasons without source text/paths.

    Ordinary file/output hashes and existing commits establish a source version.
    They never independently qualify a high-risk Fact. A confirmation must be a
    hashed JSON receipt bound to this exact claim. A tool receipt must additionally
    express a recognized structured status. The caller must still enforce scope,
    current-state policy, and any requirement for actual external attestation.
    """
    if not isinstance(kind, str) or kind not in _KINDS or not isinstance(evidence, dict):
        return _result(reason="unsupported_evidence")
    if not isinstance(claim, str) or len(claim) > MAX_CLAIM_CHARS:
        return _result(reason="invalid_claim")
    try:
        serialized = json.dumps(evidence, ensure_ascii=False, allow_nan=False)
        if len(serialized) > MAX_EVIDENCE_BYTES:
            return _result(reason="evidence_too_large")
        assert_no_secrets([serialized, claim])
        supplied_root = Path(project_root).expanduser().absolute()
        root = supplied_root.resolve(strict=True)
        if not root.is_dir():
            return _result(reason="invalid_project_root")
        _check_source_event(source_event, root)
        high_risk = is_high_risk_claim(claim)
        receipt = None
        if kind == "git_commit":
            _verify_git(evidence, root, supplied_root)
        elif kind == "file_hash":
            _read_local(evidence.get("path"), evidence.get("sha256"), root, supplied_root)
        elif kind in {"command_output", "test_result"}:
            data = _read_local(evidence.get("output_path"), evidence.get("output_hash"), root, supplied_root)
            receipt = _receipt(data)
        else:
            data = _read_local(evidence.get("source_path"), evidence.get("source_hash"), root, supplied_root)
            receipt = _receipt(data)
            if not receipt or receipt.get("receipt_type") != "user_confirmation" or receipt.get("decision") != "confirmed" or not _bound(receipt, evidence, claim):
                return _result(reason="confirmation_not_bound", level="version_verified", source_verified=True)
            return _result(True, "local_confirmation_record_matches", "claim_bound", True, True, True)
        if receipt and _tool_supports(receipt, evidence, claim):
            return _result(True, "local_tool_record_matches", "claim_bound", True, True, True)
        if high_risk:
            return _result(reason="claim_not_supported", level="version_verified", source_verified=True)
        return _result(True, "source_version_verified", "version_verified", True)
    except SecretDetected:
        return _result(reason="secret_detected")
    except ValueError as exc:
        # Only our controlled error vocabulary is exposed. Library errors may
        # contain paths, user text or source bytes and are never returned.
        reason = str(exc)
        known = {"invalid_path", "unsafe_path", "invalid_hash", "not_regular_file", "evidence_too_large", "evidence_changed_during_read", "hash_mismatch", "git_object_unavailable", "invalid_commit", "git_project_mismatch", "not_a_commit", "git_file_unavailable", "invalid_source_event", "inactive_source_event", "source_project_mismatch"}
        return _result(reason=reason if reason in known else "invalid_evidence")
    except (OSError, TypeError, OverflowError, RecursionError, subprocess.SubprocessError):
        return _result(reason="source_unavailable")
