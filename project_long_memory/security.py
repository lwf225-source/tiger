from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from .model import PLMError


class SecretDetected(PLMError):
    pass


SECRET_PATTERNS = (
    re.compile(r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)[\"']?\b(password|passwd|pwd|api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|cookie)\b[\"']?\s*[:=]\s*[\"']?([^\s\"'`,;，。}]{6,})"
    ),
)


def assert_no_secrets(values: Iterable[str]) -> None:
    text = "\n".join(value for value in values if value)
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise SecretDetected("memory rejected: probable secret or credential detected")


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(str(path), 0o700)
    except OSError:
        pass


def assert_safe_path(path: Path, root: Path) -> None:
    root_resolved = root.resolve(strict=False)
    path_abs = path.absolute()
    try:
        path_abs.relative_to(root_resolved)
    except ValueError as exc:
        raise PLMError("path escapes memory root") from exc
    current = root_resolved
    relative = path_abs.relative_to(root_resolved)
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise PLMError("symlinked memory directory rejected: %s" % current)
