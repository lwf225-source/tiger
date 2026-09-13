#!/usr/bin/env python3
"""Frozen PLM v1 writer used only while v2 runs in shadow mode."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HOME = Path.home()
BASE = Path(os.environ.get("PROJECT_LONG_MEMORY_DIR", str(HOME / ".codex" / "project-memory"))).expanduser().resolve()
MARKERS = ("AGENTS.md", "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pnpm-workspace.yaml", "yarn.lock", "vite.config.js", "vite.config.ts")


def root_for(cwd: Path) -> Path:
    try:
        result = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.stdout.strip():
            return Path(result.stdout.strip()).resolve()
    except Exception:
        pass
    for path in [cwd.resolve(), *cwd.resolve().parents]:
        if any((path / marker).exists() for marker in MARKERS):
            return path
    return cwd.resolve()


def slug(root: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(root.resolve()).strip(os.sep)) or "root"


def safe_title(title: str) -> str:
    value = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "-", title.strip().lower())
    return re.sub(r"-+", "-", value).strip("-._")[:80] or "memory-note"


def reject_secrets(text: str) -> None:
    patterns = (
        r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
        r"(?i)\b(password|passwd|pwd|api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|cookie)\b\s*[:=]\s*[^\s]{6,}",
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError("probable secret detected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--title", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--content", default="")
    args = parser.parse_args()
    content = (args.content or sys.stdin.read()).strip()
    if not content:
        print("memory_write.py: no content provided", file=sys.stderr)
        return 2
    try:
        reject_secrets("\n".join((args.title, args.tags, content)))
    except ValueError as exc:
        print("memory_write.py: %s" % exc, file=sys.stderr)
        return 2
    root = root_for(Path(args.cwd).expanduser().resolve())
    project = BASE / "projects" / slug(root)
    notes = project / "notes"
    notes.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    day = datetime.now().strftime("%Y-%m-%d")
    path = notes / (day + "-" + safe_title(args.title) + ".md")
    counter = 2
    while path.exists():
        path = notes / (day + "-" + safe_title(args.title) + "-" + str(counter) + ".md")
        counter += 1
    body = "\n".join(["---", "title: " + args.title.strip(), "time: " + stamp, "project_root: " + str(root), "tags: " + args.tags.strip(), "---", "", content, ""])
    path.write_text(body, encoding="utf-8")
    os.chmod(str(path), 0o600)
    index = project / "index.md"
    existing = index.read_text(encoding="utf-8", errors="replace") if index.exists() else "# Project Long Memory Index\n\n"
    rel = path.relative_to(project)
    line = "- %s [%s](%s) tags=%s - %s" % (stamp, args.title.strip(), rel, args.tags.strip(), re.sub(r"\s+", " ", content)[:240])
    index.write_text(existing.rstrip() + "\n" + line + "\n", encoding="utf-8")
    os.chmod(str(index), 0o600)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
