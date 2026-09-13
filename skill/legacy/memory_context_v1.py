#!/usr/bin/env python3
"""Frozen PLM v1 reader used only while v2 runs in shadow mode."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--query", default="")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    cwd = Path(args.cwd).expanduser().resolve()
    root = root_for(cwd)
    directory = BASE / "projects" / slug(root)
    keywords = set(re.findall(r"[\w\u4e00-\u9fff.-]{2,}", (args.query + " " + root.name + " " + str(root)).lower())) - {"users", "tiger", "project", "memory", "codex", "local"}
    files = []
    for path in (BASE / "global.md", directory / "index.md"):
        if path.exists():
            files.append(path)
    notes = directory / "notes"
    if notes.exists():
        files.extend(sorted(notes.glob("*.md"), reverse=True))
    matches = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")[:20000]
        score = sum(text.lower().count(word) for word in keywords)
        if score == 0 and path.name == "index.md":
            score = 1
        if score:
            matches.append((score, path.stat().st_mtime, path, text))
    matches.sort(reverse=True)
    print("# Project Long Memory Context")
    print("cwd: %s" % cwd)
    print("project_root: %s" % root)
    print("project_memory: %s" % directory)
    print("\n## Project Rule Files")
    rules = []
    for parent in reversed([cwd, *cwd.parents]):
        candidate = parent / "AGENTS.md"
        if candidate.is_file():
            rules.append(candidate)
    if not rules:
        print("(no AGENTS.md files found on the workspace path)")
    for path in rules:
        print("\n### %s" % path)
        print(path.read_text(encoding="utf-8", errors="replace")[:5000].rstrip())
    print("\n## Relevant Memory")
    if not matches:
        print("(no project long-memory notes found yet)")
    for score, _, path, text in matches[:args.limit]:
        print("\n### %s (score %d)" % (path, score))
        lines = text.splitlines()
        selected = []
        for index, line in enumerate(lines):
            if any(word in line.lower() for word in keywords):
                selected.extend(lines[max(0, index - 1):index + 2])
                selected.append("...")
            if len(selected) >= 28:
                break
        print("\n".join((selected or lines[:24])[:32]).strip())
    print("\n## Write Reminder")
    print("If this task creates durable project knowledge, write a short memory note with memory_write.py before finishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
