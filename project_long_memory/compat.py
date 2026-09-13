from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .model import PLMError
from .service import context_data, find_project_root, migrate_legacy_path, read_config, record_shadow, render_context, write_event


def _skill_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[2], here.parents[1] / "skill"]
    for candidate in candidates:
        if (candidate / "legacy" / "memory_context_v1.py").exists():
            return candidate
    raise PLMError("legacy compatibility scripts are missing")


def _run_legacy(script: str, argv: list) -> subprocess.CompletedProcess:
    path = _skill_root() / "legacy" / script
    return subprocess.run([sys.executable, str(path), *argv], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def context_main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cwd", default=str(Path.cwd()))
    parser.add_argument("--query", default="")
    parser.add_argument("--limit", type=int, default=6)
    known, _ = parser.parse_known_args(argv)
    config = read_config()
    if config["engine"] in {"v1", "shadow"}:
        result = _run_legacy("memory_context_v1.py", argv)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if config["engine"] == "shadow" and result.returncode == 0:
            try:
                start = time.perf_counter()
                project, v2 = context_data(Path(known.cwd), known.query, known.limit, "current")
                latency = (time.perf_counter() - start) * 1000.0
                legacy_refs = re.findall(r"^### (.+?) \(score \d+\)$", result.stdout, re.M)
                record_shadow(project["project_id"], known.query, legacy_refs, [item.source_path for item in v2], latency)
            except Exception:
                pass
        return result.returncode
    sys.stdout.write(render_context(Path(known.cwd), known.query, known.limit, "current", config.get("token_budget", 1800)))
    return 0


def write_main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cwd", default=str(Path.cwd()))
    parser.add_argument("--title", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--content", default="")
    known, _ = parser.parse_known_args(argv)
    config = read_config()
    if config["write_mode"] == "v1":
        result = _run_legacy("memory_write_v1.py", argv)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode == 0 and config["engine"] == "shadow":
            try:
                output_path = Path(result.stdout.strip().splitlines()[-1])
                migrate_legacy_path(output_path)
            except Exception as exc:
                print("memory_write.py: shadow indexing failed: %s" % exc, file=sys.stderr)
        return result.returncode
    content = known.content if known.content else sys.stdin.read()
    event = write_event(Path(known.cwd), known.title, content, tags=known.tags)
    print(event.path)
    return 0
