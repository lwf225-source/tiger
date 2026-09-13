"""Rebase derived legacy references without rewriting immutable Event sources."""
from __future__ import annotations

import json
from pathlib import Path

from .model import PLMError, memory_base
from .security import assert_safe_path


def restored_legacy_path(raw_path: str) -> str:
    """Map an original legacy-note path into the currently selected restored store.

    restore-origin.json records previous store roots, never an arbitrary output
    destination. The only mapping allowed is projects/<bucket>/notes/<note>.md
    under the current memory root. Outside restoration, original paths retain
    their original meaning.
    """
    if not raw_path:
        return ""
    if not isinstance(raw_path, str) or "\x00" in raw_path:
        raise PLMError("invalid legacy source path")
    raw = Path(raw_path)
    if ".." in raw.parts:
        raise PLMError("unsafe legacy source path")
    base = memory_base()
    metadata_path = base / "v2" / "restore-origin.json"
    if not metadata_path.exists():
        return raw_path
    assert_safe_path(metadata_path, base)
    if metadata_path.is_symlink() or metadata_path.stat().st_size > 1024 * 1024:
        raise PLMError("unsafe restore origin metadata")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1 or not isinstance(metadata.get("source_memory_roots", []), list):
            raise ValueError()
        origins = [str(base), metadata.get("source_memory_root", ""), *metadata.get("source_memory_roots", [])]
        for value in origins:
            if not value:
                continue
            if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError()
            try:
                relative = raw.relative_to(Path(value))
            except ValueError:
                continue
            if (len(relative.parts) != 4 or relative.parts[0] != "projects"
                    or relative.parts[2] != "notes" or relative.suffix != ".md"):
                raise PLMError("unsafe restored legacy reference")
            mapped = base / relative
            assert_safe_path(mapped, base)
            return str(mapped)
    except (ValueError, TypeError, OSError, AttributeError):
        raise PLMError("invalid restore origin metadata") from None
    raise PLMError("legacy source does not belong to a recorded restore origin")
