"""Optional, offline-only model adapters. Importing PLM never imports a model runtime.

The core search API accepts any object implementing ``embed(texts)`` or
``score(query, passages)`` plus ``describe()``. These concrete adapters require
an explicitly supplied local directory; model identifiers are provenance only.
Caches contain numeric outputs and hashes, never plaintext input.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .security import ensure_private_dir


class ProviderUnavailable(RuntimeError):
    """A safe, machine-readable reason to fall back to the lexical profile."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _artifact_fingerprint(directory: Path) -> str:
    """Content identity, including weights: overwriting a model cannot reuse a cache."""
    entries = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append((relative.as_posix(), digest.hexdigest()))
    if not entries:
        raise ProviderUnavailable("empty-local-model")
    return _digest(entries)


def _finite_vector(value: Any, dimensions: Optional[int] = None) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ProviderUnavailable("invalid-model-output")
    try:
        numbers = [float(item) for item in value]
    except (ValueError, TypeError, OverflowError):
        raise ProviderUnavailable("invalid-model-output") from None
    if any(not math.isfinite(number) for number in numbers):
        raise ProviderUnavailable("nonfinite-model-output")
    if dimensions is not None and len(numbers) != dimensions:
        raise ProviderUnavailable("embedding-dimension-mismatch")
    return numbers


class _NumericCache:
    def __init__(self, root: Optional[Path], fingerprint: str):
        # Canonicalize macOS /var -> /private/var once, but never follow a
        # symlink substituted for the explicitly selected cache directory.
        self.root = Path(root).resolve() if root is not None and not Path(root).is_symlink() else None
        self.fingerprint = fingerprint
        # Disk cache makes embeddings reusable across launches.  A single
        # Search process, however, consults the same document embeddings for
        # every query; keep validated numeric values hot so it does not parse
        # thousands of JSON files on every request.
        self._memory: Dict[str, Tuple[float, ...]] = {}

    def _path(self, key: str) -> Optional[Path]:
        return self.root / self.fingerprint / (key + ".json") if self.root is not None else None

    def get(self, key: str, dimensions: int) -> Optional[List[float]]:
        hot = self._memory.get(key)
        if hot is not None:
            try:
                return _finite_vector(hot, dimensions)
            except ProviderUnavailable:
                self._memory.pop(key, None)
        path = self._path(key)
        if path is None:
            return None
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self.fingerprint:
                return None
            values = _finite_vector(payload.get("values"), dimensions)
            self._memory[key] = tuple(values)
            return values
        except (OSError, ValueError, TypeError, AttributeError, ProviderUnavailable):
            return None

    def put(self, key: str, values: Sequence[float]) -> None:
        # Preserve a valid fresh model output even if the optional disk cache
        # is unavailable.  The caller has already validated dimensions and
        # finiteness before reaching this method.
        self._memory[key] = tuple(float(value) for value in values)
        path = self._path(key)
        if path is None:
            return
        # A cache failure must not discard valid model outputs. Safe writes reject
        # symlinks and use a private temporary file followed by atomic rename.
        temporary = None
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                return
            if self.root is not None:
                ensure_private_dir(self.root)
            ensure_private_dir(path.parent)
            descriptor, temporary = tempfile.mkstemp(prefix=".cache-", dir=str(path.parent))
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump({"fingerprint": self.fingerprint, "values": list(values)}, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, str(path))
        except (OSError, ValueError, RuntimeError):
            return
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass


class _LocalModel:
    adapter = ""
    model_class = ""

    def __init__(
        self, model_path: Path, model_id: str = "", revision: str = "",
        cache_dir: Optional[Path] = None, device: str = "cpu", batch_size: int = 16,
    ):
        self.model_path = Path(model_path).expanduser()
        self.model_id = model_id or self.model_path.name
        self.revision = revision
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.dimensions: Optional[int] = None
        self._model: Any = None
        self._fingerprint = ""
        self._artifact_hash = ""
        self._failure = ""
        self._runtime_version = ""
        self._cache: Optional[_NumericCache] = None
        self.cache_hits = 0
        self.cache_misses = 0

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.adapter, "model_id": self.model_id, "revision": self.revision,
            "dimensions": self.dimensions, "fingerprint": self._fingerprint,
            "artifact_fingerprint": self._artifact_hash,
            "runtime_version": self._runtime_version, "device": self.device,
            "local_files_only": True, "trust_remote_code": False,
            "status": "unavailable" if self._failure else ("ready" if self._model is not None else "not_loaded"),
            "fallback_reason": self._failure, "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "settings": self._settings(),
        }

    def _settings(self) -> Dict[str, Any]:
        return {}

    def _load(self) -> Any:
        if self._failure:
            raise ProviderUnavailable(self._failure)
        if self._model is not None:
            return self._model
        try:
            if not self.model_path.is_dir():
                raise ProviderUnavailable("missing-local-model")
            try:
                runtime = importlib.import_module("sentence_transformers")
            except ImportError:
                raise ProviderUnavailable("optional-dependency-missing") from None
            self._runtime_version = str(getattr(runtime, "__version__", "unknown"))
            self._artifact_hash = _artifact_fingerprint(self.model_path)
            model = getattr(runtime, self.model_class)(
                str(self.model_path.resolve()), device=self.device,
                local_files_only=True, trust_remote_code=False, token=False,
                revision=self.revision or None,
            )
            self.dimensions = int(model.get_sentence_embedding_dimension()) if self.adapter == "sentence-transformers" else 1
            if self.dimensions < 1:
                raise ProviderUnavailable("invalid-model-dimension")
            self._fingerprint = _digest({
                "adapter": self.adapter, "cache_schema": 1, "model_id": self.model_id,
                "revision": self.revision, "artifact": self._artifact_hash,
                "dimensions": self.dimensions, "runtime_version": self._runtime_version,
                "normalization": "l2" if self.adapter == "sentence-transformers" else "model-default",
                "settings": self._settings(),
            })
            self._cache = _NumericCache(self.cache_dir, self._fingerprint)
            self._model = model
            return model
        except ProviderUnavailable as exc:
            self._failure = str(exc)
            raise
        except Exception as exc:
            # Model exceptions may contain input text. Expose only the class.
            self._failure = "model-load-failed:" + type(exc).__name__
            raise ProviderUnavailable(self._failure) from None


class LocalSentenceTransformerEmbedding(_LocalModel):
    adapter = "sentence-transformers"
    model_class = "SentenceTransformer"

    def __init__(
        self, model_path: Path, model_id: str = "", revision: str = "",
        cache_dir: Optional[Path] = None, device: str = "cpu", batch_size: int = 16,
        query_prefix: str = "", document_prefix: str = "",
    ):
        super().__init__(model_path, model_id, revision, cache_dir, device, batch_size)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _settings(self) -> Dict[str, Any]:
        return {"query_prefix": self.query_prefix, "document_prefix": self.document_prefix}

    def embed_query(self, query: str) -> List[float]:
        return self.embed([self.query_prefix + query])[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed([self.document_prefix + text for text in texts])

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._load()
        assert self._cache is not None and self.dimensions is not None
        unique = list(dict.fromkeys(texts))
        encoded: Dict[str, List[float]] = {}
        missing = []
        for text in unique:
            cached = self._cache.get(_digest(text), self.dimensions)
            if cached is None:
                missing.append(text)
                self.cache_misses += 1
            else:
                encoded[text] = cached
                self.cache_hits += 1
        if missing:
            try:
                values = model.encode(missing, batch_size=self.batch_size,
                                      normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False)
                if len(values) != len(missing):
                    raise ProviderUnavailable("embedding-count-mismatch")
                validated = [_finite_vector(value, self.dimensions) for value in values]
            except ProviderUnavailable:
                raise
            except Exception as exc:
                raise ProviderUnavailable("embedding-failed:" + type(exc).__name__) from None
            for text, value in zip(missing, validated):
                encoded[text] = value
                self._cache.put(_digest(text), value)
        return [encoded[text] for text in texts]


class LocalCrossEncoderReranker(_LocalModel):
    adapter = "sentence-transformers-cross-encoder"
    model_class = "CrossEncoder"

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        model = self._load()
        assert self._cache is not None
        unique = list(dict.fromkeys(passages))
        scores: Dict[str, float] = {}
        missing = []
        for passage in unique:
            cached = self._cache.get(_digest([query, passage]), 1)
            if cached is None:
                missing.append(passage)
                self.cache_misses += 1
            else:
                scores[passage] = cached[0]
                self.cache_hits += 1
        if missing:
            try:
                values = model.predict([(query, passage) for passage in missing],
                                       batch_size=self.batch_size, show_progress_bar=False,
                                       convert_to_numpy=True)
                if hasattr(values, "tolist"):
                    values = values.tolist()
                if len(values) != len(missing):
                    raise ProviderUnavailable("reranker-count-mismatch")
                values = [value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
                          for value in values]
                validated = _finite_vector(values, len(missing))
            except ProviderUnavailable:
                raise
            except Exception as exc:
                raise ProviderUnavailable("reranker-failed:" + type(exc).__name__) from None
            for passage, value in zip(missing, validated):
                scores[passage] = value
                self._cache.put(_digest([query, passage]), [value])
        return [scores[passage] for passage in passages]
