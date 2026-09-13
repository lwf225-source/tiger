"""Optional query-understanding providers (query rewrite / HyDE).

An *optional* enhancement for retrieval.  The LLM provider asks an
OpenAI-compatible chat endpoint for (a) query expansions — synonymous
rephrasings that join multi-channel recall — and (b) a HyDE passage — a
first-person hypothetical recollection whose embedding forms one extra
recall channel (ActiveMemoryIndex-style, AML #3).  The deterministic provider
has the same seam but supplies only one local first-person recall template.

Design contract (mirrors the embedding/reranker provider precedent):

- Standard library only (``urllib``); importing PLM never imports a model
  runtime and never requires network access.  The deterministic provider has
  neither a network nor credential dependency.
- Credentials come exclusively from explicitly supplied arguments or the
  ``KIMI_API_KEY`` / ``KIMI_BASE_URL`` environment variables (or another
  OpenAI-compatible pair supplied by the caller). They are never logged,
  never persisted, and never included in diagnostics.
- Any failure — missing credentials, timeout, HTTP error, malformed or
  invalid model output — raises :class:`ProviderUnavailable`; the search
  layer treats that as "no extra queries" and deterministically degrades
  to the original single-query configuration (the competition safety
  floor). With the provider absent or failing, ids, scores, and ranking are
  identical to the frozen baseline (results carry only a fallback-reason
  marker, the same precedent as the embedding fallback).
- Results are cached on disk (private directory, atomic writes) keyed by a
  hash of (query, model, prompt version, modes): repeated evaluations do
  not re-bill the model. Caches contain the query digest and the rewrite
  output only — never credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .providers import ProviderUnavailable
from .security import ensure_private_dir

PROMPT_VERSION = 1
DEFAULT_MODEL = "k2d6-agent"
DEFAULT_TIMEOUT_S = 30.0
MAX_EXPANSIONS = 3
MAX_EXPANSION_CHARS = 300
MAX_HYDE_CHARS = 1200
MAX_QUERY_CHARS = 4000

_SYSTEM_PROMPT = (
    "You rewrite memory-retrieval queries for a personal conversation memory "
    "system. The memories are past chat messages between the user and others."
)

_USER_PROMPT = """Question: {query}

Return a JSON object with exactly two fields:
- "expansions": up to {n_exp} alternative phrasings of the question —
  synonymous rewrites, entity-focused variants, or decomposed sub-questions.
  Each is a short standalone query string.
- "hyde": a short first-person hypothetical recollection (2-4 sentences) that
  the user might have written in this memory system, containing the answer to
  the question, written as if recalling their own past conversation. Start
  with "I" and stay factual in tone.

Return only the JSON object, no markdown fences, no explanation."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class LLMQueryProvider:
    """OpenAI-compatible query rewrite provider with deterministic degradation.

    ``rewrite(query)`` returns ``{"expansions": [...], "hyde": "..."}``.
    On any operational failure it raises :class:`ProviderUnavailable` — the
    caller (the search layer) then uses the original query only.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        expansions: int = MAX_EXPANSIONS,
        hyde: bool = True,
        env_prefix: str = "KIMI",
    ):
        self.api_key = api_key or os.environ.get(env_prefix + "_API_KEY", "")
        base = base_url or os.environ.get(env_prefix + "_BASE_URL", "")
        self.base_url = base.rstrip("/")
        self.model = model or DEFAULT_MODEL
        # Canonicalize macOS /var -> /private/var once (same precedent as the
        # numeric cache in providers.py); a symlinked cache dir disables caching.
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None and not Path(cache_dir).expanduser().is_symlink()
            else None
        )
        self.timeout = float(timeout)
        self.expansions = max(0, min(int(expansions), MAX_EXPANSIONS))
        self.hyde = bool(hyde)
        self.llm_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.total_latency_ms = 0.0
        self._failure = ""
        if self.expansions == 0 and not self.hyde:
            self._failure = "no-rewrite-modes-enabled"

    # -- diagnostics ------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "provider": "openai-compatible-chat",
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "expansions": self.expansions,
            "hyde": self.hyde,
            "timeout_s": self.timeout,
            "configured": bool(self.api_key and self.base_url),
            "status": "unavailable" if self._failure else "ready",
            "fallback_reason": self._failure,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
        }

    # -- HTTP (single seam for tests to patch) -----------------------------
    def _http_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderUnavailable("http-error:%d" % exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailable("transport-failed:" + type(exc).__name__) from None
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProviderUnavailable("invalid-response-envelope") from None

    # -- cache ---------------------------------------------------------------
    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (key + ".json")

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if path is None:
            return None
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_key") != key:
                return None
            return {"expansions": payload.get("expansions"), "hyde": payload.get("hyde")}
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _cache_put(self, key: str, expansions: Sequence[str], hyde: str) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        temporary = None
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                return
            ensure_private_dir(self.cache_dir)
            descriptor, temporary = tempfile.mkstemp(prefix=".qcache-", dir=str(self.cache_dir))
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump({"cache_key": key, "expansions": list(expansions), "hyde": hyde},
                          handle, ensure_ascii=False, allow_nan=False)
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

    # -- output validation ----------------------------------------------------
    def _validate(self, query: str, payload: Dict[str, Any]) -> Tuple[List[str], str]:
        raw_expansions = payload.get("expansions")
        if raw_expansions is None:
            raw_expansions = []
        if not isinstance(raw_expansions, list):
            raise ProviderUnavailable("invalid-model-output")
        expansions: List[str] = []
        seen = {query.strip().lower()}
        for item in raw_expansions:
            if not isinstance(item, str):
                continue
            text = " ".join(item.split())
            if not text or len(text) > MAX_EXPANSION_CHARS:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            expansions.append(text)
            if len(expansions) >= self.expansions:
                break
        hyde = ""
        if self.hyde:
            raw_hyde = payload.get("hyde")
            if raw_hyde is not None:
                if not isinstance(raw_hyde, str):
                    raise ProviderUnavailable("invalid-model-output")
                hyde = raw_hyde.strip()[:MAX_HYDE_CHARS]
                if hyde.strip().lower() in seen:
                    hyde = ""
        if not expansions and not hyde:
            raise ProviderUnavailable("empty-model-output")
        return expansions, hyde

    # -- main entry -------------------------------------------------------------
    def rewrite(self, query: str) -> Dict[str, Any]:
        """Return ``{"expansions": [...], "hyde": "..."}`` for one query.

        Raises :class:`ProviderUnavailable` on missing credentials, transport
        failure, timeout, or malformed/invalid output; the search layer maps
        that to the original-query-only fallback.
        """
        if self._failure:
            raise ProviderUnavailable(self._failure)
        text = " ".join(str(query).split())[:MAX_QUERY_CHARS]
        if not text:
            raise ProviderUnavailable("empty-query")
        if not self.api_key or not self.base_url:
            self._failure = "missing-credentials"
            raise ProviderUnavailable(self._failure)
        key = _digest({
            "cache_schema": 1, "prompt_version": PROMPT_VERSION, "model": self.model,
            "expansions": self.expansions, "hyde": self.hyde, "query": text,
        })
        cached = self._cache_get(key)
        if cached is not None:
            try:
                expansions, hyde = self._validate(text, {
                    "expansions": cached.get("expansions") or [],
                    "hyde": cached.get("hyde") or "",
                })
            except ProviderUnavailable:
                expansions, hyde = [], ""
            else:
                self.cache_hits += 1
                return {"expansions": expansions, "hyde": hyde}
        started = time.perf_counter()
        try:
            payload = self._http_post(
                self.base_url + "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _USER_PROMPT.format(
                            query=text, n_exp=self.expansions)},
                    ],
                },
            )
            self.llm_calls += 1
        except ProviderUnavailable as exc:
            self.failures += 1
            raise
        finally:
            self.total_latency_ms += (time.perf_counter() - started) * 1000.0
        try:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ProviderUnavailable("invalid-response-envelope")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ProviderUnavailable("invalid-response-envelope")
            body = content.strip()
            if body.startswith("```"):
                body = body.strip("`")
                if body.lower().startswith("json"):
                    body = body[4:]
            parsed = json.loads(body.strip())
            if not isinstance(parsed, dict):
                raise ProviderUnavailable("invalid-model-output")
            expansions, hyde = self._validate(text, parsed)
        except ProviderUnavailable:
            self.failures += 1
            raise
        except (ValueError, AttributeError, TypeError):
            self.failures += 1
            raise ProviderUnavailable("invalid-model-output") from None
        self._cache_put(key, expansions, hyde)
        return {"expansions": expansions, "hyde": hyde}


class DeterministicRecallQueryProvider:
    """Offline first-person recall-query candidate.

    This deliberately does *not* attempt to answer or decompose a question.
    It only changes the embedding register from a question addressed to the
    assistant into a compact first-person recollection, mirroring the
    hypothesis tested by ActiveMemoryIndex.  It has no network or credential
    dependency and conforms to the same ``rewrite`` seam as the LLM provider.
    """

    def __init__(self) -> None:
        # Keep the same operational-statistics surface as LLMQueryProvider so
        # callers can report a provider uniformly.  All values intentionally
        # stay zero: this provider has neither calls nor a cache.
        self.llm_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.total_latency_ms = 0.0

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": "deterministic-first-person-recall",
            "status": "ready",
            "network": False,
            "credentials": False,
        }

    def rewrite(self, query: str) -> Dict[str, Any]:
        text = " ".join(str(query).split())[:MAX_QUERY_CHARS]
        if not text:
            raise ProviderUnavailable("empty-query")
        return {"expansions": [], "hyde": "I remember discussing: " + text}
