"""Optional OpenAI-compatible atomic-fact projector for AML ingestion.

The projector never replaces a source message. It creates compact retrieval
projections that carry the source event id; AML dereferences them before
returning evidence.
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
from typing import Any, Dict, List, Optional

from .providers import ProviderUnavailable
from .security import ensure_private_dir

PROMPT_VERSION = 1
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_TIMEOUT_S = 30.0
MAX_FACTS = 6
MAX_KEY_CHARS = 180
MAX_VALUE_CHARS = 600

_SYSTEM_PROMPT = (
    "Extract retrieval projections from one chat message. Preserve only facts "
    "explicitly stated in the message; do not infer, advise, or invent facts."
)
_USER_PROMPT = """Role: {role}
Observed at: {observed_at}
Message: {content}

Return JSON only: {{"facts":[{{"key":"short self-contained subject/attribute", "value":"explicit factual statement", "confidence":0.0}}]}}. Keep at most {limit} facts. Return {{"facts":[]}} when no stable fact is stated."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class LLMFactProjector:
    """A cached, bounded fact projection seam with safe operational fallback."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = DEFAULT_MODEL,
                 cache_dir: Optional[Path] = None, timeout: float = DEFAULT_TIMEOUT_S,
                 env_prefix: str = "OPENAI", allow_default_base_url: bool = True):
        self.api_key = api_key or os.environ.get(env_prefix + "_API_KEY", "")
        fallback_base_url = DEFAULT_BASE_URL if allow_default_base_url else ""
        self.base_url = (base_url or os.environ.get(env_prefix + "_BASE_URL", "") or fallback_base_url).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.cache_dir = (Path(cache_dir).expanduser().resolve()
                          if cache_dir is not None and not Path(cache_dir).expanduser().is_symlink() else None)
        self.timeout = float(timeout)
        self.llm_calls = self.cache_hits = self.failures = 0
        self.total_latency_ms = 0.0

    def describe(self) -> Dict[str, Any]:
        return {"provider": "openai-compatible-fact-projector", "model": self.model,
                "prompt_version": PROMPT_VERSION, "configured": bool(self.api_key and self.base_url),
                "llm_calls": self.llm_calls, "cache_hits": self.cache_hits, "failures": self.failures}

    def _cache_get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / (key + ".json")
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload["facts"] if payload.get("cache_key") == key and isinstance(payload.get("facts"), list) else None
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _cache_put(self, key: str, facts: List[Dict[str, Any]]) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / (key + ".json")
        temporary = None
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                return
            ensure_private_dir(self.cache_dir)
            descriptor, temporary = tempfile.mkstemp(prefix=".fact-cache-", dir=str(self.cache_dir))
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump({"cache_key": key, "facts": facts}, handle, ensure_ascii=False, allow_nan=False)
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

    def _http_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self.api_key,
                                                  "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderUnavailable("http-error:%d" % exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailable("transport-failed:" + type(exc).__name__) from None
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProviderUnavailable("invalid-response-envelope") from None

    def _validate(self, payload: Any) -> List[Dict[str, Any]]:
        rows = payload.get("facts") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ProviderUnavailable("invalid-model-output")
        facts: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = " ".join(str(row.get("key", "")).split())[:MAX_KEY_CHARS]
            value = " ".join(str(row.get("value", "")).split())[:MAX_VALUE_CHARS]
            if not key or not value or (key.casefold(), value.casefold()) in seen:
                continue
            try:
                confidence = float(row.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            seen.add((key.casefold(), value.casefold()))
            facts.append({"key": key, "value": value, "confidence": max(0.0, min(1.0, confidence))})
            if len(facts) >= MAX_FACTS:
                break
        return facts

    def project(self, content: str, role: str, observed_at: str) -> List[Dict[str, Any]]:
        text = " ".join(str(content).split())[:8000]
        if not text:
            return []
        if not self.api_key or not self.base_url:
            raise ProviderUnavailable("missing-credentials")
        key = _digest({"schema": 1, "prompt": PROMPT_VERSION, "model": self.model,
                       "content": text, "role": role, "observed_at": observed_at})
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return self._validate({"facts": cached})
        started = time.perf_counter()
        try:
            response = self._http_post(self.base_url + "/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                             {"role": "user", "content": _USER_PROMPT.format(role=role, observed_at=observed_at,
                                                                                  content=text, limit=MAX_FACTS)}],
            })
            self.llm_calls += 1
            choices = response.get("choices")
            content_out = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None
            if not isinstance(content_out, str):
                raise ProviderUnavailable("invalid-response-envelope")
            facts = self._validate(json.loads(content_out))
        except ProviderUnavailable:
            self.failures += 1
            raise
        except (TypeError, ValueError, KeyError):
            self.failures += 1
            raise ProviderUnavailable("invalid-model-output") from None
        finally:
            self.total_latency_ms += (time.perf_counter() - started) * 1000.0
        self._cache_put(key, facts)
        return facts
