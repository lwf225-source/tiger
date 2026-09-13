"""Agent Memory Leaderboard (AML) Add/Search adapter — a thin, frozen layer.

This module exposes PLM as an AML participant over two synchronous HTTP
operations, ``POST /add`` and ``POST /search`` (plus an unauthenticated
``GET /health``). It is deliberately a *thin* layer: it changes no behavior of
the existing service/search modules, adds no third-party dependency, and never
generates answers — Search returns ranked original evidence only.

Wire contract source: the official AML Add/Search API guide
(https://agentmemories.ai/api-guide), cross-checked against the public
evaluation repository (AML-memory/agent-memory-leaderboard) and the published
Top-10 submission ``flowgrid-aml-retriever`` (docs/API_CONTRACT.md), whose
validation matrix matches the official guide.

Isolation model
---------------
``user_id`` is the only retrieval scope used by the platform. The adapter maps
each ``user_id`` to its own on-disk storage directory
``<PROJECT_LONG_MEMORY_DIR>/aml_users/<slug>--<hash>/`` carrying a
``pyproject.toml`` marker, so ``find_project_root`` resolves it as a distinct
PLM project root (deterministic ``project_id`` per user) without requiring
git. As a second, independent isolation layer every event is written with
``scope="session", scope_id=<user_id>`` and every search filters on exactly
that scope pair — even if project-root resolution ever collapsed, the SQL
scope filter still cannot cross users.

Frozen submission configuration
-------------------------------
The ``AML_*`` constants below are the competition freeze point: change them
only with a new recorded version, never silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import service
from .model import memory_base
from .narrative_time import mentioned_date_tags
from .security import ensure_private_dir

# ---------------------------------------------------------------------------
# Frozen AML submission configuration (see docs/AML_SUBMISSION.md).
# ---------------------------------------------------------------------------
AML_ADAPTER_VERSION = "1.3.4"
AML_PROFILE = "lexical"          # retrieval profile: lexical | passages | legacy
AML_ADJACENT = 2                 # recorded_at neighbours appended per direct hit (0=off)
AML_RECENCY_BOOST = True         # in-memory date-term injection + micro recency tie-break
AML_EXCERPT_FALLBACK = False     # render-context-only switch; AML returns raw evidence bodies
AML_TOP_K_MAX = 100              # official evaluation top_k; larger requests clamp, never fail
AML_VIEW = "current"             # memory view: current | history | both
AML_OPTIONS_AS_QUERY_TERMS = True  # choice-question options join the recall query (never replace it)
AML_MAX_REQUEST_BYTES = 32 * 1024 * 1024
AML_SOURCE = "aml"

# Optional hybrid recall (plm-aml 1.1.0). ``local`` adds the search module's
# existing neural channels (embedding hybrid + optional cross-encoder rerank)
# on top of the unchanged lexical profile. Models come exclusively from
# explicitly configured local directories — the adapter never downloads
# anything and never sends memory text to a service. A missing or failing
# model deterministically degrades to the pure lexical configuration (the
# competition safety floor); the lexical channels themselves are untouched.
#
# Frozen after the LoCoMo re-test (.codex_work/locomo/LOCOMO_HYBRID_REPORT.md):
# BGE-small-en-v1.5 hybrid beat the lexical baseline on the primary subset
# (ER 0.8073 vs 0.7557, win 187 / tie 1751 / loss 39) and beat MiniLM-L12
# full-set; the mmarco cross-encoder reranker was a net negative in the
# current ten-session BGE pairing (overall -0.010844, temporal -0.030413,
# 3.4x mean latency) and stays off.
AML_EMBEDDING = "local"          # hybrid embedding channel: off | local
AML_EMBEDDING_MODEL_DIR = ""     # explicit local SentenceTransformer dir (env PLM_AML_EMBEDDING_MODEL_DIR overrides)
AML_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"  # provenance only; identity is the on-disk artifact hash
AML_EMBEDDING_QUERY_PREFIX = ""  # exact setting used by the full real-BGE baseline
AML_EMBEDDING_DOCUMENT_PREFIX = ""
AML_RERANKER = "off"             # cross-encoder rerank channel: off | local
AML_RERANKER_MODEL_DIR = ""      # explicit local CrossEncoder dir (env PLM_AML_RERANKER_MODEL_DIR overrides)
AML_RERANKER_MODEL_ID = ""       # provenance only

# Multi-hop entity expansion + MMR complementary selection (plm-aml 1.2.0
# candidates). Both are opt-in and frozen OFF until the LoCoMo re-test
# (.codex_work/locomo/) shows a measured gain; see docs/BATCH6_REPORT.md.
# ``AML_ENTITY_EXTRACTION`` is the ingest-side half: the platform Add path
# carries no entity metadata, so a deterministic, dependency-free proper-noun
# tagger fills event_entities/entities at write time (search-time expansion
# reads only those tables). It changes no search channel — FTS/vectors content
# is byte-identical with or without it.
AML_ENTITY_EXTRACTION = False    # ingest-time deterministic entity tagging
AML_EXPAND_ENTITIES = 0          # entity hops appended as supplementary evidence (0=off, max 2)
AML_MMR = False                  # incremental MMR over the ranking head

# Abstention floor (plm-aml 1.2.0 candidate). When > 0, candidates whose best
# absolute channel cosine falls below the threshold are withheld; if nothing
# survives, Search answers ``{"data": [], "plm_abstained": true}`` — the empty
# array is the contract-sanctioned abstention signal and the extra key is a
# non-standard observability field the platform ignores (flowgrid precedent:
# undeclared fields are tolerated). The threshold is calibrated on the frozen
# BGE-small-en-v1.5 cosine scale (docs/BATCH7_REPORT.md) and is meaningless for
# other models, so it ships OFF until the LoCoMo adversarial re-test justifies
# freezing a value.
AML_ABSTAIN_THRESHOLD = 0.0      # relevance floor for withholding weak evidence (0=off)

# Write-side narrative-time tagging (batch 11 candidate). Batch 10 measured
# that query-side valid_at filtering cannot rescue LoCoMo temporal questions:
# their anchors point at *narrative* time (dates mentioned in the message
# body), not record time. When this switch is on, ``add()`` attaches the
# normalized surface forms of every safely parseable date mention
# (``narrative_time.mentioned_date_tags`` — deterministic, zero-dependency) as
# event *tags*. The immutable body is never touched, so Search evidence stays
# verbatim; tags already flow into every index text channel (FTS tags column,
# scoped substring field, stored n-gram vector, embedding passage text), so a
# date-anchored query hits narrative-time mentions with no search-layer
# change. Frozen OFF until the LoCoMo temporal-subset re-test shows a measured
# gain without regression elsewhere; OFF writes are byte-identical to 1.1.0.
AML_NARRATIVE_TIME = False   # ingest-time narrative-time date tagging

# Query-side temporal intent (batch 10 candidate; the service layer gained the
# opt-in ``temporal_intent`` switch that runs the deterministic extractor in
# ``temporal_intent.py`` over the query — an absolute as-of anchor fills
# ``valid_at``, comparison phrasing enables ``soft_supersede``; see
# docs/BATCH10_REPORT.md). Frozen OFF until the LoCoMo temporal-subset re-test
# shows a measured gain without non-temporal regression; when on, the response
# carries a ``plm_temporal_intent`` observability key (undeclared field, the
# platform ignores it — flowgrid precedent).
AML_TEMPORAL_INTENT = False      # query-side temporal intent extraction -> auto bitemporal/soft-supersede

# Retrieval window composition fix + query-side coverage re-ranking (batch 12
# candidates, plm-aml 1.2.0 candidates). Batch 6 established that under the
# AML protocol (top_k=100, caller slices results[:top_k]) ``AML_ADJACENT=2``
# is a no-op: neighbours are appended *beyond* the window and never measured.
# ``AML_WINDOW_RESERVE`` reserves up to a quarter of the window's tail slots
# for adjacent extras (seed-rank order), displacing only the lowest-ranked
# direct hits — making adjacent actually visible under the protocol.
# ``AML_COVERAGE_RERANK`` greedily reorders the ranking head to reward
# candidates covering rare (corpus-IDF) query content words not yet covered
# by the already-selected set (aml-memory-mvp-style query-side coverage, the
# complement of result-side MMR). Both are ranking-layer only, zero new
# dependencies, and frozen OFF until the LoCoMo re-test
# (.codex_work/locomo/, docs/BATCH12_REPORT.md) shows a measured gain without
# regression elsewhere; OFF behavior is byte-identical to 1.1.0.
AML_WINDOW_RESERVE = False       # adjacent extras occupy reserved tail slots of the window
AML_COVERAGE_RERANK = False      # query-side rare information-word coverage re-ranking

# Candidate source-derived session views (inspired by FlowGrid's traceable
# sliding-window retrieval).  With the candidate enabled, Add stores only a
# deterministic digest of the external session id as an internal tag. Search
# can then score a bounded three-message view in memory while returning the
# matched source event's original body, never a derived window. It is kept off
# until paired data shows a gain; the normal 1.2.0 result shape and ranking
# remain unchanged.
AML_SESSION_VIEWS = False
# ReFind-style session-level RRF.  This shares the one-way session digest used
# by the view candidate but only reorders existing source-event candidates; it
# never returns synthesized windows or the raw platform session_id.
# Batch 20 established a full source-session paired gain; it is frozen on in 1.3.1.
AML_SESSION_RRF = True
_AML_SESSION_TAG_PREFIX = "aml-session:"

# Query understanding (batch 13 candidate, plm-aml 1.2.0 candidate). When
# "llm", Search first asks an OpenAI-compatible chat endpoint
# (``query_provider.LLMQueryProvider``, standard library only) for query
# expansions — each becomes an extra FTS + substring recall channel — and a
# first-person HyDE recollection — one extra embedding channel on top of the
# frozen BGE hybrid. Every extra channel is RRF-fused and can only add
# candidates. Credentials come exclusively from the KIMI_API_KEY /
# KIMI_BASE_URL environment variables (never hard-coded, never logged, never
# persisted); the AML protocol permits LLM use in Search (Refind #2
# precedent), and docs/AML_SUBMISSION.md discloses the model dependency.
# Missing credentials, timeouts, or malformed output deterministically degrade
# to the original single-query configuration — the competition safety floor:
# the ``data`` payload is byte-identical to the no-rewrite configuration and the response gains one
# ignorable observability key (flowgrid precedent). Rewrite results are cached on disk under
# ``memory_base()/aml_query_cache`` keyed by a hash of (query, model, prompt
# version), so replays do not re-bill the model. Frozen OFF until the LoCoMo
# re-test (.codex_work/locomo/, docs/BATCH13_REPORT.md) shows a measured gain
# without regression elsewhere. ``recall-template`` is a separate offline
# path: it supplies one deterministic first-person embedding query, uses no
# network or credentials, and is frozen ON after the clean paired evaluation
# documented in docs/BATCH14_REPORT.md.
AML_QUERY_REWRITE = "recall-template"  # frozen offline query enhancement
AML_QUERY_MODEL = "k2d6-agent"   # chat model for rewrites (env PLM_AML_QUERY_MODEL overrides)
AML_QUERY_ENV_PREFIX = "KIMI"     # credential namespace for optional LLM rewrite
AML_QUERY_EXPANSIONS = 3         # synonymous rewrites per query (0 disables the expansion channels)
AML_QUERY_HYDE = True            # first-person HyDE recollection channel

# Candidate write-time retrieval projections (plm-aml 1.2.0 candidate).
# Facts are sourced to immutable original events and dereferenced back to the
# original message by AML Search. Keep OFF until a full paired evaluation.
AML_FACT_PROJECTION = "off"      # off | llm
AML_FACT_MODEL = "gpt-4o-mini"   # official AML model when this candidate is measured
# Credential namespace for the optional projection provider.  Code-route
# deployments can inject a platform-owned OpenAI-compatible endpoint without
# coupling the adapter to a developer's personal OpenAI account.
# Deliberately separate from a developer's common OPENAI_* process variables.
# A deployment must explicitly select OPENAI only when the platform documents
# that namespace and is authorized to provide it.
AML_FACT_ENV_PREFIX = "PLM_AML_PLATFORM"  # env PLM_AML_FACT_ENV_PREFIX overrides

# Write-time near-duplicate detection (batch 9 candidate; the service layer
# gained opt-in ``consolidate(dedup_similarity=..., dedup_action=...)`` with a
# conservative human-review default — see docs/BATCH9_REPORT.md). This constant
# is deliberately 0=off and is NOT wired into ``add()``: the Add path ingests
# each platform chunk verbatim as an immutable episode, and Add's 200 promises
# the chunk is durable and searchable. Silently dropping or merging a
# near-duplicate episode would destroy original evidence text that Search is
# scored on; near-duplicate governance belongs to the candidate/consolidate
# path (agent-suggested facts with human review), not to verbatim ingest.
AML_ADD_DEDUP_SIMILARITY = 0.0   # Add-path write-time dedup threshold (0=off, unwired by design)

_MARKER_NAME = "pyproject.toml"  # PLM project-root indicator file
_MARKER_BODY = '[project]\nname = "aml-user-storage"\nversion = "0"\n'

_MODEL_SWITCHES = {"off", "local"}
# Process-wide singletons: a model loads once and is shared by every user; a
# failed build caches ``None`` so every later search degrades identically.
_MODEL_SINGLETONS: Dict[str, Any] = {}


def _model_dir(constant: str, env_name: str) -> str:
    return os.environ.get(env_name, "").strip() or constant.strip()


def _credential_env_prefix(override_env: str, configured: str, label: str) -> str:
    """Return a validated explicit namespace for one optional LLM provider."""
    prefix = os.environ.get(override_env, "").strip() or configured
    prefix = prefix.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", prefix):
        raise ValueError("AML %s credential namespace is invalid" % label)
    return prefix


def _query_env_prefix() -> str:
    """Return the explicit credential namespace for the optional query LLM."""
    return _credential_env_prefix("PLM_AML_QUERY_ENV_PREFIX", AML_QUERY_ENV_PREFIX, "query")


def _fact_env_prefix() -> str:
    """Return the credential namespace for optional fact projection."""
    return _credential_env_prefix("PLM_AML_FACT_ENV_PREFIX", AML_FACT_ENV_PREFIX, "fact projection")


def validate_model_config() -> List[str]:
    """Startup check: raise on a switch typo, warn (never crash) on a missing dir.

    An invalid switch value is a frozen-config programming error and fails
    loudly before serving. A missing model directory is an operational
    condition: the adapter must still serve, deterministically degraded to the
    lexical safety floor — a crashed endpoint costs the whole evaluation.
    """
    warnings_out: List[str] = []
    if AML_QUERY_REWRITE not in {"off", "llm", "recall-template"}:
        raise ValueError("AML_QUERY_REWRITE must be one of ['llm', 'off', 'recall-template'], got %r" % (AML_QUERY_REWRITE,))
    query_prefix = _query_env_prefix()
    query_base_ready = bool(os.environ.get(query_prefix + "_BASE_URL")) or query_prefix == "OPENAI"
    if AML_QUERY_REWRITE == "llm" and not (os.environ.get(query_prefix + "_API_KEY") and query_base_ready):
        warnings_out.append(
            "AML_QUERY_REWRITE=llm but %s_API_KEY/%s_BASE_URL are not set; "
            % (query_prefix, query_prefix)
            +
            "serving degraded to the frozen single-query configuration")
    if AML_FACT_PROJECTION not in {"off", "llm"}:
        raise ValueError("AML_FACT_PROJECTION must be one of ['llm', 'off'], got %r" % (AML_FACT_PROJECTION,))
    fact_prefix = _fact_env_prefix()
    fact_base_ready = bool(os.environ.get(fact_prefix + "_BASE_URL")) or fact_prefix == "OPENAI"
    if AML_FACT_PROJECTION == "llm" and not (os.environ.get(fact_prefix + "_API_KEY") and fact_base_ready):
        warnings_out.append(
            "AML_FACT_PROJECTION=llm but %s_API_KEY/%s_BASE_URL are not set; "
            % (fact_prefix, fact_prefix)
            + "serving without projections")
    for name, value in (("AML_EMBEDDING", AML_EMBEDDING), ("AML_RERANKER", AML_RERANKER)):
        if value not in _MODEL_SWITCHES:
            raise ValueError("%s must be one of %s, got %r" % (name, sorted(_MODEL_SWITCHES), value))
        if value == "local":
            constant, env_name = (
                (AML_EMBEDDING_MODEL_DIR, "PLM_AML_EMBEDDING_MODEL_DIR") if name == "AML_EMBEDDING"
                else (AML_RERANKER_MODEL_DIR, "PLM_AML_RERANKER_MODEL_DIR")
            )
            if not _model_dir(constant, env_name):
                warnings_out.append(
                    "%s=local but no model dir configured (%s or env %s); serving degraded as pure lexical"
                    % (name, name + "_MODEL_DIR", env_name))
    return warnings_out


def _model_provider(role: str) -> Any:
    """Build the optional local model once; any failure degrades to ``None``.

    ``None`` means the search call runs the pure lexical configuration — the
    deterministic competition safety floor. Build problems (missing directory,
    missing optional dependency, unloadable weights) never propagate.
    """
    switch = AML_EMBEDDING if role == "embedding" else AML_RERANKER
    if switch != "local":
        return None
    if role in _MODEL_SINGLETONS:
        return _MODEL_SINGLETONS[role]
    provider = None
    try:
        from .providers import LocalCrossEncoderReranker, LocalSentenceTransformerEmbedding
        cache_dir = memory_base() / "aml_model_cache"
        if role == "embedding":
            model_dir = _model_dir(AML_EMBEDDING_MODEL_DIR, "PLM_AML_EMBEDDING_MODEL_DIR")
            if not model_dir:
                raise ValueError("AML_EMBEDDING=local without a model dir")
            provider = LocalSentenceTransformerEmbedding(
                Path(model_dir), model_id=AML_EMBEDDING_MODEL_ID, cache_dir=cache_dir,
                query_prefix=AML_EMBEDDING_QUERY_PREFIX,
                document_prefix=AML_EMBEDDING_DOCUMENT_PREFIX,
            )
            provider.embed(["aml-model-warmup"])  # force the lazy load now
        else:
            model_dir = _model_dir(AML_RERANKER_MODEL_DIR, "PLM_AML_RERANKER_MODEL_DIR")
            if not model_dir:
                raise ValueError("AML_RERANKER=local without a model dir")
            provider = LocalCrossEncoderReranker(
                Path(model_dir), model_id=AML_RERANKER_MODEL_ID, cache_dir=cache_dir,
            )
            provider.score("aml-model-warmup", ["warmup"])
    except Exception:
        provider = None
    _MODEL_SINGLETONS[role] = provider
    return provider


def _query_provider_singleton() -> Any:
    """Build the optional query provider once; any failure degrades to None.

    ``None`` means Search runs the frozen single-query configuration — the
    deterministic competition safety floor. The ``llm`` variant reads an
    explicit environment namespace (KIMI by default, OPENAI when selected)
    only; the ``recall-template`` variant is local and credential-free.
    """
    if AML_QUERY_REWRITE == "off":
        return None
    if "query_rewrite" in _MODEL_SINGLETONS:
        return _MODEL_SINGLETONS["query_rewrite"]
    provider = None
    try:
        if AML_QUERY_REWRITE == "recall-template":
            from .query_provider import DeterministicRecallQueryProvider
            provider = DeterministicRecallQueryProvider()
        else:
            from .query_provider import LLMQueryProvider
            env_prefix = _query_env_prefix()
            base_url = os.environ.get(env_prefix + "_BASE_URL", "").strip()
            if not base_url and env_prefix == "OPENAI":
                base_url = "https://api.openai.com"
            provider = LLMQueryProvider(
                model=os.environ.get("PLM_AML_QUERY_MODEL", "").strip() or AML_QUERY_MODEL,
                base_url=base_url,
                cache_dir=memory_base() / "aml_query_cache",
                expansions=AML_QUERY_EXPANSIONS,
                hyde=AML_QUERY_HYDE,
                env_prefix=env_prefix,
            )
            if not provider.describe().get("configured"):
                provider = None  # no credentials: serve the frozen baseline
    except Exception:
        provider = None
    _MODEL_SINGLETONS["query_rewrite"] = provider
    return provider


def _fact_projector_singleton() -> Any:
    """Build the optional write-time projector; absent credentials mean off."""
    if AML_FACT_PROJECTION != "llm":
        return None
    if "fact_projection" in _MODEL_SINGLETONS:
        return _MODEL_SINGLETONS["fact_projection"]
    provider = None
    try:
        from .fact_provider import LLMFactProjector
        env_prefix = _fact_env_prefix()
        base_url = os.environ.get(env_prefix + "_BASE_URL", "").strip()
        if not base_url and env_prefix == "OPENAI":
            base_url = "https://api.openai.com"
        provider = LLMFactProjector(
            model=os.environ.get("PLM_AML_FACT_MODEL", "").strip() or AML_FACT_MODEL,
            base_url=base_url,
            cache_dir=memory_base() / "aml_fact_cache",
            env_prefix=env_prefix,
            allow_default_base_url=(env_prefix == "OPENAI"),
        )
        if not provider.describe().get("configured"):
            provider = None
    except Exception:
        provider = None
    _MODEL_SINGLETONS["fact_projection"] = provider
    return provider


def model_status() -> Dict[str, Any]:
    """Ops/test snapshot: configured switches and whether models are live."""
    status: Dict[str, Any] = {"adapter_version": AML_ADAPTER_VERSION}
    status["add_dedup"] = {"threshold": AML_ADD_DEDUP_SIMILARITY, "wired": False}
    status["query_rewrite"] = {"switch": AML_QUERY_REWRITE,
                               "model": AML_QUERY_MODEL if AML_QUERY_REWRITE == "llm" else "",
                               "active": _query_provider_singleton() is not None
                                         if AML_QUERY_REWRITE == "llm" else False}
    for role, switch in (("embedding", AML_EMBEDDING), ("reranker", AML_RERANKER)):
        entry: Dict[str, Any] = {"switch": switch}
        if switch == "local":
            provider = _model_provider(role)
            entry["active"] = provider is not None
            if provider is not None:
                description = provider.describe()
                entry["model_id"] = description.get("model_id", "")
                entry["artifact_fingerprint"] = description.get("artifact_fingerprint", "")
            else:
                entry["degraded_to"] = "lexical"
        status[role] = entry
    return status


class ContractError(ValueError):
    """Request violates the published Add/Search wire contract (HTTP 422)."""


# ---------------------------------------------------------------------------
# Deterministic entity extraction (opt-in ingest half of entity expansion).
# ---------------------------------------------------------------------------
# Proper-noun heuristic for English dialogue (the AML/LoCoMo traffic): capital-
# ized token runs of 2+ letters, up to three tokens. It is deliberately simple
# and deterministic — no model, no network, no third-party dependency. Common
# sentence-initial words and calendar terms are stoplisted to keep the graph
# sparse; a few false positives are harmless because expansion is capped and
# supplementary (never displaces a direct hit).
_ENTITY_RUN = re.compile(r"[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,}){0,2}")
_ENTITY_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "There", "Their", "They", "Them",
    "He", "Her", "His", "She", "We", "Our", "You", "Your", "It", "Its",
    "And", "But", "Or", "So", "If", "In", "On", "At", "As", "To", "For",
    "When", "Then", "Than", "What", "Why", "How", "Who", "Which", "Where",
    "Yes", "No", "Not", "Yeah", "Yep", "Nope", "Well", "Oh", "Ah", "Ok", "Okay",
    "Sure", "Thanks", "Thank", "Hi", "Hello", "Hey", "Bye", "Good", "Great",
    "Maybe", "Actually", "Really", "Just", "Also", "Still", "Even", "Now",
    "Here", "Let", "Like", "Wow", "Haha", "Right", "Exactly", "Absolutely",
    "Today", "Tomorrow", "Yesterday", "Tonight", "Morning", "Afternoon", "Evening",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "I", "A", "An",
})
AML_MAX_ENTITIES_PER_MESSAGE = 8


def _extract_entities(content: str) -> List[Dict[str, Any]]:
    """Deterministic proper-noun tagging; never raises on odd input."""
    found: List[str] = []
    seen = set()
    for match in _ENTITY_RUN.finditer(content[:16000]):
        name = match.group(0)
        words = name.split()
        if all(word in _ENTITY_STOPWORDS for word in words):
            continue
        if len(name) < 3:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(name)
        if len(found) >= AML_MAX_ENTITIES_PER_MESSAGE:
            break
    return [{"name": name, "type": "name", "relation": "mentions", "aliases": []} for name in found]


def _require_text(payload: Dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError("%s is required and must be a non-empty string" % field)
    return value


def _timestamp_ms(value: Any) -> Optional[int]:
    """Validate an optional Unix-millisecond timestamp (lossless floats accepted)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContractError("messages[].timestamp must be an integer in Unix milliseconds")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ContractError("messages[].timestamp must be an integer in Unix milliseconds")


def _session_tag(user_id: str, session_id: str) -> str:
    """Return a non-reversible, user-bound grouping tag for retrieval views.

    The tag never leaves the Search response and avoids persisting the
    platform's raw session identifier in event metadata merely to form an
    optional local retrieval view.
    """
    digest = hashlib.sha256((user_id + "\0" + session_id).encode("utf-8")).hexdigest()[:24]
    return _AML_SESSION_TAG_PREFIX + digest


def _iso_from_ms(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    try:
        parsed = datetime.fromtimestamp(ms / 1000.0, timezone.utc)
    except (OverflowError, OSError, ValueError):
        # Out-of-range platform timestamps must not fail the whole chunk; the
        # event falls back to the service default (recorded now).
        return None
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def user_storage_dir(user_id: str) -> Path:
    """Deterministic per-user project root under the PLM memory base."""
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", user_id).strip("-._")[:40] or "user"
    return memory_base() / "aml_users" / (slug + "--" + digest)


def ensure_user_storage(user_id: str) -> Path:
    path = user_storage_dir(user_id)
    ensure_private_dir(path)
    marker = path / _MARKER_NAME
    if not marker.exists():
        marker.write_text(_MARKER_BODY, encoding="utf-8")
    return path


def _validate_add(payload: Any) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ContractError("request body must be a JSON object")
    request_id = _require_text(payload, "request_id")
    user_id = _require_text(payload, "user_id")
    session_id = _require_text(payload, "session_id")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractError("messages is required and must be a non-empty array")
    validated: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ContractError("messages[] elements must be objects")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ContractError("messages[].role is required and must be a non-empty string")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ContractError("messages[].content is required and must be a non-empty string")
        validated.append({"role": role, "content": content, "timestamp": _timestamp_ms(message.get("timestamp"))})
    return request_id, user_id, session_id, validated


def add(request: Any) -> Dict[str, Any]:
    """Synchronously persist one platform chunk; returns only after searchable.

    Idempotency: the key is (user_id, request_id, message_index). A replayed
    request returns the same success response without duplicating events, and
    a changed payload under the same request_id is first-write-wins — the
    service layer returns the originally stored event for an existing key.
    Any genuine write failure propagates so the transport answers non-200:
    Add's 200 is the promise that the memories are durable and searchable.
    """
    request_id, user_id, session_id, messages = _validate_add(request)
    root = ensure_user_storage(user_id)
    projector = _fact_projector_singleton()
    for index, message in enumerate(messages):
        stamp = _iso_from_ms(message["timestamp"])
        day = stamp[:10] if stamp else "undated"
        entities = _extract_entities(message["content"]) if AML_ENTITY_EXTRACTION else None
        tags = [AML_SOURCE]
        if AML_SESSION_VIEWS or AML_SESSION_RRF:
            tags.append(_session_tag(user_id, session_id))
        if AML_NARRATIVE_TIME:
            # Write-side narrative-time tags: metadata only, body untouched.
            tags += mentioned_date_tags(message["content"])
        event = service.write_event(
            root,
            "%s · %s" % (message["role"], day),
            message["content"],
            tags=tags,
            kind="episode",
            scope="session",
            scope_id=user_id,
            observed_at=stamp,
            recorded_at=stamp,
            source=AML_SOURCE,
            created_by="aml-adapter",
            idempotency_key="aml:%s:%s:%d" % (user_id, request_id, index),
            extra={"entities": entities} if entities else None,
        )
        if projector is not None:
            try:
                facts = projector.project(message["content"], message["role"], stamp)
                for fact_index, fact in enumerate(facts):
                    service.write_fact(
                        root, fact["key"], fact["value"], tags=[AML_SOURCE, "aml-projection"],
                        source_event_id=event.metadata["id"], source=AML_SOURCE,
                        # Bind projection identity to the immutable source
                        # event as well as the platform request. This prevents
                        # a stale first-write projection from pointing at a
                        # later re-created event in a reused local store.
                        idempotency_key="aml:projection:%s:%s:%s:%d" %
                        (user_id, event.metadata["id"], request_id, fact_index),
                        confidence=fact["confidence"], scope="session", scope_id=user_id,
                        observed_at=stamp,
                    )
            except Exception:
                # Add's hard contract covers durable original evidence; a
                # candidate projection must never make that evidence fail.
                pass
    return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}


def _validate_search(payload: Any) -> Tuple[str, str, int, List[str]]:
    if not isinstance(payload, dict):
        raise ContractError("request body must be a JSON object")
    query = _require_text(payload, "query")
    user_id = _require_text(payload, "user_id")
    top_k = payload.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ContractError("top_k is required and must be an integer")
    if top_k < 0:
        raise ContractError("top_k must be >= 0")
    options = payload.get("options")
    if options is None:
        options = []
    if not isinstance(options, list) or any(not isinstance(option, str) for option in options):
        raise ContractError("options must be an array of strings")
    return query, user_id, min(top_k, AML_TOP_K_MAX), options


def search(request: Any) -> Dict[str, Any]:
    """Return ranked original evidence for exactly one user; never an answer.

    Retrieval failures degrade to ``200 {"data": []}`` — a contract failure
    costs the whole evaluation stage, while an empty result costs one
    question. Validation errors still raise ContractError (HTTP 422).
    """
    query, user_id, top_k, options = _validate_search(request)
    if top_k == 0 or not user_storage_dir(user_id).is_dir():
        return {"data": []}
    recall_query = query
    if AML_OPTIONS_AS_QUERY_TERMS and options:
        recall_query = query + "\n" + "\n".join(options)
    try:
        diagnostics: Dict[str, Any] = {}
        _project, results = service.context_data(
            user_storage_dir(user_id),
            recall_query,
            # Fact projections can dereference to the same source event. Ask
            # for a bounded wider candidate set, then deduplicate source ids
            # before applying the protocol's requested top_k.
            limit=top_k * 2 if AML_FACT_PROJECTION == "llm" else top_k,
            view=AML_VIEW,
            scope="session",
            scope_id=user_id,
            retrieval_profile=AML_PROFILE,
            adjacent=AML_ADJACENT,
            recency_boost=AML_RECENCY_BOOST,
            expand_entities=AML_EXPAND_ENTITIES,
            mmr=AML_MMR,
            abstain_threshold=AML_ABSTAIN_THRESHOLD,
            temporal_intent=AML_TEMPORAL_INTENT,
            window_reserve=AML_WINDOW_RESERVE,
            coverage_rerank=AML_COVERAGE_RERANK,
            session_views=AML_SESSION_VIEWS,
            session_rrf=AML_SESSION_RRF,
            diagnostics=diagnostics,
            embedding_provider=_model_provider("embedding"),
            reranker=_model_provider("reranker"),
            query_provider=_query_provider_singleton(),
        )
        if AML_FACT_PROJECTION != "llm":
            # Preserve the frozen 1.2.0 payload exactly when the optional fact
            # projection candidate is disabled, including independently-created Facts.
            data = [
                {"id": result.ref_id, "content": result.body, "score": result.score,
                 "created_at": result.valid_from}
                for result in results[:top_k]
                if isinstance(result.body, str) and result.body.strip()
            ]
        else:
            data = []
            seen_event_ids = set()
            for result in results:
                source = None
                if result.record_type == "fact" and result.source_event_id:
                    source = service.source_event_data(user_storage_dir(user_id), result.source_event_id, "session", user_id)
                    if source is None:
                        continue
                event_id = source["event_id"] if source else result.ref_id
                body = source["body"] if source else result.body
                if event_id in seen_event_ids or not isinstance(body, str) or not body.strip():
                    continue
                seen_event_ids.add(event_id)
                data.append({"id": event_id, "content": body, "score": result.score,
                             "created_at": source["observed_at"] if source else result.valid_from})
                if len(data) >= top_k:
                    break
        if AML_ABSTAIN_THRESHOLD > 0 and not data:
            # Explicit abstention signal: evidence withheld for insufficient
            # relevance (also covers "nothing matched at all"). Non-standard
            # observability key; the platform ignores undeclared fields.
            return {"data": [], "plm_abstained": True,
                    "plm_abstain": dict(diagnostics.get("abstain", {}))}
    except Exception:
        return {"data": []}
    if AML_TEMPORAL_INTENT:
        intent_info = diagnostics.get("temporal_intent", {})
        if intent_info.get("has_temporal_intent"):
            # Non-standard observability key; the platform ignores undeclared
            # fields (same precedent as ``plm_abstained``).
            return {"data": data, "plm_temporal_intent": dict(intent_info)}
    if AML_QUERY_REWRITE == "llm":
        # Non-standard observability key (model, cache/call counters, fallback
        # reason — never credentials or query text); the platform ignores
        # undeclared fields (flowgrid precedent).
        return {"data": data, "plm_query_rewrite": dict(diagnostics.get("query_rewrite", {}))}
    return {"data": data}


# ---------------------------------------------------------------------------
# HTTP transport (standard library only).
# ---------------------------------------------------------------------------

def _auth_token() -> str:
    return os.environ.get("AML_API_KEY", "")


class AMLRequestHandler(BaseHTTPRequestHandler):
    server_version = "plm-aml/" + AML_ADAPTER_VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Privacy obligation: never log request bodies or memory content.
        return

    # -- helpers ---------------------------------------------------------
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = _auth_token()
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[len("Bearer "):] == token:
            return True
        if header.startswith("Token ") and header[len("Token "):] == token:
            return True
        return self.headers.get("X-Api-Key", "") == token

    def _read_json(self) -> Any:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else 0
        except ValueError:
            raise ContractError("invalid Content-Length")
        if length <= 0:
            raise ContractError("request body is required")
        if length > AML_MAX_REQUEST_BYTES:
            raise ContractError("request body too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ContractError("request body must be valid JSON") from exc

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "version": AML_ADAPTER_VERSION})
        else:
            self._send_json(404, {"detail": {"reason": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path not in ("/add", "/search"):
            self._send_json(404, {"detail": {"reason": "not found"}})
            return
        if not self._authorized():
            self._send_json(401, {"detail": {"reason": "unauthorized"}})
            return
        try:
            payload = self._read_json()
        except ContractError as exc:
            self._send_json(400, {"detail": {"reason": str(exc)}})
            return
        if path == "/add":
            try:
                result = add(payload)
            except ContractError as exc:
                self._send_json(422, {"detail": {"reason": str(exc)}})
                return
            except Exception:
                # A durable-write failure must not answer 200: the platform
                # retries 5xx with backoff, and a false success would let it
                # search memories that do not exist yet.
                self._send_json(503, {"detail": {"reason": "write temporarily unavailable"}})
                return
            self._send_json(200, result)
            return
        try:
            result = search(payload)
        except ContractError as exc:
            self._send_json(422, {"detail": {"reason": str(exc)}})
            return
        self._send_json(200, result)

    def do_PUT(self) -> None:  # noqa: N802
        self._send_json(405, {"detail": {"reason": "method not allowed"}})

    def do_DELETE(self) -> None:  # noqa: N802
        self._send_json(405, {"detail": {"reason": "method not allowed"}})


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AMLRequestHandler)
    server.daemon_threads = True
    return server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PLM AML Add/Search adapter server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    for warning in validate_model_config():
        print("plm-aml WARNING: " + warning, file=sys.stderr, flush=True)
    print(
        "plm-aml %s listening on http://%s:%d (storage: %s, models: %s)"
        % (AML_ADAPTER_VERSION, args.host, args.port, memory_base(),
           json.dumps(model_status(), sort_keys=True)),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
