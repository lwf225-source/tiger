from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from operator import mul
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .database import searchable_tags
from .model import PLMError, SearchResult, utc_now
from .passages import split_passages
from .temporal_intent import extract_temporal_intent
from .providers import ProviderUnavailable
from .vector import cosine_blob, encode_vector


# Hashed character n-grams are intentionally a fuzzy lexical fallback, not a
# neural semantic signal.  Scores below this floor are dominated by hash
# collisions at the current 512 dimensions and should be treated as no match.
MIN_VECTOR_SIMILARITY = 0.08

# Opt-in recency micro-nudge (``recency_boost``). Hybrid Episodic Memory and
# other AML leaderboard systems apply recency inside rank space, as a small
# deterministic nudge — never a competing relevance signal. The amplitude is
# capped far below the structural bonuses already present (title bonus 0.03,
# kind bonus 0.003-0.008, one top-channel RRF step 1/61 ~= 0.016), so it can
# only break near-ties. A 30-day half-life keeps month-old memories at half
# the (already tiny) nudge; the exact value is a judgement call, not measured.
RECENCY_HALF_LIFE_DAYS = 30.0
RECENCY_MAX_BONUS = 0.004

# Opt-in multi-hop entity expansion (``expand_entities``). Chronicle (AML #7)
# expands from lexical/entity seeds over a BM25+entity graph with a <=2 hop
# bound; we do the same over the existing event_entities/entities tables. The
# per-query cap keeps the supplement bounded regardless of graph density —
# a hub entity mentioned by hundreds of memories must not flood the pack.
MAX_ENTITY_HOPS = 2
MAX_ENTITY_EXPANSION = 8

# Opt-in incremental MMR (``mmr``) at the final ranking stage, following
# aml-memory-mvp (AML #10): when picking the k-th entry, penalise candidates
# whose information overlaps what is already selected. lambda leans towards
# relevance (0.7) because the competition metric is evidence *recall* — a
# redundant near-duplicate is still evidence — while the 0.3 diversity term
# is enough to swap a near-duplicate for a distinct complementary memory.
# Judgement call anchored on classic MMR practice (Carbonell & Goldstein use
# 0.5 as the balanced point), not measured tuning.
MMR_LAMBDA = 0.7
MMR_POOL_FACTOR = 2
MMR_POOL_MIN = 16

# Opt-in AML window-composition fix (``window_reserve``, batch 12). Under the
# AML Search protocol top_k=100 is passed as the limit and the caller slices
# ``results[:limit]``; anything appended *beyond* the window (the frozen
# adjacent behaviour) can never be measured — batch 6 showed AML_ADJACENT=2 is
# a no-op there. With the switch on, up to ``limit // 4`` tail slots of the
# window are reserved for adjacent extras (same share discipline as
# ``_merge_expansion``), taken in seed-rank order (neighbours of the
# highest-ranked direct hits first), displacing only the lowest-ranked direct
# hits. Default off is byte-identical to previous behavior.
# Opt-in query-side coverage re-ranking (``coverage_rerank``, batch 12),
# modelled on aml-memory-mvp's information-word coverage scoring: greedily
# reward candidates covering rare query content words not yet covered by the
# already-selected set. This is the query-side complement of MMR's
# result-side redundancy penalty (batch 6 measured result-side MMR as
# ineffective on LoCoMo). Amplitude discipline matches the existing bonus
# system: the cap sits below one top-channel RRF step (1/61 ~= 0.016) and the
# title bonus (0.03), above the recency tie-breaker (0.004), so coverage can
# break near-ties in favour of uncovered-evidence candidates but can never
# invent relevance for an unretrievable record.
COVERAGE_MAX_BONUS = 0.01
COVERAGE_POOL_FACTOR = 2
COVERAGE_POOL_MIN = 16
COVERAGE_MAX_TERMS = 32

# Candidate-only source-derived session view.  It is deliberately bounded to
# the immediately preceding/following source events.  A view is a recall
# feature, never a persisted record or returned body, so deleting an event
# cannot leave an independently retrievable derivative behind.
SESSION_VIEW_TAG_PREFIX = "aml-session:"
SESSION_VIEW_RADIUS = 1
SESSION_RRF_K = 30.0

# Opt-in abstention floor (``abstain_threshold``). The fused RRF score carries
# no absolute-relevance meaning (rank-space only), so the floor is applied to
# each candidate's best *absolute* channel cosine — the local n-gram vector
# similarity (floored at MIN_VECTOR_SIMILARITY) and, when an embedding provider
# is configured, the neural cosine. Candidates supported only by rank-space
# lexical channels (FTS/substring) have cosine 0 and are withheld when the
# switch is on. If nothing survives, search returns an empty list and the
# diagnostics flag ``abstain.abstained`` is set — the explicit "insufficient
# evidence" signal downstream Answer models can honour instead of hallucinating
# over padding results. Cosine scales are model-specific: a threshold is only
# meaningful against the frozen model it was calibrated on (docs/BATCH7_REPORT.md).
MAX_ABSTAIN_THRESHOLD = 1.0

# Opt-in guarded soft supersession (``soft_supersede``), modelled on FlowGrid
# (AML #8): when the switch is on, a Fact version whose validity window was
# closed by a successor is not dropped from the candidate set but recalled with
# a fixed demotion, so temporal-flavoured queries can still surface the old
# value while the current value wins on score. The penalty is a ranking-layer
# constant only — larger than the recency tie-breaker cap (0.004) so the
# demotion is actually visible, yet below one top-channel RRF step
# (1/61 ~= 0.016) and the title bonus (0.03), so it can never invent relevance
# for an otherwise unretrievable record. It never touches the Fact lifecycle:
# retracted and expired memories stay excluded no matter what.
SOFT_SUPERSEDE_PENALTY = 0.01

_MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_WEEKDAYS_CN = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
_WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _temporal_point(value: Optional[str], name: str) -> str:
    """Normalise an opt-in temporal filter to the canonical UTC string.

    Empty/None means "filter off" (the default, byte-identical behavior).
    Anything else must be an ISO 8601 timestamp (date shorthand allowed);
    naive values are read as UTC, matching stored recorded_at/valid_* text.
    """
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError("invalid %s timestamp" % name)
    parsed = _iso_datetime(value.strip())
    if parsed is None:
        raise ValueError("invalid %s timestamp" % name)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _time_text(record: Dict[str, Any], now: str) -> str:
    """Date expressions injected into opt-in scoring text (never the FTS index).

    Both Chinese and English query phrasings are covered: ISO dates, CN/EN
    year-month-day and year-month forms, weekday names, and a few relative
    expressions (today/yesterday/this week/this month/this year...) resolved
    against the search call's ``now``. The persisted FTS index is deliberately
    untouched, so the default retrieval behavior and ``rebuild-index`` stay
    byte-identical; injection happens only in the in-memory scoring channels.
    """
    today_dt = _iso_datetime(now)
    days = sorted({
        parsed.date() for parsed in
        (_iso_datetime(str(record.get("recorded_at", ""))), _iso_datetime(str(record.get("valid_from", ""))))
        if parsed is not None
    })
    terms: List[str] = []
    for day in days:
        terms.extend([
            day.isoformat(),
            "%d年%d月%d日" % (day.year, day.month, day.day),
            "%d年%d月" % (day.year, day.month),
            "%s %d, %d" % (_MONTHS_EN[day.month - 1], day.day, day.year),
            "%s %d" % (_MONTHS_EN[day.month - 1], day.year),
            _WEEKDAYS_CN[day.weekday()],
            _WEEKDAYS_EN[day.weekday()],
        ])
        if today_dt is None:
            continue
        today = today_dt.date()
        delta = (today - day).days
        if delta == 0:
            terms.extend(["今天", "today"])
        elif delta == 1:
            terms.extend(["昨天", "yesterday"])
        elif delta == 2:
            terms.append("前天")
        elif delta == -1:
            terms.extend(["明天", "tomorrow"])
        if 0 <= delta <= 6:
            terms.extend(["本周", "this week"])
        elif 7 <= delta <= 13:
            terms.extend(["上周", "last week"])
        if (day.year, day.month) == (today.year, today.month):
            terms.extend(["本月", "这个月", "this month"])
        if day.year == today.year:
            terms.extend(["今年", "this year"])
        elif today.year - day.year == 1:
            terms.extend(["去年", "last year"])
    return " ".join(dict.fromkeys(terms))


def _recency_bonus(record: Dict[str, Any], now: str) -> float:
    recorded = _iso_datetime(str(record.get("recorded_at", "")))
    current = _iso_datetime(now)
    if recorded is None or current is None:
        return 0.0
    # Future timestamps (clock skew, planned valid_from) clamp to the cap.
    age_seconds = max(0.0, (current - recorded).total_seconds())
    return RECENCY_MAX_BONUS * 0.5 ** (age_seconds / (RECENCY_HALF_LIFE_DAYS * 86400.0))


def _fts_query(query: str) -> str:
    chunks = re.findall(r"[A-Za-z0-9._-]{3,}|[\u4e00-\u9fff]{3,}", query.lower())
    return " OR ".join('"%s"' % chunk.replace('"', '') for chunk in chunks[:12])


def _expand_aliases(conn: sqlite3.Connection, project_id: str, query: str) -> Tuple[str, List[str]]:
    lowered = query.lower()
    rows = conn.execute(
        "SELECT a.alias,e.canonical_name FROM entity_aliases a JOIN entities e ON e.entity_id=a.entity_id "
        "WHERE a.project_id=?",
        (project_id,),
    ).fetchall()
    additions: List[str] = []
    for row in rows:
        if row["alias"] in lowered and row["canonical_name"].lower() not in lowered:
            additions.append(row["canonical_name"])
    expanded = query + (" " + " ".join(additions) if additions else "")
    return expanded, additions


def _search_legacy(
    conn: sqlite3.Connection,
    project_id: str,
    query: str,
    limit: int = 6,
    view: str = "current",
    scope: str = "project",
    scope_id: str = "",
) -> List[SearchResult]:
    if view not in {"current", "history", "both"}:
        raise ValueError("invalid memory view")
    expanded, aliases = _expand_aliases(conn, project_id, query)
    ranked: Dict[Tuple[str, str], float] = defaultdict(float)
    reasons: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    now = utc_now()
    scope_suffix = " AND e.scope=?" + (" AND e.scope_id=?" if scope_id else "")
    event_state = " AND e.status='active'" if view == "current" else ""
    fact_state = (
        " AND f.status='active' AND f.valid_from<=? AND (f.valid_to='' OR f.valid_to>?) "
        "AND (f.expires_at='' OR f.expires_at>?)" if view == "current" else ""
    )
    eligibility = (
        "((record_type='event' AND EXISTS(SELECT 1 FROM events e WHERE e.event_id=memory_fts.ref_id "
        "AND e.project_id=? AND e.kind<>'fact' AND e.operation=''" + scope_suffix + event_state + ")) OR "
        "(record_type='fact' AND EXISTS(SELECT 1 FROM facts f JOIN events e ON e.event_id=f.assertion_event_id "
        "WHERE f.fact_id=memory_fts.ref_id AND f.project_id=?" + scope_suffix + fact_state + ")))"
    )
    eligibility_params: List[str] = [project_id, scope]
    if scope_id:
        eligibility_params.append(scope_id)
    eligibility_params.extend([project_id, scope])
    if scope_id:
        eligibility_params.append(scope_id)
    if view == "current":
        eligibility_params.extend([now, now, now])

    fts = _fts_query(expanded)
    if fts:
        try:
            rows = conn.execute(
                "SELECT ref_id,record_type,bm25(memory_fts,0.0,0.0,0.0,8.0,4.0,1.0) AS rank "
                "FROM memory_fts WHERE memory_fts MATCH ? AND project_id=? AND " + eligibility +
                " ORDER BY rank LIMIT 80",
                tuple([fts, project_id, *eligibility_params]),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for position, row in enumerate(rows, 1):
            key = (row["record_type"], row["ref_id"])
            ranked[key] += 1.0 / (60.0 + position)
            reasons[key].append("fts5-bm25")

    query_vector = encode_vector(expanded)
    vector_eligibility = eligibility.replace("memory_fts.ref_id", "vectors.ref_id")
    vector_rows = conn.execute(
        "SELECT record_type,ref_id,vector FROM vectors WHERE project_id=? AND " + vector_eligibility,
        tuple([project_id, *eligibility_params]),
    ).fetchall()
    similarities = []
    for row in vector_rows:
        similarity = cosine_blob(query_vector, row["vector"])
        if similarity >= MIN_VECTOR_SIMILARITY:
            similarities.append((similarity, row["record_type"], row["ref_id"]))
    similarities.sort(reverse=True)
    for position, (similarity, record_type, ref_id) in enumerate(similarities[:80], 1):
        key = (record_type, ref_id)
        ranked[key] += 1.0 / (60.0 + position)
        reasons[key].append("local-ngram-vector:%.3f" % similarity)

    results: List[SearchResult] = []
    for (record_type, ref_id), base_score in ranked.items():
        if record_type == "fact":
            row = conn.execute(
                "SELECT f.*,e.title,e.tags_json,e.scope,e.scope_id,COALESCE(NULLIF(e.source_legacy_path,''),e.source_path) AS source_path "
                "FROM facts f JOIN events e ON e.event_id=f.assertion_event_id WHERE f.fact_id=?",
                (ref_id,),
            ).fetchone()
            if not row:
                continue
            if row["scope"] != scope or (scope_id and row["scope_id"] != scope_id):
                continue
            if view == "current" and row["status"] != "active":
                continue
            if view == "current" and row["valid_from"] > now:
                continue
            if view == "current" and row["valid_to"] and row["valid_to"] <= now:
                continue
            if row["expires_at"] and row["expires_at"] <= now and view == "current":
                continue
            body = json.loads(row["value_json"])
            if not isinstance(body, str):
                body = json.dumps(body, ensure_ascii=False, sort_keys=True)
            title_lower = row["fact_key"].lower()
            query_lower = query.strip().lower()
            title_bonus = 0.03 if query_lower == title_lower else (0.015 if query_lower and (query_lower in title_lower or title_lower in query_lower) else 0.0)
            score = base_score + (0.008 if row["status"] == "active" else 0.0) + 0.004 * float(row["confidence"]) + title_bonus
            result = SearchResult(
                ref_id, "fact", row["fact_key"], body, row["project_id"], score,
                reasons[(record_type, ref_id)] + (["entity-alias"] if aliases else []),
                row["source_path"], row["status"], row["valid_from"], row["valid_to"],
                row["source_event_id"],
            )
        else:
            row = conn.execute(
                "SELECT *,COALESCE(NULLIF(source_legacy_path,''),source_path) AS display_source_path FROM events WHERE event_id=?",
                (ref_id,),
            ).fetchone()
            if not row or (view == "current" and row["status"] != "active") or row["operation"]:
                continue
            if row["scope"] != scope or (scope_id and row["scope_id"] != scope_id):
                continue
            query_lower = query.strip().lower()
            title_lower = row["title"].lower()
            title_bonus = 0.03 if query_lower == title_lower else (0.015 if query_lower and (query_lower in title_lower or title_lower in query_lower) else 0.0)
            kind_bonus = {"core": 0.007, "procedure": 0.004, "artifact_ref": 0.003}.get(row["kind"], 0.0)
            result = SearchResult(
                ref_id, "event", row["title"], row["body"], row["project_id"], base_score + kind_bonus + title_bonus,
                reasons[(record_type, ref_id)] + (["entity-alias"] if aliases else []),
                row["display_source_path"], row["status"], row["observed_at"], "",
            )
        results.append(result)
    results.sort(key=lambda result: (-result.score, result.record_type != "fact", result.title))
    return results[:limit]


_CJK = re.compile(r"[\u4e00-\u9fff]+$")
_QUERY_BREAKS = re.compile(r"为什么|能不能|是什么|有哪些|需要|怎么|如何|是否|哪些|什么|多少|请问|目前|现在|我们|这个|这套|本次|上次")
_STOP_TERMS = {"什么", "怎么", "如何", "是否", "哪些", "多少", "请问", "目前", "现在", "我们", "这个", "这套", "本次", "上次", "可以", "需要", "了吗", "的是", "的时候"}


def _lexical_terms(query: str) -> List[str]:
    """Small deterministic Chinese n-gram query expansion; no model-generated facts."""
    terms: List[str] = []
    for chunk in re.findall(r"[a-z0-9._-]{2,}|[\u4e00-\u9fff]{2,}", query.lower()[:2048]):
        if not _CJK.fullmatch(chunk):
            terms.append(chunk)
            continue
        for part in _QUERY_BREAKS.split(chunk):
            part = part.strip("的了吗呢啊")
            if len(part) < 2:
                continue
            if len(part) <= 12:
                terms.append(part)
            for size in (3, 2):
                terms.extend(part[index:index + size] for index in range(len(part) - size + 1))
    return list(dict.fromkeys(term for term in terms if term not in _STOP_TERMS))[:96]


def _eligibility(project_id: str, scope: str, scope_id: str, view: str, now: str, table: str,
                 valid_at: str = "", known_at: str = "", soft_supersede: bool = False) -> Tuple[str, List[str]]:
    # A scheduled successor changes stored status before it becomes valid. The
    # predecessor remains current within its effective interval. Retractions of
    # either the Fact or its assertion Event are never supplied as live context.
    # ``valid_at`` (opt-in) substitutes the given valid-time point for "now" in
    # the Fact window check (and bounds Events by observed_at); ``known_at``
    # (opt-in) bounds the system-time axis by the assertion Event's
    # recorded_at. expires_at stays a system-time discipline checked against
    # the real "now" — time travel never revives retention-expired memories.
    # ``soft_supersede`` keeps window-closed superseded versions eligible so
    # the ranking layer can demote instead of excluding them.
    point = valid_at or (now if view == "current" else "")
    event_clause = (" AND e.observed_at<=?" if valid_at else "") + (" AND e.recorded_at<=?" if known_at else "")
    event_params = ([valid_at] if valid_at else []) + ([known_at] if known_at else [])
    scope_filter = " AND e.scope=? AND e.scope_id=? AND e.status='active' AND e.operation=''" + event_clause
    fact_state = " AND f.status IN ('active','superseded')"
    fact_params: List[str] = []
    if point:
        fact_state += " AND f.valid_from<=?"
        fact_params.append(point)
        if soft_supersede:
            fact_state += " AND (f.valid_to='' OR f.valid_to>? OR f.status='superseded')"
        else:
            fact_state += " AND (f.valid_to='' OR f.valid_to>?)"
        fact_params.append(point)
    if view == "current":
        fact_state += " AND (f.expires_at='' OR f.expires_at>?)"
        fact_params.append(now)
    predicate = (
        "((record_type='event' AND EXISTS(SELECT 1 FROM events e WHERE e.event_id=" + table + ".ref_id "
        "AND e.project_id=? AND e.kind<>'fact'" + scope_filter + ")) OR "
        "(record_type='fact' AND EXISTS(SELECT 1 FROM facts f JOIN events e ON e.event_id=f.assertion_event_id "
        "WHERE f.fact_id=" + table + ".ref_id AND f.project_id=?" + scope_filter + fact_state + ")))"
    )
    params = [project_id, scope, scope_id] + event_params + [project_id, scope, scope_id] + event_params + fact_params
    return predicate, params


def _eligible_records(conn: sqlite3.Connection, project_id: str, scope: str, scope_id: str, view: str, now: str,
                      valid_at: str = "", known_at: str = "", soft_supersede: bool = False) -> Dict[Tuple[str, str], Dict[str, Any]]:
    event_clause = (" AND observed_at<=?" if valid_at else "") + (" AND recorded_at<=?" if known_at else "")
    event_params = [project_id, scope, scope_id] + ([valid_at] if valid_at else []) + ([known_at] if known_at else [])
    event_rows = conn.execute(
        "SELECT 'event' AS record_type,event_id AS ref_id,event_id AS assertion_event_id,project_id,title,tags_json,body,"
        "status,kind,observed_at AS valid_from,'' AS valid_to,recorded_at,"
        "COALESCE(NULLIF(source_legacy_path,''),source_path) AS source_path "
        "FROM events WHERE project_id=? AND scope=? AND scope_id=? AND status='active' AND operation='' AND kind<>'fact'"
        + event_clause,
        tuple(event_params),
    ).fetchall()
    point = valid_at or (now if view == "current" else "")
    time_clause = ""
    params = [project_id, scope, scope_id]
    if point:
        time_clause += " AND f.valid_from<=?"
        params.append(point)
        if soft_supersede:
            time_clause += " AND (f.valid_to='' OR f.valid_to>? OR f.status='superseded')"
        else:
            time_clause += " AND (f.valid_to='' OR f.valid_to>?)"
        params.append(point)
    if view == "current":
        time_clause += " AND (f.expires_at='' OR f.expires_at>?)"
        params.append(now)
    if known_at:
        time_clause += " AND e.recorded_at<=?"
        params.append(known_at)
    fact_rows = conn.execute(
        "SELECT 'fact' AS record_type,f.fact_id AS ref_id,f.assertion_event_id,f.source_event_id,f.project_id,f.fact_key AS title,e.tags_json,"
        "e.body,f.status,e.kind,f.valid_from,f.valid_to,e.recorded_at,COALESCE(NULLIF(e.source_legacy_path,''),e.source_path) AS source_path "
        "FROM facts f JOIN events e ON e.event_id=f.assertion_event_id "
        "WHERE f.project_id=? AND e.scope=? AND e.scope_id=? AND e.status='active' AND e.operation='' "
        "AND f.status IN ('active','superseded')" + time_clause,
        tuple(params),
    ).fetchall()
    records = {(row["record_type"], row["ref_id"]): dict(row) for row in list(event_rows) + list(fact_rows)}
    return _filter_source_chains(conn, records, project_id, scope, scope_id, view, now,
                                 valid_at=valid_at, soft_supersede=soft_supersede)


def _session_view_texts(records: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """Build bounded, source-only three-message recall views from AML tags.

    The adapter records a one-way session digest in ``tags_json``.  Only
    original event records with that tag may contribute; facts and untagged
    historical records remain untouched.  Each output maps to the centre
    event, letting the caller rank original evidence with nearby conversational
    context without persisting or exposing a synthesized view.
    """
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key, row in records.items():
        if key[0] != "event":
            continue
        try:
            tags = json.loads(row.get("tags_json", "[]"))
        except (TypeError, ValueError):
            continue
        if not isinstance(tags, list):
            continue
        session_tag = next((tag for tag in tags if isinstance(tag, str) and tag.startswith(SESSION_VIEW_TAG_PREFIX)), "")
        if session_tag:
            groups[session_tag].append(key)
    views: Dict[Tuple[str, str], str] = {}
    for members in groups.values():
        ordered = sorted(members, key=lambda key: (records[key]["valid_from"], records[key]["recorded_at"], key[1]))
        for index, key in enumerate(ordered):
            lo = max(0, index - SESSION_VIEW_RADIUS)
            hi = min(len(ordered), index + SESSION_VIEW_RADIUS + 1)
            view = "\n".join(records[item]["body"] for item in ordered[lo:hi] if records[item]["body"].strip())
            if view:
                views[key] = view
    return views


def _session_tag(row: Dict[str, Any]) -> str:
    """Return an AML one-way session digest, never its raw session id."""
    try:
        tags = json.loads(row.get("tags_json", "[]"))
    except (TypeError, ValueError):
        return ""
    if not isinstance(tags, list):
        return ""
    return next((tag for tag in tags if isinstance(tag, str) and tag.startswith(SESSION_VIEW_TAG_PREFIX)), "")


def _session_rrf_adjust(
    ranked: Dict[Tuple[str, str], float], records: Dict[Tuple[str, str], Dict[str, Any]],
    reasons: Dict[Tuple[str, str], List[str]],
) -> Dict[str, int]:
    """Fuse direct-event ranks with aggregate session ranks deterministically."""
    tags_by_key = {
        key: _session_tag(records[key])
        for key in ranked
        if key[0] == "event" and key in records
    }
    session_scores: Dict[str, float] = defaultdict(float)
    for key, tag in tags_by_key.items():
        if tag:
            session_scores[tag] += ranked[key]
    ordered = sorted(session_scores, key=lambda tag: (-session_scores[tag], tag))
    ranks = {tag: position for position, tag in enumerate(ordered, 1)}
    adjusted = 0
    for key, tag in tags_by_key.items():
        if tag in ranks:
            ranked[key] += 1.0 / (SESSION_RRF_K + ranks[tag])
            reasons[key].append("session-rrf:%d" % ranks[tag])
            adjusted += 1
    return {"sessions": len(ranks), "adjusted": adjusted}


def _filter_source_chains(
    conn: sqlite3.Connection, records: Dict[Tuple[str, str], Dict[str, Any]],
    project_id: str, scope: str, scope_id: str, view: str, now: str, valid_at: str = "",
    soft_supersede: bool = False,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """A derived memory cannot outlive its source Fact's effective interval.

    Pre-DB4 stores expose Fact provenance only. No unavailable Event provenance
    is invented; an absent, cross-scope or cyclic explicit source is rejected.
    ``valid_at`` (opt-in) moves the valid-time point of the source-window check
    while expiry stays pinned to the real "now"; ``soft_supersede`` (opt-in)
    lets window-closed superseded versions through for ranking-layer demotion.
    The default is byte-identical.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    event_source = "e.source_event_id" if "source_event_id" in columns else "''"
    rows = conn.execute(
        "SELECT e.event_id,e.status AS event_status," + event_source + " AS event_source,"
        "f.fact_id,f.status AS fact_status,f.source_event_id AS fact_source,f.valid_from,f.valid_to,f.expires_at "
        "FROM events e LEFT JOIN facts f ON f.assertion_event_id=e.event_id "
        "WHERE e.project_id=? AND e.scope=? AND e.scope_id=? AND (f.project_id IS NULL OR f.project_id=e.project_id)",
        (project_id, scope, scope_id),
    ).fetchall()
    nodes = {row["event_id"]: row for row in rows}
    resolved: Dict[str, bool] = {}
    point = valid_at or (now if view == "current" else "")

    def valid(identity: str) -> bool:
        trail = []
        seen = set()
        while identity not in resolved:
            if identity in seen:
                outcome = False
                break
            seen.add(identity)
            trail.append(identity)
            node = nodes.get(identity)
            if node is None or node["event_status"] != "active" or node["fact_status"] == "retracted":
                outcome = False
                break
            if node["fact_id"] and point and (
                node["fact_status"] not in {"active", "superseded"} or node["valid_from"] > point
                or (node["valid_to"] and node["valid_to"] <= point
                    and not (soft_supersede and node["fact_status"] == "superseded"))
                or (node["expires_at"] and node["expires_at"] <= now)
            ):
                outcome = False
                break
            parent = node["event_source"] or node["fact_source"] or ""
            if not parent or (parent == identity and node["fact_id"] and not node["event_source"]):
                outcome = True
                break
            identity = parent
        else:
            outcome = resolved[identity]
        for item in trail:
            resolved[item] = outcome
        return outcome

    return {key: record for key, record in records.items() if valid(record["assertion_event_id"])}


def _scoped_aliases(conn: sqlite3.Connection, project_id: str, query: str, records: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[str, List[str]]:
    allowed_events = {record["assertion_event_id"] for record in records.values()}
    rows = conn.execute(
        "SELECT a.alias,e.canonical_name,ee.event_id FROM entity_aliases a "
        "JOIN entities e ON e.entity_id=a.entity_id JOIN event_entities ee ON ee.entity_id=e.entity_id WHERE a.project_id=?",
        (project_id,),
    ).fetchall()
    lowered = query.lower()
    additions = []
    for row in rows:
        if row["event_id"] not in allowed_events:
            continue
        alias = row["alias"].lower()
        matches = alias in lowered if _CJK.search(alias) else bool(re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", lowered))
        if matches and row["canonical_name"].lower() not in lowered:
            additions.append(row["canonical_name"])
    additions = list(dict.fromkeys(additions))[:12]
    return query + (" " + " ".join(additions) if additions else ""), additions


def _retrieval_tags(record: Dict[str, Any]) -> List[str]:
    """Keep internal routing tags out of every retrieval scoring channel."""
    return searchable_tags(json.loads(record["tags_json"]))


def _record_text(record: Dict[str, Any]) -> str:
    return "\n".join([record["title"], " ".join(_retrieval_tags(record)), record["body"]])


def _substring_ranking(
    records: Dict[Tuple[str, str], Dict[str, Any]], terms: Sequence[str],
    suffixes: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Tuple[float, Tuple[str, str]]]:
    # FTS5 trigram cannot recall two-character Chinese words. Scoped substring
    # matches supply that channel and rank rare terms above ubiquitous fragments.
    # ``suffixes`` carries opt-in per-record text (date expressions) appended to
    # the lowest-weight body field; it never changes the persisted index.
    fields = {key: (record["title"].lower(), " ".join(_retrieval_tags(record)).lower(),
                    record["body"].lower() + ((" " + suffixes[key].lower()) if suffixes and suffixes.get(key) else ""))
              for key, record in records.items()}
    counts = {term: sum(any(term in field for field in values) for values in fields.values()) for term in terms}
    scores = []
    for key, values in fields.items():
        score = 0.0
        for term in terms:
            weight = math.log(1.0 + (len(records) + 1.0) / (counts[term] + 1.0)) * min(len(term), 6)
            score += weight * sum(boost for field, boost in zip(values, (5.0, 3.0, 1.0)) if term in field)
        if score:
            scores.append((score, key))
    return sorted(scores, key=lambda item: (-item[0], item[1]))


def _describe(provider: Any) -> Dict[str, Any]:
    try:
        return dict(provider.describe()) if hasattr(provider, "describe") else {"provider": type(provider).__name__}
    except Exception:
        return {"provider": type(provider).__name__}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) == 0:
        raise ProviderUnavailable("embedding-dimension-mismatch")
    if any(not math.isfinite(float(value)) for value in list(left) + list(right)):
        raise ProviderUnavailable("nonfinite-model-output")
    norm = math.sqrt(sum(float(value) ** 2 for value in left) * sum(float(value) ** 2 for value in right))
    return sum(float(a) * float(b) for a, b in zip(left, right)) / norm if norm else 0.0


def startup_context(conn, project_id, limit=6, view="current", scope="project", scope_id="",
                    valid_at="", known_at="", soft_supersede=False):
    scope_id = scope_id or (project_id if scope == "project" else scope)
    records = _eligible_records(conn, project_id, scope, scope_id, view, utc_now(),
                                valid_at=valid_at, known_at=known_at, soft_supersede=soft_supersede)
    priority = {"core": 4, "fact": 3, "procedure": 2, "artifact_ref": 1, "episode": 0}
    rows = sorted(records.values(), key=lambda row: (priority.get(row["kind"], 0), row["valid_from"], row["ref_id"]), reverse=True)
    return [SearchResult(row["ref_id"], row["record_type"], row["title"], row["body"], project_id, 0.0,
        ["startup-current"], row["source_path"], row["status"], row["valid_from"], row["valid_to"], row["assertion_event_id"])
        for row in rows[:max(0, limit)]]


MAX_ADJACENT_WINDOW = 8


def _entity_expand(conn, records, project_id, hits, hops, info, passage_mode=False):
    """Append records that share an entity with a direct hit, up to ``hops`` hops.

    The entity graph comes from the existing event_entities/entities tables,
    restricted to the same eligible-record universe as the direct hits, so
    project/scope/status/lifecycle/source-chain filtering cannot be bypassed.
    Supplements are appended after direct hits (and any adjacent window), carry
    score 0.0 and the ``entity-expansion:hop<N>`` reason, and are deduplicated
    against the hits and each other. Ranking within one hop prefers records
    sharing more seed entities, then recorded_at — deterministic, and never
    disturbs the direct-hit order.
    """
    event_keys: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key, record in records.items():
        event_keys[record["assertion_event_id"]].append(key)
    rows = conn.execute(
        "SELECT ee.event_id,ee.entity_id FROM event_entities ee "
        "JOIN entities e ON e.entity_id=ee.entity_id WHERE e.project_id=?",
        (project_id,),
    ).fetchall()
    event_entities: Dict[str, set] = defaultdict(set)
    entity_events: Dict[str, set] = defaultdict(set)
    for row in rows:
        if row["event_id"] in event_keys:
            event_entities[row["event_id"]].add(row["entity_id"])
            entity_events[row["entity_id"]].add(row["event_id"])
    taken = {(item.record_type, item.ref_id) for item in hits}
    frontier = set()
    for item in hits:
        # Only direct hits seed expansion; adjacent extras are temporal context,
        # not relevance anchors.
        if "adjacent-context" in item.reasons:
            continue
        frontier |= event_entities.get(item.source_event_id or item.ref_id, set())
    seen_entities = set(frontier)
    chosen: Dict[Tuple[str, str], int] = {}
    for hop in range(1, hops + 1):
        if not frontier or len(chosen) >= MAX_ENTITY_EXPANSION:
            break
        candidates = set()
        for entity_id in frontier:
            candidates |= entity_events.get(entity_id, set())
        fresh: List[Tuple[str, str]] = []
        for event_id in candidates:
            for key in event_keys[event_id]:
                if key not in taken and key not in chosen:
                    fresh.append(key)
        if not fresh:
            break

        def shared(key: Tuple[str, str]) -> int:
            return len(event_entities.get(records[key]["assertion_event_id"], set()) & frontier)

        fresh.sort(key=lambda key: (-shared(key), records[key]["recorded_at"], key[1], key[0]))
        picked = fresh[:MAX_ENTITY_EXPANSION - len(chosen)]
        next_frontier: set = set()
        for key in picked:
            chosen[key] = hop
            next_frontier |= event_entities.get(records[key]["assertion_event_id"], set())
        frontier = next_frontier - seen_entities
        seen_entities |= next_frontier
    extras = []
    for key in sorted(chosen, key=lambda item: (chosen[item], records[item]["recorded_at"], item[1], item[0])):
        row = records[key]
        result = SearchResult(
            key[1], key[0], row["title"], row["body"], project_id, 0.0,
            ["entity-expansion:hop%d" % chosen[key]], row["source_path"], row["status"],
            row["valid_from"], row["valid_to"], row["assertion_event_id"],
        )
        if passage_mode:
            spans = split_passages(row["body"], row["assertion_event_id"])
            if not spans:
                continue
            result.evidence_spans = spans[:1]
        extras.append(result)
    return extras


def _merge_expansion(hits, extras, limit):
    """Place expansion extras inside the evidence window, ahead of adjacent extras.

    Trade-off, recorded deliberately: direct hits keep their relative order and
    always dominate the window — expansion extras take at most a quarter of it
    (minimum one slot), displacing only the lowest-ranked direct hits when the
    window is full. Rationale: under the AML top_k=100 protocol anything
    appended *beyond* the limit is sliced away by the caller, so an append-only
    supplement can never be measured; and adjacent extras keep their frozen
    beyond-window semantics (plm-aml 1.1.0 unchanged) because a temporal
    neighbour is usually near-duplicate context, while an entity neighbour is
    complementary evidence — exactly what the window's tail slots are for.
    Extras beyond the window share trail at the very end for interactive use.
    """
    if not extras:
        return hits, 0
    direct = [item for item in hits if "adjacent-context" not in item.reasons]
    adjacent = [item for item in hits if "adjacent-context" in item.reasons]
    share = max(1, limit // 4)
    in_window = extras[:share]
    merged = direct[:max(0, limit - len(in_window))] + in_window + adjacent + extras[len(in_window):]
    return merged, len(in_window)


def _mmr_select(results, records, limit, info, semantic_vectors=None):
    """Incremental MMR over the head of the ranked list; the tail keeps its order.

    Relevance is the candidate's score normalised by the pool maximum;
    redundancy is the maximum cosine against already-selected entries. The
    similarity space is the existing 512-dim lexical n-gram vector (zero new
    dependencies); when this search call already computed neural embeddings
    for the embedding channel, those are reused instead at no extra cost.
    Vectors are unit-normalised once up front and each selection round only
    updates running max-similarities, so the whole pass costs
    ``pool x limit`` plain dot products.
    """
    pool_size = min(len(results), max(MMR_POOL_FACTOR * limit, MMR_POOL_MIN))
    pool = list(results[:pool_size])
    keys = [(item.record_type, item.ref_id) for item in pool]
    top_score = max((item.score for item in pool), default=0.0)
    relevance = [item.score / top_score if top_score > 0 else 0.0 for item in pool]

    def unit_vectors(raw):
        output = []
        for values in raw:
            floats = [float(value) for value in values]
            norm = math.sqrt(sum(value * value for value in floats))
            output.append([value / norm for value in floats] if norm else floats)
        return output

    if semantic_vectors:
        # Already validated finite and dimension-consistent when the embedding
        # channel scored these same vectors against the query embedding.
        vectors = unit_vectors(semantic_vectors[key] for key in keys)
        space = "semantic-reuse"
    else:
        vectors = [
            list(struct.unpack("<%df" % (len(blob) // 4), blob))  # encode_vector output is unit-norm
            for blob in (encode_vector(_record_text(records[key])) for key in keys)
        ]
        space = "lexical-ngram-512"
    selected: List[int] = []
    remaining = list(range(len(pool)))
    max_similarity = [0.0] * len(pool)
    while remaining and len(selected) < limit:
        best = max(remaining, key=lambda index: (
            MMR_LAMBDA * relevance[index] - (1.0 - MMR_LAMBDA) * max_similarity[index], -index))
        selected.append(best)
        remaining.remove(best)
        best_vector = vectors[best]
        for index in remaining:
            similarity = sum(map(mul, vectors[index], best_vector))
            if similarity > max_similarity[index]:
                max_similarity[index] = similarity
    ordered = [pool[index] for index in selected] + [pool[index] for index in remaining]
    info["mmr"] = {"status": "ready", "lambda": MMR_LAMBDA, "pool": pool_size, "space": space}
    return ordered + list(results[pool_size:])


_COVERAGE_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]{2,}")
# Minimal English stopword list (deterministic, zero-dependency). Only words
# that carry no evidence-localisation signal in questions are excluded; nouns,
# verbs, numbers and dates all stay eligible as information words.
_COVERAGE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
    "no", "not", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    "would", "you", "your",
})


def _coverage_tokens(text):
    """Content words of a text: lowercase alphanumeric runs / CJK bigram runs,
    minus stopwords and single characters."""
    return {token for token in _COVERAGE_TOKEN.findall(text.lower())
            if len(token) >= 2 and token not in _COVERAGE_STOPWORDS}


def _coverage_rerank(results, records, query, limit, info):
    """Greedy query-side coverage re-ranking over the head of the ranking.

    Information words are the query's content words (stopword-filtered); each
    is weighted by corpus IDF over the eligible-record universe so *rare*
    words dominate. Selection is greedy: at each step the candidate with the
    highest ``score + COVERAGE_MAX_BONUS * uncovered-IDF-mass fraction`` wins,
    then the words it covers leave the uncovered set — a candidate that adds
    no new information word gets no bonus. Amplitude is additive on the raw
    fused score and capped at COVERAGE_MAX_BONUS, below one top-channel RRF
    step, so coverage reorders near-ties but never overrides a clear relevance
    gap. Only the pool head is reordered; the tail keeps its order, and the
    result set is a pure permutation (no candidate is added or dropped).
    """
    pool_size = min(len(results), max(COVERAGE_POOL_FACTOR * limit, COVERAGE_POOL_MIN))
    pool = list(results[:pool_size])
    query_tokens = _coverage_tokens(query)
    if not pool or not query_tokens:
        info["coverage_rerank"] = {
            "status": "skipped-no-candidates" if query_tokens else "skipped-no-info-words",
            "info_words": len(query_tokens)}
        return results
    if len(query_tokens) > COVERAGE_MAX_TERMS:
        query_tokens = set(sorted(query_tokens)[:COVERAGE_MAX_TERMS])
    record_tokens = {key: _coverage_tokens(_record_text(records[key])) for key in records}
    total = float(len(records))
    idf = {}
    for token in query_tokens:
        df = sum(1 for tokens in record_tokens.values() if token in tokens)
        idf[token] = math.log(1.0 + (total + 1.0) / (df + 1.0))
    total_mass = sum(idf.values())
    keys = [(item.record_type, item.ref_id) for item in pool]
    candidate_tokens = [record_tokens.get(key, set()) & query_tokens for key in keys]
    uncovered = set(query_tokens)
    selected: List[int] = []
    remaining = list(range(pool_size))
    target = min(limit, pool_size)
    while remaining and len(selected) < target:
        def adjusted(index):
            gain = sum(idf[token] for token in candidate_tokens[index] & uncovered)
            return (pool[index].score + COVERAGE_MAX_BONUS * gain / total_mass, -index)
        best = max(remaining, key=adjusted)
        selected.append(best)
        remaining.remove(best)
        uncovered -= candidate_tokens[best]
    ordered = [pool[index] for index in selected] + [pool[index] for index in remaining]
    info["coverage_rerank"] = {
        "status": "ready", "max_bonus": COVERAGE_MAX_BONUS, "pool": pool_size,
        "info_words": len(query_tokens), "covered_words": len(query_tokens) - len(uncovered),
        "space": "lexical-idf-corpus",
    }
    return ordered + list(results[pool_size:])


def _adjacent_seeded(records, hits, count):
    """Recorded_at neighbours of each direct hit, keyed by first-seed rank.

    Returns ``{key: seed_position}`` where ``seed_position`` is the rank of
    the highest-ranked direct hit that first claimed the neighbour. Neighbours
    come from the same eligible-record universe as the direct hits, so
    project/scope/status/lifecycle filtering cannot be bypassed, and are
    deduplicated against the hits and each other.
    """
    ordered = sorted(records, key=lambda key: (records[key]["recorded_at"], key[1], key[0]))
    positions = {key: index for index, key in enumerate(ordered)}
    hit_keys = {(item.record_type, item.ref_id) for item in hits}
    chosen: Dict[Tuple[str, str], int] = {}
    for seed, item in enumerate(hits):
        index = positions.get((item.record_type, item.ref_id))
        if index is None:
            continue
        neighbours = [ordered[index - step] for step in range(min(count, index), 0, -1)]
        neighbours += [ordered[index + step] for step in range(1, count + 1) if index + step < len(ordered)]
        for key in neighbours:
            if key not in hit_keys and key not in chosen:
                chosen[key] = seed
    return chosen


def _adjacent_result(records, project_id, key, passage_mode):
    row = records[key]
    result = SearchResult(
        key[1], key[0], row["title"], row["body"], project_id, 0.0,
        ["adjacent-context"], row["source_path"], row["status"],
        row["valid_from"], row["valid_to"], row["assertion_event_id"],
    )
    if passage_mode:
        spans = split_passages(row["body"], row["assertion_event_id"])
        if not spans:
            return None
        result.evidence_spans = spans[:1]
    return result


def _adjacent_augment(records, project_id, hits, count, info, passage_mode=False):
    """Append up to ``count`` recorded_at neighbours before/after each direct hit.

    Neighbours come from the same eligible-record universe as the direct hits,
    so project/scope/status/lifecycle filtering cannot be bypassed. Adjacent
    entries are appended after every direct hit, carry the ``adjacent-context``
    reason and are deduplicated against the hits and each other.
    """
    chosen = _adjacent_seeded(records, hits, count)
    extras = []
    for key in sorted(chosen, key=lambda item: (records[item]["recorded_at"], item[1], item[0])):
        result = _adjacent_result(records, project_id, key, passage_mode)
        if result is not None:
            extras.append(result)
    info["adjacent"] = {"status": "ready", "window": count, "added": len(extras)}
    return list(hits) + extras


def _adjacent_augment_reserved(records, project_id, hits, limit, count, info, passage_mode=False):
    """Window-composition fix (batch 12): reserve up to ``limit // 4`` tail slots.

    The frozen behaviour appends neighbours *beyond* the window, which a
    ``results[:limit]`` caller (AML Search top_k=100) slices away — adjacent
    is then unmeasurable. With ``window_reserve`` on, the extras seeded by the
    highest-ranked direct hits take up to a quarter of the window's tail slots
    (minimum one), displacing only the lowest-ranked direct hits; the overflow
    keeps trailing beyond the window for interactive use. Same share
    discipline as ``_merge_expansion``.
    """
    chosen = _adjacent_seeded(records, hits, count)
    extras = []
    for key, seed in sorted(chosen.items(), key=lambda kv: (kv[1], records[kv[0]]["recorded_at"], kv[0][1], kv[0][0])):
        result = _adjacent_result(records, project_id, key, passage_mode)
        if result is not None:
            extras.append(result)
    share = max(1, limit // 4)
    in_window = extras[:share]
    merged = list(hits)[:max(0, limit - len(in_window))] + in_window + extras[len(in_window):]
    info["adjacent"] = {"status": "ready", "window": count, "added": len(extras),
                        "window_reserve": True, "in_window": len(in_window), "share": share}
    return merged


def search(
    conn: sqlite3.Connection, project_id: str, query: str, limit: int = 6,
    view: str = "current", scope: str = "project", scope_id: str = "",
    profile: str = "lexical", embedding_provider: Any = None, reranker: Any = None,
    diagnostics: Optional[Dict[str, Any]] = None, adjacent: int = 0,
    recency_boost: bool = False, expand_entities: int = 0, mmr: bool = False,
    abstain_threshold: float = 0.0, valid_at: str = "", known_at: str = "",
    soft_supersede: bool = False, temporal_intent: bool = False,
    window_reserve: bool = False, coverage_rerank: bool = False,
    session_views: bool = False,
    session_rrf: bool = False,
    query_provider: Any = None,
) -> List[SearchResult]:
    """Retrieve scoped evidence, optionally augmenting lexical search with local models.

    ``legacy`` preserves v2.1.0 query/ranking behavior for controlled comparisons.
    Callers selecting models should surface ``diagnostics``; fallback reasons are
    also attached to returned results, without exposing exception/input text.
    ``adjacent`` > 0 is an explicit opt-in appending up to N recorded_at
    neighbours before/after each direct hit; the frozen ``legacy`` profile and
    the default ranking are never changed by it.
    ``recency_boost`` is an explicit opt-in adding date expressions to the
    in-memory scoring text (the persisted FTS index is untouched) plus a small
    deterministic recency tie-breaker; the frozen ``legacy`` profile ignores it.
    ``expand_entities`` > 0 is an explicit opt-in appending up to
    ``MAX_ENTITY_EXPANSION`` records reachable within N entity hops from the
    direct hits over event_entities/entities (Chronicle-style graph expansion);
    the frozen ``legacy`` profile ignores it.
    ``mmr`` is an explicit opt-in reordering the head of the ranking by
    incremental MMR (relevance vs redundancy against already-selected entries);
    the ``passages`` profile reports it as skipped and the frozen ``legacy``
    profile ignores it.
    ``abstain_threshold`` > 0 is an explicit opt-in withholding candidates whose
    best absolute channel cosine (n-gram vector / neural embedding) falls below
    the threshold; when nothing survives, an empty list is returned and the
    ``abstain.abstained`` diagnostics flag is set. The frozen ``legacy``
    profile ignores it; 0.0 (default) is byte-identical to previous behavior.
    ``valid_at`` / ``known_at`` are explicit opt-in bitemporal filters (ISO
    8601, date shorthand allowed). ``valid_at`` returns only records valid at
    that valid-time point: Facts satisfy ``valid_from<=t`` and
    (``valid_to`` empty or ``>t``), Events are bounded by ``observed_at<=t``.
    ``known_at`` returns only records whose assertion Event was already
    recorded at that system-time point (``recorded_at<=t``). Both combine
    ("known by T1 and valid at T2"); with ``view="current"`` the retention
    ``expires_at`` check stays pinned to real now. The frozen ``legacy``
    profile ignores both; the default (both empty) is byte-identical.
    ``soft_supersede`` is an explicit opt-in guarded soft supersession
    (FlowGrid-style): window-closed superseded Fact versions stay eligible and
    are demoted by the fixed ``SOFT_SUPERSEDE_PENALTY`` in ranking instead of
    being excluded. It is ranking-layer only — lifecycle, forget/privacy
    semantics and retraction are untouched; retracted or expired memories
    never come back. The frozen ``legacy`` profile ignores it.
    ``temporal_intent`` is an explicit opt-in (batch 10) running the
    deterministic query-side extractor ``temporal_intent.extract_temporal_intent``
    over the query: an absolute ``as_of`` anchor ("as of March 2026", "during
    April 2022") fills ``valid_at`` and a comparison marker ("previously",
    "used to", "以前") turns on ``soft_supersede`` — but only for parameters
    the caller did not set explicitly. Relative anchors ("last week") are
    reported in diagnostics yet never applied (historical-timestamp corpora
    would be emptied by a now-anchored filter). With the flag off, or with no
    intent found, retrieval is byte-identical to previous behavior; the
    frozen ``legacy`` profile ignores it.
    ``window_reserve`` is an explicit opt-in (batch 12) changing *where*
    ``adjacent`` extras land: up to ``limit // 4`` tail slots of the evidence
    window are reserved for them (seed-rank order), displacing only the
    lowest-ranked direct hits, instead of appending every neighbour beyond
    the window where a ``results[:limit]`` caller (AML top_k=100) slices them
    away. It requires ``adjacent`` > 0; the frozen ``legacy`` profile and the
    default (off) are byte-identical to previous behavior.
    ``coverage_rerank`` is an explicit opt-in (batch 12) greedily reordering
    the ranking head by query-side coverage: candidates covering rare
    (corpus-IDF-weighted) query content words not yet covered by the
    already-selected set gain a bonus capped at COVERAGE_MAX_BONUS. The
    result set is a pure permutation of the previous head. The ``passages``
    profile reports it as skipped and the frozen ``legacy`` profile ignores
    it; default off is byte-identical.
    ``query_provider`` is an explicit opt-in (batch 13) LLM query-understanding
    provider (``query_provider.py``): its ``rewrite(query)`` returns query
    expansions — each becomes an extra FTS + substring recall channel — and a
    first-person HyDE recollection — one extra embedding channel when an
    embedding provider is also configured. Every extra channel is RRF-fused
    like the existing channels and can only add candidates, never remove
    original-query hits. A missing, failing, or misconfigured provider is a
    deterministic no-op (original query only): ids, scores, and ranking are
    identical to the frozen baseline, and results carry a
    ``query-rewrite-fallback:<reason>`` marker — the same observability
    precedent as the embedding fallback. The ``passages`` profile reports it
    as skipped and the frozen ``legacy`` profile ignores it.
    ``session_rrf`` is an explicit opt-in session-level RRF applied only to
    existing tagged source-event candidates. Add writes a one-way user-bound
    session digest only when this switch or ``session_views`` is enabled; facts
    and untagged records are untouched. The default is byte-identical.
    """
    if view not in {"current", "history", "both"}:
        raise ValueError("invalid memory view")
    if profile not in {"legacy", "lexical", "passages"}:
        raise ValueError("invalid retrieval profile")
    if not 0 <= adjacent <= MAX_ADJACENT_WINDOW:
        raise ValueError("invalid adjacent window")
    if not isinstance(recency_boost, bool):
        raise ValueError("invalid recency_boost flag")
    if isinstance(expand_entities, bool) or not isinstance(expand_entities, int) \
            or not 0 <= expand_entities <= MAX_ENTITY_HOPS:
        raise ValueError("invalid expand_entities hop count")
    if not isinstance(mmr, bool):
        raise ValueError("invalid mmr flag")
    if isinstance(abstain_threshold, bool) or not isinstance(abstain_threshold, (int, float)) \
            or not math.isfinite(float(abstain_threshold)) \
            or not 0.0 <= float(abstain_threshold) <= MAX_ABSTAIN_THRESHOLD:
        raise ValueError("invalid abstain_threshold")
    valid_point = _temporal_point(valid_at, "valid_at")
    known_point = _temporal_point(known_at, "known_at")
    if not isinstance(soft_supersede, bool):
        raise ValueError("invalid soft_supersede flag")
    if not isinstance(temporal_intent, bool):
        raise ValueError("invalid temporal_intent flag")
    if not isinstance(window_reserve, bool):
        raise ValueError("invalid window_reserve flag")
    if not isinstance(coverage_rerank, bool):
        raise ValueError("invalid coverage_rerank flag")
    if not isinstance(session_views, bool):
        raise ValueError("invalid session_views flag")
    if not isinstance(session_rrf, bool):
        raise ValueError("invalid session_rrf flag")
    intent: Optional[Dict[str, Any]] = None
    intent_applied = {"valid_at": False, "soft_supersede": False}
    if temporal_intent and profile != "legacy":
        intent = extract_temporal_intent(query)
        if intent["as_of"] and not valid_point:
            valid_point = _temporal_point(intent["as_of"], "valid_at")
            intent_applied["valid_at"] = True
        if intent["comparison"] and not soft_supersede:
            soft_supersede = True
            intent_applied["soft_supersede"] = True
    if scope in {"agent", "session"} and not scope_id:
        raise ValueError("scope-id is required for agent and session scope")
    info = diagnostics if diagnostics is not None else {}
    info.update({"profile": profile, "embedding": {"status": "disabled"}, "reranker": {"status": "disabled"},
                 "adjacent": {"status": "disabled"}, "recency_boost": {"status": "disabled"},
                 "entity_expansion": {"status": "disabled"}, "mmr": {"status": "disabled"},
                 "abstain": {"status": "disabled"}, "temporal": {"status": "disabled"},
                 "soft_supersede": {"status": "disabled"}, "temporal_intent": {"status": "disabled"},
                 "coverage_rerank": {"status": "disabled"}, "session_views": {"status": "disabled"},
                 "session_rrf": {"status": "disabled"},
                 "query_rewrite": {"status": "disabled"}})
    if intent is not None:
        info["temporal_intent"] = {
            "status": "enabled",
            "has_temporal_intent": intent["has_temporal_intent"],
            "matched": intent["matched"],
            "as_of": intent["as_of"],
            "valid_at_applied": intent_applied["valid_at"],
            "comparison": intent["comparison"],
            "soft_supersede_applied": intent_applied["soft_supersede"],
            "relative": intent["relative"],
            "relative_applied": False,
        }
    if limit <= 0:
        return []
    if profile == "legacy":
        if embedding_provider is not None or reranker is not None:
            info["model_policy"] = "ignored-by-legacy-profile"
        if adjacent:
            info["adjacent"] = {"status": "ignored-by-legacy-profile"}
        if recency_boost:
            info["recency_boost"] = {"status": "ignored-by-legacy-profile"}
        if expand_entities:
            info["entity_expansion"] = {"status": "ignored-by-legacy-profile"}
        if mmr:
            info["mmr"] = {"status": "ignored-by-legacy-profile"}
        if abstain_threshold:
            info["abstain"] = {"status": "ignored-by-legacy-profile"}
        if valid_point or known_point:
            info["temporal"] = {"status": "ignored-by-legacy-profile"}
        if soft_supersede:
            info["soft_supersede"] = {"status": "ignored-by-legacy-profile"}
        if temporal_intent:
            info["temporal_intent"] = {"status": "ignored-by-legacy-profile"}
        if window_reserve:
            info["adjacent"] = {"status": "ignored-by-legacy-profile"}
        if coverage_rerank:
            info["coverage_rerank"] = {"status": "ignored-by-legacy-profile"}
        if session_views:
            info["session_views"] = {"status": "ignored-by-legacy-profile"}
        if session_rrf:
            info["session_rrf"] = {"status": "ignored-by-legacy-profile"}
        if query_provider is not None:
            info["query_rewrite"] = {"status": "ignored-by-legacy-profile"}
        return _search_legacy(conn, project_id, query, limit, view, scope, scope_id)

    scope_id = scope_id or (project_id if scope == "project" else scope)
    now = utc_now()
    records = _eligible_records(conn, project_id, scope, scope_id, view, now,
                                valid_at=valid_point, known_at=known_point, soft_supersede=soft_supersede)
    info["eligible_records"] = len(records)
    if valid_point or known_point:
        info["temporal"] = {"status": "enabled",
                            "valid_at": valid_point or None, "known_at": known_point or None,
                            "semantics": "valid-time window + system-time recorded_at filter"}
    expanded, aliases = _scoped_aliases(conn, project_id, query, records)
    if recency_boost:
        info["recency_boost"] = {"status": "enabled", "half_life_days": RECENCY_HALF_LIFE_DAYS,
                                 "max_bonus": RECENCY_MAX_BONUS, "text_injection": "in-memory-scoring"}
    if profile == "passages":
        if reranker is not None:
            raise PLMError("passages profile does not yet support reranking")
        if mmr:
            info["mmr"] = {"status": "skipped-passages-profile"}
        if coverage_rerank:
            info["coverage_rerank"] = {"status": "skipped-passages-profile"}
        if query_provider is not None:
            info["query_rewrite"] = {"status": "skipped-passages-profile"}
        if session_views:
            info["session_views"] = {"status": "skipped-passages-profile"}
        if session_rrf:
            info["session_rrf"] = {"status": "skipped-passages-profile"}
        return _search_passages(records, project_id, expanded, limit, embedding_provider, info, aliases, view, adjacent,
                                recency_boost=recency_boost, now=now, conn=conn, expand_entities=expand_entities,
                                abstain_threshold=float(abstain_threshold),
                                soft_supersede=soft_supersede, penalty_point=valid_point or now,
                                window_reserve=window_reserve)
    ranked: Dict[Tuple[str, str], float] = defaultdict(float)
    reasons: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    fallbacks: List[str] = []
    # Opt-in LLM query understanding (batch 13): expansions become extra
    # lexical recall channels; the HyDE recollection becomes one extra
    # embedding channel below. Any provider failure degrades to the original
    # query only — a no-op against the frozen baseline.
    rewrite_expansions: List[str] = []
    rewrite_hyde = ""
    if query_provider is not None:
        info["query_rewrite"] = _describe(query_provider)
        try:
            rewrite = query_provider.rewrite(query)
            if not isinstance(rewrite, dict):
                raise ProviderUnavailable("invalid-rewrite-output")
            rewrite_expansions = [text for text in (rewrite.get("expansions") or [])
                                  if isinstance(text, str) and text.strip()]
            hyde_text = rewrite.get("hyde") or ""
            rewrite_hyde = hyde_text.strip() if isinstance(hyde_text, str) else ""
            if rewrite_expansions or rewrite_hyde:
                info["query_rewrite"] = dict(_describe(query_provider), status="ready",
                                             expansions_used=len(rewrite_expansions),
                                             hyde_used=bool(rewrite_hyde))
            else:
                info["query_rewrite"] = dict(_describe(query_provider), status="empty-output")
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ProviderUnavailable) else "provider-failed:" + type(exc).__name__
            info["query_rewrite"] = dict(_describe(query_provider), status="unavailable", fallback_reason=reason)
            fallbacks.append("query-rewrite-fallback:" + reason)
            rewrite_expansions, rewrite_hyde = [], ""
    terms = _lexical_terms(expanded)
    # Opt-in only: date expressions join the in-memory scoring text. The
    # persisted memory_fts content and stored vectors stay byte-identical, so
    # the default path and rebuild-index compatibility are unaffected.
    time_suffixes = {key: _time_text(records[key], now) for key in records} if recency_boost else None

    def record_text(key: Tuple[str, str]) -> str:
        text = _record_text(records[key])
        return text + " " + time_suffixes[key] if time_suffixes and time_suffixes.get(key) else text
    fts_terms = [term for term in terms if len(term) >= 3]
    if fts_terms:
        predicate, params = _eligibility(project_id, scope, scope_id, view, now, "memory_fts",
                                         valid_at=valid_point, known_at=known_point, soft_supersede=soft_supersede)
        fts = " OR ".join('"' + term.replace('"', '') + '"' for term in fts_terms[:64])
        try:
            rows = conn.execute(
                "SELECT record_type,ref_id,bm25(memory_fts,0.0,0.0,0.0,8.0,4.0,1.0) AS rank FROM memory_fts "
                "WHERE memory_fts MATCH ? AND project_id=? AND " + predicate + " ORDER BY rank,ref_id",
                tuple([fts, project_id, *params]),
            ).fetchall()
        except sqlite3.OperationalError:
            info["fts_fallback_reason"] = "fts-query-unavailable"
            rows = []
        eligible_hits = [row for row in rows if (row["record_type"], row["ref_id"]) in records][:80]
        for position, row in enumerate(eligible_hits, 1):
            key = (row["record_type"], row["ref_id"])
            if key in records:
                ranked[key] += 1.0 / (60.0 + position)
                reasons[key].append("fts5-bm25")
    for position, (_, key) in enumerate(_substring_ranking(records, terms, suffixes=time_suffixes)[:80], 1):
        ranked[key] += 1.0 / (60.0 + position)
        reasons[key].append("scoped-substring")

    query_vector = encode_vector(expanded)
    channel_best: Dict[Tuple[str, str], float] = {}

    # Session views are an additional deterministic recall channel, not an
    # output transform.  A response whose body has no words from a question
    # can still be found when its immediate conversational context does.  The
    # result remains that response's verbatim source event.
    if session_views:
        view_texts = _session_view_texts(records)
        view_fields = {key: {"title": "", "tags_json": "[]", "body": text}
                       for key, text in view_texts.items()}
        for position, (_, key) in enumerate(_substring_ranking(view_fields, terms)[:80], 1):
            ranked[key] += 1.0 / (60.0 + position)
            reasons[key].append("session-view-substring")
        view_vectors = []
        for key, text in view_texts.items():
            similarity = cosine_blob(query_vector, encode_vector(text))
            if similarity >= MIN_VECTOR_SIMILARITY:
                view_vectors.append((similarity, key))
        view_vectors.sort(key=lambda item: (-item[0], item[1]))
        for position, (similarity, key) in enumerate(view_vectors[:80], 1):
            ranked[key] += 1.0 / (60.0 + position)
            channel_best[key] = max(channel_best.get(key, 0.0), similarity)
            reasons[key].append("session-view-ngram:%.3f" % similarity)
        info["session_views"] = {"status": "ready", "views": len(view_texts),
                                 "radius": SESSION_VIEW_RADIUS, "candidates": len(view_vectors[:80])}
    # Extra lexical recall channels from query expansions (batch 13): each
    # expansion runs the same FTS + substring channels and is RRF-fused; it
    # can only add candidates, never remove original-query hits.
    for rewrite_index, expansion in enumerate(rewrite_expansions):
        expansion_terms = _lexical_terms(expansion)
        expansion_fts_terms = [term for term in expansion_terms if len(term) >= 3]
        if expansion_fts_terms:
            predicate, params = _eligibility(project_id, scope, scope_id, view, now, "memory_fts",
                                             valid_at=valid_point, known_at=known_point, soft_supersede=soft_supersede)
            fts = " OR ".join('"' + term.replace('"', '') + '"' for term in expansion_fts_terms[:64])
            try:
                rows = conn.execute(
                    "SELECT record_type,ref_id,bm25(memory_fts,0.0,0.0,0.0,8.0,4.0,1.0) AS rank FROM memory_fts "
                    "WHERE memory_fts MATCH ? AND project_id=? AND " + predicate + " ORDER BY rank,ref_id",
                    tuple([fts, project_id, *params]),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            eligible_hits = [row for row in rows if (row["record_type"], row["ref_id"]) in records][:80]
            for position, row in enumerate(eligible_hits, 1):
                key = (row["record_type"], row["ref_id"])
                if key in records:
                    ranked[key] += 1.0 / (60.0 + position)
                    reasons[key].append("query-expansion-%d-fts5-bm25" % rewrite_index)
        for position, (_, key) in enumerate(_substring_ranking(records, expansion_terms, suffixes=time_suffixes)[:80], 1):
            ranked[key] += 1.0 / (60.0 + position)
            reasons[key].append("query-expansion-%d-substring" % rewrite_index)

    if time_suffixes:
        # Stored vectors predate the injected date text, so with the switch on
        # the n-gram channel re-encodes the augmented text over the same
        # eligible-record universe instead of reading the vectors table.
        similarities = []
        for key in records:
            similarity = cosine_blob(query_vector, encode_vector(record_text(key)))
            if similarity >= MIN_VECTOR_SIMILARITY:
                similarities.append((similarity, key))
    else:
        predicate, params = _eligibility(project_id, scope, scope_id, view, now, "vectors",
                                         valid_at=valid_point, known_at=known_point, soft_supersede=soft_supersede)
        vector_rows = conn.execute("SELECT record_type,ref_id,vector FROM vectors WHERE project_id=? AND " + predicate,
                                   tuple([project_id, *params])).fetchall()
        similarities = []
        for row in vector_rows:
            similarity = cosine_blob(query_vector, row["vector"])
            key = (row["record_type"], row["ref_id"])
            if key in records and similarity >= MIN_VECTOR_SIMILARITY:
                similarities.append((similarity, key))
    similarities.sort(key=lambda item: (-item[0], item[1]))
    for similarity, key in similarities:
        channel_best[key] = max(channel_best.get(key, 0.0), similarity)
    for position, (similarity, key) in enumerate(similarities[:80], 1):
        ranked[key] += 1.0 / (60.0 + position)
        reasons[key].append("local-ngram-vector:%.3f" % similarity)

    semantic_vectors: Optional[Dict[Tuple[str, str], Any]] = None
    if embedding_provider is not None:
        info["embedding"] = _describe(embedding_provider)
        try:
            keys = sorted(records)
            if keys:
                passages = [record_text(key) for key in keys]
                if hasattr(embedding_provider, "embed_query") and hasattr(embedding_provider, "embed_documents"):
                    query_embedding = embedding_provider.embed_query(expanded)
                    embeddings = embedding_provider.embed_documents(passages)
                else:
                    encoded = embedding_provider.embed([expanded] + passages)
                    if len(encoded) != len(keys) + 1:
                        raise ProviderUnavailable("embedding-count-mismatch")
                    query_embedding, embeddings = encoded[0], encoded[1:]
                if len(embeddings) != len(keys):
                    raise ProviderUnavailable("embedding-count-mismatch")
                semantic_vectors = dict(zip(keys, embeddings))
                semantic = [(_cosine(query_embedding, value), key) for key, value in zip(keys, embeddings)]
                # Model cosine scales are not interchangeable; 0.0 is a minimal
                # candidate floor, not calibrated relevance or answer confidence.
                semantic = sorted((item for item in semantic if item[0] > 0.0), key=lambda item: (-item[0], item[1]))
                for similarity, key in semantic:
                    channel_best[key] = max(channel_best.get(key, 0.0), similarity)
                for position, (similarity, key) in enumerate(semantic[:80], 1):
                    ranked[key] += 1.0 / (60.0 + position)
                    reasons[key].append("local-semantic-vector:%.3f" % similarity)
                # Extra embedding recall channels from the query provider
                # (batch 13): each expansion embeds as a query; the HyDE
                # recollection embeds as a passage (ActiveMemoryIndex-style).
                extra_semantic: List[Tuple[str, str, bool]] = [
                    ("query-expansion-%d-semantic" % index, text, True)
                    for index, text in enumerate(rewrite_expansions)
                ]
                if rewrite_hyde:
                    extra_semantic.append(("hyde-semantic", rewrite_hyde, False))
                for label, text, as_query in extra_semantic:
                    try:
                        if hasattr(embedding_provider, "embed_query") and hasattr(embedding_provider, "embed_documents"):
                            extra_embedding = (embedding_provider.embed_query(text) if as_query
                                               else embedding_provider.embed_documents([text])[0])
                        else:
                            extra_embedding = embedding_provider.embed([text])[0]
                        extra_ranked = [(_cosine(extra_embedding, value), key)
                                        for key, value in zip(keys, embeddings)]
                        extra_ranked = sorted((item for item in extra_ranked if item[0] > 0.0),
                                              key=lambda item: (-item[0], item[1]))
                        for similarity, key in extra_ranked:
                            channel_best[key] = max(channel_best.get(key, 0.0), similarity)
                        for position, (similarity, key) in enumerate(extra_ranked[:80], 1):
                            ranked[key] += 1.0 / (60.0 + position)
                            reasons[key].append(label + ":%.3f" % similarity)
                    except Exception:
                        # One broken rewrite channel must not sink the main
                        # embedding channel; degradation is per-channel.
                        continue
                info["embedding"] = dict(_describe(embedding_provider), status="ready", candidates=len(semantic[:80]))
            else:
                info["embedding"]["status"] = "skipped-no-eligible-records"
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ProviderUnavailable) else "provider-failed:" + type(exc).__name__
            info["embedding"] = dict(_describe(embedding_provider), status="unavailable", fallback_reason=reason)
            fallbacks.append("embedding-fallback:" + reason)
            if diagnostics is None:
                warnings.warn("PLM embedding fallback: " + reason, RuntimeWarning, stacklevel=2)

    if session_rrf:
        info["session_rrf"] = dict({"status": "ready", "rrf_k": SESSION_RRF_K},
                                    **_session_rrf_adjust(ranked, records, reasons))

    results = []
    penalty_point = valid_point or now
    soft_penalized = 0
    for key, base_score in ranked.items():
        record = records[key]
        title_lower = record["title"].lower()
        query_lower = query.strip().lower()
        title_bonus = 0.03 if query_lower == title_lower else (0.015 if query_lower and (query_lower in title_lower or title_lower in query_lower) else 0.0)
        kind_bonus = 0.008 if key[0] == "fact" and view == "current" else {"core": 0.007, "procedure": 0.004, "artifact_ref": 0.003}.get(record["kind"], 0.0)
        recency = _recency_bonus(record, now) if recency_boost else 0.0
        # Guarded soft supersession: a superseded Fact whose validity window is
        # closed at the query point keeps its channel evidence but is demoted by
        # a fixed constant. A version still valid at ``valid_at`` is a proper
        # as-of answer and is never penalised.
        soft_penalty = 0.0
        if (soft_supersede and key[0] == "fact" and record["status"] == "superseded"
                and record["valid_to"] and record["valid_to"] <= penalty_point):
            soft_penalty = SOFT_SUPERSEDE_PENALTY
            soft_penalized += 1
        results.append(SearchResult(
            key[1], key[0], record["title"], record["body"], project_id, base_score + title_bonus + kind_bonus + recency - soft_penalty,
            reasons[key] + (["entity-alias"] if aliases else []) + (["effective-current"] if key[0] == "fact" and view == "current" and not soft_penalty else [])
            + (["recency-boost:%.6f" % recency] if recency_boost else [])
            + (["soft-supersede:-%.6f" % soft_penalty] if soft_penalty else []) + fallbacks,
            record["source_path"], record["status"], record["valid_from"], record["valid_to"],
            record.get("source_event_id") or record["assertion_event_id"],
        ))
    if soft_supersede:
        info["soft_supersede"] = {"status": "enabled", "penalty": SOFT_SUPERSEDE_PENALTY,
                                  "penalized": soft_penalized, "penalty_point": penalty_point}
    results.sort(key=lambda item: (-item.score, item.record_type != "fact", item.title, item.ref_id))
    if reranker is not None:
        info["reranker"] = _describe(reranker)
        candidates = results[:80]
        try:
            if candidates:
                scores = reranker.score(expanded, [record_text((item.record_type, item.ref_id)) for item in candidates])
                if len(scores) != len(candidates) or any(not math.isfinite(float(score)) for score in scores):
                    raise ProviderUnavailable("invalid-reranker-output")
                for result, score in zip(candidates, scores):
                    result.score = float(score)
                    result.reasons.append("local-reranker:%.6f" % float(score))
                candidates.sort(key=lambda item: (-item.score, item.record_type != "fact", item.title, item.ref_id))
                results = candidates
                info["reranker"] = dict(_describe(reranker), status="ready", candidates=len(candidates))
            else:
                info["reranker"]["status"] = "skipped-no-candidates"
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ProviderUnavailable) else "provider-failed:" + type(exc).__name__
            info["reranker"] = dict(_describe(reranker), status="unavailable", fallback_reason=reason)
            if diagnostics is None:
                warnings.warn("PLM reranker fallback: " + reason, RuntimeWarning, stacklevel=2)
            for result in results:
                result.reasons.append("reranker-fallback:" + reason)
    if mmr:
        results = _mmr_select(results, records, limit, info, semantic_vectors=semantic_vectors)
    if coverage_rerank:
        # Query-side coverage runs after result-side MMR: MMR first strips
        # redundant near-duplicates, coverage then promotes candidates holding
        # still-uncovered rare query words. Both only permute the pool head.
        results = _coverage_rerank(results, records, expanded, limit, info)
    if abstain_threshold:
        # Withhold candidates no absolute-similarity channel supports at the
        # calibrated floor; adjacent/entity supplements below are moot when no
        # direct hit survives, so an empty kept-list is the abstention signal.
        kept = [item for item in results
                if channel_best.get((item.record_type, item.ref_id), 0.0) >= abstain_threshold]
        info["abstain"] = {"status": "enabled", "threshold": float(abstain_threshold),
                           "signal": "max-channel-cosine", "candidates": len(results),
                           "kept": len(kept), "abstained": not kept}
        results = kept
    hits = results[:limit]
    if adjacent:
        if window_reserve:
            hits = _adjacent_augment_reserved(records, project_id, hits, limit, adjacent, info)
        else:
            hits = _adjacent_augment(records, project_id, hits, adjacent, info)
    elif window_reserve:
        info["adjacent"] = {"status": "skipped-reserve-requires-adjacent"}
    if expand_entities:
        extras = _entity_expand(conn, records, project_id, hits, expand_entities, info)
        hits, in_window = _merge_expansion(hits, extras, limit)
        info["entity_expansion"] = {"status": "ready", "hops": expand_entities, "added": len(extras),
                                    "in_window": in_window, "cap": MAX_ENTITY_EXPANSION}
    return hits


def _search_passages(records, project_id, query, limit, provider, info, aliases, view, adjacent=0,
                     recency_boost=False, now="", conn=None, expand_entities=0, abstain_threshold=0.0,
                     soft_supersede=False, penalty_point="", window_reserve=False):
    """Rank exact slices only after parent/scope/lifecycle filtering.

    No persistent passage index or plaintext cache is created. Parent grouping
    prevents a single long conversation from occupying every result slot. The
    limits are explicit failures, never silently incomplete corpus retrieval.
    With ``recency_boost`` only the parent-level micro-nudge is applied; exact
    span text is never polluted with injected date expressions, because a
    repeated parent date on every chunk would be false evidence localization.
    ``soft_supersede`` demotes window-closed superseded parent Facts by the
    fixed ``SOFT_SUPERSEDE_PENALTY`` after grouping; lifecycle is untouched.
    """
    if sum(len(row["body"]) for row in records.values()) > 8_000_000:
        raise PLMError("passage corpus exceeds 8000000 character work limit")
    chunks = {}
    parents = {}
    for key in sorted(records):
        row = records[key]
        for span in split_passages(row["body"], row["assertion_event_id"]):
            span_id = span["passage_id"]
            chunks[span_id] = span
            parents[span_id] = key
            if len(chunks) > 8192:
                raise PLMError("passage corpus exceeds 8192 passage work limit")
    info["passages"] = {"status": "ready", "count": len(chunks), "parent_count": len(records),
                        "max_chars": 900, "overlap_chars": 120, "cache": "none"}
    ranked = defaultdict(float)
    reasons = defaultdict(list)
    def diverse_channel(items):
        counts = defaultdict(int)
        output = []
        for item in items:
            parent = parents[item[1]]
            if counts[parent] >= 2:
                continue
            counts[parent] += 1
            output.append(item)
            if len(output) == 160:
                break
        return output
    # Chunk body is the primary signal. Repeating a parent's title on every
    # chunk would turn an early, generic title into false evidence localization.
    fields = {key: {"title": "", "tags_json": "[]", "body": span["text"]} for key, span in chunks.items()}
    for position, (_, key) in enumerate(diverse_channel(_substring_ranking(fields, _lexical_terms(query))), 1):
        ranked[key] += 1.0 / (60 + position)
        reasons[key].append("passage-substring")
    query_vector = encode_vector(query)
    vectors = [(cosine_blob(query_vector, encode_vector(span["text"])), key) for key, span in chunks.items()]
    vectors = sorted((item for item in vectors if item[0] >= MIN_VECTOR_SIMILARITY), key=lambda item: (-item[0], item[1]))
    parent_best: Dict[Tuple[str, str], float] = {}
    for score, key in vectors:
        parent = parents[key]
        parent_best[parent] = max(parent_best.get(parent, 0.0), score)
    for position, (score, key) in enumerate(diverse_channel(vectors), 1):
        ranked[key] += 1.0 / (60 + position)
        reasons[key].append("passage-ngram:%.3f" % score)
    fallback = []
    if provider is not None:
        try:
            keys = sorted(chunks)
            if keys:
                texts = [chunks[key]["text"] for key in keys]
                if hasattr(provider, "embed_query") and hasattr(provider, "embed_documents"):
                    query_embedding = provider.embed_query(query)
                    embeddings = provider.embed_documents(texts)
                else:
                    encoded = provider.embed([query] + texts)
                    if len(encoded) != len(keys) + 1:
                        raise ProviderUnavailable("embedding-count-mismatch")
                    query_embedding, embeddings = encoded[0], encoded[1:]
                if len(embeddings) != len(keys):
                    raise ProviderUnavailable("embedding-count-mismatch")
                semantic = [(_cosine(query_embedding, value), key) for key, value in zip(keys, embeddings)]
                semantic = sorted((item for item in semantic if item[0] > 0), key=lambda item: (-item[0], item[1]))
                for score, key in semantic:
                    parent = parents[key]
                    parent_best[parent] = max(parent_best.get(parent, 0.0), score)
                for position, (score, key) in enumerate(diverse_channel(semantic), 1):
                    ranked[key] += 1.0 / (60 + position)
                    reasons[key].append("passage-semantic:%.3f" % score)
                info["embedding"] = dict(_describe(provider), status="ready", candidates=len(semantic[:160]))
            else:
                info["embedding"] = {"status": "skipped-no-eligible-records"}
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ProviderUnavailable) else "provider-failed:" + type(exc).__name__
            info["embedding"] = dict(_describe(provider), status="unavailable", fallback_reason=reason)
            fallback = ["embedding-fallback:" + reason]
    # Exact Fact keys are structured identifiers, not generic session titles.
    # Only compact Facts can be localized directly from that identifier alone.
    for span_id, span in chunks.items():
        row = records[parents[span_id]]
        if row["record_type"] == "fact" and len(row["body"]) <= 900 and row["title"].casefold() == query.strip().casefold():
            ranked[span_id] += 2.0 / 61
            reasons[span_id].append("exact-fact-key")
    grouped = {}
    for span_id in sorted(ranked, key=lambda key: (-ranked[key], key)):
        key = parents[span_id]
        row = records[key]
        if key not in grouped:
            grouped[key] = SearchResult(key[1], key[0], row["title"], row["body"], project_id, ranked[span_id],
                reasons[span_id] + (["entity-alias"] if aliases else []) + (["effective-current"] if key[0] == "fact" and view == "current" else []) + fallback,
                row["source_path"], row["status"], row["valid_from"], row["valid_to"], row["assertion_event_id"])
        result = grouped[key]
        span = chunks[span_id]
        # Suppress nearly identical overlapping windows, not distant evidence.
        if len(result.evidence_spans) < 2 and all(
            max(0, min(span["end"], old["end"]) - max(span["start"], old["start"])) < .5 * min(span["end"] - span["start"], old["end"] - old["start"])
            for old in result.evidence_spans
        ):
            result.evidence_spans.append(span)
    if abstain_threshold:
        kept = {key: result for key, result in grouped.items()
                if parent_best.get(key, 0.0) >= abstain_threshold}
        info["abstain"] = {"status": "enabled", "threshold": float(abstain_threshold),
                           "signal": "max-channel-cosine", "candidates": len(grouped),
                           "kept": len(kept), "abstained": not kept}
        grouped = kept
    if recency_boost:
        info["recency_boost"] = dict(info.get("recency_boost", {}), text_injection="skipped-exact-span-profile")
        for key, result in grouped.items():
            bonus = _recency_bonus(records[key], now)
            result.score += bonus
            result.reasons.append("recency-boost:%.6f" % bonus)
    if soft_supersede:
        penalized = 0
        for key, result in grouped.items():
            row = records[key]
            if (key[0] == "fact" and row["status"] == "superseded"
                    and row["valid_to"] and row["valid_to"] <= penalty_point):
                result.score -= SOFT_SUPERSEDE_PENALTY
                result.reasons = [reason for reason in result.reasons if reason != "effective-current"]
                result.reasons.append("soft-supersede:-%.6f" % SOFT_SUPERSEDE_PENALTY)
                penalized += 1
        info["soft_supersede"] = {"status": "enabled", "penalty": SOFT_SUPERSEDE_PENALTY,
                                  "penalized": penalized, "penalty_point": penalty_point}
    if recency_boost or soft_supersede:
        hits = sorted(grouped.values(), key=lambda item: (-item.score, item.record_type != "fact", item.title, item.ref_id))[:limit]
    else:
        hits = list(grouped.values())[:limit]
    if adjacent:
        if window_reserve:
            hits = _adjacent_augment_reserved(records, project_id, hits, limit, adjacent, info, passage_mode=True)
        else:
            hits = _adjacent_augment(records, project_id, hits, adjacent, info, passage_mode=True)
    elif window_reserve:
        info["adjacent"] = {"status": "skipped-reserve-requires-adjacent"}
    if expand_entities and conn is not None:
        extras = _entity_expand(conn, records, project_id, hits, expand_entities, info, passage_mode=True)
        hits, in_window = _merge_expansion(hits, extras, limit)
        info["entity_expansion"] = {"status": "ready", "hops": expand_entities, "added": len(extras),
                                    "in_window": in_window, "cap": MAX_ENTITY_EXPANSION}
    return hits
