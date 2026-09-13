"""Isolated retrieval diagnostics; gold labels never enter the memory store.

This module neither calls a reader model nor claims benchmark QA accuracy.
Synthetic fixtures and external LongMemEval data use the same source-only ingest
boundary, but their protocols and reported results remain explicitly distinct.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .database import connect_readonly
from .model import PLMError
from .search import search
from .service import write_event


PROTOCOL_VERSION = "plm-retrieval-diagnostic-v1"
SOURCE_FIELDS = {"source_id", "project", "split", "title", "body", "observed_at", "scope", "scope_id", "tags"}
BASELINES = {"none", "raw", "legacy", "lexical", "embedding", "reranker", "provider", "recall-template"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint() -> str:
    files = sorted(Path(__file__).parent.glob("*.py"))
    return fingerprint({path.name: _file_hash(path) for path in files})


def load_dataset(folder: Path, split: str) -> Dict[str, Any]:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if split not in manifest["splits"]:
        raise PLMError("unknown evaluation split")
    for name, digest in manifest["sha256"].items():
        if Path(name).name != name or _file_hash(folder / name) != digest:
            raise PLMError("evaluation fixture fingerprint mismatch: " + name)
    sources = json.loads((folder / "sources.json").read_text(encoding="utf-8"))
    all_questions = []
    for partition in manifest["splits"]:
        all_questions.extend(json.loads((folder / ("questions." + partition + ".json")).read_text(encoding="utf-8")))
    source_ids = set()
    project_splits: Dict[str, str] = {}
    by_id = {}
    for source in sources:
        if set(source) - SOURCE_FIELDS:
            raise PLMError("source contains unknown fields or evaluation labels")
        if not all(source.get(key) for key in ("source_id", "project", "split", "title", "body")):
            raise PLMError("incomplete evaluation source")
        if source["source_id"] in source_ids:
            raise PLMError("duplicate source id")
        if project_splits.setdefault(source["project"], source["split"]) != source["split"]:
            raise PLMError("project appears in more than one split")
        source_ids.add(source["source_id"])
        by_id[source["source_id"]] = source
    question_ids = set()
    for question in all_questions:
        if question["question_id"] in question_ids:
            raise PLMError("duplicate question id")
        question_ids.add(question["question_id"])
        if project_splits.get(question["project"]) != question["split"]:
            raise PLMError("query project does not belong to its split")
        gold = question["required_source_ids"]
        if bool(gold) == bool(question.get("unanswerable", False)):
            raise PLMError("answerability and evidence labels disagree")
        for source_id in gold:
            source = by_id.get(source_id)
            if not source or (source["project"], source["split"]) != (question["project"], question["split"]):
                raise PLMError("gold evidence crosses a project or split")
            if source.get("scope", "project") != question.get("scope", "project") or source.get("scope_id", "") != question.get("scope_id", ""):
                raise PLMError("gold evidence crosses a scope")
    return {
        "manifest": manifest,
        "sources": [source for source in sources if source["split"] == split],
        "questions": [question for question in all_questions if question["split"] == split],
        "fingerprint": fingerprint(manifest),
    }


@contextlib.contextmanager
def isolated_memory():
    """Always replace, then restore, the caller's memory location.

    Process-global environment makes this context intentionally non-thread-safe.
    Run concurrent evaluations in separate processes.
    """
    previous = os.environ.get("PROJECT_LONG_MEMORY_DIR")
    with tempfile.TemporaryDirectory(prefix="plm-evaluation-") as directory:
        root = Path(directory)
        os.environ["PROJECT_LONG_MEMORY_DIR"] = str(root / "memory")
        try:
            yield root
        finally:
            if previous is None:
                os.environ.pop("PROJECT_LONG_MEMORY_DIR", None)
            else:
                os.environ["PROJECT_LONG_MEMORY_DIR"] = previous


def ingest_sources(root: Path, sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accept only sources, never a question, answer, gold label or dataset dict."""
    started = time.perf_counter()
    projects = {}
    event_to_source = {}
    for source in sources:
        if set(source) - SOURCE_FIELDS:
            raise PLMError("only source records may enter evaluation storage")
        project_key = source["project"]
        # Names from external datasets cannot become filesystem paths.
        folder = root / "workspace" / hashlib.sha256(project_key.encode("utf-8")).hexdigest()[:16]
        folder.mkdir(parents=True, exist_ok=True)
        event = write_event(
            folder, source["title"], source["body"], tags=source.get("tags", []),
            scope=source.get("scope", "project"), scope_id=source.get("scope_id", ""),
            observed_at=source.get("observed_at", "2025-01-01T00:00:00Z"),
            recorded_at=source.get("observed_at", "2025-01-01T00:00:00Z"),
            source="synthetic_evaluation" if source["split"] != "official" else "external_evaluation_history",
            created_by="evaluation", idempotency_key="eval-source:" + source["source_id"],
            event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "plm-eval:" + source["source_id"])),
        )
        projects[project_key] = event.project_id
        event_to_source[event.event_id] = source["source_id"]
    return {
        "projects": projects, "event_to_source": event_to_source,
        "ingest_ms": (time.perf_counter() - started) * 1000,
        "bytes_on_disk": sum(path.stat().st_size for path in (root / "memory").rglob("*") if path.is_file()),
        "source_count": len(event_to_source),
    }


def _features(text: str) -> Counter:
    normalized = re.sub(r"\s+", "", text.lower())
    result = Counter(normalized[i:i + 2] for i in range(max(0, len(normalized) - 1)))
    result.update(re.findall(r"[a-z0-9._-]{2,}", text.lower()))
    return result


def raw_episode_search(sources: Sequence[Dict[str, Any]], query: Dict[str, Any], limit: int) -> List[str]:
    """Frozen, no-expansion character-bigram TF cosine baseline (body only)."""
    terms = _features(query["question"])
    norm = math.sqrt(sum(count * count for count in terms.values()))
    ranked = []
    for source in sources:
        if source["project"] != query["project"] or source.get("scope", "project") != query.get("scope", "project") or source.get("scope_id", "") != query.get("scope_id", ""):
            continue
        candidate = _features(source["body"])
        denominator = norm * math.sqrt(sum(count * count for count in candidate.values()))
        score = sum(count * candidate[term] for term, count in terms.items()) / denominator if denominator else 0
        if score > 0:
            ranked.append((score, source["source_id"]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [source_id for _, source_id in ranked[:limit]]


def pack_evidence(query: Dict[str, Any], source_ids: Sequence[str], by_id: Dict[str, Dict[str, Any]], token_budget: int) -> Dict[str, Any]:
    # Conservative UTF-8-byte budget: no tokenizer/model-specific token claim.
    # Each UTF-8 byte counts as one upper-bound budget unit, including prompt.
    header = "依据给定来源作答；依据不足请说明。\n"
    if query.get("question_date"):
        header += "提问时间：" + str(query["question_date"]) + "\n"
    header += "问题：" + query["question"] + "\n"
    chunks = [header]
    used = len(header.encode("utf-8"))
    retained = []
    dropped = []
    for source_id in dict.fromkeys(source_ids):
        source = by_id.get(source_id)
        if not source:
            continue
        chunk = "\n[" + source_id + "] " + source["title"] + "\n" + source["body"] + "\n"
        size = len(chunk.encode("utf-8"))
        if used + size <= token_budget:
            chunks.append(chunk)
            used += size
            retained.append(source_id)
        else:
            dropped.append(source_id)
    return {"source_ids": retained, "dropped_source_ids": dropped, "context": "".join(chunks) if used <= token_budget else "", "budget_units": used if used <= token_budget else 0}


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percentile
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 6)


def score_evidence(question: Dict[str, Any], retrieved: Sequence[str], retained: Sequence[str], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    gold = set(question["required_source_ids"])
    hits = gold & set(retained)
    failures = []
    if not gold:
        if retained:
            failures.append("unanswerable_false_retrieval")
    elif not hits:
        failures.append("recall_miss")
    elif hits != gold:
        failures.append("incomplete_evidence")
    if (gold & set(retrieved)) - set(retained):
        failures.append("context_budget_drop")
    for source_id in retained:
        source = by_id.get(source_id)
        if not source or source["project"] != question["project"] or source.get("scope", "project") != question.get("scope", "project") or source.get("scope_id", "") != question.get("scope_id", ""):
            failures.append("scope_error")
            break
    if set(retained) & set(question.get("forbidden_source_ids", [])):
        failures.append("forbidden_time_evidence")
    return {
        "answerable": bool(gold), "evidence_recall": len(hits) / len(gold) if gold else None,
        "retrieval_recall_before_budget": len(gold & set(retrieved)) / len(gold) if gold else None,
        "full_evidence_coverage": bool(gold) and hits == gold,
        "unanswerable_false_retrieval": not gold and bool(retained),
        "failures": sorted(set(failures)),
    }


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    absent = [row for row in rows if not row["answerable"]]
    return {
        "questions": len(rows), "answerable": len(answerable), "unanswerable": len(absent),
        "evidence_recall_macro": sum(row["evidence_recall"] for row in answerable) / len(answerable) if answerable else None,
        "retrieval_recall_before_budget_macro": sum(row["retrieval_recall_before_budget"] for row in answerable) / len(answerable) if answerable else None,
        "full_evidence_coverage": sum(row["full_evidence_coverage"] for row in answerable) / len(answerable) if answerable else None,
        "unanswerable_false_retrieval_rate": sum(row["unanswerable_false_retrieval"] for row in absent) / len(absent) if absent else None,
        "failure_counts": dict(Counter(failure for row in rows for failure in row["failures"])),
    }


def _provider_info(provider: Any) -> Any:
    if provider is None:
        return None
    value = {"class": type(provider).__module__ + "." + type(provider).__qualname__}
    if callable(getattr(provider, "describe", None)):
        description = provider.describe()
        if isinstance(description, dict):
            value.update({key: item for key, item in description.items() if key not in {"status", "fallback_reason", "cache_hits", "cache_misses"}})
    for attr in ("model_id", "revision", "dimensions", "index_version", "metadata", "evaluation_runtime"):
        item = getattr(provider, attr, None)
        if item is not None and not callable(item):
            try:
                json.dumps(item)
                value[attr] = item
            except TypeError:
                value[attr] = str(item)
    return value


def _runtime_state(provider: Any) -> Any:
    if provider is None or not callable(getattr(provider, "describe", None)):
        return None
    details = provider.describe()
    return {key: details[key] for key in ("status", "cache_hits", "cache_misses") if key in details}


def run_evaluation(dataset_path: Path, split: str = "dev", baselines: Sequence[str] = ("none", "raw", "legacy", "lexical"), limit: int = 5, token_budget: int = 1800, repeats: int = 3, embedding_provider: Any = None, reranker: Any = None, reader_inputs: Optional[Path] = None, query_provider: Any = None) -> Dict[str, Any]:
    if limit < 1 or token_budget < 1 or repeats < 1 or not set(baselines) <= BASELINES:
        raise PLMError("invalid evaluation configuration")
    dataset = load_dataset(dataset_path, split)
    questions = dataset["questions"]
    sources = dataset["sources"]
    by_id = {source["source_id"]: source for source in sources}
    config = {"limit": limit, "token_budget": token_budget, "repeats": repeats, "baselines": list(baselines), "embedding_provider": _provider_info(embedding_provider), "reranker": _provider_info(reranker), "query_provider": _provider_info(query_provider)}
    report: Dict[str, Any] = {
        "protocol": PROTOCOL_VERSION, "dataset": dataset["manifest"]["dataset_id"],
        "origin": dataset["manifest"]["origin"], "split": split, "official_benchmark_score": False,
        "heldout_content_hidden": split == "heldout" and reader_inputs is None,
        "fingerprints": {"dataset": dataset["fingerprint"], "code": code_fingerprint(), "config": fingerprint(config), "protocol": PROTOCOL_VERSION},
        "configuration": config, "qa": {"status": "not_measured", "reason": "No fixed external reader predictions were evaluated."},
        "budget_accounting": "UTF-8 byte upper bound, including query, instructions, labels, titles and complete evidence; not model-token measurement",
        "latency_boundary": "query dispatch through database open/retrieval/provider calls/evidence packing; excludes reader, process startup and source ingestion. first_pass does not mean OS-cache-cold",
        "cache_policy": "provider instances are reused in listed baseline order; later first_pass timings can reuse earlier model loads/numeric cache. Use separate processes and factories for independent cold-load comparisons.",
        "results": {},
    }
    exported_inputs = []
    with isolated_memory() as root:
        ingest = ingest_sources(root, sources)
        report["build"] = {key: ingest[key] for key in ("ingest_ms", "bytes_on_disk", "source_count")}
        search_parameters = inspect.signature(search).parameters
        profile_supported = "profile" in search_parameters
        for baseline in baselines:
            use_embedding = baseline in {"embedding", "provider", "recall-template"}
            use_reranker = baseline in {"reranker", "provider"}
            use_query_provider = baseline == "recall-template"
            required_components = (["embedding"] if use_embedding else []) + (["reranker"] if use_reranker else []) + (["query_rewrite"] if use_query_provider else [])
            if baseline in {"lexical", "embedding", "reranker", "provider", "recall-template"} and not profile_supported:
                report["results"][baseline] = {"status": "not_measured", "reason": "search profile interface unavailable"}
                continue
            if ((use_embedding and embedding_provider is None) or (use_reranker and reranker is None)
                    or (use_query_provider and query_provider is None)):
                report["results"][baseline] = {"status": "not_measured", "reason": "requested model component not supplied; no model downloaded", "requested_components": required_components}
                continue
            if required_components and "diagnostics" not in search_parameters:
                report["results"][baseline] = {"status": "not_measured", "reason": "provider diagnostics unavailable; cannot distinguish model execution from fallback"}
                continue
            rows = []
            first_latencies = []
            warm_latencies = []
            component_statuses = {name: Counter() for name in required_components}
            fallback_counts = Counter()
            degraded = False
            state_before = {"embedding": _runtime_state(embedding_provider) if use_embedding else None, "reranker": _runtime_state(reranker) if use_reranker else None, "query_rewrite": _runtime_state(query_provider) if use_query_provider else None}
            try:
                for repetition in range(repeats):
                    for question in questions:
                        started = time.perf_counter()
                        if baseline == "none":
                            retrieved = []
                        elif baseline == "raw":
                            retrieved = raw_episode_search(sources, question, limit)
                        else:
                            conn = connect_readonly()
                            try:
                                kwargs = {"limit": limit, "view": question.get("view", "current"), "scope": question.get("scope", "project"), "scope_id": question.get("scope_id", "")}
                                if profile_supported:
                                    kwargs["profile"] = "legacy" if baseline == "legacy" else "lexical"
                                diagnostics: Dict[str, Any] = {}
                                if "diagnostics" in search_parameters:
                                    kwargs["diagnostics"] = diagnostics
                                if use_embedding:
                                    kwargs["embedding_provider"] = embedding_provider
                                if use_reranker:
                                    kwargs["reranker"] = reranker
                                if use_query_provider:
                                    kwargs["query_provider"] = query_provider
                                results = search(conn, ingest["projects"][question["project"]], question["question"], **kwargs)
                                for component in required_components:
                                    details = diagnostics.get(component, {})
                                    status = details.get("status", "unknown")
                                    component_statuses[component][status] += 1
                                    if status not in {"ready", "skipped-no-candidates", "skipped-no-eligible-records"}:
                                        degraded = True
                                        # Only known machine status labels enter output;
                                        # provider exception messages may contain input.
                                        reason = str(details.get("fallback_reason", "missing-provider-execution-evidence"))
                                        safe_reason = reason if re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", reason) else "provider-failure-redacted"
                                        fallback_counts[component + ":" + safe_reason] += 1
                                retrieved = [ingest["event_to_source"].get(result.ref_id, "unmapped:" + result.ref_id) for result in results]
                            finally:
                                conn.close()
                        packed = pack_evidence(question, retrieved, by_id, token_budget)
                        elapsed = (time.perf_counter() - started) * 1000
                        (first_latencies if repetition == 0 else warm_latencies).append(elapsed)
                        if repetition == 0:
                            row = score_evidence(question, retrieved, packed["source_ids"], by_id)
                            row.update({"question_id": question["question_id"], "category": question["category"], "budget_units": packed["budget_units"]})
                            if split != "heldout":
                                row.update({"question": question["question"], "retrieved_source_ids": retrieved, "retained_source_ids": packed["source_ids"]})
                            rows.append(row)
                            if reader_inputs is not None:
                                exported_inputs.append({"question_id": question["question_id"], "baseline": baseline, "question": question["question"], "question_date": question.get("question_date", ""), "context": packed["context"], "dataset_fingerprint": dataset["fingerprint"], "config_fingerprint": report["fingerprints"]["config"]})
                grouped = defaultdict(list)
                for row in rows:
                    grouped[row["category"]].append(row)
                result = {"status": "degraded" if degraded else "measured", "overall": _aggregate(rows), "by_category": {name: _aggregate(items) for name, items in sorted(grouped.items())}, "latency_ms": {"first_pass": {"p50": _percentile(first_latencies, .5), "p95": _percentile(first_latencies, .95), "n": len(first_latencies)}, "warm": {"p50": _percentile(warm_latencies, .5), "p95": _percentile(warm_latencies, .95), "n": len(warm_latencies)}}, "average_budget_units": sum(row["budget_units"] for row in rows) / len(rows) if rows else 0, "requested_components": required_components, "component_statuses": {name: dict(counts) for name, counts in component_statuses.items()}, "fallback_counts": dict(fallback_counts), "unqualified_model_comparison_valid": bool(required_components) and not degraded}
                if split != "heldout":
                    result["cases"] = rows
                result["provider_state_before"] = state_before
                result["provider_state_after"] = {"embedding": _runtime_state(embedding_provider) if use_embedding else None, "reranker": _runtime_state(reranker) if use_reranker else None, "query_rewrite": _runtime_state(query_provider) if use_query_provider else None}
                report["results"][baseline] = result
            except Exception as exc:
                # Never turn an absent/broken optional provider into a lexical win.
                report["results"][baseline] = {"status": "failed", "error_type": type(exc).__name__}
    # Resolve artifact identity after lazy models have loaded. Cache counters and
    # operational status are excluded so the configuration identity is stable.
    config.update({"embedding_provider": _provider_info(embedding_provider), "reranker": _provider_info(reranker), "query_provider": _provider_info(query_provider)})
    report["fingerprints"]["config"] = fingerprint(config)
    if reader_inputs is not None:
        for item in exported_inputs:
            item["config_fingerprint"] = report["fingerprints"]["config"]
        Path(reader_inputs).write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in exported_inputs), encoding="utf-8")
    return report


def score_reader_predictions(questions: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]], reader_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Score supplied fixed-reader outputs only; exact match is not semantic QA."""
    required = {"reader_id", "revision", "prompt_sha256", "decoding", "dataset_fingerprint"}
    if not required <= set(reader_manifest) or not re.fullmatch(r"[a-fA-F0-9]{64}", str(reader_manifest["prompt_sha256"])):
        raise PLMError("fixed reader identity, revision, prompt hash, decoding and dataset fingerprint are required")
    expected_ids = {question["question_id"] for question in questions}
    mapping = {}
    for prediction in predictions:
        qid = prediction["question_id"]
        if qid in mapping or qid not in expected_ids or "hypothesis" not in prediction:
            raise PLMError("duplicate, unknown or incomplete reader prediction")
        mapping[qid] = prediction
    normalize = lambda value: re.sub(r"\s+", "", str(value)).casefold()
    correct = 0
    abstention_correct = 0
    absent = sum(bool(question.get("unanswerable")) for question in questions)
    for question in questions:
        prediction = mapping.get(question["question_id"])
        if prediction is None:
            continue
        if question.get("unanswerable"):
            hit = prediction.get("abstained") is True
            abstention_correct += hit
        else:
            answers = question.get("acceptable_answers", [question["answer"]])
            hit = prediction.get("abstained") is not True and normalize(prediction["hypothesis"]) in {normalize(answer) for answer in answers}
        correct += hit
    return {"status": "measured", "protocol": "local_normalized_exact_match_not_official_grader", "official_benchmark_score": False, "reader": reader_manifest, "reader_fingerprint": fingerprint(reader_manifest), "prediction_fingerprint": fingerprint(predictions), "questions": len(questions), "missing_predictions": len(expected_ids - set(mapping)), "normalized_exact_match": correct / len(questions) if questions else None, "explicit_abstention_accuracy": abstention_correct / absent if absent else None, "semantic_qa_accuracy": {"status": "not_measured", "reason": "Run the fixed external/official semantic grader separately."}}


def adapt_longmemeval(records: Sequence[Dict[str, Any]], revision: str) -> Dict[str, Any]:
    """Convert official sessions to source records while separating every label.

    `has_answer`, questions, answers and answer_session_ids are never copied into
    source content or metadata. No dataset or model is downloaded by this adapter.
    """
    if not revision.strip():
        raise PLMError("external dataset revision is required")
    sources = []
    questions = []
    for record in records:
        qid = str(record["question_id"])
        sessions = record["haystack_sessions"]
        session_ids = record["haystack_session_ids"]
        dates = record["haystack_dates"]
        if len(sessions) != len(session_ids) or len(sessions) != len(dates):
            raise PLMError("misaligned official session fields")
        project = "longmemeval:" + qid
        # The pinned upstream S split contains a small number of repeated
        # session IDs within one haystack. Session position is authoritative
        # for preserving the full history. A label naming a repeated ID
        # conservatively maps to all its otherwise-indistinguishable copies.
        session_mapping = {}
        answer_turns = []
        for session_index, (session_id, date, turns) in enumerate(zip(session_ids, dates, sessions)):
            source_id = "lme-" + hashlib.sha256((qid + "\0" + str(session_index) + "\0" + str(session_id)).encode()).hexdigest()[:24]
            session_mapping.setdefault(str(session_id), []).append(source_id)
            clean_turns = []
            for index, turn in enumerate(turns):
                clean_turns.append({"role": str(turn["role"]), "content": str(turn["content"])})
                if turn.get("has_answer") is True:
                    answer_turns.append({"source_id": source_id, "turn_index": index})
            # Dates can use the official human-readable format; preserve as
            # source content rather than silently interpreting calendar strings.
            sources.append({"source_id": source_id, "project": project, "split": "official", "title": "Conversation session " + str(session_id), "body": "Session date: " + str(date) + "\n" + json.dumps(clean_turns, ensure_ascii=False), "scope": "project", "scope_id": ""})
        unanswerable = qid.endswith("_abs")
        required_ids = [] if unanswerable else [source_id for value in record["answer_session_ids"] for source_id in session_mapping[str(value)]]
        questions.append({"question_id": qid, "project": project, "split": "official", "question": record["question"], "answer": record["answer"], "question_date": record.get("question_date", ""), "category": "abstention" if unanswerable else record["question_type"], "required_source_ids": required_ids, "gold_answer_turns": answer_turns, "unanswerable": unanswerable, "scope": "project", "scope_id": ""})
    return {"sources": sources, "questions": questions, "metadata": {"dataset_id": "longmemeval-adapter-" + revision, "origin": "official_external_data_local_retrieval_diagnostic", "upstream": "https://github.com/xiaowu0162/LongMemEval", "revision": revision, "input_fingerprint": fingerprint(records), "splits": ["official"], "official_benchmark_score": False}}


def write_dataset(folder: Path, sources: Sequence[Dict[str, Any]], questions: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {"sources.json": list(sources)}
    for split in metadata["splits"]:
        outputs["questions." + split + ".json"] = [question for question in questions if question["split"] == split]
    if any((folder / name).exists() for name in [*outputs, "manifest.json"]):
        raise PLMError("evaluation output dataset already exists; choose a new destination")
    for name, content in outputs.items():
        (folder / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = dict(metadata)
    manifest["sha256"] = {name: _file_hash(folder / name) for name in outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_factory(spec: str) -> Any:
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise PLMError("factory must be an explicit module:callable")
    return getattr(importlib.import_module(module), name)()
