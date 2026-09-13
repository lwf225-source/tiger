"""Post-generation, literal evidence-span coverage; never a retrieval input.

This is a conservative text-preservation diagnostic, not semantic correctness
or proof that a legal citation supports a conclusion. Offsets are Unicode code
points, while quote hashes are SHA256 over UTF-8 bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .model import PLMError


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _range_quote(span: Any, body: str) -> Optional[str]:
    if not isinstance(span, dict):
        return None
    start, end, digest = span.get("start"), span.get("end"), span.get("quote_sha256")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(body)
        or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        return None
    quote = body[start:end]
    return quote if _sha(quote) == digest else None


def load_span_gold(dataset: Dict[str, Any], split: str, path: Optional[Path] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """Read and validate evaluator-only spans. Missing sidecar => ({}, None).

    `path` is the dataset directory, otherwise dataset['root']/['path'] is used.
    The caller must invoke this only after generation, never during retrieval.
    """
    root = path if path is not None else dataset.get("root", dataset.get("path"))
    if root is None:
        return {}, None
    root = Path(root)
    sidecar_path = root / ("evidence-spans." + split + ".json")
    if root.is_symlink() or sidecar_path.is_symlink():
        raise PLMError("symlinked span evaluation data rejected")
    if not sidecar_path.exists():
        return {}, None
    try:
        if not sidecar_path.is_file() or sidecar_path.stat().st_size > 8 * 1024 * 1024:
            raise PLMError("invalid span evaluation artifact")
        raw = sidecar_path.read_bytes()
        sidecar = json.loads(raw)
        manifest = dataset["manifest"]
        if (sidecar.get("schema_version") != 1 or sidecar.get("dataset_id") != manifest["dataset_id"]
            or sidecar.get("split") != split or sidecar.get("retrieval_use_forbidden") is not True
            or sidecar.get("sources_sha256") != manifest["sha256"]["sources.json"]
            or sidecar.get("questions_sha256") != manifest["sha256"]["questions." + split + ".json"]
            or not sidecar.get("offset_unit", "").startswith("Unicode code points")):
            raise PLMError("span gold does not match frozen dataset or offset convention")
        sources = {row["source_id"]: row for row in dataset["sources"]}
        questions = {row["question_id"]: row for row in dataset["questions"] if row["split"] == split}
        mapping: Dict[str, Any] = {}
        for row in sidecar["questions"]:
            qid = row["question_id"]
            question = questions.get(qid)
            if (not question or qid in mapping or type(row.get("unanswerable")) is not bool
                or row["unanswerable"] != bool(question.get("unanswerable", False))
                or not isinstance(row.get("spans"), list)):
                raise PLMError("invalid or duplicate span gold question")
            spans = row["spans"]
            if bool(spans) == row["unanswerable"]:
                raise PLMError("span gold answerability mismatch")
            seen, clean, hashes = set(), [], {}
            for span in spans:
                source = sources.get(span.get("source_id"))
                identity = span.get("evidence_id")
                if (not isinstance(identity, str) or not identity or identity in seen or not source
                    or source["project"] != question["project"] or source["split"] != split
                    or source.get("scope", "project") != question.get("scope", "project")
                    or source.get("scope_id", "") != question.get("scope_id", "")
                    or _range_quote(span, source["body"]) is None):
                    raise PLMError("invalid span gold identity, scope, range or hash")
                seen.add(identity)
                clean.append({key: span[key] for key in ("evidence_id", "source_id", "start", "end", "quote_sha256")})
                hashes[source["source_id"]] = _sha(source["body"])
            if {span["source_id"] for span in clean} != set(question["required_source_ids"]):
                raise PLMError("span gold required parents differ from question labels")
            absence_ids = row.get("absence_check_source_ids", [])
            for sid in absence_ids:
                source = sources.get(sid)
                if (not source or source["project"] != question["project"]
                    or source.get("scope", "project") != question.get("scope", "project")
                    or source.get("scope_id", "") != question.get("scope_id", "")):
                    raise PLMError("absence audit source crosses scope")
            mapping[qid] = {"unanswerable": row["unanswerable"], "spans": clean,
                "_project": question["project"], "_scope": question.get("scope", "project"),
                "_scope_id": question.get("scope_id", ""), "_source_hashes": hashes}
        if set(mapping) != set(questions):
            raise PLMError("span gold does not cover the complete split")
        return mapping, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise PLMError("invalid span evaluation data; no source text logged") from None


def _actual_records(case: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Only sources actually embedded in the reader's user payload count."""
    try:
        messages = case["messages"]
        if len(messages) != 2 or messages[1].get("role") != "user":
            return {}, 1
        payload = json.loads(messages[1]["content"])
        records = payload["sources"]
        if not isinstance(records, list):
            return {}, 1
        found, duplicated, invalid = {}, set(), 0
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("source_id"), str) or not isinstance(record.get("body"), str):
                invalid += 1
                continue
            sid = record["source_id"]
            if sid in found:
                duplicated.add(sid)
                invalid += 1
            found[sid] = record
        for sid in duplicated:
            found.pop(sid, None)
        return found, invalid
    except (KeyError, TypeError, ValueError, AttributeError):
        return {}, 1


def _covers(intervals, start, end):
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def score_span_coverage(question_id: str, case: Dict[str, Any], gold_mapping: Dict[str, Any], by_source: Dict[str, Any]) -> Dict[str, Any]:
    gold = gold_mapping.get(question_id)
    if not gold or gold["unanswerable"]:
        return {"status": "not_measured", "reason": "unanswerable_has_no_positive_span_gold" if gold else "no_span_gold",
            "required": None, "covered": None, "coverage": None, "full_coverage": None, "per_span": []}
    actual, invalid_records = _actual_records(case)
    intervals: Dict[str, Any] = {}
    invalid_spans = 0
    seen_metadata = set()
    metadata = case.get("sources", [])
    if not isinstance(metadata, list):
        metadata = []
        invalid_records += 1
    identity_counts = Counter(row["source_id"] for row in metadata if isinstance(row, dict) and isinstance(row.get("source_id"), str))
    duplicated_metadata = {sid for sid, count in identity_counts.items() if count > 1}
    for record in metadata:
        if not isinstance(record, dict):
            invalid_records += 1
            continue
        sid = record.get("source_id")
        parent_id = record.get("parent_source_id", sid)
        if not isinstance(sid, str) or not sid or not isinstance(parent_id, str) or not parent_id:
            invalid_records += 1
            continue
        source, presented = by_source.get(parent_id), actual.get(sid)
        if (not isinstance(source, dict) or not presented or sid in duplicated_metadata or sid in seen_metadata
            or not isinstance(source.get("body"), str) or record.get("body_sha256") != _sha(source["body"])
            or source.get("project") != gold["_project"] or source.get("scope", "project") != gold["_scope"]
            or source.get("scope_id", "") != gold["_scope_id"]
            or (parent_id in gold["_source_hashes"] and _sha(source["body"]) != gold["_source_hashes"][parent_id])
            or (record.get("presented_body_sha256") is not None and record["presented_body_sha256"] != _sha(presented["body"]))):
            invalid_records += 1
            continue
        seen_metadata.add(sid)
        spans = record.get("spans", [])
        if not isinstance(spans, list):
            invalid_records += 1
            continue
        for span in spans:
            quote = _range_quote(span, source["body"])
            # Duplicate parent locations are intentionally uncredited: the text
            # alone cannot bind a short repeated phrase to the claimed location.
            if quote is None or source["body"].count(quote) != 1 or quote not in presented["body"]:
                invalid_spans += 1
                continue
            intervals.setdefault(parent_id, []).append((span["start"], span["end"]))
    hits = [{"evidence_id": span["evidence_id"], "source_id": span["source_id"],
             "start": span["start"], "end": span["end"],
             "covered": _covers(intervals.get(span["source_id"], []), span["start"], span["end"])} for span in gold["spans"]]
    covered = sum(row["covered"] for row in hits)
    return {"status": "measured", "required": len(hits), "covered": covered,
        "coverage": covered / len(hits), "full_coverage": covered == len(hits), "per_span": hits,
        "invalid_advertised_spans": invalid_spans, "invalid_records": invalid_records,
        "measurement": "verified_literal_span_lower_bound_not_semantic_answer_or_citation_support"}
