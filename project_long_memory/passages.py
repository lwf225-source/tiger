"""Deterministic, rebuildable passages with exact parent-body provenance.

Offsets count Python Unicode code points, not UTF-8 bytes or tokenizer tokens.
Text is always an unmodified ``body[start:end]`` slice. Speaker/date labels are
read only from recognized explicit headers or role/content conversation JSON;
dates remain literal source labels, never inferred or normalized calendar facts.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


ALGORITHM = "plm-passages-v1"
UNKNOWN = "unknown"
_ROLES = {"user": "user", "assistant": "assistant", "system": "system", "developer": "developer", "tool": "tool",
          "用户": "user", "助手": "assistant", "系统": "system", "开发者": "developer", "工具": "tool"}
_ROLE_NAMES = "|".join(_ROLES)
_ROLE_LINE = re.compile(r"^\s*(?P<role>" + _ROLE_NAMES + r")\s*[:：]", re.I)
_ROLE_HEADING = re.compile(r"^\s*#{1,6}\s+(?P<role>" + _ROLE_NAMES + r")(?:\s+\[(?P<date>[^\]\r\n]+)\])?\s*$", re.I)
_DATE = re.compile(r"^\s*(?:#{1,6}\s+)?(?:Session date|Date|会话日期|日期)\s*[:：]\s*(?P<date>\S.*?)\s*$", re.I)
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_FIELDS = {"algorithm", "passage_id", "parent_id", "body_sha256", "start", "end", "line_start", "line_end", "text", "speaker", "source_date", "unit_index"}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _parent(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("parent_id must be a nonempty string")
    return value


def _identity(parent_id: str, digest: str, start: int, end: int) -> str:
    bound = json.dumps([ALGORITHM, parent_id, digest, start, end], ensure_ascii=False, separators=(",", ":"))
    return "psg-" + hashlib.sha256(bound.encode("utf-8")).hexdigest()[:32]


def _json_units(body: str) -> Optional[List[Dict[str, Any]]]:
    """Read the existing Session-date + JSON role/content adapter without re-encoding it."""
    cursor = 0
    source_date = UNKNOWN
    for line in body.splitlines(keepends=True):
        if not line.strip():
            cursor += len(line)
            continue
        date = _DATE.fullmatch(line.rstrip("\r\n"))
        if date:
            source_date = date.group("date")
            cursor += len(line)
        break
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if cursor >= len(body) or body[cursor] != "[":
        return None
    decoder = json.JSONDecoder()
    starts: List[int] = []
    roles: List[str] = []
    cursor += 1
    try:
        while True:
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if cursor >= len(body):
                return None
            if body[cursor] == "]":
                cursor += 1
                break
            start = cursor
            turn, cursor = decoder.raw_decode(body, cursor)
            if not isinstance(turn, dict) or not isinstance(turn.get("role"), str) or not isinstance(turn.get("content"), str):
                return None
            starts.append(start)
            roles.append(_ROLES.get(turn["role"].lower(), UNKNOWN))
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if cursor < len(body) and body[cursor] == ",":
                cursor += 1
                # JSON does not allow a trailing comma; fail closed to ordinary
                # unlabelled text if this is not a well-formed conversation.
                check = cursor
                while check < len(body) and body[check].isspace():
                    check += 1
                if check >= len(body) or body[check] == "]":
                    return None
            elif cursor < len(body) and body[cursor] == "]":
                cursor += 1
                break
            else:
                return None
        if body[cursor:].strip() or not starts:
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    # Prefix/date/array delimiters remain real source characters. Separators are
    # retained with the preceding turn, so the complete parent stays covered.
    starts[0] = 0
    return [{"start": start, "end": starts[index + 1] if index + 1 < len(starts) else len(body),
             "speaker": roles[index], "source_date": source_date, "unit_index": index}
            for index, start in enumerate(starts)]


def _text_units(body: str) -> List[Dict[str, Any]]:
    units = []
    start = 0
    offset = 0
    speaker = UNKNOWN
    source_date = UNKNOWN
    prefix_only = True
    fence: Optional[Tuple[str, int]] = None

    def emit(end: int) -> None:
        if end > start:
            units.append({"start": start, "end": end, "speaker": speaker,
                          "source_date": source_date, "unit_index": len(units)})

    for line in body.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        fence_match = _FENCE.match(text)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1] and not text[fence_match.end():].strip():
                fence = None
            prefix_only = False
            offset += len(line)
            continue
        if fence is not None:
            offset += len(line)
            continue
        date = _DATE.fullmatch(text)
        role = _ROLE_HEADING.fullmatch(text) or _ROLE_LINE.match(text)
        if date:
            emit(offset)
            start = offset
            speaker = UNKNOWN
            source_date = date.group("date")
            prefix_only = True
        elif role:
            if not prefix_only or speaker != UNKNOWN:
                emit(offset)
                start = offset
            speaker = _ROLES[role.group("role").lower()]
            if role.re is _ROLE_HEADING and role.group("date"):
                source_date = role.group("date")
            prefix_only = False
        elif text.strip():
            prefix_only = False
        offset += len(line)
    emit(len(body))
    return units


class _Source:
    def __init__(self, body: str):
        if not isinstance(body, str):
            raise ValueError("body must be a string")
        try:
            self.digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            raise ValueError("body must contain valid Unicode code points") from None
        self.body = body
        self.units = _json_units(body) or _text_units(body)
        self.unit_starts = [unit["start"] for unit in self.units]
        self.line_starts = [0] + [index + 1 for index, character in enumerate(body) if character == "\n"]

    def passage(self, parent_id: str, start: int, end: int) -> Dict[str, Any]:
        _parent(parent_id)
        _integer(start, "start")
        _integer(end, "end", 1)
        if not 0 <= start < end <= len(self.body):
            raise ValueError("passage offsets are outside the parent body")
        first = max(0, bisect.bisect_right(self.unit_starts, start) - 1)
        last = max(0, bisect.bisect_right(self.unit_starts, end - 1) - 1)
        touched = self.units[first:last + 1]
        speakers = {unit["speaker"] for unit in touched}
        dates = {unit["source_date"] for unit in touched}
        return {
            "algorithm": ALGORITHM, "passage_id": _identity(parent_id, self.digest, start, end),
            "parent_id": parent_id, "body_sha256": self.digest, "start": start, "end": end,
            "line_start": bisect.bisect_right(self.line_starts, start),
            "line_end": bisect.bisect_right(self.line_starts, end - 1),
            "text": self.body[start:end], "speaker": next(iter(speakers)) if len(speakers) == 1 else UNKNOWN,
            "source_date": next(iter(dates)) if len(dates) == 1 else UNKNOWN,
            "unit_index": touched[0]["unit_index"] if len(touched) == 1 else -1,
        }

    def validate(self, passage: Dict[str, Any], parent_id: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(passage, dict) or _FIELDS - set(passage):
            raise ValueError("incomplete passage provenance")
        expected_parent = _parent(parent_id) if parent_id is not None else _parent(passage["parent_id"])
        if passage["parent_id"] != expected_parent or passage["body_sha256"] != self.digest or passage["algorithm"] != ALGORITHM:
            raise ValueError("passage parent, body hash or algorithm mismatch")
        expected = self.passage(expected_parent, passage["start"], passage["end"])
        for field in _FIELDS:
            if type(passage[field]) is not type(expected[field]) or passage[field] != expected[field]:
                raise ValueError("passage provenance mismatch: " + field)
        return expected


def _window_end(body: str, start: int, stop: int, max_chars: int, overlap_chars: int) -> int:
    target = min(stop, start + max_chars)
    if target == stop:
        return target
    minimum = start + max(overlap_chars + 1, max_chars // 2)
    # Prefer blank-line paragraph boundaries, then existing line boundaries.
    # Full stops are intentionally not boundaries: they can separate a statement
    # from its speaker, negation, condition or decimal value.
    paragraph = [match.end() for match in re.finditer(r"\r?\n[\t ]*\r?\n", body[start:target]) if start + match.end() >= minimum]
    if paragraph:
        return start + paragraph[-1]
    newline = body.rfind("\n", minimum, target)
    if newline >= minimum:
        return newline + 1
    if target > start + 1 and body[target - 1:target + 1] == "\r\n":
        return target - 1
    return target


def split_passages(body: str, parent_id: str, max_chars: int = 900, overlap_chars: int = 120) -> List[Dict[str, Any]]:
    """Split all source characters into bounded windows, preserving turn boundaries.

    Overlap applies within a long turn/document unit, not across different
    speakers. Adjacent turns remain available through ``neighbor_passages``.
    Empty text produces no passage; whitespace-only text remains exact source.
    """
    parent_id = _parent(parent_id)
    max_chars = _integer(max_chars, "max_chars", 1)
    overlap_chars = _integer(overlap_chars, "overlap_chars")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    source = _Source(body)
    passages = []
    for unit in source.units:
        start = unit["start"]
        while start < unit["end"]:
            end = _window_end(body, start, unit["end"], max_chars, overlap_chars)
            passages.append(source.passage(parent_id, start, end))
            if end == unit["end"]:
                break
            start = max(start + 1, end - overlap_chars)
    return passages


def validate_passage(body: str, passage: Dict[str, Any], parent_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate the full provenance contract and return a fresh canonical record."""
    return _Source(body).validate(passage, parent_id)


def locate_passage(body: str, parent_id: str, start: int, end: int) -> Dict[str, Any]:
    """Return a canonical exact span for verified parent-event coordinates.

    This checks Unicode and offset bounds and computes all provenance fields.
    Callers must first enforce project/scope eligibility, resolve the actual
    parent Event and compare any externally supplied expected body hash.
    """
    return _Source(body).passage(parent_id, start, end)


def expand_passage(
    body: str, passage: Dict[str, Any], before_chars: int = 120,
    after_chars: int = 120, max_chars: int = 1800,
) -> Dict[str, Any]:
    """Expand a verified span with bounded real neighbors, never synthesized text.

    A cross-turn result has unit_index=-1; mixed speakers/dates are explicitly
    unknown. The original matched span is never removed to meet the allowance.
    """
    before_chars = _integer(before_chars, "before_chars")
    after_chars = _integer(after_chars, "after_chars")
    max_chars = _integer(max_chars, "max_chars", 1)
    source = _Source(body)
    original = source.validate(passage)
    remaining = max_chars - (original["end"] - original["start"])
    if remaining < 0:
        raise ValueError("max_chars cannot be smaller than the original passage")
    wanted_left = min(before_chars, original["start"])
    wanted_right = min(after_chars, len(body) - original["end"])
    left = min(wanted_left, (remaining + 1) // 2)
    right = min(wanted_right, remaining - left)
    left += min(wanted_left - left, remaining - left - right)
    return source.passage(original["parent_id"], original["start"] - left, original["end"] + right)


def neighbor_passages(
    body: str, passages: Sequence[Dict[str, Any]], passage_id: str,
    radius: int = 1, max_chars: int = 1800,
) -> List[Dict[str, Any]]:
    """Return the matched passage and bounded adjacent passages in source order.

    Budget counts the sum of returned text lengths, including overlap. An
    unknown target, stale body, mismatched parent or forged span is rejected.
    """
    radius = _integer(radius, "radius")
    max_chars = _integer(max_chars, "max_chars", 1)
    source = _Source(body)
    unique: Dict[str, Dict[str, Any]] = {}
    parent_id = None
    for item in passages:
        verified = source.validate(item, parent_id)
        parent_id = verified["parent_id"]
        unique[verified["passage_id"]] = verified
    ordered = sorted(unique.values(), key=lambda item: (item["start"], item["end"], item["passage_id"]))
    target = next((index for index, item in enumerate(ordered) if item["passage_id"] == passage_id), None)
    if target is None:
        raise ValueError("unknown passage_id")
    matched = ordered[target]
    remaining = max_chars - len(matched["text"])
    if remaining < 0:
        raise ValueError("max_chars cannot be smaller than the matched passage")
    selected = [matched]
    for distance in range(1, min(radius, len(ordered) - 1) + 1):
        for index in (target - distance, target + distance):
            if 0 <= index < len(ordered):
                candidate = ordered[index]
                if len(candidate["text"]) <= remaining:
                    selected.append(candidate)
                    remaining -= len(candidate["text"])
    return sorted(selected, key=lambda item: (item["start"], item["end"], item["passage_id"]))
