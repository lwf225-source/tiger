"""Deterministic write-side narrative-time extraction (zero dependencies).

Batch 11. Batch 10 (docs/BATCH10_REPORT.md) measured that query-side
``valid_at`` filtering cannot rescue LoCoMo temporal questions: their anchors
point at *narrative* time (the moment an event mentioned in the message body
happened, e.g. "my visit in 2021", "two weeks before August 11"), while
``valid_at`` bounds *record* time (the session timestamp) — and LoCoMo
conversations routinely recount past events in later sessions.

This module is the ingest-side half of the fix: it scans the message body at
write time, extracts every safely parseable calendar date expression, and
normalises each into a small set of standard surface forms (ISO date,
"Month D, YYYY", "Month YYYY", season, ...) that the AML adapter attaches as
event *tags* (metadata — the immutable body is never touched). Because tags
already flow into every index text channel (FTS ``tags`` column, scoped
substring field, stored n-gram vector, embedding passage text), a query
carrying a date anchor lexically/semantically hits the narrative-time mention
with no search-layer change at all.

Design discipline
-----------------
- **Extraction, not interpretation**: only explicit calendar expressions are
  taken. Relative phrases ("two weeks before ...") contribute their absolute
  anchor ("August 11") but are never resolved against the record timestamp —
  write-time resolution would bake the record date into the evidence and
  silently discard the relative semantics.
- **Recall aid, not a filter**: a false positive only adds a noise tag to one
  event; it can never exclude evidence. Still, invalid dates (2026-13-45,
  February 30) and anchor-less bare month names ("the March report") are
  rejected to keep the tag list lean.
- **Deterministic and capped**: regex + rules only, identical output for
  identical input, at most ``MAX_MENTIONS`` mentions and ``MAX_TAGS`` tag
  strings per message; extraction failure degrades to "no tags", never raises.

Output schema of :func:`extract_mentioned_dates`::

    [{"kind": "date" | "month_year" | "month_day" | "year" | "season_year",
      "iso": "2023-08-11" | None,      # full dates only
      "year": 2023 | None,
      "month": 8 | None,
      "day": 11 | None,
      "season": "summer" | None,
      "matched": "August 11, 2023"}]   # surface text (audit)

:func:`mentioned_date_tags` reduces the mentions to the deduplicated tag
strings that are attached to the event.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .temporal_intent import _MONTHS, _days_in_month

_MONTH_NAME = (
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_MONTH_FULL = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_SEASON_OF_MONTH = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall",
    11: "fall", 12: "winter",
}
_SEASON_RE = r"(spring|summer|fall|autumn|winter)"

MAX_MENTIONS = 10
MAX_TAGS = 24
_YEAR_MIN, _YEAR_MAX = 1900, 2099

# Order matters: richer forms first; spans already claimed by an earlier rule
# are masked before the bare-year fallback runs.
_ISO_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
# "August 11, 2023" / "August 11 2023" (year optional -> month_day)
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b" + _MONTH_NAME + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*((?:19|20)\d{2}))?\b", re.I)
# "11 August 2023" / "the 11th of August, 2023" (year optional)
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+" + _MONTH_NAME + r"(?:\s*,?\s*((?:19|20)\d{2}))?\b", re.I)
# "March 2026" (month + year, no day)
_MONTH_YEAR_RE = re.compile(r"\b" + _MONTH_NAME + r"\s*,?\s*((?:19|20)\d{2})\b", re.I)
# "summer 2021", "the summer of 2021"
_SEASON_YEAR_RE = re.compile(
    r"\b(?:the\s+)?" + _SEASON_RE + r"(?:\s+of)?\s+((?:19|20)\d{2})\b", re.I)
# Chinese: 2023年8月11日 / 2023年8月
_CN_RE = re.compile(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月(?:(\d{1,2})\s*日)?")
# Bare year — last resort, only where no richer rule already matched.
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _valid_date(year: int, month: int, day: int) -> bool:
    return _YEAR_MIN <= year <= _YEAR_MAX and 1 <= month <= 12 and 1 <= day <= _days_in_month(year, month)


def _mention(kind: str, matched: str, year: Optional[int] = None, month: Optional[int] = None,
             day: Optional[int] = None) -> Dict[str, Any]:
    season = _SEASON_OF_MONTH.get(month) if month else None
    iso = "%04d-%02d-%02d" % (year, month, day) if year and month and day else None
    return {"kind": kind, "iso": iso, "year": year, "month": month, "day": day,
            "season": season, "matched": matched}


def extract_mentioned_dates(text: str) -> List[Dict[str, Any]]:
    """Extract every safely parseable date mention; never raises."""
    if not isinstance(text, str) or not text.strip():
        return []
    probe = text[:16000]
    try:
        hits: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
        # Richer rules claim their span first — whether or not the candidate
        # turns out to be a valid date — so coarser fallbacks (month-year,
        # bare year) never re-report a fragment of an already-considered
        # expression ("11 August 2023" must not also yield "2023-08", and the
        # "2026" inside the invalid "2026-13-45" stays untagged).
        claimed = [False] * (len(probe) + 1)

        def iso(match):
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if not _valid_date(year, month, day):
                return None
            return _mention("date", match.group(0), year, month, day)

        def month_day_year(match):
            month = _MONTHS[match.group(1).lower()]
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else None
            if year is not None:
                if not _valid_date(year, month, day):
                    return None
                return _mention("date", match.group(0), year, month, day)
            if not 1 <= day <= _days_in_month(2001, month):  # leap-safe validity check
                return None
            return _mention("month_day", match.group(0), None, month, day)

        def day_month_year(match):
            day = int(match.group(1))
            month = _MONTHS[match.group(2).lower()]
            year = int(match.group(3)) if match.group(3) else None
            if year is not None:
                if not _valid_date(year, month, day):
                    return None
                return _mention("date", match.group(0), year, month, day)
            if not 1 <= day <= _days_in_month(2001, month):
                return None
            return _mention("month_day", match.group(0), None, month, day)

        def month_year(match):
            month = _MONTHS[match.group(1).lower()]
            year = int(match.group(2))
            if not _YEAR_MIN <= year <= _YEAR_MAX:
                return None
            return _mention("month_year", match.group(0), year, month)

        def season_year(match):
            season = match.group(1).lower()
            year = int(match.group(2))
            if not _YEAR_MIN <= year <= _YEAR_MAX:
                return None
            item = _mention("season_year", match.group(0), year)
            item["season"] = "fall" if season == "autumn" else season
            return item

        def cn(match):
            year, month = int(match.group(1)), int(match.group(2))
            day = int(match.group(3)) if match.group(3) else None
            if day is not None:
                if not _valid_date(year, month, day):
                    return None
                return _mention("date", match.group(0), year, month, day)
            if not (_YEAR_MIN <= year <= _YEAR_MAX and 1 <= month <= 12):
                return None
            return _mention("month_year", match.group(0), year, month)

        for pattern, handler in (
            (_ISO_RE, iso),
            (_MONTH_DAY_YEAR_RE, month_day_year),
            (_DAY_MONTH_YEAR_RE, day_month_year),
            (_SEASON_YEAR_RE, season_year),
            (_CN_RE, cn),
            (_MONTH_YEAR_RE, month_year),
        ):
            for match in pattern.finditer(probe):
                start, end = match.span()
                if any(claimed[start:end]):
                    continue
                for index in range(start, end):
                    claimed[index] = True
                item = handler(match)
                if item is not None:
                    hits.append(((start, end), item))

        # Bare years only where no richer rule already claimed the span.
        for match in _YEAR_RE.finditer(probe):
            if any(claimed[match.start():match.end()]):
                continue
            year = int(match.group(1))
            if _YEAR_MIN <= year <= _YEAR_MAX:
                hits.append((match.span(), _mention("year", match.group(0), year)))

        hits.sort(key=lambda entry: entry[0][0])
        mentions: List[Dict[str, Any]] = []
        seen = set()
        for _span, item in hits:
            key = (item["kind"], item["iso"], item["year"], item["month"], item["day"], item["season"])
            if key in seen:
                continue
            seen.add(key)
            mentions.append(item)
            if len(mentions) >= MAX_MENTIONS:
                break
        return mentions
    except Exception:
        # Extraction failure must never break a write: degrade to no mentions.
        return []


def mentioned_date_tags(text: str) -> List[str]:
    """Normalised surface forms of the mentioned dates, as event tag strings.

    The forms mirror the phrasings date-anchored queries actually use
    (measured on the LoCoMo temporal subset): bare years, month names,
    ISO dates, and season names — each mention contributes its own forms,
    deduplicated and capped at ``MAX_TAGS``.
    """
    tags: List[str] = []
    seen = set()

    def push(value: str) -> None:
        key = value.lower()
        if key not in seen and len(tags) < MAX_TAGS:
            seen.add(key)
            tags.append(value)

    for item in extract_mentioned_dates(text):
        year, month, day = item["year"], item["month"], item["day"]
        if item["iso"]:
            month_name = _MONTH_FULL[month - 1]
            push(item["iso"])
            push("%s %d, %d" % (month_name, day, year))
            push("%s %d" % (month_name, year))
            if item["season"]:
                push("%s %d" % (item["season"], year))
                if item["season"] == "fall":
                    push("autumn %d" % year)
        elif item["kind"] == "month_year":
            push("%s %d" % (_MONTH_FULL[month - 1], year))
            push("%04d-%02d" % (year, month))
            if item["season"]:
                push("%s %d" % (item["season"], year))
                if item["season"] == "fall":
                    push("autumn %d" % year)
        elif item["kind"] == "month_day":
            push("%s %d" % (_MONTH_FULL[month - 1], day))
        elif item["kind"] == "season_year":
            push("%s %d" % (item["season"], year))
            if item["season"] == "fall":
                push("autumn %d" % year)
        elif item["kind"] == "year":
            push(str(year))
        if item["kind"] != "year" and year is not None:
            push(str(year))
    return tags
