"""Deterministic query-side temporal-intent extraction (zero dependencies).

Given a natural-language query, decide whether it carries an explicit temporal
intent and, when safely parseable, reduce it to an absolute ``as_of`` ISO
timestamp that the search layer can feed into the opt-in ``valid_at`` filter
(batch 10; the wiring hook was announced in docs/BATCH8_REPORT.md §5).

Design discipline
-----------------
- **Conservative by construction**: when in doubt the extractor reports *no*
  intent. A false positive would point the ``valid_at`` filter at the wrong
  moment and silently exclude valid evidence; a false negative merely keeps
  the status quo. Every ``as_of`` requires an explicit absolute anchor (a
  calendar date expression with month/year or an ISO date).
- **English-first** (the AML/LoCoMo text track is English), with common
  Chinese expressions covered.
- **Relative expressions are reported, never applied by default**: "last
  week" is anchored to the query instant, but corpora whose timestamps are
  historical (LoCoMo conversations are from 2021-2023) would be emptied by a
  naive ``valid_at = now - 7d``. The extractor still returns the anchored
  value for auditability; the search wiring deliberately applies only
  absolute ``as_of`` values (see ``search(..., temporal_intent=True)``).
- ``after``/``since``/``之后`` express a *lower* bound, which the bitemporal
  ``valid_at`` mechanism (an upper bound) cannot express: they set
  ``has_temporal_intent`` but never produce an ``as_of``.

Output schema::

    {
        "has_temporal_intent": bool,
        "as_of": str | None,        # absolute ISO 8601 UTC, wiring-applicable
        "comparison": bool,         # "previously/currently/used to/以前/曾经..."
        "relative": {"expression": str, "as_of": str | None} | None,
        "matched": str,             # surface text that produced as_of (audit)
    }

Period references ("during April 2022", "in summer 2021", "March 2026")
resolve ``as_of`` to the *end* of the referenced period (the ``valid_at``
filter keeps everything at or before the point, so the end of the period is
the inclusive choice); ``before X`` resolves to the *start* of X.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_YEAR_RE = r"((?:19|20)\d{2})"
_SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (1, 2)}
_SEASON_RE = r"(spring|summer|fall|autumn|winter)"

# Prepositions that anchor a *point/period* the valid_at upper bound can serve.
_POINT_PREP = r"(?:as of|in|during|around|at the end of|by the end of)"
# Prepositions that mean "strictly before this anchor" -> start of the anchor.
_BEFORE_PREP = r"(?:before|until|prior to|by)"
# Lower-bound-only prepositions: intent yes, as_of no.
_AFTER_PREP = r"(?:after|since)"

_PATTERNS = [
    # "first/second half of September 2022"
    ("half_month", re.compile(
        r"\b(first|second|1st|2nd)\s+half\s+of\s+" + _MONTH_RE + r"\s+" + _YEAR_RE + r"\b", re.I)),
    # "between August and November 2023"
    ("between_months", re.compile(
        r"\bbetween\s+" + _MONTH_RE + r"\s+and\s+" + _MONTH_RE + r"\s+" + _YEAR_RE + r"\b", re.I)),
    # ISO date/month with an explicit preposition: "as of 2026-03", "in 2026-03-15"
    ("iso_point", re.compile(
        r"\b" + _POINT_PREP + r"\s+(\d{4})-(\d{2})(?:-(\d{2}))?\b", re.I)),
    ("iso_before", re.compile(
        r"\b" + _BEFORE_PREP + r"\s+(\d{4})-(\d{2})(?:-(\d{2}))?\b", re.I)),
    ("iso_after", re.compile(
        r"\b" + _AFTER_PREP + r"\s+(\d{4})-(\d{2})(?:-(\d{2}))?\b", re.I)),
    # Month name + year with preposition: "as of March 2026", "during April 2022"
    ("month_point", re.compile(
        r"\b" + _POINT_PREP + r"\s+" + _MONTH_RE + r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?,?\s+" + _YEAR_RE + r"\b", re.I)),
    ("month_before", re.compile(
        r"\b" + _BEFORE_PREP + r"\s+" + _MONTH_RE + r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?,?\s+" + _YEAR_RE + r"\b", re.I)),
    ("month_after", re.compile(
        r"\b" + _AFTER_PREP + r"\s+" + _MONTH_RE + r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?,?\s+" + _YEAR_RE + r"\b", re.I)),
    # Season + year: "in summer 2021", "during the summer of 2021"
    ("season_point", re.compile(
        r"\b" + _POINT_PREP + r"(?:\s+the)?\s+" + _SEASON_RE + r"(?:\s+of)?\s+" + _YEAR_RE + r"\b", re.I)),
    # Bare year with preposition: "in 2023", "before 2020"
    ("year_point", re.compile(r"\b" + _POINT_PREP + r"\s+" + _YEAR_RE + r"\b", re.I)),
    ("year_before", re.compile(r"\b" + _BEFORE_PREP + r"\s+" + _YEAR_RE + r"\b", re.I)),
    ("year_after", re.compile(r"\b" + _AFTER_PREP + r"\s+" + _YEAR_RE + r"\b", re.I)),
    # Chinese: 截至/截止到 2026年3月(15日) / 在2026年3月 / 2026年3月之前 / 之后 / 以来
    ("cn_point", re.compile(r"(?:截至(?:到)?|截止到|在|于)\s*(\d{4})\s*年\s*(\d{1,2})\s*月(?:(\d{1,2})\s*日)?")),
    ("cn_before", re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:(\d{1,2})\s*日)?\s*(?:之前|以前|前)")),
    ("cn_after", re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:(\d{1,2})\s*日)?\s*(?:之后|以后|以来)")),
]

_COMPARISON_RE = re.compile(
    r"\b(previously|used to|originally|currently|nowadays|anymore|any longer|no longer)\b"
    r"|以前|曾经|过去|目前|如今|眼下",
    re.I,
)

_RELATIVE_PARTS = [
    (r"\b(yesterday)\b", -1, "day"), (r"\b(today)\b", 0, "day"),
    (r"\b(tomorrow)\b", 1, "day"),
    (r"\b(last|this|next)\s+(week|month|year)\b", None, None),
]
_RELATIVE_RES = [(re.compile(p, re.I), off, unit) for p, off, unit in _RELATIVE_PARTS]
_RELATIVE_CN = ("昨天", "今天", "明天", "上周", "本周", "上周", "上个月", "本月", "这个月", "今年", "去年", "明年")
_RELATIVE_CN_RE = re.compile("|".join(_RELATIVE_CN))

_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_in_month(year: int, month: int) -> int:
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return 29
    return _DAYS_IN_MONTH[month - 1]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _end_of_day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 23, 59, 59, 999999, tzinfo=timezone.utc)


def _end_of_month(year: int, month: int) -> datetime:
    return _end_of_day(year, month, _days_in_month(year, month))


def _start_of_month(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _end_of_year(year: int) -> datetime:
    return _end_of_day(year, 12, 31)


def _end_of_season(year: int, season: str) -> datetime:
    _start_month, end_month = _SEASONS[season.lower()]
    # Convention: "winter YYYY" closes at the end of February of YYYY.
    return _end_of_month(year, end_month)


def _anchor_relative(expression: str, now: Optional[datetime]) -> Optional[datetime]:
    """Anchor a relative expression to the query instant (audit-only value)."""
    if now is None:
        now = datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    text = expression.lower()
    if text in ("today", "今天"):
        return _end_of_day(now.year, now.month, now.day)
    if text in ("yesterday", "昨天"):
        day = now - timedelta(days=1)
        return _end_of_day(day.year, day.month, day.day)
    if text in ("tomorrow", "明天"):
        day = now + timedelta(days=1)
        return _end_of_day(day.year, day.month, day.day)
    match = re.match(r"(last|this|next)\s+(week|month|year)", text)
    if match:
        direction, unit = match.group(1), match.group(2)
        delta = {"last": -1, "this": 0, "next": 1}[direction]
        if unit == "week":
            end = now + timedelta(days=7 * delta + (6 - now.weekday()))
            return _end_of_day(end.year, end.month, end.day)
        if unit == "month":
            month = now.month + delta
            year = now.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            return _end_of_month(year, month)
        return _end_of_year(now.year + delta)
    cn = {"上周": -7, "本周": 0}
    if expression in cn:
        end = now + timedelta(days=cn[expression] + (6 - now.weekday()))
        return _end_of_day(end.year, end.month, end.day)
    if expression in ("上个月",):
        month = now.month - 1 or 12
        year = now.year - (1 if now.month == 1 else 0)
        return _end_of_month(year, month)
    if expression in ("本月", "这个月"):
        return _end_of_month(now.year, now.month)
    if expression == "今年":
        return _end_of_year(now.year)
    if expression == "去年":
        return _end_of_year(now.year - 1)
    if expression == "明年":
        return _end_of_year(now.year + 1)
    return None


def _match_as_of(text: str) -> "tuple[Optional[str], str]":
    """Return (absolute as_of ISO or None, matched surface text)."""
    for name, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if name == "half_month":
                half, month_s, year_s = groups
                year, month = int(year_s), _MONTHS[month_s.lower()]
                if half.lower() in ("first", "1st"):
                    return _iso(_end_of_day(year, month, 15)), match.group(0)
                return _iso(_end_of_month(year, month)), match.group(0)
            if name == "between_months":
                _m1, m2, year_s = groups
                return _iso(_end_of_month(int(year_s), _MONTHS[m2.lower()])), match.group(0)
            if name.startswith("iso_"):
                year, month, day = int(groups[0]), int(groups[1]), groups[2]
                if not 1 <= month <= 12 or (day is not None and not 1 <= int(day) <= _days_in_month(year, month)):
                    continue
                if name == "iso_after":
                    return None, match.group(0)
                if name == "iso_before":
                    point = _end_of_day(year, month, int(day)) if day else _start_of_month(year, month)
                    if day:
                        point = _start_of_month(year, month) + timedelta(days=int(day) - 1)
                    return _iso(point), match.group(0)
                point = _end_of_day(year, month, int(day)) if day else _end_of_month(year, month)
                return _iso(point), match.group(0)
            if name.startswith("month_"):
                month_s, day_s, year_s = groups
                year, month = int(year_s), _MONTHS[month_s.lower()]
                day = int(day_s) if day_s else None
                if day is not None and not 1 <= day <= _days_in_month(year, month):
                    continue
                if name == "month_after":
                    return None, match.group(0)
                if name == "month_before":
                    point = _start_of_month(year, month) if day is None \
                        else datetime(year, month, day, tzinfo=timezone.utc)
                    return _iso(point), match.group(0)
                point = _end_of_day(year, month, day) if day else _end_of_month(year, month)
                return _iso(point), match.group(0)
            if name == "season_point":
                season, year_s = groups
                return _iso(_end_of_season(int(year_s), season)), match.group(0)
            if name.startswith("year_"):
                year = int(groups[0])
                if name == "year_after":
                    return None, match.group(0)
                point = datetime(year, 1, 1, tzinfo=timezone.utc) if name == "year_before" else _end_of_year(year)
                return _iso(point), match.group(0)
            if name.startswith("cn_"):
                year, month, day_s = int(groups[0]), int(groups[1]), groups[2]
                if not 1 <= month <= 12 or (day_s is not None and not 1 <= int(day_s) <= _days_in_month(year, month)):
                    continue
                if name == "cn_after":
                    return None, match.group(0)
                if name == "cn_before":
                    point = (datetime(year, month, int(day_s), tzinfo=timezone.utc) if day_s
                             else _start_of_month(year, month))
                    return _iso(point), match.group(0)
                point = _end_of_day(year, month, int(day_s)) if day_s else _end_of_month(year, month)
                return _iso(point), match.group(0)
        except (ValueError, KeyError, OverflowError):
            continue
    return None, ""


def extract_temporal_intent(text: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Extract a structured temporal intent from a query; never raises."""
    result: Dict[str, Any] = {
        "has_temporal_intent": False,
        "as_of": None,
        "comparison": False,
        "relative": None,
        "matched": "",
    }
    if not isinstance(text, str) or not text.strip():
        return result
    probe = text[:4000]
    try:
        as_of, matched = _match_as_of(probe)
        comparison = bool(_COMPARISON_RE.search(probe))
        relative = None
        for pattern, _off, _unit in _RELATIVE_RES:
            m = pattern.search(probe)
            if m:
                relative = {"expression": m.group(0), "as_of": _iso(_anchor_relative(m.group(0), now))
                            if _anchor_relative(m.group(0), now) else None}
                break
        if relative is None:
            m = _RELATIVE_CN_RE.search(probe)
            if m:
                anchored = _anchor_relative(m.group(0), now)
                relative = {"expression": m.group(0), "as_of": _iso(anchored) if anchored else None}
        result.update({
            "as_of": as_of,
            "matched": matched,
            "comparison": comparison,
            "relative": relative,
            "has_temporal_intent": bool(as_of or matched or comparison or relative),
        })
    except Exception:
        # Extraction failure must never break a search: fall back to no intent.
        return {
            "has_temporal_intent": False, "as_of": None, "comparison": False,
            "relative": None, "matched": "",
        }
    return result
