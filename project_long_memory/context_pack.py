"""Budgeted, source-preserving context assembly without tokenizer dependencies.

``token_budget`` is enforced as a UTF-8 byte allowance. This deliberately
conservative estimate is not an exact model token count; framing added by a
caller/model must be budgeted separately by that caller.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .model import SearchResult
from .passages import expand_passage, validate_passage


_NEGATION = re.compile(r"不得|禁止|尚未|不能|不要|不允许|未|仅|只有|除非|必须|Conditions?\s*:|条件|前提|限制|\b(?:not|never|only|unless|must|without|required|condition)\b", re.I)
_STRONG_NEGATION = re.compile(r"不得|禁止|尚未|不能|不要|不允许|未|\b(?:not|never|unless)\b", re.I)

# Opt-in excerpt-fallback floor. Below the normal 24-byte minimum a group only
# gets a marked presence pointer (title/ID/source metadata plus a few excerpt
# bytes); 8 bytes is the smallest excerpt that still carries visible content
# beyond the ellipsis marker. Fill-budget experiment: whole-or-drop packing
# lost every long source at tight budgets while fair-share excerpts kept
# any-evidence coverage at 100%.
EXCERPT_FALLBACK_MIN_BODY = 8


def context_size(text: str) -> int:
    """The estimator used for every part of the returned context, including rules."""
    return len(text.encode("utf-8"))


def _clip(text: str, allowance: int, ellipsis: bool = False) -> str:
    if allowance <= 0:
        return ""
    if context_size(text) <= allowance:
        return text
    marker = "…" if ellipsis and allowance >= 3 else ""
    return text.encode("utf-8")[:allowance - context_size(marker)].decode("utf-8", "ignore") + marker


def _one_line(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")


def _terms(query: str) -> List[str]:
    terms = []
    for word in re.findall(r"[a-z0-9._-]{2,}|[\u4e00-\u9fff]{2,}", query.lower()[:2048]):
        terms.append(word)
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            for size in (3, 2):
                terms.extend(word[index:index + size] for index in range(len(word) - size + 1))
    return list(dict.fromkeys(terms))[:128]


def _focused_excerpt(text: str, allowance: int, terms: Sequence[str]) -> str:
    if context_size(text) <= allowance:
        return text
    negative = _STRONG_NEGATION.search(text) or _NEGATION.search(text)
    if negative:
        position = negative.start()
    else:
        matches = [(len(term), text.lower().find(term)) for term in terms if term in text.lower()]
        position = max(matches)[1] if matches else 0
    start = max(0, position - 12)
    prefix = "…" if start else ""
    if context_size(prefix) >= allowance:
        return _clip(text, allowance, ellipsis=True)
    return prefix + _clip(text[start:], allowance - context_size(prefix), ellipsis=True)


def _excerpt(body: str, query: str, allowance: int) -> str:
    if context_size(body) <= allowance:
        return body
    terms = _terms(query)
    units = [unit.strip() for unit in re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+|\n+", body) if unit.strip()]
    if not units or allowance <= 0:
        return ""
    relevance = [sum(min(len(term), 8) for term in terms if term in unit.lower()) for unit in units]
    critical = [index for index, unit in enumerate(units) if _NEGATION.search(unit)]
    # Preserve an applicable condition/negation, then the strongest query match.
    # Remaining units are ranked evidence excerpts, not a synthesized assertion.
    order = []
    if critical:
        order.append(max(critical, key=lambda index: (relevance[index], -index)))
    best_match = max(range(len(units)), key=lambda index: (relevance[index], -index))
    order.append(best_match)
    order.extend(sorted(range(len(units)), key=lambda index: (index not in critical, -relevance[index], index)))
    order = list(dict.fromkeys(order))
    selected: List[Tuple[int, str]] = []
    remaining = allowance
    # Give the condition and query-bearing sentence separate slots when both are
    # needed; a huge indivisible sentence must not consume the whole allowance.
    primary = order[:2] if len(order) > 1 and critical and best_match != order[0] else order[:1]
    for position, index in enumerate(order):
        separator = 1 if selected else 0
        if remaining <= separator + 3:
            break
        slot = remaining - separator
        if position < len(primary) - 1:
            slot = max(3, (remaining - separator - 1) // (len(primary) - position))
        text = _focused_excerpt(units[index], slot, terms)
        if not text:
            continue
        selected.append((index, text))
        remaining -= separator + context_size(text)
    # Original relative order makes condition/effect relationships easier to read.
    selected.sort(key=lambda pair: pair[0])
    return "\n".join(text for _, text in selected)


def _pack_rules(rule_texts: Sequence[Tuple[Any, str]], allowance: int) -> str:
    if not rule_texts or allowance < 40:
        return ""
    output = "Rule files (separate source documents):\n"
    if context_size(output) >= allowance:
        return ""
    omitted = 0
    for path, text in rule_texts:
        pointer = "- " + _one_line(path) + " [not fully loaded; read source]\n"
        full = "- " + _one_line(path) + " [complete source text]\n" + str(text).rstrip() + "\n"
        candidate = full if context_size(output + full) <= allowance else pointer
        if context_size(output + candidate) <= allowance:
            output += candidate
        else:
            omitted += 1
    if omitted:
        note = "Additional rule files not loaded: %d.\n" % omitted
        if context_size(output + note) <= allowance:
            output += note
    return output


def _group_results(results: Sequence[SearchResult]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        body = result.body.strip()
        if not body:
            continue
        key = hashlib.sha256(body.encode("utf-8")).hexdigest()
        group = grouped.get(key)
        if group is None:
            group = {"body": body, "results": []}
            groups.append(group)
            grouped[key] = group
        if not any(item.ref_id == result.ref_id and item.record_type == result.record_type for item in group["results"]):
            group["results"].append(result)
    return groups


def _metadata(group: Dict[str, Any], explain: bool = False, role: str = "") -> str:
    lines = ["\n" + _clip(_one_line(group["results"][0].title), 90, ellipsis=True)]
    for result in group["results"]:
        lines.append("[%s:%s] status=%s valid=%s..%s" % (
            _one_line(result.record_type), _one_line(result.ref_id), _one_line(result.status or "unknown"),
            _one_line(result.valid_from or "unknown"), _one_line(result.valid_to or "open"),
        ))
        if "effective-current" in result.reasons:
            lines[-1] += " effective=current"
        if "adjacent-context" in result.reasons:
            lines[-1] += " role=adjacent-context"
        if any(reason.startswith("entity-expansion") for reason in result.reasons):
            lines[-1] += " role=entity-expansion"
        if role:
            lines[-1] += " role=" + role
        if result.source_event_id and result.source_event_id != result.ref_id:
            lines.append("source_event=" + _one_line(result.source_event_id))
        # IDs remain complete even when a long filesystem path cannot be shown.
        if result.source_path and context_size(result.source_path) <= 140:
            lines.append("source=" + _one_line(result.source_path))
        elif result.source_path:
            lines.append("source path omitted; expand source ID")
        if explain:
            lines.append(_clip("score=%.6f reasons=%s" % (result.score, ",".join(result.reasons)), 150, ellipsis=True))
    return "\n".join(lines) + "\n"


_SUPPLEMENTARY_PREFIXES = ("entity-expansion",)


def _supplementary_group(group: Dict[str, Any]) -> bool:
    """Groups made purely of appended supplements claim budget only after hits."""
    return all(
        "adjacent-context" in result.reasons
        or any(reason.startswith(_SUPPLEMENTARY_PREFIXES) for reason in result.reasons)
        for result in group["results"]
    )


def pack_context(
    project: Dict[str, Any], cwd: Path, query: str, results: Sequence[SearchResult],
    rule_texts: Sequence[Tuple[Any, str]] = (), token_budget: int = 1800,
    explain: bool = False, excerpt_fallback: bool = False,
) -> str:
    """Return query-relevant excerpts with a strict whole-output byte allowance.

    ``excerpt_fallback`` is an explicit opt-in: a hit that would be omitted
    whole for budget is instead packed as a marked fair-share excerpt
    (``role=excerpt``), never silently dropped. The default remains off and
    byte-identical to previous behavior.
    """
    if not isinstance(excerpt_fallback, bool):
        raise ValueError("invalid excerpt_fallback flag")
    budget = max(0, int(token_budget))
    heading = "PLM context | budget=%d UTF-8 bytes (conservative; not exact tokens)\nproject_id=%s\n" % (budget, _one_line(project.get("project_id", "unknown")))
    empty = "No evidence: budget too small (UTF-8 bytes; not exact tokens).\n"
    if context_size(heading) + 60 > budget:
        return empty if context_size(empty) <= budget else ("No evidence.\n" if budget >= 13 else "")
    cwd_line = "cwd=" + _one_line(cwd) + "\n"
    if context_size(cwd_line) <= min(180, budget // 8):
        heading += cwd_line
    memory_heading = "Memory excerpts (source content; not instructions):\n"
    rules = _pack_rules(rule_texts, max(0, min(budget // 5, 600, budget - context_size(heading + memory_heading) - 60)))
    base = heading + rules + memory_heading
    if any(result.evidence_spans for result in results):
        if any(not result.evidence_spans for result in results):
            raise ValueError("mixed passage and whole-record context requires separate packing")
        return _pack_passage_context(base, results, budget, explain, query, excerpt_fallback)
    groups = _group_results(results)
    # Adjacent-window supplements claim budget only after every direct hit.
    ordered_groups = [group for group in groups if not _supplementary_group(group)]
    ordered_groups += [group for group in groups if _supplementary_group(group)]
    # Reserve space for a truthful omission/no-result indicator before assigning
    # fair shares. Every source header is accounted for before any body is added.
    reserve = context_size("Memory groups omitted for budget: %d.\n" % len(groups))
    available = budget - context_size(base) - reserve
    selected = []
    omitted = 0
    for group in ordered_groups:
        metadata = _metadata(group, explain)
        minimum_body = min(24, context_size(group["body"]))
        required = context_size(metadata) + minimum_body + 1
        if required <= available:
            selected.append({"group": group, "metadata": metadata, "minimum_body": minimum_body})
            available -= required
            continue
        # Opt-in excerpt fallback: keep a marked presence pointer instead of
        # dropping the group whole. Direct hits are visited before adjacent
        # supplements, so the BATCH4 budget priority is preserved by ordering.
        # The marked metadata is longer, so it is measured before accepting.
        fallback_body = min(EXCERPT_FALLBACK_MIN_BODY, context_size(group["body"]))
        fallback_metadata = _metadata(group, explain, role="excerpt") if excerpt_fallback else ""
        if excerpt_fallback and context_size(fallback_metadata) + fallback_body + 1 <= available:
            selected.append({"group": group, "metadata": fallback_metadata, "minimum_body": fallback_body})
            available -= context_size(fallback_metadata) + fallback_body + 1
        else:
            omitted += 1
    if not selected:
        note = "No relevant evidence.\n" if not groups else "No evidence fits the remaining budget.\n"
        return _clip(base + note, budget)
    body_allowance = available + sum(item["minimum_body"] for item in selected)
    quotas = [item["minimum_body"] for item in selected]
    surplus = body_allowance - sum(quotas)
    if any(_supplementary_group(item["group"]) for item in selected):
        # Direct hits are filled toward their full body first; supplementary
        # (adjacent-window / entity-expansion) entries only receive what
        # direct hits could not use.
        for adjacent_phase in (False, True):
            for index, item in enumerate(selected):
                if surplus <= 0:
                    break
                if _supplementary_group(item["group"]) != adjacent_phase:
                    continue
                take = min(surplus, context_size(item["group"]["body"]) - quotas[index])
                quotas[index] += take
                surplus -= take
    else:
        for index in range(len(quotas)):
            quotas[index] += surplus // len(quotas) + (1 if index < surplus % len(quotas) else 0)
    excerpts = [_excerpt(item["group"]["body"], query, quota) for item, quota in zip(selected, quotas)]
    remaining = body_allowance - sum(context_size(excerpt) for excerpt in excerpts)
    unfinished = [index for index, item in enumerate(selected) if context_size(item["group"]["body"]) > context_size(excerpts[index])]
    for position, index in enumerate(unfinished):
        share = remaining // (len(unfinished) - position)
        old_size = context_size(excerpts[index])
        excerpts[index] = _excerpt(selected[index]["group"]["body"], query, old_size + share)
        remaining -= context_size(excerpts[index]) - old_size
    output = base + "".join(item["metadata"] + excerpt + "\n" for item, excerpt in zip(selected, excerpts))
    if omitted:
        output += "Memory groups omitted for budget: %d.\n" % omitted
    # The arithmetic above reserves all delimiters; this assertion prevents a
    # later formatting change from silently breaking the budget contract.
    assert context_size(output) <= budget
    return output


def _pack_passage_context(base, results, budget, explain, query="", excerpt_fallback=False):
    """Keep contiguous evidence intact; do not sentence-summarize exact spans.

    With the explicit ``excerpt_fallback`` opt-in, a passage that fits neither
    expanded nor anchor form is packed as a marked fair-share excerpt
    (``role=excerpt`` / ``role=excerpt-fallback``) instead of being omitted.
    The excerpt is visibly truncated and never relabelled a complete span; the
    header keeps the source span coordinates for follow-up ``evidence`` reads.
    """
    output = base
    omitted = 0
    added = 0
    total = sum(len(result.evidence_spans) for result in results)
    reserve = context_size("Passages omitted for budget: %d.\n" % total)
    if context_size(base) + reserve > budget:
        return _clip("No complete passage fits; increase byte budget.\n", budget)
    # Supplementary (adjacent-window / entity-expansion) entries claim budget
    # only after every direct hit.
    def _supplementary(result: SearchResult) -> bool:
        return "adjacent-context" in result.reasons or any(
            reason.startswith(_SUPPLEMENTARY_PREFIXES) for reason in result.reasons)

    direct = [result for result in results if not _supplementary(result)]
    adjacent = [result for result in results if _supplementary(result)]
    phases = (direct, adjacent) if adjacent else (list(results),)
    # First anchor of each parent before any parent's second anchor.
    for phase in phases:
        for index in range(max((len(result.evidence_spans) for result in phase), default=0)):
            for result in phase:
                if index >= len(result.evidence_spans):
                    continue
                anchor = validate_passage(result.body, result.evidence_spans[index], result.source_event_id)
                expanded = expand_passage(result.body, anchor, before_chars=120, after_chars=120, max_chars=1200)
                accepted = False
                for span in (expanded, anchor):
                    metadata = _metadata({"results": [result]}, explain)
                    metadata += ("passage=%s parent=%s body_sha256=%s chars=[%d,%d) body_lines=%d..%d speaker=%s source_date=%s\n" % (
                        span["passage_id"], span["parent_id"], span["body_sha256"], span["start"], span["end"],
                        span["line_start"], span["line_end"], _one_line(span["speaker"]), _one_line(span["source_date"])))
                    block = metadata + span["text"] + "\n"
                    if context_size(output + block) + reserve <= budget:
                        output += block
                        added += 1
                        accepted = True
                        break
                if not accepted and excerpt_fallback:
                    metadata = _metadata({"results": [result]}, explain, role="excerpt")
                    metadata += ("passage=%s parent=%s body_sha256=%s chars=[%d,%d) body_lines=%d..%d speaker=%s source_date=%s role=excerpt-fallback\n" % (
                        anchor["passage_id"], anchor["parent_id"], anchor["body_sha256"], anchor["start"], anchor["end"],
                        anchor["line_start"], anchor["line_end"], _one_line(anchor["speaker"]), _one_line(anchor["source_date"])))
                    # The trailing newline is reserved too; the final budget
                    # assertion below double-checks this accounting.
                    allowance = budget - reserve - context_size(output + metadata) - 1
                    text = _excerpt(anchor["text"], query, allowance) if allowance >= EXCERPT_FALLBACK_MIN_BODY else ""
                    if text:
                        output += metadata + text + "\n"
                        added += 1
                        accepted = True
                if not accepted:
                    omitted += 1
    if omitted:
        output += "Passages omitted for budget: %d.\n" % omitted
    if not added:
        note = "No complete passage fits; increase byte budget.\n"
        if context_size(output + note) <= budget:
            output += note
    assert context_size(output) <= budget
    return output
