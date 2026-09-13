from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .model import PLMError, parse_time
from .ops import backup_memory, cutover, export_v1, install_skill, rollback_skill, verify_acceptance
from .service import (
    add_project_alias, consolidate, context_data, doctor, evidence_data, fact_history, forget, migrate_legacy,
    read_config, rebuild_index, record_usage, render_context, review_candidate, shadow_report, submit_candidate,
    supersede_fact, usage_report, write_event, write_fact,
)


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plm", description="Project Long Memory v2")
    sub = parser.add_subparsers(dest="command", required=True)

    context = sub.add_parser("context")
    context.add_argument("--cwd", default=str(Path.cwd()))
    context.add_argument("--query", default="")
    context.add_argument("--limit", type=int, default=6)
    context.add_argument("--view", choices=("current", "history", "both"), default="current")
    context.add_argument("--scope", choices=("global", "project", "agent", "session", "shared"), default="project")
    context.add_argument("--scope-id", default="")
    context.add_argument("--token-budget", type=int, default=1800)
    context.add_argument("--explain", action="store_true")
    context.add_argument("--json", action="store_true")
    context.add_argument("--with-code", action="store_true",
                         help="opt-in: append current-project CodeGraph evidence to memory context")
    context.add_argument("--code-query", default="",
                         help="optional focused symbols for --with-code; defaults to --query")
    context.add_argument("--code-budget", type=int, default=16000,
                         help="maximum UTF-8 bytes of additional code text (256..64000)")
    context.add_argument("--retrieval-profile", choices=("lexical", "passages"), default="lexical")
    context.add_argument("--adjacent", type=int, default=0,
                         help="opt-in: also pack N recorded_at neighbours before/after each hit (0=off)")
    context.add_argument("--recency-boost", action="store_true",
                         help="opt-in: inject date expressions into scoring text and add a small recency tie-breaker")
    context.add_argument("--excerpt-fallback", action="store_true",
                         help="opt-in: pack a marked excerpt instead of dropping a hit that does not fit whole")
    context.add_argument("--expand-entities", type=int, default=0,
                         help="opt-in: also append records within N entity hops of each hit (0=off, max 2)")
    context.add_argument("--mmr", action="store_true",
                         help="opt-in: reorder the ranking head by incremental MMR (relevance vs redundancy)")
    context.add_argument("--abstain-threshold", type=float, default=0.0,
                         help="opt-in: withhold candidates whose best absolute channel cosine is below this floor (0=off); "
                              "empty result means insufficient evidence")
    context.add_argument("--valid-at", default="",
                         help="opt-in: ISO 8601 valid-time filter — only Facts valid at that instant "
                              "(valid_from<=t, valid_to empty or >t) and Events observed by then")
    context.add_argument("--known-at", default="",
                         help="opt-in: ISO 8601 system-time filter — only memories whose assertion Event "
                              "was recorded at or before that instant")
    context.add_argument("--soft-supersede", action="store_true",
                         help="opt-in: recall superseded Fact versions with a fixed ranking demotion "
                              "instead of excluding them (lifecycle/privacy unchanged)")
    context.add_argument("--temporal-intent", action="store_true",
                         help="opt-in: extract query-side temporal intent — an absolute as-of anchor "
                              "fills --valid-at, comparison phrasing enables --soft-supersede "
                              "(explicit flags win; relative anchors are reported but never applied)")
    context.add_argument("--window-reserve", action="store_true",
                         help="opt-in: reserve up to a quarter of the window's tail slots for "
                              "--adjacent neighbours (requires --adjacent > 0), instead of appending "
                              "them beyond the window")
    context.add_argument("--coverage-rerank", action="store_true",
                         help="opt-in: greedily reorder the ranking head to cover rare query "
                              "information words not yet covered by already-selected results")

    code = sub.add_parser("code", help="manage/query the optional local CodeGraph asset")
    code.add_argument("action", choices=("init", "sync", "status", "explore", "query", "callers", "callees", "impact", "files"))
    code.add_argument("--cwd", default=str(Path.cwd()))
    code.add_argument("--query", default="")
    code.add_argument("--max-bytes", type=int, default=16000)
    code.add_argument("--timeout", type=int, default=None)

    evidence = sub.add_parser("evidence", help="expand a verified current Event body span (read-only)")
    evidence.add_argument("--cwd", default=str(Path.cwd()))
    evidence.add_argument("--event-id", required=True)
    evidence.add_argument("--body-sha256", required=True)
    evidence.add_argument("--start", type=int, required=True, help="Unicode code-point start offset, inclusive")
    evidence.add_argument("--end", type=int, required=True, help="Unicode code-point end offset, exclusive")
    evidence.add_argument("--scope", choices=("global", "project", "agent", "session", "shared"), default="project")
    evidence.add_argument("--scope-id", default="")
    evidence.add_argument("--before-chars", type=int, default=120)
    evidence.add_argument("--after-chars", type=int, default=120)
    evidence.add_argument("--max-chars", type=int, default=1800)

    write = sub.add_parser("write")
    write.add_argument("--cwd", default=str(Path.cwd()))
    write.add_argument("--title", required=True)
    write.add_argument("--tags", default="")
    write.add_argument("--content", default="")
    write.add_argument("--type", choices=("core", "fact", "episode", "procedure", "artifact_ref"), default="episode")
    write.add_argument("--scope", choices=("global", "project", "agent", "session", "shared"), default="project")
    write.add_argument("--scope-id", default="")
    write.add_argument("--observed-at", default="")
    write.add_argument("--source", default="agent")
    write.add_argument("--idempotency-key", default="")
    write.add_argument("--fact-key", default="")
    write.add_argument("--value-json", default="")
    write.add_argument("--confidence", type=float, default=0.8)
    write.add_argument("--source-event-id", default="")
    write.add_argument("--evidence-kind", choices=("", "user_confirmation", "git_commit", "command_output", "file_hash", "test_result"), default="")
    write.add_argument("--evidence-json", default="")

    candidate = sub.add_parser("candidate")
    candidate.add_argument("--cwd", default=str(Path.cwd()))
    candidate.add_argument("--payload", default="")
    candidate.add_argument("--payload-file", default="")
    candidate.add_argument("--idempotency-key", default="")

    consolidate_parser = sub.add_parser("consolidate")
    consolidate_parser.add_argument("--limit", type=int, default=20)
    consolidate_parser.add_argument("--dedup-similarity", type=float, default=0.0,
                                    help="opt-in: near-duplicate detection threshold in (0,1] against active Facts "
                                         "of the same project+scope before writing (0=off; reference 0.85)")
    consolidate_parser.add_argument("--dedup-action", choices=("review", "skip", "supersede"), default="review",
                                    help="behavior above --dedup-similarity: review = candidate to "
                                         "needs_confirmation for candidate-review (default, conservative); "
                                         "skip = do not write the duplicate, keep the existing fact; "
                                         "supersede = write the new value superseding the match (same fact_key only)")

    candidate_review = sub.add_parser("candidate-review")
    candidate_review.add_argument("--cwd", default=str(Path.cwd()))
    candidate_review.add_argument("--candidate-id", required=True)
    candidate_review.add_argument("--decision", choices=("approve", "reject"), required=True)
    candidate_review.add_argument("--evidence-kind", choices=("", "user_confirmation", "git_commit", "command_output", "file_hash", "test_result"), default="")
    candidate_review.add_argument("--evidence-json", default="")

    history = sub.add_parser("history")
    history.add_argument("--cwd", default=str(Path.cwd()))
    history.add_argument("--fact-key", default="")
    history.add_argument("--scope", default="project", choices=("global", "project", "agent", "session", "shared"))
    history.add_argument("--scope-id", default="")

    supersede = sub.add_parser("supersede")
    supersede.add_argument("--cwd", default=str(Path.cwd()))
    supersede.add_argument("--fact-id", required=True)
    supersede.add_argument("--value", required=True)
    supersede.add_argument("--value-json", action="store_true")
    supersede.add_argument("--valid-from", default="")
    supersede.add_argument("--confidence", type=float, default=0.9)
    supersede.add_argument("--evidence-kind", choices=("", "user_confirmation", "git_commit", "command_output", "file_hash", "test_result"), default="")
    supersede.add_argument("--evidence-json", default="")

    forget_parser = sub.add_parser("forget")
    forget_parser.add_argument("--cwd", default=str(Path.cwd()))
    forget_parser.add_argument("--target-id", required=True)
    forget_parser.add_argument("--purge", action="store_true")

    sub.add_parser("doctor")
    usage = sub.add_parser("usage")
    usage.add_argument("--days", type=int, default=7)
    migrate = sub.add_parser("migrate")
    migrate.add_argument("--dry-run", action="store_true")
    sub.add_parser("rebuild-index")
    sub.add_parser("shadow-report")
    alias = sub.add_parser("alias")
    alias.add_argument("--cwd", default=str(Path.cwd()))
    alias.add_argument("--path", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--tests-passed", action="store_true", help=argparse.SUPPRESS)

    backup = sub.add_parser("backup")
    backup.add_argument("--destination", default="")
    restore = sub.add_parser("restore")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--destination", required=True, help="new or empty memory root; current store is never overwritten")
    restore.add_argument("--deletion-ledger", required=True, help="operator-supplied latest deletion ledger, not an old archive copy")
    export = sub.add_parser("export")
    export.add_argument("--destination", required=True)
    export.add_argument("--since", default="")
    install = sub.add_parser("install")
    install.add_argument("--target", default="")
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--target", default="")
    rollback.add_argument("--materialize-v1", action="store_true")
    sub.add_parser("cutover")
    sub.add_parser("config")
    return parser


def _payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.payload_file:
        raw = Path(args.payload_file).read_text(encoding="utf-8")
    elif args.payload:
        raw = args.payload
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        raise PLMError("candidate JSON payload is required")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise PLMError("candidate payload must be an object")
    return value


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "context":
            code_evidence = None
            if args.with_code:
                if args.scope != "project" or args.scope_id:
                    raise PLMError("--with-code requires the current project scope without --scope-id")
                from .codegraph import query as code_query
                code_evidence = code_query(Path(args.cwd), args.code_query or args.query,
                                           max_bytes=args.code_budget)
            try:
                valid_at = parse_time(args.valid_at) if args.valid_at else ""
                known_at = parse_time(args.known_at) if args.known_at else ""
            except PLMError as exc:
                raise PLMError("invalid --valid-at/--known-at timestamp") from exc
            if args.json:
                started = time.perf_counter()
                project, results = context_data(Path(args.cwd), args.query, args.limit, args.view, args.scope, args.scope_id, args.retrieval_profile, adjacent=args.adjacent, recency_boost=args.recency_boost, expand_entities=args.expand_entities, mmr=args.mmr, abstain_threshold=args.abstain_threshold, valid_at=valid_at, known_at=known_at, soft_supersede=args.soft_supersede, temporal_intent=args.temporal_intent, window_reserve=args.window_reserve, coverage_rerank=args.coverage_rerank)
                try:
                    record_usage(
                        project["project_id"], "context", "plm-json", args.query, len(results), True,
                        (time.perf_counter() - started) * 1000.0,
                    )
                except Exception:
                    pass
                response = {"project": project, "results": [item.as_dict() for item in results]}
                if code_evidence is not None:
                    response["code_evidence"] = code_evidence
                _json(response)
            else:
                print(render_context(
                    Path(args.cwd), args.query, args.limit, args.view, args.token_budget,
                    args.explain, args.scope, args.scope_id, args.retrieval_profile, adjacent=args.adjacent,
                    recency_boost=args.recency_boost, excerpt_fallback=args.excerpt_fallback,
                    expand_entities=args.expand_entities, mmr=args.mmr,
                    abstain_threshold=args.abstain_threshold,
                    valid_at=valid_at, known_at=known_at, soft_supersede=args.soft_supersede,
                    temporal_intent=args.temporal_intent,
                    window_reserve=args.window_reserve, coverage_rerank=args.coverage_rerank,
                ), end="")
                if code_evidence is not None:
                    print("\nCode evidence (on demand, independent of memory Events):")
                    _json(code_evidence)
        elif args.command == "code":
            from . import codegraph
            if args.action == "status":
                _json(codegraph.status(Path(args.cwd)))
            elif args.action in {"init", "sync"}:
                _json(codegraph.sync(Path(args.cwd), initialize=args.action == "init",
                                     timeout=args.timeout if args.timeout is not None else 180))
            else:
                result = codegraph.query(Path(args.cwd), args.query, args.action,
                                         args.max_bytes, args.timeout if args.timeout is not None else 30)
                _json(result)
                if result["status"] != "ready":
                    return 2
        elif args.command == "evidence":
            _json(evidence_data(
                Path(args.cwd), args.event_id, args.body_sha256, args.start, args.end,
                args.scope, args.scope_id, args.before_chars, args.after_chars, args.max_chars,
            ))
        elif args.command == "write":
            content = args.content if args.content else sys.stdin.read()
            if args.type == "fact":
                if not args.fact_key:
                    raise PLMError("--fact-key is required for --type fact")
                value = json.loads(args.value_json) if args.value_json else content
                event = write_fact(
                    Path(args.cwd), args.fact_key, value, title=args.title, tags=args.tags,
                    confidence=args.confidence, source=args.source, idempotency_key=args.idempotency_key,
                    scope=args.scope, scope_id=args.scope_id, observed_at=args.observed_at,
                    source_event_id=args.source_event_id, evidence_kind=args.evidence_kind,
                    evidence=json.loads(args.evidence_json) if args.evidence_json else {},
                )
            else:
                event = write_event(
                    Path(args.cwd), args.title, content, tags=args.tags, kind=args.type,
                    scope=args.scope, scope_id=args.scope_id, observed_at=args.observed_at,
                    source=args.source, idempotency_key=args.idempotency_key,
                )
            _json({"event_id": event.event_id, "path": str(event.path)})
        elif args.command == "candidate":
            record = submit_candidate(Path(args.cwd), _payload(args), args.idempotency_key)
            response: Dict[str, Any] = dict(record)
            if read_config().get("auto_consolidate"):
                response["consolidation"] = consolidate(1, record["candidate_id"])
            _json(response)
        elif args.command == "consolidate":
            _json(consolidate(args.limit, dedup_similarity=args.dedup_similarity, dedup_action=args.dedup_action))
        elif args.command == "candidate-review":
            evidence = json.loads(args.evidence_json) if args.evidence_json else {}
            _json(review_candidate(Path(args.cwd), args.candidate_id, args.decision, args.evidence_kind, evidence))
        elif args.command == "history":
            _json(fact_history(Path(args.cwd), args.fact_key, args.scope, args.scope_id))
        elif args.command == "supersede":
            value = json.loads(args.value) if args.value_json else args.value
            event = supersede_fact(
                Path(args.cwd), args.fact_id, value, args.valid_from, args.confidence,
                args.evidence_kind, json.loads(args.evidence_json) if args.evidence_json else {},
            )
            _json({"event_id": event.event_id, "fact_id": event.metadata["fact_id"], "path": str(event.path)})
        elif args.command == "forget":
            event = forget(Path(args.cwd), args.target_id, args.purge)
            _json({"event_id": event.event_id, "path": str(event.path)})
        elif args.command == "doctor":
            _json(doctor())
        elif args.command == "usage":
            _json(usage_report(args.days))
        elif args.command == "migrate":
            _json(migrate_legacy(args.dry_run))
        elif args.command == "rebuild-index":
            _json(rebuild_index())
        elif args.command == "shadow-report":
            _json(shadow_report())
        elif args.command == "alias":
            _json(add_project_alias(Path(args.cwd), Path(args.path)))
        elif args.command == "verify":
            _json(verify_acceptance(args.tests_passed))
        elif args.command == "backup":
            _json(backup_memory(Path(args.destination) if args.destination else None))
        elif args.command == "restore":
            from .ops import restore_backup
            _json(restore_backup(Path(args.archive), Path(args.destination), Path(args.deletion_ledger)))
        elif args.command == "export":
            _json(export_v1(Path(args.destination), args.since))
        elif args.command == "install":
            _json(install_skill(Path(args.target) if args.target else None))
        elif args.command == "rollback":
            _json(rollback_skill(Path(args.target) if args.target else None, args.materialize_v1))
        elif args.command == "cutover":
            _json(cutover())
        elif args.command == "config":
            _json(read_config())
        return 0
    except (PLMError, json.JSONDecodeError, OSError) as exc:
        print("plm: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
