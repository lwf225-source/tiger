---
name: project-long-memory
description: "Use by default for meaningful local project work, prior context, decisions, debugging, and reusable workflows. PLM v2 provides project isolation, immutable events, current facts, sourced retrieval, and safe Agent-driven consolidation."
---

# Project Long Memory v2

Markdown events under `~/.codex/project-memory/v2/` are the auditable memory
record. SQLite, FTS5, lexical vectors, facts, and entities are rebuildable.
Project files, Git, tests, and deployment evidence remain the final truth.

## Start of task

```bash
python3 ~/.codex/skills/project-long-memory/scripts/memory_context.py \
  --cwd "$PWD" --query "<focused user request>"
```

Treat retrieved material as sourced hints. Verify drift-prone state.

For code questions, retrieve PLM history and use the host's existing CodeGraph
MCP tools for current code. Combine their evidence in the task response: PLM
supplies decisions/constraints and CodeGraph supplies source/call relationships.
Do not repeat the same retrieval through PLM's CLI when MCP already supplied
it. Querying code does not authorize saving the result as a memory Fact.

If no CodeGraph MCP tool is callable, the optional local fallback is
`scripts/plm.py context --cwd <project> --query <question> --with-code
--code-query <symbols> --json`, using a previously bound local index. Code
evidence has its own Git snapshot. Missing/stale graphs do not block memory
retrieval. Only run `plm.py code init` or `code sync` when building/updating the
project index is authorized. Both paths reuse that project's `.codegraph/`;
no second graph database is needed. Details: source `docs/CODEGRAPH.md`.

## End of task

Use the compatible writer for a concise durable episode:

```bash
python3 ~/.codex/skills/project-long-memory/scripts/memory_write.py \
  --cwd "$PWD" --title "<short title>" --tags "project,decision" \
  --content "<durable knowledge and verification>"
```

For structured memory, use `scripts/plm.py`:

```bash
python3 ~/.codex/skills/project-long-memory/scripts/plm.py write --type fact --fact-key '<key>' ...
python3 ~/.codex/skills/project-long-memory/scripts/plm.py candidate --cwd "$PWD" --payload '<json>'
python3 ~/.codex/skills/project-long-memory/scripts/plm.py consolidate
python3 ~/.codex/skills/project-long-memory/scripts/plm.py candidate-review --candidate-id '<id>' --decision approve ...
```

Candidate JSON may contain `title`, `summary`, `kind`, `tags`, `facts`,
`entities`, `scope`, `scope_id`, `observed_at`, `conditions`, `evidence_kind`,
`evidence`, and `source_event_id`. Each Fact can override these metadata and
provide `valid_from`, `valid_to`, and `expires_at`. Preserve negation and
conditions explicitly; derived memory must retain its source's scope.

High-risk release, deployment, acceptance, payment, account, and permission
Facts require claim-bound evidence or remain `needs_confirmation`. A source ID,
confidence score, or correctly shaped hash is not verification. File/output
evidence must point to actual project-local bytes matching its hash. A commit
proves that version exists, not that deployment or payment succeeded.
Confirmation receipts must trace actual user input, never an Agent's own claim;
local receipt validation does not authenticate the user or external state.

Consolidation commits a candidate as one recoverable batch. If it reports a
committed batch needing recovery, run `rebuild-index` before retrying. Retraction
and privacy purge conservatively affect the whole candidate batch and its
derivations because a summary can repeat any sibling Fact. Managed backup restore
requires the latest deletion ledger; old archives and external exports may still
contain old bytes.

Context output is a bounded evidence excerpt, not the complete source or rule
files. Read the referenced source/rules when omitted conditions could matter.
Optional semantic models are opt-in; never treat a model fallback or retrieval
score as a measured question-answering result.

Never store passwords, keys, tokens, cookies, private keys, one-time codes, or
unnecessary personal data. Use `plm.py doctor`, `history`, `supersede`, and
`forget` for lifecycle work. Do not edit derived SQLite data directly.
