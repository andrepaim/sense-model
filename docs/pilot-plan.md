# English and Portuguese annotation pilot

Status: workflow implemented; human collection/review not yet executed. Scope: reproducible collection and review
workflow for 100 accepted natural examples per language. Human annotation is an
explicit step; software tests/demo records never count as completed annotation.

## Objectives and sample

Collect approximately 250 candidate occurrences per language, then review to
retain 100 per language. Aim for 40 general ambiguous, 30 technical ambiguous,
20 genuine dictionary gaps, 10 single-candidate checks, with phrases and
inflections across these groups. Buckets are planned recruitment goals, not
labels inferred from model predictions. Review identifies true none-fits cases;
never delete the correct candidate to manufacture natural absence. Report
shortfalls transparently. Prefer >=25 independent documents and cap at 5
accepted targets per document and 3 per lemma per language. Portuguese defaults
to pt-BR; retain explicit pt-PT metadata rather than silently merging variants.
The pilot is a development resource, not the future frozen test benchmark.

## Collection and inventory

Pin dictionary edition, checksum, language of definitions, candidate ordering,
and licenses before collection. Use full deployment candidate inventories,
without model-based pruning. Audit Portuguese gloss coverage before scaling.
Accept local source documents JSONL (id,text,language,domain,source_url,revision,
license,license_url,source) and local candidate inventories JSONL as already
supported by wsd prepare. Match explicit surface/lookup forms with exact code
point offsets; retain all occurrences and contiguous multiword forms. No POS
model is assumed: record inventory POS; human flags bad inventory/extraction.
Prioritize dictionary-ambiguous occurrences with deterministic seeded sampling,
per-document/per-lemma caps, normalization duplicate checks and source manifest.
The UI imports prepared candidate JSONL without known gold labels. Real examples
need complete provenance; demo mode is visibly separate and excluded from real
exports. Do not silently scrape private pdfpal libraries or change licenses.

## Local annotation application

Ship optional tooling in this repository, launched with `wsd annotate`.
Loopback-only HTTP server, SQLite workspace outside Git, vanilla HTML/CSS/JS
assets in the wheel. No inference/model import or heavyweight frontend build.
An import panel loads prepared JSONL; a dashboard shows EN/PT targets, reviewed,
independent review counts, disagreements, adjudication, assistance usage, flags,
and target bucket coverage. Named reviewer identities allow independent rounds;
they are workflow labels for one trusted workstation, not authentication.

Reviewer sees source/domain/language/license, highlighted exact occurrence,
shuffled full definitions, multiple-choice checkboxes, note and these outcomes:
selected definitions, none fits, insufficient context, extraction issue,
inventory issue, skip. Save independent first vote before AI assistance unlocks.
The same reviewer may make a separately recorded final revision after assistance;
retain the immutable initial vote and the event history. A second reviewer sees
neither previous votes nor previous AI suggestions before their first vote.
Blind order is stable per reviewer/example and stored independently of canonical
frequency order. Notes and source text render as text, never executable HTML.
Keyboard actions, progress, filtering and undo via revision are provided.

Adjudication reveals completed reviewers' votes, assistance exposure and notes.
Require two distinct independent initial votes for a real export. Agreements can
be finalized by an explicit adjudicator action. Disagreements/multiple acceptable
senses require an adjudication note. Retain flags as flags; never turn
insufficient context or extraction problems into none-fits gold labels.
No annotation label is automatically accepted because Jev agrees.

## Jev assistance and credentials

TypeSafe direct System One API via fixed `https://api.typesafe.ai/v1/systemone`.
Also support OpenRouter's fixed compatible endpoint for an OpenRouter key;
provider is explicit and never silently changed. Model is configurable with
provider default; response records the resolved version. Official TypeSafe API
shape is state + questions map, Choice with criteria map, returned answers map.
Jev is a typed decision model: show scores/suggestions, not invented explanations.
Request one primary choice over canonical meanings plus explicit none-fits and
insufficient-context options. Optional independent per-definition Noul judgments
can indicate multiple plausible meanings; they are not logical dependencies.

Settings has password input, provider/model, Save for this browser session,
Clear key, connection status, and explicit disclosure of what is sent. Browser
keeps the key in memory and clears the input after sending; server stores the key
only in its expiring session memory. Never persist in SQLite, localStorage,
URLs, logs, exports or Git. No key is requested in chat. Explicit assistance
click sends only the target sentence/span/language and candidate definitions,
without source URLs, titles, notes, reviewer IDs or local paths. Each call may
incur provider charges. Reuse recorded result for the same example/content/model
within a review session; expose explicit refresh and count requests/token usage.

Allow requesting assistance only after a recorded independent vote, enforced on
server. Record exposure before returning suggestions. Reject malformed response,
unknown choices, invalid probabilities; timeout and authentication errors are
sanitized. No automatic paid retry. Keep provider responses separate from human
gold; record model, prompt version, request fingerprint, usage and timestamps.

## Persistence, security and exports

SQLite with transactions, foreign keys, workspace schema version, immutable
initial votes and append-only audit events. Imported records immutable by ID;
identical imports idempotent, changed records rejected to protect existing votes.
Optimistic record revisions prevent stale tabs overwriting annotations.
Structured JSON audit backup includes records, review history, adjudications and AI
metadata, never keys. Gold export emits existing Example schema with provenance,
annotation protocol, review counts/AI exposure and source attribution; only
adjudicated selected/none-fit outcomes qualify. Canonical candidate order stays
unchanged. Pilot data never silently becomes frozen test. Exports include
manifest with hashes, coverage, unresolved flags and annotation agreement.

Server binds 127.0.0.1; validates Host/Origin and requires a random per-session
CSRF token for mutations, including key submission and export. Credentials are
scoped to a browser session; sessions expire and clear on process restart.
Request/body limits, fixed outbound allowlist with no redirects, bounded timeouts,
parameterized SQL, static-path allowlist and no content-bearing request logs.
This is local annotation tooling, not a public multiuser hosted service.

## Deliverables and sequence

1. Plan/protocol and shared application API contract.
2. Candidate collection and provenance/coverage audit, deterministic manifests.
3. SQLite review store and independent/adjudication workflow.
4. Local web server and accessible review/settings/import/export interface.
5. Jev adapter with direct/OpenRouter support and mocked tests.
6. Integrated browser tests, backend tests, packaging, docs and CI.
7. Public repository push and verified CI; local launch instructions.
8. Human pilot execution: acquire inventories/documents, collect pool, review
   first 20 per language, revise protocol, then finish 100 per language.

Step 8 requires actual source material, reviewers and the user's key entered
locally. Implementation must not report it complete using authored demo data.

## Acceptance and pilot report

Software: no suggestion before first vote; identity switch restores blindness;
independent votes survive refresh; multiple golds, none-fit and flags distinct;
conflict detection; failed requests preserve votes; key absent from disk/exports;
untrusted text inert; actual browser import/vote/assist/adjudicate/export tested
with a mocked Jev response, no live billed calls in automated tests. Wheel ships
all UI assets; existing model tests remain green.

Pilot report: accepted counts and bucket/document/lemma distributions per
language; missing inventories/glosses; first-vote exact-set agreement; flagged
and adjudicated rates; changed answers after AI exposure; time per review;
provider/model/token usage; total human effort and unresolved provenance issues.
Quality gates for scaling: 100 accepted per language, complete traceable
provenance, two initial votes each, resolved disagreement notes, no unresolved
flags in gold, no source leakage when allocating future splits. Review agreement
and time before deciding the larger dataset/annotation budget. No model accuracy
release gate is inferred from these pilot counts.

## API sources checked

- https://docs.typesafe.ai/api/system-one.md
- https://docs.typesafe.ai/primitives/choice.md
- https://docs.typesafe.ai/primitives/noul.md
- https://docs.typesafe.ai/models.md
- https://openrouter.ai/docs (compatible provider route verified during implementation)
