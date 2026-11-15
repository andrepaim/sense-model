# Automatic dataset labeling

The labeler runs Jev on every input occurrence, then independently asks the local
Codex CLI to judge decisions below the confidence threshold, decisions with no
confidence, and failed Jev requests. The default threshold is **0.80**, inclusive.
Jev's explicit `confidence` field controls routing; the largest candidate
probability never substitutes for missing confidence.

```bash
wsd config init
# Edit ~/.config/sense-model/config.toml and set jev.api_key.
wsd label --input data/pilot/en/samples.jsonl
# Equivalent standalone entry point:
.venv/bin/python scripts/label_dataset.py --input data/pilot/en/samples.jsonl
```

The input is ordinary `Example` JSONL with complete candidate definitions and
Unicode code point offsets. With no flags, the input is
`data/pilot/en/samples.jsonl` and output is its sibling `labeled/` directory.
Use a new output directory for a new dataset version.

```bash
wsd label --input data/pilot/en/samples.jsonl --output data/pilot/en/labeled \
  --confidence-threshold 0.80 --jev-concurrency 4 --codex-concurrency 2 \
  --limit 100 --config ~/.config/sense-model/config.toml
```

`--judge-model` overrides `codex.model`. The runner uses the configured model
without silently substituting another model. Authenticate the installed local
Codex CLI before labeling. The normal config defaults are:

```toml
[jev]
provider = "typesafe"
model = "jev-latest"
api_key = "" # Set privately; never commit.
confidence_threshold = 0.80
concurrency = 4
max_retries = 2

[codex]
model = "gpt-6-luna"
concurrency = 2
max_retries = 1
timeout_seconds = 180
```

`jev.provider` also accepts `openrouter`; set its appropriate model explicitly
(for example, `typesafe/jev-1.13`). A blank `api_key` falls back to
`TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`. An optional `api_key_env` names a
different environment variable. `max_retries` counts retries after the initial
attempt; malformed responses also count as failures. Each setting is bounded,
and each provider has its own concurrency limit. There are no unbounded retry
loops.
Retries use exponential backoff capped at two seconds. Missing credentials fail
before model calls; a resumed run with all Jev decisions already cached can run
without a Jev key.

The judge sees only the sentence, exact target span, languages, and candidate
IDs with definitions. It receives no Jev answer, confidence, provenance, source
labels, or previous human decisions. Its structured result contains
`outcome`, `selected_ids`, and `rationale`. `selected` requires exactly one
supplied candidate ID. `none_fits` means the inventory lacks the intended
meaning. `insufficient_context` means the sentence cannot support a decision.
These two outcomes require empty `selected_ids`. Unknown candidate IDs and
inconsistent outcomes fail validation and are never used as labels.

High confidence Jev selections and `none_fits` decisions become silver labels.
When Codex is needed, its valid selection becomes the proposed silver label,
including when it disagrees with Jev. Both decisions remain available for
review. `none_fits` is a usable negative label (`gold_ids: []`);
`insufficient_context` remains unresolved (`gold_ids: null`). A high confidence
Jev abstention stays unresolved without invoking Codex. Failed calls also leave
labels unresolved when no usable decision is available; they remain distinct
from successful abstentions.

The historical field name `gold_ids` holds the proposed label for schema
compatibility. Every output occurrence has `annotation: "silver"`,
`review_status: "unreviewed"`, and `provenance.automatic`:

```json
{
  "status": "labeled",
  "bucket": "jev_low_codex_disagreed",
  "selected_by": "codex",
  "jev": {"choice": "sense-a", "selected_ids": ["sense-a"], "confidence": 0.6},
  "codex": {"outcome": "selected", "selected_ids": ["sense-b"], "rationale": "Context supports this definition."},
  "escalation_reason": "jev_low_confidence",
  "threshold": 0.8,
  "label_tier": "silver",
  "jev_error": null,
  "codex_error": null
}
```

Other buckets are `jev_high_confidence`, `jev_low_codex_agreed`,
`jev_error_codex`, and `unresolved`. Escalation reasons distinguish
`jev_low_confidence`, `jev_missing_confidence`, and `jev_error`; high confidence
decisions have no escalation reason. Errors contain a fixed safe type and
attempt count. Known Codex runner failures also retain its sanitized diagnostic
(for example, missing CLI or unavailable model); raw provider response bodies
and arbitrary exception messages are never stored. The Jev
result preserves the adapter's provider/model metadata, typed choice,
probabilities, confidence, and usage. Review these proposals before evaluation;
automatic labels remain silver until explicitly approved or corrected by a
human reviewer.

Each output directory contains:

- `samples.jsonl`: schema-compatible occurrences and proposed labels.
- `judgments.jsonl`: both provider results, safe errors, attempts, request
  fingerprints, and the current routing decision for each occurrence.
- `manifest.json`: input checksum, provider/model and prompt versions,
  threshold, processing status, and completion flag.
- `labeling-report.json`: label and bucket counts, confidence histogram,
  escalation and agreement counts, errors, successful abstentions, provider
  attempts and latency, and Jev token usage. Codex token usage is `null` because
  the structured CLI response does not expose it.
- `events.jsonl`: timestamped progress events without sentence text, prompts,
  provider bodies, or credentials. Events are also emitted to stderr.

Artifacts are written atomically after each completed provider call. Resume a
partial run or reroute a finished run with:

```bash
wsd label --input data/pilot/en/samples.jsonl --resume
wsd label --input data/pilot/en/samples.jsonl --resume --confidence-threshold 0.90
wsd label --input data/pilot/en/samples.jsonl --resume --retry-failed
```

Resume reuses successful decisions only when their content, prompt/schema,
model, and provider fingerprints match. Changing only the threshold reroutes
cached decisions, invoking Codex only for newly escalated occurrences that lack
a cached judge decision. Earlier Codex judgments are retained even when a new
threshold routes an occurrence through Jev. Changing a key does not discard
decisions. Cached failures are retained unless `--retry-failed` is supplied;
successful abstentions are decisions and are not retried by that flag.

`processed: true` means every required call was attempted or had a cached result.
If a provider operation required by the current routing still has an error, status is
`completed_with_errors`, `complete` stays false, and the CLI exits nonzero even
if Codex rescued a failed Jev call with a usable silver label. Completed results
remain saved. Fix authentication, model access, connection or response issues,
then run `--resume --retry-failed`. Jev is always required; Codex is required
only for current escalations. Historical Codex errors remain in the evidence and
`codex_errors` count when lowering the threshold removes an escalation, but
`inactive_codex_errors` do not prevent completion; `active_codex_errors` counts
failures that still need a judge result. A successful run uses status `complete` and
`complete: true`; it can still contain unresolved cases from legitimate
`insufficient_context` decisions.

An interrupted run leaves `processed: false` and `complete: false` and resumes
from its saved completed calls. Interruption cancels queued calls promptly;
already-running calls can finish within their provider timeout and cannot
overwrite the saved checkpoint.
Relabeling clears active review decisions and archives them in
`provenance.previous_review`, so an old approval cannot approve newly proposed
labels. The input cannot also be an output artifact.
