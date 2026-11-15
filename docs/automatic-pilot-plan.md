# Automatic research-paper dataset pilot

Implemented workflow: local paper manifest and passage selection → Jev on every
example → independent local Codex for low/missing Jev confidence or request
failure → single-human audit → compatible reviewed JSONL. This supersedes the
human-first workflow in `pilot-plan.md` for newly generated automatic datasets.
The original manual annotation mode remains available.

## Contracts and defaults

- Input is `papers.json`, listing local paths (normally in `papers/`) and language.
  Every listed paper is processed in order. `--samples-per-paper` defaults to 10:
  5 easy, 3 medium, 2 hard. Other counts use proportional 50/30/20 integer quotas.
  Total requested examples = number of listed papers × samples per paper.
- Start with 100 per language to test collection and review; scale to 1,000 per
  language after inspecting corrections, inventory coverage and provider usage.
- Text and target spans come verbatim from extracted papers; the LLM chooses
  examples and provisional difficulty, not invented paper quotations.
- Definitions come from a supplied, pinned dictionary inventory. English and
  Portuguese definitions must declare their language. Missing coverage is reported.
- Default local CLI model is `gpt-6-luna`, explicitly selected by the user after
  the installed model catalog did not offer Luna 6.1. CLI overrides are supported;
  unavailable models never trigger a silent fallback.
- Local Codex means the locally installed CLI, using existing authentication.
  Inference is hosted: selected passages and definitions are sent to its provider.
  This is not an offline LLM pipeline.
- Every example retains the existing `Example` JSONL schema. `gold_ids: null`
  means unresolved; `[]` means none fits. Metadata lives in `provenance`.
- Automatic proposals are silver, not human benchmark gold. Human review records
  are separate from original machine judgments. Evaluation rejects unreviewed
  automatic labels. Training accepts them only with explicit `--allow-silver`.

## Configuration

Run `uv run wsd config init`. It preserves an existing file and creates
`~/.config/sense-model/config.toml` with owner-only access on Unix. Set
`jev.api_key` there. `SENSE_MODEL_CONFIG` or `--config` selects another file.
No credentials belong in Git, dataset artifacts, browser storage, or logs.

Paper acquisition is manual: put PDFs in `papers/` and list them in the JSON
manifest. Generation performs no web search or PDF downloads and needs no search
provider credentials. See `papers.example.json` for the editable template.

The shared Codex runner uses a temporary working directory, read-only sandbox,
ephemeral sessions, disabled execution/integration features, and structured JSON
output. It ignores user/project runtime customization while retaining CLI auth.
Verified with Codex CLI 0.169.0. See the [Codex configuration reference](https://developers.openai.com/codex/config-reference/).

## Script 1: collection and generation

`python scripts/generate_dataset.py --help` (also `wsd generate`).

1. Read and validate every local path and language in the supplied manifest.
2. Extract PDF text with page boundaries, preserve source metadata, reject
   unreadable/wrong-language material, and detect duplicate content.
3. Find exact dictionary occurrences; ask Codex to select the requested difficulty
   mix from these authentic passages. Validate every returned ID and source span.
4. Save accepted examples and per-paper checkpoints. Record failures and continue
   with later listed papers. Any failure makes the run incomplete; no substitutions.
5. Preserve input/inventory/model/prompt hashes for reproducibility and resume.

Artifacts: `samples.jsonl`, `papers.jsonl`, `manifest.json`,
`generation-report.json`, `events.jsonl`, and a local extraction cache.
Paper licenses are recorded for audit; availability does not grant redistribution
rights. Verify source and inventory rights before publishing excerpts.

## Script 2: automatic labeling

`python scripts/label_dataset.py --help` (also `wsd label`).

1. Validate input and configuration; refuse to overwrite the input dataset.
2. Ask Jev for all examples. Preserve its reported confidence separately from
   candidate probabilities. Threshold defaults to 0.80 and is provisional.
3. Keep valid high-confidence Jev decisions. Escalate low/missing confidence and
   exhausted request failures to Codex. A confident insufficient-context answer
   remains unresolved rather than becoming a false none-fits label.
4. Codex sees only context, target, and candidates, not Jev's answer. Validate its
   constrained selection or abstention. Preserve both judgments and disagreement.
5. Checkpoint each response; cache fingerprints include prompt, model, input and
   candidate inventory. Threshold-only changes reuse provider responses.
6. Write silver proposals, structured audit records, and progress/summary reports.

Artifacts: `samples.jsonl`, `judgments.jsonl`, `manifest.json`,
`labeling-report.json`, `events.jsonl`. Reports expose routing, failures,
confidence distribution and available usage/latency. Bounded retries and independent
provider concurrency limits keep failures and spend trackable.

## Web audit and reviewed export

`wsd annotate --import-file PATH/TO/labeled/samples.jsonl --open`.

Dark cards expose context, all meanings, Jev confidence, Codex result/rationale,
source metadata, automatic proposal and review status. Filters distinguish high
confidence, low-confidence agreement/disagreement, Jev errors, unresolved cases,
language, difficulty, paper and area. Inspect a random high-confidence sample as
well as disagreements to reveal confident errors.

One reviewer can approve, correct, mark multiple acceptable senses, choose none
fits, flag insufficient context, or reject an example. Revision history preserves
machine proposals and earlier human actions. Reviewed exports retain the input
schema and include only usable accepted examples by default; an all-status export
preserves the complete audit collection. The API key field explicitly saves to
the local config when requested; stored secrets are never returned to the browser.

## Verification and research follow-up

Offline tests cover exact spans, quotas, duplicates, routing boundaries, missing
confidence, errors, retry/resume, cache invalidation, credential redaction, stale
reviews, exports and benchmark label guards. Browser tests cover the real local
server/store, card filtering, correction and export with provider calls mocked.

A Codex structured-output smoke is integration evidence, not WSD accuracy.
Live Jev runs require the user's key. Authored test fixtures are not
paper-derived benchmark data. Separate train/development/test by paper, and keep
threshold tuning on development data. The paper-only collection measures research
reading; fiction/general-prose coverage is a later source adapter.
