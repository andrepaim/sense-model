# Automatic-first review workspace

Start the local workspace with `wsd annotate --workspace .annotation/workspace.sqlite --import-file path/to/labeled.jsonl` (or use the existing annotation server entry point). Import JSONL through the browser to add further batches. The web UI and SQLite workspace run on the local workstation; reviewers are workflow names, not authentication accounts. Optional Jev assistance sends context and definitions to the configured provider. The labeling pipeline likewise uses hosted Jev and Codex inference.

Rows containing `provenance.automatic` open in automatic audit mode. Jev's original outcome, candidate probabilities, confidence, escalation reason, and Codex's outcome and rationale are immediately visible. The automatic evidence is immutable in the workspace. Incoming proposed `gold_ids` are preserved as `provenance.proposed_gold_ids`; they do not become human gold during import. Reimporting an identical batch is idempotent; changing content under an existing example ID is rejected atomically. Importing previously reviewed data archives its review as `previous_review` and starts a fresh workspace review.

One named reviewer can:

- Approve the automatic label, using the result selected by the labeling pipeline.
- Save a correction with one or more candidate IDs, `none_fits`, or `insufficient_context`.
- Reject the example and record a note.

Approval is disabled for unresolved proposals. Notes are preserved with reviewer identity and timestamps. Revisions use optimistic concurrency; a stale browser tab must reload before saving. Every review change is recorded in the workspace event history, while the current review is shown to any local reviewer.

Unresolved examples and unreviewed Jev/Codex disagreements appear first. **Audit random high confidence** opens a random unreviewed high-confidence Jev example within the language, difficulty, area and paper filters.

Filters cover automatic bucket, reviewed/unreviewed, language, generation difficulty, area, paper, and original Jev confidence relative to the stored threshold. Buckets distinguish high-confidence Jev, low-confidence Codex agreement/disagreement, Jev-error Codex fallback, and unresolved results.

The dashboard reports observed **Jev** accuracy by original confidence band against comparable human labels. It compares Jev's original IDs/outcome, even when the pipeline selected Codex. Missing Jev results, rejections, and insufficient context do not enter the accuracy denominator. Reviewed, comparable, and correct counts are displayed separately. These are observations on the reviewed sample, subject to review-selection bias, not an independent benchmark or calibrated probability guarantee. Demo records are excluded.

## Exports

**Export human-approved gold** emits schema-compatible Example JSONL plus a manifest containing an exact JSONL SHA-256. Only `selected` and `none_fits` human reviews with status `approved` or `corrected` enter this export. Multiple acceptable IDs and empty `gold_ids` for `none_fits` are supported. Original automatic evidence remains at `provenance.automatic`; the human audit is under `provenance.review`, and `label_tier` becomes `gold`.

**Export all with status** emits every non-demo example, including unresolved, rejected, insufficient-context and unreviewed records. Only accepted human labels receive `gold_ids`; other records have `gold_ids: null`. Review status and automatic proposals remain available for analysis and recovery. Download the workspace backup for complete edit/event history.

Legacy examples without `provenance.automatic` retain the independent first-vote, two-reviewer adjudication workflow and disclosure guards. They do not acquire gold through the automatic review API.

## Jev credentials

Settings keep a submitted API key in server session memory by default. Explicitly check **Save key to local config for future runs** to save it to `~/.config/sense-model/config.toml` (or `SENSE_MODEL_CONFIG`). The file is created with owner-only permissions. Settings read the saved provider/model and whether a key is configured, and never return the saved key to the browser. Keys are excluded from annotation data, backups, and exports. Clearing the key also removes it from the local config. No provider request is made when settings are saved.

## Offline checks

Run `.venv/bin/python -m pytest tests/test_annotation_store.py tests/test_annotation_server.py` and `npm test --prefix web-tests`. Browser checks use the real local HTTP server/store with a mocked paid provider; no provider requests or model downloads are needed. The automatic scenario covers visible evidence, filters, single-review approval, multi-ID correction, unresolved/rejected outcomes, exact exports, credential settings, and a mobile-width overflow check. The legacy browser scenario continues to exercise the previous workflow.
