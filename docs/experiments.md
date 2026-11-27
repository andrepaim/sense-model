# Improvement protocol

Rounds so far: [Wiktionary round 1](wiktionary-round1.md) (English, `sign/Ettin-150m-WSD` fine-tuned on English
Wiktionary with `sign/Ettin-1B-WSD` as teacher; published as `andrepaim/Ettin-150m-WSD-wiktionary`).

Preserve English checkpoint 8751b577199d1bb95b74fa2457da7065d57100ae as the
reference. Keep fixed development, calibration and frozen test inventories.
The authored six-record fixture is not a quality benchmark.

Run one ablation at a time: realistic same-POS rejection negatives; licensed
technical contexts; candidate shuffling; calibration; stronger teacher scores;
smaller compatible students. The standalone valid-slot/multi-gold training loss
is itself a changed recipe relative to upstream: measure it as an ablation, not
as an exact training reproduction. Changing prompt language/template requires a
new checkpoint and matching metadata. Do not overwrite an existing checkpoint's
format. Portuguese starts from multilingual mmBERT-small; compare separate pt-BR
and pt-PT sets, and report Portuguese definition coverage independently.

A run receipt records hypothesis, code commit, base revision, dataset checksums,
configuration, seeds, metrics and keep/reject decision. Use at least three seeds
for finalists and paired bootstrap comparisons. Select with development only;
fit thresholds on calibration only; evaluate frozen test once for release.

Suggested annotation target: >=1000 held-out English and >=1000 Portuguese
instances, in addition to development/calibration. Include general and technical
contexts, multiword targets, no-fitting-sense examples, and single-candidate
sanity cases. Keep variants/documents together; adjudicate multiple acceptable
senses. A human-reviewed set is a remaining research deliverable, not generated
by the repository's smoke tests.

Pre-register gate thresholds from docs/evaluation.md before running comparisons.
A candidate that misses its gate remains an experimental artifact. An int8
artifact must be evaluated over the same full test data as its float parent.

# Future bilingual experiment

Train mixed English/Portuguese examples using the multilingual base only after
separate releases have credible baselines. Keep language sampling explicit;
report both languages independently. Do not merge incompatible tokenizer/head
weights or claim multilingual support from a handful of translations.
