# Implementation status

The new independent repository is `/home/andrepaim/src/sense-model`. The original
fork remains intact at `../word-sense-disambiguation`, source commit
`d1f7e107f2c072e876f031fdd5935bf496a0ef1b`.

Implemented: candidate-selection library; exact Unicode spans; explicit rejection
and abstention; local prepared-data format and leakage checks; English Raganato
and gloss-corpus adapters; Portuguese lexicon audit and context preparation;
portable supervised/multi-gold training and offline ID-aligned distillation;
checkpoint resume; evaluation/calibration/bootstrap/permutation/latency tooling;
ONNX FP32/INT8 export; actual Node tokenizer/runtime example; explicit release
validation and Hub upload with pinned redownload; locked environments and CI.

The original service, dictionary HTTP API, entity linking, spaCy, Cloud Run,
Slurm/Enroot deployment and server HTML are not imported into the new project.
The upstream fork remains the comparison reference.

## Measured evidence

See `export-results.md` for actual English checkpoint ONNX and Node parity.
See `evidence/upstream-parity.json` for original-code parity and
`portuguese-smoke.md` for actual mmBERT CPU backward/update/resume checks.
Tiny-model tests exercise training/save/reload/resume without expensive downloads.
Authored smoke examples demonstrate software paths and must not be used as
published performance claims. Actual local artifacts are ignored by Git.

## Research and release work still required

- Acquire and review licensed real-text training data and Portuguese definitions.
- Obtain independent human labels and adjudicate frozen English/Portuguese test
  and calibration sets. Proposed held-out target is 1000 per language.
- Run controlled English improvement experiments; publish only supported results.
- Train and evaluate a Portuguese WSD checkpoint on representative reviewed data.
- Calibrate each shipped float/quantized artifact; verify product release gates.
- Measure latency and memory on an ordinary Windows laptop (this environment is
  Linux CPU only).
- Authenticate Hugging Face and choose a model/dataset namespace before upload.
  No Hugging Face token was available in this session. Do not store tokens here.

These are external data, compute, annotation and release deliverables. Having
commands to run them does not mean the resulting accuracy or performance has
been established. The current project is a tested development foundation, not a
claim that improved English or production Portuguese weights already exist.

## Annotation pilot tooling

The optional `wsd annotate` local web interface and `wsd pilot collect/audit`
commands are implemented. See `pilot-plan.md` and `pilot-guide.md`. Independent
votes, Jev assistance/key entry, two-reviewer adjudication and canonical exports
are tested. This adds the human review workflow; it does not supply 100 real
reviewed examples per language or change the research prerequisites above.

## Automatic dataset pipeline

`wsd generate` / `scripts/generate_dataset.py` read a JSON manifest of local papers, request a configurable number of samples
per paper (default 10), and use
local Codex to choose dictionary-backed examples. `wsd label` /
`scripts/label_dataset.py` run Jev first and escalate low-confidence cases to
independent Codex judgment. The local annotation interface supports automatic
proposals, dark cards, single-human review and compatible reviewed exports.
See `automatic-pilot-plan.md` for the current workflow and limits. This tooling
does not itself establish a 1,000-example reviewed dataset in either language.
