# Evaluation

Evaluation consumes prepared examples with `gold_ids`. An empty list means no offered definition fits; `null` is unlabeled and is rejected. Candidate IDs may come from any dictionary. Human disagreement can be represented by multiple acceptable IDs.

```sh
wsd evaluate --model ./runs/english/final --local-files-only \
  --data data/test.jsonl --output results/english-2026-11-15 \
  --latency-repeats 3 --permutations 3
```

The model is loaded once. Prediction batches use `--batch-size` (default 16). Optional latency measurements then run each target separately, without batching, and report p50/p95 milliseconds and process peak RSS when supported. `cold_load_ms` includes loading the checkpoint. Peak RSS is the whole process high-water mark, not incremental model memory, and is unavailable through the standard library on Windows. Record actual device and thread settings alongside results. No throughput number should be presented as desktop selection latency.

Each run writes `evaluation.json` and `predictions.jsonl`. Saved predictions can be evaluated offline with `--predictions path.jsonl` in place of `--model`. Results contain the UTC date, dataset checksum, requested model revision, and calibration parameters. Pin a model revision when evaluating a Hub checkpoint.

## Measures

Top-1 accuracy always uses the highest probability across all valid candidates and the none-fits choice, before confidence thresholds. The first-candidate baseline always chooses the first offered candidate, or none for an empty inventory. Reports separate ambiguous in-inventory cases from one-sense cases. The baseline is useful only if candidate order represents dictionary frequency; preserve that order in test data.

Marked precision and coverage describe actual `selected` decisions. False-confident-none rate is the fraction of absent-meaning examples that receive a marked candidate. None-fits recall requires an explicit `none_fits` decision; `uncertain` is not counted as a successful rejection. ECE uses ten equal-width confidence bins, and the risk-coverage curve orders all raw decisions by confidence. Breakdowns include language, domain, candidate count, and target phrase length.

Paired bootstrap confidence intervals resample model/baseline outcomes on the same examples, deterministically. They use example-level resampling, which can understate uncertainty when many examples share a document; report that limitation and use a document bootstrap in a publication requiring clustered inference. `--permutations` measures changes in final decisions when candidate order is shuffled with a fixed seed.

## Score files and ranks within dictionary entries

`wsd score --model M --input A.jsonl [B.jsonl ...] --output scores.jsonl` writes one row per distinct prompt: its SHA-256 (`hash`), the logits of the offered candidates in order and then none-fits, the token count, and `type` `scored` or `rejected` (`TOO_LONG` over the model's `max_length`). It batches like `wsd teacher` (`--max-tokens`, `--batch-size`, `--dtype`, default float32, `--device`), appends, and resumes. Every row records the scorer's identity: `model` as given, `revision` (the model's `model_revision`), `dtype` and `max_length`; resuming into a file scored with another identity, or into a file that is not a score file, is refused, and `--output` may not be an input. Rows without an identity (score files written before it was recorded, such as those of the Wiktionary round) are accepted only with `--untagged-scores MODEL REVISION DTYPE MAX_LENGTH`, which states what they are and must match the model being run. `wsd rank` reads score files without these checks: it compares the files it is given. It refuses a score whose logits do not match its item's candidates or are not finite, and an empty set of items. An application that builds the same compact-v1 prompts can join the rows with its own records by that hash.

`wsd rank` measures what an application that lists a word's meanings in dictionary order, grouped by entry, shows when it reorders the meanings it offered the model: the rank of the labelled meaning within its entry. A skipped, missing or rejected prompt, or a none-fits probability at least that of every offered meaning, leaves dictionary order; a labelled meaning the prompt did not offer comes after the offered meanings of its entry. It reads item files (one JSON object per labelled item: `id`, `labelled` `{entry, sense}`, `entrySenses`, `meanings`, and `perCap` `{form: {prompt, candidates, labelledOffered, skip}}`, the files contextual-dictionary writes) and score files, and reports for each form and model first, top three, mean rank, the outcome counts, the trainer's `offered` accuracy, strata by meanings in the entry, and with `--baseline NAME` paired bootstrap intervals of "first" against that model:

```sh
wsd rank --items items.jsonl --forms '24|c1600' 127 --exclude excluded-ids.txt \
  --scores base=base.jsonl tuned=tuned.jsonl --baseline base --output rank.json
```

The paired bootstrap of `wsd rank` and `wsd evaluate` resamples items with `random.Random(seed).randrange` (2,000 resamples; `wsd rank` uses seed 1, `wsd evaluate` seed 42) and takes the 2.5th and 97.5th percentiles at indices int(0.025 R) and int(0.975 R) − 1. This is the procedure of the Wiktionary round's published intervals; `wsd evaluate`'s intervals made before it used `random.choices` and differ slightly.

## Calibration

Fit on a dedicated split, never the test set:

```sh
wsd evaluate --model ./runs/english/final --local-files-only \
  --data data/test.jsonl --calibration-data data/calibration.jsonl \
  --calibration-output runs/english/final/calibration.json \
  --target-precision .95 --output results/calibrated-english
```

This checks split leakage before fitting. Temperature is selected from a documented grid by calibration-set negative log likelihood, counting total probability of all acceptable answers. A second grid maximizes marked coverage subject to the requested empirical precision. The fitted file records whether any setting was feasible. An infeasible fit uses full abstention. These are empirical tuning results, not a guarantee of test precision; test metrics determine whether the release gate passes.

Existing artifact calibration is reset before refitting. Use `--calibration file.json` to apply a saved calibration to a model or raw predictions. Do not apply it twice to already calibrated saved probabilities. Candidate and none-fits decisions must both pass probability and top-versus-runner-up margin thresholds; otherwise the decision is `uncertain`.

## Release gates

Set thresholds before comparing releases. Initial proposed product gates are: five percentage points above the first-candidate baseline on ambiguous examples, paired improvement interval above zero, marked precision at least 95% at ambiguous in-inventory coverage at least 50%, and at most 5% confident selections on absent meanings. Quantized artifacts must still pass those gates and lose at most one percentage point of top-1 accuracy. Warm p95 must be below one second on the named Windows CPU workload.

Fixtures in this repository validate implementation, not model quality. A release needs reviewed English and Portuguese development, calibration, and frozen test sets with adequate independent source documents. Public benchmark scores must be reproduced separately, with dictionary mappings and unmatched gold senses reported. No quality claims should be made from a tiny training smoke test.

## Export and acceptance checks

Use `--backend onnx` or `--backend onnx-int8` with the same `--model` artifact directory to evaluate the exported graph on exactly the same test records. The default backend is PyTorch. CPU PyTorch thread count is explicit (`--threads 4` by default), preventing a large server's default thread pool from dominating single-selection latency. ONNX Runtime has its own thread pool; record its settings from the export/runtime setup.

Every report includes `release_gates` with individual pass/fail/unknown checks. Add `--require-gates` for a nonzero exit status unless all proposed accuracy, abstention, and latency checks pass. Missing absent-meaning cases, ambiguous cases, or latency evidence yield unknown checks and fail this mode. Hardware certification, annotation quality, and quantization comparison still require review; an automated metrics gate cannot establish that the test set is representative.

## Original implementation parity

`python scripts/check_upstream_parity.py --upstream ../word-sense-disambiguation --model sign/Ettin-150m-WSD --output docs/evidence/upstream-parity.json` compares exact prompt strings, normalized valid-choice probabilities, and selected IDs against the original code in a separate subprocess. It reads a cached checkpoint only, hides CUDA, uses four CPU threads, and requires the original low-level inference dependencies (including `requests`, used by the upstream module imports even though no dictionary call is made). The original checkout remains untouched. The fixture is illustrative and demonstrates extraction compatibility; it is not an English quality benchmark.
