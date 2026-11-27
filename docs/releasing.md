# Export and release

The model runtime is independent of dictionaries, spaCy, web services and pdfpal. Artifacts use an explicit candidate-selection contract. Downloading a model does not constitute a publication.

## Export

Install with `uv sync --extra export`. Export using representative examples, including multiple sequence lengths, candidate counts, target positions and each supported language:

```sh
uv run wsd export --model runs/english/final --fixtures data/fixtures/smoke.jsonl --output artifacts/english --quantize
```

The destination must be empty. The command saves the model and tokenizer, exports compact logits with dynamic batch and sequence dimensions, and checks float ONNX outputs against PyTorch both in batches and one example at a time. Failed parity removes the ONNX directory. It writes `parity.json` containing the exact examples, tensor inputs, reference logits, reference decisions and error tolerances. Inputs are `input_ids`, `attention_mask` and `prediction_positions`; output is `[batch,128]` logits. The first N valid slots correspond to candidates in input order, and slot 127 is rejection. Mask unsupported slots before normalization.

`--quantize` writes `onnx/model.int8.onnx` using dynamic MatMul quantization. The recipe options: `--per-channel` (INT8 weights per output channel), `--float16-embedding` (the token embedding stored in float16 with a Cast to float), and `--outlier-examples FILE ...`, which measures the input of every Linear layer on the first `--outlier-count` (200) distinct prompts by SHA-256 and keeps in float the feed-forward output MatMuls (`mlp.Wo`) whose input peaks above `--outlier-ratio` (15) times its 99.9th percentile (`activation-stats.jsonl`, `recipe.json`). `--check-examples FILE ...` scores the first `--check-count` (1,000) prompts with both graphs on the CPU and writes `int8-check.json` (same top choice, same top-three set, centred logit errors). `--max-accuracy-loss` sets the fixture budget. Its report includes decision agreement and labelled accuracy drift. More than one percentage point accuracy loss on fixtures rejects the quantized artifact. Fixtures are a mechanical smoke test, **not release-quality evidence**. Run the full held-out evaluation on the exact exported graph, report selective precision/coverage and quantization loss, and measure Windows CPU latency before claiming deployment readiness. The caller must distinguish `none_fits` from `uncertain`.

Node example (Node 22+):

```sh
cd examples/node
npm ci
node predict.mjs ../../artifacts/english ../../data/fixtures/smoke.jsonl
WSD_ARTIFACT=../../artifacts/english npm test
```

The example uses the actual saved Hugging Face tokenizer and ONNX Runtime Node, with remote loading disabled. `codePointTarget` converts JS/DOM UTF-16 offsets to exclusive-end code-point offsets and rejects a selection splitting an emoji surrogate pair. The parity test compares exported expected candidate probabilities and decisions. Python and Node consume `model_metadata.json` and `calibration.json` from the same artifact. It is an example integration, not another server or npm service.

## Model artifact

A release needs model/tokenizer files, `model_metadata.json` (template, answer symbols, maximum input length and model revision), standalone `calibration.json`, `training_config.json`, `data_manifest.json`, `evaluation.json`, and README/LICENSE/NOTICE. The README must contain YAML metadata with `license` and `base_model`. Document language and domain scope, model ancestry and exact revisions, data preparation and licensing, training command, frozen evaluation split fingerprints, calibration method, CPU measurements, and failure cases. Report illustrative fixtures as fixtures. Do not claim a Portuguese model from English inference on Portuguese text.

Model weights are saved as safetensors. Custom decoder code is supplied by this Python package: use `from wsd import Predictor` (or `from wsd.predict import Predictor`) and `Predictor.from_pretrained('namespace/model', revision='COMMIT')`. Plain stock `AutoModelForMaskedLM` is not a supported loader for the compact head. Package installation avoids executing arbitrary Hub remote code. ONNX consumers require no Python model code.

Preserve the original MIT notice for adapted code. The upstream English checkpoint declares Apache-2.0, while the proposed Portuguese mmBERT base declares MIT. These obligations are separate from the code license and source dataset conditions. Include proper model attribution and applicable license text; record exact source revisions and license review. Publish datasets separately only when redistribution is permitted. For restricted datasets publish preparation recipes and checksums instead.

## Assembling a repository

`wsd publish --artifact DIR --assemble release.json --export EXPORT [--training-config F] [--data-manifest F] [--evaluation NAME=FILE ...]` builds the repository layout in an empty `DIR` before the dry run. The release file names, relative to itself, the model card, `LICENSE` and `NOTICE`; lists the `files` copied from the export and `release_files` copied from its own folder in their place; copies the export's `recipe.json` to `onnx/recipe.json` with its `model` field replaced; and updates the export's `model_metadata.json` with its `metadata` (`"model_revision": "weights"` writes `sha256:` and the SHA-256 of `model.safetensors`), adding an `export` block made from the recipe. `releases/Ettin-150m-WSD-wiktionary/` is the release of the [Wiktionary round](wiktionary-round1.md).

## Publication

Validation and local subprocess reload happen before upload:

```sh
uv run wsd publish --artifact artifacts/english
# Only this explicit flag performs remote changes:
uv run wsd publish --artifact artifacts/english --repo-id YOUR_NAMESPACE/sense-en --upload
```

The dry run creates `checksums.json`, validates release completeness and the compact mapping, then reloads locally with Hub networking disabled from a temporary working directory. Changing a checksummed artifact causes validation to fail; after reviewing the changes explicitly remove the stale checksum file to regenerate it. Publishing uses the authenticated Hugging Face Hub client and never stores the token in config files. `--private` selects a private newly created repository; it does not change an existing repository's privacy. `--revision` targets an existing branch (default main).

Upload returns an immutable commit ID. The command downloads that exact commit into a fresh temporary directory, verifies all checksums and reloads it. A network verification failure after upload is a failed verification, not a rollback; use the returned Hub revision to investigate. Publication is independent of training, and no training command implicitly uploads a model. Publishing a baseline must attribute it as a repackaged upstream checkpoint; improvements and Portuguese support need measured evidence.
