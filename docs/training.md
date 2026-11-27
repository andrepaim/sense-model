# Training and distillation

Use `wsd train --config configs/english_baseline.json`. Configuration paths are relative to the working directory; supply your prepared files first. `--model` loads a local checkpoint or explicitly named Hub model. Pin `--revision` for reproducibility. CLI options override JSON configuration. All dataset access is local. CPU is the default; use `--device cuda --bf16` on a compatible GPU. W&B is opt-in via `--report-to wandb`; metrics otherwise stay local.

Candidate order is shuffled once deterministically per run (`--candidate-order given` keeps each example's order, for applications that always present candidates in dictionary order). Training uses the same prompt construction and valid candidate/rejection slots as inference. The loss marginalizes all acceptable gold definitions; none-fits labels supervise the rejection slot. Optional label smoothing is applied only over valid slots. Development accuracy likewise considers valid slots. Best checkpoint selection uses development accuracy, and early stopping uses `--patience` evaluation events. `--save-steps` controls saving/evaluation; use 1 for tiny smoke runs. Change one variable per experiment, repeat seeds for finalists, and keep test held out.

`--grad-accum N` divides each micro-batch's mean loss by N, so the logged loss and the gradient equal those of one batch of `--batch-size` times N. Gradients are clipped to norm 1.0 (the Trainer default) and the optimizer is AdamW (the Trainer default). Before this was fixed, runs with `--grad-accum` above 1, such as `configs/english_baseline.json` (8), logged losses and gradient norms N times too large. AdamW's update is largely insensitive to a constant gradient scale, so the main effect on the weights came through clipping: the inflated norm crossed 1.0 on steps whose true norm did not, and those steps were scaled down relative to the others. Such runs are not comparable to runs made after the fix.

```sh
wsd train --config configs/portuguese_pilot.json --max-steps 10 --save-steps 5
wsd train --config configs/portuguese_pilot.json --resume runs/portuguese-pilot/checkpoints/checkpoint-5
```

Resuming restores Trainer optimizer, scheduler, RNG and global-step state. Keep the same data/config/model settings when resuming. `resolved_config.json`, `environment.json`, `data_manifest.json`, `metrics.jsonl`, checkpoint metadata and the final portable model are saved under each run. Use separate output directories for separate experiments. The final checkpoint is uncalibrated; calibrate with held-out calibration data using the evaluation tooling before deployment.

## Several files, development forms and batches by length

`--train` takes several files, read as one split: their examples are concatenated in the order given and shuffled once with the seed (example IDs must be unique across files). One file keeps its order, as before, unless `--shuffle-examples` asks for the same shuffle (the Wiktionary round's configurations set it). Unlabelled rows (`gold_ids` null) need teacher scores. `--drop-labels` trains teacher-only on labelled files: the cache's fingerprints are checked against the examples as stored, then their labels are dropped. `--data-dir DIR` makes relative data paths (train, dev, extra dev, calibration, teacher scores) refer to `DIR`, for example a dataset downloaded with `huggingface_hub.snapshot_download(..., repo_type="dataset", revision=...)`, and `data_manifest.json` then names the files relative to it.

`--dev` chooses the best checkpoint and stops early. `--extra-dev NAME=PATH` evaluates other forms of the development data alongside it (in a configuration, `"extra_dev": {"NAME": "PATH"}`); metrics are then logged as `eval_<name>_accuracy` with `--dev-name` naming the first. `--dev-limit N` evaluates N examples of each form, drawn with the seed. Development metrics are `accuracy` (the top choice among candidates and none-fits is a gold one), `top3_accuracy` (a gold candidate among the three best offered candidates, over rows whose gold is offered) and `none_rate`. `--evaluate-before-training` evaluates the starting model first (`baseline.json`, step-0 rows in `metrics.jsonl`). `--eval-batch-size` and `--logging-steps` set the development batch and the logging interval. The logged loss comes with its parts, `label_ce` (mean label cross-entropy over labelled rows) and `teacher_kl` (mean T² × KL over rows with teacher scores).

`--group-by-length` batches examples of similar length (transformers' length-grouped mega-batches; the development sampler follows the same strategy) in an order drawn from a generator seeded by the seed and the epoch. transformers' own sampler draws from torch's global random state, so a resumed run trained on a new batch order whenever that state differed from the fresh run's at the start of the epoch (for example after `--evaluate-before-training`); with this sampler a run resumed mid-epoch replays the epoch's batches. `--resume latest` resumes from the newest checkpoint that has every file a resume needs (a run stopped while saving leaves one without).

The model trains on float32 weights, also when the checkpoint is stored in bfloat16; `--bf16` is autocast. Split validation runs over all files of each split; `--allow-split-overlap` records cross-split overlap in `data_manifest.json` (`leakage: failed`, the count and the first 50 findings) instead of failing, for data whose leak checks were made when it was built: say so wherever the run is reported. `--report-to tensorboard` writes TensorBoard events under the run's `checkpoints/runs/` (install the `tensorboard` extra).

The configurations `configs/wiktionary_round1_*.json` are the three runs of the [Wiktionary round](wiktionary-round1.md).

## What changed for existing configurations

A configuration that uses none of the options above trains as before the Wiktionary round's options, with these exceptions:

- **Float32 weights for a bfloat16 checkpoint.** A base stored in bfloat16 (`sign/Ettin-150m-WSD` is) was trained with bfloat16 weights and optimizer state; it now trains on float32 weights (`--bf16` is autocast over them). Its `model.safetensors` is about twice the size (598,827,368 bytes against 299,420,880 for the bfloat16 base) and its SHA-256, hence `model_revision`, differs from what the old behaviour gave. A float32 base is unaffected.
- **Saved tokenizers** have no `padding` block in `tokenizer.json` (encoding with padding left it there); token ids are unchanged.
- **Logs and metrics.** Loss rows also carry `label_ce` and `teacher_kl`; development metrics add `top3_accuracy` and `none_rate`, and `accuracy` is computed in float64. Best-checkpoint selection still uses `accuracy`.
- **Resume.** A run recorded before these options resumes: a missing setting counts as its default and a `train` path recorded as a string equals a list of it. `--resume latest` with no complete checkpoint starts from the beginning, says so, and writes a `restart` row in `metrics.jsonl`.
- **Teacher scores** (`wsd teacher`): prompts are batched by length instead of in file order, which changes float32 logits only in the last digits and bfloat16 ones more (a check on 2026-11-27 found 2,984 of 3,000 top choices the same); the checkpoint's stored dtype is kept unless `--dtype` is given; a prompt over `max_length` still raises unless `--skip-too-long`; cache probabilities are computed in Python floats; `main()` now returns a report (`rows`, `skipped`, `too_long`, `scored`), which it also prints.
- **Paired bootstrap** intervals of `wsd evaluate` use the resampling of `wsd rank` (docs/evaluation.md), so they differ slightly for the same inputs.

## Distillation

Distill by generating teacher scores offline:

```sh
wsd teacher --model sign/Ettin-1B-WSD --revision COMMIT --input data/english/train.jsonl --output data/english/teacher.jsonl --temperature 2 --device cuda --dtype bfloat16 --max-tokens 8192 --batch-size 64 --scores data/english/teacher-logits.jsonl
wsd train --config configs/english_improved.json --teacher-scores data/english/teacher.jsonl
```

The teacher scores each distinct prompt once (`wsd.scoring`: prompts sorted by length, batches of at most `--batch-size` prompts and, with `--max-tokens`, at most that many padded tokens; a batch that runs out of GPU memory is split). `--scores` keeps the raw logits by prompt SHA-256 in a file that is appended to and resumable; prompts already in it are not scored again, so a cache at another temperature, or for more examples, reuses them. `scripts/build_teacher_scores.py` is the same command. Score rows record the scorer's identity, and a `--scores` file of another model, revision, dtype or `max_length` is refused (`docs/evaluation.md`); `--output` must differ from `--scores` and the inputs. The cache includes example fingerprints, teacher revision, temperature, and candidate-ID probabilities. Students tokenize independently, so teacher and student vocabularies may differ. A cache mismatch is an error, not a fallback. KL loss is computed at the cache temperature, multiplied by temperature squared. Labeled teacher rows mix gold and KL losses; unlabeled rows use KL alone. Do not use an English teacher to fabricate Portuguese labels without language-specific validation.

The English recipe starts from the published WSD checkpoint. The Portuguese pilot starts from mmBERT-small with a new language-specific prompt and answer mapping; it does not claim Portuguese quality until evaluated on natural, reviewed data. Current configs are portable starting recipes, not a claim of reproduced upstream training or completed language training. Large runs, dataset acquisition, and annotation are intentionally explicit operations.
