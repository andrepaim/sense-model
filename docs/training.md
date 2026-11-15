# Training and distillation

Use `wsd train --config configs/english_baseline.json`. Configuration paths are relative to the working directory; supply your prepared files first. `--model` loads a local checkpoint or explicitly named Hub model. Pin `--revision` for reproducibility. CLI options override JSON configuration. All dataset access is local. CPU is the default; use `--device cuda --bf16` on a compatible GPU. W&B is opt-in via `--report-to wandb`; metrics otherwise stay local.

Candidate order is shuffled once deterministically per run. Training uses the same prompt construction and valid candidate/rejection slots as inference. The loss marginalizes all acceptable gold definitions; none-fits labels supervise the rejection slot. Optional label smoothing is applied only over valid slots. Development accuracy likewise considers valid slots. Best checkpoint selection uses development accuracy, and early stopping uses `--patience` evaluation events. `--save-steps` controls saving/evaluation; use 1 for tiny smoke runs. Change one variable per experiment, repeat seeds for finalists, and keep test held out.

```sh
wsd train --config configs/portuguese_pilot.json --max-steps 10 --save-steps 5
wsd train --config configs/portuguese_pilot.json --resume runs/portuguese-pilot/checkpoints/checkpoint-5
```

Resuming restores Trainer optimizer, scheduler, RNG and global-step state. Keep the same data/config/model settings when resuming. `resolved_config.json`, `environment.json`, `data_manifest.json`, `metrics.jsonl`, checkpoint metadata and the final portable model are saved under each run. Use separate output directories for separate experiments. The final checkpoint is uncalibrated; calibrate with held-out calibration data using the evaluation tooling before deployment.

Distill by generating teacher scores offline:

```sh
python scripts/build_teacher_scores.py --model sign/Ettin-1B-WSD --input data/english/train.jsonl --output data/english/teacher.jsonl --temperature 2 --device cuda
wsd train --config configs/english_improved.json --teacher-scores data/english/teacher.jsonl
```

The cache includes example fingerprints, teacher revision, temperature, and candidate-ID probabilities. Students tokenize independently, so teacher and student vocabularies may differ. A cache mismatch is an error, not a fallback. KL loss is computed at the cache temperature, multiplied by temperature squared. Labeled teacher rows mix gold and KL losses; unlabeled rows use KL alone. Do not use an English teacher to fabricate Portuguese labels without language-specific validation.

The English recipe starts from the published WSD checkpoint. The Portuguese pilot starts from mmBERT-small with a new language-specific prompt and answer mapping; it does not claim Portuguese quality until evaluated on natural, reviewed data. Current configs are portable starting recipes, not a claim of reproduced upstream training or completed language training. Large runs, dataset acquisition, and annotation are intentionally explicit operations.
