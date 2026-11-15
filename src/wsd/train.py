"""Supervised fine-tuning and tokenizer-independent offline distillation."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from wsd.data import file_sha256, read_jsonl, validate_splits
from wsd.predict import Predictor, validate_metadata
from wsd.tokenizer import encode_examples


def read_teacher(path, examples, temperature):
    if path is None:
        return {}
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"Duplicate teacher row {row['id']}")
        if abs(row.get("temperature", 1.0) - temperature) > 1e-6:
            raise ValueError("Teacher cache temperature must equal distillation temperature")
        rows[row["id"]] = row
    for ex in examples:
        if ex.id not in rows:
            if ex.gold_ids is None:
                raise ValueError(f"Unlabeled example {ex.id} has no teacher scores")
            continue
        row = rows[ex.id]
        if row.get("example_sha256") != example_fingerprint(ex):
            raise ValueError(f"Teacher example fingerprint mismatch: {ex.id}")
        expected = {c.id for c in ex.candidates}
        if set(row["probabilities"]) != expected:
            raise ValueError(f"Teacher candidate IDs mismatch: {ex.id}")
        probs = [*row["probabilities"].values(), row["none_probability"]]
        if any(not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probs) or abs(sum(probs) - 1) > 1e-4:
            raise ValueError(f"Invalid teacher distribution: {ex.id}")
    return rows


def example_fingerprint(example):
    import hashlib

    # Candidate order is a property of presentation, not example identity.
    data = example.to_dict()
    data["candidates"] = sorted(data["candidates"], key=lambda c: c["id"])
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class PreparedDataset(Dataset):
    def __init__(self, examples, predictor, seed=42, shuffle=False, teacher=None):
        self.rows = []
        rng = random.Random(seed)
        teacher = teacher or {}
        nslots = len(predictor.metadata["letters"])
        nota = predictor.metadata.get("nota_index", nslots - 1)
        for original in examples:
            if not original.candidates:
                raise ValueError(f"Training/evaluation example {original.id} has no candidates")
            ex = copy.deepcopy(original)
            if shuffle:
                rng.shuffle(ex.candidates)
            encoding = encode_examples([ex], predictor.tokenizer, predictor.metadata)
            gold = torch.zeros(nslots, dtype=torch.float32)
            valid = torch.zeros(nslots, dtype=torch.bool)
            valid[: len(ex.candidates)] = True
            valid[nota] = True
            if ex.gold_ids is not None:
                if not ex.gold_ids:
                    gold[nota] = 1
                else:
                    for i, candidate in enumerate(ex.candidates):
                        gold[i] = candidate.id in ex.gold_ids
            scores = torch.zeros(nslots, dtype=torch.float32)
            row = teacher.get(ex.id)
            if row:
                for i, candidate in enumerate(ex.candidates):
                    scores[i] = row["probabilities"][candidate.id]
                scores[nota] = row["none_probability"]
            elif ex.gold_ids is None:
                raise ValueError(f"Unlabeled example {ex.id} requires teacher scores")
            self.rows.append(
                {
                    **{k: v[0].cpu() for k, v in encoding.items()},
                    "labels": gold,
                    "valid_mask": valid,
                    "teacher_probs": scores,
                    "teacher_available": torch.tensor(row is not None),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(rows, pad_id):
    from torch.nn.utils.rnn import pad_sequence

    output = {}
    for key in rows[0]:
        tensors = [row[key] for row in rows]
        output[key] = (
            pad_sequence(tensors, batch_first=True, padding_value=pad_id if key == "input_ids" else 0)
            if key in ("input_ids", "attention_mask")
            else torch.stack(tensors)
        )
    return output


def choice_loss(
    logits, labels, valid_mask, teacher_probs=None, teacher_available=None, alpha=0.0, temperature=2.0, smoothing=0.0
):
    """Marginal likelihood over acceptable golds; mask absent answer slots."""
    logits = logits.float().masked_fill(~valid_mask, -1e9)
    log_probs = F.log_softmax(logits, dim=-1)
    labeled = labels.sum(-1) > 0
    gold_logprob = torch.logsumexp(log_probs.masked_fill(~labels.bool(), -1e9), dim=-1)
    ce = -gold_logprob
    if smoothing:
        uniform = -(log_probs.masked_fill(~valid_mask, 0).sum(-1) / valid_mask.sum(-1))
        ce = (1 - smoothing) * ce + smoothing * uniform
    teacher_available = torch.zeros_like(labeled) if teacher_available is None else teacher_available.bool()
    if teacher_probs is not None and teacher_available.any():
        student_log = F.log_softmax(logits / temperature, dim=-1)
        teacher_log = teacher_probs.clamp_min(1e-30).log()
        kl = (teacher_probs * (teacher_log - student_log)).sum(-1) * temperature**2
        loss = torch.where(labeled & teacher_available, (1 - alpha) * ce + alpha * kl, torch.where(labeled, ce, kl))
    else:
        loss = ce
    if not (labeled | teacher_available).all():
        raise ValueError("Unlabeled rows require a teacher distribution")
    return loss.mean(), logits


class ChoiceTrainer(Trainer):
    def __init__(self, *args, alpha=0, temperature=2, smoothing=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha, self.temperature, self.smoothing = alpha, temperature, smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels, valid = inputs.pop("labels"), inputs.pop("valid_mask")
        teacher, available = inputs.pop("teacher_probs"), inputs.pop("teacher_available")
        outputs = model(**inputs)
        loss, logits = choice_loss(
            outputs.logits, labels, valid, teacher, available, self.alpha, self.temperature, self.smoothing
        )
        return (loss, {"logits": logits}) if return_outputs else loss


def _metrics(pred):
    import numpy as np

    predictions, labels = pred
    chosen = np.argmax(predictions, axis=-1)
    labeled = labels.sum(-1) > 0
    return {"accuracy": float(labels[np.arange(len(labels))[labeled], chosen[labeled]].mean()) if labeled.any() else 0}


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args(argv)
    config = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--train")
    parser.add_argument("--dev")
    parser.add_argument("--calibration")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--teacher-scores")
    parser.add_argument("--distill-alpha", type=float, default=0.7)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--resume", help="Checkpoint directory with optimizer/scheduler state")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--none-text")
    parser.add_argument("--separator")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--bf16", action="store_true")
    allowed = {a.dest for a in parser._actions}
    unknown = set(config) - allowed
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for required in ("model", "train", "dev", "output"):
        if not getattr(args, required):
            parser.error(f"--{required} is required (or set it in --config)")
    if args.distill_temperature <= 0 or not 0 <= args.distill_alpha <= 1 or not 0 <= args.label_smoothing < 1:
        parser.error("Invalid distillation temperature, weight or label smoothing")
    if args.batch_size < 1 or args.grad_accum < 1 or args.save_steps < 1:
        parser.error("batch-size, grad-accum and save-steps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    output = Path(args.output)
    if (output / "final").exists() and not args.resume:
        raise ValueError("Output already contains final model; choose a new run directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": args.train, "dev": args.dev}
    if args.calibration:
        splits["calibration"] = args.calibration
    manifest = validate_splits(splits)
    if args.resume and (output / "data_manifest.json").exists():
        previous_manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
        for name, info in manifest["splits"].items():
            if previous_manifest["splits"].get(name, {}).get("sha256") != info["sha256"]:
                raise ValueError(f"Resume dataset changed: {name}")
    train, dev = read_jsonl(args.train), read_jsonl(args.dev)
    if any(ex.provenance.get("review_status") == "unreviewed" for ex in train + dev):
        raise ValueError("Unreviewed generated examples must be reviewed before training")
    if not train or not dev:
        raise ValueError("Training and development files must be nonempty")
    if any(ex.gold_ids is None for ex in dev):
        raise ValueError("Development examples must have gold labels")
    predictor = Predictor.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only, device=args.device
    )
    predictor.metadata["max_length"] = args.max_length
    if args.none_text:
        predictor.metadata["none_text"] = args.none_text
    if args.separator:
        predictor.metadata["separator"] = args.separator
    validate_metadata(predictor.metadata, predictor.tokenizer, predictor.model.config.max_position_embeddings)
    teacher = read_teacher(args.teacher_scores, train, args.distill_temperature)
    train_dataset = PreparedDataset(train, predictor, args.seed, shuffle=True, teacher=teacher)
    dev_dataset = PreparedDataset(dev, predictor)
    predictor.metadata.setdefault("base_model", args.model)
    predictor.metadata.setdefault("base_revision", predictor.metadata.get("model_revision", args.revision))
    predictor.model.train()
    predictor.model.config.label_smoothing = 0  # custom objective handles it once
    predictor.metadata["calibration"] = {
        "temperature": 1.0,
        "min_probability": 0.0,
        "min_margin": 0.0,
        "version": "uncalibrated",
    }
    resolved = vars(args)
    if args.resume:
        previous = output / "resolved_config.json"
        if previous.exists():
            old = json.loads(previous.read_text(encoding="utf-8"))
            for key in (
                "model",
                "revision",
                "train",
                "dev",
                "seed",
                "learning_rate",
                "teacher_scores",
                "distill_alpha",
                "distill_temperature",
                "label_smoothing",
                "max_length",
                "none_text",
                "separator",
            ):
                if old.get(key) != resolved.get(key):
                    raise ValueError(f"Resume configuration changed {key}; start a new run")
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    if args.teacher_scores:
        manifest["teacher_scores"] = {"sha256": file_sha256(args.teacher_scores), "path": args.teacher_scores}
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        git_commit = ""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "device": args.device,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "base_revision": predictor.metadata.get("model_revision", args.revision),
    }
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    if args.bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("--bf16 requires a CUDA device with bf16 support")
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        bf16=args.bf16,
        use_cpu=args.device == "cpu",
        report_to=args.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )
    from functools import partial

    from transformers import EarlyStoppingCallback, TrainerCallback

    class RunCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state.global_step, "time": time.time(), **(logs or {})}) + "\n")

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            predictor.tokenizer.save_pretrained(checkpoint)
            (checkpoint / "model_metadata.json").write_text(
                json.dumps(predictor.metadata, indent=2) + "\n", encoding="utf-8"
            )

    trainer = ChoiceTrainer(
        model=predictor.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=partial(collate, pad_id=predictor.tokenizer.pad_token_id),
        compute_metrics=_metrics,
        callbacks=[RunCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience)],
        alpha=args.distill_alpha if teacher else 0,
        temperature=args.distill_temperature,
        smoothing=args.label_smoothing,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    metrics = trainer.evaluate()
    final = output / "final"
    predictor.model.eval()
    predictor.save_pretrained(final)
    predictor.metadata["model_revision"] = "sha256:" + file_sha256(final / "model.safetensors")
    (final / "model_metadata.json").write_text(json.dumps(predictor.metadata, indent=2) + "\n", encoding="utf-8")
    (final / "training_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    for filename in ("resolved_config.json", "data_manifest.json", "environment.json"):
        (final / filename).write_bytes((output / filename).read_bytes())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final": str(final), "metrics": metrics}, indent=2))
    return final


if __name__ == "__main__":
    main()
