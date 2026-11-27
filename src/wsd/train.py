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
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler

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


def choice_terms(
    logits, labels, valid_mask, teacher_probs=None, teacher_available=None, alpha=0.0, temperature=2.0, smoothing=0.0
):
    """Per-row loss and its parts: label cross-entropy ``ce``, ``kl`` (T² × KL to the teacher, or None when no row of
    the batch has teacher scores), the ``labeled`` and ``teacher_available`` masks, and the masked float logits."""
    logits = logits.float().masked_fill(~valid_mask, -1e9)
    log_probs = F.log_softmax(logits, dim=-1)
    labeled = labels.sum(-1) > 0
    gold_logprob = torch.logsumexp(log_probs.masked_fill(~labels.bool(), -1e9), dim=-1)
    ce = -gold_logprob
    if smoothing:
        uniform = -(log_probs.masked_fill(~valid_mask, 0).sum(-1) / valid_mask.sum(-1))
        ce = (1 - smoothing) * ce + smoothing * uniform
    teacher_available = torch.zeros_like(labeled) if teacher_available is None else teacher_available.bool()
    kl = None
    if teacher_probs is not None and teacher_available.any():
        student_log = F.log_softmax(logits / temperature, dim=-1)
        teacher_log = teacher_probs.clamp_min(1e-30).log()
        kl = (teacher_probs * (teacher_log - student_log)).sum(-1) * temperature**2
        loss = torch.where(labeled & teacher_available, (1 - alpha) * ce + alpha * kl, torch.where(labeled, ce, kl))
    else:
        loss = ce
    if not (labeled | teacher_available).all():
        raise ValueError("Unlabeled rows require a teacher distribution")
    return {
        "loss": loss,
        "logits": logits,
        "ce": ce,
        "kl": kl,
        "labeled": labeled,
        "teacher_available": teacher_available,
    }


def choice_loss(
    logits, labels, valid_mask, teacher_probs=None, teacher_available=None, alpha=0.0, temperature=2.0, smoothing=0.0
):
    """Marginal likelihood over acceptable golds; mask absent answer slots."""
    terms = choice_terms(logits, labels, valid_mask, teacher_probs, teacher_available, alpha, temperature, smoothing)
    return terms["loss"].mean(), terms["logits"]


class EpochSeededLengthGroupedSampler(LengthGroupedSampler):
    """Length-grouped order drawn from a generator seeded by (seed, epoch) instead of torch's global random state.

    transformers' LengthGroupedSampler draws its permutation from the global state when the epoch's iterator is
    made, so a resumed run, whose global state differs at that moment (for example when it skips the evaluation a
    fresh run made before training), trains on a new batch order. Here a run resumed mid-epoch replays that epoch's
    order, and skipping the batches already trained leaves the rest of the epoch as it was."""

    def __init__(self, batch_size, dataset, seed=42):
        super().__init__(batch_size, dataset=dataset, model_input_name="input_ids")
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        self.generator = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return super().__iter__()


class ChoiceTrainer(Trainer):
    # compute_loss returns the mean over one micro-batch and ignores num_items_in_batch, so Trainer must divide it by
    # the gradient accumulation steps. Trainer infers "already normalised over the accumulated batch" from a
    # forward(**kwargs) signature, which WSDModernBertForMaskedLM has; without these two settings, loss and gradients
    # are gradient_accumulation_steps times too large. The trainer's num_items_in_batch would also be wrong here: it
    # counts labels != -100, every one of the 128 slots of the multi-hot labels. loss_is_scaled_for_ga is read by
    # transformers 5.19 and later; model_accepts_loss_kwargs = False gives the same scaling from 5.0.
    loss_is_scaled_for_ga = False

    def __init__(self, *args, alpha=0, temperature=2, smoothing=0, length_grouped=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.alpha, self.temperature, self.smoothing = alpha, temperature, smoothing
        self.length_grouped = length_grouped
        self._parts = {}

    def _get_train_sampler(self, train_dataset=None):
        if not self.length_grouped:
            return super()._get_train_sampler(train_dataset)
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return EpochSeededLengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps, dataset, seed=self.args.seed
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels, valid = inputs.pop("labels"), inputs.pop("valid_mask")
        teacher, available = inputs.pop("teacher_probs"), inputs.pop("teacher_available")
        outputs = model(**inputs)
        terms = choice_terms(
            outputs.logits, labels, valid, teacher, available, self.alpha, self.temperature, self.smoothing
        )
        if model.training:
            # The parts of the loss, summed on the device until the next log (no synchronisation per micro-batch).
            with torch.no_grad():
                parts = self._parts
                labeled, teacher_rows = terms["labeled"], terms["teacher_available"]
                parts["ce"] = parts.get("ce", 0) + terms["ce"][labeled].sum()
                parts["ce_n"] = parts.get("ce_n", 0) + labeled.sum()
                if terms["kl"] is not None:
                    parts["kl"] = parts.get("kl", 0) + terms["kl"][teacher_rows].sum()
                    parts["kl_n"] = parts.get("kl_n", 0) + teacher_rows.sum()
        loss = terms["loss"].mean()
        return (loss, {"logits": terms["logits"]}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._parts:
            parts = self._parts
            if int(parts.get("ce_n", 0)):
                logs["label_ce"] = float(parts["ce"] / parts["ce_n"])
            if int(parts.get("kl_n", 0)):
                logs["teacher_kl"] = float(parts["kl"] / parts["kl_n"])
            self._parts = {}
        super().log(logs, *args, **kwargs)


def _metrics(pred):
    """Accuracy (the top choice among the offered candidates and none-fits is a gold one), top-3 accuracy (a gold
    candidate among the three best offered candidates, over rows whose gold is offered) and the none-fits rate."""
    import numpy as np

    predictions, labels = pred
    chosen = np.argmax(predictions, axis=-1)
    labeled = labels.sum(-1) > 0
    result = {
        "accuracy": float(labels[np.arange(len(labels))[labeled], chosen[labeled]].astype(np.float64).mean())
        if labeled.any()
        else 0
    }
    nota = labels.shape[-1] - 1
    offered = predictions.copy()
    offered[:, nota] = -1e30
    top3 = np.argsort(-offered, axis=-1)[:, :3]
    inventory = labels[:, :nota].sum(-1) > 0
    if inventory.any():
        hits = np.take_along_axis(labels, top3, axis=-1).sum(-1) > 0
        result["top3_accuracy"] = float(hits[inventory].mean())
    result["none_rate"] = float((chosen == nota).mean())
    return result


class PrimaryEarlyStopping(EarlyStoppingCallback):
    """Early stopping on the primary development form only: the evaluations of the other forms (separate
    on_evaluate calls with their own metrics) are passed over silently."""

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric = args.metric_for_best_model
        if (metric if metric.startswith("eval_") else f"eval_{metric}") in metrics:
            super().on_evaluate(args, state, control, metrics, **kwargs)


CHECKPOINT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")


def latest_checkpoint(folder):
    """The highest-step checkpoint-N in folder that holds every file a resume needs (a crash mid-save leaves one
    without), or None."""
    found = []
    for path in Path(folder).glob("checkpoint-*"):
        step = path.name.rsplit("-", 1)[-1]
        if not step.isdigit() or not all((path / name).is_file() for name in CHECKPOINT_FILES):
            continue
        try:
            json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        except ValueError:
            continue
        found.append((int(step), path))
    return max(found)[1] if found else None


def read_training_examples(paths, seed):
    """The examples of every file, in the order given, shuffled once with the run's seed."""
    examples, seen = [], set()
    for path in paths:
        for example in read_jsonl(path):
            if example.id in seen:
                raise ValueError(f"Duplicate training example ID across files: {example.id}")
            seen.add(example.id)
            examples.append(example)
    random.Random(seed).shuffle(examples)
    return examples


def limit_examples(examples, limit, seed):
    """At most ``limit`` examples, drawn with the seed (all of them, in file order, when limit is None)."""
    if limit is None:
        return examples
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args(argv)
    config = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--train", nargs="+", help="Training example files, read as one split")
    parser.add_argument("--dev", help="The development file used to choose the best checkpoint and stop early")
    parser.add_argument("--dev-name", default="dev", help="Name of --dev in the logged metrics when --extra-dev is set")
    parser.add_argument(
        "--extra-dev", nargs="*", default=[], help="NAME=PATH development forms evaluated alongside --dev"
    )
    parser.add_argument("--dev-limit", type=int, help="Evaluate at most this many examples of each development form")
    parser.add_argument("--calibration")
    parser.add_argument("--output")
    parser.add_argument("--data-dir", help="Folder that relative data paths (train, dev, teacher scores) refer to")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, help="Development batch size (default: --batch-size)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--teacher-scores")
    parser.add_argument("--distill-alpha", type=float, default=0.7)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--drop-labels", action="store_true", help="Teacher only: train on labelled examples without their gold_ids"
    )
    parser.add_argument(
        "--candidate-order",
        choices=("shuffle", "given"),
        default="shuffle",
        help="shuffle candidates once per run (default) or keep each example's order",
    )
    parser.add_argument(
        "--group-by-length", action="store_true", help="Batches of similar length, in an order seeded by epoch"
    )
    parser.add_argument(
        "--resume", help="Checkpoint directory with optimizer/scheduler state, or 'latest' for the newest complete one"
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--evaluate-before-training", action="store_true", help="Evaluate the starting model first (baseline.json)"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--none-text")
    parser.add_argument("--separator")
    parser.add_argument("--report-to", default="none", help="none, tensorboard (extra 'tensorboard') or wandb")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--allow-silver", action="store_true", help="Explicitly allow automatic labels in training only"
    )
    parser.add_argument(
        "--allow-split-overlap",
        action="store_true",
        help="Record cross-split overlap in data_manifest.json instead of failing (leak checks made elsewhere)",
    )
    allowed = {a.dest for a in parser._actions}
    unknown = set(config) - allowed
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    if isinstance(config.get("train"), str):
        config["train"] = [config["train"]]
    if isinstance(config.get("extra_dev"), dict):
        config["extra_dev"] = [f"{name}={path}" for name, path in config["extra_dev"].items()]
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for required in ("model", "train", "dev", "output"):
        if not getattr(args, required):
            parser.error(f"--{required} is required (or set it in --config)")
    if args.distill_temperature <= 0 or not 0 <= args.distill_alpha <= 1 or not 0 <= args.label_smoothing < 1:
        parser.error("Invalid distillation temperature, weight or label smoothing")
    if args.batch_size < 1 or args.grad_accum < 1 or args.save_steps < 1 or args.logging_steps < 1:
        parser.error("batch-size, grad-accum, save-steps and logging-steps must be positive")
    if (args.eval_batch_size is not None and args.eval_batch_size < 1) or (
        args.dev_limit is not None and args.dev_limit < 1
    ):
        parser.error("eval-batch-size and dev-limit must be positive")
    if any("=" not in item for item in args.extra_dev):
        parser.error("--extra-dev takes NAME=PATH")
    args.extra_dev = dict(item.split("=", 1) for item in args.extra_dev)
    if args.dev_name in args.extra_dev:
        parser.error("--dev-name must differ from the --extra-dev names")
    if args.drop_labels and not args.teacher_scores:
        parser.error("--drop-labels needs --teacher-scores")
    return args


def _data_path(args, path):
    if path is None or args.data_dir is None or Path(path).is_absolute():
        return path
    return str(Path(args.data_dir) / path)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    output = Path(args.output)
    if (output / "final").exists() and (not args.resume or args.resume == "latest"):
        raise ValueError("Output already contains final model; choose a new run directory or --resume CHECKPOINT")
    output.mkdir(parents=True, exist_ok=True)
    resume = args.resume
    if resume == "latest":
        resume = latest_checkpoint(output / "checkpoints")
        resume = str(resume) if resume else None
    train_paths = [_data_path(args, path) for path in args.train]
    dev_paths = {args.dev_name: _data_path(args, args.dev)}
    dev_paths.update({name: _data_path(args, path) for name, path in args.extra_dev.items()})
    teacher_scores = _data_path(args, args.teacher_scores)
    calibration_path = _data_path(args, args.calibration)
    splits = {
        "train": train_paths if len(train_paths) > 1 else train_paths[0],
        "dev": list(dev_paths.values()) if len(dev_paths) > 1 else dev_paths[args.dev_name],
    }
    if calibration_path:
        splits["calibration"] = calibration_path
    manifest = validate_splits(splits, strict=not args.allow_split_overlap)
    if resume and (output / "data_manifest.json").exists():
        previous_manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
        for name, info in manifest["splits"].items():
            if previous_manifest["splits"].get(name, {}).get("sha256") != info["sha256"]:
                raise ValueError(f"Resume dataset changed: {name}")
    train = read_training_examples(train_paths, args.seed)
    devs = {name: limit_examples(read_jsonl(path), args.dev_limit, args.seed) for name, path in dev_paths.items()}
    from wsd.pipeline import require_reviewed

    require_reviewed(train, allow_silver=args.allow_silver)
    for dev in devs.values():
        require_reviewed(dev)
    if calibration_path:
        require_reviewed(read_jsonl(calibration_path))
    if not train or not all(devs.values()):
        raise ValueError("Training and development files must be nonempty")
    if any(ex.gold_ids is None for dev in devs.values() for ex in dev):
        raise ValueError("Development examples must have gold labels")
    predictor = Predictor.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only, device=args.device
    )
    # Optimizer state and master weights in float32, also when the checkpoint is stored in bfloat16 (--bf16 trains
    # with autocast over float32 weights).
    predictor.model.float()
    predictor.metadata["max_length"] = args.max_length
    if args.none_text:
        predictor.metadata["none_text"] = args.none_text
    if args.separator:
        predictor.metadata["separator"] = args.separator
    validate_metadata(predictor.metadata, predictor.tokenizer, predictor.model.config.max_position_embeddings)
    # The cache's fingerprints cover gold_ids, so it is checked against the examples as stored, before labels drop.
    teacher = read_teacher(teacher_scores, train, args.distill_temperature)
    if args.drop_labels:
        for ex in train:
            ex.gold_ids = None
    train_dataset = PreparedDataset(
        train, predictor, args.seed, shuffle=args.candidate_order == "shuffle", teacher=teacher
    )
    dev_datasets = {name: PreparedDataset(dev, predictor) for name, dev in devs.items()}
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
                "extra_dev",
                "dev_limit",
                "seed",
                "learning_rate",
                "teacher_scores",
                "distill_alpha",
                "distill_temperature",
                "label_smoothing",
                "drop_labels",
                "candidate_order",
                "group_by_length",
                "max_length",
                "none_text",
                "separator",
            ):
                if old.get(key) != resolved.get(key):
                    raise ValueError(f"Resume configuration changed {key}; start a new run")
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    if teacher_scores:
        manifest["teacher_scores"] = {"sha256": file_sha256(teacher_scores), "path": teacher_scores}
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
    named = len(dev_datasets) > 1
    extra = {"train_sampling_strategy": "group_by_length"} if args.group_by_length else {}
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
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
        metric_for_best_model=f"{args.dev_name}_accuracy" if named else "accuracy",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        bf16=args.bf16,
        use_cpu=args.device == "cpu",
        report_to=args.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        **extra,
    )
    from functools import partial

    from transformers import TrainerCallback

    from wsd.predict import save_tokenizer

    class RunCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state.global_step, "time": time.time(), **(logs or {})}) + "\n")

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            save_tokenizer(predictor.tokenizer, checkpoint)
            (checkpoint / "model_metadata.json").write_text(
                json.dumps(predictor.metadata, indent=2) + "\n", encoding="utf-8"
            )

    trainer = ChoiceTrainer(
        model=predictor.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets if named else dev_datasets[args.dev_name],
        data_collator=partial(collate, pad_id=predictor.tokenizer.pad_token_id),
        compute_metrics=_metrics,
        callbacks=[RunCallback(), PrimaryEarlyStopping(early_stopping_patience=args.patience)],
        alpha=args.distill_alpha if teacher else 0,
        temperature=args.distill_temperature,
        smoothing=args.label_smoothing,
        length_grouped=args.group_by_length,
    )
    if resume:
        step = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"resume_from_step": step, "checkpoint": Path(resume).name, "time": time.time()}))
            stream.write("\n")
    baseline = output / "baseline.json"
    if args.evaluate_before_training and not (resume and baseline.exists()):
        baseline.write_text(json.dumps(trainer.evaluate(), indent=2) + "\n", encoding="utf-8")
    trainer.train(resume_from_checkpoint=resume)
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
