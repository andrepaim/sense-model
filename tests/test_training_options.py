"""Training options of the Wiktionary round: several files, teacher-only rows, given candidate order,
length-grouped batches that survive a resume, capped development forms, and the logged loss parts."""

import json
import random

import pytest
import torch
from test_core import example, tiny_predictor

from wsd.data import validate_splits, write_jsonl
from wsd.schema import Candidate, Example, Target
from wsd.train import (
    EpochSeededLengthGroupedSampler,
    example_fingerprint,
    latest_checkpoint,
    limit_examples,
    main,
    read_training_examples,
)


def make(id, words, gold=("a",), document=None, tail=""):
    text = " ".join(["bank"] * words) + tail
    ex = Example(id, text, Target(0, 4, "bank"), [Candidate("a", "seat"), Candidate("b", "money")])
    ex.gold_ids = None if gold is None else list(gold)
    ex.provenance = {"source": "fixture", "document_id": document or id}
    return ex


def teacher_row(ex):
    return {
        "id": ex.id,
        "example_sha256": example_fingerprint(ex),
        "probabilities": {"a": 0.6, "b": 0.3},
        "none_probability": 0.1,
        "temperature": 2.0,
    }


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_training_files_are_concatenated_then_shuffled_once(tmp_path):
    first, second = [make(f"a{i}", 2) for i in range(5)], [make(f"b{i}", 3) for i in range(4)]
    write_jsonl(tmp_path / "a.jsonl", first)
    write_jsonl(tmp_path / "b.jsonl", second)
    expected = [ex.id for ex in first + second]
    random.Random(7).shuffle(expected)
    assert [ex.id for ex in read_training_examples([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], 7)] == expected
    write_jsonl(tmp_path / "c.jsonl", first[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        read_training_examples([tmp_path / "a.jsonl", tmp_path / "c.jsonl"], 7)


def test_development_cap_is_a_seeded_sample():
    examples = [make(str(i), 2) for i in range(10)]
    assert limit_examples(examples, None, 42) == examples
    expected = list(examples)
    random.Random(42).shuffle(expected)
    assert limit_examples(examples, 3, 42) == expected[:3]


def test_length_sampler_is_seeded_by_epoch_not_by_global_state():
    dataset = [{"input_ids": torch.zeros(n)} for n in [random.Random(i).randint(3, 40) for i in range(300)]]
    torch.manual_seed(0)
    first = list(EpochSeededLengthGroupedSampler(4, dataset, seed=42))
    torch.manual_seed(1)
    torch.rand(10)
    again = list(EpochSeededLengthGroupedSampler(4, dataset, seed=42))
    assert first == again and sorted(first) == list(range(300))
    later = EpochSeededLengthGroupedSampler(4, dataset, seed=42)
    later.set_epoch(1)
    assert list(later) != first
    # Mega-batches are sorted by length: the first index is the longest of its mega-batch of 50 batches.
    assert len(dataset[first[0]]["input_ids"]) == max(len(dataset[i]["input_ids"]) for i in first[:200])


def test_latest_checkpoint_needs_every_resume_file(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    names = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")
    for step, files, state in ((2, names, "{}"), (5, names[:-1], "{}"), (7, names, "{broken")):
        folder = tmp_path / f"checkpoint-{step}"
        folder.mkdir()
        for name in files:
            (folder / name).write_text(state if name == "trainer_state.json" else "x")
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-2"


def test_split_validation_takes_file_lists_and_can_record_overlap(tmp_path):
    write_jsonl(tmp_path / "t1.jsonl", [make("t1", 2, document="d1")])
    write_jsonl(tmp_path / "t2.jsonl", [make("t2", 3, document="d1")])
    write_jsonl(tmp_path / "dev.jsonl", [make("dev", 4, document="d2")])
    report = validate_splits({"train": [tmp_path / "t1.jsonl", tmp_path / "t2.jsonl"], "dev": tmp_path / "dev.jsonl"})
    assert report["leakage"] == "passed"
    assert report["splits"]["train"]["count"] == 2 and len(report["splits"]["train"]["files"]) == 2
    write_jsonl(tmp_path / "dev2.jsonl", [make("dev2", 2, document="d3")])
    splits = {"train": [tmp_path / "t1.jsonl"], "dev": [tmp_path / "dev.jsonl", tmp_path / "dev2.jsonl"]}
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits)
    recorded = validate_splits(splits, strict=False)
    assert recorded["leakage"] == "failed" and recorded["leakage_findings"] == 1


def round_files(tmp_path):
    data = tmp_path / "data"
    labelled = [make(f"l{i}", 2 + i % 5) for i in range(6)]
    notoffered = [make("n0", 3, gold=None)]
    unlabelled = [make(f"u{i}", 2 + i, gold=None) for i in range(3)]
    write_jsonl(data / "labelled.jsonl", labelled)
    write_jsonl(data / "notoffered.jsonl", notoffered)
    write_jsonl(data / "unlabelled.jsonl", unlabelled)
    (data / "teacher.jsonl").write_text(
        "".join(json.dumps(teacher_row(ex)) + "\n" for ex in labelled + notoffered + unlabelled)
    )
    write_jsonl(data / "dev-capped.jsonl", [make(f"d{i}", 6 + i, document=f"dd{i}", tail=" dev") for i in range(4)])
    write_jsonl(data / "dev-uncapped.jsonl", [make(f"e{i}", 6 + i, document=f"dd{i}", tail=" dev") for i in range(4)])
    model = tmp_path / "model"
    tiny_predictor().save_pretrained(model)
    return model


def round_args(tmp_path, model, output, *extra):
    return [
        "--model", str(model), "--local-files-only", "--max-length", "256",
        "--data-dir", str(tmp_path / "data"),
        "--train", "labelled.jsonl", "notoffered.jsonl", "unlabelled.jsonl",
        "--teacher-scores", "teacher.jsonl",
        "--dev", "dev-capped.jsonl", "--dev-name", "c1600", "--extra-dev", "uncapped=dev-uncapped.jsonl",
        "--dev-limit", "3", "--candidate-order", "given", "--group-by-length",
        "--batch-size", "2", "--grad-accum", "2", "--eval-batch-size", "3", "--logging-steps", "1",
        "--evaluate-before-training", "--save-steps", "1", "--max-steps", "2",
        "--output", str(output), *extra,
    ]  # fmt: skip


def test_round_options_train_and_log(tmp_path):
    model = round_files(tmp_path)
    final = main(round_args(tmp_path, model, tmp_path / "run"))
    metrics = rows(tmp_path / "run" / "metrics.jsonl")
    before = [r for r in metrics if r["step"] == 0]
    assert {"eval_c1600_accuracy", "eval_c1600_top3_accuracy", "eval_c1600_none_rate"} <= set(before[0])
    assert "eval_uncapped_accuracy" in before[1]
    losses = [r for r in metrics if "loss" in r]
    assert [r["step"] for r in losses] == [1, 2]
    assert all("label_ce" in r and "teacher_kl" in r for r in losses)
    assert (
        json.loads((tmp_path / "run" / "baseline.json").read_text())["eval_c1600_accuracy"]
        == before[0]["eval_c1600_accuracy"]
    )
    manifest = json.loads((final / "data_manifest.json").read_text())
    assert [f["count"] for f in manifest["splits"]["train"]["files"]] == [6, 1, 3]
    # Paths under --data-dir are recorded relative to it, so the manifest names the dataset's own files.
    assert [f["path"] for f in manifest["splits"]["train"]["files"]] == [
        "labelled.jsonl",
        "notoffered.jsonl",
        "unlabelled.jsonl",
    ]
    assert manifest["teacher_scores"]["path"] == "teacher.jsonl"
    assert manifest["splits"]["dev"]["count"] == 8
    config = json.loads((final / "training_config.json").read_text())
    assert config["candidate_order"] == "given" and config["group_by_length"] is True


def test_teacher_only_drops_labels_after_checking_the_cache(tmp_path):
    model = round_files(tmp_path)
    main(round_args(tmp_path, model, tmp_path / "run", "--drop-labels"))
    losses = [r for r in rows(tmp_path / "run" / "metrics.jsonl") if "loss" in r]
    assert losses and all("teacher_kl" in r and "label_ce" not in r for r in losses)


def test_resume_latest_replays_the_length_grouped_batches(tmp_path):
    # 120 rows of different lengths: more than one mega-batch, so the order depends on the sampler's random draw.
    data = tmp_path / "data"
    write_jsonl(data / "train.jsonl", [make(f"t{i}", 2 + (i * 7) % 30) for i in range(120)])
    write_jsonl(data / "dev.jsonl", [make("dev", 3, document="dev", tail=" dev")])
    model = tmp_path / "model"
    tiny_predictor().save_pretrained(model)
    args = [
        "--model", str(model), "--local-files-only", "--max-length", "256", "--data-dir", str(data),
        "--train", "train.jsonl", "--dev", "dev.jsonl", "--group-by-length", "--evaluate-before-training",
        "--batch-size", "1", "--logging-steps", "1", "--save-steps", "1", "--max-steps", "3",
        "--output", str(tmp_path / "run"),
    ]  # fmt: skip
    main(args)
    run = tmp_path / "run"
    uninterrupted = {r["step"]: r["loss"] for r in rows(run / "metrics.jsonl") if "loss" in r}
    import shutil

    shutil.rmtree(run / "final")
    shutil.rmtree(run / "checkpoints" / "checkpoint-3")
    main([*args, "--resume", "latest"])
    logged = rows(run / "metrics.jsonl")
    marker = next(r for r in logged if "resume_from_step" in r)
    resumed = [r for r in logged[logged.index(marker) :] if "loss" in r]
    assert marker["resume_from_step"] < 3
    assert [r["step"] for r in resumed] == list(range(marker["resume_from_step"] + 1, 4))
    for r in resumed:
        assert r["loss"] == pytest.approx(uninterrupted[r["step"]], rel=1e-6)


def test_checkpoint_stored_in_bfloat16_trains_in_float32(tmp_path):
    from safetensors import safe_open

    p = tiny_predictor()
    p.model.to(torch.bfloat16)
    p.save_pretrained(tmp_path / "model")
    a, b = example(), example()
    a.id, b.id, b.text = "train", "dev", "bank bank independently"
    write_jsonl(tmp_path / "train.jsonl", [a])
    write_jsonl(tmp_path / "dev.jsonl", [b])
    final = main(
        [
            "--model",
            str(tmp_path / "model"),
            "--local-files-only",
            "--max-length",
            "256",
            "--train",
            str(tmp_path / "train.jsonl"),
            "--dev",
            str(tmp_path / "dev.jsonl"),
            "--max-steps",
            "1",
            "--save-steps",
            "1",
            "--batch-size",
            "1",
            "--output",
            str(tmp_path / "run"),
        ]  # fmt: skip
    )
    with safe_open(final / "model.safetensors", "pt") as weights:
        assert {weights.get_tensor(k).dtype for k in weights.keys()} == {torch.float32}


def test_saved_tokenizer_carries_no_batch_padding(tmp_path):
    p = tiny_predictor()
    a, b = example(), example()
    b.id, b.text = "longer", "bank bank bank bank"
    p.logits([a, b])
    p.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "tokenizer.json").read_text())["padding"] is None
    assert p.logits([a, b]).shape == (2, 128)


def small_run(tmp_path):
    data = tmp_path / "data"
    write_jsonl(data / "train.jsonl", [make(f"t{i}", 2 + i) for i in range(4)])
    write_jsonl(data / "dev.jsonl", [make("dev", 3, document="dev", tail=" dev")])
    model = tmp_path / "model"
    tiny_predictor().save_pretrained(model)
    return [
        "--model", str(model), "--local-files-only", "--max-length", "256", "--data-dir", str(data),
        "--train", "train.jsonl", "--dev", "dev.jsonl", "--batch-size", "1", "--save-steps", "1",
        "--max-steps", "3", "--output", str(tmp_path / "run"),
    ]  # fmt: skip


# resolved_config.json as wsd train wrote it before the round's options (origin/main 5a91847).
OLD_RESOLVED = {
    "config": None, "model": None, "revision": None, "train": None, "dev": None, "calibration": None,
    "output": None, "device": "cpu", "local_files_only": True, "seed": 42, "epochs": 3, "max_steps": 3,
    "batch_size": 1, "grad_accum": 1, "learning_rate": 5e-05, "weight_decay": 0.01, "label_smoothing": 0.1,
    "teacher_scores": None, "distill_alpha": 0.7, "distill_temperature": 2.0, "resume": None, "patience": 2,
    "save_steps": 1, "max_length": 256, "none_text": None, "separator": None, "report_to": "none",
    "bf16": False, "allow_silver": False,
}  # fmt: skip


def test_run_recorded_before_the_round_options_resumes(tmp_path):
    import shutil

    args = small_run(tmp_path)
    main(args)
    run = tmp_path / "run"
    old = dict(OLD_RESOLVED, model=args[1], train="train.jsonl", dev="dev.jsonl", output=str(run))
    (run / "resolved_config.json").write_text(json.dumps(old))
    shutil.rmtree(run / "final")
    shutil.rmtree(run / "checkpoints" / "checkpoint-3")
    main([*args, "--resume", str(run / "checkpoints" / "checkpoint-1")])
    assert (run / "final").is_dir()
    # A setting that changes the run is still refused.
    old["learning_rate"] = 1e-3
    (run / "resolved_config.json").write_text(json.dumps(old))
    shutil.rmtree(run / "final")
    with pytest.raises(ValueError, match="learning_rate"):
        main([*args, "--resume", str(run / "checkpoints" / "checkpoint-1")])


def test_single_training_file_keeps_file_order_unless_asked(tmp_path, monkeypatch):
    import wsd.train as train

    built = {}
    real = train.PreparedDataset

    def capture(examples, predictor, seed=42, shuffle=False, teacher=None):
        dataset = real(examples, predictor, seed, shuffle, teacher)
        built.setdefault("train", (list(examples), seed, shuffle, dataset))
        return dataset

    monkeypatch.setattr(train, "PreparedDataset", capture)
    args = small_run(tmp_path)
    main(args)
    examples, seed, shuffle, dataset = built.pop("train")
    from wsd.data import read_jsonl

    # As before the round's options: file order, candidates shuffled by PreparedDataset with the seed.
    in_file = read_jsonl(tmp_path / "data" / "train.jsonl")
    assert [ex.id for ex in examples] == [ex.id for ex in in_file] and shuffle is True and seed == 42
    expected = real(in_file, tiny_predictor(), 42, shuffle=True)
    assert all(torch.equal(a["labels"], b["labels"]) for a, b in zip(dataset.rows, expected.rows))
    main([*args[:-1], str(tmp_path / "shuffled"), "--shuffle-examples"])
    examples = built.pop("train")[0]
    order = [ex.id for ex in in_file]
    random.Random(42).shuffle(order)
    assert [ex.id for ex in examples] == order


def test_resume_latest_without_a_checkpoint_says_so_and_marks_the_restart(tmp_path, capsys):
    args = small_run(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"step": 1, "loss": 1.0}\n')
    (run / "data_manifest.json").write_text(json.dumps({"splits": {"train": {"sha256": "other"}}}))
    with pytest.raises(ValueError, match="Resume dataset changed"):
        main([*args, "--resume", "latest"])
    (run / "data_manifest.json").unlink()
    main([*args, "--resume", "latest"])
    assert "no complete checkpoint" in capsys.readouterr().out
    logged = rows(run / "metrics.jsonl")
    assert logged[1].get("restart") == "no complete checkpoint"


def test_length_grouping_arguments_follow_the_transformers_version():
    from wsd.train import length_grouping_arguments

    class New:
        __dataclass_fields__ = {"train_sampling_strategy": None}

    class Old:
        __dataclass_fields__ = {"group_by_length": None}

    assert length_grouping_arguments(New) == {"train_sampling_strategy": "group_by_length"}
    assert length_grouping_arguments(Old) == {"group_by_length": True}
    from transformers import TrainingArguments

    assert length_grouping_arguments(TrainingArguments) == {"train_sampling_strategy": "group_by_length"}


def test_early_stopping_counts_only_the_primary_development_form(tmp_path):
    from transformers import TrainerControl, TrainerState, TrainingArguments

    from wsd.train import PrimaryEarlyStopping

    args = TrainingArguments(
        output_dir=str(tmp_path),
        metric_for_best_model="c1600_accuracy",
        load_best_model_at_end=True,
        eval_strategy="steps",
        save_strategy="steps",
        use_cpu=True,
        report_to="none",
    )
    state, control = TrainerState(), TrainerControl()
    state.best_metric = 0.9
    stop = PrimaryEarlyStopping(early_stopping_patience=2)
    for _ in range(3):
        stop.on_evaluate(args, state, control, metrics={"eval_uncapped_accuracy": 0.1})
    assert stop.early_stopping_patience_counter == 0 and not control.should_training_stop
    stop.on_evaluate(args, state, control, metrics={"eval_c1600_accuracy": 0.5})
    stop.on_evaluate(args, state, control, metrics={"eval_c1600_accuracy": 0.5})
    assert control.should_training_stop
