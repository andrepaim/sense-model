import json
from pathlib import Path

import pytest
import torch
from test_core import example, tiny_predictor

from wsd.data import prepare_raganato, read_lexicon, split_examples, validate_splits, write_jsonl
from wsd.schema import Target
from wsd.train import PreparedDataset, choice_loss, example_fingerprint, main, read_teacher


def test_split_families_and_leakage(tmp_path):
    xs = []
    for i in range(12):
        x = example()
        x.id = str(i)
        x.text = f"bank bank {i}"
        x.provenance = {"source": "fixture", "document_id": str(i // 2)}
        xs.append(x)
    groups = split_examples(xs)
    paths = {}
    for name, records in groups.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], records)
    assert validate_splits(paths)["leakage"] == "passed"
    assert sum(len(rows) for rows in groups.values()) == 12
    write_jsonl(tmp_path / "a.jsonl", xs[:1])
    write_jsonl(tmp_path / "b.jsonl", xs[:1])
    with pytest.raises(ValueError, match="leakage"):
        validate_splits({"train": tmp_path / "a.jsonl", "test": tmp_path / "b.jsonl"})


def test_teacher_remaps_candidate_ids(tmp_path):
    ex = example()
    row = {
        "id": ex.id,
        "example_sha256": example_fingerprint(ex),
        "temperature": 2,
        "probabilities": {"a": 0.2, "b": 0.7},
        "none_probability": 0.1,
    }
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps(row) + "\n")
    teacher = read_teacher(path, [ex], 2)
    p = tiny_predictor()
    data = PreparedDataset([ex], p, seed=1, shuffle=True, teacher=teacher)
    # Seed 1 swaps a/b, gold follows a and scores follow candidate IDs.
    assert data[0]["labels"].argmax().item() == 1
    torch.testing.assert_close(data[0]["teacher_probs"][:2], torch.tensor([0.7, 0.2]))
    row["example_sha256"] = "stale"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        read_teacher(path, [ex], 2)


def test_multi_gold_mask_and_none_loss():
    logits = torch.tensor([[2.0, 2.0, 100.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, False, True]])
    loss, masked = choice_loss(logits, labels, valid)
    assert loss < 0.1
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert masked.argmax(-1).item() in (0, 1)
    # No-gold + no-teacher should never silently train a rejection label.
    with pytest.raises(ValueError):
        choice_loss(logits, torch.zeros_like(labels), valid)


def test_raganato_repeated_target_and_missing_mapping(tmp_path):
    prefix = tmp_path / "corpus"
    Path(str(prefix) + ".data.xml").write_text(
        '<corpus><text><sentence><wf>bank</wf><instance id="doc.s0.t1" lemma="bank" pos="NOUN">bank</instance><wf>.</wf></sentence></text></corpus>'
    )
    Path(str(prefix) + ".gold.key.txt").write_text("doc.s0.t1 bank%1:00:00::\n")
    index = tmp_path / "index.sense"
    index.write_text("bank%1:00:00:: 00000001 1 1\n")
    from wsd.schema import Candidate

    rows, counts = prepare_raganato(
        prefix,
        index,
        {("bank", "n"): [Candidate("omw-en-00000001-n", "seat")]},
        {"source": "fixture", "license": "CC0", "version": "1"},
    )
    assert rows[0].target.start == 5
    assert rows[0].text == "bank bank."
    assert counts == {"accepted": 1}


def test_tiny_training_and_resume(tmp_path):
    p = tiny_predictor()
    model = tmp_path / "model"
    p.save_pretrained(model)
    a, b = example(), example()
    a.id, b.id = "train", "dev"
    b.text = "bank bank independently"
    write_jsonl(tmp_path / "train.jsonl", [a])
    write_jsonl(tmp_path / "dev.jsonl", [b])
    args = [
        "--model",
        str(model),
        "--train",
        str(tmp_path / "train.jsonl"),
        "--dev",
        str(tmp_path / "dev.jsonl"),
        "--output",
        str(tmp_path / "run"),
        "--max-steps",
        "1",
        "--save-steps",
        "1",
        "--batch-size",
        "1",
        "--max-length",
        "256",
        "--local-files-only",
    ]
    final = main(args)
    assert (final / "model_metadata.json").exists()
    assert (final / "training_config.json").exists()
    main([*args, "--max-steps", "2", "--resume", str(tmp_path / "run/checkpoints/checkpoint-1")])
    state = json.loads((tmp_path / "run/checkpoints/checkpoint-2/trainer_state.json").read_text())
    assert state["global_step"] == 2


def test_gradient_accumulation_matches_full_batch(tmp_path):
    # One optimizer step over the same four examples, as one batch of four or as four accumulated batches of one,
    # must log the same mean loss and the same pre-clipping gradient norm. A trainer that leaves out the division by
    # the accumulation steps logs both four times too large.
    p = tiny_predictor()
    model = tmp_path / "model"
    p.save_pretrained(model)
    train = []
    for i, (word, gold) in enumerate([("river", "a"), ("money", "b"), ("seat", "a"), ("loan", "b")]):
        x = example()
        x.id, x.text, x.target, x.gold_ids = f"train-{i}", f"bank {word}", Target(0, 4, "bank"), [gold]
        train.append(x)
    dev = example()
    dev.id, dev.text = "dev", "bank bank independently"
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "dev.jsonl", [dev])

    def first_step(name, batch_size, grad_accum):
        main(
            [
                "--model",
                str(model),
                "--train",
                str(tmp_path / "train.jsonl"),
                "--dev",
                str(tmp_path / "dev.jsonl"),
                "--output",
                str(tmp_path / name),
                "--max-steps",
                "1",
                "--save-steps",
                "1",
                "--batch-size",
                str(batch_size),
                "--grad-accum",
                str(grad_accum),
                "--max-length",
                "256",
                "--local-files-only",
            ]
        )
        rows = [json.loads(line) for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
        return next(row for row in rows if "loss" in row and "grad_norm" in row)

    full = first_step("full", batch_size=4, grad_accum=1)
    accumulated = first_step("accumulated", batch_size=1, grad_accum=4)
    assert accumulated["loss"] == pytest.approx(full["loss"], rel=1e-3)
    assert accumulated["grad_norm"] == pytest.approx(full["grad_norm"], rel=1e-3)


def test_portuguese_context_join_and_generation(tmp_path, monkeypatch):
    import importlib.util

    from wsd.data import prepare_contexts

    lex = tmp_path / "lexicon.jsonl"
    lex.write_text(
        json.dumps(
            {"lemma": "banco", "pos": "n", "language": "pt", "candidates": [{"id": "seat", "definition": "Assento."}]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    context = tmp_path / "context.jsonl"
    context.write_text(
        json.dumps(
            {
                "id": "pt",
                "text": "O banco caiu.",
                "target": {"start": 2, "end": 7, "text": "banco"},
                "lemma": "banco",
                "pos": "n",
                "gold_ids": ["seat"],
                "provenance": {"annotation": "human"},
            }
        )
        + "\n"
    )
    examples, report = prepare_contexts(context, read_lexicon(lex), {"source": "fixture", "license": "CC0"}, "pt-BR")
    assert examples[0].gold_ids == ["seat"]
    spec = importlib.util.spec_from_file_location(
        "generate_contexts", Path(__file__).parent.parent / "scripts/generate_contexts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import io

    response = {"response": json.dumps({"text": "O banco caiu.", "target_text": "banco", "start": 2})}
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    output = tmp_path / "generated.jsonl"
    module.main(
        ["--lexicon", str(lex), "--model", "test", "--language", "pt-BR", "--output", str(output), "--limit", "1"]
    )
    row = json.loads(output.read_text())
    assert row["provenance"]["annotation"] == "synthetic"
    assert row["provenance"]["review_status"] == "unreviewed"


def test_glosstag_adapter(tmp_path):
    from wsd.corpora import prepare_glosstag
    from wsd.schema import Candidate

    merged = tmp_path / "merged"
    merged.mkdir()
    xml = '<root><synset ofs="00000001" pos="n"><def><wf tag="man" sep=" "><id sk="bank%1:00:00::" lemma="bank"/>bank</wf><wf sep="">.</wf></def></synset></root>'
    for name in ("noun.xml", "verb.xml", "adj.xml", "adv.xml"):
        (merged / name).write_text(xml if name == "noun.xml" else "<root/>")
    index = tmp_path / "index.sense"
    index.write_text("bank%1:00:00:: 00000001 1 1\n")
    rows, report = prepare_glosstag(
        tmp_path,
        index,
        {("bank", "n"): [Candidate("omw-en-00000001-n", "seat")]},
        {"source": "fixture", "license": "CC0"},
    )
    assert rows[0].target.text == "bank"
    assert report == {"accepted": 1}
