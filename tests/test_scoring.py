"""Batched scoring by prompt (wsd score) and the teacher cache made from it (wsd teacher)."""

import hashlib
import json
import math

import torch
from test_core import example, tiny_predictor

from wsd.data import write_jsonl
from wsd.schema import Candidate, Example, Target
from wsd.scoring import prompt_sha256, read_scores, score_examples
from wsd.tokenizer import format_prompt
from wsd.train import example_fingerprint, read_teacher


def examples():
    rows = []
    for i, words in enumerate([2, 9, 4, 4, 12]):
        text = " ".join(["bank"] * words)
        candidates = [Candidate(f"c{j}", "seat money"[: 4 + j]) for j in range(2 + i % 3)]
        rows.append(Example(f"e{i}", text, Target(0, 4, "bank"), candidates, gold_ids=["c0"]))
    return rows


def test_prompt_hash_is_the_sha256_of_the_formatted_prompt():
    p, ex = tiny_predictor(), example()
    prompt = format_prompt(ex, p.tokenizer, p.metadata)
    assert prompt_sha256(ex, p.tokenizer, p.metadata) == hashlib.sha256(prompt.encode()).hexdigest()


def test_batched_scores_match_single_examples_and_resume(tmp_path):
    p = tiny_predictor()
    xs = examples()
    xs.append(Example("same-prompt", xs[0].text, xs[0].target, xs[0].candidates, gold_ids=["c1"]))
    out = tmp_path / "scores.jsonl"
    rows = score_examples(p, xs[:3], out, max_tokens=40, max_batch=2, label="tiny")
    assert len(rows) == 3
    # A cut last line (a killed run) is dropped and the run goes on with the prompts not yet scored.
    data = out.read_bytes()
    out.write_bytes(data[: data.rfind(b"\n", 0, len(data) - 1) + 1] + b'{"hash": "cut')
    rows = score_examples(p, xs, out, max_tokens=40, max_batch=2)
    scores = read_scores(out)
    assert len(scores) == 5 and len(out.read_text().splitlines()) == 5
    for ex in xs[:5]:
        row = scores[prompt_sha256(ex, p.tokenizer, p.metadata)]
        assert row["type"] == "scored" and row["offered"] == len(ex.candidates)
        single = p.logits([ex])[0]
        expected = [*single[: len(ex.candidates)].tolist(), single[127].item()]
        assert torch.allclose(torch.tensor(row["logits"]), torch.tensor(expected), atol=1e-5)


def test_too_long_prompts_are_rejected_not_scored(tmp_path):
    p = tiny_predictor()
    short, long = examples()[0], examples()[4]
    p.metadata["max_length"] = len(p.tokenizer(format_prompt(short, p.tokenizer, p.metadata))["input_ids"])
    rows = score_examples(p, [long, short], tmp_path / "s.jsonl")
    kinds = {r["type"]: r for r in rows}
    assert kinds["rejected"]["code"] == "TOO_LONG" and kinds["rejected"]["logits"] is None
    assert kinds["scored"]["logits"]


def test_teacher_cache_from_scores_without_scoring_again(tmp_path):
    from wsd.teacher import main

    p = tiny_predictor()
    p.save_pretrained(tmp_path / "teacher")
    xs = examples()
    unlabelled = Example("u", "bank river", Target(0, 4, "bank"), [Candidate("a", "seat"), Candidate("b", "money")])
    write_jsonl(tmp_path / "a.jsonl", xs[:3])
    write_jsonl(tmp_path / "b.jsonl", [*xs[3:], unlabelled])
    args = [
        "--model",
        str(tmp_path / "teacher"),
        "--local-files-only",
        "--input",
        str(tmp_path / "a.jsonl"),
        str(tmp_path / "b.jsonl"),
        "--scores",
        str(tmp_path / "scores.jsonl"),
        "--max-tokens",
        "64",
        "--output",
        str(tmp_path / "cache.jsonl"),
        "--temperature",
        "2",
    ]
    report = main(args)
    assert report["rows"] == 6 and report["skipped"] == 0 and report["scored"] == 6
    cache = read_teacher(tmp_path / "cache.jsonl", [*xs, unlabelled], 2.0)
    scores = read_scores(tmp_path / "scores.jsonl")
    for ex in [*xs, unlabelled]:
        row = cache[ex.id]
        assert row["example_sha256"] == example_fingerprint(ex)
        logits = scores[prompt_sha256(ex, p.tokenizer, p.metadata)]["logits"]
        z = [v / 2 for v in logits]
        top = max(z)
        weights = [math.exp(v - top) for v in z]
        assert math.isclose(row["none_probability"], weights[-1] / sum(weights), rel_tol=1e-12)
    first = (tmp_path / "cache.jsonl").read_bytes()
    # Every prompt is in the score file now: a second run scores nothing and writes the same cache.
    assert main(args)["scored"] == 0
    assert (tmp_path / "cache.jsonl").read_bytes() == first
    assert json.loads(first.splitlines()[0])["teacher_model"] == str(tmp_path / "teacher")
