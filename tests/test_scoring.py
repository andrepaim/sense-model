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


def score_cli(*args):
    from wsd.scoring import main

    return main([str(a) for a in args])


def saved(tmp_path, name, dtype=None, max_length=None):
    p = tiny_predictor()
    p.metadata["model_revision"] = f"rev-{name}"
    if dtype is not None:
        p.model.to(dtype)
    if max_length is not None:
        p.metadata["max_length"] = max_length
    p.save_pretrained(tmp_path / name)
    return tmp_path / name


def test_score_rows_record_the_model_and_a_mismatch_is_refused(tmp_path):
    import pytest

    a, b = saved(tmp_path, "a"), saved(tmp_path, "b")
    write_jsonl(tmp_path / "in.jsonl", examples())
    out = tmp_path / "scores.jsonl"
    score_cli("--model", a, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", out)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["model"] == str(a) and row["dtype"] == "float32" and row["max_length"] == 256 and "revision" in row
    for other in (["--model", b], ["--model", a, "--dtype", "bfloat16"]):
        with pytest.raises(ValueError, match="scored by another model"):
            score_cli(*other, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", out)
    shorter = saved(tmp_path, "short", max_length=128)
    with pytest.raises(ValueError, match="scored by another model"):
        score_cli("--model", shorter, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", out)


def test_rows_without_a_recorded_model_need_an_explicit_identity(tmp_path):
    import pytest

    a = saved(tmp_path, "a")
    write_jsonl(tmp_path / "in.jsonl", examples())
    out = tmp_path / "scores.jsonl"
    score_cli("--model", a, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", out)
    old = [json.loads(line) for line in out.read_text().splitlines()]
    identity = {k: old[0][k] for k in ("model", "revision", "dtype", "max_length")}
    out.write_text("".join(json.dumps({k: v for k, v in r.items() if k not in identity}) + "\n" for r in old[:3]))
    args = ["--model", a, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", out]
    with pytest.raises(ValueError, match="no recorded model"):
        score_cli(*args)
    with pytest.raises(ValueError, match="scored by another model"):
        score_cli(*args, "--untagged-scores", "other", identity["revision"], "float32", "256")
    report = score_cli(*args, "--untagged-scores", a, identity["revision"], "float32", "256")
    assert report["scored"] == 2 and report["total"] == 5


def test_output_paths_must_not_clobber_inputs(tmp_path):
    import pytest

    from wsd.teacher import main as teacher

    a = saved(tmp_path, "a")
    write_jsonl(tmp_path / "in.jsonl", examples())
    before = (tmp_path / "in.jsonl").read_bytes()
    with pytest.raises(SystemExit):
        score_cli(
            "--model", a, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", tmp_path / "in.jsonl"
        )
    for extra in (
        ["--scores", tmp_path / "x.jsonl", "--output", tmp_path / "x.jsonl"],
        ["--output", tmp_path / "in.jsonl"],
    ):
        with pytest.raises(SystemExit):
            teacher(["--model", str(a), "--local-files-only", "--input", str(tmp_path / "in.jsonl"), *map(str, extra)])
    (tmp_path / "copy.jsonl").write_bytes(before)
    with pytest.raises(ValueError, match="not a score file"):
        score_cli(
            "--model", a, "--local-files-only", "--input", tmp_path / "in.jsonl", "--output", tmp_path / "copy.jsonl"
        )
    assert (tmp_path / "in.jsonl").read_bytes() == before == (tmp_path / "copy.jsonl").read_bytes()


def test_teacher_keeps_the_stored_dtype_and_refuses_too_long_prompts_unless_asked(tmp_path):
    import pytest

    from wsd.teacher import main as teacher

    stored = saved(tmp_path, "bf16", dtype=torch.bfloat16)
    write_jsonl(tmp_path / "in.jsonl", examples())
    base = ["--model", str(stored), "--local-files-only", "--input", str(tmp_path / "in.jsonl")]
    teacher([*base, "--scores", str(tmp_path / "s.jsonl"), "--output", str(tmp_path / "c.jsonl")])
    assert {json.loads(line)["dtype"] for line in (tmp_path / "s.jsonl").read_text().splitlines()} == {"bfloat16"}
    teacher([*base, "--dtype", "float32", "--scores", str(tmp_path / "f.jsonl"), "--output", str(tmp_path / "c.jsonl")])
    assert {json.loads(line)["dtype"] for line in (tmp_path / "f.jsonl").read_text().splitlines()} == {"float32"}
    p = tiny_predictor()
    short = examples()[0]
    tight = len(p.tokenizer(format_prompt(short, p.tokenizer, p.metadata))["input_ids"])
    small = saved(tmp_path, "small", max_length=tight)
    args = ["--model", str(small), "--local-files-only", "--input", str(tmp_path / "in.jsonl")]
    with pytest.raises(ValueError, match="max_length"):
        teacher([*args, "--output", str(tmp_path / "t.jsonl")])
    report = teacher([*args, "--output", str(tmp_path / "t.jsonl"), "--skip-too-long"])
    assert report["too_long"] == 4 and report["rows"] == 1


def test_teacher_refuses_scores_of_another_model(tmp_path):
    import pytest

    from wsd.teacher import main as teacher

    a, b = saved(tmp_path, "a"), saved(tmp_path, "b")
    write_jsonl(tmp_path / "in.jsonl", examples())
    teacher(["--model", str(a), "--local-files-only", "--input", str(tmp_path / "in.jsonl"),
             "--scores", str(tmp_path / "s.jsonl"), "--output", str(tmp_path / "c.jsonl")])  # fmt: skip
    with pytest.raises(ValueError, match="scored by another model"):
        teacher(["--model", str(b), "--local-files-only", "--input", str(tmp_path / "in.jsonl"),
                 "--scores", str(tmp_path / "s.jsonl"), "--output", str(tmp_path / "c.jsonl")])  # fmt: skip


def test_every_prompt_of_a_batch_needs_exactly_one_mask(tmp_path):
    p = tiny_predictor()
    real = p.tokenizer

    class TwoMasks:
        """The tokenizer, with a second mask in the ids of the longest prompt (no Example can produce one)."""

        def __getattr__(self, name):
            return getattr(real, name)

        def __call__(self, text, *args, **kwargs):
            out = real(text, *args, **kwargs)
            if text.startswith("*bank* " + "bank " * 10):
                out = {**out, "input_ids": [*out["input_ids"], real.mask_token_id]}
            return out

    p.tokenizer = TwoMasks()
    rows = score_examples(p, examples(), None, max_batch=64)
    kinds = {r["hash"]: (r["type"], r["code"]) for r in rows}
    assert sorted(kinds.values()) == [("rejected", "MASK")] + [("scored", None)] * 4
