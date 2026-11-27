"""The ranking rule of training round 1 (wsd rank): the labelled meaning's rank within its entry when the model
reorders the meanings it was offered, dictionary order otherwise; paired bootstrap between score files."""

import json
import math

import pytest

from wsd.evaluate import paired_bootstrap
from wsd.ranking import STRATA, main, offered_accuracy, outcome, summary


def item(id, sense, senses, candidates, prompt="p1", meanings=None, **info):
    return {
        "id": id,
        "labelled": {"entry": 0, "sense": sense},
        "entrySenses": senses,
        "meanings": meanings or senses[0],
        "perCap": {"24|c1600": {"prompt": prompt, "candidates": candidates, **info}},
    }


def scored(logits, hash="p1"):
    return {"hash": hash, "type": "scored", "code": None, "logits": logits}


def test_outcome_reorders_offered_meanings_of_the_labelled_entry():
    it = item("a", 2, [4, 2], [[0, 0], [1, 0], [0, 2], [0, 1]])
    # Offered: entry 0 senses 0, 2, 1 and entry 1 sense 0; then none-fits last.
    assert outcome(it, "24|c1600", {"p1": scored([0.0, 5.0, 3.0, 1.0, -9.0])}) == (1, "applied")
    assert outcome(it, "24|c1600", {"p1": scored([3.0, 5.0, 1.0, 2.0, -9.0])}) == (3, "applied")
    # Ties keep dictionary order.
    assert outcome(it, "24|c1600", {"p1": scored([1.0, 0.0, 1.0, 1.0, -9.0])}) == (3, "applied")


def test_outcome_falls_back_to_dictionary_order():
    it = item("a", 2, [4], [[0, 0], [0, 2]])
    assert outcome(it, "24|c1600", {"p1": scored([1.0, 0.0, 1.0])}) == (3, "none")
    assert outcome(it, "24|c1600", {}) == (3, "missing")
    assert outcome(it, "24|c1600", {"p1": {"hash": "p1", "type": "rejected", "code": "TOO_LONG"}}) == (
        3,
        "rejected:TOO_LONG",
    )
    assert outcome(item("b", 1, [3], [], skip="not English"), "24|c1600", {}) == (2, "skipped")
    # The labelled meaning not offered: after the offered meanings of its entry, in dictionary order.
    it = item("c", 1, [4], [[0, 3], [0, 0]])
    assert outcome(it, "24|c1600", {"p1": scored([2.0, 1.0, -9.0])}) == (3, "applied")


def test_summary_and_offered_accuracy():
    assert summary([1, 2, 4]) == (pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(7 / 3))
    items = [
        item("a", 1, [3], [[0, 0], [0, 1], [0, 2]], labelledOffered=True),
        item("b", 0, [3], [[0, 0], [0, 1]], prompt="p2", labelledOffered=True),
        item("c", 2, [3], [[0, 0]], prompt="p3", labelledOffered=False),
    ]
    scores = {"p1": scored([0.0, 3.0, 1.0, 5.0]), "p2": scored([0.0, 2.0, -9.0], "p2")}
    first, top3, n = offered_accuracy(items, "24|c1600", scores)
    # a: none-fits wins (not first) but is in the top three offered; b: second.
    assert (first, top3, n) == (0.0, 1.0, 2)
    assert STRATA[0] == (2, 2) and STRATA[-1] == (33, math.inf)


def test_paired_bootstrap_is_deterministic_and_brackets_the_difference():
    a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    b = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    result = paired_bootstrap(a, b, 2000, seed=1)
    assert result == paired_bootstrap(a, b, 2000, seed=1)
    assert result["difference"] == pytest.approx(0.4)
    low, high = result["ci95"]
    assert low <= 0.4 <= high and low < high
    assert paired_bootstrap(a, a, 200, seed=1)["ci95"] == [0, 0]


def test_rank_command_reports_forms_models_and_intervals(tmp_path, capsys):
    items = [item(f"i{n}", n % 2, [2], [[0, 0], [0, 1]], prompt=f"p{n}", labelledOffered=True) for n in range(8)]
    items.append(item("gone", 0, [2], [[0, 0], [0, 1]], prompt="px", labelledOffered=True))
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    (tmp_path / "exclude.txt").write_text("gone\n")
    right = [scored([5.0, 0.0, -9.0] if n % 2 == 0 else [0.0, 5.0, -9.0], f"p{n}") for n in range(8)]
    wrong = [scored([0.0, 5.0, -9.0], f"p{n}") for n in range(8)]
    for name, rows in (("good", right), ("base", wrong)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = main(
        [
            "--items",
            str(tmp_path / "items.jsonl"),
            "--exclude",
            str(tmp_path / "exclude.txt"),
            "--forms",
            "24|c1600",
            "--scores",
            f"base={tmp_path / 'base.jsonl'}",
            f"good={tmp_path / 'good.jsonl'}",
            "--baseline",
            "base",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert report["items"] == 8 and report["excluded"] == 1
    assert report["dictionary"]["first"] == 0.5
    form = report["forms"]["24|c1600"]
    assert form["good"]["first"] == 1.0 and form["base"]["first"] == 0.5
    assert form["good"]["offered"]["first"] == 1.0 and form["good"]["by_meanings"]["2"] == {"first": 1.0, "items": 8}
    assert form["good"]["vs_baseline"]["difference"] == 0.5 and "vs_baseline" not in form["base"]
    assert json.loads((tmp_path / "report.json").read_text()) == report
    printed = capsys.readouterr().out
    assert "| good | 100.0% |" in printed and "+50.0" in printed


def test_paired_bootstrap_reproduces_the_published_procedure():
    # Values from compare.py of training round 1 (contextual-dictionary), the procedure of its published intervals.
    a = [int((i * 7) % 3 != 0) for i in range(57)]
    b = [int((i * 5) % 4 == 0) for i in range(57)]
    assert paired_bootstrap(a, b, 2000, seed=1)["ci95"] == [0.24561403508771928, 0.5614035087719298]
    assert paired_bootstrap(a, b, 500, seed=7)["ci95"] == [0.24561403508771928, 0.5789473684210527]


def test_a_score_that_does_not_fit_its_prompt_is_reported():
    it = item("bad", 2, [4], [[0, 0], [0, 2]])
    with pytest.raises(ValueError, match="bad.*2 logits for 2 candidates"):
        outcome(it, "24|c1600", {"p1": scored([1.0, 2.0])})
    with pytest.raises(ValueError, match="bad.*not finite"):
        outcome(it, "24|c1600", {"p1": scored([float("nan"), 1.0, 0.0])})


def test_ranking_no_items_is_an_error():
    from wsd.ranking import rank_report

    with pytest.raises(ValueError, match="No items"):
        rank_report([], ["24|c1600"], {"m": {}})
