import math
from types import SimpleNamespace

import pytest

from wsd.evaluate import apply_calibration, fit_calibration, paired_bootstrap, summarize


def example(id="a", gold=None, language="en", candidates=2):
    return SimpleNamespace(
        id=id,
        candidates=[SimpleNamespace(id=f"c{i}") for i in range(candidates)],
        gold_ids=["c0"] if gold is None else gold,
        language=language,
        target=SimpleNamespace(text="banco"),
        provenance={"domain": "prose"},
    )


def prediction(id="a", probs=(0.8, 0.1), none=0.1, selected="c0", reason="selected"):
    return {
        "example_id": id,
        "probabilities": {f"c{i}": p for i, p in enumerate(probs)},
        "none_probability": none,
        "selected_id": selected,
        "reason": reason,
    }


def test_metrics_separate_absence_uncertainty_and_precision():
    examples = [example(), example("b", ["c1"], "pt-BR"), example("c", []), example("d", [], candidates=0)]
    predictions = [
        prediction(),
        prediction("b", (0.4, 0.5), selected=None, reason="uncertain"),
        prediction("c"),
        prediction("d", (), 1, None, "no_candidates"),
    ]
    report = summarize(examples, predictions, 100)
    assert report["top1_accuracy"] == 0.75
    assert report["first_candidate_accuracy"] == 0.5
    assert report["marked_precision"] == 0.5
    assert report["coverage"] == 0.5
    assert report["false_confident_none_rate"] == 0.5
    assert report["ambiguous_coverage"] == 0.5
    assert report["ambiguous_top1_accuracy"] == 1
    assert report["by_language"]["pt-BR"]["count"] == 1
    assert report["risk_coverage"][-1]["risk"] == 0.25
    assert report["calibration"]["ece"] >= 0


def test_none_fit_prediction_is_not_a_marked_meaning():
    report = summarize(
        [example(gold=[])], [prediction(probs=(0.1, 0.1), none=0.8, selected=None, reason="none_fits")], 10
    )
    assert report["none_fits_recall"] == 1
    assert report["marked_precision"] is None
    assert report["top1_accuracy"] == 1


def test_temperature_thresholds_and_margin_apply_to_none_as_well():
    source = [prediction(probs=(0.4, 0.1), none=0.5, selected=None, reason="none_fits")]
    result = apply_calibration(source, {"temperature": 2, "min_probability": 0.6})
    assert result[0]["reason"] == "uncertain"
    assert source[0]["none_probability"] == 0.5
    assert math.isclose(sum(result[0]["probabilities"].values()) + result[0]["none_probability"], 1)
    assert apply_calibration(source, {"min_margin": 0.2})[0]["reason"] == "uncertain"
    assert apply_calibration(source, {})[0]["reason"] == "none_fits"


def test_calibrate_uses_labeled_examples_and_can_abstain_wrong_ones():
    examples = [example("a"), example("b", ["c1"])]
    predictions = [prediction("a", (0.95, 0.025), 0.025), prediction("b", (0.45, 0.3), 0.25)]
    fitted = fit_calibration(examples, predictions, 1)
    result = apply_calibration(predictions, fitted)
    assert fitted["feasible"]
    assert result[0]["selected_id"] == "c0"
    assert result[1]["reason"] == "uncertain"
    assert fitted["calibration_count"] == 2


def test_no_feasible_calibration_is_recorded():
    fitted = fit_calibration([example(gold=["c1"])], [prediction()], 1)
    assert fitted["feasible"] is False
    assert apply_calibration([prediction()], fitted)[0]["reason"] == "uncertain"


@pytest.mark.parametrize(
    "change",
    [
        {"example_id": "wrong"},
        {"probabilities": {"c0": 0.8}},
        {"none_probability": float("nan")},
        {"none_probability": 0.8},
        {"reason": "other"},
        {"selected_id": "unknown"},
    ],
)
def test_malformed_predictions_fail(change):
    row = prediction()
    row.update(change)
    with pytest.raises(ValueError):
        summarize([example()], [row], 10)


def test_unlabeled_and_count_mismatch_fail():
    ex = example()
    ex.gold_ids = None
    with pytest.raises(ValueError, match="gold_ids"):
        summarize([ex], [prediction()], 10)
    with pytest.raises(ValueError, match="count"):
        summarize([example()], [], 10)


def test_bootstrap_is_paired_and_reproducible():
    a = paired_bootstrap([True, True, True], [False, False, False], 100)
    assert a["ci95"] == [1, 1]
    assert a == paired_bootstrap([True, True, True], [False, False, False], 100)
    assert paired_bootstrap([], [], 10)["difference"] is None


def test_infeasible_fit_abstains_even_with_exact_one_probability():
    raw = [prediction(probs=(1.0, 0.0), none=0.0)]
    fitted = fit_calibration([example(gold=["c1"])], raw, 1)
    assert fitted["abstain_all"]
    assert apply_calibration(raw, fitted)[0]["reason"] == "uncertain"


def test_release_gate_requires_all_evidence():
    from wsd.evaluate import release_gates

    assert not release_gates({})["passed"]
    report = {
        "ambiguous_paired_improvement": {"difference": 0.1, "ci95": [0.05, 0.15]},
        "ambiguous_marked_precision": 0.95,
        "ambiguous_coverage": 0.5,
        "false_confident_none_rate": 0.05,
        "latency": {"p95": 999},
    }
    assert release_gates(report)["passed"]
    report["latency"]["p95"] = 1000
    assert not release_gates(report)["passed"]
