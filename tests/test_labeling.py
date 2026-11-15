import json
import threading
from collections import Counter

import pytest

from wsd import Candidate, Example, Target
from wsd.data import read_jsonl, write_jsonl
from wsd.labeling import judge_request, label_dataset, validate_judgment


def example(eid="one"):
    return Example(
        eid,
        "She sat on the bank.",
        Target(15, 19, "bank"),
        [Candidate("river", "the land beside a river"), Candidate("finance", "a financial institution")],
        provenance={"source": "offline-fixture", "note": "private-review-note"},
    )


def jev_result(choice="river", confidence=0.8):
    return {
        "choice": choice,
        "selected_ids": [choice] if choice in {"river", "finance"} else [],
        "confidence": confidence,
        "provider": "typesafe",
        "model": "jev-resolved",
        "probabilities": {"river": 0.9, "finance": 0.05},
    }


def judgment(outcome="selected", selected_ids=None):
    return {
        "outcome": outcome,
        "selected_ids": selected_ids if selected_ids is not None else ["river"],
        "rationale": "The surrounding words support this meaning.",
    }


@pytest.fixture
def settings(monkeypatch):
    config = {
        "jev": {
            "provider": "typesafe",
            "model": "jev-test",
            "api_key": "private-api-key",
            "confidence_threshold": 0.8,
            "concurrency": 2,
            "max_retries": 0,
        },
        "codex": {"model": "test-local-model", "concurrency": 2, "max_retries": 0, "timeout_seconds": 12},
    }
    monkeypatch.setattr("wsd.labeling.load_config", lambda path: config)
    return config


def artifacts(output):
    rows = read_jsonl(output / "samples.jsonl")
    return {ex.id: ex for ex in rows}


def test_routing_decisions_and_silver_contract(tmp_path, monkeypatch, settings):
    cases = {
        "boundary": (jev_result(confidence=0.8), None, ["river"], "jev_high_confidence", "jev"),
        "agree": (jev_result(confidence=0.79), judgment(), ["river"], "jev_low_codex_agreed", "codex"),
        "disagree": (
            jev_result(confidence=0.3),
            judgment(selected_ids=["finance"]),
            ["finance"],
            "jev_low_codex_disagreed",
            "codex",
        ),
        "missing": (jev_result(confidence=None), judgment(), ["river"], "jev_low_codex_agreed", "codex"),
        "error": (None, judgment("none_fits", []), [], "jev_error_codex", "codex"),
        "jev_abstain": (jev_result("insufficient_context", 0.9), None, None, "unresolved", None),
        "codex_abstain": (jev_result(confidence=0.4), judgment("insufficient_context", []), None, "unresolved", None),
        "jev_none": (jev_result("none_fits", 0.95), None, [], "jev_high_confidence", "jev"),
    }
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    inputs = [example(eid) for eid in cases]
    # Give each row distinct visible context, without changing target offsets.
    for ex in inputs:
        ex.text += " " + ex.id
    write_jsonl(source, inputs)
    seen_jev, seen_codex = [], []

    def jev_call(ex, key, provider, model):
        assert (key, provider, model) == ("private-api-key", "typesafe", "jev-test")
        seen_jev.append(ex.id)
        if cases[ex.id][0] is None:
            raise RuntimeError("private-api-key private response")
        return cases[ex.id][0]

    def codex_call(prompt, schema, *, model, timeout):
        data = json.loads(prompt.split("\n")[-1])
        eid = data["text"].split()[-1]
        seen_codex.append(eid)
        assert model == "test-local-model" and timeout == 12
        assert "private-review-note" not in prompt
        assert "confidence" not in prompt and '"jev"' not in prompt
        assert schema["properties"]["selected_ids"]["items"]["enum"] == ["river", "finance"]
        return cases[eid][1]

    monkeypatch.setattr("wsd.labeling.jev.assist", jev_call)
    monkeypatch.setattr("wsd.labeling.run_codex", codex_call)
    report = label_dataset(source, output)
    assert set(seen_jev) == set(cases)
    assert set(seen_codex) == {"agree", "disagree", "missing", "error", "codex_abstain"}
    assert report["labeled"] == 6 and report["unresolved"] == 2
    assert report["jev_errors"] == 1
    assert report["jev_abstentions"] == report["codex_abstentions"] == 1
    rows = artifacts(output)
    for eid, (_, c, gold, bucket, by) in cases.items():
        ex = rows[eid]
        auto = ex.provenance["automatic"]
        assert ex.gold_ids == gold
        assert auto["status"] == ("labeled" if gold is not None else "unresolved")
        assert auto["bucket"] == bucket and auto["selected_by"] == by
        assert auto["codex"] == c
        assert auto["label_tier"] == "silver" and auto["threshold"] == 0.8
        assert ex.provenance["annotation"] == "silver" and ex.provenance["review_status"] == "unreviewed"
    assert rows["missing"].provenance["automatic"]["escalation_reason"] == "jev_missing_confidence"
    assert rows["error"].provenance["automatic"]["jev_error"]["type"] == "request_failed"
    assert rows["jev_abstain"].provenance["automatic"]["jev_error"] is None
    assert set(p.name for p in output.iterdir()) == {
        "samples.jsonl",
        "judgments.jsonl",
        "manifest.json",
        "labeling-report.json",
        "events.jsonl",
    }
    for path in output.iterdir():
        assert "private-api-key" not in path.read_text()


def test_judge_is_independent_and_validates_constrained_outcomes():
    ex = example("private-row-id")
    ex.gold_ids = ["finance"]
    ex.provenance["automatic"] = {"jev": {"choice": "finance", "confidence": 0.1}}
    prompt, _ = judge_request(ex)
    assert "private-row-id" not in prompt and "private-review-note" not in prompt
    assert "gold_ids" not in prompt and "confidence" not in prompt
    assert validate_judgment(judgment("none_fits", []), ex)["selected_ids"] == []
    for invalid in (
        judgment(selected_ids=["invented"]),
        judgment(selected_ids=[]),
        judgment(selected_ids=["river", "finance"]),
        judgment("none_fits", ["river"]),
        judgment("insufficient_context", ["river"]),
        {**judgment(), "rationale": ""},
        {**judgment(), "extra": "ignored?"},
    ):
        with pytest.raises(ValueError):
            validate_judgment(invalid, ex)


def test_resume_reroutes_threshold_without_repeating_calls(tmp_path, monkeypatch, settings):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    calls = Counter()

    def j(*args):
        calls["jev"] += 1
        return jev_result(confidence=0.7)

    def c(*args, **kwargs):
        calls["codex"] += 1
        return judgment(selected_ids=["finance"])

    monkeypatch.setattr("wsd.labeling.jev.assist", j)
    monkeypatch.setattr("wsd.labeling.run_codex", c)
    label_dataset(source, output)
    assert artifacts(output)["one"].gold_ids == ["finance"]
    label_dataset(source, output, resume=True, confidence_threshold=0.6)
    row = artifacts(output)["one"]
    assert row.gold_ids == ["river"] and row.provenance["automatic"]["codex"] == judgment(selected_ids=["finance"])
    label_dataset(source, output, resume=True, confidence_threshold=0.9)
    assert artifacts(output)["one"].gold_ids == ["finance"]
    assert calls == {"jev": 1, "codex": 1}
    # Keys are deliberately absent from fingerprints and artifacts.
    settings["jev"]["api_key"] = "new-private-api-key"
    label_dataset(source, output, resume=True)
    assert calls == {"jev": 1, "codex": 1}
    settings["codex"]["model"] = "other-local-model"
    label_dataset(source, output, resume=True)
    assert calls == {"jev": 1, "codex": 2}
    settings["jev"]["provider"] = "openrouter"
    label_dataset(source, output, resume=True)
    assert calls == {"jev": 2, "codex": 2}
    changed = example()
    changed.candidates[0].definition += " or stream"
    write_jsonl(source, [changed])
    label_dataset(source, output, resume=True)
    assert calls == {"jev": 3, "codex": 3}


def test_missing_confidence_is_not_probability_and_failed_calls_are_bounded(tmp_path, monkeypatch, settings, capsys):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    settings["jev"]["max_retries"] = 2
    settings["codex"]["max_retries"] = 1
    calls = Counter()

    def j(*args):
        calls["jev"] += 1
        raise ValueError("private-api-key provider-body-private")

    def c(*args, **kwargs):
        calls["codex"] += 1
        return judgment(selected_ids=["hallucinated"])

    monkeypatch.setattr("wsd.labeling.jev.assist", j)
    monkeypatch.setattr("wsd.labeling.run_codex", c)
    label_dataset(source, output)
    assert calls == {"jev": 3, "codex": 2}
    row = artifacts(output)["one"]
    assert row.gold_ids is None
    assert row.provenance["automatic"]["jev_error"]["attempts"] == 3
    assert row.provenance["automatic"]["codex_error"]["attempts"] == 2
    label_dataset(source, output, resume=True)
    assert calls == {"jev": 3, "codex": 2}
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result(confidence=None))
    monkeypatch.setattr("wsd.labeling.run_codex", lambda *args, **kwargs: judgment())
    label_dataset(source, output, resume=True, retry_failed=True)
    row = artifacts(output)["one"]
    assert row.gold_ids == ["river"] and row.provenance["automatic"]["selected_by"] == "codex"
    assert row.provenance["automatic"]["jev_error"] is None
    assert row.provenance["automatic"]["codex_error"] is None
    captured = capsys.readouterr()
    assert "private-api-key" not in captured.err and "provider-body-private" not in captured.err


def test_checkpoint_recovers_completed_jev_after_interruption(tmp_path, monkeypatch, settings):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    calls = Counter()

    def j(*args):
        calls["jev"] += 1
        return jev_result(confidence=0.3)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("wsd.labeling.jev.assist", j)
    monkeypatch.setattr("wsd.labeling.run_codex", interrupted)
    with pytest.raises(KeyboardInterrupt):
        label_dataset(source, output)
    assert json.loads((output / "manifest.json").read_text())["complete"] is False
    checkpoint = json.loads((output / "judgments.jsonl").read_text())
    assert checkpoint["jev"] == jev_result(confidence=0.3)
    monkeypatch.setattr("wsd.labeling.run_codex", lambda *args, **kwargs: judgment())
    report = label_dataset(source, output, resume=True)
    assert calls["jev"] == 1 and report["labeled"] == 1 and report["complete"]


def test_input_limit_default_output_and_no_accidental_overwrite(tmp_path, monkeypatch, settings):
    source = tmp_path / "samples.jsonl"
    write_jsonl(source, [example("one"), example("two")])
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result())
    monkeypatch.setattr(
        "wsd.labeling.run_codex", lambda *args, **kwargs: pytest.fail("high confidence must skip Codex")
    )
    report = label_dataset(source, limit=1)
    assert report["total"] == 1 and list(artifacts(tmp_path / "labeled")) == ["one"]
    with pytest.raises(ValueError, match="exists"):
        label_dataset(source)
    with pytest.raises(ValueError, match="threshold"):
        label_dataset(source, confidence_threshold=float("nan"), resume=True)
    with pytest.raises(ValueError, match="requires"):
        label_dataset(source, tmp_path / "new", retry_failed=True)


def test_credentials_preflight_and_input_overwrite_guard(tmp_path, monkeypatch, settings):
    source = tmp_path / "samples.jsonl"
    write_jsonl(source, [example()])
    settings["jev"]["api_key"] = ""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: pytest.fail("missing key must fail before Jev"))
    monkeypatch.setattr(
        "wsd.labeling.run_codex", lambda *args, **kwargs: pytest.fail("missing key must not cascade to Codex")
    )
    with pytest.raises(ValueError, match="API key is missing"):
        label_dataset(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="Input cannot"):
        label_dataset(source, source.parent, resume=True)
    assert source.read_bytes() == original


def test_new_threshold_escalates_once_and_resume_can_use_cached_jev_without_key(tmp_path, monkeypatch, settings):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    calls = Counter()

    def j(*args):
        calls["jev"] += 1
        return jev_result(confidence=0.85)

    def c(*args, **kwargs):
        calls["codex"] += 1
        return judgment()

    monkeypatch.setattr("wsd.labeling.jev.assist", j)
    monkeypatch.setattr("wsd.labeling.run_codex", c)
    label_dataset(source, output)
    assert calls == {"jev": 1}
    settings["jev"]["api_key"] = ""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    label_dataset(source, output, resume=True, confidence_threshold=0.9)
    assert calls == {"jev": 1, "codex": 1}
    label_dataset(source, output, resume=True, confidence_threshold=0.95)
    assert calls == {"jev": 1, "codex": 1}
    monkeypatch.setattr("wsd.labeling.JUDGE_PROMPT_VERSION", "changed-prompt")
    label_dataset(source, output, resume=True, confidence_threshold=0.95)
    assert calls == {"jev": 1, "codex": 2}


def test_old_human_approval_is_archived_before_relabeling(tmp_path, monkeypatch, settings):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    ex = example()
    ex.gold_ids = ["finance"]
    ex.provenance.update(
        annotation="human",
        label_tier="gold",
        review_count=1,
        annotation_protocol="automatic-first-v1",
        review_status="approved",
        proposed_gold_ids=["finance"],
        review={"status": "approved", "gold_ids": ["finance"]},
    )
    write_jsonl(source, [ex])
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result())
    label_dataset(source, output)
    row = artifacts(output)["one"]
    assert row.gold_ids == ["river"] and row.provenance["review_status"] == "unreviewed"
    assert "review" not in row.provenance and "proposed_gold_ids" not in row.provenance
    assert row.provenance["previous_review"]["review"]["status"] == "approved"
    assert row.provenance["previous_review"]["gold_ids"] == ["finance"]
    assert row.provenance["label_tier"] == "silver"
    assert "review_count" not in row.provenance and "annotation_protocol" not in row.provenance


def test_provider_pools_bound_concurrency_and_jev_finishes_first(tmp_path, monkeypatch, settings):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example(str(i)) for i in range(4)])
    lock, barrier = threading.Lock(), threading.Barrier(2, timeout=5)
    active, maximum, completed = Counter(), Counter(), Counter()

    def call(stage, result):
        with lock:
            active[stage] += 1
            maximum[stage] = max(maximum[stage], active[stage])
            if stage == "codex":
                assert completed["jev"] == 4
        barrier.wait()
        with lock:
            active[stage] -= 1
            completed[stage] += 1
        return result

    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: call("jev", jev_result(confidence=0.4)))
    monkeypatch.setattr("wsd.labeling.run_codex", lambda *args, **kwargs: call("codex", judgment()))
    report = label_dataset(source, output)
    assert maximum == {"jev": 2, "codex": 2} and report["labeled"] == 4
    assert report["provider_attempts"] == {"jev": 4, "codex": 4}
    assert report["agreement"] == {"agreed": 4}
    assert report["confidence_histogram"] == {"0.40-0.60": 4}


def test_operational_errors_are_processed_but_incomplete_and_codex_diagnostic_is_safe(tmp_path, monkeypatch, settings):
    from wsd.pipeline import CodexError

    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result(confidence=0.4))

    def failed(*args, **kwargs):
        raise CodexError("Codex CLI is missing; install it and run codex login")

    monkeypatch.setattr("wsd.labeling.run_codex", failed)
    report = label_dataset(source, output)
    assert report["processed"] is True and report["complete"] is False
    assert report["status"] == "completed_with_errors" and report["unresolved"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["processed"] is True and manifest["complete"] is False
    error = artifacts(output)["one"].provenance["automatic"]["codex_error"]
    assert error["type"] == "codex_error" and "codex login" in error["message"]
    monkeypatch.setattr("wsd.labeling.run_codex", lambda *a, **kw: judgment())
    cached = label_dataset(source, output, resume=True)
    assert cached["status"] == "completed_with_errors"  # Failed calls require explicit retry.
    recovered = label_dataset(source, output, resume=True, retry_failed=True)
    assert recovered["processed"] and recovered["complete"] and recovered["status"] == "complete"


def test_jev_error_rescued_by_codex_still_has_operational_error(tmp_path, monkeypatch, settings):
    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])

    def failed(*args):
        raise RuntimeError("private-api-key arbitrary provider error body")

    monkeypatch.setattr("wsd.labeling.jev.assist", failed)
    monkeypatch.setattr("wsd.labeling.run_codex", lambda *a, **kw: judgment())
    report = label_dataset(source, output)
    assert report["labeled"] == 1 and report["processed"] is True
    assert report["complete"] is False and report["status"] == "completed_with_errors"
    for path in output.iterdir():
        assert "private-api-key" not in path.read_text() and "arbitrary provider error body" not in path.read_text()


def test_successful_abstention_is_complete_and_cli_failures_exit_nonzero(tmp_path, monkeypatch, settings):
    from wsd import labeling

    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result("insufficient_context", 0.95))
    report = label_dataset(source, output)
    assert report["unresolved"] == 1 and report["processed"] and report["complete"]
    assert report["status"] == "complete"
    monkeypatch.setattr(
        labeling, "label_dataset", lambda **kwargs: {"complete": False, "status": "completed_with_errors"}
    )
    with pytest.raises(SystemExit) as exit_info:
        labeling.main([])
    assert exit_info.value.code == 1


def test_interrupt_cancels_queued_provider_calls_without_waiting_for_inflight(tmp_path, monkeypatch, settings):
    import time

    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example(str(i)) for i in range(8)])
    settings["codex"]["concurrency"] = 1
    release, finished = threading.Event(), threading.Event()
    calls = []
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result(confidence=0.4))

    def interrupted(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise KeyboardInterrupt()
        release.wait(timeout=4)
        finished.set()
        return judgment()

    monkeypatch.setattr("wsd.labeling.run_codex", interrupted)
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            label_dataset(source, output)
        assert time.monotonic() - started < 2
        assert len(calls) <= 2  # At most one queued call raced with cancellation.
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["complete"] is False and manifest["processed"] is False
    finally:
        release.set()
        if len(calls) > 1:
            assert finished.wait(timeout=2)


def test_threshold_change_keeps_inactive_codex_failure_without_blocking_completion(tmp_path, monkeypatch, settings):
    from wsd.pipeline import CodexError

    source, output = tmp_path / "input.jsonl", tmp_path / "out"
    write_jsonl(source, [example()])
    monkeypatch.setattr("wsd.labeling.jev.assist", lambda *args: jev_result(confidence=0.7))

    def failed(*args, **kwargs):
        raise CodexError("Codex CLI is missing; install it and run codex login")

    monkeypatch.setattr("wsd.labeling.run_codex", failed)
    report = label_dataset(source, output)
    assert not report["complete"] and report["active_codex_errors"] == 1
    monkeypatch.setattr("wsd.labeling.run_codex", lambda *a, **k: pytest.fail("Lowered threshold needs no Codex"))
    report = label_dataset(source, output, resume=True, retry_failed=True, confidence_threshold=0.6)
    assert report["complete"] and report["processed"] and report["status"] == "complete"
    assert report["codex_errors"] == report["inactive_codex_errors"] == 1
    assert report["active_codex_errors"] == 0
    row = artifacts(output)["one"]
    assert row.gold_ids == ["river"] and row.provenance["automatic"]["codex_error"] is not None
