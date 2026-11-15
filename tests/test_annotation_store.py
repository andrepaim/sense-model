import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from wsd.annotation_store import Store


def row(eid="a"):
    return {
        "id": eid,
        "text": "Ela sentou no banco.",
        "target": {"start": 14, "end": 19, "text": "banco"},
        "candidates": [
            {"id": "seat", "definition": "Assento para várias pessoas."},
            {"id": "financial", "definition": "Instituição financeira."},
        ],
        "language": "pt-BR",
        "definition_language": "pt-BR",
        "gold_ids": ["seat"],
        "provenance": {
            "source": "pilot source",
            "document_id": eid,
            "source_url": "https://example.org/" + eid,
            "revision": "1",
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "domain": "general",
            "lemma": "banco",
        },
    }


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "workspace.sqlite")
    s.import_examples([row()])
    return s


def vote(s, reviewer, outcome="selected", gold=None, note=""):
    d = s.get_example("a", reviewer)
    return s.vote("a", reviewer, outcome, (["seat"] if gold is None else gold), note, d["revision"])


def assistance():
    return {
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "prompt_version": "v1",
        "request_fingerprint": "x",
        "choice": "seat",
        "selected_ids": ["seat"],
        "probabilities": {"seat": 0.8, "financial": 0.1},
        "none_probability": 0.05,
        "insufficient_probability": 0.05,
        "confidence": 0.8,
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "created_at": "2026-11-15T00:00:00Z",
    }


def test_import_strips_gold_and_rejects_changed_identity_atomically(store):
    detail = store.get_example("a", "alice")
    assert "gold_ids" not in detail["example"]
    assert detail["my_vote"] is None and "reviews" not in detail
    assert store.import_examples([row()]) == {"imported": 0, "existing": 1}
    changed = row()
    changed["text"] = "bad"
    with pytest.raises(ValueError):
        store.import_examples([row("new"), changed])
    assert store.stats()["total"] == 1
    changed = row()
    changed["candidates"][0]["definition"] = "different"
    with pytest.raises(ValueError, match="Changed"):
        store.import_examples([changed])


def test_independent_votes_and_ai_visibility(store):
    with pytest.raises(PermissionError):
        store.record_assistance("a", "alice", assistance())
    alice = vote(store, "alice")
    assert alice["my_vote"]["initial"]["gold_ids"] == ["seat"]
    store.record_assistance("a", "alice", assistance())
    bob = store.get_example("a", "bob")
    assert bob["my_vote"] is None and bob["assistance"] == []
    assert "adjudication" not in bob and "reviews" not in bob
    with pytest.raises(PermissionError):
        store.get_example("a", "bob", adjudicate=True)
    vote(store, "alice", gold=["financial"])
    alice = store.get_example("a", "alice")
    assert alice["my_vote"]["initial"]["gold_ids"] == ["seat"]
    assert alice["my_vote"]["gold_ids"] == ["financial"]
    assert alice["my_vote"]["ai_exposed"]
    assert len(alice["assistance"]) == 1
    assert store.stats()["answers_changed_after_ai"] == 1


def test_candidate_order_stable_and_canonical_untouched(store):
    a = store.get_example("a", "alice")
    assert a["candidate_order"] == store.get_example("a", "alice")["candidate_order"]
    assert [c["id"] for c in a["example"]["candidates"]] == ["seat", "financial"]
    assert set(a["candidate_order"]) == {"seat", "financial"}


def test_optimistic_conflicts_and_concurrent_writes(store):
    initial = store.get_example("a", "alice")["revision"]
    vote(store, "alice")
    with pytest.raises(RuntimeError):
        store.vote("a", "bob", "selected", ["seat"], expected_revision=initial)

    def save(name):
        try:
            store.vote("a", name, "selected", ["seat"], expected_revision=1)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(save, ["bob", "carol"])) == 1


def test_adjudication_requires_two_votes_and_note_for_disagreement(store):
    a = vote(store, "alice")
    with pytest.raises(PermissionError):
        store.adjudicate("a", "alice", "selected", ["seat"], expected_revision=a["revision"])
    b = vote(store, "bob", gold=["financial"])
    with pytest.raises(ValueError, match="Explain"):
        store.adjudicate("a", "alice", "selected", ["seat"], expected_revision=b["revision"])
    final = store.adjudicate("a", "alice", "selected", ["seat"], "Context indicates sitting, not money.", b["revision"])
    assert final["adjudication"]["gold_ids"] == ["seat"]
    assert len(final["reviews"]) == 2
    exported = store.export_gold()
    assert exported["examples"][0]["gold_ids"] == ["seat"]
    assert exported["examples"][0]["provenance"]["review_count"] == 2
    assert exported["manifest"]["count"] == 1
    assert store.stats()["initial_agreement"] == 0
    assert "adjudication" not in store.get_example("a", "bob")
    vote(store, "bob")
    assert store.export_gold()["examples"] == []


def test_agreement_export_multiple_gold_and_none_fit(store):
    vote(store, "alice", gold=["seat", "financial"])
    d = vote(store, "bob", gold=["financial", "seat"])
    with pytest.raises(ValueError):
        store.adjudicate("a", "alice", "selected", ["seat", "financial"], expected_revision=d["revision"])
    store.adjudicate("a", "alice", "selected", ["seat", "financial"], "Both readings acceptable.", d["revision"])
    assert set(store.export_gold()["examples"][0]["gold_ids"]) == {"seat", "financial"}
    vote(store, "alice", "none_fits", [])
    d = vote(store, "bob", "none_fits", [])
    store.adjudicate("a", "alice", "none_fits", [], expected_revision=d["revision"])
    assert store.export_gold()["examples"][0]["gold_ids"] == []


def test_flags_never_become_gold_implicitly(store):
    vote(store, "alice", "insufficient_context", [])
    d = vote(store, "bob", "insufficient_context", [])
    assert store.export_gold()["examples"] == []
    with pytest.raises(ValueError):
        store.adjudicate("a", "alice", "insufficient_context", [], expected_revision=d["revision"])
    with pytest.raises(ValueError):
        store.adjudicate("a", "alice", "none_fits", [], expected_revision=d["revision"])


def test_demo_excluded_and_provenance_required(tmp_path):
    s = Store(tmp_path / "s.sqlite")
    data = row()
    data["provenance"] = {}
    with pytest.raises(ValueError, match="provenance"):
        s.import_examples([data])
    s.import_examples([data], demo=True)
    vote(s, "alice")
    d = vote(s, "bob")
    s.adjudicate("a", "alice", "selected", ["seat"], expected_revision=d["revision"])
    assert s.export_gold()["examples"] == []
    assert s.export_gold(include_demo=True)["examples"][0]["provenance"]["annotation"] == "synthetic"
    assert s.stats()["by_language"]["pt-BR"]["accepted"] == 0


def test_backup_contains_history_but_no_credentials(store):
    vote(store, "alice")
    store.record_assistance("a", "alice", assistance())
    value = store.backup()
    assert len(value["events"]) == 3
    assert value["votes"][0]["ai_exposed"] == 1
    bad = assistance()
    bad["api_key"] = "secret"
    with pytest.raises(ValueError):
        store.record_assistance("a", "alice", bad)
    assert "secret" not in json.dumps(store.backup())
    assert store.stats()["usage"]["input_tokens"] == 100


@pytest.mark.parametrize(
    "outcome,gold", [("selected", []), ("none_fits", ["seat"]), ("bogus", []), ("selected", ["missing"])]
)
def test_invalid_votes_rejected(store, outcome, gold):
    with pytest.raises(ValueError):
        vote(store, "alice", outcome, gold)


def test_cache_hits_are_exposure_not_additional_paid_requests(store):
    vote(store, "alice")
    store.record_assistance("a", "alice", dict(assistance(), cache_hit=False, requested_model="jev-latest"))
    store.record_assistance("a", "alice", dict(assistance(), cache_hit=True, requested_model="jev-latest"))
    assert store.stats()["ai_requests"] == 1
    assert store.stats()["usage"]["input_tokens"] == 100
    assert len(store.get_example("a", "alice")["assistance"]) == 2
    with pytest.raises(ValueError, match="cache_hit"):
        store.record_assistance("a", "alice", dict(assistance(), cache_hit="true"))


def test_disclosure_guard_prevents_backup_and_export_unblinding(store):
    vote(store, "alice")
    with pytest.raises(PermissionError):
        store.assert_disclosure_allowed("bob", backup=True)
    assert store.assert_disclosure_allowed("alice", backup=True)
    d = vote(store, "bob")
    store.adjudicate("a", "alice", "selected", ["seat"], expected_revision=d["revision"])
    with pytest.raises(PermissionError):
        store.assert_disclosure_allowed("carol")
    assert store.assert_disclosure_allowed("alice")
    assert store.assert_disclosure_allowed("bob")


def test_portuguese_queue_and_postvote_status_filters_do_not_unblind(store):
    assert len(store.list_examples("alice", language="pt")) == 1
    vote(store, "alice")
    d = vote(store, "bob", gold=["financial"])
    assert store.list_examples("alice", status="disagreement")[0]["status"] == "disagreement"
    assert store.list_examples("carol", status="disagreement") == []
    assert store.list_examples("carol")[0]["status"] == "pending"
    store.adjudicate("a", "alice", "selected", ["seat"], "Sitting context.", d["revision"])
    assert len(store.list_examples("alice", status="adjudicated")) == 1
    assert len(store.list_examples("alice", status="reviewed")) == 1
    assert store.list_examples("carol", status="adjudicated") == []


def test_gold_export_checksum_matches_exact_jsonl(store):
    import hashlib

    vote(store, "alice")
    d = vote(store, "bob")
    store.adjudicate("a", "alice", "selected", ["seat"], expected_revision=d["revision"])
    result = store.export_gold()
    assert result["manifest"]["sha256"] == hashlib.sha256(result["jsonl"].encode("utf-8")).hexdigest()
    assert [json.loads(line) for line in result["jsonl"].splitlines()] == result["examples"]


def test_skip_is_audit_only_and_cannot_unlock_ai_or_count_as_review(store):
    skipped = vote(store, "alice", "skip", [])
    assert skipped["my_vote"] is None
    assert store.stats()["by_language"]["pt-BR"]["reviewed"] == 0
    with pytest.raises(PermissionError):
        store.record_assistance("a", "alice", assistance())
    vote(store, "bob")
    with pytest.raises(PermissionError):
        store.get_example("a", "alice", adjudicate=True)
    assert any(e["action"] == "skip" for e in store.backup()["events"])
    voted = vote(store, "alice")
    assert voted["my_vote"]["initial"]["outcome"] == "selected"
    skipped = vote(store, "alice", "skip", [])
    assert skipped["my_vote"]["gold_ids"] == ["seat"]
    assert skipped["my_vote"]["initial"]["outcome"] == "selected"


def automatic_row(eid="auto", bucket="jev_high_confidence", ids=None, confidence=0.9):
    data = row(eid)
    data["gold_ids"] = ["seat"] if ids is None else ids
    data["provenance"]["generation"] = {"difficulty": "hard", "area": "linguistics", "paper_id": "paper-1"}
    data["provenance"]["automatic"] = {
        "status": "labeled",
        "bucket": bucket,
        "selected_by": "jev",
        "threshold": 0.8,
        "label_tier": "silver",
        "jev": {"choice": "seat", "selected_ids": data["gold_ids"], "confidence": confidence},
        "codex": None,
        "escalation_reason": None,
    }
    return data


def test_automatic_review_preserves_proposal_and_exports_one_human_approval(tmp_path):
    s = Store(tmp_path / "auto.sqlite")
    source = automatic_row()
    s.import_examples([source])
    detail = s.get_example("auto", "owner")
    assert detail["mode"] == "automatic" and detail["review"] is None
    assert detail["example"]["provenance"]["proposed_gold_ids"] == ["seat"]
    assert "gold_ids" not in detail["example"]
    assert detail["automatic"] == source["provenance"]["automatic"]
    assert not s.export_gold()["examples"]
    s.review("auto", "owner", "approve", expected_revision=detail["revision"])
    gold = s.export_gold()["examples"][0]
    assert gold["gold_ids"] == ["seat"]
    assert gold["provenance"]["review"]["status"] == "approved"
    assert gold["provenance"]["automatic"] == source["provenance"]["automatic"]
    assert s.assert_disclosure_allowed("another-owner", backup=True)
    with pytest.raises(RuntimeError):
        s.review("auto", "owner", "reject", expected_revision=detail["revision"])
    with pytest.raises(ValueError, match="automatic review"):
        s.vote("auto", "owner", "none_fits", [], expected_revision=1)
    assert s.get_example("auto", "another-owner")["review"]["reviewer"] == "owner"


def test_automatic_correction_multi_ids_none_insufficient_reject_and_audit(tmp_path):
    s = Store(tmp_path / "auto.sqlite")
    s.import_examples([automatic_row()])
    for action, outcome, ids, status in [
        ("correct", "selected", ["seat", "financial"], "corrected"),
        ("correct", "none_fits", [], "corrected"),
        ("correct", "insufficient_context", [], "insufficient_context"),
        ("reject", None, [], "rejected"),
    ]:
        detail = s.get_example("auto", "owner")
        s.review("auto", "owner", action, outcome, ids, "Audit note", detail["revision"])
        detail = s.get_example("auto", "owner")
        assert detail["review"]["status"] == status
        assert detail["review"]["note"] == "Audit note"
        exported = s.export_gold()["examples"]
        assert bool(exported) == (status == "corrected")
        if exported:
            assert exported[0]["gold_ids"] == ids
        assert detail["example"]["provenance"]["proposed_gold_ids"] == ["seat"]
    assert len(s.backup()["human_reviews"]) == 1
    assert len([e for e in s.backup()["events"] if e["action"] == "human_review"]) == 4
    all_status = s.export_all()["examples"][0]
    assert all_status["provenance"]["review_status"] == "rejected"
    assert all_status["gold_ids"] is None


def test_automatic_filters_and_observed_jev_accuracy_denominators(tmp_path):
    s = Store(tmp_path / "auto.sqlite")
    corrected_by_codex = automatic_row("b")
    corrected_by_codex["provenance"]["automatic"].update(
        selected_by="codex", codex={"outcome": "selected", "selected_ids": ["financial"]}
    )
    s.import_examples([automatic_row("a"), corrected_by_codex, automatic_row("c", confidence=0.4)])
    for eid, ids in [("a", ["seat"]), ("b", ["financial"])]:
        s.review(eid, "owner", "correct", "selected", ids, expected_revision=0)
    assert len(s.list_examples("owner", status="reviewed")) == 2
    assert [r["id"] for r in s.list_examples("owner", status="unreviewed")] == ["c"]
    assert (
        len(
            s.list_examples(
                "owner", language="pt", difficulty="hard", area="linguistics", paper="paper-1", confidence="high"
            )
        )
        == 2
    )
    assert s.list_examples("owner", bucket="unresolved") == []
    report = s.stats()["automatic_review"]
    assert report["reviewed"] == 2 and report["accepted"] == 2
    assert report["by_confidence"]["90–100%"]["accuracy"] == 0.5
    assert report["by_confidence"]["90–100%"]["comparable"] == 2
    assert s.export_all()["manifest"]["count"] == 3


def test_unresolved_automatic_proposal_requires_correction(tmp_path):
    s = Store(tmp_path / "auto.sqlite")
    data = automatic_row()
    data.pop("gold_ids")
    data["provenance"]["automatic"].update(status="unresolved", bucket="unresolved", selected_by=None)
    s.import_examples([data])
    with pytest.raises(ValueError, match="Resolve"):
        s.review("auto", "owner", "approve")
    s.review("auto", "owner", "correct", "none_fits", [])
    assert s.export_gold()["examples"][0]["gold_ids"] == []
