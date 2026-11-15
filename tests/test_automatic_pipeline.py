"""Offline integration: actual PDF extraction and full artifact/review round trip.

Documents and definitions here are authored software fixtures, not real papers.
"""

import json
import shutil

import pytest

from wsd import generation, labeling
from wsd.annotation_store import Store
from wsd.data import read_jsonl
from wsd.pipeline import load_config, require_reviewed, save_config


def _pdf(text):
    content = f"BT /F1 12 Tf 40 740 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    payload = b"%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(payload)
    payload += b"xref\n0 6\n0000000000 65535 f \n"
    payload += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    payload += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return payload


def test_real_pdf_extraction(tmp_path):
    if not shutil.which("pdftotext"):
        pytest.importorskip("pypdf")
    source = tmp_path / "fixture.pdf"
    source.write_bytes(_pdf("The bank beside the river was covered in green grass."))
    assert "bank beside the river" in generation.extract_pdf(source)


def test_generate_label_review_evaluate_contract(tmp_path, monkeypatch):
    paper = tmp_path / "authored.txt"
    paper.write_text(
        "The river flooded the grassy bank during the heavy rainfall.\n"
        "The bank approved a loan to finance a small business yesterday.",
        encoding="utf-8",
    )
    inventory = tmp_path / "dictionary.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "lemma": "bank",
                "pos": "n",
                "language": "en",
                "candidates": [
                    {"id": "river", "definition": "Land beside a body of water."},
                    {"id": "finance", "definition": "A financial institution."},
                ],
            }
        ),
        encoding="utf-8",
    )
    sources = tmp_path / "papers.jsonl"
    sources.write_text(
        json.dumps(
            {
                "title": "Authored integration fixture",
                "language": "en",
                "path": paper.name,
                "license": "CC0 fixture",
                "area": "software testing",
            }
        )
    )
    config_path = tmp_path / "private" / "config.toml"
    config = load_config(config_path)
    config["jev"].update(api_key="fixture-not-a-real-key", max_retries=0)
    save_config(config, config_path)

    def generate(prompt, schema, **kwargs):
        pool = json.loads(prompt.split("\n", 1)[1])["pool"]
        return {
            "detected_language": "en",
            "selections": [
                {"pool_id": row["pool_id"], "difficulty": "easy", "reason": "Authored fixture context"}
                for row in pool[:2]
            ],
        }

    monkeypatch.setattr(generation, "run_codex", generate)
    generated = tmp_path / "generated"
    generation.generate(n=2, output=generated, lexicons={"en": inventory}, sources=sources, languages=("en",))
    inputs = read_jsonl(generated / "samples.jsonl")
    assert len(inputs) == 2 and all(ex.gold_ids is None for ex in inputs)

    def jev(ex, *args):
        finance = "approved" in ex.text
        return {
            "choice": "finance",
            "selected_ids": ["finance"],
            "confidence": 0.95 if finance else 0.2,
            "probabilities": {"finance": 0.7, "river": 0.2},
            "none_probability": 0.05,
            "insufficient_probability": 0.05,
            "model": "fixture-jev",
            "provider": "typesafe",
        }

    monkeypatch.setattr(labeling.jev, "assist", jev)
    monkeypatch.setattr(
        labeling,
        "run_codex",
        lambda *args, **kw: {
            "outcome": "selected",
            "selected_ids": ["river"],
            "rationale": "The passage describes river land.",
        },
    )
    labeled = tmp_path / "labeled"
    report = labeling.label_dataset(generated / "samples.jsonl", labeled, config_path=config_path)
    assert report["labeled"] == 2
    rows = read_jsonl(labeled / "samples.jsonl")
    with pytest.raises(ValueError, match="silver"):
        require_reviewed(rows)
    assert {ex.provenance["automatic"]["bucket"] for ex in rows} == {"jev_high_confidence", "jev_low_codex_disagreed"}
    store = Store(tmp_path / "review.sqlite")
    assert store.import_examples([ex.to_dict() for ex in rows])["imported"] == 2
    for ex in rows:
        detail = store.get_example(ex.id, "human")
        store.review(ex.id, "human", "approve", expected_revision=detail["revision"])
    exported = store.export_gold()
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text(exported["jsonl"], encoding="utf-8")
    reviewed_examples = read_jsonl(reviewed)
    require_reviewed(reviewed_examples)
    assert len(reviewed_examples) == 2
    assert all(ex.provenance["review"]["status"] == "approved" for ex in reviewed_examples)
    assert all(ex.provenance["automatic"] for ex in reviewed_examples)
    # Canonical artifact is directly usable by the existing evaluation machinery.
    from wsd.evaluate import summarize

    predictions = [
        {
            "example_id": ex.id,
            "selected_id": ex.gold_ids[0],
            "reason": "selected",
            "probabilities": {c.id: float(c.id in ex.gold_ids) for c in ex.candidates},
            "none_probability": 0.0,
        }
        for ex in reviewed_examples
    ]
    assert summarize(reviewed_examples, predictions, bootstrap_samples=10)
