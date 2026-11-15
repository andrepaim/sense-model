import json

import pytest

from wsd.pilot import audit_examples, collect


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def fixture(tmp_path):
    lexicon = write(
        tmp_path / "lexicon.jsonl",
        [
            {
                "lemma": "banco",
                "language": "pt",
                "pos": "n",
                "candidates": [
                    {"id": "seat", "definition": "Assento."},
                    {"id": "finance", "definition": "Instituição financeira."},
                ],
            },
            {
                "lemma": "banco de dados",
                "language": "pt",
                "pos": "n",
                "candidates": [{"id": "database", "definition": "Conjunto estruturado de dados."}],
            },
        ],
    )
    docs = write(
        tmp_path / "docs.jsonl",
        [
            {
                "id": "doc",
                "text": "😀 O banco de dados está no banco. O banco fica na praça.",
                "language": "pt-BR",
                "source": "licensed-fixture",
                "domain": "technical",
                "source_url": "https://example.org/doc",
                "revision": "v1",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        ],
    )
    return docs, lexicon


def run(docs, lexicon, **kwargs):
    return collect(
        docs,
        lexicon,
        definition_language="pt-BR",
        inventory_version="fixture-v1",
        inventory_license="CC0-1.0",
        **kwargs,
    )


def test_collection_spans_full_inventory_no_labels(tmp_path):
    docs, lexicon = fixture(tmp_path)
    examples, manifest = run(docs, lexicon, limit_per_language=10)
    assert len(examples) == 4  # overlapping phrase, two occurrences, next sentence
    assert all(e.gold_ids is None for e in examples)
    assert sum(e.target.text == "banco" for e in examples) == 3
    assert any(e.target.text == "banco de dados" for e in examples)
    for e in examples:
        assert e.text[e.target.start : e.target.end] == e.target.text
        assert e.provenance["inventory_sha256"]
        if e.target.text == "banco":
            assert len(e.candidates) == 2
    assert manifest["shortfalls"]["pt-BR"] == 6
    assert not audit_examples(examples)["missing_source_provenance"]


def test_caps_deterministic_and_input_unchanged(tmp_path):
    docs, lexicon = fixture(tmp_path)
    before = docs.read_bytes()
    examples, a = run(docs, lexicon, max_per_document=2, max_per_lemma=1)
    repeated, b = run(docs, lexicon, max_per_document=2, max_per_lemma=1)
    assert len(examples) == 2
    assert [e.to_dict() for e in examples] == [e.to_dict() for e in repeated]
    assert a == b and before == docs.read_bytes()


def test_provenance_and_language_required(tmp_path):
    docs, lexicon = fixture(tmp_path)
    row = json.loads(docs.read_text())
    del row["license"]
    write(docs, [row])
    with pytest.raises(ValueError, match="Document requires"):
        run(docs, lexicon)
    write(lexicon, [{"lemma": "banco", "language": "en", "candidates": [{"id": "x", "definition": "seat"}]}])
    with pytest.raises(ValueError, match="language"):
        run(docs, lexicon)
