"""Offline acquisition tests use quoted fixture text, not benchmark samples."""

import json
import urllib.request
from collections import Counter

import pytest

from wsd import generation as g
from wsd.schema import Example


def fixture_inputs(tmp_path, languages=("en", "pt"), copies=1):
    lexicons, sources = {}, []
    for language in languages:
        lexicon = tmp_path / f"lexicon-{language}.jsonl"
        lexicon.write_text(
            json.dumps(
                {
                    "lemma": "bank" if language == "en" else "banco",
                    "pos": "n",
                    "language": language,
                    "candidates": [
                        {"id": language + ":1", "definition": "Financial institution"},
                        {"id": language + ":2", "definition": "Land or seat"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        lexicons[language] = str(lexicon)
        for copy in range(copies):
            text = tmp_path / f"paper-{language}-{copy}.txt"
            target = "bank" if language == "en" else "banco"
            text.write_text(
                "\n".join(
                    f"😀 Experiência {copy} número {i}: the {target} provides substantial context for study."
                    for i in range(12)
                ),
                encoding="utf-8",
            )
            sources.append(
                {
                    "title": f"Fixture {language} {copy}",
                    "language": language,
                    "path": text.name,
                    "license": "CC0 fixture",
                }
            )
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources), encoding="utf-8")
    return lexicons, path


def choose_pool(prompt, schema, **kwargs):
    offered = json.JSONDecoder().raw_decode(prompt.split("\n", 1)[1])[0]["pool"]
    # The requested schedule is the only list before the data payload.
    schedule = json.loads(prompt.split("schedule: ", 1)[1].split(". Easy:", 1)[0])
    return {
        "detected_language": "pt" if "Fixture pt" in prompt else "en",
        "selections": [
            {"pool_id": row["pool_id"], "difficulty": difficulty, "reason": "Fixture heuristic"}
            for row, difficulty in zip(offered, schedule)
        ],
    }


def test_manifest_hash_and_dictionary_language_guard(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path)
    with pytest.raises(g.GenerationError, match="must be pt"):
        g.load_inventory(lexicons["en"], "pt")
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(samples_per_paper=1, output=output, lexicons=lexicons, sources=sources)
    with open(lexicons["en"], "a") as stream:
        stream.write("\n")
    with pytest.raises(g.GenerationError, match="changed"):
        g.generate(samples_per_paper=1, output=output, lexicons=lexicons, sources=sources, resume=True)


def test_pdf_extraction_external_binary_is_offline_and_uses_utf8(tmp_path, monkeypatch):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    calls = []
    monkeypatch.setattr(g.shutil, "which", lambda name: "/usr/bin/pdftotext")

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "Ação 😀 extracted".encode()})()

    monkeypatch.setattr(g.subprocess, "run", run)
    assert g.extract_pdf(source) == "Ação 😀 extracted"
    assert calls[0][0] == ["pdftotext", "-enc", "UTF-8", str(source), "-"]


def test_source_pool_handles_repeated_unicode_occurrences_exactly():
    inventory = {("banco", "n"): [g.Candidate("1", "one"), g.Candidate("2", "two")]}
    text = "😀 Ação: o banco é descrito neste trecho; outro banco aparece no mesmo contexto."
    pool = g.target_pool(text, inventory, 7)
    assert len(pool) == 2
    for row in pool:
        target = row["target"]
        assert row["text"][target["start"] : target["end"]] == "banco"
        assert text[row["source_start"] : row["source_end"]] == row["text"]


def test_inventory_pools_all_pos_deduplicates_ids_and_rejects_conflicts(tmp_path):
    path = tmp_path / "lexicon.jsonl"
    rows = [
        {
            "lemma": "bank",
            "language": "en",
            "pos": "n",
            "candidates": [{"id": "n", "definition": "a financial institution"}],
        },
        {
            "lemma": "bank",
            "language": "en",
            "pos": "v",
            "candidates": [
                {"id": "v", "definition": "tilt an aircraft"},
                {"id": "n", "definition": "a financial institution"},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    inventory, meta = g.load_inventory(path, "en")
    assert [vars(c) for c in inventory[("bank", "all")]] == [
        {"id": "n", "definition": "a financial institution"},
        {"id": "v", "definition": "tilt an aircraft"},
    ]
    assert meta["candidate_scope"] == "all_inventory_pos"
    rows[1]["candidates"][-1]["definition"] = "conflicting definition"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(g.GenerationError, match="Conflicting"):
        g.load_inventory(path, "en")
    rows[1]["candidates"] = [{"id": str(i), "definition": f"meaning {i}"} for i in range(127)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(g.GenerationError, match="pooled.*127"):
        g.load_inventory(path, "en")


@pytest.mark.parametrize("change", ["run_id", "language", "candidate", "passage", "text_cache", "metadata", "manifest"])
def test_resume_revalidates_batches_without_overwriting_existing_exports(tmp_path, monkeypatch, change):
    lexicons, sources = fixture_inputs(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(g, "run_codex", choose_pool)
    g.generate(samples_per_paper=1, output=output, lexicons=lexicons, sources=sources)
    slot_path = output / "cache" / "slot-000000.json"
    batch = json.loads(slot_path.read_text())
    if change == "run_id":
        batch["run_id"] = "different-run"
    elif change == "language":
        batch["samples"][0]["language"] = "pt"
    elif change == "candidate":
        batch["samples"][0]["candidates"][0]["definition"] = "changed definition"
    elif change == "passage":
        batch["samples"][0]["text"] += " added text"
    elif change == "text_cache":
        with open(batch["paper"]["text_path"], "a") as stream:
            stream.write(" changed cache")
    elif change == "metadata":
        batch["samples"][0]["provenance"]["generation"]["detected_language"] = "pt"
    elif change == "manifest":
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["model"] = "changed-with-original-run-id"
        manifest_path.write_text(json.dumps(manifest))
    # A self-consistent checksum alone must not accept changed batch semantics.
    batch["batch_fingerprint"] = g.fingerprint({"samples": batch["samples"], "paper": batch["paper"]})
    slot_path.write_text(json.dumps(batch))
    names = ("samples.jsonl", "papers.jsonl", "manifest.json", "generation-report.json")
    before = {name: (output / name).read_bytes() for name in names}
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: pytest.fail("invalid cache must fail before model calls"))
    with pytest.raises(g.GenerationError, match="Resume"):
        g.generate(samples_per_paper=1, output=output, lexicons=lexicons, sources=sources, resume=True)
    assert {name: (output / name).read_bytes() for name in names} == before


def test_selection_schema_has_exact_count_candidates_and_source_audit_metadata(tmp_path, monkeypatch):
    lexicons, manifest = fixture_inputs(tmp_path, languages=("en",))
    sources = json.loads(manifest.read_text())
    sources[0].update(url="https://example.org/paper", area="ecology", domain="scientific")
    text_path = tmp_path / sources[0]["path"]
    text_path.write_text("Introduction without a dictionary target.\f" + text_path.read_text())
    manifest.write_text(json.dumps(sources))

    def select(prompt, schema, **kwargs):
        choices = schema["properties"]["selections"]
        assert choices["minItems"] == choices["maxItems"] == 1
        assert choices["items"]["properties"]["pool_id"]["enum"]
        assert "Return exactly 1 selections" in prompt
        return choose_pool(prompt, schema, **kwargs)

    monkeypatch.setattr(g, "run_codex", select)
    output = tmp_path / "run"
    g.generate(samples_per_paper=1, output=output, lexicons=lexicons, sources=manifest)
    row = json.loads((output / "samples.jsonl").read_text())
    p = row["provenance"]
    assert p["source_url"] == "https://example.org/paper"
    assert p["area"] == p["generation"]["area"] == "ecology" and p["domain"] == "scientific"
    assert p["page"] == p["generation"]["page"] == 2


def test_pypdf_fallback_uses_bounded_subprocess_and_preserves_interrupts(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-offline-fixture")
    monkeypatch.setattr(g.shutil, "which", lambda name: None)
    monkeypatch.setattr(g.importlib.util, "find_spec", lambda name: object())
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "Page 1\n\f\nPágina 2 😀".encode()})()

    monkeypatch.setattr(g.subprocess, "run", run)
    assert g.extract_pdf(path) == "Page 1\n\f\nPágina 2 😀"
    args, kwargs = calls[0]
    assert args[0] == g.sys.executable and args[1] == "-c" and args[-1] == str(path)
    assert kwargs["timeout"] == 120 and "shell" not in kwargs

    def timed_out(*args, **kwargs):
        raise g.subprocess.TimeoutExpired("private-path", 120)

    monkeypatch.setattr(g.subprocess, "run", timed_out)
    with pytest.raises(g.GenerationError, match="timed out"):
        g.extract_pdf(path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(g.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        g.extract_pdf(path)


def test_every_paper_in_manifest_order_default_quota_unicode_and_resume(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, copies=2)
    calls = []

    def select(prompt, schema, **kwargs):
        calls.append((prompt, kwargs))
        return choose_pool(prompt, schema, **kwargs)

    monkeypatch.setattr(g, "run_codex", select)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("No network acquisition"))
    output = tmp_path / "run"
    report = g.generate(output=output, lexicons=lexicons, sources=sources)
    assert report["status"] == "complete"
    assert report["requested"] == report["accepted"] == 40
    assert report["accepted_papers"] == 4
    assert report["languages"] == {"en": 20, "pt": 20}
    assert report["difficulties"] == {"easy": 20, "medium": 12, "hard": 8}
    papers = [json.loads(line) for line in (output / "papers.jsonl").read_text().splitlines()]
    assert [row["title"] for row in papers] == [row["title"] for row in json.loads(sources.read_text())]
    assert len(calls) == 4 and all(kw["model"] == g.DEFAULT_MODEL for _, kw in calls)
    for line in (output / "samples.jsonl").read_text().splitlines():
        ex = Example.from_dict(json.loads(line))
        assert ex.gold_ids is None
        assert [c.id for c in ex.candidates] == [ex.language + ":1", ex.language + ":2"]
        metadata = ex.provenance["generation"]
        source = next(paper for paper in papers if paper["paper_id"] == metadata["paper_id"])
        text = g._exact_text(source["text_path"])
        assert text[metadata["source_start"] : metadata["source_end"]] == ex.text
        assert ex.text[ex.target.start : ex.target.end] == ex.target.text
    before = (output / "samples.jsonl").read_bytes()
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: pytest.fail("Completed papers must not call the model"))
    report = g.generate(output=output, lexicons=lexicons, sources=sources, resume=True)
    assert report["accepted"] == 40
    assert (output / "samples.jsonl").read_bytes() == before
    with pytest.raises(g.ResumeError, match="changed"):
        g.generate(output=output, lexicons=lexicons, sources=sources, samples_per_paper=9, resume=True)


@pytest.mark.parametrize(
    "count,expected",
    [
        (1, {"easy": 1}),
        (2, {"easy": 1, "medium": 1}),
        (3, {"easy": 1, "medium": 1, "hard": 1}),
        (7, {"easy": 4, "medium": 2, "hard": 1}),
        (10, {"easy": 5, "medium": 3, "hard": 2}),
        (13, {"easy": 6, "medium": 4, "hard": 3}),
    ],
)
def test_difficulty_largest_remainder(count, expected):
    assert Counter(g.difficulty_schedule(count)) == expected


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_quota_rejected(count):
    with pytest.raises(g.GenerationError, match="positive integer"):
        g.difficulty_schedule(count)


def test_quota_above_default_pool_limit_and_too_few_targets(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    paper = tmp_path / json.loads(sources.read_text())[0]["path"]
    paper.write_text(
        "\n".join(f"Occurrence {i}: the bank offers authentic context for this research." for i in range(170))
    )
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    report = g.generate(sources=sources, output=output, lexicons=lexicons, samples_per_paper=165)
    assert report["accepted"] == 165 and report["status"] == "complete"
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: pytest.fail("Resume should not call model"))
    assert (
        g.generate(sources=sources, output=output, lexicons=lexicons, samples_per_paper=165, resume=True)["accepted"]
        == 165
    )
    report = g.generate(sources=sources, output=tmp_path / "short", lexicons=lexicons, samples_per_paper=171)
    assert report["status"] == "incomplete" and report["accepted"] == 0
    assert "Only 170" in report["failures"][0]["error"]


def test_failed_middle_paper_continues_and_resume_only_retries_failure(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=3)
    calls = []

    def select(prompt, schema, **kwargs):
        calls.append(prompt)
        if "Fixture en 1" in prompt:
            raise g.CodexError("Requested model unavailable")
        return choose_pool(prompt, schema, **kwargs)

    output = tmp_path / "run"
    monkeypatch.setattr(g, "run_codex", select)
    report = g.generate(sources=sources, output=output, lexicons=lexicons)
    assert report["status"] == "incomplete" and report["requested"] == 30 and report["accepted"] == 20
    assert len(calls) == 3
    assert report["failures"][0]["slot"] == 1
    assert report["failures"][0]["status"] == "incomplete"
    assert not (output / "cache/slot-000001.json").exists()
    calls.clear()
    monkeypatch.setattr(
        g, "run_codex", lambda prompt, schema, **kw: calls.append(prompt) or choose_pool(prompt, schema)
    )
    report = g.generate(sources=sources, output=output, lexicons=lexicons, resume=True)
    assert report["status"] == "complete" and report["accepted"] == 30 and not report["failures"]
    assert len(calls) == 1 and "Fixture en 1" in calls[0]
    assert [json.loads(line)["slot"] for line in (output / "papers.jsonl").read_text().splitlines()] == [0, 1, 2]


@pytest.mark.parametrize("wrong", ["selection", "language"])
def test_invalid_selection_or_language_is_retried_without_fabrication(tmp_path, monkeypatch, wrong):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    attempts = []

    def invalid(prompt, schema, **kwargs):
        attempts.append(prompt)
        if wrong == "language":
            return {**choose_pool(prompt, schema), "detected_language": "pt"}
        return {"detected_language": "en", "selections": [{"pool_id": "invented", "difficulty": "easy", "reason": ""}]}

    monkeypatch.setattr(g, "run_codex", invalid)
    output = tmp_path / "run"
    report = g.generate(sources=sources, output=output, lexicons=lexicons, samples_per_paper=1)
    assert report["status"] == "incomplete" and report["accepted"] == 0
    assert len(attempts) == 3
    assert not (output / "samples.jsonl").read_text()
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert Counter(row["event"] for row in events)["selection_rejected"] == 3


@pytest.mark.parametrize(
    "defect,match",
    [
        ("missing", "does not exist"),
        ("path", "Duplicate source path"),
        ("content", "Duplicate paper content"),
        ("remote", "local path"),
        ("language", "English.*Portuguese"),
    ],
)
def test_manifest_preflight_rejects_invalid_sources_before_model(tmp_path, monkeypatch, defect, match):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=2)
    rows = json.loads(sources.read_text())
    if defect == "missing":
        rows[1]["path"] = "papers/missing.pdf"
    elif defect == "path":
        rows[1]["path"] = rows[0]["path"]
    elif defect == "content":
        (tmp_path / rows[1]["path"]).write_bytes((tmp_path / rows[0]["path"]).read_bytes())
    elif defect == "remote":
        del rows[1]["path"]
        rows[1]["pdf_url"] = "https://example.invalid/p.pdf"
    else:
        del rows[1]["language"]
    sources.write_text(json.dumps({"papers": rows}))
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: pytest.fail("Invalid inputs must not call model"))
    with pytest.raises(g.GenerationError, match=match):
        g.generate(sources=sources, output=tmp_path / "run", lexicons=lexicons)
    assert not (tmp_path / "run").exists()


def test_manifest_relative_paths_optional_title_and_default_language(tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    paper = folder / "source.txt"
    paper.write_text("Authentic paper text.")
    manifest = tmp_path / "papers.json"
    manifest.write_text(json.dumps({"language": "en-US", "papers": [{"path": "papers/source.txt"}]}))
    (row,) = g.read_sources(manifest)
    assert row["path"] == str(paper.resolve())
    assert row["title"] == "source" and row["language"] == "en"
    assert row["input_sha256"] == g.file_sha256(paper)


def test_cli_input_alias_inventory_inference_defaults_and_config(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=2)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "papers.json").write_bytes(sources.read_bytes())
    monkeypatch.setattr(g, "run_codex", choose_pool)
    report = g.main(["--inventory", lexicons["en"]])
    assert report["accepted"] == 20
    assert (tmp_path / "data/pilot/generated/samples.jsonl").exists()
    report = g.main(
        [
            "--sources",
            str(sources),
            "--inventory",
            lexicons["en"],
            "--samples-per-paper",
            "3",
            "--output",
            str(tmp_path / "custom"),
        ]
    )
    assert report["accepted"] == 6 and report["difficulties"] == {"easy": 2, "medium": 2, "hard": 2}
    monkeypatch.setattr(
        g,
        "load_config",
        lambda *a: {
            "generation": {"samples_per_paper": 2, "lexicon_en": lexicons["en"]},
            "codex": {"model": "custom-model"},
        },
    )
    report = g.main(["--input", str(sources), "--output", str(tmp_path / "configured")])
    assert report["accepted"] == 4


def test_cli_incomplete_exits_nonzero_after_saving_report(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: (_ for _ in ()).throw(g.CodexError("Unavailable")))
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        g.main(["--input", str(sources), "--inventory", lexicons["en"], "--output", str(output)])
    assert exc.value.code == 1
    assert json.loads((output / "generation-report.json").read_text())["status"] == "incomplete"


def test_resume_requires_original_manifest_and_source_hash(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(sources=sources, output=output, lexicons=lexicons)
    before = (output / "samples.jsonl").read_bytes()
    manifest = output / "manifest.json"
    payload = manifest.read_bytes()
    manifest.unlink()
    with pytest.raises(g.ResumeError, match="original manifest"):
        g.generate(sources=sources, output=output, lexicons=lexicons, resume=True)
    manifest.write_bytes(payload)
    paper = tmp_path / json.loads(sources.read_text())[0]["path"]
    paper.write_text(paper.read_text() + " Changed.")
    with pytest.raises(g.ResumeError, match="hashes changed"):
        g.generate(sources=sources, output=output, lexicons=lexicons, resume=True)
    assert (output / "samples.jsonl").read_bytes() == before


def test_missing_earlier_checkpoint_never_substitutes_later_completed_paper(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=2)
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(sources=sources, output=output, lexicons=lexicons)
    (output / "cache/slot-000000.json").unlink()
    first = g.read_sources(sources)[0]["paper_id"]

    def inaccessible(source, *args):
        assert source["paper_id"] == first
        raise g.GenerationError("First source unavailable")

    monkeypatch.setattr(g, "acquire_source", inaccessible)
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: pytest.fail("No substitution allowed"))
    report = g.generate(sources=sources, output=output, lexicons=lexicons, resume=True)
    assert report["status"] == "incomplete" and report["accepted"] == 10
    assert (output / "cache/slot-000001.json").exists()


def test_failed_extraction_continues_to_next_listed_paper(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=2)
    rows = json.loads(sources.read_text())
    pdf = tmp_path / "corrupt.pdf"
    pdf.write_bytes(b"%PDF-corrupt-fixture")
    rows[0]["path"] = pdf.name
    sources.write_text(json.dumps({"papers": rows}))
    monkeypatch.setattr(g, "extract_pdf", lambda *a: (_ for _ in ()).throw(g.GenerationError("Unreadable PDF")))
    monkeypatch.setattr(g, "run_codex", choose_pool)
    report = g.generate(sources=sources, output=tmp_path / "run", lexicons=lexicons)
    assert report["status"] == "incomplete" and report["accepted"] == 10
    assert report["failures"][0]["slot"] == 0


def test_duplicate_extracted_text_fails_without_substitution(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=2)
    rows = json.loads(sources.read_text())
    for index, row in enumerate(rows):
        paper = tmp_path / f"paper-{index}.pdf"
        paper.write_bytes(f"%PDF-distinct-fixture-{index}".encode())
        row["path"] = paper.name
    sources.write_text(json.dumps(rows))
    monkeypatch.setattr(
        g,
        "extract_pdf",
        lambda *a: "\n".join(
            f"Occurrence {i}: the bank offers authentic context for this research." for i in range(12)
        ),
    )
    calls = []
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: calls.append(1) or choose_pool(*a, **kw))
    report = g.generate(sources=sources, output=tmp_path / "run", lexicons=lexicons)
    assert report["status"] == "incomplete" and report["accepted"] == 10
    assert len(calls) == 1 and "Duplicate extracted" in report["failures"][0]["error"]


def test_text_fixture_keeps_original_unicode_and_crlf_offsets(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    paper = tmp_path / json.loads(sources.read_text())[0]["path"]
    original = paper.read_text().replace("\n", "\r\n")
    paper.write_bytes(original.encode("utf-8"))
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(sources=sources, output=output, lexicons=lexicons)
    (saved,) = (output / "cache").glob("paper-*.txt")
    assert saved.read_bytes() == original.encode("utf-8")
    g.generate(sources=sources, output=output, lexicons=lexicons, resume=True)


def test_cli_config_input_and_output_paths(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    output = tmp_path / "configured-output"
    monkeypatch.setattr(
        g,
        "load_config",
        lambda *a: {
            "generation": {
                "input": str(sources),
                "output": str(output),
                "lexicon_en": lexicons["en"],
                "samples_per_paper": 2,
            }
        },
    )
    monkeypatch.setattr(g, "run_codex", choose_pool)
    assert g.main([])["accepted"] == 2
    assert (output / "samples.jsonl").exists()
