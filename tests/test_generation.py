"""Offline acquisition tests use quoted fixture text, not benchmark samples."""

import json
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


def test_exact_unicode_offsets_pinned_candidates_bilingual_trim_and_resume(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path)
    calls = []

    def select(*args, **kwargs):
        calls.append(kwargs)
        return choose_pool(*args, **kwargs)

    monkeypatch.setattr(g, "run_codex", select)
    output = tmp_path / "run"
    report = g.generate(n=17, output=output, lexicons=lexicons, sources=sources)
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert report["accepted"] == 17
    assert report["accepted_papers"] == 2
    assert report["languages"] == {"en": 10, "pt": 7}
    assert report["difficulties"] == {"easy": 10, "medium": 5, "hard": 2}
    assert len(calls) == 2
    assert all(call["model"] == g.DEFAULT_MODEL for call in calls)
    for row in rows:
        ex = Example.from_dict(row)
        assert ex.gold_ids is None
        assert [c.id for c in ex.candidates] == [ex.language + ":1", ex.language + ":2"]
        metadata = ex.provenance["generation"]
        text = (output / "cache" / (metadata["paper_id"].replace(":", "-") + ".txt")).read_text()
        assert text[metadata["source_start"] : metadata["source_end"]] == ex.text
        assert ex.text[ex.target.start : ex.target.end] == ex.target.text
    before = (output / "samples.jsonl").read_bytes()
    monkeypatch.setattr(g, "run_codex", lambda *a, **k: pytest.fail("resume must not call the model"))
    g.generate(n=17, output=output, lexicons=lexicons, sources=sources, resume=True)
    assert (output / "samples.jsonl").read_bytes() == before
    with pytest.raises(g.GenerationError, match="changed"):
        g.generate(n=18, output=output, lexicons=lexicons, sources=sources, resume=True)


def test_partial_failure_persists_then_resumes_without_repeating_completed_paper(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path)
    seen = []

    def select(prompt, schema, **kwargs):
        seen.append(prompt)
        if "Fixture pt" in prompt:
            raise g.CodexError("Requested model unavailable")
        return choose_pool(prompt, schema, **kwargs)

    output = tmp_path / "run"
    monkeypatch.setattr(g, "run_codex", select)
    with pytest.raises(g.CodexError, match="unavailable"):
        g.generate(n=20, output=output, lexicons=lexicons, sources=sources)
    report = json.loads((output / "generation-report.json").read_text())
    assert report["status"] == "incomplete" and report["accepted"] == 10
    seen.clear()
    monkeypatch.setattr(g, "run_codex", lambda prompt, schema, **kw: seen.append(prompt) or choose_pool(prompt, schema))
    report = g.generate(n=20, output=output, lexicons=lexicons, sources=sources, resume=True)
    assert report["accepted"] == 20
    assert len(seen) == 1 and "Fixture pt" in seen[0]


def test_invalid_model_selection_retried_and_never_labels_or_invents(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    attempts = []

    def invalid(prompt, schema, **kwargs):
        attempts.append(prompt)
        return {"detected_language": "en", "selections": [{"pool_id": "invented", "difficulty": "easy", "reason": ""}]}

    monkeypatch.setattr(g, "run_codex", invalid)
    output = tmp_path / "run"
    with pytest.raises(g.GenerationError, match="Not enough usable"):
        g.generate(n=1, output=output, lexicons=lexicons, sources=sources, languages=("en",))
    assert len(attempts) == 3
    assert not (output / "samples.jsonl").read_text()
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert Counter(row["event"] for row in events)["selection_rejected"] == 3


def test_default_scholar_discovery_uses_diverse_seeded_queries_and_cache(tmp_path, monkeypatch):
    lexicons, explicit = fixture_inputs(tmp_path)
    rows = g.read_sources(explicit)
    calls = []

    def search(query, language, key, **kwargs):
        calls.append((query, language, key))
        return [{**r, "discovery": "google-scholar-serpapi", "area": query} for r in rows if r["language"] == language]

    monkeypatch.setattr(g, "scholar_search", search)
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(n=20, output=output, lexicons=lexicons, scholar_api_key="private-credential")
    assert {c[1] for c in calls} == {"en", "pt"}
    assert len({c[0] for c in calls}) >= 6
    assert "private-credential" not in "".join(p.read_text() for p in output.rglob("*.json*"))
    calls.clear()
    g.generate(n=20, output=output, lexicons=lexicons, scholar_api_key="private-credential", resume=True)
    assert calls == []


def test_scholar_search_parses_documented_pdf_resources_and_hides_credentials(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return json.dumps(
                {
                    "organic_results": [
                        {
                            "title": "Paper",
                            "result_id": "r1",
                            "link": "https://example.org/paper",
                            "resources": [{"file_format": "PDF", "link": "https://example.org/p.pdf"}],
                        }
                    ]
                }
            ).encode()

    requests = []
    monkeypatch.setattr(g.urllib.request, "urlopen", lambda request, **kwargs: requests.append(request) or Response())
    results = g.scholar_search("ecology", "pt", "secret-key")
    query = g.urllib.parse.parse_qs(g.urllib.parse.urlsplit(requests[0].full_url).query)
    assert query["engine"] == ["google_scholar"] and query["lr"] == ["lang_pt"]
    assert results[0]["pdf_url"] == "https://example.org/p.pdf"
    assert "secret-key" not in json.dumps(results)
    monkeypatch.setattr(
        g.urllib.request,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(g.urllib.error.URLError("https://serpapi.com/?api_key=secret-key")),
    )
    with pytest.raises(g.GenerationError) as error:
        g.scholar_search("ecology", "pt", "secret-key")
    assert "secret-key" not in str(error.value)


def test_manifest_hash_and_dictionary_language_guard(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    with pytest.raises(g.GenerationError, match="must be pt"):
        g.load_inventory(lexicons["en"], "pt")
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(n=1, output=output, lexicons=lexicons, sources=sources, languages=("en",))
    with open(lexicons["en"], "a") as stream:
        stream.write("\n")
    with pytest.raises(g.GenerationError, match="changed"):
        g.generate(n=1, output=output, lexicons=lexicons, sources=sources, languages=("en",), resume=True)


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


def test_single_source_jsonl_and_cli_aliases(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    single = json.loads(sources.read_text())[0]
    sources.write_text(json.dumps(single) + "\n")
    assert len(g.read_sources(sources)) == 1
    monkeypatch.setattr(g, "run_codex", choose_pool)
    report = g.main(
        [
            "--samples",
            "1",
            "--language",
            "en",
            "--inventory",
            lexicons["en"],
            "--sources",
            str(sources),
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert report["accepted"] == 1


def test_wrong_detected_language_rejected(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))

    def wrong(prompt, schema, **kwargs):
        return {**choose_pool(prompt, schema), "detected_language": "pt"}

    monkeypatch.setattr(g, "run_codex", wrong)
    output = tmp_path / "run"
    with pytest.raises(g.GenerationError, match="Not enough usable"):
        g.generate(n=1, output=output, lexicons=lexicons, sources=sources, languages=("en",))
    report = json.loads((output / "generation-report.json").read_text())
    assert "different language" in report["failures"][0]["error"]
    assert report["accepted"] == 0


def test_area_balance_and_duplicate_paper_content(tmp_path, monkeypatch):
    sources = [
        {"language": "en", "area": area, "paper_id": f"{area}{index}"}
        for area in ("science", "arts", "health")
        for index in range(4)
    ]
    balanced = g.balance_sources(sources, 42)
    assert len({s["area"] for s in balanced[:3]}) == 3
    lexicons, manifest = fixture_inputs(tmp_path, languages=("en",), copies=3)
    sources = json.loads(manifest.read_text())
    (tmp_path / sources[1]["path"]).write_bytes((tmp_path / sources[0]["path"]).read_bytes())
    manifest.write_text(json.dumps(sources))
    monkeypatch.setattr(g, "run_codex", choose_pool)
    with pytest.raises(g.GenerationError, match="Not enough usable"):
        g.generate(n=30, output=tmp_path / "run", lexicons=lexicons, sources=manifest, languages=("en",))
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["accepted"] == 20
    assert any("Duplicate" in f["error"] for f in report["failures"])


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
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    output = tmp_path / "run"
    monkeypatch.setattr(g, "run_codex", choose_pool)
    g.generate(n=1, output=output, lexicons=lexicons, sources=sources, languages=("en",))
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
        g.generate(n=1, output=output, lexicons=lexicons, sources=sources, languages=("en",), resume=True)
    assert {name: (output / name).read_bytes() for name in names} == before


def test_resume_requires_manifest_and_original_discovery_plan(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",))
    plan = g.read_sources(sources)
    monkeypatch.setattr(g, "discover_sources", lambda *a, **kw: plan)
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(n=1, output=output, lexicons=lexicons, languages=("en",))
    before = (output / "samples.jsonl").read_bytes()
    discovery_path = output / "cache" / "discovery.json"
    discovery = discovery_path.read_bytes()
    discovery_path.unlink()
    monkeypatch.setattr(
        g, "discover_sources", lambda *a, **kw: pytest.fail("missing discovery must not silently search again")
    )
    with pytest.raises(g.ResumeError, match="discovery plan is missing"):
        g.generate(n=1, output=output, lexicons=lexicons, languages=("en",), resume=True)
    discovery_path.write_bytes(discovery)
    (output / "manifest.json").unlink()
    with pytest.raises(g.ResumeError, match="original manifest"):
        g.generate(n=1, output=output, lexicons=lexicons, languages=("en",), resume=True)
    assert (output / "samples.jsonl").read_bytes() == before


def test_remote_pdf_download_resume_raw_hash_and_redirect_scheme_guards(tmp_path, monkeypatch):
    lexicons, manifest = fixture_inputs(tmp_path, languages=("en",))
    manifest.write_text(
        json.dumps([{"title": "Fixture en remote", "language": "en", "pdf_url": "https://example.invalid/p.pdf"}])
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return b"%PDF-offline-fixture"

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(g.urllib.request, "build_opener", lambda handler: Opener())
    monkeypatch.setattr(
        g, "extract_pdf", lambda path: "An authentic bank passage with enough words to select an exact occurrence."
    )
    monkeypatch.setattr(g, "run_codex", choose_pool)
    output = tmp_path / "run"
    g.generate(n=1, output=output, lexicons=lexicons, sources=manifest, languages=("en",))
    pdf = next((output / "cache").glob("*.pdf"))
    pdf.write_bytes(b"%PDF-changed-fixture")
    before = (output / "samples.jsonl").read_bytes()
    with pytest.raises(g.ResumeError, match="validation failed"):
        g.generate(n=1, output=output, lexicons=lexicons, sources=manifest, languages=("en",), resume=True)
    assert (output / "samples.jsonl").read_bytes() == before
    handler = g.HttpPdfRedirects()
    request = g.urllib.request.Request("https://example.invalid/p.pdf")
    for url in ("file:///private.pdf", "ftp://example.invalid/p.pdf"):
        with pytest.raises(g.GenerationError, match="redirect must use"):
            handler.redirect_request(request, None, 302, "", {}, url)
    assert handler.redirect_request(request, None, 302, "", {}, "https://example.invalid/new.pdf").full_url.endswith(
        "new.pdf"
    )


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
    g.generate(n=1, output=output, lexicons=lexicons, sources=manifest, languages=("en",))
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


def test_missing_earlier_slot_cannot_reuse_a_later_cached_paper(tmp_path, monkeypatch):
    lexicons, sources = fixture_inputs(tmp_path, languages=("en",), copies=2)
    output = tmp_path / "run"
    monkeypatch.setattr(g, "run_codex", choose_pool)
    g.generate(n=20, output=output, lexicons=lexicons, sources=sources, languages=("en",))
    slot_path = output / "cache" / "slot-000000.json"
    first = json.loads(slot_path.read_text())["paper"]["paper_id"]
    slot_path.unlink()

    def inaccessible(source, *args):
        if source["paper_id"] == first:
            raise g.GenerationError("First source unavailable")
        pytest.fail("A later cached paper must remain reserved for its own slot")

    monkeypatch.setattr(g, "acquire_source", inaccessible)
    monkeypatch.setattr(g, "run_codex", lambda *a, **kw: pytest.fail("No unused source remains"))
    with pytest.raises(g.GenerationError, match="Not enough usable"):
        g.generate(n=20, output=output, lexicons=lexicons, sources=sources, languages=("en",), resume=True)
    assert (output / "cache" / "slot-000001.json").exists()
