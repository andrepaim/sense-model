# Generate paper-passage review data

`wsd generate` (also `python scripts/generate_dataset.py`) builds **unlabelled** `Example` records from authentic extracted paper text. It asks the local authenticated Codex CLI to select dictionary occurrences and judge difficulty. It never asks the model to compose contexts or definitions. All `gold_ids` remain `null` until a separate annotation step.

`--n` / `--samples` is the total across languages. The generator accepts exactly `ceil(n/10)` papers, with 5 easy, 3 medium and 2 hard occurrences per full paper, in that order. The last paper's schedule is truncated to reach exactly `n`; for example `n=17` produces EN 10 + PT 7 and totals 10 easy, 5 medium, 2 hard. The default paper rotation is `en,pt`, starting with English. Thus `n<=10` uses one English paper. Use `--languages pt,en` to start with Portuguese or `--language pt` for Portuguese only. Difficulty and language detection are model judgments requiring review, not benchmark labels.

## Local prerequisites

Install the project and the optional PDF parser, or install Poppler's `pdftotext` binary. `pdftotext` is preferred when available; `pypdf` is the fallback. Both run in a subprocess with a 120-second extraction timeout. Scanned PDFs without extractable text require a separately obtained OCR text file; the generator does not perform OCR.

```bash
uv sync --extra pilot
codex login
uv run wsd config init
```

The default Codex model is `gpt-6-luna`. `--model` or `[codex].model` overrides it. An unavailable model, missing CLI or failed authentication stops the run with an actionable error and retains completed papers; there is no fallback model. `[codex].timeout_seconds` controls each selection call (default 180).

Supply pinned, locally acquired dictionaries in the existing lexicon JSONL format. Each entry needs `lemma`, `pos`, `language` and `candidates: [{id, definition}]`. The generator pools **all POS entries for each lemma** and offers lemmas with at least two candidates. It preserves first-seen candidate order, IDs and definitions, deduplicates identical IDs, rejects conflicting definitions, and rejects pooled inventories exceeding 127 candidates. It never narrows a candidate list using a guessed POS; metadata records `pos: "all"` and `candidate_scope: "all_inventory_pos"`. English dictionaries must declare English and Portuguese dictionaries must declare Portuguese. The entire dictionary file's SHA-256 is recorded. Dictionaries are not generated, translated or downloaded automatically. See [data preparation](data.md) for exporting already-installed versioned WordNet dictionaries. Check Portuguese gloss coverage yourself; language declarations alone do not establish it.

Configure absolute dictionary paths in the private `~/.config/sense-model/config.toml`, or pass `--lexicon-en` and `--lexicon-pt`:

```toml
[generation]
lexicon_en = "/data/dictionaries/english-v1.jsonl"
lexicon_pt = "/data/dictionaries/portuguese-v1.jsonl"
samples = 1000
samples_per_paper = 10
seed = 42
search_source = "google-scholar"
scholar_api_key = "YOUR_PRIVATE_SERPAPI_KEY"
```

Keep real keys in this private file or use `SERPAPI_API_KEY`. The generator does not place provider credentials in manifests, caches, prompts or logs.

## Default Google Scholar discovery

```bash
wsd generate --n 100 --output data/pilot/paper-review-v1
```

Discovery uses the documented [SerpAPI Google Scholar engine](https://serpapi.com/google-scholar-api), with `engine=google_scholar`, query, language restrictions and the result's PDF resources. It requires your own SerpAPI credentials and quota. It does not scrape Scholar HTML or bypass CAPTCHA, paywalls or access restrictions. Accessible PDF resources are downloaded directly with a 40 MiB cap and finite timeouts; redirects must also use HTTP or HTTPS. Inaccessible, non-PDF, empty or unsuitable papers are reported and skipped in favor of reserve results.

The seed shuffles predefined English and Portuguese research areas covering ecology, economics, computing, psychology, materials, health, language, education, physics, history, geography and agriculture. Discovery queries at least three areas per language where available. Acquisition rotates across areas, rather than taking every result from the first query. The seed also controls occurrence pools. Results themselves can change over time: resuming reuses the cached discovery plan rather than rerunning searches. No claim of uniform sampling of Google Scholar is made.

```bash
wsd generate --n 40 --areas "ecology,economics,linguistics" --max-papers 20 --output data/pilot/custom-areas
```

`--areas` overrides the query areas for each requested language. `--max-papers` bounds newly attempted acquisitions in a run, including rejected papers; it must be at least `ceil(n/10)`. Successful cached papers do not consume that attempt bound during resume. Without this flag, all discovered reserve sources can be attempted. Exact paper counts refer to accepted papers; discovery can inspect additional candidates.

## Explicit sources

Use `--sources` to replace Scholar discovery entirely. This supports JSON arrays, `{ "papers": [...] }`, a single paper object, or JSONL. Local paths resolve relative to the manifest. Each source requires title, language, and either a local UTF-8 `.txt`/`.text` or PDF `path`, or an HTTP(S) `pdf_url`. Supply citation URL, research area, authors, version and license if known; these source fields are retained in `papers.jsonl`. An unknown license is recorded as unspecified and needs review before redistribution.

```json
[
  {"title": "English source", "language": "en", "path": "papers/english.pdf", "area": "ecology", "license": "YOUR_VERIFIED_LICENSE", "url": "https://example.org/english"},
  {"title": "Fonte portuguesa", "language": "pt", "path": "papers/portuguese.txt", "area": "educação", "license": "YOUR_VERIFIED_LICENSE", "url": "https://example.org/portuguese"}
]
```

```bash
wsd generate --n 20 --sources sources.json --output data/pilot/explicit-review
wsd generate --samples 10 --language pt --inventory /data/pt-dictionary.jsonl --sources sources.json --output data/pilot/pt-review
```

The declared language is checked against Codex's detected language in the actual extracted passages. Wrong-language batches are rejected. Candidate matching uses exact dictionary lemmas and phrases (case insensitive); it does not infer inflected forms. Papers with insufficient ambiguous dictionary occurrences are rejected instead of inventing data. Include reserve papers with adequate coverage.

## Artifacts, validation and resume

Each accepted passage is an unchanged substring of cached extracted text. Target offsets are Python Unicode code points with exclusive end, not UTF-8 bytes or JavaScript UTF-16 offsets. The generator derives offsets itself and validates the existing `Example` schema. Selection references a finite pool of exact occurrences; the model cannot replace text, target offsets or dictionary candidates. Distinct occurrences within a paper are required. Identical downloaded files and extracted paper text are rejected across accepted papers, even if source URLs differ.

The output directory contains:

- `samples.jsonl`: existing portable `Example` schema, unlabelled and unreviewed.
- `papers.jsonl`: accepted paper metadata, source/extracted text hashes, extractor and cache path.
- `manifest.json`: settings, pinned dictionary/input hashes, schedule, run fingerprint and status.
- `generation-report.json`: requested/accepted counts, language/difficulty counts and rejected sources.
- `events.jsonl`: structured events also printed live to stderr, including searches, extraction, selection retries, acceptance, cache hits and failures.
- `cache/`: downloaded PDFs, exact extracted UTF-8 text, extraction metadata, discovery plan and atomic per-paper completed batches.

`provenance.generation` stores difficulty, reason, paper ID, model, prompt version, pooled lemma scope, seed, source passage offsets and content hashes. It also records research area and the extracted page number (counted from PDF form-feed page delimiters). Top-level provenance includes the paper title, citation URL as both `paper_url` and `source_url`, area, domain and page for audit and filtering. `provenance.document_id` groups all examples from one paper for leakage-safe splits. The model's selected meaning is deliberately not collected during generation.

```bash
wsd generate --n 100 --output data/pilot/paper-review-v1 --resume
```

Use the same flags and unchanged inputs. Resume skips completed paper batches and reuses cached extraction and discovery. The original manifest and discovery plan are required. Each batch is tied to the run fingerprint and revalidated against the planned source, raw file and extracted-text hashes, exact passage substring, languages, full pinned candidate list, unlabeled status and difficulty schedule. A batch's self-checksum alone cannot approve altered content. Invalid checkpoints are rejected before replacing existing exports. Settings, source contents and dictionary hashes must match; changed inputs require a new output directory. A failed acquisition run writes incomplete status and partial outputs, never claims it reached the request. Completed paper batches survive interruption even if final export was interrupted. A directory with existing contents requires `--resume`. Generation creates review data; it does not train, upload or publish it.

Offline tests mock Scholar responses and local Codex selections and exercise exact Unicode offsets, fixed dictionary candidates, language checks, paper quotas, truncation, failure recovery, credential handling and PDF extraction. Fixture text in those tests is synthetic test input, not a benchmark or scientific result.
