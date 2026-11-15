# Generate samples from a local paper list

The input is a JSON manifest listing local papers. The argument controls **samples
per paper**, default **10**. Every listed paper is processed in manifest order.
For example, 12 papers × 10 samples requests 120 examples; 12 × 20 requests 240.
There is no total-sample target, automatic paper discovery, or PDF download.

JSON is useful here because each path can have language and source metadata.
Generated datasets remain JSONL, matching the labeler, reviewer and model tools.

## Files and commands

```text
sense-model/
  papers.json
  papers/
    attention.pdf
    aprendizagem.pdf
```

Copy `papers.example.json` to `papers.json`, put your PDFs in `papers/`, and edit:

```json
{
  "papers": [
    {"path": "papers/attention.pdf", "language": "en"},
    {"path": "papers/aprendizagem.pdf", "language": "pt"}
  ]
}
```

Paths resolve relative to the JSON file. The minimum fields are `path` and
`language` (`en` or `pt`). `title` defaults to the filename stem. Optional fields
include `area`, `url`, `license` and citation metadata. A JSON array of entries
also works. PDFs and `papers.json` are ignored by Git; the template is tracked.
UTF-8 text files are supported for already extracted text and testing.

```sh
uv sync --extra pilot
uv run wsd config init
# Configure generation.lexicon_en / generation.lexicon_pt, then:
uv run wsd generate --input papers.json
uv run wsd label
uv run wsd annotate --import-file data/pilot/generated/labeled/samples.jsonl --open
```

Specify a different count or output:

```sh
uv run wsd generate --input papers.json --samples-per-paper 20 --output data/pilot/custom
# Equivalent standalone script:
uv run python scripts/generate_dataset.py --input papers.json --samples-per-paper 20 --output data/pilot/custom
```

`--sources` is an alias for `--input`. The default input is `papers.json`; the
default output is `data/pilot/generated`. Old `--n`, `--samples`, language rotation,
research-area search and paper-limit arguments no longer apply.

## Samples, definitions and difficulty

At the default 10, each paper contributes 5 easy, 3 medium and 2 hard examples.
Other counts use 50%/30%/20% proportions, rounding by largest fractional remainder;
ties use easy, medium, hard order. Twenty requests 10/6/4. Every paper has the
same requested count; there is no truncated final paper.

Codex selects exact dictionary occurrences from the extracted text. It cannot
invent or rewrite passages, target spans or definitions. All POS entries for a
lemma are pooled; repeated IDs are deduplicated and conflicting definitions are
rejected. Offsets are Unicode code points, end-exclusive. Language and difficulty
are model judgments for your review. `gold_ids` remain null during generation.

Supply pinned dictionary JSONL paths in `generation.lexicon_en` and
`generation.lexicon_pt`, or via `--lexicon-en` / `--lexicon-pt`. Each dictionary
row contains `lemma`, `pos`, `language`, and `candidates: [{id, definition}]`.
With a single-language manifest, `--inventory dictionary.jsonl` is a shortcut.
See [data preparation](data.md) for exporting an installed WordNet edition.
Portuguese definitions need actual coverage, not only a language declaration.

Codex uses your local CLI login with hosted inference. The default model is
`gpt-6-luna`; `--model` overrides it. There is no silent fallback. PDF extraction
uses Poppler's `pdftotext` when available, otherwise `pypdf` from the `pilot` extra.
Both run with a bounded timeout. Scanned PDFs need separately obtained OCR text.

## Configuration, logs and resume

The private home config supports:

```toml
[generation]
input = "papers.json"
output = "data/pilot/generated"
samples_per_paper = 10
seed = 42
lexicon_en = "/data/dictionaries/en.jsonl"
lexicon_pt = "/data/dictionaries/pt.jsonl"
```

CLI flags override config. Jev credentials are needed by the labeler, not the
generator. No search API key is needed. Previous search settings are ignored.

Artifacts: `samples.jsonl`, `papers.jsonl`, `manifest.json`,
`generation-report.json`, `events.jsonl`, and a local extraction/checkpoint cache.
Logs stream progress to stderr and JSONL. The report records requested versus
accepted samples and per-paper failures. An unreadable or unsuitable paper is
reported, later listed papers are still attempted, and the CLI exits nonzero if
the run is incomplete. Missing input files fail validation before model calls.
The generator never substitutes a different paper or invents missing examples.

```sh
uv run wsd generate --input papers.json --samples-per-paper 20 --output data/pilot/custom --resume
```

Resume validates the manifest, source/dictionary hashes and cached examples,
skips successful papers, and retries failed papers. Use unchanged inputs and
settings. Changed paper lists, file contents or sample counts require a fresh
output directory. Older generation checkpoints are not compatible with this
manifest-per-paper contract. Invalid checkpoints cannot overwrite prior exports.

Source licenses are recorded for audit; check source and dictionary rights before
redistributing examples. Software fixtures and provisional difficulty labels are
not benchmark evidence.
