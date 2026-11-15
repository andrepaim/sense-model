# Collecting the first natural pilot pool

The pilot targets 100 **reviewed** examples per language. The collector prepares
an unlabeled pool, initially up to 250 per language; collecting is not annotation.
No documents or dictionary data are downloaded implicitly. Select sources whose
licenses permit your intended use and record the exact revision. Dictionaries,
source passages, code, and annotation licenses remain separate.

## Prepare sources

One local UTF-8 JSON object per document, per line:

```json
{"id":"document-001","text":"The original licensed document text.","language":"en","domain":"general","source":"publisher-or-corpus","source_url":"https://example.org/document","revision":"pinned-version-or-date","license":"YOUR_VERIFIED_LICENSE","license_url":"https://example.org/license"}
```

Required fields are `id`, `text`, `language`, `domain`, `source`, `source_url`,
`revision`, `license` and `license_url`, all nonempty strings. Keep independent
publisher/document identities stable. Use `pt-BR` and `pt-PT` distinctly. The
collector is an acquisition-independent local tool; it does not imply any license
verification has been performed. It saves source-document and file checksums.

## Prepare the pinned inventory

Use the local lexicon JSONL from `wsd prepare lexicon`, after auditing definition
language and coverage, or a versioned inventory with this shape:

```json
{"lemma":"bank","pos":"n","language":"en","surface_forms":["banks"],"candidates":[{"id":"source:river","definition":"The inventory's original definition."},{"id":"source:financial","definition":"The inventory's original financial definition."}]}
```

Every candidate stays in canonical input order; no model screens or ranks them.
`surface_forms` is optional and lists explicit inflections or aliases. The
collector performs literal Unicode token matching, including overlapping word
and contiguous phrase targets. It pools entries across POS for a matched form
because it does not assume a POS model. Record this inventory policy consistently
when evaluating deployment. Conflicting same-ID definitions and inventories over
127 candidates are rejected rather than silently pruned. Language declarations
are required; an inventory tagged Portuguese still needs a human gloss audit.

## Collect, inspect and import

```sh
wsd pilot collect --documents data/raw/english-documents.jsonl \
  --lexicon data/raw/english-inventory.jsonl --definition-language en \
  --inventory-version YOUR_PINNED_VERSION --inventory-license YOUR_VERIFIED_LICENSE \
  --limit-per-language 250 --max-per-document 5 --max-per-lemma 3 \
  --seed 42 --output data/prepared/pilot-en.jsonl

wsd pilot collect --documents data/raw/portuguese-documents.jsonl \
  --lexicon data/raw/portuguese-inventory.jsonl --definition-language pt-BR \
  --inventory-version YOUR_PINNED_VERSION --inventory-license YOUR_VERIFIED_LICENSE \
  --output data/prepared/pilot-pt.jsonl

wsd pilot audit --input data/prepared/pilot-en.jsonl \
  --output data/prepared/pilot-en.audit.json
```

Output examples contain original sentence text and exact code-point spans,
`gold_ids: null`, full candidate definitions and source/inventory provenance.
A sibling `.manifest.json` records checksums, language/domain bucket counts,
shortfalls and filters. Import the resulting JSONL with the annotation interface.
Outputs are versioned by using new paths; existing output files are not replaced.

Sampling is deterministic from a seed. Ten percent of requested slots are
initially reserved for single-candidate sanity checks when available; remaining
slots prioritize ambiguous dictionary forms. Per-document and per-lemma caps
apply, with shortfalls reported instead of duplicating data. Document pools must
be sufficiently broad (at least 50 documents to fill 250 examples under a cap of
five). The collector cannot infer which 20 examples will be genuine dictionary
gaps; those emerge from review. Do not remove correct definitions to manufacture
natural none-fits examples. Conservative sentence splitting can mishandle
abbreviations; mark extraction issues during review.

## Jev protocol checked for this implementation

Direct TypeSafe: `https://api.typesafe.ai/v1/systemone`, Bearer authorization,
`state` plus a `questions` map; the `meaning` Choice question has generated
opaque slots and descriptions. Two additional choices distinguish absent meaning
and insufficient context. Candidate IDs, source URLs, provenance, human votes and
reviewer notes are not sent. The returned resolved model, distribution, confidence,
limited token usage, prompt version and a request fingerprint are stored as
assistance metadata. Jev does not generate an explanation. There is no automatic
retry; request failures preserve the existing human annotation.

OpenRouter supports the same `/v1/systemone` protocol under its `/api` base URL.
Select the provider and matching key explicitly. `typesafe/jev-1.13` is the initial
OpenRouter default; the UI model field can select a later available version.
Direct `jev-latest` is an alias; the resolved response model is always recorded.

Primary sources checked:

- [TypeSafe System One API](https://docs.typesafe.ai/api/system-one.md)
- [TypeSafe OpenAPI](https://api.typesafe.ai/openapi.json)
- [Choice primitive](https://docs.typesafe.ai/primitives/choice.md)
- [OpenRouter's official Jev integration guide](https://openrouter.ai/blog/insights/what-is-jev/)

Adapter automated tests use the documented raw response shape and mocked
transport. No live paid request has been made with a user key in this session.
