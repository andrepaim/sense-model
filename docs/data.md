# Local preparation and provenance

Training and evaluation read JSONL examples with exact code-point target offsets, dictionary candidates and `gold_ids`. `gold_ids: []` explicitly means no supplied definition fits; omitted/null means unlabeled and requires teacher scores for training. Several acceptable gold IDs are supported. Every real record should identify `source`, `version`, `license`, `document_id`, and annotation type (`human`, `synthetic`, `translated`, or `teacher`) in provenance. Synthetic/translated variants should reference `original_id`/`translation_of`/`synthetic_parent`. Do not invent these labels or treat generated data as human gold.

A local lexicon is JSONL with `lemma`, coarse `pos` (`n`, `v`, `a`, `r`), declared `language`, and `candidates: [{id, definition}]`. Candidate IDs must match annotation IDs. Satellite adjectives are grouped with adjectives. Dictionary frequency order should be preserved for the first-meaning baseline.

Export an already installed `wn` lexicon (install the optional `prepare` dependencies first):

```sh
wsd prepare lexicon --wn-spec omw-en:1.4 --language en --output data/english-lexicon.jsonl
wsd prepare lexicon --wn-spec own-pt:1.0.0 --language pt --output data/portuguese-lexicon.jsonl
```

These commands never download lexicons. Acquire resources explicitly under their own licenses. `wn` installation and package availability do not prove Portuguese gloss coverage. Review language and definition quality before training; the Portuguese audit reports only declared language and available gloss counts.

Prepare SemCor or Raganato-format evaluation material:

```sh
python scripts/prepare_english.py --corpus /data/SemCor/semcor --sense-index /data/WordNet-3.0/dict/index.sense --lexicon data/english-lexicon.jsonl --source SemCor --version YOUR_VERSION --license YOUR_VERIFIED_LICENSE --output data/english-all.jsonl
```

Sense key mappings default to `omw-en-OFFSET-POS`; choose `--namespace` only when the supplied candidate IDs use that namespace. Mapping errors and absent inventories are counted in the output manifest and never silently converted to rejection examples. XML reconstructs target offsets directly, so repeated occurrences stay distinct. Cross-edition mappings require explicit review.

Portuguese annotated context rows add `lemma` and `pos` to `id,text,target,gold_ids,provenance`. Join them with a Portuguese inventory:

```sh
python scripts/prepare_portuguese.py --lexicon data/portuguese-lexicon.jsonl --contexts /data/reviewed-contexts.jsonl --source YOUR_SOURCE --version YOUR_VERSION --license YOUR_VERIFIED_LICENSE --output data/portuguese-all.jsonl
```

Omit `--contexts` to produce a lexicon audit only. English fallback glosses must be kept separately. Translated English labels are not automatically Portuguese gold labels. For synthetic contexts, use a separate offline generator of your choice, save structured rows with generation provenance, then validate and review before this import. This project does not run paid generation or silently download private corpora.

```sh
wsd prepare split --input data/portuguese-all.jsonl --output data/portuguese --seed 42
wsd prepare validate --split train=data/portuguese/train.jsonl --split dev=data/portuguese/dev.jsonl --split calibration=data/portuguese/calibration.jsonl --split test=data/portuguese/test.jsonl --output data/portuguese/manifest.json
```

Documents, duplicate contexts, original/translation/synthetic families stay together. Validation also flags high-overlap long passages using five-word shingles; review/merge such groups if the split fails. Freeze test after review. Never include test material in teacher scoring or augmentation. Manifests contain source file hashes and counts. For nonredistributable inputs, publish scripts/manifests rather than corpus text.

For the manually annotated WordNet Gloss Corpus, use the English preparation command with `--glosstag /data/WordNet-3.0/glosstag` instead of `--corpus`. Automatic tags are excluded by default. The adapter derives from upstream's loader and handles contiguous collocations. Provenance and dictionary hashes accompany the prepared examples.

Optional explicit local generation:

```sh
python scripts/generate_contexts.py --lexicon data/portuguese-lexicon.jsonl --model YOUR_OLLAMA_MODEL --language pt-BR --limit 100 --output data/unreviewed-portuguese-contexts.jsonl
```

This invokes the endpoint you select (`--endpoint`, local Ollama by default). It stores prompts/model/seed, rejects malformed spans, refuses overwrite, and marks every accepted row synthetic and unreviewed. Review each context for the intended sense before importing; schema validation cannot establish semantic correctness. This is supplementary data, never a replacement for natural human-annotated evaluation.
