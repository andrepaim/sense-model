# Wiktionary round 1: Ettin-150m-WSD fine-tuned on English Wiktionary

On 2026-11-25 and 26, `sign/Ettin-150m-WSD` was fine-tuned on English Wiktionary's own usage examples and quotations,
with `sign/Ettin-1B-WSD` as a teacher, for the Chrome extension
[Contextual Dictionary](https://github.com/andrepaim/contextual-dictionary). The run that ships is published as
[`andrepaim/Ettin-150m-WSD-wiktionary`](https://huggingface.co/andrepaim/Ettin-150m-WSD-wiktionary/tree/0edcbab04bb30eadeeef6babb26e8055b173b8d7)
(revision `0edcbab04bb30eadeeef6babb26e8055b173b8d7`), and its INT8 graph is the extension's release `assets-v2`. It was first published at revision
`affc67b9a87c51a07ce642726cee993c593ad744` (2026-11-26); `0edcbab0` (2026-11-27, `wsd publish` from this repository) has the same weights and ONNX
graphs and adds the extension's `tokenizer.json`, the card of `releases/Ettin-150m-WSD-wiktionary/` and the run's
records (`training_config.json`, `data_manifest.json`, `evaluation.json`). The
training and evaluation data are the dataset
[`andrepaim/wiktionary-wsd`](https://huggingface.co/datasets/andrepaim/wiktionary-wsd/tree/84b6444e83219aaa76c29a517a64a640c1df9a46)
(revision `84b6444e83219aaa76c29a517a64a640c1df9a46`).

This page holds the model-level method and numbers and the rerun with `wsd` commands. The round ran with scripts of the
extension's repository over this package (`validation/training/` as it was at
[`6cdc449`](https://github.com/andrepaim/contextual-dictionary/tree/6cdc449/validation/training), removed in `7947129`); their options are now part of `wsd train`, `wsd teacher`, `wsd score`,
`wsd rank`, `wsd export` and `wsd publish`, and "Checked after the move" below shows that these commands give the
round's results again. What the extension's reader gets (accuracy in the browser at its time budget, latency, threads,
memory, the cards) is measured in the extension's
[`docs/validation.md`](https://github.com/andrepaim/contextual-dictionary/blob/7947129/docs/validation.md), "Training round
1 (assets-v2)", with the scripts of [`validation/candidate-model/`](https://github.com/andrepaim/contextual-dictionary/tree/7947129/validation/candidate-model), and
is not repeated here. The data were built by the extension's builders (`validation/build-train-set.ts`,
`validation/build-unlabelled-set.ts`) and exported to the dataset by
[`validation/dataset/`](https://github.com/andrepaim/contextual-dictionary/tree/7947129/validation/dataset) (all at commit `7947129`).

**Where the numbers come from.** Every number under "Data", "Recipe", "Pilot", "Runs", "Development", "Test", "Which run
ships" and "Export" is copied from that section of the extension's `docs/validation.md` as it was at commit
[`6cdc449`](https://github.com/andrepaim/contextual-dictionary/blob/6cdc449/docs/validation.md), which counts the round's files (at `7947129` the section keeps a
summary and points here). The development and test tables, the export's outlier ratios, sizes and INT8 check, and
the teacher cache were also recomputed from the round's kept files on 2026-11-27 with the commands of this package and
came out the same ("Checked after the move"). The split-check counts under "Limits" are new, computed on 2026-11-27.

## Data

**Labelled examples.** From the Kaikki.org extract of English Wiktionary of 2026-10-30 (SHA-256
`4d03b6cba283744756c57ff661c2c6cbfe34fcf8ff513a966576cf7d97e91a99`). Wiktionary's editors attach every usage example and
quotation to the meaning it illustrates; that attachment is the label. The extension's builders keep English records
whose headword is one lower-case word and whose entry keeps at least two meanings in one part of speech, look the word up
in the extension's own dictionary pack, and build the prompt as the extension builds it: every usable example, at most 12
per entry (seeded), in two forms, the extension's defaults (at most 24 meanings and 1,600 characters, `24|c1600`) and every
meaning (`127`). The headwords of the extension's English Wiktionary and hand-written test sets (2,197) are held out in
every part of speech (3,754 entries skipped, and 2,274 items dropped because their lookup returned an entry of a held-out
lemma). The other headwords are split by `sha256("cd-train-r1\n" + headword)`: the first 4% of the hash space is
development.

| | Items | Headwords | 2 meanings | 3–4 | 5–8 | 9–16 | 17–32 | 33+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 125,031 | 34,993 | 48,394 | 41,108 | 25,404 | 9,846 | 279 | 0 |
| Development | 4,951 | 1,378 | 1,863 | 1,756 | 952 | 380 | 0 | 0 |

Training items: 69,789 nouns, 24,365 verbs, 25,837 adjectives, 3,462 adverbs, 1,578 others; 96,528 quotations and 28,503
usage examples. The held-out headwords take nearly all entries with many meanings: 464 entries with 17–32 meanings and
all 63 with 33+ are held out, so 45 such entries (279 items) remain for training and none for development.

Each prompt became a sense-model example, and `format_prompt` rebuilt every prompt byte for byte. The no-cap form differs
from the defaults for 5.6% of the items. Where the cap leaves the labelled meaning out (1,667 training items), the prompt
is not made a none-fits example (`gold_ids: []` would say no offered meaning fits, which is not true there); it is used
with the teacher's answers only (`gold_ids: null`).

**Leak checks** (when the data was built): no training or development headword, and no lemma the unlabelled stream looked
up, is held out; training and development headwords are disjoint. 46 training passages and 2 development passages were
equal to a Wiktionary test passage after normalisation, and 557 more shared a run of ten words with a test passage; all
605 were dropped, and 4 identifiers that appeared twice were kept once. The same check on the final files found no equal
passage and no shared run of ten words with any of the three test sets. Final training examples: 129,093 labelled and
1,658 teacher-only; the 4,951 development items become 4,931 and the 125,031 training items 124,442.

**Unlabelled stream** (teacher only). One shard of English Wikipedia,
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, file `20231101.en/train-00001-of-00041.parquet` (SHA-256
`6cd57a5825312ca966313d4a1b614690b5c4fb53cee9b124bc0e999977eb36cf`): a seeded sample of 60,000 prose paragraphs of
120–700 characters, one lower-case word per paragraph whose lemmas are neither held out nor on the development side, at
most 15 per lemma: 47,910 items, 55,138 examples.

**Files** (the dataset; the configurations name them relative to its root, its layout is that of the round's working
folder):

| File | Rows | Used for |
| --- | ---: | --- |
| `data/train-labelled.clean.jsonl` | 129,093 | training, all runs (labels only reads it alone) |
| `data/train-notoffered.clean.jsonl` | 1,658 | training with the teacher, `gold_ids` null |
| `data/unlabelled.clean.jsonl` | 55,138 | training with the teacher, `gold_ids` null |
| `data/dev-c1600.clean.jsonl`, `data/dev-127.clean.jsonl` | 4,846, 4,907 | the trainer's development forms (2,000 of each) |
| `data/dev-notoffered.clean.jsonl` | 61 | with the two above, every development prompt, for scoring |
| `data/dev-items.jsonl`, `reports/excluded-overlap-ids.txt` | 4,951, 605 | the development table (4,931 items) |
| `teacher/cache-stage2-T2.jsonl` | 185,889 | the teacher cache at T = 2 |
| `teacher/teacher-bf16.jsonl` | 190,855 | the teacher's raw bf16 logits by prompt SHA-256 |
| `eval/{wiktionary,written,simple}-{c1600,127}.jsonl` | 2,287 / 104 / 4,157 per form | the test prompts, for scoring |
| `eval/items/{wiktionary,written,simple}-items.jsonl` | 2,300 / 104 / 4,167 | the test tables |

## Recipe

As the upstream card describes its own: loss 0.3 × label cross-entropy (label smoothing 0.1) + 0.7 × T² × KL to the
teacher at T = 2 (rows without a label: the KL term alone); learning rate 5e-5, cosine schedule, batch 64 (4 × 16
accumulation), seed 42, at most 3 epochs; evaluation every 300 steps on 2,000 development prompts of each form, the best
checkpoint by "labelled meaning first" at the defaults, early stopping after 4 evaluations without improvement. Also:
AdamW as the trainer's default (PyTorch's fused AdamW, betas 0.9 and 0.999, ε 10⁻⁸), weight decay 0.01, gradients
clipped to norm 1; a linear warm-up over the first 10% of the planned steps (606 for labels only, 872 for the other two)
and then the cosine decay towards zero at the planned end, which early stopping never reached; bf16 mixed precision;
candidates in the extension's order; batches grouped by length (transformers' length-grouped sampler, seeded by epoch);
prompts of at most 2,048 tokens (no training prompt was longer); the 2,000 development prompts of each form drawn with
seed 42. Teacher: `sign/Ettin-1B-WSD` revision `e6fc37a7e11fee91a600a0cd8e7423bcc639a1cc`, soft labels in bf16, batches
of at most 8,192 tokens (bf16 agreed with fp32 on 99.4% of top choices, with the same development accuracy; batched
scores equalled one-at-a-time scores). The student: `sign/Ettin-150m-WSD` revision
`8751b577199d1bb95b74fa2457da7065d57100ae`. Every accuracy below is fp32.

Two defects of this package showed up on the way and were worked around in the round's script: under transformers 5 the
loss and gradients came out multiplied by the gradient accumulation steps (fixed in #1), and a resumed run drew a new
batch order (the length-grouped sampler; `--group-by-length` now seeds it by epoch).

The configurations are `configs/wiktionary_round1_labels_only.json`, `configs/wiktionary_round1_teacher_only.json` and
`configs/wiktionary_round1_labels_teacher.json`. They hold every setting above and the ones that only shape the logs
(evaluation batch 16, a log every 10 steps, a baseline evaluation before training). They set `shuffle_examples`: the
round's trainer shuffled the examples once with the seed in every run, the single-file labels-only run included.

## Pilot

On 2026-11-25, labels and teacher on 39,999 of the labelled training examples, one epoch of 625 steps of 64 (about 2
seconds a step), tried the trainer, the teacher cache, the resume and the export. The trainer's development score
(labelled meaning first on 2,000 development prompts whose labelled meaning is offered) went from 69.5% to 74.8%. Its
first attempt, resumed from step 300, trained on a new batch order (training loss 0.772 against 0.834 at step 310 over
the same steps); the second, with the sampler seeded by epoch, resumed onto the same batches (0.8531 against 0.8534).

## Runs

| Run | Data | Stopped at | Chosen checkpoint |
| --- | --- | ---: | ---: |
| Labels only | 129,093 labelled | step 4,200 of 6,054 | 3,000 |
| Teacher only | the labelled prompts without their labels, the teacher-only ones and the unlabelled stream (185,889) | step 4,200 of 8,715 | 3,000 |
| Labels and teacher | the same 185,889, labels where there are | step 4,200 of 8,715 | 3,000 (1.03 epochs) |

On an NVIDIA GeForce RTX 3060 (12 GB): 1 h 36 min (labels only), 50 min and then 1 h 2 min from checkpoint 1,800 (teacher
only, stopped and resumed), 1 h 48 min (labels and teacher), after 2 to 5 minutes of encoding; teacher scores for the
145,729 prompts not scored in the pilot took 58 minutes.

## Development

All 4,931 development items, the labelled meaning's rank within its entry (`wsd rank`; "first" at the defaults, top
three, and first with every meaning offered):

| Model | First | Top 3 | First, no cap |
| --- | ---: | ---: | ---: |
| Dictionary order | 38.1% | 84.3% | 38.1% |
| Untrained (float) | 72.1% | 96.6% | 72.8% |
| Teacher (Ettin-1B-WSD, float) | 79.3% | 97.3% | 80.1% |
| Labels only | **78.6%** | 97.5% | 79.5% |
| Teacher only | 76.7% | 97.3% | 77.7% |
| Labels and teacher | 78.0% | 97.5% | 79.0% |

## Test

The extension's three test sets (described in its `docs/validation.md`, "Method"): 2,300 English Wiktionary items
stratified by meanings in the entry, headwords held out of training; 104 hand-written web-style sentences, **written and
labelled by a model with no human review**; 4,167 Simple English Wiktionary items, a separate wiki with its own
definitions. Scored once per finished run, float PyTorch, labelled meaning first (paired bootstrap 95% interval of the
difference from the untrained float model, 2,000 resamples, seed 1):

| Set, form | Untrained | Labels only | Teacher only | Labels and teacher |
| --- | ---: | ---: | ---: | ---: |
| Wiktionary, defaults | 59.0% | 66.1% (+5.5 to +8.6) | 62.7% (+2.4 to +5.1) | 64.2% (+3.8 to +6.7) |
| Wiktionary, no cap | 66.0% | 75.1% (+7.3 to +10.9) | 71.3% (+3.9 to +6.8) | 73.3% (+5.8 to +8.9) |
| Hand-written, defaults | 65.4% | 69.2% (−1.9 to +9.6) | 67.3% (−2.9 to +6.7) | 70.2% (+0.0 to +9.6) |
| Hand-written, no cap | 75.0% | 80.8% (−1.0 to +12.5) | 79.8% (−1.0 to +10.6) | 83.7% (+3.8 to +14.4) |
| Simple English, defaults | 86.6% | 87.0% (−0.4 to +1.3) | 87.6% (+0.3 to +1.8) | 88.5% (+1.1 to +2.7) |

## Which run ships

Development chose labels only (78.6% against 78.0%), and the round's export step exported that one. The owner chose
labels and teacher **after the test results above were known**. Against labels only, it is better on the hand-written and
Simple English sets and one point worse on the Wiktionary set, and it still answers none-fits (labels only did on no test
item). Because that choice looked at the test sets, the test numbers of the run that ships are not an unbiased estimate of
its accuracy; the difference between the two runs is within the noise of the hand-written set.

## Export

The outlier layers measured for this model on the first 200 development prompts by SHA-256: the input of the feed-forward
output (`mlp.Wo`) peaks at 42.9 (layer 4), 1,089 (7), 24.8 (9), 1,663 (11) and 740 (15) times its 99.9th percentile; the
next are 13.5 (12) and 13.1 (14), under the cut-off of 15. The ONNX export is this package's (float parity with PyTorch:
maximum logit error 4.3 × 10⁻⁶); then `quantize_dynamic` with INT8 weights per channel for the MatMuls except those five,
and the token embedding stored in float16 with a cast to float. INT8 graph: 203,516,610 bytes (SHA-256
`6bac203419a0ad43b09b3806c22abb4870c48adf21657f3ef2a73a5d8369d6e4`); 185,262,587 as the extension's release compresses it
(gzip level 9). INT8 against float on the first 1,000 development prompts by SHA-256: the same top choice for 96.6%.

**Where the weights are.** The float weights are the step-3,000 checkpoint, `model.safetensors` of
`andrepaim/Ettin-150m-WSD-wiktionary` (598,827,368 bytes, SHA-256
`49bc2634bcf696171f367b59cedb961fcd71bb7d825ad6680ef5ed1ed8d25be3`, the `model_revision` of its metadata and of the
extension), with the float ONNX export (`onnx/model.onnx`, SHA-256 `bd31a4ff6fe8d279d29be9ab392cac90b8ff8f74039d4df5a3284b66462c4ab4`),
the INT8 graph and the recipe. The other two runs, the checkpoints (16 GB with their optimiser states) and the training
curves stay on the maintainer's machine.

## Limits

- The development set has no entry with more than 16 meanings, and training has 279 items with 17–32 and none with more:
  what was learnt about long entries comes from the teacher and from short entries.
- The run that ships was chosen after the test results were known.
- The hand-written set is small (its intervals are about ±10 points) and was written and labelled by a model.
- The Wikipedia shard is one of 41, and the unlabelled stream avoids the held-out headwords, which are many of the
  commonest words.
- **Development passages overlap training passages** (found on 2026-11-27). Training and development headwords are
  disjoint, and the build checked training passages only against the test sets. This package's split check
  (`validate_splits`, normalized passage equality and a five-word-shingle Jaccard of at least 0.8 for passages of 15
  words or more) finds 758 findings between the training files and the two development forms, 90 equal passages and
  668 near duplicates. They involve 232 of the 4,931 development items (451 development examples across the two forms)
  and 317 training items (341 training examples): the same quotation filed under another headword. The development numbers may be somewhat optimistic; the choice between runs was made on them. The
  configurations set `allow_split_overlap`, so a run records these findings in its `data_manifest.json` instead of
  stopping.
- One seed per run, one machine.

## Rerun

Needs this package, the dataset and a CUDA GPU (the round used an RTX 3060 with 12 GB: micro-batches of 4 keep the
longest batch at 6.85 GB). The environment the round ran in, and in which the checks below reproduced it, is
[`wiktionary-round1-requirements.txt`](wiktionary-round1-requirements.txt) (Python 3.12, PyTorch 2.14.1 with CUDA 13.0,
transformers 5.23.0, onnx 1.23.2, onnxruntime 1.32.0); its header gives the install commands. The locked environment of
this repository (CPU PyTorch) runs every step that does not need the GPU. Run from the repository root.

```sh
# 0. The data, downloaded explicitly at the pinned revision (or any local folder with the same layout).
DATA=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(
  "andrepaim/wiktionary-wsd", repo_type="dataset", revision="84b6444e83219aaa76c29a517a64a640c1df9a46"))')
R=runs/wiktionary-round1; mkdir -p $R

# 1. Teacher soft labels (58 minutes on the RTX 3060 for 145,729 prompts). The dataset already holds them:
#    with a copy of its raw logits as --scores nothing is scored and the cache comes out byte for byte the same.
#    Its rows predate the recorded scorer identity, so --untagged-scores states what they are.
cp $DATA/teacher/teacher-bf16.jsonl $R-teacher-bf16.jsonl
wsd teacher --model sign/Ettin-1B-WSD --revision e6fc37a7e11fee91a600a0cd8e7423bcc639a1cc --device cuda \
  --dtype bfloat16 --max-tokens 8192 --batch-size 64 --temperature 2 --scores $R-teacher-bf16.jsonl \
  --untagged-scores sign/Ettin-1B-WSD e6fc37a7e11fee91a600a0cd8e7423bcc639a1cc bfloat16 2048 \
  --input $DATA/data/train-labelled.clean.jsonl $DATA/data/train-notoffered.clean.jsonl $DATA/data/unlabelled.clean.jsonl \
  --output $R-cache-T2.jsonl

# 2. The three runs (about 1.5 to 2 hours each). The configurations read teacher/cache-stage2-T2.jsonl of the
#    dataset; pass --teacher-scores $R-cache-T2.jsonl to use the cache of step 1.
for run in labels_only teacher_only labels_teacher; do
  wsd train --config configs/wiktionary_round1_$run.json --data-dir $DATA
done

# 3. Development and test scores, fp32 (about 1 and 2 minutes per model), for the untrained student, the teacher
#    (development only) and each run.
DEV="$DATA/data/dev-c1600.clean.jsonl $DATA/data/dev-127.clean.jsonl $DATA/data/dev-notoffered.clean.jsonl"
score() { wsd score --device cuda --max-tokens 8192 "$@"; }
score --model sign/Ettin-150m-WSD --revision 8751b577199d1bb95b74fa2457da7065d57100ae --input $DEV --output $R/dev-untrained.jsonl
score --model sign/Ettin-1B-WSD --revision e6fc37a7e11fee91a600a0cd8e7423bcc639a1cc --input $DEV --output $R/dev-teacher.jsonl
for run in labels-only teacher-only labels-teacher; do score --model $R/$run/final --input $DEV --output $R/dev-$run.jsonl; done
for s in wiktionary written simple; do
  score --model sign/Ettin-150m-WSD --revision 8751b577199d1bb95b74fa2457da7065d57100ae \
    --input $DATA/eval/$s-c1600.jsonl $DATA/eval/$s-127.jsonl --output $R/test-untrained.jsonl
  for run in labels-only teacher-only labels-teacher; do
    score --model $R/$run/final --input $DATA/eval/$s-c1600.jsonl $DATA/eval/$s-127.jsonl --output $R/test-$run.jsonl
  done
done

# 4. The development table and the test table with its intervals (seconds; no model).
wsd rank --items $DATA/data/dev-items.jsonl --exclude $DATA/reports/excluded-overlap-ids.txt --output $R/dev.json \
  --scores untrained=$R/dev-untrained.jsonl teacher=$R/dev-teacher.jsonl labels-only=$R/dev-labels-only.jsonl \
  teacher-only=$R/dev-teacher-only.jsonl labels-teacher=$R/dev-labels-teacher.jsonl
for s in wiktionary written simple; do
  wsd rank --items $DATA/eval/items/$s-items.jsonl --baseline untrained --output $R/test-$s.json \
    --scores untrained=$R/test-untrained.jsonl labels-only=$R/test-labels-only.jsonl \
    teacher-only=$R/test-teacher-only.jsonl labels-teacher=$R/test-labels-teacher.jsonl
done

# 5. Export with the recipe and the INT8 check (about 9 minutes, mostly the check on the CPU).
wsd export --model $R/labels-teacher/final --fixtures data/fixtures/smoke.jsonl --output $R/export-labels-teacher \
  --quantize --per-channel --float16-embedding --outlier-examples $DEV --outlier-device cuda --check-examples $DEV

# 6. The repository layout, validated and reloaded offline; nothing is uploaded.
wsd publish --assemble releases/Ettin-150m-WSD-wiktionary/release.json --export $R/export-labels-teacher \
  --data-manifest $R/labels-teacher/final/data_manifest.json --artifact $R/release \
  --evaluation development=$R/dev.json test-wiktionary=$R/test-wiktionary.json test-written=$R/test-written.json \
  test-simple=$R/test-simple.json
# Publishing is a separate, explicit action:
#   wsd publish --artifact $R/release --repo-id andrepaim/Ettin-150m-WSD-wiktionary --upload
```

A rerun of step 2 gives its own checkpoints: the round's first logged steps come out again to within bf16 noise (below),
but not bit for bit, so a full rerun's accuracies and its chosen checkpoint can differ slightly. To export and publish
the round's own model, give step 5 `--model` a download of `andrepaim/Ettin-150m-WSD-wiktionary` at revision `0edcbab0`
(its `model.safetensors` is the checkpoint, the same file as at `affc67b9`). The round measured outliers and the INT8 check on all 5,221 development
prompts, before the leak check dropped 20 development items; the cleaned files of the dataset hold 5,200, so their first
200 prompts differ from the round's by one and their first 1,000 by three. With them the same five layers are chosen
(ratios within 0.3 of the round's), and the graph is the same.

## Checked after the move

On 2026-11-27, in the environment of the round (the virtual environment of `wiktionary-round1-requirements.txt`, with
this package from the commit that added this page), from the round's kept files on the maintainer's machine:

- **Export.** `wsd export` as in step 5, from the kept checkpoint of labels and teacher, with the round's uncleaned
  development files: `onnx/model.onnx` and `onnx/model.int8.onnx` are byte for byte those of the round and of the Hugging
  Face repository (SHA-256 `bd31a4ff…4ab4` and `6bac2034…d6e4`, 203,516,610 bytes), and so are `model.safetensors`,
  `config.json`, the tokenizer files and the metadata of the export folder. The activation statistics equal the round's
  exactly; the layers are 4, 7, 9, 11 and 15; float parity 4.3 × 10⁻⁶; INT8 against float on 1,000 prompts: same top
  choice 96.6%, same top-three set 93.1%, centred RMSE median 0.113, as the round's check. The INT8 graph passes the
  fixture budget (agreement 100%, maximum logit error 0.30). 8 min 38 s.
- **Teacher cache.** Step 1 with a copy of the kept raw logits wrote `cache-stage2-T2.jsonl` byte for byte (185,889 rows,
  no prompt scored again, 54 seconds). Scoring the first 3,000 labelled training examples' prompts again in bf16 on the
  GPU agreed with the kept logits on 2,984 top choices (maximum logit difference 0.22: bf16 with other batch neighbours).
- **Evaluation.** `wsd rank` on the kept score files gives every value of the development table (and of the round's
  development summary file, by form and stratum) and every cell of the test table, intervals included. `wsd score` of the
  labels-and-teacher checkpoint on the development files in fp32 agreed with the kept scores on all 5,200 top choices
  (maximum logit difference 1 × 10⁻⁵) and gives 78.0% first again.
- **Training.** `wsd train --config configs/wiktionary_round1_labels_teacher.json --data-dir <the round's working
  folder>`, stopped after the step-60 log (5 minutes, about 2 of them on the GPU). The step-0 evaluations equal the kept
  run's (c1600: loss 1.1259613037109375, accuracy 0.695, top three 0.9595, none-fits 0.002; uncapped: 1.1312685012817383,
  0.69, 0.953, 0.003); the learning rate equals the kept run's at every logged step; loss and gradient norm:

  | Step | Loss, round | Loss, `wsd train` | Gradient norm, round | Gradient norm, `wsd train` |
  | ---: | ---: | ---: | ---: | ---: |
  | 10 | 0.862366 | 0.862603 | 5.1931 | 5.1981 |
  | 20 | 0.800255 | 0.800744 | 5.3407 | 5.4130 |
  | 30 | 0.624098 | 0.624637 | 3.9030 | 3.9068 |
  | 40 | 0.520393 | 0.519648 | 3.8754 | 3.8690 |
  | 50 | 0.460938 | 0.461437 | 4.8948 | 4.8994 |
  | 60 | 0.833973 | 0.833950 | 4.5384 | 4.5267 |

  The differences (under 0.1% of the loss) are of the size the round's own resume onto the same batches showed (0.8531
  against 0.8534), while a different batch order moved the loss by about 8% (0.772 against 0.834). The jump at step 60 is
  the move to another length-grouped mega-batch.
