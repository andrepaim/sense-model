---
license: apache-2.0
base_model: sign/Ettin-150m-WSD
base_model_relation: finetune
language: en
datasets:
- andrepaim/wiktionary-wsd
- wikimedia/wikipedia
tags:
- word-sense-disambiguation
- wiktionary
- modernbert
- ettin
- onnx
---

# Ettin-150m-WSD-wiktionary

A word-sense model that picks which dictionary meaning of a word fits a passage. It is
[`sign/Ettin-150m-WSD`](https://huggingface.co/sign/Ettin-150m-WSD) (revision `8751b577199d1bb95b74fa2457da7065d57100ae`),
fine-tuned on English Wiktionary's own usage examples and quotations, with soft labels from the larger
[`sign/Ettin-1B-WSD`](https://huggingface.co/sign/Ettin-1B-WSD) (revision `e6fc37a7e11fee91a600a0cd8e7423bcc639a1cc`).
The base model was trained on WordNet definitions; this one was tuned for Wiktionary's definitions.

It was made for the Chrome extension [Contextual Dictionary](https://github.com/andrepaim/contextual-dictionary), which shows
English Wiktionary meanings of a selected word with the meaning that fits the passage first. The extension runs the INT8
ONNX file of this repository in the browser (ONNX Runtime Web, WebAssembly); nothing the reader selects leaves the computer.
The extension downloads that file from its own GitHub release
[`assets-v2`](https://github.com/andrepaim/contextual-dictionary/releases/tag/assets-v2), not from this repository. This
repository holds the same graph together with the float weights and the float ONNX export, so they can be reused.

How it was made and measured at model level is written up in sense-model's
[`docs/wiktionary-round1.md`](https://github.com/andrepaim/sense-model/blob/main/docs/wiktionary-round1.md), with the `wsd` commands that rerun it, and the
training and evaluation data are the dataset [`andrepaim/wiktionary-wsd`](https://huggingface.co/datasets/andrepaim/wiktionary-wsd/tree/84b6444e83219aaa76c29a517a64a640c1df9a46) (revision
`84b6444e83219aaa76c29a517a64a640c1df9a46`). The measurements in the browser are in the extension's
[`docs/validation.md`](https://github.com/andrepaim/contextual-dictionary/blob/main/docs/validation.md), section "Training
round 1 (assets-v2)". The numbers on this card come from those two documents; the example's output, the file sizes and
digests, and the warm-up, weight decay and mixed precision (from the trainer's arguments) were checked for this card.

## What it does

Word-sense disambiguation as multiple choice, the same way as the base model: the prompt holds the passage with the target
word marked, one definition per answer letter, and a "none of the above" letter; the model reads the `[MASK]` at the end and
a 128-way answer decoder (in place of the usual vocabulary decoder) gives one logit per letter. The architecture is
ModernBERT (22 layers, hidden size 768, about 150M parameters), as in the base model. The weights here are float32.

`config.json` names the class `WSDModernBertForMaskedLM`, as the base model does; that class lives in the base model's code
(<https://github.com/sign/word-sense-disambiguation>, per the base model's card) and is not part of transformers. The
example below loads the model with plain transformers instead.

## Prompt format (`compact-v1`)

Taken from the extension's prompt builder (`packages/core/src/prompt.ts`) and `model_metadata.json`:

```
<passage with the target word between asterisks>
A. <first definition>
B. <second definition>
...
а. none of the above
[unused0] [MASK]
```

- Lines are joined with a single `\n`. Each choice is `<letter>. <definition>`.
- The letters are `model_metadata.json["letters"]`, in order: `A`–`Z`, `a`–`z`, `0`–`9`, punctuation, Greek, Cyrillic. Up to
  127 choices use letters 0–126.
- "None of the above" always uses letter 127 (`nota_index`), which is the Cyrillic `а` (U+0430), not the Latin `a`, and
  always comes last, however many choices there are.
- The last line is the separator `[unused0]`, a space and `[MASK]`. The tokenizer adds `[CLS]` and `[SEP]`.
- The target is marked by wrapping it in `*`. The extension refuses passages or definitions that contain `*`, `[MASK]`,
  `[unused…]` or other special tokens, or control characters.
- The extension offers meanings in dictionary order, at most 24 per prompt and at most 1,600 characters per prompt (the
  trained model saw both these prompts and prompts with every meaning; see "Training"). The model's limit is 2,048 tokens.
- Read the answer as a softmax over the logits of the offered letters plus letter 127 (`calibration.json` is the identity:
  temperature 1.0, "uncalibrated"). The extension keeps dictionary order when "none of the above" scores at least as high as
  every meaning; otherwise it sorts the meanings by probability.

## Use with transformers

Tested with transformers 5.23.0 and PyTorch 2.14.1 on the CPU. The sentence and definitions are made up for this card.

```python
import json

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, ModernBertForMaskedLM

path = snapshot_download("andrepaim/Ettin-150m-WSD-wiktionary", allow_patterns=["*.json", "model.safetensors"])
meta = json.load(open(f"{path}/model_metadata.json"))
letters, nota = meta["letters"], meta["nota_index"]

tokenizer = AutoTokenizer.from_pretrained(path)
# ModernBERT with a 128-way answer decoder in place of the vocabulary decoder: load the rest, then the decoder.
model = ModernBertForMaskedLM.from_pretrained(path, ignore_mismatched_sizes=True).eval()
model.decoder = torch.nn.Linear(model.config.hidden_size, 128)
with safe_open(f"{path}/model.safetensors", "pt") as f:
    model.decoder.load_state_dict({"weight": f.get_tensor("decoder.weight"), "bias": f.get_tensor("decoder.bias")})


def prompt_for(passage, start, end, definitions):
    marked = passage[:start] + "*" + passage[start:end] + "*" + passage[end:]
    choices = [f"{letters[i]}. {d}" for i, d in enumerate(definitions)]
    none = f"{letters[nota]}. {meta['none_text']}"
    return "\n".join([marked, *choices, none, f"{meta['separator']} {tokenizer.mask_token}"])


@torch.inference_mode()
def choose(passage, word, definitions):
    start = passage.index(word)
    enc = tokenizer(prompt_for(passage, start, start + len(word), definitions), return_tensors="pt")
    mask = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()
    logits = model(**enc).logits[0, mask]  # 128 answer logits
    scores = torch.cat([logits[: len(definitions)], logits[nota : nota + 1]])
    return scores.softmax(-1)  # the offered meanings, then "none of the above"


definitions = [
    "A small flying mammal that hunts at night and finds its way by echoes.",
    "A club of wood or metal used to strike the ball in games such as cricket and baseball.",
    "A turn at hitting the ball in such a game.",
    "To open and close one's eyes quickly; to blink.",
]
for passage in ["She swung the bat and hit the ball over the fence.",
                "A bat flew out of the cave just as the sun went down."]:
    probs = choose(passage, "bat", definitions)
    labels = [*letters[: len(definitions)], "none"]
    print(passage)
    print("  " + "  ".join(f"{l}={p:.3f}" for l, p in zip(labels, probs.tolist())))
```

Output (transformers prints a load report saying that `decoder.weight` and `decoder.bias` did not match the vocabulary
decoder's shape; the code then loads them into the 128-way decoder):

```
She swung the bat and hit the ball over the fence.
  A=0.002  B=0.986  C=0.007  D=0.002  none=0.003
A bat flew out of the cave just as the sun went down.
  A=0.986  B=0.003  C=0.002  D=0.003  none=0.007
```

On these two prompts the logits of this loader match the float ONNX export to within 3 × 10⁻⁶.

## ONNX files

Both graphs take `input_ids`, `attention_mask` and `prediction_positions` (int64; `prediction_positions` is the index of
`[MASK]` in each row) and return `logits` of shape `[batch, 128]`.

- `onnx/model.int8.onnx` (203,516,610 bytes, SHA-256 `6bac203419a0ad43b09b3806c22abb4870c48adf21657f3ef2a73a5d8369d6e4`):
  **the file the extension runs**, byte for byte the graph of the release `assets-v2` (published there gzip-compressed as
  `ettin-150m-wsd-int8.onnx.gz`). Against the float model it gives the same top choice on 96.6% of 1,000 development prompts.
- `onnx/model.onnx` (599,536,273 bytes): the float export, maximum logit error 4.3 × 10⁻⁶ against PyTorch.
- `onnx/recipe.json`: the quantization recipe, below.

### Quantization recipe

Dynamic quantization with ONNX Runtime's `quantize_dynamic`: INT8 weights per channel for the MatMul operations,
activations quantized at run time, except the feed-forward output MatMul (`mlp.Wo`) of layers 4, 7, 9, 11 and 15, which stays
in float. Those are the layers whose `mlp.Wo` input peaks above 15 times its 99.9th percentile, measured for this model on
200 development prompts: 42.9 (layer 4), 1,089 (7), 24.8 (9), 1,663 (11) and 740 (15); the next are 13.5 (12) and
13.1 (14). The token embedding is stored in float16 with a cast to float. The recipe is `wsd export --quantize
--per-channel --float16-embedding --outlier-examples …` in [sense-model](https://github.com/andrepaim/sense-model), which makes both graphs of this
repository again byte for byte from `model.safetensors` (see `docs/wiktionary-round1.md` there).

## Training

### Data

Labelled examples come from the [Kaikki.org](https://kaikki.org/) extract of English Wiktionary of 2026-10-30. Wiktionary's
editors attach every usage example and quotation to the meaning it illustrates; that attachment is the label. The prompts
were built by the extension's own code from the extension's own dictionary pack (one lower-case headword, at least two
meanings in one part of speech, at most 12 examples per entry), in two forms: the extension's defaults (at most 24 meanings,
1,600 characters) and every meaning.

| | Items | Headwords | 2 meanings | 3–4 | 5–8 | 9–16 | 17–32 | 33+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 125,031 | 34,993 | 48,394 | 41,108 | 25,404 | 9,846 | 279 | 0 |
| Development | 4,951 | 1,378 | 1,863 | 1,756 | 952 | 380 | 0 | 0 |

Training items: 69,789 nouns, 24,365 verbs, 25,837 adjectives, 3,462 adverbs, 1,578 others; 96,528 quotations and 28,503
usage examples. After the leak checks below, the training examples were 129,093 labelled and 1,658 teacher-only (prompts
where the 24-meaning cap left the labelled meaning out: these were used with the teacher's answers only, not as "none of the
above").

The unlabelled stream (teacher only) is one shard of English Wikipedia from
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, file `20231101.en/train-00001-of-00041.parquet`: a seeded sample of 60,000
prose paragraphs of 120–700 characters, one lower-case word per paragraph, at most 15 per lemma: 47,910 items, 55,138
examples.

**Held out.** No example of any headword of the project's English Wiktionary test set or hand-written test set was used,
in any part of speech: 2,197 headwords (3,754 entries skipped, and 2,274 more items dropped because their lookup returned an
entry of a held-out lemma). The other headwords were split into training and development by a hash of the headword (4% to
development). The Simple English Wiktionary test set is not held out by headword, but its definitions and examples were
never training data.

**Leak checks.** No training or development headword, and no lemma the unlabelled stream looked up, is held out; training
and development headwords are disjoint. The same quotation is often filed under several headwords, so the headword rule did
not catch everything: 46 training and 2 development passages were equal to a test passage after normalisation, and 557 more
shared a run of ten words with one. All 605 were dropped; the final files have no passage equal to, and no ten-word run
shared with, any of the three test sets.

### Recipe

The loss is the base model's: 0.3 × label cross-entropy (label smoothing 0.1) + 0.7 × T² × KL divergence to the teacher, at
T = 2. Rows without a label use the KL term only. Teacher soft labels from `sign/Ettin-1B-WSD`, computed in bf16 (99.4%
agreement with fp32 on the top choice).

- Learning rate 5e-5, cosine schedule, warm-up over the first tenth of the planned steps, weight decay 0.01, batch 64 (4 × 16
  accumulation), seed 42, bf16 mixed precision, at most 3 epochs.
- Candidates kept in the extension's (dictionary) order; batches grouped by length.
- Evaluation every 300 steps on 2,000 development prompts of each form; the best checkpoint by "labelled meaning first" at
  the extension's defaults; early stopping after 4 evaluations without improvement.
- This run (training round 1, "labels and teacher": 185,889 examples, labels where there are) stopped at step 4,200 of 8,715
  planned. **The weights here are the step-3,000 checkpoint (1.03 epochs).**
- One NVIDIA GeForce RTX 3060 (12 GB); Python 3.12, PyTorch 2.14.1, transformers 5.23.0. The run used a trainer script
  of the extension's repository over sense-model's dataset, loss and trainer class; its options are now part of
  [sense-model](https://github.com/andrepaim/sense-model)'s `wsd train`, and `configs/wiktionary_round1_labels_teacher.json` there reproduces the run's first
  logged steps (`training_config.json` here).

### Three runs, and why this one

| Run | Data | Stopped at | Chosen checkpoint |
| --- | --- | ---: | ---: |
| Labels only | 129,093 labelled | step 4,200 of 6,054 | 3,000 |
| Teacher only | the labelled prompts without their labels, the teacher-only ones and the unlabelled stream (185,889) | step 4,200 of 8,715 | 3,000 |
| **Labels and teacher (this model)** | the same 185,889, labels where there are | step 4,200 of 8,715 | 3,000 |

Development data favoured **labels only** (78.6% against 78.0%). This run was chosen instead, **after the test results below
were known**: it is better than labels only on the hand-written and Simple English Wiktionary sets and about one point worse
on the Wiktionary set. Because the choice looked at the test sets, the test numbers of this model are not an unbiased
estimate of its accuracy. The other two runs are not published.

## Results

"First" is the share of items whose labelled meaning comes first within its entry. The test sets are described in
`docs/validation.md`:

- **Wiktionary**: 2,300 English Wiktionary examples, stratified by number of meanings in the entry (448, 446, 447, 447, 449
  and 63 for 2, 3–4, 5–8, 9–16, 17–32 and 33+), headwords held out of training.
- **Hand-written**: 104 web-style sentences (news, documentation, forums, recipes), mostly common words with many meanings.
  **Written and labelled by a model, with no human review.**
- **Simple English**: 4,167 examples from Simple English Wiktionary, a separate wiki with its own definitions; mostly short
  entries.

### Development (float PyTorch)

4,931 development items, at the extension's defaults (and with every meaning offered):

| Model | First | Top 3 | First, every meaning |
| --- | ---: | ---: | ---: |
| Dictionary order | 38.1% | 84.3% | 38.1% |
| Base, `sign/Ettin-150m-WSD` | 72.1% | 96.6% | 72.8% |
| Teacher, `sign/Ettin-1B-WSD` | 79.3% | 97.3% | 80.1% |
| Labels only | 78.6% | 97.5% | 79.5% |
| Teacher only | 76.7% | 97.3% | 77.7% |
| **Labels and teacher (this model)** | 78.0% | 97.5% | 79.0% |

### Test (float PyTorch)

Labelled meaning first; in brackets the paired bootstrap 95% interval of the difference from the base model (2,000
resamples):

| Set, prompt form | Base | Labels only | Teacher only | **This model** |
| --- | ---: | ---: | ---: | ---: |
| Wiktionary, defaults | 59.0% | 66.1% (+5.5 to +8.6) | 62.7% (+2.4 to +5.1) | 64.2% (+3.8 to +6.7) |
| Wiktionary, every meaning | 66.0% | 75.1% (+7.3 to +10.9) | 71.3% (+3.9 to +6.8) | 73.3% (+5.8 to +8.9) |
| Hand-written, defaults | 65.4% | 69.2% (−1.9 to +9.6) | 67.3% (−2.9 to +6.7) | 70.2% (+0.0 to +9.6) |
| Hand-written, every meaning | 75.0% | 80.8% (−1.0 to +12.5) | 79.8% (−1.0 to +10.6) | 83.7% (+3.8 to +14.4) |
| Simple English, defaults | 86.6% | 87.0% (−0.4 to +1.3) | 87.6% (+0.3 to +1.8) | 88.5% (+1.1 to +2.7) |

### In the extension (browser, INT8)

`onnx/model.int8.onnx` in the built extension (ONNX Runtime Web, WebAssembly, 8 threads), on the prompts the extension
builds at its defaults, against the previously shipped model: `assets-v1`, an INT8 export of the base checkpoint itself.

| Set | First, `assets-v1` | First, this model | Paired 95% interval of the difference | Top 3, `assets-v1` / this model | "None of the above", `assets-v1` / this model |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wiktionary | 55.4% | **64.3%** | +7.3 to +10.6 | 77.9% / 81.2% | 7 / 22 of 2,287 |
| Hand-written | 64.4% | **69.2%** | +0.0 to +9.6 | 75.0% / 76.0% | 6 / 8 of 104 |
| Simple English | 84.6% | **88.0%** | +2.5 to +4.3 | 99.7% / 99.6% | 42 / 18 of 4,157 |

Wiktionary set by meanings in the entry, first: 85.0% (2), 78.9% (3–4), 73.2% (5–8), 53.7% (9–16), 37.0% (17–32), 22.2%
(33+), against 78.1%, 72.0%, 58.8%, 43.0%, 30.5% and 19.0% for `assets-v1`.

Item by item, this model does not only add correct answers; it also loses some that `assets-v1` had:

| Set | Both right | `assets-v1` right, this model wrong | `assets-v1` wrong, this model right | Both wrong | `assets-v1`'s correct items lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wiktionary | 1,175 | 100 | 305 | 720 | 100 of 1,275 (7.8%) |
| Hand-written | 66 | 1 | 6 | 31 | 1 of 67 (1.5%) |
| Simple English | 3,399 | 126 | 266 | 376 | 126 of 3,525 (3.6%) |

**Speed and memory in the extension** (12th Gen Intel Core i9-12900F, Chromium headless, Linux under WSL2):

- 8 threads, 600 prompts each run twice: p50 421 ms, p99 633 ms, maximum 731 ms per prompt (the base checkpoint exported
  with the same recipe: 412 / 640 / 846 ms). The extension waits at most 750 ms for an answer.
- Fewer threads, 512 prompts, against `assets-v1`: within 750 ms 100% / 100% on 8 threads; on 4 threads 100% in all three
  runs for `assets-v1` and 94%, 100%, 100% for this model; on 2 threads 72% / 71%. These runs used the machine's fast cores,
  so a small machine is slower for both.
- Memory of the extension's process with the model loaded: 1,183–1,188 MB resident (1,007–1,009 MB proportional), against
  1,294–1,302 MB (1,118–1,126 MB) for `assets-v1`.
- Download in the extension: 188.8 MB, against 147.0 MB for `assets-v1`.

## Limitations

- **This run was chosen after the test results were known.** Development favoured another run, so the test numbers above
  are not an unbiased estimate of this model's accuracy.
- **The gains are largest on Wiktionary-style text** (literary and quoted examples like the training data) and smaller
  elsewhere: on Simple English Wiktionary +3.4 points first in the extension against `assets-v1` (+1.9 in float against
  the base), and on the hand-written set intervals that start at zero (+4.8 points in the extension, +4.8 in float).
- **The hand-written set was written and labelled by a model, with no human review**, and is small (104 items; its
  intervals span about ±10 points).
- **Entries with many meanings were barely trained**: the held-out headwords take nearly all long entries, so training has
  279 items in entries with 17–32 meanings and none with more, and development has no entry with more than 16 meanings.
  What it does on long entries comes from the teacher and from short entries. In the extension the labelled meaning is
  first in 37.0% of Wiktionary items with 17–32 meanings and 22.2% with 33 or more.
- **It answers "none of the above" more often than the base on the Wiktionary set** (22 against 7 of 2,287 prompts; on the
  Simple English set it does so less often, 18 against 42). In the extension such an answer leaves dictionary order.
- **It loses some answers the base got right**: 7.8% of the base's correct Wiktionary items, 3.6% of Simple English and 1 of
  67 hand-written, while gaining more. Among about twelve demonstration lookups the base got right, two became wrong.
- **Development passages overlap training passages.** Training and development headwords are disjoint, but the same
  quotation is often filed under several headwords: sense-model's split check finds 438 of the 4,931 development items
  with a passage equal to (90 findings) or nearly the same as (668) that of a training item, under another headword. The
  development numbers may be somewhat optimistic for that reason; the test sets were checked against training passages
  when the data was built (no equal passage and no shared run of ten words).
- All measurements were taken on one machine.
- The unlabelled stream is one Wikipedia shard of 41, and it avoids the held-out headwords, which are many of the
  commonest words.
- Not measured on WordNet or SemEval benchmarks, where the base model was evaluated; it may be worse there than the base.
- English only. The labels are editors' attachments of examples to meanings, not adjudicated annotations, and an example
  can fit more than one meaning.

## Files

| File | What it is |
| --- | --- |
| `model.safetensors` | Float32 weights of the step-3,000 checkpoint. SHA-256 `49bc2634bcf696171f367b59cedb961fcd71bb7d825ad6680ef5ed1ed8d25be3`, which is the `model_revision` in `model_metadata.json` and the revision the extension checks. |
| `config.json` | ModernBERT configuration, with `answer_vocab_size` 128. |
| `tokenizer.json`, `tokenizer_config.json` | The base model's tokenizer as sense-model exports it, byte for byte the files of the release `assets-v2`: no truncation setting (the copy in `sign/Ettin-150m-WSD` truncates at 2,048 tokens; sense-model refuses longer prompts instead) and no padding setting (the checkpoint's copy, published here until 2026-11-27, pads a batch to its longest prompt). The token ids of a prompt are the same with any of the three. |
| `model_metadata.json` | The answer letters, `nota_index` 127, template `compact-v1`, separator, maximum length 2,048 tokens, the revision, and the provenance of the training and the export; the file published in `assets-v2`. |
| `calibration.json` | Identity calibration (temperature 1.0), as published in `assets-v2`. |
| `onnx/` | The ONNX exports and the quantization recipe (above). |
| `LICENSE` | The Apache License 2.0. |
| `NOTICE` | The model notice published with `assets-v2`: the attributions, the training data and their licences, what was changed, and the Apache-2.0 and MIT licence texts. It names the release's file names; the extra files here (the float weights, `config.json`, the float ONNX graph, the recipe and the records below) are offered on the same terms. |
| `training_config.json` | The `wsd train` configuration of this run (sense-model's `configs/wiktionary_round1_labels_teacher.json`). |
| `data_manifest.json` | The files of the dataset that run reads, with their SHA-256 and counts, and the outcome of sense-model's split check. |
| `evaluation.json` | The development and float test tables above, as `wsd rank` reports them from the score files of the round. |
| `checksums.json` | The SHA-256 of every other file, written by `wsd publish`. |

## Licence and attribution

These weights and files are offered under the **Apache License 2.0**, the licence of the checkpoint they were fine-tuned
from. See `LICENSE` and `NOTICE`.

- Fine-tuned from [`sign/Ettin-150m-WSD`](https://huggingface.co/sign/Ettin-150m-WSD/tree/8751b577199d1bb95b74fa2457da7065d57100ae)
  by sign (Nagish Inc.), Apache-2.0, itself fine-tuned from
  [`jhu-clsp/ettin-encoder-150m`](https://huggingface.co/jhu-clsp/ettin-encoder-150m) by the Ettin authors (MIT).
- Teacher: [`sign/Ettin-1B-WSD`](https://huggingface.co/sign/Ettin-1B-WSD/tree/e6fc37a7e11fee91a600a0cd8e7423bcc639a1cc)
  by sign (Nagish Inc.), Apache-2.0. Only its answers were used; none of its weights are here.
- Training data (no training text is in this repository):
  - English Wiktionary usage examples and quotations, by the Wiktionary contributors, through the Kaikki.org extract
    (made with wiktextract) of 2026-10-30. Wiktionary's text is available under
    [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and the GFDL.
  - English Wikipedia paragraphs, by the Wikipedia contributors, from
    [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (dump of 2023-11-01, revision
    `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, file `20231101.en/train-00001-of-00041.parquet`), used without labels.
    That dataset states the licences [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and GFDL.

This card states the licences of the sources and the licence under which these weights are offered; it does not say
whether the terms of the training data's licences extend to the weights.

## More

- The extension: <https://github.com/andrepaim/contextual-dictionary>
- Full measurements, methods and caveats: [`docs/validation.md`](https://github.com/andrepaim/contextual-dictionary/blob/main/docs/validation.md), "Training round 1 (assets-v2)"
- Model-level method, numbers and the rerun with `wsd` commands: [`docs/wiktionary-round1.md`](https://github.com/andrepaim/sense-model/blob/main/docs/wiktionary-round1.md) in sense-model
- Training and evaluation data: [`andrepaim/wiktionary-wsd`](https://huggingface.co/datasets/andrepaim/wiktionary-wsd/tree/84b6444e83219aaa76c29a517a64a640c1df9a46)
