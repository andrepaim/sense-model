# sense-model

Train, evaluate, and publish compact models that choose a word's meaning in context.
Supply a sentence, an exact target span, and dictionary definitions; receive scores,
a selected candidate, or abstention. English and Portuguese share one Python API
and prepared-data format. Dictionary lookup belongs to the application.

## Install

Python 3.11+ and uv:

```sh
uv sync --extra train --extra export --extra dev
uv run wsd --help
uv run pytest
```

The lock defaults to CPU PyTorch. For GPU training use a separate environment with
a CUDA PyTorch wheel appropriate for the machine (`uv pip install` with the
PyTorch CUDA index), then install this package with training extras. Never install
this project from the sibling upstream directory: both packages use `wsd`.

## Predict

```sh
uv run wsd predict --model sign/Ettin-150m-WSD \
  --input data/fixtures/smoke.jsonl --output runs/predictions.jsonl
```

The first explicit model load downloads weights. Use `--local-files-only` to
require cached files. No dictionary server, spaCy, entity linker, or account is
needed once model files are cached. Literal `*` markers and mask tokens are
rejected by the current compact prompt format; oversized prompts are refused.
The checkpoint limit is 127 candidates. Single-candidate inputs still permit
rejection. Empty candidate lists return `no_candidates` without model execution.

```python
from wsd import Predictor, Example, Target, Candidate

text = 'She sat on the bank.'
start = text.index('bank')
example = Example(
    id='example', text=text, target=Target(start, start + 4, 'bank'),
    candidates=[Candidate('river', 'sloping land beside a body of water'),
                Candidate('finance', 'a financial institution')],
)
model = Predictor.from_pretrained('sign/Ettin-150m-WSD')
print(model.predict([example]))
```

Target offsets count Unicode code points and end exclusively. Node consumers must
convert UTF-16 selection offsets; the Node example provides that conversion.
Candidate IDs are opaque, unique strings. They never have to be WordNet IDs.
`none_fits` is the model's explicit rejection; `uncertain` is threshold-based
abstention. Uncalibrated scores are not confidence guarantees.

## Lifecycle

```sh
uv run wsd prepare --help
uv run wsd train --help
uv run wsd teacher --help
uv run wsd evaluate --help
uv run wsd export --help
uv run wsd publish --help
```

See [data preparation](docs/data.md), [training](docs/training.md),
[evaluation](docs/evaluation.md), [release/export](docs/releasing.md), and
[implementation status](docs/status.md) and [Portuguese training smoke receipt](docs/portuguese-smoke.md). Configuration examples live in `configs`.
Small authored smoke examples are tests of software mechanics, not a benchmark.

## Portuguese

The starting candidate is `jhu-clsp/mmBERT-small`, a multilingual ModernBERT.
Initialize/train its compact answer decoder with Portuguese contexts and
Portuguese definitions. The same inference code accepts an artifact produced by
training. Until representative training, calibration, and evaluation are done,
this project does not claim a production-ready Portuguese checkpoint. A generic
multilingual base is not a trained WSD model.

The package supports separate English/Portuguese artifacts and independent
teacher/student tokenizers. A bilingual checkpoint is an optional experiment,
not an assumption. Source language and definition language are recorded separately.

## Origins and attribution

Model, decoder-pruning and answer-letter code is adapted from
[sign/word-sense-disambiguation](https://github.com/sign/word-sense-disambiguation),
using local fork commit `d1f7e107f2c072e876f031fdd5935bf496a0ef1b`.
The sibling fork is retained as the reference implementation. This project
extracts candidate selection and provides portable datasets, training,
evaluation, ONNX/Node deployment, and explicit Hub release tooling.

Original code copyright (c) 2026 Nagish Inc., MIT. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). Model weights and datasets retain their own terms. The upstream
English WSD checkpoint declares Apache-2.0; mmBERT-small declares MIT. Every
release must document exact base/data revisions and applicable notices.
