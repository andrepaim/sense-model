"""Cache teacher distributions by example and candidate ID for different-tokenizer students.

The teacher scores each distinct prompt once (wsd.scoring: length-sorted batches under an optional token budget,
float32 or bfloat16). With --scores the raw logits are kept in that file, which is appended to and resumable, and
prompts already in it are not scored again; the cache is then made from it: for every example,
softmax(logits / T) over its candidates and none-fits, keyed by candidate ID, with the example's fingerprint.
"""

import argparse
import json
import math
from pathlib import Path

from wsd.data import read_jsonl
from wsd.scoring import add_scoring_arguments, load_scorer, prompt_sha256, read_scores, score_examples
from wsd.train import example_fingerprint


def cache_rows(examples, scores, tokenizer, metadata, temperature, teacher_model, teacher_revision):
    """Teacher cache rows of the examples whose prompt was scored; returns (rows, skipped example IDs)."""
    rows, skipped = [], []
    for ex in examples:
        score = scores.get(prompt_sha256(ex, tokenizer, metadata))
        logits = score.get("logits") if score and score.get("type") == "scored" else None
        if logits is None or len(logits) != len(ex.candidates) + 1:
            skipped.append(ex.id)
            continue
        z = [value / temperature for value in logits]
        top = max(z)
        weights = [math.exp(value - top) for value in z]
        total = sum(weights)
        probs = [weight / total for weight in weights]
        rows.append(
            {
                "id": ex.id,
                "example_sha256": example_fingerprint(ex),
                "probabilities": {c.id: p for c, p in zip(ex.candidates, probs[:-1], strict=True)},
                "none_probability": probs[-1],
                "temperature": temperature,
                "teacher_model": teacher_model,
                "teacher_revision": teacher_revision,
            }
        )
    return rows, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_scoring_arguments(parser)
    parser.set_defaults(batch_size=8)
    parser.add_argument("--output", required=True, help="Teacher cache (JSONL) that wsd train --teacher-scores reads")
    parser.add_argument("--scores", help="Raw teacher logits by prompt (JSONL): reused, extended and resumable")
    parser.add_argument("--temperature", type=float, default=2.0)
    a = parser.parse_args(argv)
    if a.temperature <= 0 or a.batch_size < 1 or (a.max_tokens is not None and a.max_tokens < 1):
        parser.error("Temperature, batch size and max tokens must be positive")
    predictor = load_scorer(
        a.model, revision=a.revision, local_files_only=a.local_files_only, device=a.device, dtype=a.dtype
    )
    examples, seen = [], set()
    for path in a.input:
        for example in read_jsonl(path):
            if example.id in seen:
                raise ValueError(f"Duplicate example ID across inputs: {example.id}")
            seen.add(example.id)
            examples.append(example)
    new = score_examples(predictor, examples, a.scores, max_tokens=a.max_tokens, max_batch=a.batch_size, progress=True)
    scores = read_scores(a.scores) if a.scores else {row["hash"]: row for row in new}
    revision = predictor.metadata.get("model_revision", a.revision)
    rows, skipped = cache_rows(
        examples, scores, predictor.tokenizer, predictor.metadata, a.temperature, a.model, revision
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.output).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"rows": len(rows), "skipped": len(skipped), "scored": sum(r["type"] == "scored" for r in new)}
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
