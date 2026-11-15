"""Cache teacher distributions by example and candidate ID for different-tokenizer students."""

import argparse
import json
from pathlib import Path

import torch

from wsd.data import read_jsonl
from wsd.predict import Predictor
from wsd.train import example_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    a = parser.parse_args(argv)
    if a.temperature <= 0 or a.batch_size < 1:
        parser.error("Temperature and batch size must be positive")
    predictor = Predictor.from_pretrained(
        a.model, revision=a.revision, local_files_only=a.local_files_only, device=a.device
    )
    examples = read_jsonl(a.input)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.output).open("w", encoding="utf-8") as stream:
        for start in range(0, len(examples), a.batch_size):
            batch = examples[start : start + a.batch_size]
            if any(not ex.candidates for ex in batch):
                raise ValueError("Teacher scoring requires candidate definitions")
            logits = predictor.logits(batch).detach().float().cpu()
            for ex, scores in zip(batch, logits, strict=True):
                indices = [*range(len(ex.candidates)), predictor.metadata.get("nota_index", 127)]
                probs = torch.softmax(scores[indices] / a.temperature, dim=-1).tolist()
                row = {
                    "id": ex.id,
                    "example_sha256": example_fingerprint(ex),
                    "probabilities": {c.id: prob for c, prob in zip(ex.candidates, probs[:-1], strict=True)},
                    "none_probability": probs[-1],
                    "temperature": a.temperature,
                    "teacher_model": a.model,
                    "teacher_revision": predictor.metadata.get("model_revision", a.revision),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return a.output


if __name__ == "__main__":
    main()
