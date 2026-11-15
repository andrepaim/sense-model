"""Explicit synthetic-data generation through a user-selected Ollama endpoint.

Generated rows need human review; they are never automatically mixed into gold.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from wsd.data import file_sha256, read_lexicon
from wsd.schema import Example, Target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexicon", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    p.add_argument("--language", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if a.limit < 1:
        p.error("--limit must be positive")
    if Path(a.output).exists():
        raise ValueError("Choose a new output file; generated records are never overwritten")
    inventory = read_lexicon(a.lexicon)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    accepted, rejected = 0, []
    with Path(a.output).open("w", encoding="utf-8") as stream:
        for i, ((lemma, pos), candidates) in enumerate(inventory.items()):
            if i >= a.limit:
                break
            if not candidates:
                continue
            gold = candidates[i % len(candidates)]
            prompt = (
                f"Write one natural sentence in {a.language} using {lemma!r} with this meaning: {gold.definition}. "
                "Return JSON with text (the sentence), target_text (the exact word or phrase in it), and start "
                "(Unicode character offset of that occurrence). No other fields or explanation."
            )
            request = urllib.request.Request(
                a.endpoint,
                data=json.dumps(
                    {
                        "model": a.model,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {"seed": a.seed + i},
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=a.timeout) as response:
                generated = json.loads(json.loads(response.read())["response"])
            try:
                text, target, start = generated["text"], generated["target_text"], generated["start"]
                eid = "synthetic:" + hashlib.sha256((a.model + str(i) + text).encode()).hexdigest()[:20]
                provenance = {
                    "source": "ollama-synthetic",
                    "model": a.model,
                    "annotation": "synthetic",
                    "review_status": "unreviewed",
                    "seed": a.seed + i,
                    "document_id": eid,
                    "lexicon_sha256": file_sha256(a.lexicon),
                    "prompt": prompt,
                }
                ex = Example(
                    eid,
                    text,
                    Target(start, start + len(target), target),
                    candidates,
                    language=a.language,
                    definition_language=a.language,
                    gold_ids=[gold.id],
                    provenance=provenance,
                )
                ex.validate()
                row = {
                    "id": eid,
                    "text": text,
                    "target": ex.to_dict()["target"],
                    "lemma": lemma,
                    "pos": pos,
                    "gold_ids": [gold.id],
                    "provenance": provenance,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append({"entry": lemma, "error": str(exc)})
    report = {
        "accepted_unreviewed": accepted,
        "rejected": rejected,
        "model": a.model,
        "language": a.language,
        "lexicon_sha256": file_sha256(a.lexicon),
    }
    Path(a.output + ".manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
