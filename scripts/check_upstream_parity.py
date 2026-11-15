#!/usr/bin/env python3
"""Compare the extracted English model to original code in an isolated subprocess."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

UPSTREAM = r"""
import json,sys,torch
from pathlib import Path
from wsd.masked_language_model import load_model, unmask_token_batch
from wsd.prompt import Definition,create_multiple_choice_prompt
from wsd.word_sense_disambiguation import _result_from_probs

torch.set_num_threads(int(sys.argv[3]))
examples=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
components=load_model(sys.argv[2])
prompts=[]
definitions=[]
for e in examples:
    t=e['target']
    marked=e['text'][:t['start']]+'*'+e['text'][t['start']:t['end']]+'*'+e['text'][t['end']:]
    defs=[Definition(c['id'],c['definition']) for c in e['candidates']]
    definitions.append(defs)
    prompts.append(create_multiple_choice_prompt(components.tokenizer.mask_token,marked,defs,components.tokenizer))
answers=unmask_token_batch(prompts)
rows=[]
for e,p,defs,a in zip(examples,prompts,definitions,answers):
    probs=[a.probabilities[i] for i in range(len(defs))]+[a.probabilities[127]]
    total=sum(probs)
    probs=[x/total for x in probs]
    result=_result_from_probs(a.probabilities,defs)
    rows.append({'example_id':e['id'],'prompt':p,'probabilities':probs,'selected_id':result.synset_id or None})
print(json.dumps(rows,ensure_ascii=False))
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--model", default="sign/Ettin-150m-WSD")
    p.add_argument("--revision")
    p.add_argument("--data", type=Path, default=Path("data/fixtures/smoke.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=1e-5)
    args = p.parse_args(argv)
    from tempfile import TemporaryDirectory

    import torch
    from huggingface_hub import snapshot_download

    from wsd.data import read_jsonl
    from wsd.predict import Predictor
    from wsd.tokenizer import format_prompt

    torch.set_num_threads(args.threads)
    examples = [e for e in read_jsonl(args.data) if e.language.startswith("en") and e.candidates]
    if not examples:
        p.error("no English candidate examples found")
    path = Path(args.model)
    if not path.is_dir():
        path = Path(snapshot_download(args.model, revision=args.revision, local_files_only=True))
    start = time.perf_counter()
    model = Predictor.from_pretrained(path, local_files_only=True)
    predictions = model.predict(examples)
    # Force original checkpoint identity through environment; no WordNet requests are made.
    env = {
        **os.environ,
        "WSD_MODEL": str(path.resolve()),
        "WORDNET_URL": "http://unused.invalid",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(args.upstream.resolve()),
    }
    with TemporaryDirectory(prefix="wsd-parity-") as temporary:
        fixture = Path(temporary) / "examples.json"
        fixture.write_text(json.dumps([e.to_dict() for e in examples]), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", UPSTREAM, str(fixture), str(path.resolve()), str(args.threads)],
            cwd=args.upstream,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise RuntimeError("Upstream subprocess failed: " + proc.stderr)
        original = json.loads(proc.stdout)
    rows = []
    for ex, new, old in zip(examples, predictions, original, strict=True):
        prompt = format_prompt(ex, model.tokenizer, model.metadata)
        probs = [new["probabilities"][c.id] for c in ex.candidates] + [new["none_probability"]]
        delta = max(abs(a - b) for a, b in zip(probs, old["probabilities"], strict=True))
        rows.append(
            {
                "example_id": ex.id,
                "prompt_equal": prompt == old["prompt"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "selected_id_equal": new["selected_id"] == old["selected_id"],
                "max_probability_delta": delta,
            }
        )
    passed = all(
        r["prompt_equal"] and r["selected_id_equal"] and r["max_probability_delta"] <= args.tolerance for r in rows
    )
    report = {
        "passed": passed,
        "model": args.model,
        "checkpoint": str(path),
        "resolved_revision": model.metadata.get("model_revision"),
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip(),
        "examples": rows,
        "count": len(rows),
        "tolerance": args.tolerance,
        "threads": args.threads,
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "Extraction equivalence on illustrative English fixtures; not a quality benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
