"""Score examples in batches: the logits of the offered candidates and none-fits, one row per distinct prompt.

Rows are keyed by the SHA-256 of the prompt string that format_prompt builds, so an application that builds the
same compact-v1 prompts can join them with its own records. Prompts are sorted by length and batched under a token
budget; a batch that runs out of GPU memory is split in halves. The output file is appended to and can be resumed:
prompts already in it are skipped. Row fields: hash, type ("scored" or "rejected"), code (TOO_LONG for prompts over
the model's max_length), logits (offered candidates in order, then none-fits), ms (per prompt), tokens, offered,
label, and the scorer's identity: model (as given), revision (its metadata's model_revision), dtype and max_length.
Resuming into a file scored with another identity is refused; rows without one (files written before it was
recorded) are accepted only when --untagged-scores names the identity they were scored with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from wsd.tokenizer import format_prompt


def prompt_sha256(example, tokenizer, metadata):
    return hashlib.sha256(format_prompt(example, tokenizer, metadata).encode("utf-8")).hexdigest()


def read_scores(path):
    """Score rows by prompt hash; a partial last line (a killed run) is ignored."""
    rows = {}
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "hash" in row:
            rows[row["hash"]] = row
    return rows


IDENTITY = ("model", "revision", "dtype", "max_length")


def scorer_identity(predictor, model=None):
    dtype = str(next(predictor.model.parameters()).dtype).removeprefix("torch.")
    return {
        "model": None if model is None else str(model),
        "revision": predictor.metadata.get("model_revision"),
        "dtype": dtype,
        "max_length": predictor.metadata["max_length"],
    }


def untagged_identity(values):
    """The identity named by --untagged-scores MODEL REVISION DTYPE MAX_LENGTH."""
    if values is None:
        return None
    model, revision, dtype, max_length = values
    return {"model": model, "revision": revision, "dtype": dtype, "max_length": int(max_length)}


def check_score_file(path, identity, untagged=None):
    """Refuse a file that is not a score file, or whose rows were scored with another identity."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError:
            if number == len(lines):  # a cut last line, dropped on resume
                continue
            raise ValueError(f"{path}:{number} is not a score file row") from None
        if not isinstance(row, dict) or "hash" not in row or "type" not in row:
            raise ValueError(f"{path}:{number}: not a score file (rows need hash and type)")
        recorded = {key: row[key] for key in IDENTITY if key in row}
        if not recorded:
            if untagged is None:
                raise ValueError(
                    f"{path} has rows with no recorded model; name the model they were scored with "
                    "(--untagged-scores MODEL REVISION DTYPE MAX_LENGTH) or use another file"
                )
            recorded = untagged
        if recorded != identity:
            raise ValueError(f"{path} was scored by another model: {recorded}, not {identity}")


def _logits(model, batch, mask_id, pad_id, device):
    width = max(len(ids) for ids in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    positions = torch.zeros(len(batch), dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention[i, : len(ids)] = 1
        positions[i] = ids.index(mask_id)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
            prediction_positions=positions.to(device),
        )
    return output.logits.float().cpu()


def _safe_logits(model, batch, mask_id, pad_id, device):
    try:
        return _logits(model, batch, mask_id, pad_id, device)
    except (torch.OutOfMemoryError, getattr(torch, "AcceleratorError", torch.OutOfMemoryError)) as error:
        if "out of memory" not in str(error) or len(batch) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = len(batch) // 2
        return torch.cat(
            [
                _safe_logits(model, batch[:half], mask_id, pad_id, device),
                _safe_logits(model, batch[half:], mask_id, pad_id, device),
            ]
        )


def score_examples(
    predictor,
    examples,
    output=None,
    *,
    max_tokens=None,
    max_batch=64,
    label=None,
    progress=False,
    identity=None,
    untagged=None,
):
    """Score the distinct prompts of ``examples`` not yet in ``output`` (appended to when given); returns the new rows.

    A batch grows, in length order, while it has at most ``max_batch`` prompts and, when ``max_tokens`` is set, its
    padded size (prompts times the longest prompt's tokens) stays within ``max_tokens``. ``identity`` (default:
    scorer_identity(predictor)) is recorded in every row and checked against an existing ``output``."""
    tokenizer, metadata = predictor.tokenizer, predictor.metadata
    identity = identity or scorer_identity(predictor)
    prompts = {}
    for example in examples:
        if not example.candidates:
            raise ValueError(f"Scoring requires candidate definitions: {example.id}")
        prompt = format_prompt(example, tokenizer, metadata)
        prompts.setdefault(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), (prompt, len(example.candidates)))
    done = set()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            check_score_file(output, identity, untagged)
            data = output.read_bytes()
            if data and not data.endswith(b"\n"):
                output.write_bytes(data[: data.rfind(b"\n") + 1])
            done = set(read_scores(output))
    todo = [h for h in prompts if h not in done]
    encoded = {h: tokenizer(prompts[h][0])["input_ids"] for h in todo}
    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
    nota, limit = metadata.get("nota_index", 127), metadata["max_length"]

    def problem(h):
        ids = encoded[h]
        return "TOO_LONG" if len(ids) > limit else "MASK" if ids.count(mask_id) != 1 else None

    # Every prompt is checked before batching; the rejected ones get a row each and are not scored.
    rejected = [h for h in todo if problem(h)]
    order = sorted((h for h in todo if not problem(h)), key=lambda h: len(encoded[h]))
    rows, started = [], time.time()
    stream = output.open("a", encoding="utf-8") if output is not None else None

    def emit(new):
        for row in new:
            row.update(identity)
            if stream is not None:
                stream.write(json.dumps(row) + "\n")
        rows.extend(new)

    try:
        emit(
            [
                dict(hash=h, type="rejected", code=problem(h), logits=None, ms=0, tokens=len(encoded[h]))
                | dict(offered=prompts[h][1], label=label)
                for h in rejected
            ]
        )
        i = 0
        while i < len(order):
            j = i
            while (
                j < len(order)
                and j - i < max_batch
                and (max_tokens is None or (j - i + 1) * len(encoded[order[j]]) <= max_tokens)
            ):
                j += 1
            j = max(j, i + 1)
            batch = order[i:j]
            start = time.perf_counter()
            logits = _safe_logits(predictor.model, [encoded[h] for h in batch], mask_id, pad_id, predictor.device)
            ms = round((time.perf_counter() - start) * 1000 / len(batch), 2)
            new = []
            for h, row in zip(batch, logits.tolist(), strict=True):
                offered = prompts[h][1]
                new.append(
                    dict(hash=h, type="scored", code=None, logits=row[:offered] + [row[nota]], ms=ms)
                    | dict(tokens=len(encoded[h]), offered=offered, label=label)
                )
            i = j
            emit(new)
            if progress and len(rows) % 2000 < len(new):
                elapsed = time.time() - started
                print(f"{len(rows)}/{len(todo)} in {elapsed / 60:.1f} min", file=sys.stderr, flush=True)
    finally:
        if stream is not None:
            stream.close()
    return rows


def load_scorer(model, *, revision=None, local_files_only=False, device="cpu", dtype=None):
    """The model to score with, cast to ``dtype`` when given (else in the checkpoint's stored dtype)."""
    from wsd.predict import Predictor

    predictor = Predictor.from_pretrained(model, revision=revision, local_files_only=local_files_only, device=device)
    if dtype is not None:
        predictor.model.to({"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype])
    return predictor


def same_path(a, b):
    return Path(a).resolve() == Path(b).resolve()


def add_scoring_arguments(parser, dtype="float32"):
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default=dtype, help=f"Default: {dtype or 'the stored dtype'}"
    )
    parser.add_argument("--input", nargs="+", required=True, help="Example JSONL files")
    parser.add_argument("--max-tokens", type=int, help="Token budget of a padded batch")
    parser.add_argument("--batch-size", type=int, default=64, help="Most prompts in a batch")
    parser.add_argument(
        "--untagged-scores",
        nargs=4,
        metavar=("MODEL", "REVISION", "DTYPE", "MAX_LENGTH"),
        help="The identity of score rows that record none (files written before it was recorded)",
    )


def main(argv=None):
    from wsd.data import read_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    add_scoring_arguments(parser)
    parser.add_argument("--output", required=True, help="Score file (JSONL), appended to and resumable")
    parser.add_argument("--label", help="Recorded in each row")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("batch-size and max-tokens must be positive")
    if any(same_path(path, args.output) for path in args.input):
        parser.error("--output must not be one of the --input files")
    predictor = load_scorer(
        args.model, revision=args.revision, local_files_only=args.local_files_only, device=args.device, dtype=args.dtype
    )
    examples = [example for path in args.input for example in read_jsonl(path)]
    rows = score_examples(
        predictor,
        examples,
        args.output,
        max_tokens=args.max_tokens,
        max_batch=args.batch_size,
        label=args.label,
        progress=True,
        identity=scorer_identity(predictor, args.model),
        untagged=untagged_identity(args.untagged_scores),
    )
    report = {
        "scored": sum(r["type"] == "scored" for r in rows),
        "rejected": sum(r["type"] == "rejected" for r in rows),
    }
    report["total"] = len(read_scores(args.output))
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    main()
