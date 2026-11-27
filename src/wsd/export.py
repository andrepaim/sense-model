"""Export compact answer logits and verify the deployment graph against PyTorch."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, prediction_positions):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, prediction_positions=prediction_positions
        ).logits


def export_model(
    predictor,
    examples,
    output: Path,
    *,
    quantize=False,
    atol=1e-4,
    rtol=1e-3,
    max_accuracy_loss=0.01,
    per_channel=False,
    exclude_nodes=(),
    float16_embedding=False,
):
    """Export and check every fixture singly and together (variable batch/sequence sizes).

    Quantized parity reports probability drift and decision agreement. A quantized
    artifact is rejected if its labelled accuracy drops by more than the budget.
    Fixtures are a mechanical check, not a substitute for a full release evaluation.
    ``per_channel`` quantizes MatMul weights per output channel, ``exclude_nodes`` names MatMul nodes that stay in
    float, and ``float16_embedding`` stores the token embedding in float16 with a Cast back to float.
    """
    import onnxruntime as ort

    from wsd.tokenizer import encode_examples

    if not examples or any(not e.candidates for e in examples):
        raise ValueError("Export fixtures must contain at least one candidate each")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    predictor.model.to("cpu").float().eval()
    predictor.save_pretrained(output)
    graph = output / "onnx" / "model.onnx"
    graph.parent.mkdir()
    wrapper = ExportModel(predictor.model).eval()
    batch = encode_examples(examples, predictor.tokenizer, predictor.metadata)
    inputs = ("input_ids", "attention_mask", "prediction_positions")
    args = tuple(batch[k].cpu() for k in inputs)
    try:
        torch.onnx.export(
            wrapper,
            args,
            str(graph),
            input_names=list(inputs),
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "prediction_positions": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
        checks = []
        for group in [examples, *[[e] for e in examples]]:
            encoded = encode_examples(group, predictor.tokenizer, predictor.metadata)
            feed = {k: encoded[k].cpu().numpy() for k in inputs}
            with torch.inference_mode():
                expected = wrapper(*(encoded[k].cpu() for k in inputs)).numpy()
            actual = session.run(["logits"], feed)[0]
            if not np.allclose(expected, actual, atol=atol, rtol=rtol):
                raise ValueError(f"ONNX parity failed: max logit error={np.max(np.abs(expected - actual))}")
            checks.append(
                {
                    "batch": len(group),
                    "sequence": feed["input_ids"].shape[1],
                    "max_logit_error": float(np.max(np.abs(expected - actual))),
                }
            )
        with torch.inference_mode():
            reference_logits = wrapper(*args).numpy()
        report = {
            "format": 1,
            "float_parity": checks,
            "atol": atol,
            "rtol": rtol,
            "examples": [e.to_dict() for e in examples],
            "inputs": {k: batch[k].tolist() for k in inputs},
            "expected_logits": reference_logits.tolist(),
            "expected_predictions": predictor.predict(examples),
        }
        if quantize:
            import onnx
            from onnxruntime.quantization import QuantType, quantize_dynamic

            target = graph.with_name("model.int8.onnx")
            names = {n.name for n in onnx.load(str(graph), load_external_data=False).graph.node}
            missing = [name for name in exclude_nodes if name not in names]
            if missing:
                raise ValueError(f"Node names not in the graph: {missing}")
            # The float16 embedding is written in a second pass over the quantized graph, as training round 1 did.
            quantized = graph.with_name("model.pc-int8.onnx") if float16_embedding else target
            quantize_dynamic(
                str(graph),
                str(quantized),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
                per_channel=per_channel,
                nodes_to_exclude=list(exclude_nodes),
            )
            if float16_embedding:
                store_embedding_float16(quantized, target)
                quantized.unlink()
            quant_session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            actual = quant_session.run(["logits"], {k: batch[k].numpy() for k in inputs})[0]
            original = decisions(reference_logits, examples, predictor.metadata)
            compressed = decisions(actual, examples, predictor.metadata)
            labeled = [(i, e) for i, e in enumerate(examples) if e.gold_ids is not None]
            loss = 0.0
            if labeled:
                before = sum(original[i] in (e.gold_ids or [None]) for i, e in labeled) / len(labeled)
                after = sum(compressed[i] in (e.gold_ids or [None]) for i, e in labeled) / len(labeled)
                loss = before - after
                if loss > max_accuracy_loss + 1e-12:
                    raise ValueError(f"INT8 fixture accuracy loss {loss:.4f} exceeds {max_accuracy_loss:.4f}")
            report["int8"] = {
                "decision_agreement": sum(a == b for a, b in zip(original, compressed)) / len(examples),
                "labelled_accuracy_loss": loss if labeled else None,
                "max_logit_error": float(np.max(np.abs(reference_logits - actual))),
                "requires_full_evaluation": True,
                "per_channel": per_channel,
                "excluded_nodes": list(exclude_nodes),
                "float16_embedding": float16_embedding,
            }
        (output / "parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception:
        # Do not leave an apparently deployable graph after a failed parity check.
        shutil.rmtree(graph.parent, ignore_errors=True)
        raise


def decisions(logits, examples, metadata):
    """Deployment selection for quantization comparisons (including abstention)."""
    calibration = metadata.get("calibration", {})
    temp = calibration.get("temperature", 1.0)
    selected = []
    for row, example in zip(logits, examples):
        scores = np.asarray([*row[: len(example.candidates)], row[metadata.get("nota_index", 127)]], dtype=float) / temp
        scores -= scores.max()
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities, kind="stable")
        best = int(order[0])
        margin = probabilities[best] - probabilities[order[1]]
        if (
            best == len(example.candidates)
            or probabilities[best] < calibration.get("min_probability", 0)
            or margin < calibration.get("min_margin", 0)
        ):
            selected.append(None)
        else:
            selected.append(example.candidates[best].id)
    return selected


EMBEDDING = "model.model.embeddings.tok_embeddings.weight"


def store_embedding_float16(source, target, name=EMBEDDING):
    """Store the token embedding of an ONNX graph in float16, with a Cast to float as the graph's first node."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(source))
    graph = model.graph
    initializer = next(i for i in graph.initializer if i.name == name)
    half = numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16), name + "_fp16")
    graph.initializer.remove(initializer)
    graph.initializer.append(half)
    graph.node.insert(0, helper.make_node("Cast", [half.name], [name], to=onnx.TensorProto.FLOAT, name=name + "_cast"))
    onnx.save(model, str(target))


def first_prompts(examples, tokenizer, metadata, count):
    """One example per distinct prompt, the first ``count`` prompts in order of their SHA-256."""
    from wsd.scoring import prompt_sha256

    by_hash = {}
    for example in examples:
        by_hash.setdefault(prompt_sha256(example, tokenizer, metadata), example)
    return [by_hash[h] for h in sorted(by_hash)[:count]]


def activation_outliers(predictor, examples, count=200, device="cpu"):
    """Input statistics of every nn.Linear over the first ``count`` prompts (one at a time, float32): the mean and
    maximum over prompts of max |x|, and the mean over prompts of max |x| divided by its 99.9th percentile (taken
    over the first 2,000,000 values of |x|), as training round 1 measured them (act_stats_model.py)."""
    from collections import defaultdict

    from wsd.tokenizer import encode_examples

    model = predictor.model.to(device).float().eval()
    names = {module: name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    stats = defaultdict(lambda: defaultdict(list))

    def hook(module, inputs, output):
        x = inputs[0].detach().float().reshape(-1)
        s = stats[names[module]]
        s["max"].append(x.abs().max().item())
        s["outlier"].append((x.abs().max() / torch.quantile(x.abs()[:2_000_000], 0.999).clamp_min(1e-9)).item())

    handles = [module.register_forward_hook(hook) for module in names]
    try:
        with torch.inference_mode():
            for example in first_prompts(examples, predictor.tokenizer, predictor.metadata, count):
                batch = encode_examples([example], predictor.tokenizer, predictor.metadata)
                model(**{k: v.to(device) for k, v in batch.items()})
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    for name in names.values():
        s = stats[name]
        rows.append(
            {
                "name": name,
                "act_max_mean": sum(s["max"]) / len(s["max"]),
                "act_max_max": max(s["max"]),
                "outlier_ratio": sum(s["outlier"]) / len(s["outlier"]),
            }
        )
    return rows


def outlier_layers(rows, ratio=15.0):
    """Layers whose feed-forward output (mlp.Wo) input peaks above ``ratio`` times its 99.9th percentile."""
    return sorted(
        int(r["name"].split(".")[-3]) for r in rows if r["name"].endswith("mlp.Wo") and r["outlier_ratio"] > ratio
    )


def outlier_nodes(layers):
    return [f"/model/model/layers.{i}/mlp/Wo/MatMul" for i in layers]


def compare_graphs(reference, candidate, examples, tokenizer, metadata, threads=16):
    """Score each prompt alone with two ONNX graphs on the CPU and compare: the same top choice (offered candidates
    and none-fits), the same set of the three best offered candidates, and the logit error after removing each
    prompt's mean (order depends only on differences). The measures of screen.py in training round 1."""
    import onnxruntime as ort

    from wsd.tokenizer import encode_examples

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    sessions = [
        ort.InferenceSession(str(g), options, providers=["CPUExecutionProvider"]) for g in (reference, candidate)
    ]
    nota = metadata.get("nota_index", 127)
    top = top3 = 0
    errors, largest = [], []
    for example in examples:
        feed = {k: v.numpy() for k, v in encode_examples([example], tokenizer, metadata).items()}
        r, c = (
            np.array([*row[: len(example.candidates)], row[nota]], dtype=np.float64)
            for row in (s.run(["logits"], feed)[0][0] for s in sessions)
        )
        top += int(r.argmax() == c.argmax())
        k = min(3, len(r) - 1)
        top3 += int(set(np.argsort(-r[:-1])[:k]) == set(np.argsort(-c[:-1])[:k]))
        d = (c - c.mean()) - (r - r.mean())
        errors.append(float(np.sqrt(np.mean(d**2))))
        largest.append(float(np.abs(d).max()))
    n = len(examples)
    return {
        "n": n,
        "top1_agree": top / n,
        "top3_set_agree": top3 / n,
        "rmse_median": float(np.median(errors)),
        "rmse_mean": float(np.mean(errors)),
        "maxerr_p90": float(np.percentile(largest, 90)),
        "threads": threads,
    }


class ONNXPredictor:
    """CPU deployment graph using exactly the Python tokenizer and decoder contract."""

    def __init__(self, directory, *, quantized=False):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "model_metadata.json").read_text(encoding="utf-8"))
        calibration = self.directory / "calibration.json"
        if calibration.exists():
            self.metadata["calibration"] = json.loads(calibration.read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
        name = "model.int8.onnx" if quantized else "model.onnx"
        self.session = ort.InferenceSession(str(self.directory / "onnx" / name), providers=["CPUExecutionProvider"])

    @classmethod
    def from_pretrained(cls, path_or_id, revision=None, local_files_only=False, device="cpu", quantized=False):
        if device != "cpu":
            raise ValueError("ONNX runtime currently supports CPU only")
        directory = Path(path_or_id)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            directory = Path(
                snapshot_download(
                    str(path_or_id),
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=["*.json", "*.model", "*.txt", "onnx/*"],
                )
            )
        return cls(directory, quantized=quantized)

    def logits(self, examples):
        from wsd.tokenizer import encode_examples

        if not examples:
            return torch.empty(0, 128)
        encoded = encode_examples(examples, self.tokenizer, self.metadata)
        output = self.session.run(["logits"], {k: value.numpy() for k, value in encoded.items()})[0]
        return torch.from_numpy(output)

    def predict(self, examples):
        from wsd.predict import decode

        for example in examples:
            example.validate()
        valid = [e for e in examples if e.candidates]
        rows = iter(self.logits(valid))
        return [decode(e, next(rows) if e.candidates else torch.zeros(128), self.metadata) for e in examples]


def main(argv=None):
    import gzip

    from wsd.data import read_jsonl
    from wsd.predict import Predictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-accuracy-loss", type=float, default=0.01, help="INT8 labelled fixture accuracy budget")
    parser.add_argument("--per-channel", action="store_true", help="INT8 MatMul weights per output channel")
    parser.add_argument(
        "--float16-embedding", action="store_true", help="Store the token embedding in float16 with a Cast"
    )
    parser.add_argument(
        "--outlier-examples", nargs="+", type=Path, help="Measure activation outliers on these examples' prompts"
    )
    parser.add_argument("--outlier-count", type=int, default=200, help="First N distinct prompts by SHA-256")
    parser.add_argument("--outlier-ratio", type=float, default=15.0, help="Keep mlp.Wo in float above this ratio")
    parser.add_argument("--outlier-device", default="cpu")
    parser.add_argument("--check-examples", nargs="+", type=Path, help="Compare INT8 with float on these prompts")
    parser.add_argument("--check-count", type=int, default=1000, help="First N distinct prompts by SHA-256")
    parser.add_argument("--check-threads", type=int, default=16)
    args = parser.parse_args(argv)
    if (args.outlier_examples or args.per_channel or args.float16_embedding or args.check_examples) and not (
        args.quantize
    ):
        parser.error("the quantization options need --quantize")
    predictor = Predictor.from_pretrained(args.model, revision=args.revision, local_files_only=args.local_files_only)
    stats, layers = None, []
    if args.outlier_examples:
        examples = [example for path in args.outlier_examples for example in read_jsonl(path)]
        stats = activation_outliers(predictor, examples, args.outlier_count, args.outlier_device)
        layers = outlier_layers(stats, args.outlier_ratio)
    report = export_model(
        predictor,
        read_jsonl(args.fixtures),
        args.output,
        quantize=args.quantize,
        atol=args.atol,
        rtol=args.rtol,
        max_accuracy_loss=args.max_accuracy_loss,
        per_channel=args.per_channel,
        exclude_nodes=outlier_nodes(layers),
        float16_embedding=args.float16_embedding,
    )
    if stats is not None:
        (args.output / "activation-stats.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in stats), encoding="utf-8"
        )
    if args.quantize and (stats is not None or args.per_channel or args.float16_embedding):
        data = (args.output / "onnx" / "model.int8.onnx").read_bytes()
        recipe = {"model": args.model}
        if stats is not None:
            recipe.update(
                outlier_rule=f"mlp.Wo input max/p99.9 > {args.outlier_ratio}",
                outlier_layers=layers,
                excluded_nodes=outlier_nodes(layers),
            )
        recipe["quantize_dynamic"] = {
            "weight_type": "QInt8",
            "per_channel": args.per_channel,
            "op_types_to_quantize": ["MatMul"],
        }
        if args.float16_embedding:
            recipe["token_embedding"] = "float16 + Cast"
        recipe.update(stored_bytes=len(data), gzip6_bytes=len(gzip.compress(data, 6)))
        (args.output / "recipe.json").write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    if args.check_examples:
        examples = [example for path in args.check_examples for example in read_jsonl(path)]
        chosen = first_prompts(examples, predictor.tokenizer, predictor.metadata, args.check_count)
        onnx_dir = args.output / "onnx"
        check = compare_graphs(
            onnx_dir / "model.onnx",
            onnx_dir / "model.int8.onnx",
            chosen,
            predictor.tokenizer,
            predictor.metadata,
            args.check_threads,
        )
        (args.output / "int8-check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
        report["int8_check"] = check
    print(json.dumps({k: v for k, v in report.items() if k in ("float_parity", "int8", "int8_check")}, indent=2))


if __name__ == "__main__":
    main()
