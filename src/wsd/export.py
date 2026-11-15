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


def export_model(predictor, examples, output: Path, *, quantize=False, atol=1e-4, rtol=1e-3, max_accuracy_loss=0.01):
    """Export and check every fixture singly and together (variable batch/sequence sizes).

    Quantized parity reports probability drift and decision agreement. A quantized
    artifact is rejected if its labelled accuracy drops by more than the budget.
    Fixtures are a mechanical check, not a substitute for a full release evaluation.
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
            from onnxruntime.quantization import QuantType, quantize_dynamic

            target = graph.with_name("model.int8.onnx")
            quantize_dynamic(str(graph), str(target), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
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
    args = parser.parse_args(argv)
    predictor = Predictor.from_pretrained(args.model, revision=args.revision, local_files_only=args.local_files_only)
    report = export_model(
        predictor, read_jsonl(args.fixtures), args.output, quantize=args.quantize, atol=args.atol, rtol=args.rtol
    )
    print(json.dumps({k: v for k, v in report.items() if k in ("float_parity", "int8")}, indent=2))


if __name__ == "__main__":
    main()
