"""Evaluate candidate selection, abstention, calibration, and CPU latency.

Metrics operate on saved predictions as well as live models. No benchmark scores
are bundled: reports describe exactly the examples supplied to this command.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _ratio(a, b):
    return a / b if b else None


def _gold(example):
    if example.gold_ids is None:
        raise ValueError(f"{example.id}: evaluation requires gold_ids (use [] for none fits)")
    return set(example.gold_ids)


def _distribution(example, prediction):
    if prediction.get("example_id") != example.id:
        raise ValueError(f"prediction/example ID mismatch for {example.id}")
    values = prediction.get("probabilities", {})
    ids = [c.id for c in example.candidates]
    if set(values) != set(ids):
        raise ValueError(f"{example.id}: predictions must score every candidate exactly once")
    probabilities = [float(values[c]) for c in ids] + [float(prediction["none_probability"])]
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError(f"{example.id}: invalid probability")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-4):
        raise ValueError(f"{example.id}: probabilities must sum to one")
    return ids + [None], probabilities


def _is_correct(choice, gold):
    return choice in gold if gold else choice is None


def paired_bootstrap(model_correct, baseline_correct, samples=2000, seed=42):
    """Paired example bootstrap; samples sharing documents should be interpreted cautiously.

    Each resample draws len(delta) indices with random.Random(seed).randrange; the interval is the 2.5th and 97.5th
    percentile of the sorted resample means (indices int(0.025 R) and int(0.975 R) - 1), the procedure of training
    round 1's published intervals."""
    if len(model_correct) != len(baseline_correct):
        raise ValueError("paired results have different lengths")
    if not model_correct:
        return {"difference": None, "ci95": [None, None], "samples": samples}
    rng = random.Random(seed)
    delta = [int(a) - int(b) for a, b in zip(model_correct, baseline_correct)]
    n = len(delta)
    estimates = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return {
        "difference": statistics.mean(delta),
        "ci95": [estimates[int(0.025 * samples)], estimates[max(int(0.975 * samples) - 1, 0)]],
        "samples": samples,
        "seed": seed,
        "unit": "example",
    }


def _aggregate(rows):
    marked = [r for r in rows if r["reason"] == "selected"]
    absent = [r for r in rows if not r["gold"]]
    inventory = [r for r in rows if r["gold"]]
    ambiguous = [r for r in inventory if r["candidate_count"] > 1]
    single = [r for r in inventory if r["candidate_count"] == 1]
    return {
        "count": len(rows),
        "top1_accuracy": _ratio(sum(r["correct"] for r in rows), len(rows)),
        "first_candidate_accuracy": _ratio(sum(r["baseline_correct"] for r in rows), len(rows)),
        "marked_precision": _ratio(sum(r["marked_correct"] for r in marked), len(marked)),
        "coverage": _ratio(len(marked), len(rows)),
        "in_inventory_coverage": _ratio(sum(r["reason"] == "selected" for r in inventory), len(inventory)),
        "ambiguous_top1_accuracy": _ratio(sum(r["correct"] for r in ambiguous), len(ambiguous)),
        "ambiguous_first_candidate_accuracy": _ratio(sum(r["baseline_correct"] for r in ambiguous), len(ambiguous)),
        "ambiguous_marked_precision": _ratio(
            sum(r["marked_correct"] for r in ambiguous if r["reason"] == "selected"),
            sum(r["reason"] == "selected" for r in ambiguous),
        ),
        "ambiguous_coverage": _ratio(sum(r["reason"] == "selected" for r in ambiguous), len(ambiguous)),
        "single_sense_accuracy": _ratio(sum(r["correct"] for r in single), len(single)),
        "none_fits_count": len(absent),
        "false_confident_none_rate": _ratio(sum(r["reason"] == "selected" for r in absent), len(absent)),
        "none_fits_recall": _ratio(sum(r["reason"] == "none_fits" for r in absent), len(absent)),
    }


def summarize(examples, predictions, bootstrap_samples=2000):
    if len(examples) != len(predictions):
        raise ValueError("prediction count differs from evaluation count")
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("duplicate example IDs")
    rows = []
    for example, prediction in zip(examples, predictions):
        gold = _gold(example)
        ids, probabilities = _distribution(example, prediction)
        top = max(range(len(ids)), key=probabilities.__getitem__)
        reason = prediction.get("reason")
        selected = prediction.get("selected_id")
        if reason not in {"selected", "none_fits", "uncertain", "no_candidates"}:
            raise ValueError(f"{example.id}: invalid decision reason")
        if reason == "selected" and selected not in ids[:-1]:
            raise ValueError(f"{example.id}: selected_id is not a candidate")
        if reason != "selected" and selected is not None:
            raise ValueError(f"{example.id}: abstention must have selected_id=null")
        rows.append(
            {
                "gold": gold,
                "candidate_count": len(example.candidates),
                "correct": _is_correct(ids[top], gold),
                "confidence": probabilities[top],
                "baseline_correct": _is_correct(ids[0] if example.candidates else None, gold),
                "reason": reason,
                "marked_correct": selected in gold,
                "language": example.language,
                "domain": example.provenance.get("domain", "unspecified"),
                "phrase_length": len(example.target.text.split()),
            }
        )
    report = _aggregate(rows)
    report["paired_improvement"] = paired_bootstrap(
        [r["correct"] for r in rows], [r["baseline_correct"] for r in rows], bootstrap_samples
    )
    ambiguous = [r for r in rows if r["gold"] and r["candidate_count"] > 1]
    report["ambiguous_paired_improvement"] = paired_bootstrap(
        [r["correct"] for r in ambiguous], [r["baseline_correct"] for r in ambiguous], bootstrap_samples
    )
    bins = []
    for i in range(10):
        members = [r for r in rows if i / 10 <= r["confidence"] < (i + 1) / 10 or i == 9 and r["confidence"] == 1]
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": len(members),
                "confidence": statistics.mean(r["confidence"] for r in members) if members else None,
                "accuracy": statistics.mean(r["correct"] for r in members) if members else None,
            }
        )
    report["calibration"] = {
        "bins": bins,
        "ece": sum(b["count"] * abs(b["confidence"] - b["accuracy"]) for b in bins if b["count"]) / len(rows)
        if rows
        else None,
    }
    ordered = sorted(rows, key=lambda r: r["confidence"], reverse=True)
    report["risk_coverage"] = []
    for threshold in range(1, 11):
        n = math.ceil(len(rows) * threshold / 10)
        report["risk_coverage"].append(
            {
                "coverage": _ratio(n, len(rows)),
                "risk": 1 - statistics.mean(r["correct"] for r in ordered[:n]) if n else None,
            }
        )
    for field in ("language", "domain", "candidate_count", "phrase_length"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        report[f"by_{field}"] = {key: _aggregate(value) for key, value in sorted(groups.items())}
    return report


def apply_calibration(predictions, calibration):
    """Apply temperature before deciding, over candidates plus the none-fits slot."""
    temperature = float(calibration.get("temperature", 1.0))
    probability_threshold = float(calibration.get("min_probability", 0.0))
    margin_threshold = float(calibration.get("min_margin", 0.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not 0 <= probability_threshold <= 1 or not 0 <= margin_threshold <= 1:
        raise ValueError("probability and margin thresholds must be within [0,1]")
    output = []
    for original in predictions:
        row = copy.deepcopy(original)
        ids = list(row["probabilities"])
        raw = list(row["probabilities"].values()) + [row["none_probability"]]
        # Stable softmax of log probabilities. Zero remains exactly zero.
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in raw) or not math.isclose(sum(raw), 1.0, abs_tol=1e-4):
            raise ValueError("calibration input must be a complete finite probability distribution")
        logs = [math.log(p) / temperature if p > 0 else -math.inf for p in raw]
        maximum = max(logs)
        weights = [math.exp(p - maximum) for p in logs]
        probs = [p / sum(weights) for p in weights]
        order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
        top = order[0]
        runner = probs[order[1]] if len(order) > 1 else 0.0
        row["probabilities"] = dict(zip(ids, probs[:-1]))
        row["none_probability"] = probs[-1]
        row["selected_id"] = None
        if not ids:
            row["reason"] = "no_candidates"
        elif (
            calibration.get("abstain_all", False)
            or probs[top] < probability_threshold
            or probs[top] - runner < margin_threshold
        ):
            row["reason"] = "uncertain"
        elif top == len(ids):
            row["reason"] = "none_fits"
        else:
            row["reason"], row["selected_id"] = "selected", ids[top]
        row["calibration_version"] = calibration.get("version", "uncalibrated")
        output.append(row)
    return output


def fit_calibration(examples, predictions, target_precision=0.95):
    """Choose temperature by NLL, then maximize marked coverage at requested precision.

    Calibration only; its achieved precision is not a test-set guarantee.
    """
    if not examples:
        raise ValueError("calibration requires nonempty labeled examples")
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0,1]")
    for example, prediction in zip(examples, predictions):
        _gold(example)
        _distribution(example, prediction)
    if len(examples) != len(predictions):
        raise ValueError("calibration count mismatch")

    def nll(rows):
        return -statistics.mean(
            math.log(
                max(
                    sum(row["probabilities"].get(g, 0) for g in example.gold_ids)
                    if example.gold_ids
                    else row["none_probability"],
                    1e-30,
                )
            )
            for example, row in zip(examples, rows)
        )

    temperatures = (0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0)
    temperature = min(temperatures, key=lambda t: nll(apply_calibration(predictions, {"temperature": t})))
    best = None
    for threshold in [i / 100 for i in range(101)]:
        for margin in (0.0, 0.05, 0.1, 0.2, 0.3):
            calibration = {"temperature": temperature, "min_probability": threshold, "min_margin": margin}
            rows = apply_calibration(predictions, calibration)
            marked = [(e, r) for e, r in zip(examples, rows) if r["reason"] == "selected"]
            correct = sum(r["selected_id"] in e.gold_ids for e, r in marked)
            precision = _ratio(correct, len(marked))
            if precision is not None and precision >= target_precision:
                score = (len(marked), precision, -threshold, -margin)
                if best is None or score > best[0]:
                    best = (score, calibration)
    selected = best[1] if best else {"temperature": temperature, "min_probability": 1.0, "min_margin": 1.0}
    selected.update(
        {
            "version": "calibration-v1",
            "target_precision": target_precision,
            "calibration_count": len(examples),
            "feasible": best is not None,
            "abstain_all": best is None,
            "calibration_ids_sha256": hashlib.sha256("\n".join(e.id for e in examples).encode()).hexdigest(),
        }
    )
    return selected


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if values else None


def benchmark_latency(predictor, examples, repeats=1):
    durations = []
    for _ in range(repeats):
        for example in examples:
            start = time.perf_counter()
            predictor.predict([example])
            durations.append((time.perf_counter() - start) * 1000)
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    except ImportError:
        rss_bytes = None
    return {
        "unit": "ms",
        "count": len(durations),
        "p50": _percentile(durations, 0.5),
        "p95": _percentile(durations, 0.95),
        "peak_process_rss_bytes": rss_bytes,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "note": "Process peak includes model load and evaluation; not incremental model memory.",
    }


def release_gates(report):
    ambiguous = report.get("ambiguous_paired_improvement", {})
    difference, interval = ambiguous.get("difference"), ambiguous.get("ci95", [None, None])

    def at_least(value, minimum):
        return None if value is None else value >= minimum

    def at_most(value, maximum):
        return None if value is None else value <= maximum

    checks = {
        "ambiguous_accuracy_gain_at_least_5pp": at_least(difference, 0.05),
        "ambiguous_paired_ci_above_zero": None if interval[0] is None else interval[0] > 0,
        "ambiguous_marked_precision_at_least_95pct": at_least(report.get("ambiguous_marked_precision"), 0.95),
        "ambiguous_coverage_at_least_50pct": at_least(report.get("ambiguous_coverage"), 0.5),
        "false_confident_none_at_most_5pct": at_most(report.get("false_confident_none_rate"), 0.05),
        "warm_p95_below_1s": None if report.get("latency", {}).get("p95") is None else report["latency"]["p95"] < 1000,
    }
    return {
        "passed": all(value is True for value in checks.values()),
        "checks": checks,
        "note": "Null checks lack evidence and fail the gate. Hardware must be reviewed separately; this command does not certify a Windows laptop.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--predictions", type=Path, help="Saved prediction JSONL; no model loading")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="Explicit PyTorch CPU thread count")
    parser.add_argument("--backend", choices=("torch", "onnx", "onnx-int8"), default="torch")
    parser.add_argument(
        "--require-gates", action="store_true", help="Exit nonzero unless every product gate has evidence and passes"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--latency-repeats", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        args.threads < 1
        or args.batch_size < 1
        or args.bootstrap_samples < 1
        or args.latency_repeats < 0
        or args.permutations < 0
    ):
        parser.error("batch/bootstrap counts must be positive; repeats/permutations nonnegative")
    if args.calibration_data and (not args.model or not args.calibration_output or args.calibration):
        parser.error("calibration fitting needs --model and --calibration-output, without --calibration")
    if args.predictions and (args.latency_repeats or args.permutations):
        parser.error("latency/permutation evaluation requires --model")
    from wsd.data import read_jsonl, validate_splits
    from wsd.pipeline import require_reviewed

    examples = read_jsonl(args.data)
    require_reviewed(examples)
    predictor = None
    cold_load = None
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    if args.model:
        import torch

        torch.set_num_threads(args.threads)
        from wsd.predict import Predictor

        if args.backend != "torch":
            from wsd.export import ONNXPredictor

            Predictor = ONNXPredictor
        start = time.perf_counter()
        predictor = Predictor.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=args.local_files_only,
            device=args.device,
            **({"quantized": args.backend == "onnx-int8"} if args.backend != "torch" else {}),
        )
        cold_load = (time.perf_counter() - start) * 1000
        # Remove existing calibration to fit/apply exactly once.
        if args.calibration_data or calibration:
            predictor.metadata["calibration"] = {
                "temperature": 1.0,
                "min_probability": 0.0,
                "min_margin": 0.0,
                "version": "uncalibrated",
            }

        def predict(data):
            return [
                r
                for i in range(0, len(data), args.batch_size)
                for r in predictor.predict(data[i : i + args.batch_size])
            ]

        if args.calibration_data:
            validate_splits({"test": args.data, "calibration": args.calibration_data})
            calibration_examples = read_jsonl(args.calibration_data)
            require_reviewed(calibration_examples)
            calibration = fit_calibration(calibration_examples, predict(calibration_examples), args.target_precision)
            args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
            args.calibration_output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        predictions = predict(examples)
    else:
        predictions = [
            json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    if calibration:
        predictions = apply_calibration(predictions, calibration)
    report = summarize(examples, predictions, args.bootstrap_samples)
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "requested_revision": args.revision,
            "cold_load_ms": cold_load,
            "resolved_model_revisions": sorted({r.get("model_revision", "unknown") for r in predictions}),
            "backend": args.backend,
            "torch_threads": args.threads if args.model else None,
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "calibration_parameters": calibration,
        }
    )
    if args.latency_repeats:
        report["latency"] = benchmark_latency(predictor, examples, args.latency_repeats)
    if args.permutations:
        from dataclasses import replace

        rng = random.Random(42)
        changed, count = 0, 0
        for _ in range(args.permutations):
            shuffled = []
            for example in examples:
                candidates = list(example.candidates)
                rng.shuffle(candidates)
                shuffled.append(replace(example, candidates=candidates))
            permuted = predict(shuffled)
            if calibration:
                permuted = apply_calibration(permuted, calibration)
            for a, b in zip(predictions, permuted):
                changed += (a["selected_id"], a["reason"]) != (b["selected_id"], b["reason"])
                count += 1
        report["candidate_order"] = {"comparisons": count, "decision_change_rate": _ratio(changed, count), "seed": 42}
    report["release_gates"] = release_gates(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("count", "top1_accuracy", "marked_precision", "coverage")}, indent=2))
    if args.require_gates and not report["release_gates"]["passed"]:
        raise SystemExit("Product release gates failed or lack required evidence; see evaluation.json")
    return report


if __name__ == "__main__":
    main()
