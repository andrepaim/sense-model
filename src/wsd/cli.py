"""Lazily dispatched model lifecycle commands."""

import argparse
import importlib
import json
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="wsd", description="Prepare, train, evaluate, export and publish WSD models")
    parser.add_argument(
        "command",
        choices=[
            "predict",
            "prepare",
            "train",
            "evaluate",
            "export",
            "publish",
            "teacher",
            "annotate",
            "pilot",
            "generate",
            "label",
            "config",
        ],
    )
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    if command == "predict":
        p = argparse.ArgumentParser(prog="wsd predict")
        p.add_argument("--model", required=True)
        p.add_argument("--revision")
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--local-files-only", action="store_true")
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=4)
        args = p.parse_args(argv[1:])
        import torch

        if args.threads < 1:
            p.error("threads must be positive")
        torch.set_num_threads(args.threads)
        from .data import read_jsonl
        from .predict import Predictor

        if args.batch_size < 1:
            p.error("batch-size must be positive")
        examples = read_jsonl(args.input)
        model = Predictor.from_pretrained(
            args.model, revision=args.revision, local_files_only=args.local_files_only, device=args.device
        )
        from pathlib import Path

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for i in range(0, len(examples), args.batch_size):
                for result in model.predict(examples[i : i + args.batch_size]):
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return 0
    module = {
        "prepare": "data",
        "teacher": "teacher",
        "annotate": "annotation_server",
        "generate": "generation",
        "label": "labeling",
        "config": "pipeline",
    }.get(command, command)
    result = importlib.import_module("wsd." + module).main(argv[1:])
    # Console entry points call sys.exit(main()); report dictionaries are successful results.
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    main()
