"""Validate and checksum a release; upload only when --upload is explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "model_metadata.json",
    "calibration.json",
    "training_config.json",
    "data_manifest.json",
    "evaluation.json",
)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(directory):
    directory = Path(directory).resolve()
    missing = [f for f in REQUIRED if not (directory / f).is_file()]
    if missing:
        raise ValueError(f"Missing release files: {', '.join(missing)}")
    files = [p for p in directory.rglob("*") if p.is_file() and p.name != "checksums.json"]
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Release artifacts must not contain symbolic links")
    prohibited = [
        str(p.relative_to(directory))
        for p in files
        if any(part.startswith(".") for part in p.relative_to(directory).parts)
        or p.suffix in (".bin", ".pt", ".pth")
        or p.name in ("config.toml", "credentials.json")
    ]
    if prohibited:
        raise ValueError(f"Unexpected or sensitive release files: {prohibited}")
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model_metadata.json",
        "calibration.json",
        "training_config.json",
        "data_manifest.json",
        "evaluation.json",
    ):
        if not isinstance(json.loads((directory / name).read_text(encoding="utf-8")), dict):
            raise TypeError(f"{name} must contain a JSON object")
    metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    if len(metadata.get("letters", [])) != 128 or metadata.get("nota_index") != 127:
        raise ValueError("Invalid compact answer mapping")
    if not metadata.get("template_version"):
        raise ValueError("Missing prompt format version")
    card = (directory / "README.md").read_text(encoding="utf-8")
    if len(card.strip()) < 100 or "license:" not in card or "base_model:" not in card:
        raise ValueError("Model card needs YAML license/base_model metadata and usage/evaluation text")
    manifest = {str(p.relative_to(directory)): checksum(p) for p in sorted(files)}
    old = directory / "checksums.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8")) != manifest:
        raise ValueError(
            "Artifact has changed since checksums were generated; explicitly remove stale checksums after review"
        )
    return manifest


def clean_reload(directory):
    """Load in a separate process with offline flags and a temporary working directory."""
    directory = str(Path(directory).resolve())
    with tempfile.TemporaryDirectory(prefix="wsd-release-check-") as temporary:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from wsd.predict import Predictor; import sys; "
                    "Predictor.from_pretrained(sys.argv[1], local_files_only=True)"
                ),
                directory,
            ],
            cwd=temporary,
            check=True,
            env=_offline_environment(),
        )


def _offline_environment():
    import os

    return {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def describe_recipe(recipe):
    """The export block of model_metadata.json, from an export's recipe.json."""
    granularity = "per channel" if recipe.get("quantize_dynamic", {}).get("per_channel") else "per tensor"
    block = {"graph": f"dynamic quantization, INT8 weights {granularity} for MatMul, activations uint8 at run time"}
    block["float_matmuls"] = list(recipe.get("excluded_nodes", []))
    if recipe.get("token_embedding"):
        block["token_embedding"] = "float16, cast to float at session creation"
    return block


def assemble_release(release, artifact, *, export=None, training_config=None, data_manifest=None, evaluation=None):
    """Build the folder of a model repository from an export and a release file (JSON).

    The release file names, relative to itself, the model card (``card``), ``license`` and ``notice``; it lists the
    ``files`` copied from the export folder, and ``release_files`` (name -> path) copied from the release's own
    folder in their place; ``recipe`` copies the export's recipe.json to ``to`` with its ``model``
    field replaced; ``metadata`` updates the export's model_metadata.json (``model_revision: "weights"`` means
    "sha256:" + the SHA-256 of model.safetensors) and an ``export`` block made from the recipe is added last. The
    run's training configuration, data manifest and evaluation reports (a mapping name -> file, merged into
    evaluation.json) are copied in. ``export``, ``training_config``, ``data_manifest`` and ``evaluation`` given here
    take the place of the release file's own entries (paths relative to the working directory)."""
    import shutil

    release = Path(release)
    config = json.loads(release.read_text(encoding="utf-8"))
    base = release.parent

    def source(value, override):
        if override is not None:
            return Path(override)
        if value is None:
            return None
        return Path(value) if Path(value).is_absolute() else base / value

    export = source(config.get("export"), export)
    if export is None:
        raise ValueError("The release needs an export folder (--export)")
    names = [*config["files"], *config.get("release_files", {})]
    if config.get("recipe"):
        names.append(config["recipe"].get("to", "onnx/recipe.json"))
    for name in names:
        inside = PurePosixPath(name)
        if not name or inside.is_absolute() or ".." in inside.parts or "\\" in name:
            raise ValueError(f"Release file name {name!r} is outside the artifact folder")
    artifact = Path(artifact)
    if artifact.exists() and any(artifact.iterdir()):
        raise ValueError("Release artifact directory must be empty")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    # Built in a temporary folder beside the artifact and moved into place, so a failure leaves nothing behind.
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact.name}.", dir=artifact.parent))
    try:
        for name in config["files"]:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(export / name, staging / name)
        for name, path in config.get("release_files", {}).items():
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source(path, None), staging / name)
        for key, name in (("card", "README.md"), ("license", "LICENSE"), ("notice", "NOTICE")):
            shutil.copyfile(source(config[key], None), staging / name)
        recipe = None
        if config.get("recipe"):
            recipe = json.loads((export / config["recipe"].get("from", "recipe.json")).read_text(encoding="utf-8"))
            if "model" in config["recipe"]:
                recipe["model"] = config["recipe"]["model"]
            target = staging / config["recipe"].get("to", "onnx/recipe.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(recipe, indent=1) + "\n", encoding="utf-8")
        metadata = json.loads((export / "model_metadata.json").read_text(encoding="utf-8"))
        metadata.pop("export", None)
        for key, value in config.get("metadata", {}).items():
            if key == "model_revision" and value == "weights":
                value = "sha256:" + checksum(staging / "model.safetensors")
            metadata[key] = value
        if recipe is not None:
            metadata["export"] = describe_recipe(recipe)
        (staging / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, value, override in (
            ("training_config.json", config.get("training_config"), training_config),
            ("data_manifest.json", config.get("data_manifest"), data_manifest),
        ):
            path = source(value, override)
            if path is not None:
                shutil.copyfile(path, staging / name)
        if evaluation is not None:
            reports = {name: Path(path) for name, path in evaluation.items()}
        else:
            reports = {name: source(path, None) for name, path in config.get("evaluation", {}).items()}
        if reports:
            merged = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in reports.items()}
            (staging / "evaluation.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if artifact.exists():
        artifact.rmdir()
    staging.rename(artifact)
    return artifact


def publish(directory, *, repo_id=None, upload=False, private=False, revision="main"):
    directory = Path(directory).resolve()
    manifest = validate_artifact(directory)
    clean_reload(directory)
    (directory / "checksums.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"validated": True, "files": len(manifest), "uploaded": False, "directory": str(directory)}
    if not upload:
        return result
    if not repo_id or "/" not in repo_id:
        raise ValueError("--upload requires an explicit namespace/repository --repo-id")
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(directory),
        revision=revision,
        commit_message="Publish validated WSD model artifact",
    )
    # No cached local artifact is used to stand in for verification of the uploaded revision.
    with tempfile.TemporaryDirectory(prefix="wsd-hub-check-") as temporary:
        downloaded = Path(snapshot_download(repo_id, revision=commit.oid, local_dir=temporary))
        for name, expected in manifest.items():
            if checksum(downloaded / name) != expected:
                raise ValueError(f"Uploaded artifact checksum mismatch: {name}")
        clean_reload(downloaded)
    result.update(uploaded=True, repo_id=repo_id, commit=commit.oid)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--assemble", type=Path, help="Release file (JSON): build --artifact from an export first")
    parser.add_argument("--export", type=Path, help="Export folder (wsd export) for --assemble")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--evaluation", nargs="+", help="NAME=FILE reports merged into evaluation.json")
    parser.add_argument("--repo-id")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    if args.assemble:
        if args.evaluation and any("=" not in item for item in args.evaluation):
            parser.error("--evaluation takes NAME=FILE")
        assemble_release(
            args.assemble,
            args.artifact,
            export=args.export,
            training_config=args.training_config,
            data_manifest=args.data_manifest,
            evaluation=dict(item.split("=", 1) for item in args.evaluation) if args.evaluation else None,
        )
    elif args.export or args.training_config or args.data_manifest or args.evaluation:
        parser.error("--export, --training-config, --data-manifest and --evaluation need --assemble")
    print(
        json.dumps(
            publish(
                args.artifact, repo_id=args.repo_id, upload=args.upload, private=args.private, revision=args.revision
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
