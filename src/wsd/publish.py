"""Validate and checksum a release; upload only when --upload is explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

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
    parser.add_argument("--repo-id")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
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
