"""wsd publish --assemble: the layout of a published model repository, built from an export and a release file,
then validated as a dry run. Nothing is uploaded without --upload."""

import hashlib
import json

import pytest
from test_core import example, tiny_predictor

from wsd.publish import assemble_release, main

CARD = "---\nlicense: apache-2.0\nbase_model: local/tiny-test\n---\n" + "A synthetic model for release tests. " * 5


def export_folder(tmp_path):
    pytest.importorskip("onnxruntime")
    from wsd.export import export_model

    out = tmp_path / "export"
    export_model(tiny_predictor(), [example()], out, quantize=True, per_channel=True, max_accuracy_loss=1)
    recipe = {
        "model": "runs/x/final",
        "outlier_layers": [0],
        "excluded_nodes": ["/model/model/layers.0/mlp/Wo/MatMul"],
        "quantize_dynamic": {"weight_type": "QInt8", "per_channel": True, "op_types_to_quantize": ["MatMul"]},
        "token_embedding": "float16 + Cast",
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=1))
    return out


def release_file(tmp_path):
    folder = tmp_path / "release"
    folder.mkdir()
    (folder / "README.md").write_text(CARD)
    (folder / "LICENSE").write_text("Synthetic licence text for a test.")
    (folder / "NOTICE").write_text("Synthetic notice for a test.")
    (folder / "calibration.json").write_text('{"temperature": 1.0, "version": "release copy"}')
    release = {
        "card": "README.md",
        "license": "LICENSE",
        "notice": "NOTICE",
        "files": ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "calibration.json"]
        + ["onnx/model.onnx", "onnx/model.int8.onnx"],
        "recipe": {"to": "onnx/recipe.json", "model": "model.safetensors (this repository)"},
        "release_files": {"calibration.json": "calibration.json"},
        "metadata": {
            "model_revision": "weights",
            "base_model": "local/tiny-test",
            "base_revision": "abc",
            "training": {"round": "a test"},
        },
    }
    (folder / "release.json").write_text(json.dumps(release))
    (tmp_path / "training_config.json").write_text(json.dumps({"seed": 42}))
    (tmp_path / "data_manifest.json").write_text(json.dumps({"leakage": "passed"}))
    (tmp_path / "dev.json").write_text(json.dumps({"items": 3}))
    (tmp_path / "test.json").write_text(json.dumps({"items": 4}))
    return folder / "release.json"


def test_assemble_builds_the_repository_layout(tmp_path):
    export = export_folder(tmp_path)
    out = assemble_release(
        release_file(tmp_path),
        tmp_path / "artifact",
        export=export,
        training_config=tmp_path / "training_config.json",
        data_manifest=tmp_path / "data_manifest.json",
        evaluation={"development": tmp_path / "dev.json", "test": tmp_path / "test.json"},
    )
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == sorted(
        [
            "LICENSE",
            "NOTICE",
            "README.md",
            "calibration.json",
            "config.json",
            "data_manifest.json",
            "evaluation.json",
            "model.safetensors",
            "model_metadata.json",
            "onnx/model.int8.onnx",
            "onnx/model.onnx",
            "onnx/recipe.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "training_config.json",
        ]
    )
    assert (out / "onnx" / "model.int8.onnx").read_bytes() == (export / "onnx" / "model.int8.onnx").read_bytes()
    # A file of the release folder takes the place of the export's copy.
    assert json.loads((out / "calibration.json").read_text())["version"] == "release copy"
    meta = json.loads((out / "model_metadata.json").read_text())
    weights = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    assert meta["model_revision"] == "sha256:" + weights
    assert list(meta)[-4:] == ["base_model", "base_revision", "training", "export"]
    assert meta["export"] == {
        "graph": "dynamic quantization, INT8 weights per channel for MatMul, activations uint8 at run time",
        "float_matmuls": ["/model/model/layers.0/mlp/Wo/MatMul"],
        "token_embedding": "float16, cast to float at session creation",
    }
    assert (out / "onnx" / "recipe.json").read_text().endswith("}\n")
    recipe = json.loads((out / "onnx" / "recipe.json").read_text())
    assert recipe["model"] == "model.safetensors (this repository)" and recipe["outlier_layers"] == [0]
    assert json.loads((out / "evaluation.json").read_text()) == {"development": {"items": 3}, "test": {"items": 4}}


def test_publish_assembles_then_validates_without_uploading(tmp_path, capsys):
    export = export_folder(tmp_path)
    release = release_file(tmp_path)
    args = [
        "--artifact",
        str(tmp_path / "artifact"),
        "--assemble",
        str(release),
        "--export",
        str(export),
        "--training-config",
        str(tmp_path / "training_config.json"),
        "--data-manifest",
        str(tmp_path / "data_manifest.json"),
        "--evaluation",
        f"development={tmp_path / 'dev.json'}",
    ]
    main(args)
    result = json.loads(capsys.readouterr().out)
    assert result["validated"] is True and result["uploaded"] is False
    assert (tmp_path / "artifact" / "checksums.json").is_file()
    with pytest.raises(ValueError, match="empty"):
        main(args)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["files"].append("../outside.json"),
        lambda r: r.update(release_files={"/tmp/abs.json": "README.md"}),
        lambda r: r["recipe"].update(to="../../recipe.json"),
    ],
)
def test_names_that_leave_the_artifact_are_refused(tmp_path, change):
    export = export_folder(tmp_path)
    release = release_file(tmp_path)
    config = json.loads(release.read_text())
    change(config)
    release.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="outside the artifact"):
        assemble_release(release, tmp_path / "artifact", export=export)
    assert not (tmp_path / "artifact").exists() and not (tmp_path / "outside.json").exists()


def test_a_failed_assembly_leaves_nothing_behind(tmp_path):
    export = export_folder(tmp_path)
    release = release_file(tmp_path)
    config = json.loads(release.read_text())
    config["files"].append("missing.json")
    release.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError):
        assemble_release(release, tmp_path / "artifact", export=export)
    assert not (tmp_path / "artifact").exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".artifact")] == []
    config["files"].remove("missing.json")
    release.write_text(json.dumps(config))
    assert (assemble_release(release, tmp_path / "artifact", export=export) / "model_metadata.json").is_file()
