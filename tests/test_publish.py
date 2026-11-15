import pytest
from test_core import tiny_predictor

from wsd.publish import clean_reload, publish, validate_artifact


def artifact(tmp_path):
    p = tiny_predictor()
    p.save_pretrained(tmp_path)
    (tmp_path / "README.md").write_text(
        "---\nlicense: mit\nbase_model: local/tiny-test\n---\n"
        + ("A synthetic model used only for release tests. " * 5)
    )
    for name in ["LICENSE", "NOTICE"]:
        (tmp_path / name).write_text("Synthetic fixture; not a trained release.")
    for name in ["training_config.json", "data_manifest.json", "evaluation.json"]:
        (tmp_path / name).write_text("{}")
    return tmp_path


def test_publish_is_dry_run_and_checksum_guard(tmp_path):
    directory = artifact(tmp_path)
    result = publish(directory)
    assert result["uploaded"] is False
    assert (directory / "checksums.json").is_file()
    clean_reload(directory)
    (directory / "NOTICE").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        validate_artifact(directory)


def test_publish_missing_and_symlink(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        validate_artifact(tmp_path)
    artifact(tmp_path)
    (tmp_path / "linked").symlink_to(tmp_path / "README.md")
    with pytest.raises(ValueError, match="symbolic"):
        validate_artifact(tmp_path)
