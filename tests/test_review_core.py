import json

import pytest
import torch
from test_core import example, tiny_predictor

from wsd.predict import Predictor, decode, validate_metadata


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", float("inf")),
        ("temperature", float("nan")),
        ("temperature", 0),
        ("min_probability", -1),
        ("min_margin", 2),
    ],
)
def test_invalid_calibration_fails(key, value):
    predictor = tiny_predictor()
    predictor.metadata["calibration"] = {key: value}
    with pytest.raises(ValueError):
        validate_metadata(predictor.metadata, predictor.tokenizer)
    with pytest.raises(ValueError):
        decode(example(), torch.zeros(128), predictor.metadata)


def test_unknown_compact_checkpoint_requires_prompt_metadata(tmp_path):
    tiny_predictor().save_pretrained(tmp_path)
    (tmp_path / "model_metadata.json").unlink()
    with pytest.raises(ValueError, match="prompt format cannot be inferred"):
        Predictor.from_pretrained(tmp_path, local_files_only=True)


def test_saved_mapping_is_authoritative(tmp_path):
    predictor = tiny_predictor()
    predictor.metadata["letters"][0], predictor.metadata["letters"][1] = (
        predictor.metadata["letters"][1],
        predictor.metadata["letters"][0],
    )
    predictor.save_pretrained(tmp_path)
    loaded = Predictor.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.metadata["letters"] == predictor.metadata["letters"]


def test_unknown_template_fails_loading(tmp_path):
    tiny_predictor().save_pretrained(tmp_path)
    meta = json.loads((tmp_path / "model_metadata.json").read_text())
    meta["template_version"] = "unrecognized-v9"
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="template_version"):
        Predictor.from_pretrained(tmp_path, local_files_only=True)


def test_infeasible_calibration_abstains_for_exact_one_score():
    predictor = tiny_predictor()
    predictor.metadata["calibration"] = {"abstain_all": True}
    logits = torch.zeros(128)
    logits[0] = 1000
    assert decode(example(), logits, predictor.metadata)["reason"] == "uncertain"
