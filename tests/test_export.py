import json

import numpy as np
import pytest
from test_core import example, tiny_predictor

from wsd.export import decisions, export_model
from wsd.schema import Candidate, Example, Target


def test_export_dynamic_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    p = tiny_predictor()
    examples = [
        example(),
        Example("pt", "um banco", Target(3, 8, "banco"), [Candidate("seat", "um assento")], gold_ids=["seat"]),
    ]
    report = export_model(p, examples, tmp_path / "export", quantize=True, max_accuracy_loss=1)
    assert (tmp_path / "export" / "onnx" / "model.onnx").is_file()
    assert (tmp_path / "export" / "onnx" / "model.int8.onnx").is_file()
    assert [c["batch"] for c in report["float_parity"]] == [2, 1, 1]
    assert json.loads((tmp_path / "export" / "parity.json").read_text())["expected_predictions"] == p.predict(examples)


def test_export_decisions_match_predictor():
    from wsd.predict import decode

    p = tiny_predictor()
    e = example()
    for chosen in (0, 1, 127):
        logits = np.zeros((1, 128))
        logits[0, chosen] = 8
        assert decisions(logits, [e], p.metadata)[0] == decode(e, logits[0], p.metadata)["selected_id"]
    p.metadata["calibration"] = {"min_probability": 0.99}
    assert decisions(np.zeros((1, 128)), [e], p.metadata) == [None]


def test_onnx_predictor_shared_decode(tmp_path):
    pytest.importorskip("onnxruntime")
    from wsd.export import ONNXPredictor

    p = tiny_predictor()
    examples = [example()]
    export_model(p, examples, tmp_path / "model")
    q = ONNXPredictor.from_pretrained(tmp_path / "model", local_files_only=True)
    before = p.predict(examples)
    after = q.predict(examples)
    assert before[0]["selected_id"] == after[0]["selected_id"]
    assert before[0]["reason"] == after[0]["reason"]
    q.metadata["calibration"] = {"min_probability": 0.9999}
    assert q.predict(examples)[0]["reason"] == "uncertain"
