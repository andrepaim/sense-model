"""Local artifacts must not access the network during prediction."""

import socket

from test_core import example, tiny_predictor

from wsd.predict import Predictor


def test_local_prediction_needs_no_network(tmp_path, monkeypatch):
    tiny_predictor().save_pretrained(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("network access during offline inference")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    model = Predictor.from_pretrained(tmp_path, local_files_only=True)
    assert len(model.predict([example()])) == 1
