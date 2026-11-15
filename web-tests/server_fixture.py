"""Real HTTP/store/UI with mocked paid provider in an isolated temporary workspace."""

import tempfile
from pathlib import Path

import wsd.jev
from wsd.annotation_server import AnnotationServer
from wsd.annotation_store import Store


def fake_assist(example, key, provider="typesafe", model="jev-latest"):
    if model == "fixture-failure":
        raise wsd.jev.JevError("Jev connection failed or timed out. Your annotation is unchanged.")
    assert key == "browser-fixture-key"
    probs = {f"c{i}": (0.9 if i == 0 else 0.05 / (len(example.candidates) - 1)) for i in range(len(example.candidates))}
    probs.update(none=0.025, insufficient=0.025)
    response = {
        "model": "jev-fixture-1",
        "answers": {"meaning": dict(type="choice", choice="c0", confidence=0.8, probabilities=probs)},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    return wsd.jev.parse_response(response, example, provider, model, "browser-fixture")


wsd.jev.assist = fake_assist
with tempfile.TemporaryDirectory(prefix="sense-review-e2e-") as directory:
    with AnnotationServer(("127.0.0.1", 18865), Store(Path(directory) / "workspace.sqlite")) as server:
        print("Annotation browser fixture ready", flush=True)
        server.serve_forever()
