import copy
import json
import urllib.error

import pytest

from wsd import Candidate, Example, Target
from wsd.jev import ENDPOINTS, JevError, NoRedirect, assist, build_request


def example():
    return Example(
        "private-id",
        "Ela sentou no banco.",
        Target(14, 19, "banco"),
        [Candidate("opaque/id", "assento"), Candidate("none", "instituição financeira")],
        "pt-BR",
        "pt-BR",
        ["opaque/id"],
        {"source_url": "private/path", "note": "private note"},
    )


def response():
    return {
        "model": "jev-1.14.0",
        "answers": {
            "meaning": {
                "type": "choice",
                "choice": "c0",
                "confidence": 0.8,
                "probabilities": {"c0": 0.8, "c1": 0.1, "none": 0.05, "insufficient": 0.05},
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 20, "key": "ignored"},
    }


class Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.body[:limit]


def install(monkeypatch, data, capture=None):
    class Opener:
        def open(self, request, timeout):
            if capture is not None:
                capture.append((request, timeout))
            if isinstance(data, Exception):
                raise data
            return Reply(json.dumps(data).encode())

    monkeypatch.setattr("urllib.request.build_opener", lambda *handlers: Opener())


def test_request_minimization_and_opaque_ids(monkeypatch):
    capture = []
    install(monkeypatch, response(), capture)
    result = assist(example(), "secret-key")
    request, timeout = capture[0]
    payload = json.loads(request.data)
    assert request.full_url == ENDPOINTS["typesafe"]
    assert timeout == 30
    assert set(payload["questions"]["meaning"]["criteria"]) == {"c0", "c1", "none", "insufficient"}
    serialized = json.dumps(payload)
    assert "private-id" not in serialized and "private/path" not in serialized
    assert "private note" not in serialized and "gold_ids" not in serialized and "opaque/id" not in serialized
    assert result["selected_ids"] == ["opaque/id"]
    assert result["probabilities"]["none"] == 0.1  # genuine candidate ID distinct from rejection
    assert result["none_probability"] == 0.05
    assert result["model"] == "jev-1.14.0"
    assert "secret-key" not in json.dumps(result)
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 20}


@pytest.mark.parametrize("provider", ["typesafe", "openrouter"])
def test_provider_explicit_and_fingerprint(monkeypatch, provider):
    capture = []
    install(monkeypatch, response(), capture)
    a = assist(example(), "key1", provider, "jev-pinned")
    b = assist(example(), "key2", provider, "jev-pinned")
    assert capture[0][0].full_url == ENDPOINTS[provider]
    assert a["request_fingerprint"] == b["request_fingerprint"]


@pytest.mark.parametrize(
    "change",
    [
        {"choice": "unknown"},
        {"choice": "none"},
        {"type": "score"},
        {"confidence": float("nan")},
        {"probabilities": {"c0": 1}},
        {"probabilities": {"c0": 0.8, "c1": 0.1, "none": 0.1, "insufficient": 0.1}},
        {"probabilities": {"c0": True, "c1": 0, "none": 0, "insufficient": 0}},
    ],
)
def test_malformed_responses_rejected(monkeypatch, change):
    data = copy.deepcopy(response())
    data["answers"]["meaning"].update(change)
    install(monkeypatch, data)
    with pytest.raises(JevError):
        assist(example(), "secret-key")


def test_errors_do_not_expose_key_or_body(monkeypatch):
    error = urllib.error.HTTPError("https://api.typesafe.ai", 401, "secret-key private sentence", {}, None)
    install(monkeypatch, error)
    with pytest.raises(JevError) as exc:
        assist(example(), "secret-key")
    assert "secret-key" not in str(exc.value) and "private sentence" not in str(exc.value)
    with pytest.raises(JevError):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid")
    with pytest.raises(JevError):
        assist(example(), "secret-key", provider="https://evil.invalid")


def test_no_generated_explanation():
    request = build_request(example(), "jev-latest")
    assert request["questions"]["meaning"]["type"] == "choice"
