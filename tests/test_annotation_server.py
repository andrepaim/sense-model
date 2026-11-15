"""HTTP workflow, cross-origin protection and credential non-persistence."""

import http.client
import json
import threading

import pytest

from wsd.annotation_server import AnnotationServer
from wsd.annotation_store import Store


def record():
    return dict(
        id="real-1",
        text="The bank is nearby.",
        target=dict(start=4, end=8, text="bank"),
        candidates=[
            dict(id="finance", definition="A financial institution"),
            dict(id="seat", definition="A long seat"),
        ],
        language="en",
        definition_language="en",
        gold_ids=["seat"],
        provenance=dict(
            source="pilot-test",
            source_url="https://example.org/source",
            revision="1",
            license="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            document_id="doc-1",
            domain="general",
            lemma="bank",
            inventory_version="fixture-1",
        ),
    )


@pytest.fixture
def api(tmp_path):
    store = Store(tmp_path / "workspace.sqlite")
    server = AnnotationServer(("127.0.0.1", 0), store)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    class Client:
        csrf = ""
        cookie = ""

        def call(self, method, path, body=None, headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            h = {
                "Cookie": self.cookie,
                "X-CSRF-Token": self.csrf,
                "Content-Type": "application/json",
                **(headers or {}),
            }
            conn.request(method, path, json.dumps(body) if body is not None else None, h)
            response = conn.getresponse()
            raw = response.read()
            cookie = response.getheader("Set-Cookie")
            if cookie:
                self.cookie = cookie.split(";")[0]
            value = json.loads(raw)
            if "csrf" in value:
                self.csrf = value["csrf"]
            conn.close()
            return response.status, value

    client = Client()
    client.call("GET", "/api/session")
    client.call("POST", "/api/session", {"reviewer": "Alice"})
    yield client, server, tmp_path
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def test_origins_host_and_csrf(api):
    client, server, _ = api
    assert client.call("POST", "/api/import", {"examples": []}, headers={"Origin": "https://evil.example"})[0] == 403
    assert client.call("POST", "/api/import", {"examples": []}, headers={"X-CSRF-Token": ""})[0] == 403
    assert client.call("GET", "/api/session", headers={"Host": "evil.example"})[0] == 403
    with pytest.raises(ValueError):
        AnnotationServer(("0.0.0.0", 0), server.store)


def test_blind_vote_assistance_and_key_is_memory_only(api, monkeypatch):
    client, server, path = api
    assert client.call("POST", "/api/import", {"examples": [record()]})[0] == 200
    status, detail = client.call("GET", "/api/example?id=real-1")
    assert status == 200
    assert not detail["example"].get("gold_ids")
    assert client.call("POST", "/api/assist", {"id": "real-1"})[0] == 403
    secret = "test-secret-never-store-12345"
    assert client.call("POST", "/api/key", {"key": secret, "provider": "typesafe", "model": "jev-latest"})[0] == 200
    vote = dict(
        id="real-1", outcome="selected", gold_ids=["finance"], note="independent", expected_revision=detail["revision"]
    )
    assert client.call("POST", "/api/vote", vote)[0] == 200
    calls = []

    def fake(example, key, provider, model):
        calls.append(key)
        return dict(
            provider=provider,
            model="jev-test",
            prompt_version="test",
            request_fingerprint="x",
            choice="finance",
            selected_ids=["finance"],
            probabilities={"finance": 0.9, "seat": 0.02},
            none_probability=0.04,
            insufficient_probability=0.04,
            confidence=0.8,
            usage={"input_tokens": 10},
        )

    monkeypatch.setattr("wsd.jev.assist", fake)
    assert client.call("POST", "/api/assist", {"id": "real-1"})[0] == 200
    assert client.call("POST", "/api/assist", {"id": "real-1"})[1]["cache_hit"] is True
    assert len(calls) == 1
    status, backup = client.call("POST", "/api/backup", {})
    assert status == 200 and secret not in json.dumps(backup)
    for file in path.iterdir():
        assert secret.encode() not in file.read_bytes()
    client.call("POST", "/api/session", {"reviewer": "Bob"})
    status, bob = client.call("GET", "/api/example?id=real-1")
    assert status == 200 and bob["my_vote"] is None and not bob["assistance"]
    assert not bob.get("reviews")
    assert client.call("GET", "/api/adjudication?id=real-1")[0] == 403
    assert client.call("DELETE", "/api/key")[1]["configured"] is False
    assert not next(iter(server.sessions.values()))["key"]


def test_api_errors_do_not_echo_keys(api):
    client, _, _ = api
    key = "should-not-appear"
    status, value = client.call("POST", "/api/key", {"provider": "bad", "model": "jev-latest", "key": key})
    assert status == 400 and key not in json.dumps(value)


def test_label_downloads_and_aggregate_stats_preserve_blindness(api):
    client, server, _ = api
    server.store.import_examples([record()])
    detail = server.store.get_example("real-1", "Other")
    server.store.vote("real-1", "Other", "none_fits", [], expected_revision=detail["revision"])
    assert client.call("POST", "/api/backup", {})[0] == 403
    status, stats = client.call("GET", "/api/stats")
    assert status == 200 and stats["blind_mode"] is True
    assert "initial_agreement" not in stats and "buckets" not in stats
    assert client.call("GET", "/api/example?id=real-1")[1]["my_vote"] is None
