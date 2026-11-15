import json
import os
from pathlib import Path

import pytest

from wsd import pipeline
from wsd.schema import Candidate, Example, Target


def example():
    return Example("x", "bank", Target(0, 4, "bank"), [Candidate("a", "finance")], gold_ids=["a"])


def test_config_roundtrip_private_preserves_unknown(tmp_path, monkeypatch):
    path = tmp_path / "private" / "config.toml"
    monkeypatch.setenv("SENSE_MODEL_CONFIG", str(path))
    cfg = pipeline.load_config()
    cfg["jev"]["api_key"] = 'secret-with-"quotes"\\and\nnewlines'
    cfg["generation"]["areas"] = ["economics", "computing"]
    cfg["extra"] = {"enabled": True, "nested": {"n": 3}}
    assert pipeline.save_config(cfg) == path
    assert pipeline.load_config() == cfg
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    pipeline.main(["init", "--config", str(path)])
    assert pipeline.load_config() == cfg


def test_config_errors_do_not_disclose_secret(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[jev]\napi_key = unquoted-secret")
    with pytest.raises(ValueError) as error:
        pipeline.load_config(path)
    assert "unquoted-secret" not in str(error.value)
    path.write_text('jev = "bad"')
    with pytest.raises(ValueError, match="table"):
        pipeline.load_config(path)


def test_event_log_redacts_registered_key_and_sensitive_fields(tmp_path, capsys):
    cfg = pipeline.load_config(tmp_path / "missing")
    cfg["jev"]["api_key"] = "do-not-leak-12345"
    pipeline.save_config(cfg, tmp_path / "config.toml")
    log = pipeline.EventLog(tmp_path / "events.jsonl")
    log.emit("failed", detail="URL includes do-not-leak-12345", api_key="some-other-secret", count=3)
    raw = log.path.read_text()
    assert "do-not-leak-12345" not in raw + capsys.readouterr().err
    assert "some-other-secret" not in raw
    assert json.loads(raw)["count"] == 3


def test_atomic_artifacts_and_fingerprint(tmp_path):
    pipeline.atomic_json(tmp_path / "x.json", {"a": "ação"})
    pipeline.atomic_jsonl(tmp_path / "x.jsonl", [{"a": 1}, {"a": 2}])
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": "ação"}
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 2
    assert pipeline.fingerprint({"a": 1, "b": 2}) == pipeline.fingerprint({"b": 2, "a": 1})
    assert not list(tmp_path.glob(".*"))


def test_automatic_labels_require_matching_human_review():
    ex = example()
    ex.provenance = {"automatic": {"label_tier": "silver"}, "review_status": "unreviewed"}
    with pytest.raises(ValueError, match="silver"):
        pipeline.require_reviewed([ex])
    pipeline.require_reviewed([ex], allow_silver=True)
    ex.provenance["review"] = {"status": "approved", "gold_ids": ["a"]}
    pipeline.require_reviewed([ex])
    ex.provenance["review"]["gold_ids"] = []
    with pytest.raises(ValueError, match="do not match"):
        pipeline.require_reviewed([ex])
    ex.provenance["review"]["status"] = "rejected"
    with pytest.raises(ValueError, match="rejected"):
        pipeline.require_reviewed([ex], allow_silver=True)
    pipeline.require_reviewed([example()])


def test_codex_command_isolated_and_json_output(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-child")
    monkeypatch.setenv("CUSTOM_CREDENTIAL", "private-custom-key-value")
    pipeline.save_config({"jev": {"api_key_env": "CUSTOM_CREDENTIAL"}}, tmp_path / "config.toml")
    pipeline.load_config(tmp_path / "config.toml")

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            seen.update(args=args, kwargs=kwargs)
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text('{"answer": "a"}')

        def communicate(self, text, timeout):
            seen["prompt"] = text

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    result = pipeline.run_codex("untrusted passage", {"type": "object"}, model="test-model")
    assert result == {"answer": "a"}
    assert "--ignore-user-config" in seen["args"]
    assert "--ignore-rules" in seen["args"]
    assert "--ephemeral" in seen["args"]
    assert "read-only" in seen["args"]
    assert "TYPESAFE_API_KEY" not in seen["kwargs"]["env"]
    assert "CUSTOM_CREDENTIAL" not in seen["kwargs"]["env"]
    assert "untrusted passage" not in " ".join(seen["args"])
    assert not Path(seen["args"][seen["args"].index("--cd") + 1]).exists()


def test_codex_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/usr/bin/codex")

    class Process:
        returncode = 2

        def __init__(self, args, **kwargs):
            kwargs["stderr"].write(b"secret-prompt secret-token")

        def communicate(self, text, timeout):
            pass

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    with pytest.raises(pipeline.CodexError, match="No fallback") as error:
        pipeline.run_codex("private", {}, model="unavailable-model")
    assert "secret" not in str(error.value)
    with pytest.raises(ValueError, match="model"):
        pipeline.run_codex("private", {}, model="--bad-flag")


def test_codex_interrupt_terminates_child_and_propagates(monkeypatch):
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/usr/bin/codex")
    killed = []

    class Process:
        pid = 123456

        def __init__(self, args, **kwargs):
            pass

        def communicate(self, text, timeout):
            raise KeyboardInterrupt()

        def kill(self):
            killed.append(self.pid)

        def wait(self):
            pass

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    monkeypatch.setattr(pipeline.os, "killpg", lambda pid, sig: killed.append(pid))
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_codex("private", {})
    assert killed == [123456]


def test_console_cli_returns_exit_code_not_report(monkeypatch):
    from types import SimpleNamespace

    from wsd import cli

    monkeypatch.setattr(
        cli.importlib, "import_module", lambda name: SimpleNamespace(main=lambda argv: {"complete": True})
    )
    assert cli.main(["generate", "--samples-per-paper", "1"]) == 0
    assert cli.main(["label"]) == 0
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(main=lambda argv: 2))
    assert cli.main(["evaluate"]) == 2
