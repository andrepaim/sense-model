"""Optional loopback-only annotation workspace. No model imports or hosting stack."""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .pipeline import load_config
from .schema import Example

MAX_BODY = 8 * 1024 * 1024
SESSION_TTL = 8 * 3600
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        if address[0] != "127.0.0.1":
            raise ValueError("Annotation server binds only to 127.0.0.1")
        super().__init__(address, Handler)
        self.store = store
        self.sessions = {}
        self.session_lock = threading.RLock()
        self.assist_lock = threading.Lock()
        self.assist_cache = {}

    def session(self, token):
        now = time.monotonic()
        with self.session_lock:
            for key in [k for k, s in self.sessions.items() if now - s["touched"] > SESSION_TTL]:
                expired_csrf = self.sessions[key]["csrf"]
                del self.sessions[key]
                self.assist_cache = {k: v for k, v in self.assist_cache.items() if k[0] != expired_csrf}
            if token not in self.sessions:
                from .pipeline import load_config

                configured = load_config().get("jev", {})
                if len(self.sessions) >= 256:
                    oldest = min(self.sessions, key=lambda k: self.sessions[k]["touched"])
                    expired_csrf = self.sessions.pop(oldest)["csrf"]
                    self.assist_cache = {k: v for k, v in self.assist_cache.items() if k[0] != expired_csrf}
                token = secrets.token_urlsafe(32)
                self.sessions[token] = dict(
                    csrf=secrets.token_urlsafe(32),
                    reviewer="",
                    key=configured.get("api_key") or None,
                    provider=configured.get("provider", "typesafe"),
                    model=configured.get("model", "jev-latest"),
                    persisted=bool(configured.get("api_key")),
                    credential_revision=0,
                    touched=now,
                )
            self.sessions[token]["touched"] = now
            return token, self.sessions[token]


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(35)

    server_version = "SenseReview"

    def log_message(self, *args):
        pass  # No request URLs, source text, headers or credentials in logs.

    def _respond(self, value, status=200, cookie=None, content_type="application/json; charset=utf-8"):
        body = json.dumps(value, ensure_ascii=False).encode() if content_type.startswith("application/json") else value
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        if cookie:
            self.send_header("Set-Cookie", f"sense_session={cookie}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def _guard(self):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            raise PermissionError("Invalid local Host header")
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in hosts}:
            raise PermissionError("Cross-origin requests are not allowed")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("Cross-site requests are not allowed")

    def _session(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            pass
        token = cookie.get("sense_session")
        return self.server.session(token.value if token else None)

    def _body(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("JSON request body required")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid content length") from None
        if size < 1 or size > MAX_BODY:
            raise ValueError("Request body size must be between 1 byte and 8 MiB")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        try:
            self._guard()
            path = urlsplit(self.path).path
            if method == "GET" and path in ASSETS:
                filename, mime = ASSETS[path]
                asset = Path(__file__).parent / "annotation_web" / filename
                return self._respond(asset.read_bytes(), content_type=mime)
            token, session = self._session()
            if method != "GET" and not secrets.compare_digest(self.headers.get("X-CSRF-Token", ""), session["csrf"]):
                raise PermissionError("Session expired or invalid CSRF token; reload the page")
            if method == "GET" and path == "/api/session":
                return self._respond(
                    dict(
                        csrf=session["csrf"],
                        reviewer=session["reviewer"],
                        credential=dict(
                            configured=bool(session["key"]),
                            provider=session["provider"],
                            model=session["model"],
                            persisted=session["persisted"],
                        ),
                    ),
                    cookie=token,
                )
            body = self._body() if method == "POST" else {}
            if path == "/api/session" and method == "POST":
                reviewer = body.get("reviewer", "").strip()
                if not reviewer or len(reviewer) > 80:
                    raise ValueError("Reviewer name must contain 1–80 characters")
                session["reviewer"] = reviewer
                return self._respond({"reviewer": reviewer})
            if path == "/api/key":
                if method == "DELETE":
                    from .pipeline import load_config, save_config

                    config = load_config()
                    if config.setdefault("jev", {}).get("api_key"):
                        config["jev"]["api_key"] = ""
                        save_config(config)
                    session["key"] = None
                    session["persisted"] = False
                    session["credential_revision"] += 1
                    self.server.assist_cache = {
                        k: v for k, v in self.server.assist_cache.items() if k[0] != session["csrf"]
                    }
                    return self._respond({"configured": False})
                if method == "POST":
                    from .jev import PROVIDERS

                    provider = body.get("provider", "typesafe")
                    key = body.get("key", "")
                    model = body.get("model", "").strip()
                    if provider not in PROVIDERS or not isinstance(key, str) or not 1 <= len(key.strip()) <= 4096:
                        raise ValueError("Select a supported provider and enter an API key")
                    if not model or len(model) > 200 or any(ord(c) < 32 for c in model):
                        raise ValueError("Enter a model identifier")
                    if any(ord(c) < 33 or ord(c) > 126 for c in key.strip()):
                        raise ValueError("Invalid API key format")
                    persist = body.get("save_to_config", False)
                    if type(persist) is not bool:
                        raise ValueError("save_to_config must be boolean")
                    if persist:
                        from .pipeline import load_config, save_config

                        config = load_config()
                        config.setdefault("jev", {}).update(api_key=key.strip(), provider=provider, model=model)
                        save_config(config)
                    session.update(key=key.strip(), provider=provider, model=model)
                    session["persisted"] = persist
                    session["credential_revision"] += 1
                    self.server.assist_cache = {
                        k: v for k, v in self.server.assist_cache.items() if k[0] != session["csrf"]
                    }
                    return self._respond(
                        {"configured": True, "provider": provider, "model": model, "persisted": persist}
                    )
            reviewer = session["reviewer"]
            if not reviewer:
                raise PermissionError("Set a reviewer name first")
            store = self.server.store
            query = parse_qs(urlsplit(self.path).query)
            if method == "GET" and path == "/api/examples":
                return self._respond(
                    {
                        "examples": store.list_examples(
                            reviewer,
                            query.get("language", [""])[0],
                            query.get("status", [""])[0],
                            **{
                                key: query.get(key, [""])[0]
                                for key in ("bucket", "difficulty", "area", "paper", "confidence")
                            },
                        )
                    }
                )
            if method == "GET" and path in ("/api/example", "/api/adjudication"):
                return self._respond(
                    store.get_example(query.get("id", [""])[0], reviewer, adjudicate=path == "/api/adjudication")
                )
            if method == "GET" and path == "/api/stats":
                stats = store.stats()
                try:
                    store.assert_disclosure_allowed(reviewer, backup=True)
                    stats["blind_mode"] = False
                except PermissionError:
                    for field in ("buckets", "flags", "initial_agreement", "answers_changed_after_ai"):
                        stats.pop(field, None)
                    stats["blind_mode"] = True
                return self._respond(stats)
            if method == "POST" and path == "/api/import":
                rows = body.get("examples")
                if not isinstance(rows, list) or len(rows) > 5000:
                    raise ValueError("Import a JSONL batch of at most 5000 examples")
                return self._respond(store.import_examples(rows, demo=bool(body.get("demo", False))))
            if method == "POST" and path in ("/api/vote", "/api/adjudicate"):
                action = store.vote if path == "/api/vote" else store.adjudicate
                return self._respond(
                    action(
                        body["id"],
                        reviewer,
                        body["outcome"],
                        body.get("gold_ids", []),
                        note=body.get("note", ""),
                        expected_revision=body["expected_revision"],
                    )
                )
            if method == "POST" and path == "/api/review":
                return self._respond(
                    store.review(
                        body["id"],
                        reviewer,
                        body["action"],
                        body.get("outcome"),
                        body.get("gold_ids"),
                        body.get("note", ""),
                        body["expected_revision"],
                    )
                )
            if method == "POST" and path == "/api/assist":
                return self._respond(self._assist(body, session))
            if method == "POST" and path == "/api/export":
                all_status = body.get("scope") == "all"
                store.assert_disclosure_allowed(
                    reviewer, backup=all_status, include_demo=bool(body.get("include_demo", False))
                )
                export = store.export_all if all_status else store.export_gold
                return self._respond(export(include_demo=bool(body.get("include_demo", False))))
            if method == "POST" and path == "/api/backup":
                store.assert_disclosure_allowed(reviewer, backup=True)
                return self._respond(store.backup())
            self._respond({"error": "Not found"}, 404)
        except PermissionError as exc:
            self._respond({"error": str(exc)}, 403)
        except RuntimeError as exc:
            self._respond({"error": str(exc)}, 409)
        except (ValueError, TypeError, KeyError) as exc:
            # Input errors never include submitted bodies/keys or upstream error bodies.
            message = str(exc) if isinstance(exc, ValueError) else "Invalid request fields"
            self._respond({"error": message}, 400)
        except Exception:
            self._respond({"error": "Annotation request failed; your previous saved work is preserved."}, 500)

    def _assist(self, body, session):
        from .jev import assist

        detail = self.server.store.get_example(body["id"], session["reviewer"])
        if not detail["my_vote"]:
            raise PermissionError("Save an independent first vote before requesting Jev")
        if not session["key"]:
            raise ValueError("Enter a Jev API key in settings first")
        # Immutable examples allow content-safe session-local caching. Different
        # browser sessions never share paid request state or credentials.
        cachekey = (session["csrf"], body["id"], session["provider"], session["model"], session["credential_revision"])
        with self.server.assist_lock:
            result = self.server.assist_cache.get(cachekey)
            cached = result is not None and not body.get("refresh", False)
            if not cached:
                result = assist(
                    Example.from_dict(detail["example"]), session["key"], session["provider"], session["model"]
                )
                if len(self.server.assist_cache) >= 512:
                    self.server.assist_cache.pop(next(iter(self.server.assist_cache)))
                self.server.assist_cache[cachekey] = result
        result = dict(result, cache_hit=cached)
        self.server.store.record_assistance(body["id"], session["reviewer"], result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".annotation/workspace.sqlite"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--import-file", type=Path)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    settings = load_config().get("review", {})
    if settings.get("host", "127.0.0.1") != "127.0.0.1":
        parser.error("The review server must bind to 127.0.0.1")
    args.port = args.port if args.port is not None else settings.get("port", 8765)
    if type(args.port) is not int or not 0 <= args.port <= 65535:
        parser.error("Review port must be between 0 and 65535")
    from .annotation_store import Store

    store = Store(args.workspace)
    if args.import_file:
        rows = [json.loads(line) for line in args.import_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(store.import_examples(rows)))
    with AnnotationServer(("127.0.0.1", args.port), store) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"Sense Model annotation workspace: {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.sessions.clear()
            server.assist_cache.clear()


if __name__ == "__main__":
    main()
