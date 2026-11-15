"""Local annotation persistence, blind votes and explicit gold adjudication.

Reviewer names are workflow identities on a trusted workstation, not login accounts.
No credentials belong in this store. Each operation opens its own connection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sqlite3
import statistics
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .schema import Example

OUTCOMES = {"selected", "none_fits", "insufficient_context", "extraction_issue", "inventory_issue", "skip"}
FLAGS = OUTCOMES - {"selected", "none_fits"}
PROTOCOL = "pilot-blind-first-v1"
REAL_PROVENANCE = ("source", "document_id", "source_url", "revision", "license", "license_url")
ASSISTANCE_FIELDS = {
    "provider",
    "requested_model",
    "cache_hit",
    "model",
    "prompt_version",
    "request_fingerprint",
    "choice",
    "selected_ids",
    "probabilities",
    "none_probability",
    "insufficient_probability",
    "confidence",
    "usage",
    "created_at",
}


def _now():
    return datetime.now(UTC).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _reviewer(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
        raise ValueError("Choose a reviewer name of 1–100 characters")
    return value.strip()


def _signature(vote):
    return vote["outcome"], tuple(sorted(vote["gold_ids"]))


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported annotation workspace schema version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS examples (
                  id TEXT PRIMARY KEY, data TEXT NOT NULL, fingerprint TEXT NOT NULL,
                  demo INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS votes (
                  example_id TEXT NOT NULL REFERENCES examples(id), reviewer TEXT NOT NULL,
                  initial TEXT NOT NULL, current TEXT NOT NULL, ai_exposed INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY(example_id,reviewer));
                CREATE TABLE IF NOT EXISTS assistance (
                  id INTEGER PRIMARY KEY, example_id TEXT NOT NULL REFERENCES examples(id),
                  reviewer TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS adjudications (
                  example_id TEXT PRIMARY KEY REFERENCES examples(id), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS presentations (
                  example_id TEXT NOT NULL REFERENCES examples(id), reviewer TEXT NOT NULL,
                  opened_at TEXT NOT NULL, PRIMARY KEY(example_id,reviewer));
                CREATE TABLE IF NOT EXISTS human_reviews (
                  example_id TEXT PRIMARY KEY REFERENCES examples(id), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY, example_id TEXT REFERENCES examples(id),
                  reviewer TEXT, action TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _event(db, eid, reviewer, action, data):
        db.execute(
            "INSERT INTO events(example_id,reviewer,action,data,created_at) VALUES(?,?,?,?,?)",
            (eid, reviewer, action, _json(data), _now()),
        )

    @staticmethod
    def _row(db, eid):
        row = db.execute("SELECT * FROM examples WHERE id=?", (eid,)).fetchone()
        if row is None:
            raise ValueError("Unknown example")
        return row

    @staticmethod
    def _check_revision(row, revision):
        if type(revision) is not int or row["revision"] != revision:
            raise RuntimeError("Example changed in another tab; reload before saving")

    @staticmethod
    def _validate_vote(example, outcome, gold_ids, note):
        if outcome not in OUTCOMES:
            raise ValueError("Unknown annotation outcome")
        if not isinstance(gold_ids, list) or any(not isinstance(x, str) for x in gold_ids):
            raise ValueError("gold_ids must be a list of candidate IDs")
        if len(gold_ids) != len(set(gold_ids)):
            raise ValueError("Duplicate gold IDs")
        candidates = {c["id"] for c in example["candidates"]}
        if any(x not in candidates for x in gold_ids):
            raise ValueError("Gold IDs must be supplied candidates")
        if outcome == "selected" and not gold_ids:
            raise ValueError("Select at least one definition")
        if outcome != "selected" and gold_ids:
            raise ValueError("Only selected outcome can contain gold IDs")
        if not isinstance(note, str) or len(note) > 10000:
            raise ValueError("Note must be text under 10000 characters")
        return {"outcome": outcome, "gold_ids": list(gold_ids), "note": note.strip(), "created_at": _now()}

    def import_examples(self, rows, demo=False):
        if not isinstance(rows, list) or not rows or len(rows) > 10000:
            raise ValueError("Import 1–10000 examples")
        if type(demo) is not bool:
            raise ValueError("demo must be boolean")
        prepared = []
        for source in rows:
            if not isinstance(source, dict):
                raise ValueError("Each example must be an object")
            source = copy.deepcopy(source)
            proposed = source.pop("gold_ids", None)
            # Known gold/AI judgments must not enter blind source provenance.
            p = source.get("provenance", {})
            if not isinstance(p, dict):
                raise ValueError("Provenance must be an object")
            if isinstance(p.get("automatic"), dict):
                # A proposal is evidence, never human-approved training gold.
                if proposed is not None:
                    candidate_ids = {c.get("id") for c in source.get("candidates", [])}
                    if not isinstance(proposed, list) or any(i not in candidate_ids for i in proposed):
                        raise ValueError("Proposed gold IDs must be supplied candidates")
                    p["proposed_gold_ids"] = proposed
                if "review" in p:
                    p["previous_review"] = p.pop("review")
                p["review_status"] = "unreviewed"
                p["label_tier"] = "silver"
            p = {
                k: v
                for k, v in p.items()
                if k
                not in {"gold_ids", "gold", "answer", "expected", "selected_id", "adjudication", "reviews", "ai_result"}
            }
            if demo:
                p.update(annotation="synthetic", demo=True, review_status="demo")
            else:
                missing = [k for k in REAL_PROVENANCE if not isinstance(p.get(k), str) or not p[k].strip()]
                if missing and not isinstance(p.get("automatic"), dict):
                    raise ValueError("Real examples need provenance: " + ", ".join(missing))
                if p.get("demo") or p.get("annotation") == "synthetic":
                    raise ValueError("Synthetic/demo examples must use demo import")
                p["demo"] = False
            source["provenance"] = p
            try:
                ex = Example.from_dict(source)
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError("Invalid example shape") from exc
            data = ex.to_dict()
            data.pop("gold_ids", None)
            fingerprint = hashlib.sha256(_json(data).encode()).hexdigest()
            prepared.append((ex.id, data, fingerprint))
        added = existing = 0
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            for eid, data, fingerprint in prepared:
                old = db.execute("SELECT fingerprint,demo FROM examples WHERE id=?", (eid,)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint or old["demo"] != int(demo):
                        raise ValueError("Changed example ID rejected: " + eid)
                    existing += 1
                    continue
                db.execute(
                    "INSERT INTO examples(id,data,fingerprint,demo,created_at) VALUES(?,?,?,?,?)",
                    (eid, _json(data), fingerprint, int(demo), _now()),
                )
                self._event(db, eid, None, "import", {"fingerprint": fingerprint, "demo": demo})
                added += 1
        return {"imported": added, "existing": existing}

    def list_examples(self, reviewer, language="", status="", **filters):
        reviewer = _reviewer(reviewer)
        result = []
        with self._db() as db:
            for row in db.execute("SELECT * FROM examples ORDER BY id"):
                ex = json.loads(row["data"])
                if (
                    language
                    and ex["language"] != language
                    and not (language in {"pt", "en"} and ex["language"].startswith(language + "-"))
                ):
                    continue
                vote = db.execute(
                    "SELECT current FROM votes WHERE example_id=? AND reviewer=?", (row["id"], reviewer)
                ).fetchone()
                mine = json.loads(vote["current"]) if vote else None
                automatic = ex["provenance"].get("automatic")
                review_row = db.execute("SELECT data FROM human_reviews WHERE example_id=?", (row["id"],)).fetchone()
                review = json.loads(review_row["data"]) if review_row else None
                state = "reviewed" if mine else "pending"
                if mine:
                    initials = [
                        json.loads(v["initial"])
                        for v in db.execute("SELECT initial FROM votes WHERE example_id=?", (row["id"],))
                    ]
                    if len({_signature(v) for v in initials}) > 1:
                        state = "disagreement"
                    if db.execute("SELECT 1 FROM adjudications WHERE example_id=?", (row["id"],)).fetchone():
                        state = "adjudicated"
                if automatic is not None:
                    state = "reviewed" if review else "unreviewed"
                    mine = review
                if status == "pending" and automatic is not None:
                    status_match = not review
                else:
                    status_match = (
                        not status
                        or (status == "reviewed" and mine)
                        or status in (state, mine["outcome"] if mine else "")
                    )
                if not status_match:
                    continue
                p = ex["provenance"]
                generation = p.get("generation", {})
                confidence = (automatic or {}).get("jev", {}) or {}
                confidence = confidence.get("confidence")
                meta = dict(
                    bucket=(automatic or {}).get("bucket", ""),
                    difficulty=generation.get("difficulty", p.get("difficulty", "")),
                    area=generation.get("area", p.get("area", p.get("domain", ""))),
                    paper=generation.get("paper_id", p.get("paper_id", p.get("document_id", ""))),
                )
                if any(
                    value and str(meta.get(key, "")) != value for key, value in filters.items() if key != "confidence"
                ):
                    continue
                confidence_filter = filters.get("confidence", "")
                if confidence_filter == "unknown" and confidence is not None:
                    continue
                if confidence_filter in {"high", "low"} and (
                    confidence is None
                    or (confidence >= (automatic or {}).get("threshold", 0.8)) != (confidence_filter == "high")
                ):
                    continue
                result.append(
                    {
                        "id": row["id"],
                        "text": ex["text"],
                        "target": ex["target"],
                        "language": ex["language"],
                        "domain": ex["provenance"].get("domain", "unspecified"),
                        "status": state,
                        "outcome": mine["outcome"] if mine else None,
                        "demo": bool(row["demo"]),
                        "revision": row["revision"],
                        "automatic": automatic,
                        "review": review,
                        "confidence": confidence,
                        **meta,
                    }
                )

        def priority(item):
            if item["status"] in {"reviewed", "adjudicated"}:
                return 3, item["id"]
            return {"unresolved": 0, "jev_low_codex_disagreed": 1}.get(item["bucket"], 2), item["id"]

        return sorted(result, key=priority)

    def get_example(self, id, reviewer, adjudicate=False):
        reviewer = _reviewer(reviewer)
        with self._db() as db:
            row = self._row(db, id)
            ex = json.loads(row["data"])
            db.execute(
                "INSERT OR IGNORE INTO presentations(example_id,reviewer,opened_at) VALUES(?,?,?)",
                (id, reviewer, _now()),
            )
            vote = db.execute("SELECT * FROM votes WHERE example_id=? AND reviewer=?", (id, reviewer)).fetchone()
            my_vote = None
            if vote:
                my_vote = {
                    **json.loads(vote["current"]),
                    "initial": json.loads(vote["initial"]),
                    "ai_exposed": bool(vote["ai_exposed"]),
                }
            order = [c["id"] for c in ex["candidates"]]
            random.Random(hashlib.sha256((id + "\0" + reviewer).encode()).digest()).shuffle(order)
            detail = {
                "example": ex,
                "revision": row["revision"],
                "my_vote": my_vote,
                "assistance": [],
                "candidate_order": order,
                "demo": bool(row["demo"]),
            }
            if isinstance(ex["provenance"].get("automatic"), dict):
                review = db.execute("SELECT data FROM human_reviews WHERE example_id=?", (id,)).fetchone()
                detail["mode"] = "automatic"
                detail["review"] = json.loads(review["data"]) if review else None
                detail["automatic"] = ex["provenance"]["automatic"]
            if vote:
                detail["assistance"] = [
                    json.loads(a["result"])
                    for a in db.execute(
                        "SELECT result FROM assistance WHERE example_id=? AND reviewer=? ORDER BY id", (id, reviewer)
                    )
                ]
            if adjudicate:
                votes = db.execute("SELECT * FROM votes WHERE example_id=? ORDER BY reviewer", (id,)).fetchall()
                if not vote or len(votes) < 2:
                    raise PermissionError("Adjudication requires your independent vote and two distinct reviewers")
                detail["reviews"] = [
                    {
                        "reviewer": v["reviewer"],
                        **json.loads(v["current"]),
                        "initial": json.loads(v["initial"]),
                        "ai_exposed": bool(v["ai_exposed"]),
                    }
                    for v in votes
                ]
                final = db.execute("SELECT data FROM adjudications WHERE example_id=?", (id,)).fetchone()
                detail["adjudication"] = json.loads(final["data"]) if final else None
            return detail

    def vote(self, id, reviewer, outcome, gold_ids, note="", expected_revision=0):
        reviewer = _reviewer(reviewer)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, id)
            self._check_revision(row, expected_revision)
            if isinstance(json.loads(row["data"])["provenance"].get("automatic"), dict):
                raise ValueError("Use automatic review to approve, correct, or reject this example")
            value = self._validate_vote(json.loads(row["data"]), outcome, gold_ids, note)
            if outcome == "skip":
                # Navigation event only: no initial judgment and no AI unlock.
                self._event(db, id, reviewer, "skip", value)
                db.execute("UPDATE examples SET revision=revision+1 WHERE id=?", (id,))
            else:
                presentation = db.execute(
                    "SELECT opened_at FROM presentations WHERE example_id=? AND reviewer=?", (id, reviewer)
                ).fetchone()
                if presentation:
                    value["elapsed_seconds"] = max(
                        0.0,
                        (
                            datetime.fromisoformat(value["created_at"])
                            - datetime.fromisoformat(presentation["opened_at"])
                        ).total_seconds(),
                    )
                old = db.execute("SELECT * FROM votes WHERE example_id=? AND reviewer=?", (id, reviewer)).fetchone()
                if old:
                    db.execute(
                        "UPDATE votes SET current=? WHERE example_id=? AND reviewer=?", (_json(value), id, reviewer)
                    )
                else:
                    db.execute(
                        "INSERT INTO votes(example_id,reviewer,initial,current) VALUES(?,?,?,?)",
                        (id, reviewer, _json(value), _json(value)),
                    )
                db.execute("UPDATE examples SET revision=revision+1 WHERE id=?", (id,))
                # Any review revision requires explicit adjudication again.
                db.execute("DELETE FROM adjudications WHERE example_id=?", (id,))
                self._event(db, id, reviewer, "revise_vote" if old else "initial_vote", value)
        return self.get_example(id, reviewer)

    def review(self, id, reviewer, action, outcome=None, gold_ids=None, note="", expected_revision=0):
        """Audit a frozen automatic proposal; the evidence never changes."""
        reviewer = _reviewer(reviewer)
        if action not in {"approve", "correct", "reject"}:
            raise ValueError("Choose approve, correct, or reject")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, id)
            self._check_revision(row, expected_revision)
            ex = json.loads(row["data"])
            automatic = ex["provenance"].get("automatic")
            if not isinstance(automatic, dict):
                raise ValueError("This example uses the independent review workflow")
            if action == "approve":
                if automatic.get("status") != "labeled":
                    raise ValueError("Resolve the automatic result before approving")
                outcome, gold_ids = self._proposal(ex)
                if outcome not in {"selected", "none_fits"}:
                    raise ValueError("This automatic result needs a correction")
            if action == "reject":
                outcome, gold_ids = "insufficient_context", []
            value = self._validate_vote(ex, outcome, gold_ids or [], note)
            if action == "correct" and outcome not in {"selected", "none_fits", "insufficient_context"}:
                raise ValueError("Choose selected, none_fits, or insufficient_context")
            value.update(
                status={"approve": "approved", "correct": "corrected", "reject": "rejected"}[action],
                reviewer=reviewer,
                protocol="automatic-first-v1",
            )
            if outcome == "insufficient_context" and action != "reject":
                value["status"] = "insufficient_context"
            db.execute("INSERT OR REPLACE INTO human_reviews(example_id,data) VALUES(?,?)", (id, _json(value)))
            db.execute("UPDATE examples SET revision=revision+1 WHERE id=?", (id,))
            self._event(db, id, reviewer, "human_review", value)
        return self.get_example(id, reviewer)

    @staticmethod
    def _proposal(ex):
        p = ex["provenance"]
        automatic = p.get("automatic", {})
        result = automatic.get(automatic.get("selected_by")) or {}
        ids = result.get("selected_ids", p.get("proposed_gold_ids", []))
        choice = result.get("outcome", result.get("choice"))
        outcome = (
            "selected" if ids else {"none": "none_fits", "insufficient": "insufficient_context"}.get(choice, choice)
        )
        return outcome, ids

    def record_assistance(self, id, reviewer, result):
        reviewer = _reviewer(reviewer)
        if not isinstance(result, dict) or set(result) - ASSISTANCE_FIELDS:
            raise ValueError("Unexpected assistance fields; credentials and raw provider responses cannot be stored")
        # Validate serializability, bounded result and never accept credential-like fields.
        if "cache_hit" in result and type(result["cache_hit"]) is not bool:
            raise ValueError("cache_hit must be boolean")
        if "requested_model" in result and (
            not isinstance(result["requested_model"], str) or len(result["requested_model"]) > 200
        ):
            raise ValueError("Invalid requested model metadata")
        encoded = _json(result)
        if len(encoded.encode()) > 200000:
            raise ValueError("Assistance result too large")
        usage = result.get("usage", {})
        if not isinstance(usage, dict) or any(
            k not in {"input_tokens", "output_tokens", "total_tokens", "cost"}
            or type(v) not in (int, float)
            or not math.isfinite(v)
            or v < 0
            for k, v in usage.items()
        ):
            raise ValueError("Invalid usage metadata")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            ex = json.loads(self._row(db, id)["data"])
            candidates = {c["id"] for c in ex["candidates"]}
            probs = result.get("probabilities")
            if not isinstance(probs, dict) or set(probs) != candidates:
                raise ValueError("Assistance must score exactly the supplied candidates")
            values = [*probs.values(), result.get("none_probability"), result.get("insufficient_probability")]
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Invalid assistance probabilities")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-4):
                raise ValueError("Assistance probabilities must sum to one")
            if not isinstance(result.get("selected_ids"), list) or any(
                i not in candidates for i in result["selected_ids"]
            ):
                raise ValueError("Invalid assistance selected IDs")
            vote = db.execute("SELECT * FROM votes WHERE example_id=? AND reviewer=?", (id, reviewer)).fetchone()
            if not vote:
                raise PermissionError("Record your independent first vote before requesting assistance")
            db.execute(
                "INSERT INTO assistance(example_id,reviewer,result,created_at) VALUES(?,?,?,?)",
                (id, reviewer, encoded, _now()),
            )
            db.execute("UPDATE votes SET ai_exposed=1 WHERE example_id=? AND reviewer=?", (id, reviewer))
            db.execute("UPDATE examples SET revision=revision+1 WHERE id=?", (id,))
            db.execute("DELETE FROM adjudications WHERE example_id=?", (id,))
            self._event(
                db,
                id,
                reviewer,
                "ai_exposure",
                {
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                    "request_fingerprint": result.get("request_fingerprint"),
                },
            )
        return self.get_example(id, reviewer)

    def adjudicate(self, id, reviewer, outcome, gold_ids, note="", expected_revision=0):
        reviewer = _reviewer(reviewer)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, id)
            self._check_revision(row, expected_revision)
            if outcome not in ("selected", "none_fits"):
                raise ValueError("Only selected or none_fits can be adjudicated as gold")
            if isinstance(json.loads(row["data"])["provenance"].get("automatic"), dict):
                raise ValueError("Use automatic review to approve, correct, or reject this example")
            value = self._validate_vote(json.loads(row["data"]), outcome, gold_ids, note)
            votes = db.execute("SELECT * FROM votes WHERE example_id=?", (id,)).fetchall()
            if len(votes) < 2 or not any(v["reviewer"] == reviewer for v in votes):
                raise PermissionError("Adjudication requires your independent vote and two distinct reviewers")
            initials = [json.loads(v["initial"]) for v in votes]
            currents = [json.loads(v["current"]) for v in votes]
            needs_note = (
                len({_signature(v) for v in initials}) != 1
                or len(gold_ids) > 1
                or any(v["outcome"] in FLAGS or _signature(v) != _signature(value) for v in currents)
            )
            if needs_note and not note.strip():
                raise ValueError("Explain disagreement, multiple senses, flag resolution or consensus override")
            value.update(reviewer=reviewer, review_count=len(votes), protocol=PROTOCOL)
            db.execute("INSERT OR REPLACE INTO adjudications(example_id,data) VALUES(?,?)", (id, _json(value)))
            db.execute("UPDATE examples SET revision=revision+1 WHERE id=?", (id,))
            self._event(db, id, reviewer, "adjudicate", value)
        return self.get_example(id, reviewer, adjudicate=True)

    def export_gold(self, include_demo=False):
        if type(include_demo) is not bool:
            raise ValueError("include_demo must be boolean")
        examples = []
        with self._db() as db:
            for row in db.execute(
                "SELECT e.*,h.data AS review FROM examples e JOIN human_reviews h ON e.id=h.example_id ORDER BY e.id"
            ):
                if row["demo"] and not include_demo:
                    continue
                review = json.loads(row["review"])
                if review["status"] not in {"approved", "corrected"} or review["outcome"] not in {
                    "selected",
                    "none_fits",
                }:
                    continue
                ex = json.loads(row["data"])
                ex["gold_ids"] = review["gold_ids"]
                ex["provenance"].update(
                    annotation="human",
                    review_status=review["status"],
                    review=review,
                    annotation_protocol="automatic-first-v1",
                    label_tier="gold",
                    review_count=1,
                    ai_assisted=True,
                )
                Example.from_dict(ex)
                examples.append(ex)
            for row in db.execute(
                "SELECT e.*,a.data AS final FROM examples e JOIN adjudications a ON e.id=a.example_id ORDER BY e.id"
            ):
                if row["demo"] and not include_demo:
                    continue
                ex = json.loads(row["data"])
                if isinstance(ex["provenance"].get("automatic"), dict):
                    continue
                final = json.loads(row["final"])
                votes = db.execute("SELECT * FROM votes WHERE example_id=?", (row["id"],)).fetchall()
                if len(votes) < 2:
                    continue
                ex["gold_ids"] = final["gold_ids"]
                ex["provenance"].update(
                    annotation="human",
                    review_status="adjudicated",
                    annotation_protocol=PROTOCOL,
                    review_count=len(votes),
                    ai_assisted=any(v["ai_exposed"] for v in votes),
                    pilot=True,
                    demo=bool(row["demo"]),
                )
                if row["demo"]:
                    ex["provenance"]["annotation"] = "synthetic"
                Example.from_dict(ex)
                examples.append(ex)
        content = "".join(_json(e) + "\n" for e in examples)
        protocols = {e["provenance"].get("annotation_protocol", PROTOCOL) for e in examples}
        return {
            "examples": examples,
            "jsonl": content,
            "manifest": {
                "format_version": 1,
                "protocol": next(iter(protocols))
                if len(protocols) == 1
                else ("mixed" if protocols else "automatic-first-v1"),
                "created_at": _now(),
                "count": len(examples),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "languages": dict(Counter(e["language"] for e in examples)),
                "include_demo": include_demo,
                "scope": "Human-approved labels; automatic evidence preserved"
                if "automatic-first-v1" in protocols
                else "Reviewed development pilot; not frozen test or training data",
                "workspace": self.stats(),
            },
        }

    def export_all(self, include_demo=False):
        """Schema-compatible audit export, with proposals separate from human gold."""
        examples = []
        with self._db() as db:
            for row in db.execute("SELECT * FROM examples ORDER BY id"):
                if row["demo"] and not include_demo:
                    continue
                ex = json.loads(row["data"])
                ex["gold_ids"] = None
                review = db.execute("SELECT data FROM human_reviews WHERE example_id=?", (row["id"],)).fetchone()
                if review:
                    value = json.loads(review["data"])
                    ex["provenance"].update(review=value, review_status=value["status"])
                    if value["status"] in {"approved", "corrected"} and value["outcome"] in {"selected", "none_fits"}:
                        ex["gold_ids"] = value["gold_ids"]
                        ex["provenance"].update(annotation="human", label_tier="gold")
                else:
                    ex["provenance"]["review_status"] = "unreviewed"
                    if not isinstance(ex["provenance"].get("automatic"), dict):
                        final = db.execute("SELECT data FROM adjudications WHERE example_id=?", (row["id"],)).fetchone()
                        if final:
                            ex["gold_ids"] = json.loads(final["data"])["gold_ids"]
                            ex["provenance"].update(
                                annotation="human", review_status="adjudicated", annotation_protocol=PROTOCOL
                            )
                        elif db.execute("SELECT 1 FROM votes WHERE example_id=?", (row["id"],)).fetchone():
                            ex["provenance"]["review_status"] = "pending_adjudication"
                examples.append(ex)
        content = "".join(_json(ex) + "\n" for ex in examples)
        return {
            "examples": examples,
            "jsonl": content,
            "manifest": {
                "format_version": 1,
                "protocol": "automatic-first-v1",
                "scope": "all-with-status",
                "count": len(examples),
                "created_at": _now(),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            },
        }

    def assert_disclosure_allowed(self, reviewer, backup=False, include_demo=False):
        """Avoid accidental unblinding via gold or raw backup downloads.

        Local reviewer names are not authentication. The workspace owner can
        always inspect SQLite directly; this guard protects the review workflow.
        """
        reviewer = _reviewer(reviewer)
        with self._db() as db:
            if backup:
                rows = db.execute("SELECT DISTINCT example_id FROM votes").fetchall()
            else:
                rows = db.execute(
                    "SELECT a.example_id FROM adjudications a JOIN examples e ON a.example_id=e.id WHERE e.demo=0 OR ?",
                    (int(include_demo),),
                ).fetchall()
            for row in rows:
                ex = json.loads(self._row(db, row["example_id"])["data"])
                if isinstance(ex["provenance"].get("automatic"), dict):
                    continue
                own = db.execute(
                    "SELECT 1 FROM votes WHERE example_id=? AND reviewer=?", (row["example_id"], reviewer)
                ).fetchone()
                if not own:
                    raise PermissionError(
                        "Complete your independent votes before downloading other reviewers’ annotations"
                    )
        return True

    def backup(self):
        with self._db() as db:
            return {
                "format_version": 1,
                "schema_version": 1,
                "created_at": _now(),
                "examples": [dict(row) for row in db.execute("SELECT * FROM examples ORDER BY id")],
                "votes": [dict(row) for row in db.execute("SELECT * FROM votes ORDER BY example_id,reviewer")],
                "assistance": [dict(row) for row in db.execute("SELECT * FROM assistance ORDER BY id")],
                "adjudications": [dict(row) for row in db.execute("SELECT * FROM adjudications ORDER BY example_id")],
                "human_reviews": [dict(row) for row in db.execute("SELECT * FROM human_reviews ORDER BY example_id")],
                "presentations": [
                    dict(row) for row in db.execute("SELECT * FROM presentations ORDER BY example_id,reviewer")
                ],
                "events": [dict(row) for row in db.execute("SELECT * FROM events ORDER BY id")],
            }

    def stats(self):
        by_language, buckets, flags = {}, Counter(), Counter()
        agreements = comparisons = changes = exposures = 0
        durations = []
        with self._db() as db:
            total = db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
            for row in db.execute("SELECT * FROM examples ORDER BY id"):
                ex = json.loads(row["data"])
                lang = ex["language"]
                counts = by_language.setdefault(
                    lang,
                    {
                        "total": 0,
                        "real": 0,
                        "demo": 0,
                        "reviewed": 0,
                        "double_reviewed": 0,
                        "adjudicated": 0,
                        "accepted": 0,
                        "target": 100,
                        "documents": set(),
                        "lemmas": set(),
                    },
                )
                counts["total"] += 1
                counts["demo" if row["demo"] else "real"] += 1
                votes = db.execute("SELECT * FROM votes WHERE example_id=?", (row["id"],)).fetchall()
                counts["reviewed"] += bool(votes)
                counts["double_reviewed"] += len(votes) >= 2
                if len(votes) >= 2:
                    comparisons += 1
                    agreements += len({_signature(json.loads(v["initial"])) for v in votes}) == 1
                for vote in votes:
                    initial, current = json.loads(vote["initial"]), json.loads(vote["current"])
                    if "elapsed_seconds" in initial:
                        durations.append(initial["elapsed_seconds"])
                    if current["outcome"] in FLAGS:
                        flags[current["outcome"]] += 1
                    exposures += vote["ai_exposed"]
                    changes += bool(vote["ai_exposed"] and _signature(initial) != _signature(current))
                final = db.execute("SELECT data FROM adjudications WHERE example_id=?", (row["id"],)).fetchone()
                if final:
                    counts["adjudicated"] += 1
                    if not row["demo"]:
                        counts["accepted"] += 1
                        p = ex["provenance"]
                        counts["documents"].add((p["source"], p["document_id"]))
                        counts["lemmas"].add(p.get("lemma", ex["target"]["text"].casefold()))
                        final_data = json.loads(final["data"])
                        bucket = (
                            "none_fits"
                            if final_data["outcome"] == "none_fits"
                            else ("single" if len(ex["candidates"]) == 1 else p.get("domain", "general"))
                        )
                        buckets[lang + ":" + bucket] += 1
            usage = Counter()
            request_count = 0
            for row in db.execute("SELECT result FROM assistance"):
                result = json.loads(row["result"])
                if result.get("cache_hit", False):
                    continue
                request_count += 1
                for key, value in result.get("usage", {}).items():
                    usage[key] += value
            for counts in by_language.values():
                counts["documents"] = len(counts["documents"])
                counts["lemmas"] = len(counts["lemmas"])
            return {
                "automatic_review": self.automatic_stats(),
                "total": total,
                "by_language": by_language,
                "buckets": dict(buckets),
                "flags": dict(flags),
                "initial_agreement": agreements / comparisons if comparisons else None,
                "double_reviewed_examples": comparisons,
                "ai_exposed_reviews": exposures,
                "answers_changed_after_ai": changes,
                "ai_requests": request_count,
                "usage": dict(usage),
                "protocol": PROTOCOL,
                "target_per_language": 100,
                "initial_review_seconds": {
                    "count": len(durations),
                    "median": statistics.median(durations) if durations else None,
                    "note": "Elapsed first-view to initial vote, including breaks; not active working time.",
                },
            }

    def automatic_stats(self):
        buckets, accuracy = Counter(), {}
        reviewed = accepted = total = 0
        with self._db() as db:
            for row in db.execute(
                "SELECT e.*, h.data AS review FROM examples e LEFT JOIN human_reviews h ON e.id=h.example_id"
            ):
                ex = json.loads(row["data"])
                automatic = ex["provenance"].get("automatic")
                if not isinstance(automatic, dict) or row["demo"]:
                    continue
                total += 1
                buckets[automatic.get("bucket", "unresolved")] += 1
                if not row["review"]:
                    continue
                review = json.loads(row["review"])
                reviewed += 1
                accepted += review["status"] in {"approved", "corrected"} and review["outcome"] in {
                    "selected",
                    "none_fits",
                }
                confidence = (automatic.get("jev") or {}).get("confidence")
                band = (
                    "unknown"
                    if confidence is None
                    else f"{min(9, int(confidence * 10)) * 10}–{min(10, int(confidence * 10) + 1) * 10}%"
                )
                counts = accuracy.setdefault(band, {"reviewed": 0, "comparable": 0, "correct": 0, "accuracy": None})
                counts["reviewed"] += 1
                jev = automatic.get("jev") or {}
                jev_ids = jev.get("selected_ids", [])
                jev_outcome = (
                    "selected"
                    if jev_ids
                    else {"none": "none_fits", "insufficient": "insufficient_context"}.get(
                        jev.get("outcome", jev.get("choice")), jev.get("outcome", jev.get("choice"))
                    )
                )
                if (
                    jev_outcome in {"selected", "none_fits"}
                    and review["status"] != "rejected"
                    and review["outcome"] in {"selected", "none_fits"}
                ):
                    counts["comparable"] += 1
                    counts["correct"] += (jev_outcome, sorted(jev_ids)) == (
                        review["outcome"],
                        sorted(review["gold_ids"]),
                    )
                    counts["accuracy"] = counts["correct"] / counts["comparable"]
        return {
            "total": total,
            "reviewed": reviewed,
            "unreviewed": total - reviewed,
            "accepted": accepted,
            "buckets": dict(buckets),
            "by_confidence": accuracy,
            "note": "Observed Jev accuracy against reviewed human labels, grouped by original Jev confidence; selected review sample, not benchmark accuracy. Missing Jev judgments, rejections and insufficient context are not comparable.",
        }
