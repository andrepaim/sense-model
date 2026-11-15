"""Collect unlabeled pilot occurrences from local, licensed documents and lexicons."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import file_sha256, read_jsonl, write_jsonl
from .schema import Candidate, Example, Target

DOCUMENT_FIELDS = ("id", "text", "language", "domain", "source_url", "revision", "license", "license_url", "source")


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    raise ValueError(f"{path}:{number}: invalid JSON") from None
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield row


def _normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def load_inventory(path, definition_language):
    """Pool all POS for a surface form; never narrow candidate senses using a guessed POS."""
    forms = {}
    for row in _rows(path):
        for key in ("lemma", "language", "candidates"):
            if not row.get(key):
                raise ValueError(f"Inventory entry requires {key}")
        if not isinstance(row["lemma"], str) or not isinstance(row["language"], str):
            raise ValueError("Inventory lemma and language must be strings")
        if not isinstance(row.get("candidates"), list) or not isinstance(row.get("surface_forms", []), list):
            raise ValueError("Inventory candidates and surface_forms must be lists")
        if not isinstance(row.get("pos", "unknown"), str):
            raise ValueError("Inventory POS must be a string")
        if row["language"].split("-")[0] != definition_language.split("-")[0]:
            raise ValueError("Inventory language and definition language differ; audit fallback glosses separately")
        candidates = [Candidate(**value) for value in row["candidates"]]
        lookup_forms = [row["lemma"].replace("_", " ")] + row.get("surface_forms", [])
        for form in lookup_forms:
            if not isinstance(form, str) or not form.strip():
                raise ValueError("Inventory surface forms must be nonempty strings")
            form = form.casefold()
            slot = forms.setdefault(form, {"lemma": row["lemma"].replace("_", " "), "candidates": {}, "pos": set()})
            slot["pos"].add(row.get("pos", "unknown"))
            for candidate in candidates:
                if (
                    not isinstance(candidate.id, str)
                    or not candidate.id
                    or not isinstance(candidate.definition, str)
                    or not candidate.definition.strip()
                ):
                    raise ValueError("Inventory candidates need IDs and definitions")
                old = slot["candidates"].get(candidate.id)
                if old and old.definition != candidate.definition:
                    raise ValueError("Conflicting definitions for the same inventory candidate ID")
                slot["candidates"][candidate.id] = candidate
    if not forms:
        raise ValueError("Inventory has no usable forms")
    for slot in forms.values():
        if len(slot["candidates"]) > 127:
            raise ValueError("Complete candidate inventory exceeds model limit; do not silently prune it")
    return forms


def collect(
    documents,
    lexicon,
    *,
    definition_language,
    inventory_version,
    inventory_license,
    limit_per_language=250,
    max_per_document=5,
    max_per_lemma=3,
    seed=42,
):
    if min(limit_per_language, max_per_document, max_per_lemma) < 1:
        raise ValueError("Collection limits must be positive")
    if not inventory_version.strip() or not inventory_license.strip():
        raise ValueError("Pin an inventory version and license")
    forms = load_inventory(lexicon, definition_language)
    # Index token sequences by first token rather than building a huge lexicon
    # regex. This supports overlapping word/phrase targets and original offsets.
    form_index = defaultdict(list)
    for form in sorted(forms, key=lambda s: (-len(s), s)):
        tokens = tuple(re.findall(r"\w+|[^\w\s]", form.casefold()))
        if tokens:
            form_index[tokens[0]].append((tokens, form))

    def matches(sentence):
        tokens = list(re.finditer(r"\w+|[^\w\s]", sentence))
        values = [token.group().casefold() for token in tokens]
        for start, value in enumerate(values):
            for sequence, form in form_index.get(value, []):
                end = start + len(sequence)
                if tuple(values[start:end]) == sequence:
                    yield tokens[start].start(), tokens[end - 1].end(), form

    pools = defaultdict(list)
    seen_ids, contexts, report = set(), set(), Counter()
    document_hash = file_sha256(documents)
    inventory_hash = file_sha256(lexicon)
    for document in _rows(documents):
        if any(not isinstance(document.get(k), str) or not document[k].strip() for k in DOCUMENT_FIELDS):
            raise ValueError("Document requires nonempty strings: " + ", ".join(DOCUMENT_FIELDS))
        identity = (document["source"], document["id"])
        if identity in seen_ids:
            raise ValueError("Duplicate document identity")
        seen_ids.add(identity)
        if document["language"].split("-")[0] != definition_language.split("-")[0]:
            report["documents_other_language"] += 1
            continue
        report["documents_scanned"] += 1
        # Conservative boundaries only: abbreviations may need human extraction flags.
        for sentence_match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", document["text"]):
            sentence = sentence_match.group().strip()
            if not sentence:
                continue
            for match_start, match_end, form in matches(sentence):
                inventory = forms[form]
                signature = (_normalized(sentence), match_start, match_end, form)
                if signature in contexts:
                    report["duplicate_occurrences"] += 1
                    continue
                contexts.add(signature)
                target = Target(match_start, match_end, sentence[match_start:match_end])
                candidates = list(inventory["candidates"].values())
                if "*" in sentence or any("*" in c.definition for c in candidates):
                    report["unsupported_marker"] += 1
                    continue
                domain = document["domain"]
                bucket = (
                    (
                        "technical_ambiguous"
                        if domain in ("technical", "academic", "scientific")
                        else "general_ambiguous"
                    )
                    if len(candidates) > 1
                    else "single_candidate"
                )
                source_offset = (
                    sentence_match.start() + len(sentence_match.group()) - len(sentence_match.group().lstrip())
                )
                source_position = source_offset + match_start
                eid = hashlib.sha256(
                    json.dumps(
                        [identity, document["revision"], source_position, inventory_hash, form], ensure_ascii=False
                    ).encode()
                ).hexdigest()[:24]
                provenance = {
                    k: document[k] for k in ("source", "source_url", "revision", "license", "license_url", "domain")
                }
                provenance.update(
                    document_id=document["id"],
                    source_document_sha256=hashlib.sha256(document["text"].encode()).hexdigest(),
                    source_offset=source_position,
                    documents_sha256=document_hash,
                    inventory_sha256=inventory_hash,
                    inventory_version=inventory_version,
                    inventory_license=inventory_license,
                    lemma=inventory["lemma"],
                    inventory_pos=sorted(inventory["pos"]),
                    bucket=bucket,
                    annotation="unlabeled",
                    review_status="unreviewed",
                    collection_seed=seed,
                )
                example = Example(
                    eid, sentence, target, candidates, document["language"], definition_language, None, provenance
                )
                example.validate()
                pools[document["language"]].append(example)
                report["matched_occurrences"] += 1
    selected = []
    rng = random.Random(seed)
    for language in sorted(pools):
        pool = pools[language]
        rng.shuffle(pool)
        # Hold roughly 10% slots for single-candidate sanity checks; fill any shortfall
        # from the remaining pool. No candidate receives a semantic gold label here.
        ambiguous = [e for e in pool if len(e.candidates) > 1]
        singles = [e for e in pool if len(e.candidates) == 1]
        target_single = min(len(singles), max(1, limit_per_language // 10))
        ordered = singles[:target_single] + ambiguous + singles[target_single:]
        doc_counts, lemma_counts = Counter(), Counter()
        count = 0
        for example in ordered:
            doc = (example.provenance["source"], example.provenance["document_id"])
            lemma = example.provenance["lemma"].casefold()
            if count >= limit_per_language:
                break
            if doc_counts[doc] >= max_per_document or lemma_counts[lemma] >= max_per_lemma:
                report["capped_occurrences"] += 1
                continue
            doc_counts[doc] += 1
            lemma_counts[lemma] += 1
            selected.append(example)
            count += 1
    audit = audit_examples(selected)
    audit.update(
        format_version=1,
        kind="unlabeled-pilot-pool",
        seed=seed,
        documents_sha256=document_hash,
        inventory_sha256=inventory_hash,
        inventory_version=inventory_version,
        inventory_license=inventory_license,
        counts=dict(report),
        requested_per_language=limit_per_language,
        shortfalls={
            lang: max(0, limit_per_language - sum(e.language == lang for e in selected))
            for lang in (sorted(pools) or [definition_language])
        },
        limitations=[
            "Literal inventory forms and explicit surface aliases only; no morphological/POS inference.",
            "Sentence boundaries are conservative and require human review.",
            "No matching-sense or none-fits labels have been inferred.",
        ],
    )
    return selected, audit


def audit_examples(examples):
    by_language = {}
    for language in sorted({e.language for e in examples}):
        group = [e for e in examples if e.language == language]
        by_language[language] = {
            "count": len(group),
            "documents": len({(e.provenance.get("source"), e.provenance.get("document_id")) for e in group}),
            "lemmas": len({e.provenance.get("lemma", e.target.text.casefold()) for e in group}),
            "buckets": dict(Counter(e.provenance.get("bucket", "unspecified") for e in group)),
            "candidate_counts": dict(Counter(len(e.candidates) for e in group)),
            "labeled": sum(e.gold_ids is not None for e in group),
        }
    return {
        "total": len(examples),
        "by_language": by_language,
        "missing_source_provenance": [
            e.id
            for e in examples
            if any(
                not e.provenance.get(k)
                for k in (
                    "source",
                    "source_url",
                    "document_id",
                    "revision",
                    "license",
                    "license_url",
                    "inventory_version",
                    "inventory_sha256",
                )
            )
        ],
        "pilot_goal_per_language": 100,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--documents", required=True)
    p.add_argument("--lexicon", required=True)
    p.add_argument("--definition-language", required=True)
    p.add_argument("--inventory-version", required=True)
    p.add_argument("--inventory-license", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit-per-language", type=int, default=250)
    p.add_argument("--max-per-document", type=int, default=5)
    p.add_argument("--max-per-lemma", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = sub.add_parser("audit")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Output exists; use a new path to preserve the dataset version")
    if args.command == "collect":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        output = kwargs.pop("output")
        examples, report = collect(**kwargs)
        write_jsonl(output, examples)
        report["examples_sha256"] = file_sha256(output)
        report_path = Path(output + ".manifest.json")
    else:
        report = audit_examples(read_jsonl(args.input))
        report["examples_sha256"] = file_sha256(args.input)
        report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
