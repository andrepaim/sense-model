"""Portable annotated examples, local lexicon adapters, and leakage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from wsd.schema import Candidate, Example, Target


def read_jsonl(path):
    examples = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                example = Example.from_dict(json.loads(line))
                example.validate()
                if example.id in seen:
                    raise ValueError(f"duplicate example ID {example.id}")
                seen.add(example.id)
                examples.append(example)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return examples


def write_jsonl(path, examples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            example.validate()
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(text):
    # Case, accents and punctuation variations should not disguise copied contexts.
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _group_keys(example):
    p = example.provenance
    keys = [("id", example.id), ("context", _normalized(example.text))]
    for field in ("document_id", "group_id", "original_id", "translation_of", "synthetic_parent"):
        if p.get(field):
            # original_id and translation_of reference the same example family.
            namespace = "family" if field in ("original_id", "translation_of", "synthetic_parent") else field
            value = str(p[field])
            if field == "document_id":
                value = str(p.get("source", "")) + ":" + value
            keys.append((namespace, value))
    keys.append(("family", example.id))
    return keys


def validate_splits(paths):
    """Reject cross-split document, identity, normalized-context and family leakage.

    A five-token shingle audit also flags highly overlapping long contexts. Short
    contexts use exact normalized matching to avoid common-word false positives.
    """
    owners, shingles, errors, report = {}, [], [], {}
    for name, path in paths.items():
        examples = read_jsonl(path)
        report[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "count": len(examples),
            "languages": dict(Counter(ex.language for ex in examples)),
            "annotations": dict(Counter(str(ex.provenance.get("annotation", "unspecified")) for ex in examples)),
        }
        for ex in examples:
            for key in _group_keys(ex):
                if key in owners and owners[key][0] != name:
                    errors.append(f"{name}/{ex.id} overlaps {owners[key][0]}/{owners[key][1]} ({key[0]})")
                owners[key] = (name, ex.id)
            words = _normalized(ex.text).split()
            if len(words) >= 15:
                grams = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
                shingles.append((name, ex.id, grams))
    # Inverted index avoids quadratic comparisons on realistic corpora.
    inverted = defaultdict(list)
    for split, eid, grams in shingles:
        overlaps = Counter()
        for gram in grams:
            overlaps.update(inverted[gram])
        for (other_split, other_id, other_size), shared in overlaps.items():
            if other_split != split and shared / (len(grams) + other_size - shared) >= 0.8:
                errors.append(f"{split}/{eid} near-duplicates {other_split}/{other_id}")
        for gram in grams:
            inverted[gram].append((split, eid, len(grams)))
    if errors:
        raise ValueError("Cross-split leakage:\n" + "\n".join(errors[:50]))
    return {"format_version": 1, "splits": report, "leakage": "passed"}


def split_examples(examples, seed=42, ratios=(0.8, 0.1, 0.05, 0.05)):
    """Split connected document/translation/duplicate groups as indivisible units."""
    if len(ratios) != 4 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("Four nonnegative train/dev/calibration/test ratios must sum to 1")
    parents = list(range(len(examples)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, ex in enumerate(examples):
        for key in _group_keys(ex):
            if key in owners:
                parents[root(i)] = root(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[root(i)].append(ex)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    names = ("train", "dev", "calibration", "test")
    outputs = {name: [] for name in names}
    targets = [len(examples) * r for r in ratios]
    for group in groups:
        # Largest deficit keeps large groups intact without starving later splits.
        idx = max(range(4), key=lambda j: targets[j] - len(outputs[names[j]]) if ratios[j] else -float("inf"))
        outputs[names[idx]].extend(group)
    return outputs


def read_lexicon(path):
    """Read a local JSONL lexicon: lemma, pos, language, candidates[{id,definition}]."""
    lexicon = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["lemma"].replace("_", " ").casefold(), row.get("pos", ""))
            defs = [Candidate(**item) for item in row.get("candidates", []) if item.get("definition", "").strip()]
            if key in lexicon:
                raise ValueError(f"Duplicate lemma/POS at {path}:{line_number}: {key}")
            if len({d.id for d in defs}) != len(defs):
                raise ValueError(f"Duplicate candidate ID at {path}:{line_number}")
            lexicon[key] = defs
    return lexicon


def export_wn(spec, output, language):
    """Export an already-installed wn lexicon. Never downloads or starts services."""
    import wn

    lex = wn.Wordnet(spec)
    grouped = defaultdict(dict)
    definitions = Counter()
    for word in lex.words():
        pos = "a" if word.pos == "s" else word.pos
        key = (word.lemma().replace("_", " "), pos)
        for sense in word.senses():
            synset = sense.synset()
            definition = synset.definition()
            definitions["with_definition" if definition else "missing_definition"] += 1
            if definition:
                grouped[key].setdefault(synset.id, {"id": synset.id, "definition": definition})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as stream:
        for (lemma, pos), candidates in sorted(grouped.items()):
            stream.write(
                json.dumps(
                    {"lemma": lemma, "pos": pos, "language": language, "candidates": list(candidates.values())},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "lexicon": spec,
        "entries": len(grouped),
        "sense_definitions": dict(definitions),
        "sha256": file_sha256(output),
    }


def _sense_index(path, namespace):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, offset, *_ = line.split()
        pos = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "s"}[key.split("%")[1][0]]
        result[key] = f"{namespace}-{offset}-{pos}"
    return result


def prepare_raganato(prefix, index_sense, lexicon, provenance, language="en", namespace="omw-en"):
    """Prepare corpus XML using explicit local candidate inventories.

    Reconstruct offsets token by token, retaining punctuation spacing from a
    conservative detokenizer. Missing mappings are reported, never relabeled none.
    """
    mapping = _sense_index(index_sense, namespace)
    gold = {}
    for line in Path(str(prefix) + ".gold.key.txt").read_text(encoding="utf-8").splitlines():
        eid, *keys = line.split()
        gold[eid] = keys
    report, examples = Counter(), []
    for _, sentence in ET.iterparse(str(prefix) + ".data.xml", events=("end",)):
        if sentence.tag != "sentence":
            continue
        text, offsets = "", []
        for el in sentence:
            token = "".join(el.itertext()).strip()
            attach = bool(re.match(r"^[,.;:!?%)\]}]", token)) or token.startswith(
                ("'s", "n't", "'re", "'ve", "'ll", "'d", "'m")
            )
            if text and not attach and text[-1] not in "([{":
                text += " "
            start = len(text)
            text += token
            offsets.append((start, len(text), token, el))
        for start, end, token, el in offsets:
            if el.tag != "instance":
                continue
            eid = el.get("id")
            keys = gold.get(eid, [])
            if not keys or any(k not in mapping for k in keys):
                report["missing_gold_mapping"] += 1
                continue
            pos = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}.get(el.get("pos"), el.get("pos", ""))
            candidates = lexicon.get((el.get("lemma", token).replace("_", " ").casefold(), pos), [])
            ids = [mapping[k] for k in keys]
            if not candidates or not any(s in {c.id for c in candidates} for s in ids):
                report["missing_candidate_inventory"] += 1
                continue
            covered = [s for s in ids if s in {c.id for c in candidates}]
            p = dict(provenance, document_id=eid.split(".s")[0], annotation="human")
            ex = Example(
                id=eid,
                text=text,
                target=Target(start, end, token),
                candidates=candidates,
                language=language,
                definition_language=language,
                gold_ids=covered,
                provenance=p,
            )
            try:
                ex.validate()
            except ValueError:
                report["invalid"] += 1
                continue
            examples.append(ex)
            report["accepted"] += 1
        sentence.clear()
    return examples, dict(report)


def prepare_contexts(path, lexicon, provenance, language):
    """Join reviewed context annotations with local dictionary candidates.

    Context rows contain id,text,target,lemma,pos,gold_ids,provenance. Explicit
    gold_ids=[] denotes none fits; omitted gold_ids remains unlabeled.
    """
    examples, report = [], Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("provenance", {}).get("review_status") == "unreviewed":
            report["unreviewed"] += 1
            continue
        candidates = lexicon.get((row["lemma"].replace("_", " ").casefold(), row.get("pos", "")), [])
        if not candidates:
            report["missing_inventory"] += 1
            continue
        data = {k: row[k] for k in ("id", "text", "target")}
        data.update(
            candidates=[{"id": c.id, "definition": c.definition} for c in candidates],
            language=language,
            definition_language=language,
            gold_ids=row.get("gold_ids"),
            provenance={**provenance, **row.get("provenance", {})},
        )
        ex = Example.from_dict(data)
        ex.validate()
        examples.append(ex)
        report["accepted"] += 1
    return examples, dict(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--split", action="append", required=True, help="NAME=PATH; repeat for all splits")
    check.add_argument("--output")
    split = sub.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--ratios", default="0.8,0.1,0.05,0.05")
    lex = sub.add_parser("lexicon")
    lex.add_argument("--wn-spec", required=True)
    lex.add_argument("--language", required=True)
    lex.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_splits(dict(item.split("=", 1) for item in args.split))
    elif args.command == "lexicon":
        report = export_wn(args.wn_spec, args.output, args.language)
    else:
        outputs = split_examples(read_jsonl(args.input), args.seed, tuple(map(float, args.ratios.split(","))))
        paths = {}
        for name, examples in outputs.items():
            paths[name] = Path(args.output) / f"{name}.jsonl"
            write_jsonl(paths[name], examples)
        report = validate_splits(paths)
        report["seed"] = args.seed
        Path(args.output, "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if getattr(args, "output", None) and args.command == "validate":
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
