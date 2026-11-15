"""Audit Portuguese gloss coverage and join locally reviewed Portuguese contexts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from wsd.data import file_sha256, prepare_contexts, read_lexicon, write_jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexicon", required=True, help="JSONL from wsd prepare lexicon, or a licensed local inventory")
    p.add_argument("--contexts", help="Reviewed JSONL containing id,text,target,lemma,pos,gold_ids,provenance")
    p.add_argument("--language", default="pt-BR")
    p.add_argument("--source", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    raw = [json.loads(line) for line in Path(a.lexicon).read_text(encoding="utf-8").splitlines() if line.strip()]
    # Language tags are declarations, not an automatic quality guarantee.
    language_counts = Counter(row.get("language", "unspecified") for row in raw)
    if any(not lang.startswith("pt") for lang in language_counts):
        raise ValueError(
            "Portuguese preparation requires explicitly Portuguese-tagged gloss inventory; audit English fallback separately"
        )
    lexicon = read_lexicon(a.lexicon)
    provenance = {
        "source": a.source,
        "version": a.version,
        "license": a.license,
        "lexicon_sha256": file_sha256(a.lexicon),
    }
    examples, counts = prepare_contexts(a.contexts, lexicon, provenance, a.language) if a.contexts else ([], {})
    if a.contexts:
        write_jsonl(a.output, examples)
    report = {
        "provenance": provenance,
        "languages_declared": dict(language_counts),
        "entries": len(raw),
        "entries_with_glosses": sum(bool(defs) for defs in lexicon.values()),
        "glosses": sum(len(defs) for defs in lexicon.values()),
        "counts": counts,
        "limitations": [
            "Language labels are not automatic Portuguese gloss validation.",
            "No corpus download, translation, annotation or training is performed by this command.",
        ],
    }
    report_path = a.output + ".manifest.json" if a.contexts else a.output
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
