"""Prepare local Raganato annotations with an explicitly versioned local lexicon."""

import argparse
import json
from pathlib import Path

from wsd.data import file_sha256, prepare_raganato, read_lexicon, write_jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--corpus", help="Prefix of .data.xml and .gold.key.txt")
    inputs.add_argument("--glosstag", help="WordNet Gloss Corpus directory containing merged/*.xml")
    p.add_argument("--sense-index", required=True)
    p.add_argument("--lexicon", required=True, help="Local JSONL lemma/POS candidate inventory")
    p.add_argument("--source", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--namespace", default="omw-en")
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    provenance = {
        "source": a.source,
        "version": a.version,
        "license": a.license,
        "lexicon_sha256": file_sha256(a.lexicon),
        "sense_index_sha256": file_sha256(a.sense_index),
    }
    if a.glosstag:
        if a.namespace != "omw-en":
            p.error("Gloss Corpus adapter currently requires omw-en IDs")
        from wsd.corpora import prepare_glosstag

        examples, counts = prepare_glosstag(a.glosstag, a.sense_index, read_lexicon(a.lexicon), provenance)
    else:
        examples, counts = prepare_raganato(
            a.corpus, a.sense_index, read_lexicon(a.lexicon), provenance, namespace=a.namespace
        )
    write_jsonl(a.output, examples)
    report = {"provenance": provenance, "counts": counts, "sha256": file_sha256(a.output)}
    Path(a.output + ".manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
