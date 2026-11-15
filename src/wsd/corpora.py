"""Princeton WordNet Gloss Corpus (WordNet-3.0-glosstag) as WordNetExamples.

Every synset's definition (``<def>``) and example sentences (``<ex>``) come
sense-tagged (~340k manual + ~120k automatic tags). Download and unpack
https://wordnetcode.princeton.edu/glosstag-files/WordNet-3.0-glosstag.tar.bz2
and point ``--wngt`` at its ``glosstag`` directory.

Example sentences of held-out synsets are skipped (``exclude_synsets``) so the
WordNet held-out benchmark stays clean: the benchmark is split by synset.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WordNetExample:
    synset_id: str
    word_form: str
    lemma: str
    pos: str
    marked_text: str
    sentence: str
    gold_ids: tuple[str, ...] = ()


_FILES = ("noun.xml", "verb.xml", "adj.xml", "adv.xml")


def _text(tok) -> str:
    return "".join(t for t in tok.itertext() if t).strip()


def _join(tokens: list[tuple[str, bool]]) -> str:
    """Join ``(text, space_after)`` pairs the way the corpus marks separators."""
    out = []
    for text, space in tokens:
        out.append(text)
        if space:
            out.append(" ")
    return "".join(out).strip()


def _sense_tag(el, tags: frozenset[str]) -> tuple[str, list[str]] | None:
    """``(lemma, sense_keys)`` of a tagged ``<wf>``/``<glob>``, or None."""
    if el.get("tag") not in tags:
        return None
    ids = el.findall("id")
    keys = [i.get("sk") for i in ids if i.get("sk")]
    lemma = next((i.get("lemma") for i in ids if i.get("lemma")), None)
    return (lemma, keys) if keys and lemma else None


def _annotations(block, tags: frozenset[str]):
    """Return the token list ``[(text, space_after)]`` and ``[(token_indices, lemma, sense_keys)]``
    for tagged words in a def/ex block. Multiword collocations (``<cf>`` + ``<glob>``) span several tokens."""
    tokens: list[tuple[str, bool]] = []
    anns = []
    coll_spans: dict[str, list[int]] = {}
    coll_meta: dict[str, tuple[str, list[str]]] = {}
    for el in block:
        if el.tag not in ("wf", "cf"):
            continue
        idx = len(tokens)
        tokens.append((_text(el), el.get("sep") != ""))
        if el.tag == "wf":
            if tag := _sense_tag(el, tags):
                anns.append(([idx], *tag))
            continue
        for coll in (el.get("coll") or "").split(","):  # collocation piece; the glob carries the annotation
            coll_spans.setdefault(coll, []).append(idx)
        for glob in el.findall("glob"):
            if tag := _sense_tag(glob, tags):
                coll_meta[glob.get("coll")] = tag
    for coll, (lemma, keys) in coll_meta.items():
        span = coll_spans.get(coll)
        if span and span == list(range(span[0], span[-1] + 1)):  # contiguous only
            anns.append((span, lemma, keys))
    return tokens, anns


def load_wngt(
    glosstag_dir: Path,
    sense_index: dict[str, str],
    parts: frozenset[str] = frozenset({"def", "ex"}),
    tags: frozenset[str] = frozenset({"man", "auto"}),
    exclude_synsets: frozenset[str] = frozenset(),
) -> list[WordNetExample]:
    examples: list[WordNetExample] = []
    for name in _FILES:
        synset_id = None
        for event, el in ET.iterparse(glosstag_dir / "merged" / name, events=("start", "end")):
            if event == "start" and el.tag == "synset":
                synset_id = f"omw-en-{el.get('ofs')}-{el.get('pos')}"
                continue
            if event != "end" or el.tag not in parts:
                continue
            if el.tag == "ex" and synset_id in exclude_synsets:
                el.clear()
                continue
            tokens, anns = _annotations(el, tags)
            if any("*" in t for t, _ in tokens):
                el.clear()
                continue
            if tokens and tokens[-1][0] == ";":  # every gloss ends in a separator token
                tokens = tokens[:-1]
                anns = [a for a in anns if a[0][-1] < len(tokens)]
            sentence = _join(tokens)
            for span, lemma, keys in anns:
                gold = [sid for k in keys if (sid := sense_index.get(k))]
                if not gold:
                    continue
                pos = gold[0].rsplit("-", 1)[1]  # omw-en-<offset>-<pos>
                marked_tokens = list(tokens)  # markers around the whole (possibly multiword) span
                marked_tokens[span[0]] = ("*" + marked_tokens[span[0]][0], marked_tokens[span[0]][1])
                marked_tokens[span[-1]] = (marked_tokens[span[-1]][0] + "*", marked_tokens[span[-1]][1])
                examples.append(
                    WordNetExample(
                        synset_id=gold[0],
                        word_form=" ".join(t for t, _ in tokens[span[0] : span[-1] + 1]),
                        lemma=lemma.replace("_", " "),
                        pos="a" if pos == "s" else pos,
                        marked_text=_join(marked_tokens),
                        sentence=sentence,
                        gold_ids=tuple(gold[1:]),
                    )
                )
            el.clear()
    return examples


def prepare_glosstag(directory, sense_index, lexicon, provenance, tags=frozenset({"man"})):
    """Adapt annotated glosses to portable candidate examples; report exclusions."""
    from collections import Counter

    from wsd.data import _sense_index
    from wsd.schema import Example, Target

    examples, report = [], Counter()
    for i, row in enumerate(load_wngt(Path(directory), _sense_index(sense_index, "omw-en"), tags=tags)):
        candidates = lexicon.get((row.lemma.casefold(), row.pos), [])
        ids = {candidate.id for candidate in candidates}
        gold = [sid for sid in (row.synset_id, *row.gold_ids) if sid in ids]
        if not gold:
            report["missing_inventory"] += 1
            continue
        # The source loader refuses literal asterisks and adds exactly two.
        start = row.marked_text.index("*")
        end = row.marked_text.index("*", start + 1) - 1
        text = row.marked_text.replace("*", "")
        annotation = "human" if tags == frozenset({"man"}) else "mixed"
        p = dict(provenance, document_id="gloss:" + row.synset_id, annotation=annotation)
        ex = Example(
            "glosstag:" + str(i),
            text,
            Target(start, end, text[start:end]),
            candidates,
            language="en",
            definition_language="en",
            gold_ids=gold,
            provenance=p,
        )
        ex.validate()
        examples.append(ex)
        report["accepted"] += 1
    return examples, dict(report)
