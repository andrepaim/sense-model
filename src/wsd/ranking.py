"""Rank of the labelled meaning within its dictionary entry, from score files (the rule of training round 1).

An application that shows a word's meanings in dictionary order, grouped by entry, reorders the meanings it offered
the model by the model's probabilities; the labelled meaning's rank within its entry is then the measure ("first",
"top 3", mean rank), counted over every item. A prompt that was skipped, is missing from the score file or was
rejected, or whose none-fits probability is at least that of every offered meaning, leaves dictionary order. A
labelled meaning the prompt did not offer comes after the offered meanings of its entry, in dictionary order.

Items are JSONL rows with ``id``, ``labelled`` ({entry, sense}, zero-based), ``entrySenses`` (meanings per entry),
``meanings`` (of the labelled entry, for the strata) and ``perCap`` ({form: {prompt (the prompt's SHA-256),
candidates ([entry, sense] in prompt order), labelledOffered, skip (present when the application did not ask the
model)}}): the item files contextual-dictionary writes. Score files are the rows of ``wsd score``.

``offered`` is the trainer's development metric over the items whose labelled meaning was offered and scored: the
top choice among the offered meanings and none-fits is the labelled one, and the labelled one is among the three
highest offered meanings. Paired bootstrap intervals compare "first" of each model with a baseline, item by item.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from wsd.evaluate import paired_bootstrap
from wsd.scoring import read_scores

STRATA = [(2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, math.inf)]


def _stratum(low, high):
    return f"{low}+" if high == math.inf else str(low) if low == high else f"{low}-{high}"


def outcome(item, form, scores):
    """(rank of the labelled meaning within its entry, kind): applied, none, missing, rejected:CODE or skipped."""
    entry, sense = item["labelled"]["entry"], item["labelled"]["sense"]
    dictionary = sense + 1
    info = item["perCap"][form]
    if "skip" in info:
        return dictionary, "skipped"
    score = scores.get(info["prompt"])
    if score is None:
        return dictionary, "missing"
    if score.get("type") != "scored" or not score.get("logits"):
        return dictionary, f"rejected:{score.get('code')}"
    values = score["logits"]
    if len(values) != len(info["candidates"]) + 1:
        raise ValueError(
            f"{item['id']} ({form}): score has {len(values)} logits for {len(info['candidates'])} candidates "
            "and none-fits; the score file does not fit these items"
        )
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{item['id']} ({form}): logits are not finite")
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    total = sum(weights)
    probs = [w / total for w in weights]
    none = probs.pop()
    if none >= max(probs):
        return dictionary, "none"
    offered = {(e, s): probs[i] for i, (e, s) in enumerate(info["candidates"])}
    count = item["entrySenses"][entry]
    ranked = sorted((s for s in range(count) if (entry, s) in offered), key=lambda s: (-offered[(entry, s)], s))
    rest = [s for s in range(count) if (entry, s) not in offered]
    return (ranked + rest).index(sense) + 1, "applied"


def summary(ranks):
    """First, top three and mean rank."""
    n = len(ranks)
    return sum(r == 1 for r in ranks) / n, sum(r <= 3 for r in ranks) / n, sum(ranks) / n


def offered_accuracy(items, form, scores):
    """(first, top three, items counted) over the items whose labelled meaning was offered and scored."""
    first = top3 = n = 0
    for it in items:
        info = it["perCap"].get(form, {})
        score = scores.get(info.get("prompt")) if info.get("labelledOffered") else None
        if not score or score.get("type") != "scored":
            continue
        gold = [tuple(c) for c in info["candidates"]].index((it["labelled"]["entry"], it["labelled"]["sense"]))
        logits = score["logits"]
        n += 1
        first += max(range(len(logits)), key=lambda i: logits[i]) == gold
        top3 += gold in sorted(range(len(logits) - 1), key=lambda i: -logits[i])[:3]
    return first / max(n, 1), top3 / max(n, 1), n


def rank_report(items, forms, named_scores, baseline=None, samples=2000, seed=1):
    if not items:
        raise ValueError("No items to rank (empty item file, or every item excluded)")
    report = {"items": len(items)}
    first, top3, mean = summary([it["labelled"]["sense"] + 1 for it in items])
    report["dictionary"] = {"first": first, "top3": top3, "mean_rank": mean}
    report["forms"] = {}
    for form in forms:
        results, firsts = {}, {}
        for name, scores in named_scores.items():
            outcomes = [outcome(it, form, scores) for it in items]
            ranks = [rank for rank, _ in outcomes]
            first, top3, mean = summary(ranks)
            kinds = {}
            for _, kind in outcomes:
                kinds[kind] = kinds.get(kind, 0) + 1
            o_first, o_top3, o_n = offered_accuracy(items, form, scores)
            strata = {}
            for low, high in STRATA:
                members = [i for i, it in enumerate(items) if low <= it["meanings"] <= high]
                if members:
                    strata[_stratum(low, high)] = {
                        "first": sum(ranks[i] == 1 for i in members) / len(members),
                        "items": len(members),
                    }
            results[name] = {
                "first": first,
                "top3": top3,
                "mean_rank": mean,
                "outcomes": dict(sorted(kinds.items())),
                "offered": {"first": o_first, "top3": o_top3, "items": o_n},
                "by_meanings": strata,
            }
            firsts[name] = [rank == 1 for rank in ranks]
        if baseline is not None:
            for name in results:
                if name != baseline:
                    results[name]["vs_baseline"] = {
                        "baseline": baseline,
                        **paired_bootstrap(firsts[name], firsts[baseline], samples, seed),
                    }
        report["forms"][form] = results
    return report


def markdown(report):
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    points = lambda x: f"{100 * x:+.1f}"  # noqa: E731
    d = report["dictionary"]
    lines = [f"{report['items']} items; dictionary order first {pct(d['first'])}, top 3 {pct(d['top3'])}", ""]
    for form, results in report["forms"].items():
        lines += [f"## {form}", "", "| Model | First | Top 3 | Mean rank | Offered first | Difference in first (95%) |"]
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for name, r in results.items():
            versus = r.get("vs_baseline")
            interval = (
                f"{points(versus['difference'])} ({points(versus['ci95'][0])} to {points(versus['ci95'][1])})"
                if versus
                else ""
            )
            lines.append(
                f"| {name} | {pct(r['first'])} | {pct(r['top3'])} | {r['mean_rank']:.2f} | "
                f"{pct(r['offered']['first'])} | {interval} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--forms", nargs="+", default=["24|c1600", "127"], help="perCap keys to count")
    parser.add_argument("--scores", nargs="+", required=True, help="NAME=FILE score files (wsd score)")
    parser.add_argument("--baseline", help="NAME of the score file the paired intervals compare with")
    parser.add_argument("--exclude", type=Path, help="Item IDs to leave out, one per line")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, help="JSON report")
    args = parser.parse_args(argv)
    if any("=" not in spec for spec in args.scores):
        parser.error("--scores takes NAME=FILE")
    named = {name: read_scores(path) for name, path in (spec.split("=", 1) for spec in args.scores)}
    if args.baseline is not None and args.baseline not in named:
        parser.error("--baseline must name one of --scores")
    excluded = set(args.exclude.read_text(encoding="utf-8").split()) if args.exclude else set()
    rows = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [it for it in rows if it["id"] not in excluded]
    report = {"excluded": len(rows) - len(items)}
    report.update(rank_report(items, args.forms, named, args.baseline, args.bootstrap_samples, args.seed))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(markdown(report))
    return report


if __name__ == "__main__":
    main()
