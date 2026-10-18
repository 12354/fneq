"""Probable callee correspondences, from where callees are called.

In each compared pair the direct calls of both sides are put in address order and
aligned, with the callees the correspondence already names as anchors: unnamed callees
that sit at the same position between two anchors (on both sides, in runs of equal
length) are probably the same function. Votes are summed over all pairs of a batch.
The result is a list to review, not an assumption of any proof: an accepted line goes
into the correspondence file as a [[callees]] entry.

    fneq suggest-corr REPORT_DIR [-o corr_suggestions.csv]
"""
from __future__ import annotations

import collections
import csv
import difflib
import os

FIELDS = ["confidence", "a", "a_name", "b", "b_name", "call_sites", "callers",
          "a_consistency", "b_consistency", "alternatives"]


def call_sequence(rep: dict, side: str) -> list[tuple[int, str | None]]:
    """(callee address, its name or None) of the direct calls of `side`, in address order."""
    inp = (rep.get("inputs") or {}).get(side) or {}
    names = (rep.get("callees") or {}).get(side) or {}
    out = []
    for c in inp.get("calls", []):
        if not str(c.get("callee", "")).startswith("0x"):
            continue                    # (indirect)
        info = names.get(c["callee"]) or {}
        name = info.get("name")
        out.append((int(c["at"], 16), int(c["callee"], 16),
                    None if not name or name.startswith("0x") else name))
    out.sort()
    return [(callee, name) for _, callee, name in out]


def evidence(rep: dict) -> dict | None:
    """What suggest() needs of one report (small: kept per pair during a batch)."""
    if "A" not in (rep.get("inputs") or {}) or "B" not in rep["inputs"]:
        return None
    return {"caller": rep["inputs"]["A"].get("function"),
            "A": call_sequence(rep, "A"), "B": call_sequence(rep, "B")}


def suggest(items: list[dict]) -> list[dict]:
    votes: collections.Counter = collections.Counter()
    callers: dict = collections.defaultdict(set)
    named_a: dict[int, str] = {}
    named_b: dict[int, str] = {}
    for ev in items:
        if ev is None:
            continue
        sa, sb = ev["A"], ev["B"]
        for c, n in sa:
            if n:
                named_a[c] = n
        for c, n in sb:
            if n:
                named_b[c] = n
        ta = [n or f"?A{c:x}" for c, n in sa]
        tb = [n or f"?B{c:x}" for c, n in sb]
        sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace" or i2 - i1 != j2 - j1:
                continue
            for i, j in zip(range(i1, i2), range(j1, j2)):
                # at least one side unnamed: two different names are a real difference
                if sa[i][1] is None or sb[j][1] is None:
                    k = (sa[i][0], sb[j][0])
                    votes[k] += 1
                    callers[k].add(ev["caller"])
    per_a: dict = collections.defaultdict(list)
    per_b: collections.Counter = collections.Counter()
    for (a, b), n in votes.items():
        per_a[a].append((n, b))
        per_b[b] += n
    rows = []
    for a, lst in per_a.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        n, b = lst[0]
        ca = n / sum(x for x, _ in lst)
        cb = n / per_b[b]
        conf = "high" if ca >= 0.8 and cb >= 0.8 and n >= 3 else \
            "medium" if ca >= 0.8 and cb >= 0.8 else "low"
        rows.append({
            "confidence": conf, "a": f"0x{a:x}", "a_name": named_a.get(a, ""),
            "b": f"0x{b:x}", "b_name": named_b.get(b, ""), "call_sites": n,
            "callers": len(callers[(a, b)]), "a_consistency": round(ca, 2),
            "b_consistency": round(cb, 2),
            "alternatives": "; ".join(f"0x{bb:x}:{x}" for x, bb in lst[1:4]),
        })
    order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda r: (order[r["confidence"]], -r["call_sites"], r["a"]))
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def from_dir(report_dir: str) -> list[dict]:
    import json
    items = []
    for name in sorted(os.listdir(report_dir)):
        if name.endswith(".json"):
            try:
                with open(os.path.join(report_dir, name), encoding="utf-8") as f:
                    items.append(evidence(json.load(f)))
            except (OSError, ValueError):
                continue
    return suggest(items)


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="fneq suggest-corr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", help="directory of per-pair JSON reports (fneq batch --out-dir)")
    ap.add_argument("-o", "--out", default="corr_suggestions.csv")
    args = ap.parse_args(argv)
    rows = from_dir(args.reports)
    write_csv(rows, args.out)
    counts = collections.Counter(r["confidence"] for r in rows)
    print(f"{len(rows)} suggestions ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}) -> {args.out}")
    return 0
