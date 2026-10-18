"""Callee correspondence suggestions from aligned call sites (fneq suggest-corr)."""
from __future__ import annotations

from fneq import suggest


def rep(calls_a, calls_b):
    """A report with the given (callee address, name or None) calls on each side."""
    def side(calls):
        return ({"function": "0x1000", "calls": [{"at": hex(0x1000 + 8 * i), "callee": hex(c), "tail": False}
                                                 for i, (c, _) in enumerate(calls)]},
                {hex(c): {"name": n or hex(c)} for c, n in calls})
    ia, na = side(calls_a)
    ib, nb = side(calls_b)
    return {"inputs": {"A": ia, "B": ib}, "callees": {"A": na, "B": nb}}


def test_unnamed_between_anchors():
    r = rep([(0x10, "lock"), (0x20, None), (0x30, "unlock")],
            [(0x91, "lock"), (0x92, "roundf"), (0x93, "unlock")])
    rows = suggest.suggest([suggest.evidence(r)] * 3)
    assert len(rows) == 1
    row = rows[0]
    assert (row["a"], row["b"], row["b_name"], row["call_sites"], row["confidence"]) == \
        ("0x20", "0x92", "roundf", 3, "high")


def test_different_names_are_not_suggested():
    r = rep([(0x10, "lock"), (0x20, "memset"), (0x30, "unlock")],
            [(0x91, "lock"), (0x92, "memcpy"), (0x93, "unlock")])
    assert suggest.suggest([suggest.evidence(r)]) == []


def test_inconsistent_votes_are_low():
    r1 = rep([(0x10, "f"), (0x20, None), (0x30, "g")], [(0x91, "f"), (0x92, None), (0x93, "g")])
    r2 = rep([(0x10, "f"), (0x20, None), (0x30, "g")], [(0x91, "f"), (0x95, None), (0x93, "g")])
    rows = suggest.suggest([suggest.evidence(r1), suggest.evidence(r2)])
    assert [r["confidence"] for r in rows] == ["low"]
