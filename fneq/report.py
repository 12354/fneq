"""Human-readable and JSON reports."""
from __future__ import annotations

import difflib
import json

from fneq.pipeline import Report


def to_json(rep: Report) -> str:
    out = {"verdict": rep.verdict_label, "reason": rep.reason, "backend": rep.backend,
           "exit_code": rep.exit_code, **rep.data}
    return json.dumps(out, indent=2, default=str)


def _ir_diff(ir: dict) -> list[str]:
    a = ir["A"].replace("@src(", "@F(").splitlines()
    b = ir["B"].replace("@tgt(", "@F(").splitlines()
    if ir.get("textually_identical"):
        return ["(normalised IR of both functions is textually identical; the proof still ran)"]
    lines = list(difflib.unified_diff(a, b, "A (normalised)", "B (normalised)", lineterm="", n=2))
    return lines[:200] + (["... (diff truncated)"] if len(lines) > 200 else [])


def to_text(rep: Report) -> str:
    d = rep.data
    L: list[str] = []
    L.append(f"VERDICT: {rep.verdict_label}")
    L.append(f"  reason : {rep.reason}")
    L.append(f"  backend: {rep.backend}")
    if d.get("tool_bug"):
        L.append(f"  TOOL BUG: {d['tool_bug']}")
    if d.get("tool_error"):
        L.append(f"  TOOL ERROR: {d['tool_error']}")
    L.append("")
    if "definition" in d:
        L.append("Definition of equivalence used:")
        L.append("  " + d["definition"])
        dis = d.get("checks_disabled") or []
        L.append("  Checks disabled: " + (", ".join(dis) if dis else "none"))
        L.append("")
    for side in ("A", "B"):
        inp = d.get("inputs", {}).get(side)
        if inp:
            L.append(f"{side}: {inp['binary']} :: {inp['function']} @ {inp['address']} "
                     f"({inp['format']}/{inp['arch']}, {inp['size']} bytes, {inp['blocks']} blocks"
                     f"{', has loops' if inp['has_loop'] else ''})")
            for c in inp["calls"]:
                L.append(f"     call at {c['at']} -> {c['callee']}{' (tail call)' if c['tail'] else ''}")
    if "conventions" in d:
        for side, c in d["conventions"].items():
            how = "inferred: " + "; ".join(c["evidence"]) if c["inferred"] else "specified"
            L.append(f"Convention {side}: {c['name']} ({how})")
    if "return" in d:
        L.append(f"Return value compared as: {d['return']['kind']} ({d['return']['how']})")
    if "signature" in d:
        L.append("Outputs compared: " + ", ".join(d["signature"]["outputs"]))
    L.append("")
    if d.get("assumptions"):
        L.append("Assumptions:")
        L += [f"  - {a}" for a in d["assumptions"]]
        L.append("")
    if d.get("hazards"):
        L.append("Why the functions could not be analysed:")
        L += [f"  - {h}" for h in d["hazards"]]
        L.append("")
    L.append("Per-check result:")
    for check, label in (("return", "return value"), ("memory", "memory writes"),
                         ("calls", "external calls"), ("preserve", "callee-saved regs / stack delta")):
        L.append(f"  {label:32} {_check_status(rep, check)}")
    L.append("")
    if d.get("backends"):
        L.append("Backends:")
        for b in d["backends"]:
            L.append(f"  {b['backend']:13} {b['result']:10} {b.get('seconds', '')}s  {b.get('reason', '')}")
        L.append("")
    ls = d.get("lockstep")
    if ls and ls.get("segment_pairs") is not None:
        L.append(f"Lockstep: {ls['result']}, {ls['segment_pairs']} segment pairs, {ls['visits']} visits, "
                 f"{ls['obligations_proved']} obligations proved by Z3")
        diff = ls.get("first_difference")
        if diff:
            L.append(f"  first difference: A {diff['a']} / B {diff['b']} (segment A {diff['segment'][0]}"
                     f" / B {diff['segment'][1]}): {diff['what']}")
        elif ls.get("stopped_at"):
            L.append(f"  stopped in segment A {ls['stopped_at'][0]} / B {ls['stopped_at'][1]}")
        L.append("")
    t = d.get("testing")
    if t and t.get("stopped"):
        L.append(f"Testing stopped early: {t['stopped']}")
        L.append("")
    cex = d.get("counterexample")
    if cex:
        L.append(f"Counterexample (from {cex.get('source', '?')}, replayed concretely):")
        L.append("  inputs: " + ", ".join(f"{k}={v}" for k, v in cex["inputs"].items()))
        if cex.get("memory"):
            L.append("  memory: " + ", ".join(f"[{k}]={v:#04x}" for k, v in cex["memory"].items()))
        L.append(f"  memory seed: {cex['memory_seed']}" + (f"  ({cex['note']})" if cex.get("note") else ""))
        L.append(f"  first differing effect: {cex['first_difference']}")
        for side in ("A", "B"):
            s = cex[side]
            L.append(f"  {side} outputs: " + ", ".join(f"{k}={v}" for k, v in s["outputs"].items()))
            if s["calls"]:
                L.append(f"  {side} calls  : " + "; ".join(f"{n}({', '.join(a)})" for n, a in s["calls"]))
            if s["writes"]:
                L.append(f"  {side} writes : " + "; ".join(f"[{a}]<-{v} ({n}B)" for a, n, v in s["writes"][:12]))
        for u in cex.get("unmapped_accesses", []):
            L.append(f"  unmapped {u['kind']} at {u['address']:#x} ({u['size']}B) by insn {u['pc']:#x} ({u['side']})")
        L.append("")
    for key in ("testing", "cross_check"):
        t = d.get(key)
        if t:
            title = "Testing" if key == "testing" else "Concrete cross-check of the proof"
            L.append(f"{title}: {t['runs']} runs")
            for side, c in t["coverage"].items():
                L.append(f"  coverage {side}: {c['covered']}/{c['blocks']} blocks ({c['percent']}%)"
                         + (f", unreached: {', '.join(c['unreached'][:20])}" if c["unreached"] else ""))
            if t.get("runs_hit_instruction_limit"):
                L.append(f"  runs stopped by the instruction limit: {t['runs_hit_instruction_limit']}")
            for u in t.get("unmapped_accesses", [])[:10]:
                L.append(f"  unmapped {u['kind']} at {u['address']:#x} ({u['size']}B) "
                         f"by insn {u['pc']:#x} (side {u['side']})")
            L.append("")
    if d.get("notes"):
        L.append("Notes:")
        L += [f"  - {n}" for n in d["notes"]]
        L.append("")
    if d.get("ir"):
        L.append("Normalised IR diff (A -> B):")
        L += ["  " + ln for ln in _ir_diff(d["ir"])]
    return "\n".join(L)


def _check_status(rep: Report, check: str) -> str:
    d = rep.data
    if check in ("return", "memory", "calls", "preserve") and check_disabled(d, check):
        return "disabled"
    if rep.verdict in ("EQUIVALENT", "EQUIVALENT_BOUNDED"):
        return "equal" + (f" (loops up to {rep.bound} iterations)" if rep.bound else "")
    if rep.verdict == "NOT_EQUIVALENT":
        diffs = d.get("counterexample", {}).get("differences", [])
        hit = any(x.startswith(check) or (check == "preserve" and x.startswith("fault")) for x in diffs)
        return "DIFFERENT" if hit else "equal on the counterexample"
    if rep.verdict == "NO_DIFFERENCE_FOUND":
        return "no difference found (testing only)"
    return "unknown"


def check_disabled(d: dict, check: str) -> bool:
    labels = {"return": "return value", "memory": "memory writes", "calls": "external calls",
              "preserve": "callee-saved registers / stack delta"}
    return labels[check] in (d.get("checks_disabled") or [])
