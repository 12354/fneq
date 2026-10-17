"""Primary backend: lift with remill (fneq-lift), normalise with LLVM, prove with Alive2."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile

from fneq import tools
from fneq.binary import Binary
from fneq.bounding import FunctionInfo
from fneq.compare import Checks
from fneq.errors import ToolError
from fneq.model import RET_ADDR, STACK_TOP, CalleeSpec
from fneq.signature import Signature

# The spec's required passes, then: full unrolling of loops with a constant trip
# count (exact, and loop-free code is much easier for Alive2), reassociate (sorts
# operands of commutative operations by a stable rank), and div-rem-pairs to undo
# remill's inconsistent expansion of remainders (x - y*(x/y)), which is very hard
# for the solver. All of these are semantics-preserving LLVM passes.
NORMALISE = ("function(mem2reg,sroa,instcombine,simplifycfg,early-cse,"
             "loop-simplify,lcssa,loop(indvars),"
             "loop-unroll<O2;no-partial;no-peeling;no-runtime;no-upperbound;full-unroll-max=1024>,"
             "instcombine,simplifycfg,div-rem-pairs,reassociate,instcombine,early-cse,simplifycfg)")
MAX_RODATA_WINDOW = 0x4000


@dataclasses.dataclass
class SolverResult:
    status: str                    # equivalent | bounded | different | unknown
    reason: str
    bound: int | None = None
    counterexample: dict | None = None   # {"inputs": {...}} (slot name -> value)
    textual_identical: bool = False
    ir_a: str = ""
    ir_b: str = ""
    log: str = ""


def _rodata(binary: Binary, info: FunctionInfo) -> list[dict]:
    """Read-only data the function references: constant tables for the lowering."""
    chunks: dict[int, bytes] = {}
    for ref in sorted(info.data_refs):
        sec = binary.section_at(ref)
        if sec is None or sec.writable or sec.executable:
            continue
        if sec.vsize <= MAX_RODATA_WINDOW:
            start, end = sec.addr, sec.end
        else:
            start = max(sec.addr, ref - 0x100)
            end = min(sec.end, ref + MAX_RODATA_WINDOW)
        data = binary.read(start, end - start)
        if data is not None:
            chunks[start] = data
    # Merge overlaps.
    merged: list[list] = []
    for start in sorted(chunks):
        data = chunks[start]
        if merged and start <= merged[-1][0] + len(merged[-1][1]):
            prev_start, prev = merged[-1]
            end = max(prev_start + len(prev), start + len(data))
            merged[-1][1] = binary.read(prev_start, end - prev_start)
        else:
            merged.append([start, data])
    return [{"addr": s, "hex": d.hex()} for s, d in merged]


def lifter_spec(name: str, binary: Binary, info: FunctionInfo, sig: Signature,
                callees: dict[int, CalleeSpec], remap: list[tuple[int, int, int]]) -> dict:
    word = sig.word
    return {
        "arch": "x86" if sig.arch == "x86" else "amd64",
        "os": binary.os,
        "name": name,
        "entry": info.entry,
        "code": [{"addr": a, "hex": d.hex()} for a, d in sorted(info.code.items())],
        "rodata": _rodata(binary, info),
        "in_regs": [s.name for s in sig.inputs if s.kind == "reg"],
        "stack_args": [{"offset": s.offset, "size": s.size} for s in sig.inputs if s.kind == "stack"],
        "out_regs": sig.outputs,
        "esp0": STACK_TOP[sig.arch],
        "ret_addr": RET_ADDR[sig.arch],
        "frame_below": sig.frame_below,
        "frame_above": sig.frame_above,
        "callees": [c.to_lifter(word) for c in callees.values()],
        "remap": [{"from": here, "to": a, "size": size} for here, a, size in remap if here != a],
    }


def lift(spec: dict, workdir: str, tag: str) -> str:
    spec_path = os.path.join(workdir, f"{tag}.spec.json")
    out = os.path.join(workdir, f"{tag}.ll")
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    p = tools.run(["fneq-lift", "--spec", spec_path, "--out", out], ok_codes=(0, 4), timeout=600)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise ToolError(f"fneq-lift produced no diagnostics (stderr: {p.stderr[-800:]})")
    diag = json.loads(lines[-1])
    if not diag.get("ok"):
        reasons = diag.get("unsupported") or [diag.get("error", "unknown lifter failure")]
        raise _Unsupported("; ".join(reasons))
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise ToolError("fneq-lift reported success but wrote no IR")
    return out


class _Unsupported(Exception):
    pass


def normalise(ll: str, workdir: str, tag: str) -> str:
    out = os.path.join(workdir, f"{tag}.norm.ll")
    tools.run(["opt", "-S", f"-passes={NORMALISE}", ll, "-o", out], timeout=300)
    text = open(out).read()
    if "define " not in text:
        raise ToolError("opt produced IR without a function definition")
    return finalize(out, workdir, tag)


def finalize(ll: str, workdir: str, tag: str) -> str:
    """Rewrite fault markers to `unreachable` after the last LLVM pass (fneq-lift --finalize).

    The lifter keeps faults as an opaque call so that optimisations cannot exploit
    them; Alive2 must see them as UB (see the lifter's comment on fault_exit).
    """
    out = os.path.join(workdir, f"{tag}.final.ll")
    p = tools.run(["fneq-lift", "--finalize", ll, "--out", out], ok_codes=(0, 4), timeout=120)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines or not json.loads(lines[-1]).get("ok"):
        raise ToolError("fneq-lift --finalize failed: " + (lines[-1] if lines else p.stderr[-800:]))
    if "define " not in open(out).read():
        raise ToolError("finalized IR has no function definition")
    return out


def ir_has_loops(ll: str) -> bool:
    """Does the (normalised) IR still contain a loop?"""
    p = tools.run(["opt", "-passes=print<loops>", "-disable-output", ll], timeout=120)
    return "Loop at depth" in (p.stdout + p.stderr)


def function_text(ll: str, name: str) -> str:
    text = open(ll).read()
    m = re.search(r"define [^@]*@" + re.escape(name) + r"\(.*?\n}\n", text, re.S)
    if not m:
        raise ToolError(f"function @{name} not found in {ll}")
    return m.group(0)


def _parse_alive(out: str) -> tuple[str, str]:
    """('correct'|'incorrect'|'unknown', detail) from alive-tv output for one pair.

    alive-tv exits 0 whatever the outcome, so the outcome is recognised from the text;
    output that matches none of the known outcomes is a tool error.
    """
    errors = re.findall(r"^ERROR: (.*)$", out, re.M)
    if "doesn't type check" in out:
        # A malformed comparison (our lowering), never a verdict about the functions.
        raise ToolError("Alive2: the lowered functions do not type-check")
    if "Transformation doesn't verify!" in out:
        return "incorrect", errors[0] if errors else "transformation does not verify"
    if "Transformation seems to be correct!" in out:
        if errors:
            return "unknown", errors[0]
        return "correct", ""
    if errors or "Couldn't prove the correctness" in out or "approximat" in out.lower():
        return "unknown", errors[0] if errors else "Alive2 could not prove the transformation"
    raise ToolError("unrecognised alive-tv output: " + out[-1500:])


_CEX_LINE = re.compile(r"^\s*(\S+) %([\w.]+) = (?:#x([0-9a-fA-F]+)|#b([01]+)|(-?\d+))", re.M)


def _parse_cex(out: str) -> dict[str, int]:
    m = re.search(r"Example:\n(.*?)(?:\n\n|\nSource:)", out, re.S)
    if not m:
        return {}
    vals = {}
    for mm in _CEX_LINE.finditer(m.group(1)):
        name = mm.group(2)
        if mm.group(3):
            vals[name] = int(mm.group(3), 16)
        elif mm.group(4):
            vals[name] = int(mm.group(4), 2)
        elif mm.group(5):
            vals[name] = int(mm.group(5))
    return vals


def check(sig: Signature, side_a: tuple, side_b: tuple, checks: Checks,
          timeout_ms: int, unroll: int, machine_loops: bool, keep_dir: str | None = None) -> SolverResult:
    """side_x = (binary, info, callees, remap)."""
    workdir = keep_dir or tempfile.mkdtemp(prefix="fneq-alive-")
    os.makedirs(workdir, exist_ok=True)
    try:
        ll_a = normalise(lift(lifter_spec("src", *side_a[:2], sig, side_a[2], side_a[3]), workdir, "a"), workdir, "a")
        ll_b = normalise(lift(lifter_spec("tgt", *side_b[:2], sig, side_b[2], side_b[3]), workdir, "b"), workdir, "b")
    except _Unsupported as e:
        return SolverResult("unknown", f"lifting: {e}")
    fa, fb = function_text(ll_a, "src"), function_text(ll_b, "tgt")
    # Loops that normalisation fully unrolled (constant trip count) need no bound.
    has_loops = ir_has_loops(ll_a) or ir_has_loops(ll_b)
    identical = fa.replace("@src(", "@F(") == fb.replace("@tgt(", "@F(")

    if not (checks.ret and checks.memory and checks.calls and checks.preserve):
        return SolverResult("unknown", "Alive2 checks all effects at once; disabled checks need the angr backend",
                            textual_identical=identical, ir_a=fa, ir_b=fb)
    if checks.write_order:
        return SolverResult("unknown", "Alive2 cannot check write order; using the angr backend",
                            textual_identical=identical, ir_a=fa, ir_b=fb)

    linked = os.path.join(workdir, "linked.ll")
    tools.run(["llvm-link", "-S", ll_a, ll_b, "-o", linked], timeout=120)
    base = ["alive-tv", "--disable-undef-input", "--disable-poison-input", f"--smt-to={timeout_ms}"]
    if has_loops:
        base += [f"--src-unroll={unroll}", f"--tgt-unroll={unroll}"]
    # Both directions always run (in parallel): one direction may time out while
    # the other finds a counterexample at once.
    import concurrent.futures as cf

    def direction(src: str, tgt: str):
        p = tools.run(base + [f"--src-fn={src}", f"--tgt-fn={tgt}", linked],
                      timeout=max(60, 6 * timeout_ms / 1000), ok_codes=(0, 1))
        out = p.stdout + p.stderr
        return src, tgt, out, _parse_alive(out)

    with cf.ThreadPoolExecutor(2) as ex:
        results = list(ex.map(lambda st: direction(*st), (("src", "tgt"), ("tgt", "src"))))
    logs = [r[2] for r in results]
    log = "\n".join(logs)
    for src, tgt, out, (status, detail) in results:
        if status == "incorrect":
            return SolverResult("different", f"Alive2 ({src} => {tgt}): {detail}",
                                counterexample={"inputs": _parse_cex(out)},
                                textual_identical=identical, ir_a=fa, ir_b=fb, log=log)
    unknown = [(s, t, d) for s, t, _, (st, d) in results if st == "unknown"]
    if unknown:
        if identical:
            # Identical lowered IR is equivalent by construction (same function).
            return SolverResult("equivalent", "normalised IR of both functions is identical "
                                f"(Alive2 itself did not finish: {unknown[0][2]})",
                                textual_identical=True, ir_a=fa, ir_b=fb, log=log)
        reason = "; ".join(f"{s} => {t}: {d}" for s, t, d in unknown)
        if "timeout" in reason.lower() and any(k in fa + fb for k in ("double", "float", "x86_fp80")):
            reason += " (floating-point arithmetic made the solver time out)"
        return SolverResult("unknown", f"Alive2: {reason}", textual_identical=identical,
                            ir_a=fa, ir_b=fb, log=log)
    if has_loops:
        return SolverResult("bounded", f"Alive2 proved both directions with loops unrolled {unroll} times",
                            bound=unroll, textual_identical=identical, ir_a=fa, ir_b=fb, log="\n".join(logs))
    return SolverResult("equivalent", "Alive2 proved refinement in both directions",
                        textual_identical=identical, ir_a=fa, ir_b=fb, log="\n".join(logs))
