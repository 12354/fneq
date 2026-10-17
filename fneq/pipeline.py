"""Orchestration: load, bound, model, run the backends, decide the verdict."""
from __future__ import annotations

import dataclasses
import time

from fneq import alive_backend, binary as binmod, bounding, callees as calleemod
from fneq import correspondence, fuzz, signature
from fneq.compare import DEFINITION, Checks, compare
from fneq.errors import FneqError, InputError, ToolError, Unsupported
from fneq.harness import Environment, run
from fneq.model import DEFAULT_MXCSR, DEFAULT_X87_CW

VERDICT_EXIT = {"EQUIVALENT": 0, "EQUIVALENT_BOUNDED": 0, "NOT_EQUIVALENT": 1,
                "NO_DIFFERENCE_FOUND": 2, "UNKNOWN": 2}


@dataclasses.dataclass
class SideOpts:
    path: str
    func: str
    cc: str | None = None
    raw_base: int | None = None
    arch: str | None = None


@dataclasses.dataclass
class Options:
    a: SideOpts
    b: SideOpts
    ret: str | None = None
    corr: str | None = None
    map_by_symbols: bool = False
    checks: Checks = dataclasses.field(default_factory=Checks)
    backend: str = "auto"            # auto | alive | angr | fuzz
    timeout_ms: int = 20000
    unroll: int = 4
    fuzz_iters: int = 2000
    sanity_iters: int = 300
    seed: int = 1
    x87_cw: int = DEFAULT_X87_CW
    mxcsr: int = DEFAULT_MXCSR
    max_insns: int = 50_000
    fuzz_time: float = 60.0
    no_alias: bool = False
    keep_dir: str | None = None


@dataclasses.dataclass
class Report:
    verdict: str
    reason: str
    backend: str
    bound: int | None = None
    data: dict = dataclasses.field(default_factory=dict)

    @property
    def verdict_label(self) -> str:
        return f"{self.verdict}({self.bound})" if self.verdict == "EQUIVALENT_BOUNDED" else self.verdict

    @property
    def exit_code(self) -> int:
        if self.data.get("tool_bug") or self.data.get("tool_error"):
            return 3
        return VERDICT_EXIT[self.verdict]


class _Ctx:
    """Everything prepared for both sides."""

    def __init__(self, opts: Options):
        self.opts = opts
        self.data: dict = {"definition": DEFINITION, "inputs": {}, "assumptions": [],
                           "checks_disabled": opts.checks.disabled(), "backends": [],
                           "notes": []}


def _side_inputs(side: SideOpts, b: binmod.Binary, entry: int, fi) -> dict:
    return {"binary": side.path, "format": b.fmt, "arch": b.arch, "function": side.func,
            "address": hex(entry), "symbol": b.symbol_at(entry), "size": fi.size,
            "blocks": len(fi.code), "has_loop": fi.has_loop,
            "calls": [{"at": hex(c.addr), "callee": c.name, "tail": c.tail} for c in fi.calls]}


def analyse(opts: Options) -> Report:
    ctx = _Ctx(opts)
    d = ctx.data
    try:
        return _analyse(ctx)
    except Unsupported as e:
        return Report("UNKNOWN", str(e), "frontend", data=d)


def _analyse(ctx: _Ctx) -> Report:
    opts, d = ctx.opts, ctx.data
    ba = binmod.load(opts.a.path, opts.a.raw_base, opts.a.arch)
    bb = binmod.load(opts.b.path, opts.b.raw_base, opts.b.arch)
    if ba.arch != bb.arch:
        raise Unsupported(f"architectures differ ({ba.arch} vs {bb.arch}); "
                          "cross-architecture comparison is not supported in v1")
    arch = ba.arch
    ea, eb = ba.resolve(opts.a.func), bb.resolve(opts.b.func)
    fa, fb = bounding.bound(ba, ea), bounding.bound(bb, eb)
    d["inputs"] = {"A": _side_inputs(opts.a, ba, ea, fa), "B": _side_inputs(opts.b, bb, eb, fb)}
    d["notes"] += [f"A: {n}" for n in fa.notes] + [f"B: {n}" for n in fb.notes]

    hazards = [f"A: {h}" for h in fa.hazards] + [f"B: {h}" for h in fb.hazards]
    if hazards:
        d["hazards"] = hazards
        return Report("UNKNOWN", "; ".join(hazards[:5]), "frontend", data=d)

    inf_a = signature.infer_convention(fa, ba.fmt, opts.a.cc)
    inf_b = signature.infer_convention(fb, bb.fmt, opts.b.cc)
    d["conventions"] = {
        "A": {"name": inf_a.convention.name, "inferred": inf_a.inferred, "evidence": inf_a.evidence},
        "B": {"name": inf_b.convention.name, "inferred": inf_b.inferred, "evidence": inf_b.evidence}}
    if inf_a.convention != inf_b.convention:
        raise Unsupported(f"calling conventions differ (A {inf_a.convention.name}, "
                          f"B {inf_b.convention.name}); override with --cc-a/--cc-b if inference is wrong")
    cc = inf_a.convention
    ret_kind, ret_why = signature.infer_return(fa, fb, arch, opts.ret)
    d["return"] = {"kind": ret_kind, "how": ret_why}
    sig = signature.build_signature(arch, cc, ret_kind, [fa, fb])
    d["signature"] = sig.describe()

    corr = correspondence.load(opts.corr, ba, bb, opts.map_by_symbols).writable_only(bb)
    d["correspondence"] = corr.describe()
    d["notes"] += corr.notes
    ca, na = calleemod.callee_specs(ba, fa, cc, corr.overrides, corr.names_a)
    cb, nb = calleemod.callee_specs(bb, fb, cc, corr.overrides, corr.names_b)
    d["callees"] = {"A": {hex(k): dataclasses.asdict(v) for k, v in ca.items()},
                    "B": {hex(k): dataclasses.asdict(v) for k, v in cb.items()}}
    d["notes"] += [f"A: {n}" for n in na] + [f"B: {n}" for n in nb]

    d["assumptions"] += [
        "memory is flat; every address outside the image is readable/writable and holds "
        "arbitrary initial contents (writable globals included)",
        "pointer arguments may alias each other and globals" if not opts.no_alias else
        "--no-alias: counterexamples in which an argument points into another argument's "
        "or a global's region are discarded",
        "pointers that do not derive from the stack pointer never point into the callee's own frame",
        f"stack frame modelled as [SP-{sig.frame_below}, SP+{sig.frame_above}); its final contents "
        "are not observable",
        "status flags are 0 and DF=0 at entry; the x87 stack is empty",
        f"x87 control word pinned to {opts.x87_cw:#06x}, MXCSR pinned to {opts.mxcsr:#06x}",
        "external callees are uninterpreted functions of their arguments (and the memory they see); "
        "their arity is inferred from their code when available",
    ]
    if fa.x87_arith or fb.x87_arith:
        d["assumptions"].append("x87 arithmetic present: transcendental x87 instructions are "
                                "uninterpreted; precision control is only modelled concretely (Unicorn)")

    env = Environment(x87_cw=opts.x87_cw, mxcsr=opts.mxcsr, max_insns=opts.max_insns)
    side_a = fuzz.Side(ba, fa, ca, corr.regions_a())
    side_b = fuzz.Side(bb, fb, cb, corr.regions_b())
    ctx.sig, ctx.env, ctx.side_a, ctx.side_b, ctx.corr = sig, env, side_a, side_b, corr

    backends = {"auto": ["alive", "angr", "fuzz"], "alive": ["alive", "fuzz"],
                "angr": ["angr", "fuzz"], "fuzz": ["fuzz"]}[opts.backend]
    unknown_reasons = []
    for name in backends:
        t0 = time.monotonic()
        if name == "fuzz":
            rep = _run_fuzz(ctx, opts.fuzz_iters, unknown_reasons)
            d["backends"].append({"backend": "unicorn-fuzz", "seconds": round(time.monotonic() - t0, 2),
                                  "result": rep.verdict})
            return rep
        try:
            res = _run_solver(ctx, name)
        except ToolError as e:
            d["backends"].append({"backend": name, "result": "tool error", "reason": str(e)})
            raise
        d["backends"].append({"backend": name, "seconds": round(time.monotonic() - t0, 2),
                              "result": res.status, "reason": res.reason})
        if res.status == "unknown":
            unknown_reasons.append(f"{name}: {res.reason}")
            continue
        if res.status in ("equivalent", "bounded"):
            bug = _sanity_fuzz(ctx, res, name)
            if bug:
                return bug
            if res.status == "equivalent":
                return Report("EQUIVALENT", res.reason, name, data=d)
            return Report("EQUIVALENT_BOUNDED", res.reason, name, bound=res.bound, data=d)
        return _confirm(ctx, res, name)
    return Report("UNKNOWN", "; ".join(unknown_reasons), backends[-1], data=d)


def _run_solver(ctx: _Ctx, name: str):
    opts, d = ctx.opts, ctx.data
    sa = (ctx.side_a.binary, ctx.side_a.info, ctx.side_a.callees, [])
    sb = (ctx.side_b.binary, ctx.side_b.info, ctx.side_b.callees, ctx.corr.regions_b())
    loops = ctx.side_a.info.has_loop or ctx.side_b.info.has_loop
    if name == "alive":
        res = alive_backend.check(ctx.sig, sa, sb, opts.checks, opts.timeout_ms, opts.unroll,
                                  loops, opts.keep_dir)
        if res.ir_a:
            d["ir"] = {"A": res.ir_a, "B": res.ir_b, "textually_identical": res.textual_identical}
        return res
    if name == "angr":
        try:
            from fneq import angr_backend
        except ImportError:
            return alive_backend.SolverResult("unknown", "angr backend not available")
        return angr_backend.check(ctx.sig, ctx.side_a, ctx.side_b, opts.checks, ctx.env,
                                  opts.timeout_ms, opts.unroll, ctx.corr.map_b)
    raise InputError(f"unknown backend {name}")


def _slot_inputs(ctx: _Ctx, raw: dict[str, int]) -> dict[str, int]:
    """Map solver variable names (eax, stk_4, xmm0, ...) to signature slot names."""
    names = {s.name.lower(): s.name for s in ctx.sig.inputs}
    return {names[k.lower()]: v for k, v in raw.items() if k.lower() in names}


def _replay(ctx: _Ctx, inputs: dict[str, int], memory: dict[int, int], seed: int):
    ea = Environment(**{**dataclasses.asdict(ctx.env), "memory_seed": seed, "memory": memory,
                        "regions": ctx.side_a.regions})
    eb = Environment(**{**dataclasses.asdict(ctx.env), "memory_seed": seed,
                        "memory": {_inv(ctx, a): v for a, v in memory.items()},
                        "regions": ctx.side_b.regions})
    ra = run(ctx.side_a.binary, ctx.side_a.info.entry, ctx.sig, inputs, ctx.side_a.callees, ea,
             ctx.side_a.info.code)
    rb = run(ctx.side_b.binary, ctx.side_b.info.entry, ctx.sig, inputs, ctx.side_b.callees, eb,
             ctx.side_b.info.code)
    return ra, rb, compare(ctx.sig, ra, rb, ctx.opts.checks, ctx.corr.map_b)


def _inv(ctx: _Ctx, a: int) -> int:
    for r in ctx.corr.regions:
        if r.a <= a < r.a + r.size:
            return r.b + (a - r.a)
    return a


def _shrink(ctx: _Ctx, inputs: dict[str, int], memory: dict[int, int], seed: int,
            budget: int = 300) -> dict[str, int]:
    """Simplify a confirmed counterexample: every candidate is re-checked by replay."""
    cur = dict(inputs)
    tries = 0

    def still_differs(cand: dict[str, int]) -> bool:
        nonlocal tries
        tries += 1
        return bool(_replay(ctx, cand, memory, seed)[2])

    for name in list(cur):
        if tries >= budget:
            break
        v = cur[name]
        if v == 0:
            continue
        for cand in (0, 1, v & 0xFF, v & 0xFFFF, v & 0xFFFFFFFF):
            if cand >= v or tries >= budget:
                continue
            trial = dict(cur, **{name: cand})
            if still_differs(trial):
                cur = trial
                break
        else:
            # Halve towards zero while the difference persists.
            while cur[name] > 1 and tries < budget:
                trial = dict(cur, **{name: cur[name] >> 1})
                if not still_differs(trial):
                    break
                cur = trial
    return cur


def _cex_report(ctx: _Ctx, inputs, memory, seed, ra, rb, diffs) -> dict:
    return {"inputs": {k: hex(v) for k, v in inputs.items()},
            "memory": {hex(k): v for k, v in sorted(memory.items())},
            "memory_seed": seed,
            "first_difference": str(diffs[0]),
            "differences": [str(x) for x in diffs],
            "A": ra.summary(), "B": rb.summary(),
            "unmapped_accesses": [dict(u, side="A") for u in ra.unmapped[:10]]
                                 + [dict(u, side="B") for u in rb.unmapped[:10]]}


def _alias_violation(ctx: _Ctx, inputs: dict[str, int]) -> bool:
    """With --no-alias: do two argument values point into the same 4 KiB region?"""
    args = [inputs.get(s.name, 0) for s in ctx.sig.inputs
            if s.name in ctx.sig.convention.int_args or s.kind == "stack"]
    pages = [a >> 12 for a in args if a > 0x10000]
    return len(pages) != len(set(pages))


def _confirm(ctx: _Ctx, res, backend: str) -> Report:
    """Replay a solver counterexample concretely; never report it unconfirmed."""
    d = ctx.data
    cex = res.counterexample or {}
    inputs = _slot_inputs(ctx, cex.get("inputs", {}))
    memory = cex.get("memory", {})
    tried = []
    for seed in [0] + list(range(1, 64)):
        ra, rb, diffs = _replay(ctx, inputs, memory, seed)
        tried.append(seed)
        if diffs:
            if ctx.opts.no_alias and _alias_violation(ctx, inputs):
                d["notes"].append("solver counterexample discarded: it violates --no-alias")
                break
            original = dict(inputs)
            inputs = _shrink(ctx, inputs, memory, seed)
            ra, rb, diffs = _replay(ctx, inputs, memory, seed)
            d["counterexample"] = _cex_report(ctx, inputs, memory, seed, ra, rb, diffs)
            d["counterexample"]["source"] = backend
            if inputs != original:
                d["counterexample"]["unshrunk_inputs"] = {k: hex(v) for k, v in original.items()}
            if seed:
                d["counterexample"]["note"] = ("the solver did not report memory contents; the "
                                               f"difference reproduces with memory seed {seed}")
            return Report("NOT_EQUIVALENT", f"{res.reason}; confirmed by concrete replay: {diffs[0]}",
                          backend, data=d)
    # Not reproduced: a tool bug by definition. Report it as such, and still try to
    # give the user an answer by testing.
    d["tool_bug"] = (f"{backend} reported a counterexample that does not reproduce concretely "
                     f"(inputs {', '.join(f'{k}={v:#x}' for k, v in inputs.items())}; "
                     f"memory seeds tried: {len(tried)})")
    return Report("UNKNOWN", "solver counterexample did not reproduce (tool bug, see report)",
                  backend, data=d)


def _probe_switches(ctx: _Ctx, proven: bool) -> dict:
    """Which expensive probe families can still find something.

    After a full proof the cross-check only guards against tool bugs, so it stays
    light. Pairwise float probes need floating point in either function; pointer /
    length probes (buffers crossing the end of the address space) need a loop that
    dereferences a pointer, and no full proof.
    """
    a, b = ctx.side_a.info, ctx.side_b.info
    return {
        "slot_probes": not proven,
        "uses_fp": a.uses_fp or b.uses_fp,
        "float_pair_probes": not proven and (a.uses_fp or b.uses_fp),
        "pointer_probes": not proven and (a.has_loop or b.has_loop)
                          and (a.pointer_access or b.pointer_access),
    }


def _sanity_fuzz(ctx: _Ctx, res, backend: str) -> Report | None:
    """Cross-check of a proof by testing; for a full proof a difference is a tool bug."""
    fr = fuzz.fuzz(ctx.sig, ctx.side_a, ctx.side_b, ctx.opts.checks, ctx.env,
                   iterations=ctx.opts.sanity_iters, seed=ctx.opts.seed, map_b=ctx.corr.map_b,
                   **_probe_switches(ctx, proven=res.status == "equivalent"))
    ctx.data["cross_check"] = {"backend": "unicorn-fuzz", "runs": fr.runs,
                               "coverage": {"A": fuzz.coverage_report(ctx.side_a.info, fr.coverage_a),
                                            "B": fuzz.coverage_report(ctx.side_b.info, fr.coverage_b)}}
    if fr.counterexample is None:
        return None
    if res.status == "bounded":
        # Not a contradiction: the proof only covers loops of up to k iterations.
        # The concrete difference (replayed below) is the answer.
        inputs, seed = fr.counterexample["inputs"], fr.counterexample["memory_seed"]
        ra, rb, diffs = _replay(ctx, inputs, {}, seed)
        if diffs:
            inputs = _shrink(ctx, inputs, {}, seed)
            ra, rb, diffs = _replay(ctx, inputs, {}, seed)
            ctx.data["counterexample"] = _cex_report(ctx, inputs, {}, seed, ra, rb, diffs)
            ctx.data["counterexample"]["source"] = "unicorn-fuzz (beyond the unroll bound)"
            return Report("NOT_EQUIVALENT",
                          f"{res.reason}, but concrete testing found a difference for longer "
                          f"executions: {diffs[0]}", "unicorn-fuzz", data=ctx.data)
    ctx.data["tool_bug"] = (f"{backend} proved equivalence but concrete testing found a difference: "
                            f"{fr.diffs[0]}")
    ctx.data["counterexample"] = _cex_report(ctx, fr.counterexample["inputs"], {},
                                             fr.counterexample["memory_seed"], fr.result_a,
                                             fr.result_b, fr.diffs)
    return Report("UNKNOWN", "backends disagree (tool bug, see report)", backend, data=ctx.data)


def _run_fuzz(ctx: _Ctx, iters: int, unknown_reasons: list[str]) -> Report:
    d = ctx.data
    fr = fuzz.fuzz(ctx.sig, ctx.side_a, ctx.side_b, ctx.opts.checks, ctx.env,
                   iterations=iters, seed=ctx.opts.seed, map_b=ctx.corr.map_b,
                   time_budget=ctx.opts.fuzz_time, **_probe_switches(ctx, proven=False))
    cov = {"A": fuzz.coverage_report(ctx.side_a.info, fr.coverage_a),
           "B": fuzz.coverage_report(ctx.side_b.info, fr.coverage_b)}
    d["testing"] = {"runs": fr.runs, "coverage": cov, "runs_hit_instruction_limit": fr.exhausted_runs,
                    "unmapped_accesses": fr.unmapped[:20]}
    why = ("; ".join(unknown_reasons) + "; ") if unknown_reasons else ""
    if fr.counterexample:
        inputs = fr.counterexample["inputs"]
        if ctx.opts.no_alias and _alias_violation(ctx, inputs):
            d["notes"].append("a difference found by testing violates --no-alias and was discarded")
        else:
            seed = fr.counterexample["memory_seed"]
            ra, rb, diffs = _replay(ctx, inputs, {}, seed)
            if not diffs:
                d["tool_bug"] = "a difference found by testing did not reproduce on replay"
                return Report("UNKNOWN", "non-deterministic testing result (tool bug)", "unicorn-fuzz", data=d)
            original = dict(inputs)
            inputs = _shrink(ctx, inputs, {}, seed)
            ra, rb, diffs = _replay(ctx, inputs, {}, seed)
            d["counterexample"] = _cex_report(ctx, inputs, {}, seed, ra, rb, diffs)
            if inputs != original:
                d["counterexample"]["unshrunk_inputs"] = {k: hex(v) for k, v in original.items()}
            d["counterexample"]["source"] = "unicorn-fuzz"
            return Report("NOT_EQUIVALENT", f"{why}testing found a difference: {diffs[0]}",
                          "unicorn-fuzz", data=d)
    return Report("NO_DIFFERENCE_FOUND",
                  f"{why}no difference in {fr.runs} concrete runs "
                  f"(coverage A {cov['A']['percent']}%, B {cov['B']['percent']}%)",
                  "unicorn-fuzz", data=d)


def safe_analyse(opts: Options) -> Report:
    """analyse() that turns tool/input errors into an error report (exit code 3)."""
    try:
        return analyse(opts)
    except (InputError, ToolError) as e:
        return Report("UNKNOWN", f"{type(e).__name__}: {e}", "fneq", data={"tool_error": str(e)})
    except FneqError as e:
        return Report("UNKNOWN", str(e), "fneq", data={"tool_error": str(e)})
    except Exception as e:  # an internal error must never pass silently
        import traceback
        return Report("UNKNOWN", f"internal error: {type(e).__name__}: {e}", "fneq",
                      data={"tool_error": f"internal error: {type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()})
