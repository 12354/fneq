"""fneq command line.

    fneq A.bin:func_a B.bin:func_b [options]

A side is PATH:SYMBOL or PATH:0xADDRESS. Raw code blobs need --raw-base-a/-b and --arch.

    fneq batch PAIRS.toml [--jobs N] [--json]

compares every pair listed in PAIRS.toml ([[pair]] tables with a, b and optional cc,
ret, corr, map_by_symbols, backend) and prints a summary table; the exit code is the
worst per-pair exit code. With --suggest-corr CSV it also writes probable callee
correspondences, from callees called at the same place on both sides.

    fneq suggest-corr REPORT_DIR [-o CSV]

does the same for the reports of an earlier batch (--out-dir).
Exit codes: 0 equivalent (also bounded), 1 not equivalent, 2 unknown / no difference
found, 3 tool error (including a detected tool bug).
"""
from __future__ import annotations

import argparse
import json
import sys

from fneq import report, tools
from fneq.compare import Checks
from fneq.pipeline import Options, SideOpts, safe_analyse


def _side(text: str) -> tuple[str, str]:
    path, sep, func = text.rpartition(":")
    if not sep or not path or len(path) == 1:  # "C:" drive letters are not a separator
        raise argparse.ArgumentTypeError(f"expected PATH:FUNCTION, got {text!r}")
    return path, func


def _int(text: str) -> int:
    return int(text, 0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fneq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=_side, nargs="?", help="side A: PATH:FUNCTION")
    ap.add_argument("b", type=_side, nargs="?", help="side B: PATH:FUNCTION")
    ap.add_argument("--cc", help="calling convention for both sides "
                    "(cdecl, stdcall, fastcall, thiscall, sysv, ms); default: inferred")
    ap.add_argument("--cc-a", help="calling convention of side A")
    ap.add_argument("--cc-b", help="calling convention of side B")
    ap.add_argument("--ret", help="return kind: void, i8, i16, i32, i64, ptr, f32, f64, f80")
    ap.add_argument("--arch", choices=["x86", "x64"], help="architecture of raw blobs")
    ap.add_argument("--raw-base-a", type=_int, help="side A is a raw blob loaded at this address")
    ap.add_argument("--raw-base-b", type=_int, help="side B is a raw blob loaded at this address")
    ap.add_argument("--corr", help="correspondence file (JSON or TOML)")
    ap.add_argument("--map-by-symbols", action="store_true",
                    help="map globals present in both binaries by symbol name")
    ap.add_argument("--backend", choices=["auto", "lockstep", "alive", "angr", "fuzz"],
                    default="auto")
    ap.add_argument("--timeout", type=int, default=20, help="solver timeout per query (s)")
    ap.add_argument("--unroll", type=int, default=4, help="loop unroll bound k")
    ap.add_argument("--fuzz-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--x87-cw", type=_int, default=0x027F, help="pinned x87 control word")
    ap.add_argument("--mxcsr", type=_int, default=0x1F80, help="pinned MXCSR")
    ap.add_argument("--fuzz-time", type=float, default=60.0,
                    help="wall-clock budget for the whole testing campaign (s)")
    ap.add_argument("--lockstep-time", type=float, default=60.0,
                    help="wall-clock budget for the lockstep (per-segment) proof (s)")
    ap.add_argument("--max-insns", type=int, default=50_000,
                    help="instruction limit per concrete run")
    ap.add_argument("--no-alias", action="store_true",
                    help="assume pointer arguments do not alias each other or globals")
    ap.add_argument("--no-check-ret", action="store_true")
    ap.add_argument("--no-check-mem", action="store_true")
    ap.add_argument("--no-check-calls", action="store_true")
    ap.add_argument("--no-check-preserve", action="store_true")
    ap.add_argument("--write-order", action="store_true",
                    help="also require the same sequence of memory writes")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--keep", help="keep intermediate files (specs, IR) in this directory")
    ap.add_argument("--check-tools", action="store_true", help="verify pinned tool versions")
    return ap


_PAIR_FLAGS = (("cc", "--cc"), ("cc_a", "--cc-a"), ("cc_b", "--cc-b"), ("ret", "--ret"),
               ("corr", "--corr"), ("backend", "--backend"), ("arch", "--arch"))


def _pair_argv(p: dict) -> list[str]:
    argv = [p["a"], p["b"]]
    for key, flag in _PAIR_FLAGS:
        if key in p:
            argv += [flag, str(p[key])]
    if p.get("map_by_symbols"):
        argv.append("--map-by-symbols")
    return argv + [str(x) for x in p.get("args", [])]


def _batch_worker(conn) -> None:
    """Analyse pairs sent over `conn` until None arrives. Binaries, angr projects,
    bounded functions and the correspondence stay cached across pairs."""
    import time
    while True:
        task = conn.recv()
        if task is None:
            return
        index, argv = task
        t0 = time.monotonic()
        try:
            rep = safe_analyse(options_from_args(build_parser().parse_args(argv)))
            out = report.to_json(rep)
        except BaseException as e:  # a worker must always answer
            out = json.dumps({"verdict": "UNKNOWN", "reason": f"internal error: {e!r}", "backend": "fneq",
                              "exit_code": 3, "tool_error": repr(e)})
        conn.send((index, out, round(time.monotonic() - t0, 2)))


def _preload(pairs: list[dict]) -> None:
    """Load the binaries, their angr projects and the correspondence of every
    distinct pair of binaries once, before the workers are forked: they share them
    (copy-on-write) instead of each loading them again."""
    import gc
    import time
    from fneq import bounding, pipeline
    t0 = time.monotonic()
    done, n = set(), 0
    for p in pairs:
        try:
            opts = options_from_args(build_parser().parse_args(_pair_argv(p)))
        except SystemExit:
            continue
        key = (opts.a.path, opts.b.path, opts.corr, opts.map_by_symbols)
        if key in done:
            continue
        done.add(key)
        try:
            ba, bb = pipeline._load_binary(opts.a), pipeline._load_binary(opts.b)
            bounding.angr_project(ba)
            bounding.angr_project(bb)
            pipeline._load_corr(opts, ba, bb)
            n += 1
        except Exception as e:      # a pair that cannot load fails in its worker
            print(f"preload: {e}", file=sys.stderr)
    gc.collect()
    gc.freeze()     # keep the collector from touching (and so copying) shared objects
    print(f"preloaded {n} binary pair(s) in {time.monotonic() - t0:.1f}s", file=sys.stderr, flush=True)


def batch(argv: list[str]) -> int:
    """Compare many pairs in persistent worker processes with a per-pair time limit."""
    import multiprocessing as mp
    import os
    import re as _re
    import time
    import tomllib
    from multiprocessing.connection import wait

    ap = argparse.ArgumentParser(prog="fneq batch")
    ap.add_argument("pairs", help="TOML file with [[pair]] tables, or a JSON list of pairs")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=float, default=900, help="wall-clock limit per pair (s)")
    ap.add_argument("--out-dir", help="write each pair's full JSON report here")
    ap.add_argument("--jsonl", help="append one result line per pair as it finishes")
    ap.add_argument("--max-tasks", type=int, default=200, help="recycle a worker after N pairs")
    ap.add_argument("--json", action="store_true", help="print all results as JSON")
    ap.add_argument("--suggest-corr", metavar="CSV",
                    help="write probable callee correspondences (from aligned call sites) here")
    args = ap.parse_args(argv)
    if args.pairs.endswith(".json"):
        with open(args.pairs) as f:
            pairs = json.load(f)
    else:
        with open(args.pairs, "rb") as f:
            pairs = tomllib.load(f).get("pair", [])
    if not pairs:
        print(f"no pairs in {args.pairs}", file=sys.stderr)
        return 3
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    _preload(pairs)
    ctx = mp.get_context("fork")

    class Worker:
        def __init__(self):
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_batch_worker, args=(child,), daemon=True)
            self.proc.start()
            child.close()
            self.task: int | None = None
            self.started = 0.0
            self.done = 0

        def kill(self):
            self.proc.kill()
            self.proc.join()

    results: list[dict | None] = [None] * len(pairs)
    call_evidence: list = []
    # Longest first (by a pair's `expect_s`, e.g. its time in an earlier run; pairs
    # without one first of all): a slow pair started last would end the batch late.
    queue = sorted(range(len(pairs)), key=lambda i: -float(pairs[i].get("expect_s", float("inf"))))
    workers = [Worker() for _ in range(min(args.jobs, len(pairs)))]
    finished = 0

    def record(index: int, rep: dict, seconds: float) -> None:
        nonlocal finished
        p = pairs[index]
        name = p.get("name", f"{p['a']} ~ {p['b']}")
        res = {"name": name, "verdict": rep.get("verdict"), "backend": rep.get("backend"),
               "reason": rep.get("reason", ""), "exit": rep.get("exit_code", 3), "seconds": seconds}
        if args.out_dir:
            path = os.path.join(args.out_dir, _re.sub(r"[^\w.-]+", "_", name) + ".json")
            with open(path, "w") as f:
                json.dump(rep, f, indent=1, default=str)
            res["report"] = path
        results[index] = res
        if args.suggest_corr:
            from fneq import suggest
            call_evidence.append(suggest.evidence(rep))
        finished += 1
        if args.jsonl:
            with open(args.jsonl, "a") as f:
                f.write(json.dumps(res) + "\n")
        print(f"[{finished}/{len(pairs)}] {str(res['verdict']):<22} {seconds:>7.1f}s  {name}",
              file=sys.stderr, flush=True)

    def dispatch(w: Worker) -> None:
        if queue and w.task is None:
            w.task = queue.pop(0)
            w.started = time.monotonic()
            w.conn.send((w.task, _pair_argv(pairs[w.task])))

    for w in workers:
        dispatch(w)
    while any(w.task is not None for w in workers):
        busy = [w for w in workers if w.task is not None]
        now = time.monotonic()
        next_deadline = min(w.started + args.timeout for w in busy)
        ready = wait([w.conn for w in busy] + [w.proc.sentinel for w in busy],
                     timeout=max(0.0, next_deadline - now))
        now = time.monotonic()
        for i, w in enumerate(workers):
            if w.task is None:
                continue
            replaced = False
            if w.conn in ready:
                try:
                    index, out, seconds = w.conn.recv()
                    record(index, json.loads(out), seconds)
                    w.task = None
                    w.done += 1
                    if w.done >= args.max_tasks:
                        w.conn.send(None)
                        w.proc.join(timeout=10)
                        replaced = True
                except (EOFError, OSError):
                    pass
            if w.task is not None and (w.proc.sentinel in ready or not w.proc.is_alive()):
                record(w.task, {"verdict": "UNKNOWN", "reason": "tool error: worker process died",
                                "backend": "fneq", "exit_code": 3}, round(now - w.started, 2))
                w.task = None
                replaced = True
            elif w.task is not None and now >= w.started + args.timeout:
                w.kill()
                record(w.task, {"verdict": "TIMEOUT", "reason": f"no result within {args.timeout:g}s",
                                "backend": "fneq", "exit_code": 2}, round(now - w.started, 2))
                w.task = None
                replaced = True
            if replaced:
                if w.proc.is_alive():
                    w.kill()
                workers[i] = w = Worker()
            dispatch(w)
    for w in workers:
        try:
            w.conn.send(None)
        except OSError:
            pass
        w.proc.join(timeout=10)

    if args.suggest_corr:
        from fneq import suggest
        rows = suggest.suggest(call_evidence)
        suggest.write_csv(rows, args.suggest_corr)
        print(f"{len(rows)} callee correspondence suggestions -> {args.suggest_corr}",
              file=sys.stderr)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        width = max(len(r["name"]) for r in results)
        print(f"{'pair':{width}}  {'verdict':24} backend")
        for r in results:
            print(f"{r['name']:{width}}  {str(r['verdict']):24} {r['backend']}")
        counts: dict[str, int] = {}
        for r in results:
            key = str(r["verdict"]).split("(")[0]
            counts[key] = counts.get(key, 0) + 1
        print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return max(r["exit"] for r in results)


def options_from_args(args: argparse.Namespace) -> Options:
    checks = Checks(ret=not args.no_check_ret, memory=not args.no_check_mem,
                    calls=not args.no_check_calls, preserve=not args.no_check_preserve,
                    write_order=args.write_order)
    return Options(
        a=SideOpts(args.a[0], args.a[1], args.cc_a or args.cc, args.raw_base_a, args.arch),
        b=SideOpts(args.b[0], args.b[1], args.cc_b or args.cc, args.raw_base_b, args.arch),
        ret=args.ret, corr=args.corr, map_by_symbols=args.map_by_symbols, checks=checks,
        backend=args.backend, timeout_ms=args.timeout * 1000, unroll=args.unroll,
        fuzz_iters=args.fuzz_iters, seed=args.seed, x87_cw=args.x87_cw, mxcsr=args.mxcsr,
        max_insns=args.max_insns, fuzz_time=args.fuzz_time, no_alias=args.no_alias,
        lockstep_time=args.lockstep_time,
        keep_dir=args.keep)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "batch":
        return batch(argv[1:])
    if argv and argv[0] == "suggest-corr":
        from fneq import suggest
        return suggest.main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.check_tools:
        problems = tools.check_pins()
        print("\n".join(f"{k}: {v}" for k, v in tools.versions().items()))
        for p in problems:
            print("MISMATCH:", p)
        return 3 if problems else 0
    if not args.a or not args.b:
        build_parser().error("two functions are required")
    rep = safe_analyse(options_from_args(args))
    print(report.to_json(rep) if args.json else report.to_text(rep))
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
