"""fneq command line.

    fneq A.bin:func_a B.bin:func_b [options]

A side is PATH:SYMBOL or PATH:0xADDRESS. Raw code blobs need --raw-base-a/-b and --arch.

    fneq batch PAIRS.toml [--jobs N] [--json]

compares every pair listed in PAIRS.toml ([[pair]] tables with a, b and optional cc,
ret, corr, map_by_symbols, backend) and prints a summary table; the exit code is the
worst per-pair exit code.
Exit codes: 0 equivalent (also bounded), 1 not equivalent, 2 unknown / no difference
found, 3 tool error (including a detected tool bug).
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--backend", choices=["auto", "alive", "angr", "fuzz"], default="auto")
    ap.add_argument("--timeout", type=int, default=20, help="solver timeout per query (s)")
    ap.add_argument("--unroll", type=int, default=4, help="loop unroll bound k")
    ap.add_argument("--fuzz-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--x87-cw", type=_int, default=0x027F, help="pinned x87 control word")
    ap.add_argument("--mxcsr", type=_int, default=0x1F80, help="pinned MXCSR")
    ap.add_argument("--fuzz-time", type=float, default=60.0,
                    help="wall-clock budget for the whole testing campaign (s)")
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


def batch(argv: list[str]) -> int:
    import concurrent.futures as cf
    import json
    import os
    import subprocess
    import tomllib

    ap = argparse.ArgumentParser(prog="fneq batch")
    ap.add_argument("pairs", help="TOML file with [[pair]] tables")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    with open(args.pairs, "rb") as f:
        pairs = tomllib.load(f).get("pair", [])
    if not pairs:
        print(f"no [[pair]] entries in {args.pairs}", file=sys.stderr)
        return 3

    def one(p: dict) -> dict:
        cmd = [sys.executable, "-m", "fneq", p["a"], p["b"], "--json"]
        for key, flag in (("cc", "--cc"), ("cc_a", "--cc-a"), ("cc_b", "--cc-b"), ("ret", "--ret"),
                          ("corr", "--corr"), ("backend", "--backend"), ("arch", "--arch")):
            if key in p:
                cmd += [flag, str(p[key])]
        if p.get("map_by_symbols"):
            cmd.append("--map-by-symbols")
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            rep = json.loads(r.stdout)
        except json.JSONDecodeError:
            rep = {"verdict": "UNKNOWN", "reason": "tool error: " + (r.stderr or r.stdout)[-300:]}
        return {"name": p.get("name", f"{p['a']} ~ {p['b']}"), "verdict": rep.get("verdict"),
                "backend": rep.get("backend"), "reason": rep.get("reason", ""), "exit": r.returncode}

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(one, pairs))
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        w = max(len(r["name"]) for r in results)
        print(f"{'pair':{w}}  {'verdict':24} backend")
        for r in results:
            print(f"{r['name']:{w}}  {str(r['verdict']):24} {r['backend']}")
        counts: dict[str, int] = {}
        for r in results:
            key = str(r["verdict"]).split("(")[0]
            counts[key] = counts.get(key, 0) + 1
        print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return max(r["exit"] for r in results)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "batch":
        return batch(argv[1:])
    args = build_parser().parse_args(argv)
    if args.check_tools:
        problems = tools.check_pins()
        print("\n".join(f"{k}: {v}" for k, v in tools.versions().items()))
        for p in problems:
            print("MISMATCH:", p)
        return 3 if problems else 0
    if not args.a or not args.b:
        build_parser().error("two functions are required")
    checks = Checks(ret=not args.no_check_ret, memory=not args.no_check_mem,
                    calls=not args.no_check_calls, preserve=not args.no_check_preserve,
                    write_order=args.write_order)
    opts = Options(
        a=SideOpts(args.a[0], args.a[1], args.cc_a or args.cc, args.raw_base_a, args.arch),
        b=SideOpts(args.b[0], args.b[1], args.cc_b or args.cc, args.raw_base_b, args.arch),
        ret=args.ret, corr=args.corr, map_by_symbols=args.map_by_symbols, checks=checks,
        backend=args.backend, timeout_ms=args.timeout * 1000, unroll=args.unroll,
        fuzz_iters=args.fuzz_iters, seed=args.seed, x87_cw=args.x87_cw, mxcsr=args.mxcsr,
        max_insns=args.max_insns, fuzz_time=args.fuzz_time, no_alias=args.no_alias,
        keep_dir=args.keep)
    rep = safe_analyse(opts)
    print(report.to_json(rep) if args.json else report.to_text(rep))
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
