"""Self-test corpus: build function pairs with known answers and check fneq's verdicts.

    python -m fneq.corpus [--filter SUBSTR] [--target T ...] [--jobs N] [--list]

A case passes only when fneq's verdict matches the expected class:
  equivalent -> EQUIVALENT or EQUIVALENT_BOUNDED(k)
  different  -> NOT_EQUIVALENT (which fneq only reports after concrete replay)
Cases whose toolchain is missing are SKIPPED; skips are reported separately and never
count as passes. The exit status is 0 only if there are no failures.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Source checkout: <repo>/corpus. Installed package (Docker image): FNEQ_CORPUS_DIR.
CORPUS = Path(os.environ.get("FNEQ_CORPUS_DIR") or ROOT / "corpus")
SRC_IN = CORPUS / "src"
# Build outputs (and a snapshot of the sources) live outside the source tree so
# that a container can keep them on a local volume instead of a host bind mount.
BUILD = Path(os.environ.get("FNEQ_BUILD_DIR", CORPUS / "build"))
SRC = BUILD / "src"

COMMON_CFLAGS = ["-fno-asynchronous-unwind-tables", "-fno-stack-protector", "-fno-pic",
                 "-fno-pie", "-ffreestanding", "-fno-builtin", "-fcf-protection=none"]

# target name -> (arch, convention)
TARGETS = {
    "x86-cdecl": ("x86", "cdecl"),
    "x86-stdcall": ("x86", "stdcall"),
    "x86-fastcall": ("x86", "fastcall"),
    "x86-thiscall": ("x86", "thiscall"),
    "x64-sysv": ("x64", "sysv"),
    "x64-ms": ("x64", "ms"),
}


@dataclasses.dataclass(frozen=True)
class Build:
    name: str
    toolchain: str          # gcc | clang | mingw | msvc
    flags: tuple[str, ...]

    @property
    def fmt(self) -> str:
        return "pe" if self.toolchain in ("mingw", "msvc") else "elf"

    def compiler(self, arch: str) -> list[str] | None:
        """Compiler command prefix for `arch`, or None if unavailable."""
        if self.toolchain == "gcc":
            cmd = ["gcc", "-m32" if arch == "x86" else "-m64"]
        elif self.toolchain == "clang":
            cmd = ["clang", "-m32" if arch == "x86" else "-m64"]
        elif self.toolchain == "mingw":
            cmd = ["i686-w64-mingw32-gcc" if arch == "x86" else "x86_64-w64-mingw32-gcc"]
        else:
            return None
        return cmd if shutil.which(cmd[0]) else None

    def unavailable_reason(self, arch: str) -> str | None:
        if self.toolchain == "msvc":
            return "MSVC (cl.exe) is not available in the fneq container"
        if self.compiler(arch) is None:
            return f"{self.toolchain} toolchain for {arch} not found"
        return None


BUILDS = {b.name: b for b in [
    Build("gcc-O0", "gcc", ("-O0",)),
    Build("gcc-O2", "gcc", ("-O2",)),
    Build("gcc-O2-sse", "gcc", ("-O2", "-msse2", "-mfpmath=sse")),
    Build("clang-O0", "clang", ("-O0",)),
    Build("clang-O2", "clang", ("-O2",)),
    Build("clang-O2-sse", "clang", ("-O2", "-msse2", "-mfpmath=sse")),
    Build("clang-O2-novec", "clang", ("-O2", "-fno-vectorize", "-fno-slp-vectorize")),
    Build("mingw-O0", "mingw", ("-O0",)),
    Build("mingw-O2", "mingw", ("-O2",)),
    Build("msvc-O2", "msvc", ("/O2",)),
]}


@dataclasses.dataclass
class Case:
    pair: str
    expect: str
    target: str
    a: tuple[str, str, str]   # (build, source, function)
    b: tuple[str, str, str]
    ret: str | None = None
    skip: str | None = None
    verdict: str | None = None
    status: str | None = None     # PASS | FAIL | SKIP | ERROR
    detail: str = ""
    seconds: float = 0.0

    @property
    def id(self) -> str:
        (ba, sa, fa), (bb, sb, fb) = self.a, self.b
        left = f"{fa}@{ba}"
        right = f"{fb}@{bb}"
        return f"{self.pair}/{self.target}/{left}~{right}"


def expand(manifest: dict, targets: list[str]) -> list[Case]:
    rets = manifest.get("ret", {})
    cases = []
    for pair in manifest["pair"]:
        pair_targets = [t for t in pair.get("targets", TARGETS) if t in targets]
        for target in pair_targets:
            if pair["mode"] == "variant":
                sa, fa = pair["a"].split(":")
                sb, fb = pair["b"].split(":")
                for build in pair["builds"]:
                    cases.append(Case(pair["name"], pair["expect"], target,
                                      (build, sa, fa), (build, sb, fb), rets.get(fa)))
            elif pair["mode"] == "cross":
                for func in pair["funcs"]:
                    s, f = func.split(":")
                    for ba, bb in pair["cross"]:
                        cases.append(Case(pair["name"], pair["expect"], target,
                                          (ba, s, f), (bb, s, f), rets.get(f)))
            else:
                raise SystemExit(f"corpus: unknown mode {pair['mode']!r} in {pair['name']}")
    return cases


def run_checked(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def binary_path(target: str, build: Build, source: str) -> Path:
    ext = ".exe" if build.fmt == "pe" else ".elf"
    return BUILD / target / build.name / (Path(source).stem + ext)


def snapshot_sources() -> None:
    """Copy corpus sources next to the build outputs (sequentially, once)."""
    SRC.mkdir(parents=True, exist_ok=True)
    for f in SRC_IN.iterdir():
        dst = SRC / f.name
        data = f.read_bytes()
        if not dst.exists() or dst.read_bytes() != data:
            dst.write_bytes(data)


def build_binary(target: str, build: Build, source: str) -> Path:
    """Compile `source` plus externs.c for `target` with `build`; cached by mtime."""
    arch, conv = TARGETS[target]
    out = binary_path(target, build, source)
    inputs = [SRC / source, SRC / "externs.c", SRC / "cc.h"]
    if out.exists() and all(out.stat().st_mtime >= p.stat().st_mtime for p in inputs):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cc = build.compiler(arch)
    flags = [*COMMON_CFLAGS, f"-DFNEQ_CC_{conv}"]
    if build.toolchain == "mingw":
        flags = [f for f in flags if f not in ("-fno-pie",)]
    objs = []
    for src, opt in ((source, list(build.flags)), ("externs.c", ["-O0"])):
        obj = out.with_name(Path(source).stem + "." + Path(src).stem + ".o")
        run_checked([*cc, "-c", *opt, *flags, str(SRC / src), "-o", str(obj)])
        objs.append(str(obj))
    entry = "_fneq_entry" if (build.toolchain == "mingw" and arch == "x86") else "fneq_entry"
    link = [*cc, "-nostdlib", f"-Wl,-e,{entry}", *objs, "-o", str(out)]
    if build.fmt == "elf":
        link[1:1] = ["-static", "-no-pie"]
    run_checked(link)
    return out


def run_case(case: Case, fneq_cmd: list[str], timeout: int) -> Case:
    if case.skip:
        case.status = "SKIP"
        return case
    arch, conv = TARGETS[case.target]
    bins = []
    for build_name, source, _ in (case.a, case.b):
        bins.append(binary_path(case.target, BUILDS[build_name], source))
    cmd = [*fneq_cmd, f"{bins[0]}:{case.a[2]}", f"{bins[1]}:{case.b[2]}",
           "--cc", conv, "--json", "--map-by-symbols"]
    if case.ret:
        cmd += ["--ret", case.ret]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        case.status, case.detail = "ERROR", f"fneq exceeded the corpus timeout ({timeout}s)"
        return case
    finally:
        case.seconds = time.monotonic() - t0
    try:
        report = json.loads(p.stdout)
        case.verdict = report["verdict"]
        case.detail = report.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        case.status = "ERROR"
        case.detail = f"fneq exit {p.returncode}, no JSON report: {(p.stderr or p.stdout)[-400:]}"
        return case
    expected_exit = {"EQUIVALENT": 0, "NOT_EQUIVALENT": 1}.get(case.verdict.split("(")[0], 2)
    if case.verdict.startswith("EQUIVALENT_BOUNDED"):
        expected_exit = 0
    if p.returncode != expected_exit:
        case.status = "ERROR"
        case.detail = f"exit code {p.returncode} does not match verdict {case.verdict}"
        return case
    if case.expect == "equivalent":
        ok = case.verdict == "EQUIVALENT" or case.verdict.startswith("EQUIVALENT_BOUNDED")
    else:
        ok = case.verdict == "NOT_EQUIVALENT"
    case.status = "PASS" if ok else "FAIL"
    return case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", action="append", default=[], help="only cases whose id contains this")
    ap.add_argument("--target", action="append", choices=list(TARGETS), help="restrict targets")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=900, help="per-case wall-clock limit (s)")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--build-only", action="store_true", help="build binaries and exit")
    ap.add_argument("--json-out", type=Path, help="write per-case results as JSON")
    args = ap.parse_args(argv)

    manifest = tomllib.loads((CORPUS / "corpus.toml").read_text())
    cases = expand(manifest, args.target or list(TARGETS))
    if args.filter:
        cases = [c for c in cases if any(f in c.id for f in args.filter)]
    if args.list:
        for c in cases:
            print(c.id)
        return 0

    snapshot_sources()

    # Build every binary that a runnable case needs; a missing toolchain skips the case.
    needed: dict[tuple[str, str, str], str | None] = {}
    for c in cases:
        arch = TARGETS[c.target][0]
        for build_name, source, _ in (c.a, c.b):
            build = BUILDS[build_name]
            key = (c.target, build_name, source)
            if key not in needed:
                needed[key] = build.unavailable_reason(arch)
    build_errors: dict[tuple[str, str, str], str] = {}
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(build_binary, t, BUILDS[b], s): (t, b, s)
                for (t, b, s), reason in needed.items() if reason is None}
        for fut in cf.as_completed(futs):
            try:
                fut.result()
            except RuntimeError as e:
                build_errors[futs[fut]] = str(e)
    for c in cases:
        for build_name, source, _ in (c.a, c.b):
            key = (c.target, build_name, source)
            if needed[key]:
                c.skip = needed[key]
            elif key in build_errors:
                c.status, c.detail = "ERROR", "corpus build failed: " + build_errors[key][:300]

    if args.build_only:
        for key, err in build_errors.items():
            print("BUILD ERROR", key, err[:2000])
        return 1 if build_errors else 0

    fneq_cmd = [sys.executable, "-m", "fneq"]
    todo = [c for c in cases if c.status is None]
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda c: run_case(c, fneq_cmd, args.timeout), todo))

    counts: dict[str, int] = {}
    for c in cases:
        counts[c.status] = counts.get(c.status, 0) + 1
        if c.status != "PASS":
            reason = c.skip if c.status == "SKIP" else c.detail
            print(f"{c.status:5} {c.id}  verdict={c.verdict}  {reason[:200]}")
    print()
    by_pair: dict[str, dict[str, int]] = {}
    for c in cases:
        by_pair.setdefault(c.pair, {}).setdefault(c.status, 0)
        by_pair[c.pair][c.status] += 1
    print(f"{'pair':28} {'PASS':>5} {'FAIL':>5} {'ERROR':>5} {'SKIP':>5}")
    for pair, st in by_pair.items():
        print(f"{pair:28} {st.get('PASS', 0):5} {st.get('FAIL', 0):5} "
              f"{st.get('ERROR', 0):5} {st.get('SKIP', 0):5}")
    total = len(cases)
    print(f"\n{total} cases: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("(skipped cases are not passes)")
    if args.json_out:
        args.json_out.write_text(json.dumps([dataclasses.asdict(c) for c in cases], indent=1))
    return 0 if counts.get("FAIL", 0) == 0 and counts.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
