#!/usr/bin/env python3
"""Feasibility spike: remill -> fneq-lift -> opt -> alive-tv on a+b vs b+a.

Run inside the fneq image:  python3 spike/spike.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

X86_GPR = ["EAX", "ECX", "EDX", "EBX", "ESI", "EDI", "EBP"]
X64_GPR = ["RAX", "RCX", "RDX", "RBX", "RSI", "RDI", "RBP",
           "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15"]

CONFIGS = {
    "x86-cdecl": dict(cflags=["-m32"], arch="x86", in_regs=X86_GPR,
                      stack_args=[{"offset": 4, "size": 4}, {"offset": 8, "size": 4}],
                      out_regs=["EAX", "EBX", "ESI", "EDI", "EBP"],
                      esp0=0x7FF00000, ret_addr=0xDEAD0000),
    "x64-sysv": dict(cflags=["-m64"], arch="amd64", in_regs=X64_GPR, stack_args=[],
                     out_regs=["EAX", "RBX", "RBP", "R12", "R13", "R14", "R15"],
                     esp0=0x7FFFFF000000, ret_addr=0xDEADBEEF0000),
}


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout


def function_bytes(obj, name):
    """Return the bytes of symbol `name` from an ELF object (section-relative)."""
    run(["llvm-objcopy", "-O", "binary", "--only-section=.text", obj, obj + ".text"])
    text = open(obj + ".text", "rb").read()
    for line in run(["llvm-nm", "--print-size", "--defined-only", obj]).splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == name:
            off, size = int(parts[0], 16), int(parts[1], 16)
            return text[off:off + size]
    sys.exit(f"symbol {name} not found in {obj}")


def lift(cfg, code, fname, work):
    base = 0x1000
    spec = dict(arch=cfg["arch"], os="linux", name=fname, entry=base,
                code=[{"addr": base, "hex": code.hex()}],
                in_regs=cfg["in_regs"], stack_args=cfg["stack_args"],
                out_regs=cfg["out_regs"], esp0=cfg["esp0"], ret_addr=cfg["ret_addr"],
                frame_below=512, frame_above=64)
    sp = os.path.join(work, fname + ".json")
    out = os.path.join(work, fname + ".ll")
    json.dump(spec, open(sp, "w"))
    diag = json.loads(run(["fneq-lift", "--spec", sp, "--out", out]).strip().splitlines()[-1])
    if not diag["ok"]:
        sys.exit(f"fneq-lift failed: {diag}")
    return out


def normalise(ll, work, tag):
    out = os.path.join(work, tag + ".norm.ll")
    run(["opt", "-S", "-passes=function(mem2reg,sroa,instcombine,simplifycfg,early-cse)",
         ll, "-o", out])
    return out


def body(ll, fname):
    text = open(ll).read()
    m = re.search(r"define [^@]*@" + re.escape(fname) + r"\(.*?\n}\n", text, re.S)
    return m.group(0).replace("@" + fname + "(", "@F(") if m else None


def alive(src_ll, tgt_ll, work, tag):
    linked = os.path.join(work, tag + ".linked.ll")
    run(["llvm-link", "-S", src_ll, tgt_ll, "-o", linked])
    p = subprocess.run(["alive-tv", "--src-fn=src", "--tgt-fn=tgt", "--bidirectional",
                        "--disable-undef-input", "--disable-poison-input",
                        "--smt-to=60000", linked], capture_output=True, text=True)
    out = p.stdout + p.stderr
    if "Transformation seems to be correct!" in out and "ERROR" not in out:
        return "EQUIVALENT", out
    if "Transformation doesn't verify!" in out or "Reverse transformation doesn't verify" in out:
        return "NOT_EQUIVALENT", out
    return "UNKNOWN", out


def main():
    results = []
    for cname, cfg in CONFIGS.items():
        work = tempfile.mkdtemp(prefix="spike-" + cname + "-")
        obj = os.path.join(work, "add.o")
        run(["clang", "-O2", "-fno-asynchronous-unwind-tables", "-c", *cfg["cflags"],
             os.path.join(HERE, "add.c"), "-o", obj])
        src = normalise(lift(cfg, function_bytes(obj, "add_ab"), "src", work), work, "src")
        for other, expect in (("add_ba", "EQUIVALENT"), ("add_bad", "NOT_EQUIVALENT")):
            tgt = normalise(lift(cfg, function_bytes(obj, other), "tgt", work), work, "tgt")
            same_text = body(src, "src") == body(tgt, "tgt")
            verdict, out = alive(src, tgt, work, other)
            ok = verdict == expect
            results.append((cname, other, verdict, expect, ok, same_text))
            print(f"== {cname}: add_ab vs {other}: {verdict} (expected {expect})"
                  f"{' [normalised IR textually identical]' if same_text else ''}"
                  f" -> {'OK' if ok else 'MISMATCH'}")
            if verdict != "EQUIVALENT":
                ex = re.search(r"Example:\n(.*?)\n\n", out, re.S)
                print(ex.group(1) if ex else out[-2000:])
    return 0 if all(r[4] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
