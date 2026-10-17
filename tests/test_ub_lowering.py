"""The lowering must not introduce UB that lets Alive2 prove a false equivalence.

Each pair below would be 'equivalent' under C/LLVM semantics (signed overflow,
division by zero, and over-wide shifts are UB/poison there) but differs on the
machine, or the reverse. fneq must classify them by machine semantics.
Run inside the fneq image: pytest tests/test_ub_lowering.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("fneq-lift") is None or shutil.which("alive-tv") is None,
                                reason="needs the fneq toolchain (run inside the fneq image)")

# x86-64 System V machine code (raw blobs loaded at 0x1000).
X_PLUS_1_GT_X = bytes.fromhex("8d4701" "39f8" "0f9fc0" "0fb6c0" "c3")   # lea eax,[rdi+1]; cmp eax,edi; setg al; movzx eax,al; ret
RETURN_1 = bytes.fromhex("b801000000" "c3")                             # mov eax,1; ret
DIV_UNCHECKED = bytes.fromhex("89f8" "31d2" "f7f6" "c3")                # mov eax,edi; xor edx,edx; div esi; ret
DIV_CHECKED = bytes.fromhex("85f6" "7407" "89f8" "31d2" "f7f6" "c3" "31c0" "c3")
#   test esi,esi; jz +7; mov eax,edi; xor edx,edx; div esi; ret; xor eax,eax; ret
SHL_CL = bytes.fromhex("89f8" "89f1" "d3e0" "c3")                       # mov eax,edi; mov ecx,esi; shl eax,cl; ret
SHL_CL_MASKED = bytes.fromhex("89f8" "89f1" "83e11f" "d3e0" "c3")       # ... and ecx,31; shl eax,cl; ret


def run_fneq(tmp_path, a: bytes, b: bytes, *extra: str) -> dict:
    pa, pb = tmp_path / "a.bin", tmp_path / "b.bin"
    pa.write_bytes(a)
    pb.write_bytes(b)
    cmd = [sys.executable, "-m", "fneq", f"{pa}:0x1000", f"{pb}:0x1000", "--arch", "x64",
           "--raw-base-a", "0x1000", "--raw-base-b", "0x1000", "--cc", "sysv", "--ret", "i32",
           "--json", "--keep", str(tmp_path / "keep"), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))})
    rep = json.loads(p.stdout)
    rep["_exit"] = p.returncode
    return rep


def test_signed_overflow_is_not_ub(tmp_path):
    rep = run_fneq(tmp_path, X_PLUS_1_GT_X, RETURN_1, "--backend", "alive")
    assert rep["verdict"] == "NOT_EQUIVALENT", rep["reason"]
    assert rep["_exit"] == 1
    cex = rep["counterexample"]
    assert int(cex["inputs"]["RDI"], 16) & 0xFFFFFFFF == 0x7FFFFFFF


def test_division_by_zero_is_a_fault_not_ub(tmp_path):
    rep = run_fneq(tmp_path, DIV_UNCHECKED, DIV_CHECKED, "--backend", "alive")
    assert rep["verdict"] == "NOT_EQUIVALENT", rep["reason"]
    assert int(rep["counterexample"]["inputs"]["RSI"], 16) & 0xFFFFFFFF == 0
    assert "fault" in rep["counterexample"]["first_difference"]


def test_shift_count_is_masked_like_the_cpu(tmp_path):
    rep = run_fneq(tmp_path, SHL_CL, SHL_CL_MASKED, "--backend", "alive")
    assert rep["verdict"] == "EQUIVALENT", rep["reason"]


def test_lifter_output_has_no_poison_or_undef(tmp_path):
    run_fneq(tmp_path, X_PLUS_1_GT_X, DIV_UNCHECKED, "--backend", "alive")
    for side in ("a", "b"):
        text = (tmp_path / "keep" / f"{side}.ll").read_text()
        body = re.search(r"define .*?\n}\n", text, re.S).group(0)
        for token in (" nsw", " nuw", " exact", "undef", "poison", " nnan", " ninf", "llvm.assume"):
            assert token not in body, f"{token!r} in lifter output for side {side}"
