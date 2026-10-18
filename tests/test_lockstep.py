"""The lockstep stage alone (--backend lockstep) on small hand-assembled raw blobs.

Run inside the fneq image: pytest tests/test_lockstep.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

angr = pytest.importorskip("angr")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# x86-64 System V
ADD = bytes.fromhex("8d0437c3")                          # lea eax,[rdi+rsi]; ret
ADD_SWAPPED = bytes.fromhex("8d043ec3")                  # lea eax,[rsi+rdi]; ret
ADD_PLUS_1 = bytes.fromhex("8d443701c3")                 # lea eax,[rdi+rsi+1]; ret
ADD_ECX = bytes.fromhex("89f9" "01f1" "89c8" "c3")       # mov ecx,edi; add ecx,esi; mov eax,ecx; ret
# sum = 0; while (n) { sum += n; n--; } return sum;  (accumulator in EAX / in ECX)
SUM_EAX = bytes.fromhex("31c0" "85ff" "7406" "01f8" "ffcf" "ebf6" "c3")
SUM_ECX = bytes.fromhex("31c9" "85ff" "7406" "01f9" "ffcf" "ebf6" "89c8" "c3")
SUB_EAX = bytes.fromhex("31c0" "85ff" "7406" "29f8" "ffcf" "ebf6" "c3")   # sum -= n


def lockstep(tmp_path, a: bytes, b: bytes, ret: str = "i32", extra: tuple = ()) -> dict:
    pa, pb = tmp_path / "a.bin", tmp_path / "b.bin"
    pa.write_bytes(a)
    pb.write_bytes(b)
    cmd = [sys.executable, "-m", "fneq", f"{pa}:0x1000", f"{pb}:0x1000", "--arch", "x64",
           "--raw-base-a", "0x1000", "--raw-base-b", "0x1000", "--cc", "sysv", "--ret", ret,
           "--json", "--backend", "lockstep", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "PYTHONPATH": ROOT})
    rep = json.loads(p.stdout)
    assert [b["backend"] for b in rep["backends"]] == ["lockstep"], "only the lockstep stage runs"
    return rep


def test_commuted_operands(tmp_path):
    rep = lockstep(tmp_path, ADD, ADD_SWAPPED)
    assert rep["verdict"] == "EQUIVALENT" and rep["backend"] == "lockstep"


def test_register_allocation(tmp_path):
    assert lockstep(tmp_path, ADD, ADD_ECX)["verdict"] == "EQUIVALENT"


def test_loop_with_different_registers(tmp_path):
    rep = lockstep(tmp_path, SUM_EAX, SUM_ECX)
    assert rep["verdict"] == "EQUIVALENT"
    assert rep["lockstep"]["segment_pairs"] >= 2       # the entry and the loop head


def test_difference_is_localised(tmp_path):
    rep = lockstep(tmp_path, ADD, ADD_PLUS_1)
    assert rep["verdict"] == "UNKNOWN"
    diff = rep["lockstep"]["first_difference"]
    assert "return value" in diff["what"] and diff["a"] == "0x1003" and diff["b"] == "0x1004"


def test_loop_body_difference_is_not_proven(tmp_path):
    rep = lockstep(tmp_path, SUM_EAX, SUB_EAX)
    assert rep["verdict"] == "UNKNOWN"
    assert rep["lockstep"]["result"] == "differs"


# `if (x == 0) return 0; return 1;` vs `if (x == 0) return x; return 1;`: B relies on the
# branch taken (EDI == 0) in the next segment, which only a path-sensitive boundary
# (refinement after the first failed obligation) sees.
ZERO_CONST = bytes.fromhex("85ff" "7506" "b800000000" "c3" "b801000000" "c3")
ZERO_EDI = bytes.fromhex("85ff" "7503" "89f8" "c3" "b801000000" "c3")
ZERO_EDI_PLUS_1 = bytes.fromhex("85ff" "7506" "8d4701" "90" "90" "c3" "b801000000" "c3")
# 64-bit test: the guard pins RDI to 0, a fact carried into the next segment's entry.
ZERO_CONST_64 = bytes.fromhex("4885ff" "7506" "b800000000" "c3" "b801000000" "c3")
ZERO_EDI_64 = bytes.fromhex("4885ff" "7503" "89f8" "c3" "b801000000" "c3")
# 4 * x by a loop with a constant trip count vs by an unrolled computation.
LOOP4 = bytes.fromhex("31c0" "b904000000" "01f8" "ffc9" "75fa" "c3")
LOOP5 = bytes.fromhex("31c0" "b905000000" "01f8" "ffc9" "75fa" "c3")
TIMES4 = bytes.fromhex("8d04bd00000000" "c3")


def test_path_fact_used_by_one_side(tmp_path):
    assert lockstep(tmp_path, ZERO_CONST, ZERO_EDI)["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, ZERO_CONST_64, ZERO_EDI_64)["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, ZERO_CONST, ZERO_EDI_PLUS_1)["verdict"] == "UNKNOWN"


def test_constant_trip_count_loop_vs_unrolled(tmp_path):
    assert lockstep(tmp_path, LOOP4, TIMES4)["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, LOOP5, TIMES4)["verdict"] == "UNKNOWN"


# The same after a branch (`if (y) return 0;`): the path reaches the loop head after a
# fork and runs on through it speculatively, since the loop is decided by constants.
GUARD_LOOP4 = bytes.fromhex("85f6" "750e" "31c0" "b904000000" "01f8" "ffc9" "75fa" "c3" "31c0" "c3")
GUARD_LOOP5 = bytes.fromhex("85f6" "750e" "31c0" "b905000000" "01f8" "ffc9" "75fa" "c3" "31c0" "c3")
GUARD_TIMES4 = bytes.fromhex("85f6" "7508" "8d04bd00000000" "c3" "31c0" "c3")


def test_constant_trip_count_loop_after_a_branch(tmp_path):
    assert lockstep(tmp_path, GUARD_LOOP4, GUARD_TIMES4)["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, GUARD_LOOP5, GUARD_TIMES4)["verdict"] == "UNKNOWN"


# `return x && y;` as two branches vs branch-free (setne/and): the branches have no
# one-to-one counterpart, the paths of the region are paired by feasibility instead.
AND_BRANCHES = bytes.fromhex("85ff" "740a" "85f6" "7406" "b801000000" "c3" "31c0" "c3")
AND_SETCC = bytes.fromhex("85ff" "0f95c0" "85f6" "0f95c1" "20c8" "0fb6c0" "c3")
OR_SETCC = bytes.fromhex("85ff" "0f95c0" "85f6" "0f95c1" "08c8" "0fb6c0" "c3")


def test_different_branch_structure(tmp_path):
    rep = lockstep(tmp_path, AND_BRANCHES, AND_SETCC)
    assert rep["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, AND_BRANCHES, OR_SETCC)["verdict"] == "UNKNOWN"


# SSE arithmetic is uninterpreted: the same operation on the same operands is equal,
# a different operation or rounding mode is not.
ADDSD = bytes.fromhex("f20f58c1" "c3")                                # addsd xmm0,xmm1
ADDSD_VIA_XMM2 = bytes.fromhex("660f28d0" "f20f58d1" "660f28c2" "c3")
SUBSD = bytes.fromhex("f20f5cc1" "c3")
CVTTSD2SI = bytes.fromhex("f20f2cc0" "c3")                            # truncating
CVTSD2SI = bytes.fromhex("f20f2dc0" "c3")                             # round to nearest


def test_floating_point_uninterpreted(tmp_path):
    assert lockstep(tmp_path, ADDSD, ADDSD_VIA_XMM2, "f64")["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, ADDSD, SUBSD, "f64")["verdict"] == "UNKNOWN"
    assert lockstep(tmp_path, CVTTSD2SI, CVTTSD2SI)["verdict"] == "EQUIVALENT"
    assert lockstep(tmp_path, CVTTSD2SI, CVTSD2SI)["verdict"] == "UNKNOWN"


# `*p = 1; return *q;` vs `r = *q; *p = 1; return r;`: different only when p and q
# overlap -- proven only under --no-alias, which the proof then says it assumed.
STORE_THEN_READ = bytes.fromhex("c70701000000" "8b06" "c3")
READ_THEN_STORE = bytes.fromhex("8b06" "c70701000000" "c3")


def test_no_alias_assumption(tmp_path):
    assert lockstep(tmp_path, STORE_THEN_READ, READ_THEN_STORE)["verdict"] == "UNKNOWN"
    rep = lockstep(tmp_path, STORE_THEN_READ, READ_THEN_STORE, extra=("--no-alias",))
    assert rep["verdict"] == "EQUIVALENT"
    assert "assuming pointers do not alias" in rep["lockstep"]["reason"]


# return h(x) with h(x) = x + 1 (call h; ret / h: lea eax,[rdi+1]; ret)  vs  x + 1 inline
CALLS_HELPER = bytes.fromhex("e801000000" "c3" "8d4701c3")
HELPER_INLINED = bytes.fromhex("8d4701c3")
CALLS_HELPER_PLUS_2 = bytes.fromhex("e801000000" "c3" "8d4702c3")


def test_helper_inlined_on_one_side(tmp_path):
    rep = lockstep(tmp_path, CALLS_HELPER, HELPER_INLINED)
    assert rep["verdict"] == "EQUIVALENT"
    assert rep["lockstep"]["inlined_callees"] == ["A:0x1006"]
    assert lockstep(tmp_path, CALLS_HELPER_PLUS_2, HELPER_INLINED)["verdict"] == "UNKNOWN"


# int x; f(&x); g(); return x;   -- A reads x after g(), B before (f: mov eax,[rdi]; g: ret)
OUT_PARAM_LATE = bytes.fromhex("4883ec18" "488d7c2408" "e80e000000" "e80c000000" "8b442408"
                               "4883c418" "c3" "8b07c3" "c3")
OUT_PARAM_EARLY = bytes.fromhex("4883ec18" "488d7c2408" "e814000000" "8b442408" "890424"
                                "e80b000000" "8b0424" "4883c418" "c3" "8b07c3" "c3")


def test_no_alias_callees_keep_no_frame_address(tmp_path):
    # g() could change x through an address f() kept -- unless --no-alias rules that out
    assert lockstep(tmp_path, OUT_PARAM_LATE, OUT_PARAM_EARLY)["verdict"] == "UNKNOWN"
    rep = lockstep(tmp_path, OUT_PARAM_LATE, OUT_PARAM_EARLY, extra=("--no-alias",))
    assert rep["verdict"] == "EQUIVALENT"
