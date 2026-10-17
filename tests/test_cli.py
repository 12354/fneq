"""End-to-end behaviour of the fneq CLI on small hand-assembled raw blobs.

Run inside the fneq image: pytest tests/test_cli.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("fneq-lift") is None or shutil.which("alive-tv") is None,
                                reason="needs the fneq toolchain (run inside the fneq image)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# x86-64 System V
ADD = bytes.fromhex("8d0437c3")                    # lea eax,[rdi+rsi]; ret
ADD_SWAPPED = bytes.fromhex("8d043ec3")            # lea eax,[rsi+rdi]; ret
ADD_PLUS_1 = bytes.fromhex("8d443701c3")           # lea eax,[rdi+rsi+1]; ret
LOAD_2000 = bytes.fromhex("8b042500200000c3")      # mov eax,[0x2000]; ret
LOAD_3000 = bytes.fromhex("8b042500300000c3")      # mov eax,[0x3000]; ret
JMP_RAX = bytes.fromhex("ffe0")                    # jmp rax
SMC = bytes.fromhex("c605f9ffffff90c3")            # mov byte [rip-7],0x90; ret
CLOBBER_RBX = bytes.fromhex("bb01000000" "89f8c3")  # mov ebx,1; mov eax,edi; ret
IDENTITY = bytes.fromhex("89f8c3")                 # mov eax,edi; ret


def fneq(tmp_path, a: bytes, b: bytes, *extra: str, arch: str = "x64", cc: str = "sysv") -> dict:
    pa, pb = tmp_path / "a.bin", tmp_path / "b.bin"
    pa.write_bytes(a)
    pb.write_bytes(b)
    cmd = [sys.executable, "-m", "fneq", f"{pa}:0x1000", f"{pb}:0x1000", "--arch", arch,
           "--raw-base-a", "0x1000", "--raw-base-b", "0x1000", "--cc", cc, "--ret", "i32",
           "--json", "--fuzz-time", "10", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": ROOT})
    rep = json.loads(p.stdout)
    rep["_exit"] = p.returncode
    return rep


def test_equivalent_exit_code_and_report_fields(tmp_path):
    rep = fneq(tmp_path, ADD, ADD_SWAPPED)
    assert rep["verdict"] == "EQUIVALENT" and rep["_exit"] == 0
    assert "definition" in rep and rep["checks_disabled"] == [
        "write order (final memory state is still compared)"]
    assert rep["conventions"]["A"]["name"] == "sysv"


def test_not_equivalent_exit_code_and_replayed_counterexample(tmp_path):
    rep = fneq(tmp_path, ADD, ADD_PLUS_1)
    assert rep["verdict"] == "NOT_EQUIVALENT" and rep["_exit"] == 1
    cex = rep["counterexample"]
    assert cex["first_difference"].startswith("return: EAX")
    assert cex["A"]["outputs"]["EAX"] != cex["B"]["outputs"]["EAX"]


def test_globals_differ_without_mapping(tmp_path):
    rep = fneq(tmp_path, LOAD_2000, LOAD_3000)
    assert rep["verdict"] == "NOT_EQUIVALENT"


def test_correspondence_file_maps_globals(tmp_path):
    corr = tmp_path / "corr.toml"
    corr.write_text('[[globals]]\na = "0x2000"\nb = "0x3000"\nsize = 4\n')
    rep = fneq(tmp_path, LOAD_2000, LOAD_3000, "--corr", str(corr))
    assert rep["verdict"] == "EQUIVALENT", rep["reason"]
    assert rep["correspondence"]["globals"] == [{"a": "0x2000", "b": "0x3000", "size": 4}]


def test_unresolved_indirect_jump_is_unknown(tmp_path):
    rep = fneq(tmp_path, JMP_RAX, ADD)
    assert rep["verdict"] == "UNKNOWN" and rep["_exit"] == 2
    assert "indirect jump" in rep["reason"]


def test_self_modifying_code_is_unknown(tmp_path):
    rep = fneq(tmp_path, SMC, ADD)
    assert rep["verdict"] == "UNKNOWN"
    assert "self-modifying" in rep["reason"]


def test_callee_saved_clobber_and_disabled_check(tmp_path):
    rep = fneq(tmp_path, CLOBBER_RBX, IDENTITY)
    assert rep["verdict"] == "NOT_EQUIVALENT"
    assert "RBX" in rep["counterexample"]["first_difference"]
    rep = fneq(tmp_path, CLOBBER_RBX, IDENTITY, "--no-check-preserve")
    assert rep["verdict"] == "EQUIVALENT", rep["reason"]
    assert "callee-saved registers / stack delta" in rep["checks_disabled"]


def test_x86_raw_cdecl(tmp_path):
    a = bytes.fromhex("8b442404" "03442408" "c3")   # mov eax,[esp+4]; add eax,[esp+8]; ret
    b = bytes.fromhex("8b442408" "03442404" "c3")
    rep = fneq(tmp_path, a, b, arch="x86", cc="cdecl")
    assert rep["verdict"] == "EQUIVALENT", rep["reason"]
