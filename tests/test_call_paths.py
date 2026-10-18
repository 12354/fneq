"""A shared opaque call's result excludes infeasible pairs of later paths."""
from __future__ import annotations

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding, callees, fuzz, lockstep, signature
from fneq.compare import Checks


@pytest.mark.parametrize("same_probe", [True, False])
def test_shared_call_prefix_before_different_call_counts(tmp_path, same_probe):
    # f(); if (rax != 0) g(); return. f and g stay opaque. Pair A's zero
    # branch (one call) with B's nonzero branch (two calls): impossible exactly
    # when both first calls have the same model.
    body = bytes.fromhex("4883ec28 e817000000 4885c0 7405 e81d000000 4883c428 c3")
    body += b"\xcc" * (32 - len(body)) + b"\xc3" + b"\xcc" * 15 + b"\xc3"
    sides = []
    infos = []
    cc = signature.CONVENTIONS["ms"]
    for tag in ("a", "b"):
        path = tmp_path / f"{tag}.bin"
        path.write_bytes(body)
        binary = binmod.load(str(path), 0x1000, "x64")
        info = bounding.bound(binary, 0x1000)
        names = {0x1020: "probe" if tag == "a" or same_probe else "independent_probe",
                 0x1030: "consume"}
        specs, _ = callees.callee_specs(binary, info, cc, rename=names)
        sides.append(fuzz.Side(binary, info, specs, []))
        infos.append(info)
    sig = signature.build_signature("x64", cc, "i64", infos)
    ck = lockstep.Checker(sig, *sides, Checks(), lambda x: x, no_alias=True)
    ab = ck._entry_abs()
    a, b = ck._entry_values(ab, 1)
    ex_a = ck._run_segment(ck.A, 0x1000, *a, ab.escaped[0], 1)
    ex_b = ck._run_segment(ck.B, 0x1000, *b, ab.escaped[1], 1)
    count = lambda e: sum(ev[0] == "call" for ev in e.state.lsmem.events)
    ea, eb = next(e for e in ex_a if count(e) == 1), next(e for e in ex_b if count(e) == 2)
    match = lockstep._Match(ck, 1, (0x1000, 0x1000), ea, eb, flat=False)
    with pytest.raises(lockstep._Differs):
        match.unify()  # calls still differ; this does not waive the call check
    assert match.feasible(ea, eb) is (not same_probe)
