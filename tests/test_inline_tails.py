"""Exact inlining of direct tail wrappers, including inside an inlined call."""
from __future__ import annotations

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding, pipeline


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("increment", [1, 2])
def test_tail_wrapper_vs_inlined_body(tmp_path, monkeypatch, tail, increment):
    # Root calls (or tail-jumps to) a wrapper, which adjusts an unused register
    # and tail-jumps to the actual helper. PE function boundaries keep the
    # three entries separate. The other binary inlines x + 1 in its root.
    root = (bytes.fromhex("e909000000") + b"\xcc" * 9 if tail else
            bytes.fromhex("4883ec28 e805000000 4883c428 c3"))
    wrapper = bytes.fromhex("31d2e900000000")
    helper = bytes([0x8D, 0x41, increment, 0xC3])
    pa, pb = tmp_path / "a.bin", tmp_path / "b.bin"
    pa.write_bytes(root + wrapper + helper)
    pb.write_bytes(bytes.fromhex("8d4101c3"))
    binaries = {}
    for path, ranges in [(pa, [(0x1000, 0x100E, 0), (0x100E, 0x1015, 0),
                              (0x1015, 0x1019, 0)]), (pb, [(0x1000, 0x1004, 0)])]:
        b = binmod.load(str(path), 0x1000, "x64")
        bounding.angr_project(b)
        b.fmt, b.pdata_ranges = "pe", ranges
        binaries[str(path)] = b
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binaries[side.path])
    opts = pipeline.Options(
        pipeline.SideOpts(str(pa), "0x1000", cc="ms"),
        pipeline.SideOpts(str(pb), "0x1000", cc="ms"),
        backend="lockstep", no_alias=True, ret="i32")
    rep = pipeline.analyse(opts)
    assert rep.verdict == ("EQUIVALENT" if increment == 1 else "UNKNOWN"), rep.reason
    assert rep.data["lockstep"]["inlined_callees"]
