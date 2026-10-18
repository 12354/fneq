"""A 32-bit x64 stack parameter does not consume the residue above its low dword."""
from __future__ import annotations

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding, pipeline


@pytest.mark.parametrize("size_a,size_b,low_b,expected", [
    (4, 4, 7, "EQUIVALENT"),
    (4, 4, 8, "UNKNOWN"),
    (8, 8, 7, "UNKNOWN"),
    (4, 8, 7, "UNKNOWN"),
])
def test_stack_argument_consumed_width(tmp_path, monkeypatch, size_a, size_b, low_b, expected):
    binaries = {}
    for tag, size, value in [("a", size_a, 0x1122334400000007),
                             ("b", size_b, 0x5566778800000000 | low_b)]:
        # Allocate home space + one argument, write its whole slot, call helper,
        # restore SP, return. The helper reads either eax or rax from its arg5.
        root = bytes.fromhex("4883ec38 48b8") + value.to_bytes(8, "little") + \
            bytes.fromhex("4889442420 e805000000 4883c438 c3")
        helper = (b"\x48" if size == 8 else b"") + bytes.fromhex("8b442428 c3")
        path = tmp_path / f"{tag}.bin"
        path.write_bytes(root + helper)
        b = binmod.load(str(path), 0x1000, "x64")
        bounding.angr_project(b)
        b.fmt = "pe"
        b.pdata_ranges = [(0x1000, 0x101D, 0), (0x101D, 0x101D + len(helper), 0)]
        binaries[tag] = b
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binaries[side.path])
    opts = pipeline.Options(pipeline.SideOpts("a", "0x1000", cc="ms"),
                            pipeline.SideOpts("b", "0x1000", cc="ms"),
                            backend="lockstep", no_alias=True, ret="i64")
    rep = pipeline.analyse(opts)
    assert rep.verdict == expected, rep.reason
