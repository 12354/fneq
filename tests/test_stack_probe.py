"""The exact MSVC x64 stack probe has a nonstandard register contract."""
from __future__ import annotations

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding, callees, pipeline


@pytest.mark.parametrize("same_size", [True, False])
@pytest.mark.parametrize("modified_probe", [False, True])
def test_verified_probe_preserves_allocation_size(tmp_path, monkeypatch, same_size, modified_probe):
    binaries = {}
    for tag, size in [("a", 0x40), ("b", 0x40 if same_size else 0x50)]:
        root = b"\xb8" + size.to_bytes(4, "little") + bytes.fromhex("e836000000 482be0 c7042400000000")
        root += bytes.fromhex("4883c4") + bytes([size]) + b"\xc3"
        probe = (bytes.fromhex("31c0") if modified_probe else b"") + callees.MSVC_X64_CHKSTK
        path = tmp_path / f"{tag}.bin"
        path.write_bytes(root + b"\xcc" * (0x40 - len(root)) + probe)
        binary = binmod.load(str(path), 0x1000, "x64")
        bounding.angr_project(binary)
        binary.fmt = "pe"
        binary.pdata_ranges = [(0x1000, 0x1000 + len(root), 0),
                               (0x1040, 0x1040 + len(probe), 0)]
        binaries[tag] = binary
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binaries[side.path])
    opts = pipeline.Options(pipeline.SideOpts("a", "0x1000", cc="ms"),
                            pipeline.SideOpts("b", "0x1000", cc="ms"),
                            backend="lockstep", no_alias=True, ret="i64")
    rep = pipeline.analyse(opts)
    assert rep.verdict == ("EQUIVALENT" if same_size and not modified_probe else "UNKNOWN"), rep.reason
