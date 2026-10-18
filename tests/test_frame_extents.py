"""Explicit callee pointee extents exclude unrelated locals, preserving their effects."""
from __future__ import annotations

import json
import pytest

pytest.importorskip("angr")
from fneq import binary as binmod, bounding, pipeline


@pytest.mark.parametrize("mode", ["fixed", "argument", "indirect", "absent"])
@pytest.mark.parametrize("difference", ["outside", "inside", "observed"])
def test_declared_frame_extent(tmp_path, monkeypatch, mode, difference):
    binaries = {}
    for tag in ("a", "b"):
        # Two local dwords, only the first passed as a four-byte buffer.
        value = 2 if tag == "b" and difference == "inside" else 1
        suffix = 6 if tag == "b" else 5
        code = bytes.fromhex("4883ec48 c7442420") + value.to_bytes(4, "little")
        code += bytes.fromhex("c7442424") + suffix.to_bytes(4, "little")
        code += bytes.fromhex("488d4c2420")
        if mode == "indirect":
            code += bytes.fromhex("c744243004000000 488d542430")
        else:
            code += bytes.fromhex("ba04000000")
        code += b"\xe8" + (0x100 - len(code) - 5).to_bytes(4, "little")
        code += bytes.fromhex("8b442424" if difference == "observed" else "31c0")
        code += bytes.fromhex("4883c448 c3")
        helper = bytes.fromhex("8b01 0302 c3" if mode == "indirect" else "8b01 01d0 c3")
        path = tmp_path / (tag + ".bin")
        path.write_bytes(code + b"\xcc" * (0x100 - len(code)) + helper)
        b = binmod.load(str(path), 0x1000, "x64")
        bounding.angr_project(b)
        b.fmt, b.pdata_ranges = "pe", [(0x1000, 0x1000 + len(code), 0),
                                       (0x1100, 0x1100 + len(helper), 0)]
        binaries[tag] = b
    models = {"fixed": [{"arg": 0, "size": 4}],
              "argument": [{"arg": 0, "size_arg": 1}],
              "indirect": [{"arg": 0, "size_arg": 1, "indirect": 4},
                           {"arg": 1, "size": 4}], "absent": []}
    corr = tmp_path / "corr.json"
    corr.write_text(json.dumps({"callees": [{"a": "0x1100", "b": "0x1100", "name": "helper"}],
                               "callee_contracts": {"helper": {"frame_args": models[mode]}}}))
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binaries[side.path])
    opts = pipeline.Options(pipeline.SideOpts("a", "0x1000", cc="ms"),
                            pipeline.SideOpts("b", "0x1000", cc="ms"),
                            backend="lockstep", no_alias=True, ret="i32", corr=str(corr))
    rep = pipeline.analyse(opts)
    expected = "EQUIVALENT" if difference == "outside" and mode != "absent" else "UNKNOWN"
    assert rep.verdict == expected, rep.reason


@pytest.mark.parametrize("length", [None, 0x100])
def test_symbolic_or_saved_slot_crossing_extent_stays_unknown(tmp_path, monkeypatch, length):
    code = bytes.fromhex("4883ec28 488d4c2420")
    if length is not None:
        code += b"\xba" + length.to_bytes(4, "little")
    code += b"\xe8" + (0x40 - len(code) - 5).to_bytes(4, "little")
    code += bytes.fromhex("31c0 4883c428 c3")
    helper = bytes.fromhex("8b01 01d0 c3")
    path = tmp_path / "a.bin"
    path.write_bytes(code + b"\xcc" * (0x40 - len(code)) + helper)
    binary = binmod.load(str(path), 0x1000, "x64")
    bounding.angr_project(binary)
    binary.fmt, binary.pdata_ranges = "pe", [(0x1000, 0x1000 + len(code), 0),
                                             (0x1040, 0x1040 + len(helper), 0)]
    corr = tmp_path / "corr.json"
    corr.write_text(json.dumps({"callees": [{"a": "0x1040", "b": "0x1040", "name": "helper"}],
                               "callee_contracts": {"helper": {"frame_args": [{"arg": 0, "size_arg": 1}]}}}))
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binary)
    opts = pipeline.Options(pipeline.SideOpts("a", "0x1000", cc="ms"),
                            pipeline.SideOpts("b", "0x1000", cc="ms"),
                            backend="lockstep", no_alias=True, ret="i32", corr=str(corr))
    rep = pipeline.analyse(opts)
    assert rep.verdict == "UNKNOWN", rep.reason
    assert "extent" in rep.reason
