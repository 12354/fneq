"""Bounded tail wrappers still consume the arguments forwarded to their targets."""
from __future__ import annotations

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding, callees, pipeline
from fneq.errors import Unsupported


@pytest.mark.parametrize('stack', [False, True])
@pytest.mark.parametrize('same', [False, True])
def test_conditional_tail_wrapper_keeps_forwarded_inputs(tmp_path, monkeypatch, stack, same):
    binaries = {}
    for tag, value in [('a',7),('b',7 if same else 8)]:
        root=bytes.fromhex('4883ec38 b901000000 ba')+value.to_bytes(4,'little')
        root+=bytes.fromhex('c7442420')+value.to_bytes(4,'little')
        root+=b'\xe8'+(0x40-len(root)-5).to_bytes(4,'little')+bytes.fromhex('4883c438 c3')
        wrapper=bytes.fromhex('4885c9 7405 e903000000 31c0 c3')
        helper=bytes.fromhex('8b442428 c3' if stack else '89d0 c3')
        path=tmp_path/(tag+'.bin');path.write_bytes(root+b'\xcc'*(0x40-len(root))+wrapper+helper)
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        b.pdata_ranges=[(0x1000,0x1000+len(root),0),(0x1040,0x104D,0),
                        (0x104D,0x104D+len(helper),0)]
        binaries[tag]=b
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i32')
    rep=pipeline.analyse(opts)
    assert rep.verdict==('EQUIVALENT' if same else 'UNKNOWN'),rep.reason


def test_partial_register_write_does_not_kill_forwarded_high_bits(tmp_path,monkeypatch):
    binaries={}
    for tag,value in [('a',0x1111222233334444),('b',0x5555666677774444)]:
        root=bytes.fromhex('4883ec28 48ba')+value.to_bytes(8,'little')
        root+=b'\xe8'+(0x40-len(root)-5).to_bytes(4,'little')+bytes.fromhex('4883c428 c3')
        stub=bytes.fromhex('6631d2 e900000000')
        helper=bytes.fromhex('4889d0 c3')
        path=tmp_path/(tag+'.bin');path.write_bytes(root+b'\xcc'*(0x40-len(root))+stub+helper)
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        b.pdata_ranges=[(0x1000,0x1000+len(root),0),(0x1040,0x1048,0),(0x1048,0x104C,0)]
        binaries[tag]=b
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i64')
    rep=pipeline.analyse(opts)
    assert rep.verdict=='UNKNOWN',rep.reason


def test_definition_on_one_branch_does_not_hide_an_input(tmp_path,monkeypatch):
    binaries={}
    for tag,value in [('a',7),('b',8)]:
        root=bytes.fromhex('4883ec28 85c9 ba')+value.to_bytes(4,'little')
        root+=b'\xe8'+(0x40-len(root)-5).to_bytes(4,'little')+bytes.fromhex('4883c428 c3')
        # Only one branch defines EDX before the shared tail jump.
        stub=bytes.fromhex('85c9 7405 ba07000000 e900000000')
        helper=bytes.fromhex('89d0 c3')
        path=tmp_path/(tag+'.bin');path.write_bytes(root+b'\xcc'*(0x40-len(root))+stub+helper)
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        end=0x1040+len(stub)
        b.pdata_ranges=[(0x1000,0x1000+len(root),0),(0x1040,end,0),(end,end+len(helper),0)]
        binaries[tag]=b
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i32')
    rep=pipeline.analyse(opts)
    assert rep.verdict=='UNKNOWN',rep.reason


@pytest.mark.parametrize('same',[False,True])
def test_conditional_fallthrough_to_import_thunk(tmp_path,monkeypatch,same):
    binaries={}
    for tag,value in [('a',7),('b',7 if same else 8)]:
        root=bytes.fromhex('4883ec38 b901000000 31d2 4531c0 41b9')+value.to_bytes(4,'little')
        root+=bytes.fromhex('48c744242000000000')
        root+=b'\xe8'+(0x40-len(root)-5).to_bytes(4,'little')+bytes.fromhex('4883c438 c3')
        wrapper=bytes.fromhex('4885c9 7406 ff2535000000 c3')
        data=root+b'\xcc'*(0x40-len(root))+wrapper
        data+=b'\xcc'*(0x80-len(data))+bytes(8)
        path=tmp_path/(tag+'.bin');path.write_bytes(data)
        b=binmod.load(str(path),0x1000,'x64');bounding.angr_project(b);b.fmt='pe'
        b.imports[0x1080]='kernel32.dll!WriteFile'
        b.pdata_ranges=[(0x1000,0x1000+len(root),0),(0x1040,0x1045,0),
                        (0x1045,0x104B,0),(0x104B,0x104C,0)]
        binaries[tag]=b
    monkeypatch.setattr(pipeline,'_load_binary',lambda side:binaries[side.path])
    opts=pipeline.Options(pipeline.SideOpts('a','0x1000',cc='ms'),pipeline.SideOpts('b','0x1000',cc='ms'),
                          backend='lockstep',no_alias=True,ret='i32')
    rep=pipeline.analyse(opts)
    assert rep.verdict==('EQUIVALENT' if same else 'UNKNOWN'),rep.reason


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("same_argument", [False, True])
def test_forwarded_argument_is_not_lost(tmp_path, monkeypatch, chain, same_argument):
    binaries = {}
    for tag, value in [("a", 7), ("b", 7 if same_argument else 8)]:
        root = bytes.fromhex("4883ec28 b9") + value.to_bytes(4, "little")
        root += bytes.fromhex("e812000000 4883c428 c3")
        stub = bytes.fromhex("4531c0 e900000000")  # xor r8d,r8d; jmp target
        stubs = stub + (bytes.fromhex("4531c9 e900000000") if chain else b"")
        helper = bytes.fromhex("8d4101 c3")  # lea eax,[rcx+1]; ret
        path = tmp_path / f"{tag}.bin"
        path.write_bytes(root + b"\xcc" * (0x20 - len(root)) + stubs + helper)
        b = binmod.load(str(path), 0x1000, "x64")
        bounding.angr_project(b)
        b.fmt = "pe"
        end = 0x1020 + len(stubs)
        b.pdata_ranges = [(0x1000, 0x1000 + len(root), 0), (0x1020, 0x1028, 0)]
        if chain:
            b.pdata_ranges.append((0x1028, 0x1030, 0))
        b.pdata_ranges.append((end, end + len(helper), 0))
        binaries[tag] = b
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binaries[side.path])
    opts = pipeline.Options(pipeline.SideOpts("a", "0x1000", cc="ms"),
                            pipeline.SideOpts("b", "0x1000", cc="ms"),
                            backend="lockstep", no_alias=True, ret="i32")
    rep = pipeline.analyse(opts)
    assert rep.verdict == ("EQUIVALENT" if same_argument else "UNKNOWN"), rep.reason


@pytest.mark.parametrize("cycle", [False, True])
def test_unresolved_wrapper_chain_does_not_invent_an_arity(tmp_path, cycle):
    if cycle:
        data = bytes.fromhex("4531c0 e9f8ffffff")
    else:
        data = bytes.fromhex("4531c0 e900000000") * 9 + bytes.fromhex("8d4101 c3")
    path = tmp_path / "unresolved.bin"
    path.write_bytes(data)
    binary = binmod.load(str(path), 0x1000, "x64")
    with pytest.raises(Unsupported, match="tail-wrapper argument inference"):
        callees._stub_info(binary, 0x1000)


@pytest.mark.parametrize("import_name,same_argument,expected", [
    ("KERNEL32.dll!WriteFile", True, "EQUIVALENT"),
    ("KERNEL32.dll!WriteFile", False, "UNKNOWN"),
    ("custom.dll!unprototyped", True, "UNKNOWN"),
])
def test_import_tail_wrapper_uses_sdk_arity(tmp_path, monkeypatch, import_name, same_argument, expected):
    binaries = {}
    for tag, value in [("a", 7), ("b", 7 if same_argument else 8)]:
        root = bytes.fromhex("4883ec28 41b9") + value.to_bytes(4, "little")
        root += bytes.fromhex("48c744242000000000 e828000000 4883c428 c3")
        stub = bytes.fromhex("4531c0 e900000000 ff2532000000")
        data = root + b"\xcc" * (0x40 - len(root)) + stub
        data += b"\xcc" * (0x80 - len(data)) + bytes(8)
        path = tmp_path / f"{tag}.bin"
        path.write_bytes(data)
        binary = binmod.load(str(path), 0x1000, "x64")
        bounding.angr_project(binary)
        binary.fmt = "pe"
        binary.imports[0x1080] = import_name
        binary.pdata_ranges = [(0x1000, 0x1000 + len(root), 0),
                               (0x1040, 0x1048, 0), (0x1048, 0x104E, 0)]
        binaries[tag] = binary
    monkeypatch.setattr(pipeline, "_load_binary", lambda side: binaries[side.path])
    opts = pipeline.Options(pipeline.SideOpts("a", "0x1000", cc="ms"),
                            pipeline.SideOpts("b", "0x1000", cc="ms"),
                            backend="lockstep", no_alias=True, ret="i32")
    rep = pipeline.analyse(opts)
    assert rep.verdict == expected, rep.reason
