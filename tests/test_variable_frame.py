"""Exhaustively bounded stack indexing, including conditional writes and guard use."""
from __future__ import annotations
import pytest

pytest.importorskip('angr')
from fneq import binary as binmod, pipeline


def compare(tmp_path, monkeypatch, a, b, cc='ms', arch='x64'):
    binaries = {}
    for tag, code in [('a', a), ('b', b)]:
        path = tmp_path / (tag + '.bin')
        path.write_bytes(code)
        binaries[tag] = binmod.load(str(path), 0x1000, arch)
    monkeypatch.setattr(pipeline, '_load_binary', lambda side: binaries[side.path])
    opts = pipeline.Options(pipeline.SideOpts('a', '0x1000', cc=cc),
                            pipeline.SideOpts('b', '0x1000', cc=cc),
                            backend='lockstep', no_alias=True, ret='i32')
    return pipeline.analyse(opts)


def table(values, x64=True):
    prefix = bytes.fromhex('4883ec28' if x64 else '83ec10')
    for i, value in enumerate(values):
        prefix += bytes.fromhex('c74424') + bytes([(0x10 if x64 else 0) + 4*i]) + value.to_bytes(4, 'little')
    return prefix


@pytest.mark.parametrize('changed', [False, True])
@pytest.mark.parametrize('guarded', [False, True])
def test_indexed_load_matches_arithmetic(tmp_path, monkeypatch, changed, guarded):
    # The guarded form needs the preceding branch condition to bound ECX.
    a = table([7, 8, 9, 10])
    a += bytes.fromhex('8b448c10 4883c428 c3')
    b = bytes.fromhex('8d4108c3' if changed else '8d4107c3')
    if guarded:
        def guard(code):
            return bytes.fromhex('83f903 77') + bytes([len(code)]) + code + bytes.fromhex('31c0c3')
        # A 32-bit comparison alone does not constrain RCX's high half.
        a, b = guard(bytes.fromhex('89c9') + a), guard(b)
    else:
        a, b = bytes.fromhex('83e103') + a, bytes.fromhex('83e103') + b
    rep = compare(tmp_path, monkeypatch, a, b)
    assert rep.verdict == ('UNKNOWN' if changed else 'EQUIVALENT'), rep.reason


@pytest.mark.parametrize('changed', [False, True])
def test_variable_store_keeps_unselected_bytes(tmp_path, monkeypatch, changed):
    a = bytes.fromhex('83e103') + table([0, 0, 0, 0])
    a += bytes.fromhex('c7448c102a000000 8b442410 4883c428 c3')
    value = 43 if changed else 42
    b = bytes.fromhex('83e103 b8') + value.to_bytes(4, 'little')
    b += bytes.fromhex('ba00000000 85c9 0f45c2 c3')
    rep = compare(tmp_path, monkeypatch, a, b)
    assert rep.verdict == ('UNKNOWN' if changed else 'EQUIVALENT'), rep.reason


def test_x86_variable_frame_load(tmp_path, monkeypatch):
    a = bytes.fromhex('83e103') + table([7, 8, 9, 10], False)
    a += bytes.fromhex('8b048c 83c410 c3')
    b = bytes.fromhex('83e103 8d4107 c3')
    rep = compare(tmp_path, monkeypatch, a, b, cc='fastcall', arch='x86')
    assert rep.verdict == 'EQUIVALENT', rep.reason


def test_low_dword_guard_does_not_bound_full_register(tmp_path, monkeypatch):
    body = table([7, 8, 9, 10]) + bytes.fromhex('8b448c10 4883c428 c3')
    a = bytes.fromhex('83f903 77') + bytes([len(body)]) + body + bytes.fromhex('31c0c3')
    rep = compare(tmp_path, monkeypatch, a, a)
    assert rep.verdict == 'UNKNOWN', rep.reason
    assert 'variable stack access' in rep.reason


@pytest.mark.parametrize('bounded', [False, True])
def test_unbounded_or_return_slot_access_stays_unknown(tmp_path, monkeypatch, bounded):
    # A four-element index at RSP+20h can reach the root return slot.
    a = (bytes.fromhex('83e103') if bounded else b'')
    a += bytes.fromhex('4883ec28 8b448c20 4883c428 c3')
    rep = compare(tmp_path, monkeypatch, a, a)
    assert rep.verdict == 'UNKNOWN', rep.reason
    assert 'variable stack access' in rep.reason
