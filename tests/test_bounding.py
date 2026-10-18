"""Function recovery in a restricted PE code region (especially tail wrappers)."""
from __future__ import annotations

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding


def restricted_pe(tmp_path, code, entry_offset, region_end):
    path = tmp_path / "code.bin"
    path.write_bytes(code)
    binary = binmod.load(str(path), 0x1000, "x64")
    # Use a blob-backed angr project with PE-style function boundaries. This
    # isolates CFG recovery from a linker/SDK and from the actual Parsec image.
    bounding.angr_project(binary)
    binary.fmt = "pe"
    binary.pdata_ranges = [(0x1000 + entry_offset, 0x1000 + region_end, 0)]
    return binary


@pytest.mark.parametrize("adjust", [b"", bytes.fromhex("4531c0")])
def test_restricted_backward_tail_wrapper(tmp_path, adjust):
    # target: lea eax,[rcx+rdx]; ret; padding; wrapper: [xor r8d,r8d;] jmp target
    prefix = bytes.fromhex("8d0411c3") + b"\xcc" * 12
    jump_at = 0x1010 + len(adjust)
    wrapper = adjust + b"\xe9" + (0x1000 - jump_at - 5).to_bytes(4, "little", signed=True)
    binary = restricted_pe(tmp_path, prefix + wrapper, 16, 16 + len(wrapper))
    info = bounding.bound(binary, 0x1010)
    assert b"".join(info.code.values()) == wrapper
    assert not info.hazards
    assert [(c.addr, c.target, c.tail) for c in info.calls] == [(jump_at, 0x1000, True)]


def test_restricted_conditional_tail_wrapper(tmp_path):
    # cmp ecx,0; je target; jmp target, with both tails outside the region.
    prefix = bytes.fromhex("31c0c3") + b"\xcc" * 13
    wrapper = bytes.fromhex("83f90074eb") + b"\xe9" + (-26).to_bytes(4, "little", signed=True)
    binary = restricted_pe(tmp_path, prefix + wrapper, 16, 26)
    info = bounding.bound(binary, 0x1010)
    assert info.entry in info.code
    assert not info.hazards
    assert any(c.target == 0x1000 and c.tail for c in info.calls)


def test_restricted_system_instruction_stays_unsupported(tmp_path):
    binary = restricted_pe(tmp_path, bytes.fromhex("0f05c3"), 0, 3)
    info = bounding.bound(binary, 0x1000)
    assert any("system instruction" in h for h in info.hazards)


def test_local_buffer_is_not_an_incoming_stack_argument(tmp_path):
    # push rbx; sub rsp,0x80; lea rcx,[rsp+0x50]; call external;
    # add rsp,0x80; pop rbx; ret; external: ret
    body = bytes.fromhex("53 4881ec80000000 488d4c2450 e909000000")
    # Use a CALL, so the buffer setup and stack adjustment are in one VEX block.
    body = body[:-5] + bytes.fromhex("e809000000 4881c480000000 5b c3")
    binary = restricted_pe(tmp_path, body + b"\xc3", 0, len(body))
    info = bounding.bound(binary, 0x1000)
    assert info.max_stack_arg == 0
    assert info.min_sp == -0x88


def test_real_incoming_stack_argument_survives_stack_adjustment(tmp_path):
    # sub rsp,0x88; mov rax,[rsp+0xb0]; add rsp,0x88; ret
    body = bytes.fromhex("4881ec88000000 488b8424b0000000 4881c488000000 c3")
    binary = restricted_pe(tmp_path, body, 0, len(body))
    info = bounding.bound(binary, 0x1000)
    assert info.max_stack_arg == 0x28
    assert info.min_sp == -0x88
    assert info.stack_read_sizes[0x28] == 8


def test_stack_argument_width_includes_alias_reads(tmp_path):
    # eax = [rsp+0x28]; r11 = rsp; edx = [r11+0x2c]; ret
    body = bytes.fromhex("8b442428 4c8bdc 418b532c c3")
    binary = restricted_pe(tmp_path, body, 0, len(body))
    info = bounding.bound(binary, 0x1000)
    assert info.stack_read_sizes[0x28] == 8


def test_incoming_rbp_is_not_assumed_to_be_a_frame_pointer(tmp_path):
    body = bytes.fromhex("8b4528 c3")    # mov eax,[rbp+0x28]; ret
    binary = restricted_pe(tmp_path, body, 0, len(body))
    info = bounding.bound(binary, 0x1000)
    assert info.max_stack_arg == 0
    assert not info.stack_read_sizes
