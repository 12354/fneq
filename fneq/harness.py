"""Concrete execution of one function in Unicorn, in the shared abstract environment."""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Callable

import unicorn as uc
from unicorn import x86_const as R

from fneq.binary import Binary
from fneq.model import (DEFAULT_MXCSR, DEFAULT_X87_CW, RET_ADDR,
                        STACK_TOP, CalleeSpec)
from fneq.signature import Signature

PAGE = 0x1000

_REG = {name: getattr(R, f"UC_X86_REG_{name}") for name in (
    "EAX ECX EDX EBX ESP EBP ESI EDI EIP RAX RCX RDX RBX RSP RBP RSI RDI RIP "
    "R8 R9 R10 R11 R12 R13 R14 R15 AL AX FPCW MXCSR EFLAGS FS GS".split())}
for _i in range(16):
    _REG[f"XMM{_i}"] = getattr(R, f"UC_X86_REG_XMM{_i}")
for _i in range(8):
    _REG[f"ST{_i}"] = getattr(R, f"UC_X86_REG_ST{_i}")


@dataclasses.dataclass
class Environment:
    """Everything outside the function that both sides must agree on."""
    memory_seed: int = 0
    x87_cw: int = DEFAULT_X87_CW
    mxcsr: int = DEFAULT_MXCSR
    max_insns: int = 50_000
    # Explicit memory contents (address -> byte); overrides the seeded background.
    memory: dict[int, int] = dataclasses.field(default_factory=dict)
    # Globals mapped to the other side: (address here, address in side-A space, size).
    regions: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)

    def space(self, addr: int) -> int:
        """Address in the shared (side-A) space, used to derive background content."""
        for here, a, size in self.regions:
            if here <= addr < here + size:
                return a + (addr - here)
        return addr


@dataclasses.dataclass
class RunResult:
    outputs: dict[str, int]
    writes: list[tuple[int, int, int]]        # (address, size, value), outside the frame
    calls: list[tuple[str, tuple[int, ...]]]  # (callee name, argument values)
    coverage: set[int]
    unmapped: list[dict]                      # first-touch accesses outside the image/stack
    fault: str | None                         # e.g. "divide error", "invalid instruction"
    exhausted: bool                           # stopped by the instruction limit
    final_bytes: dict[int, int]               # value of every byte written outside the frame
    read_byte: Callable[[int], int]           # final memory (background where untouched)

    def summary(self) -> dict:
        return {"outputs": {k: hex(v) for k, v in self.outputs.items()},
                "writes": [(hex(a), s, hex(v)) for a, s, v in self.writes],
                "calls": [(n, [hex(x) for x in a]) for n, a in self.calls],
                "fault": self.fault, "exhausted": self.exhausted}


def background_byte(seed: int, addr: int) -> int:
    """Deterministic 'arbitrary' memory content shared by both sides."""
    h = hashlib.blake2b(f"{seed}:{addr >> 6}".encode(), digest_size=64).digest()
    return h[addr & 63]


def callee_result(name: str, reg: str, args: tuple[int, ...], bits: int) -> int:
    """Deterministic result of an uninterpreted callee: a function of its arguments."""
    h = hashlib.blake2b(repr((name, reg, args)).encode(), digest_size=16).digest()
    return int.from_bytes(h, "little") & ((1 << bits) - 1)


def _read(mu, name: str) -> int:
    """Read NAME or NAME:BITS (the low BITS bits)."""
    reg, _, bits = name.partition(":")
    if reg.startswith("ST"):
        # Unicorn's ST<i> returns only the 64-bit mantissa; the full 80-bit value is
        # the physical register FP[(TOP + i) & 7], TOP = FPSW bits 11..13.
        top = (mu.reg_read(R.UC_X86_REG_FPSW) >> 11) & 7
        phys = (top + int(reg[2:])) & 7
        val = _f80_to_int(mu.reg_read(getattr(R, f"UC_X86_REG_FP{phys}")))
    else:
        val = mu.reg_read(_REG[reg])
    return val & ((1 << int(bits)) - 1) if bits else val


def _f80_to_int(val) -> int:
    if isinstance(val, tuple):           # (mantissa, exponent) from unicorn
        mant, exp = val
        return (exp << 64) | mant
    return int(val)


_MEM_OPERANDS: dict[tuple, list] = {}


def _memory_operands(arch: str, code: dict[int, bytes]) -> list[tuple]:
    """(insn address, base, index, scale, disp) of every non-RIP-relative memory operand."""
    key = (arch, tuple(sorted((a, d) for a, d in code.items())))
    if key not in _MEM_OPERANDS:
        import capstone
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64 if arch == "x64" else capstone.CS_MODE_32)
        md.detail = True
        out = []
        for start, data in code.items():
            for insn in md.disasm(data, start):
                if insn.mnemonic in ("lea", "nop") or insn.mnemonic.startswith("prefetch"):
                    continue
                for op in insn.operands:
                    if op.type == capstone.x86.X86_OP_MEM:
                        m = op.mem
                        base = insn.reg_name(m.base).upper() if m.base else None
                        if base in ("RIP", "RSP"):
                            continue  # RIP/RSP-relative: never non-canonical here
                        index = insn.reg_name(m.index).upper() if m.index else None
                        out.append((insn.address, base, index, m.scale, m.disp))
        _MEM_OPERANDS[key] = out
    return _MEM_OPERANDS[key]


def run(binary: Binary, entry: int, sig: Signature, inputs: dict[str, int],
        callees: dict[int, CalleeSpec], env: Environment,
        code: dict[int, bytes] | None = None) -> RunResult:
    arch = sig.arch
    word = sig.word
    bits = 8 * word
    mask = (1 << bits) - 1
    mu = uc.Uc(uc.UC_ARCH_X86, uc.UC_MODE_32 if arch == "x86" else uc.UC_MODE_64)

    mapped: set[int] = set()

    def bg(addr: int) -> int:
        return background_byte(env.memory_seed, env.space(addr))

    def map_page(page: int, fill_background: bool) -> None:
        if page in mapped:
            return
        mu.mem_map(page, PAGE, uc.UC_PROT_ALL)
        mapped.add(page)
        if fill_background:
            mu.mem_write(page, bytes(bg(page + i) for i in range(PAGE)))

    # Image sections. Read-only sections keep their contents (constants, code);
    # writable data is an input like all other memory, so it gets the shared
    # arbitrary background, exactly as in the symbolic backends.
    for sec in binary.sections:
        start = sec.addr & ~(PAGE - 1)
        end = (sec.end + PAGE - 1) & ~(PAGE - 1)
        for page in range(start, end, PAGE):
            map_page(page, True)
        if not sec.writable:
            mu.mem_write(sec.addr, sec.data + bytes(sec.vsize - len(sec.data)))

    # Stack: background content (the caller's garbage), then RA and arguments.
    sp0 = STACK_TOP[arch]
    frame_lo = sp0 - sig.frame_below
    frame_hi = sp0 + sig.frame_above
    for page in range((frame_lo - PAGE) & ~(PAGE - 1), ((frame_hi + PAGE) & ~(PAGE - 1)) + PAGE, PAGE):
        map_page(page, True)
    for addr, val in env.memory.items():
        map_page(addr & ~(PAGE - 1), True)
        mu.mem_write(addr, bytes([val]))
    ra = RET_ADDR[arch]
    mu.mem_write(sp0, ra.to_bytes(word, "little"))
    for slot in sig.inputs:
        v = inputs.get(slot.name, 0)
        if slot.kind == "stack":
            mu.mem_write(sp0 + slot.offset, (v & mask).to_bytes(word, "little"))
        else:
            mu.reg_write(_REG[slot.name], v)
    mu.reg_write(_REG["ESP" if arch == "x86" else "RSP"], sp0)
    mu.reg_write(_REG["FPCW"], env.x87_cw)
    mu.reg_write(_REG["MXCSR"], env.mxcsr)
    mu.reg_write(_REG["EFLAGS"], 0x2)

    # Pages for callees and the return address: execution there is intercepted.
    map_page(ra & ~(PAGE - 1), False)
    for c in callees.values():
        if (c.addr & ~(PAGE - 1)) not in mapped:
            map_page(c.addr & ~(PAGE - 1), False)

    writes: list[tuple[int, int, int]] = []
    calls: list[tuple[str, tuple[int, ...]]] = []
    coverage: set[int] = set()
    unmapped: list[dict] = []
    state = {"fault": None, "insns": 0}
    sp_reg = _REG["ESP" if arch == "x86" else "RSP"]
    pc_reg = _REG["EIP" if arch == "x86" else "RIP"]

    def in_frame(addr: int) -> bool:
        return frame_lo <= addr < frame_hi

    limit = 1 << 47 if arch == "x64" else 1 << 32

    def on_write(mu_, access, addr, size, value, _):
        if addr + size > limit:
            return  # faults; never happens
        if not in_frame(addr):
            writes.append((addr, size, value & ((1 << (8 * size)) - 1)))

    def on_unmapped(mu_, access, addr, size, value, _):
        kind = {uc.UC_MEM_READ_UNMAPPED: "read", uc.UC_MEM_WRITE_UNMAPPED: "write",
                uc.UC_MEM_FETCH_UNMAPPED: "fetch"}.get(access, str(access))
        pc = mu_.reg_read(pc_reg)
        unmapped.append({"address": addr, "size": size, "kind": kind, "pc": pc})
        if kind == "fetch" or addr + size > limit:
            return False  # outside the canonical user range: a fault, as in the model
        # Memory outside the image is part of the input: map it with the shared
        # background content and continue.
        for page in range(addr & ~(PAGE - 1), ((addr + size - 1) & ~(PAGE - 1)) + 1, PAGE):
            map_page(page, True)
        return True

    def on_block(mu_, addr, size, _):
        coverage.add(addr)

    def on_code(mu_, addr, size, _):
        if addr == ra:
            mu_.emu_stop()
            return
        callee = callees.get(addr)
        if callee is None:
            return
        sp = mu_.reg_read(sp_reg)
        args = [_read(mu_, r) for r in callee.reg_args]
        for off in callee.stack_offsets(word):
            args.append(int.from_bytes(mu_.mem_read(sp + off, word), "little"))
        args_t = tuple(args)
        calls.append((callee.name, args_t))
        for r in callee.results:
            mu_.reg_write(_REG[r], callee_result(callee.name, r, args_t, bits))
        ret_to = int.from_bytes(mu_.mem_read(sp, word), "little")
        mu_.reg_write(sp_reg, (sp + word + callee.pop) & mask)
        mu_.reg_write(pc_reg, ret_to)

    def on_edge(mu_, access, addr, size, value, _):
        # An access that runs past the end of the user address space faults in
        # the model; QEMU would wrap it around instead.
        if addr + size > limit:
            state["fault"] = "access beyond the end of the address space"
            mu_.emu_stop()

    mu.hook_add(uc.UC_HOOK_MEM_READ | uc.UC_HOOK_MEM_WRITE, on_edge, begin=limit - 16, end=limit - 1)
    if arch == "x64" and code:
        # QEMU truncates non-canonical addresses to 48 bits before any memory hook
        # sees them, so compute effective addresses of the function's memory
        # operands ourselves and fault like the CPU (and the model) does.
        operands: dict[int, list] = {}
        for addr, base, index, scale, disp in _memory_operands(arch, code):
            operands.setdefault(addr, []).append((base, index, scale, disp))

        def on_mem_insn(mu_, addr, size, _):
            for base, index, scale, disp in operands.get(addr, ()):
                ea = disp
                if base:
                    ea += mu_.reg_read(_REG[base]) if base in _REG else 0
                if index:
                    ea += (mu_.reg_read(_REG[index]) if index in _REG else 0) * scale
                ea &= (1 << 64) - 1
                if ea >= limit:
                    state["fault"] = "non-canonical / kernel address"
                    mu_.emu_stop()
                    return

        for addr in operands:
            mu.hook_add(uc.UC_HOOK_CODE, on_mem_insn, begin=addr, end=addr)
    mu.hook_add(uc.UC_HOOK_MEM_WRITE, on_write)
    mu.hook_add(uc.UC_HOOK_MEM_UNMAPPED, on_unmapped)
    mu.hook_add(uc.UC_HOOK_BLOCK, on_block)
    mu.hook_add(uc.UC_HOOK_CODE, on_code, begin=ra, end=ra)
    for c in callees.values():
        mu.hook_add(uc.UC_HOOK_CODE, on_code, begin=c.addr, end=c.addr)

    exhausted = False
    try:
        mu.emu_start(entry, ra, count=env.max_insns)
        if mu.reg_read(pc_reg) != ra:
            exhausted = True
    except uc.UcError as e:
        state["fault"] = {uc.UC_ERR_EXCEPTION: "cpu exception (e.g. divide error)",
                          uc.UC_ERR_INSN_INVALID: "invalid instruction",
                          uc.UC_ERR_FETCH_UNMAPPED: "jump to unmapped code"}.get(e.errno, str(e))

    outputs: dict[str, int] = {}
    if state["fault"]:
        # A fault ends the function: register outputs are zero (as in the model).
        outputs = {name: 0 for name in sig.outputs}
        outputs.update(sp_delta=0, return_address=0, fault=1)
    else:
        for name in sig.outputs:
            outputs[name] = _read(mu, name)
        outputs["sp_delta"] = (mu.reg_read(sp_reg) - sp0) & mask
        outputs["return_address"] = mu.reg_read(pc_reg)
        outputs["fault"] = 0

    final_bytes: dict[int, int] = {}
    for addr, size, _ in writes:
        for a in range(addr, addr + size):
            final_bytes[a] = mu.mem_read(a, 1)[0]
    def read_byte(a: int) -> int:
        page = a & ~(PAGE - 1)
        if page not in mapped:
            return bg(a)
        return mu.mem_read(a, 1)[0]

    return RunResult(outputs, writes, calls, coverage, unmapped, state["fault"], exhausted,
                     final_bytes, read_byte)
