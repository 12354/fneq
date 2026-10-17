"""Find a function's full extent and the facts later stages need.

Bounding is delegated to angr's CFGFast (tail calls, jump tables, padding). On top of
its result we collect what fneq needs: call sites and their callees, `ret N` values,
register/stack-argument usage for convention inference, comparison constants for the
fuzzer, and hazards that force an UNKNOWN verdict.
"""
from __future__ import annotations

import dataclasses
import logging

from fneq.binary import Binary, undecorate
from fneq.errors import Unsupported

for _name in ("angr", "cle", "pyvex", "claripy"):
    logging.getLogger(_name).setLevel(logging.ERROR)


@dataclasses.dataclass
class CallSite:
    addr: int              # address of the call/jmp instruction
    target: int | None     # None for indirect calls
    name: str              # symbol/import name, "self" for recursion, or hex address
    tail: bool = False     # jmp used as a call (tail call)


@dataclasses.dataclass
class FunctionInfo:
    entry: int
    name: str
    arch: str
    code: dict[int, bytes]                     # block start -> bytes
    calls: list[CallSite]
    ret_pops: set[int]                         # N of every `ret N` (0 for plain ret)
    jump_tables: dict[int, list[int]]          # indirect jmp addr -> targets
    hazards: list[str]                         # reasons for UNKNOWN
    notes: list[str]
    constants: set[int]                        # immediates (fuzzing seeds)
    data_refs: set[int]                        # absolute / RIP-relative data addresses
    regs_read_first: set[str]                  # canonical registers read before written
    read_widths: dict[str, int]                # canonical register -> widest read (bits)
    max_stack_arg: int                         # highest byte offset above the RA that is read
    min_sp: int                                # lowest SP offset reached (<= 0)
    uses_x87: bool
    x87_arith: bool
    sets_fpu_cw: bool
    has_loop: bool = False
    uses_fp: bool = False          # x87 or XMM registers
    pointer_access: bool = False   # memory accessed through a non-stack, non-RIP base/index

    @property
    def size(self) -> int:
        return sum(len(b) for b in self.code.values())

    def code_ranges(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for start in sorted(self.code):
            end = start + len(self.code[start])
            if out and start <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], end))
            else:
                out.append((start, end))
        return out


# ------------------------------------------------------------ angr project

_PROJECTS: dict[str, object] = {}


def angr_project(binary: Binary):
    import angr

    key = binary.path
    if key not in _PROJECTS:
        if binary.fmt == "raw":
            arch = "x86" if binary.arch == "x86" else "amd64"
            _PROJECTS[key] = angr.Project(
                binary.path, auto_load_libs=False,
                main_opts={"backend": "blob", "arch": arch, "base_addr": binary.image_base,
                           "entry_point": binary.image_base})
        else:
            _PROJECTS[key] = angr.Project(binary.path, auto_load_libs=False)
    return _PROJECTS[key]


# ------------------------------------------------------- register helpers

_FAMILIES = {
    "RAX": "rax eax ax al ah", "RBX": "rbx ebx bx bl bh", "RCX": "rcx ecx cx cl ch",
    "RDX": "rdx edx dx dl dh", "RSI": "rsi esi si sil", "RDI": "rdi edi di dil",
    "RBP": "rbp ebp bp bpl", "RSP": "rsp esp sp spl",
}
CANON = {alias: fam for fam, names in _FAMILIES.items() for alias in names.split()}
for _n in range(8, 16):
    for _suffix in ("", "d", "w", "b"):
        CANON[f"r{_n}{_suffix}"] = f"R{_n}"
for _n in range(32):
    for _p in ("xmm", "ymm", "zmm"):
        CANON[f"{_p}{_n}"] = f"XMM{_n}"


def _full_write(name: str, arch: str) -> bool:
    """Does writing this register name define the whole canonical register?"""
    if name.startswith(("xmm", "ymm", "zmm")):
        return True
    if arch == "x64":
        return name.startswith("r") and not name.endswith(("w", "b")) or name.startswith("e")
    return name.startswith("e")


def _is_zero_idiom(insn) -> bool:
    ops = insn.op_str.replace(" ", "").split(",")
    return (insn.mnemonic in ("xor", "sub", "pxor", "xorps", "xorpd", "vpxor", "vxorps")
            and len(ops) >= 2 and ops[0] == ops[1])


# ----------------------------------------------------------- the analysis

def bound(binary: Binary, entry: int) -> FunctionInfo:
    proj = angr_project(binary)
    name = binary.symbol_at(entry) or f"sub_{entry:x}"
    if not binary.is_exec(entry):
        raise Unsupported(f"0x{entry:x} is not in an executable section")

    exec_regions = [(s.addr, s.end) for s in binary.sections if s.executable]
    cfg = proj.analyses.CFGFast(
        function_starts=[entry], start_at_entry=False, force_complete_scan=False,
        regions=exec_regions, normalize=True, resolve_indirect_jumps=True,
        function_prologues=False, symbols=binary.fmt != "raw", data_references=False)
    funcs = cfg.kb.functions
    if entry not in funcs:
        raise Unsupported(f"angr could not recover a function at 0x{entry:x}")
    func = funcs[entry]

    # GCC's hot/cold splitting puts part of a function in `<name>.cold`; angr sees a
    # transition to a separate function. Merge those parts into this one.
    parts = [func]
    base_name = undecorate(name)
    for f in list(funcs.values()):
        sym = binary.symbol_at(f.addr) or ""
        if f.addr != entry and undecorate(sym).startswith(base_name + ".cold"):
            parts.append(f)
    part_addrs = {p.addr for p in parts}

    hazards: list[str] = []
    notes: list[str] = []
    code: dict[int, bytes] = {}
    calls: list[CallSite] = []
    ret_pops: set[int] = set()
    constants: set[int] = set()
    data_refs: set[int] = set()
    uses_x87 = x87_arith = sets_cw = False
    uses_fp = pointer_access = False
    insns_by_block: dict[int, list] = {}

    def callee_name(target: int | None) -> str:
        if target is None:
            return "<indirect>"
        if target == entry:
            return "self"
        if target in binary.imports:
            return binary.imports[target]
        sym = binary.symbol_at(target)
        if sym:
            return undecorate(sym)
        plt = funcs.get(target)
        if plt is not None and plt.is_plt and plt.name:
            return plt.name
        return f"0x{target:x}"

    n_insns = 0
    for part in parts:
        for block in part.blocks:
            data = binary.read(block.addr, block.size)
            if data is None:
                hazards.append(f"block 0x{block.addr:x} lies outside the image")
                continue
            code[block.addr] = data
            insns = block.capstone.insns
            insns_by_block[block.addr] = insns
            n_insns += len(insns)
            for ci in insns:
                insn = ci.insn
                _scan_insn(binary, insn, hazards, notes, constants, data_refs)
                mnem = insn.mnemonic
                if mnem.startswith("f") and mnem not in ("fence",):
                    uses_x87 = True
                    if mnem.rstrip("p").rstrip("r") in ("fadd", "fsub", "fmul", "fdiv",
                                                        "fiadd", "fisub", "fimul", "fidiv",
                                                        "fsqrt", "fsubr", "fdivr", "fprem",
                                                        "fprem1", "fscale", "frndint") \
                            or mnem in ("fsqrt", "fsin", "fcos", "fsincos", "fptan",
                                        "fpatan", "f2xm1", "fyl2x", "fyl2xp1"):
                        x87_arith = True
                    if mnem == "fldcw":
                        sets_cw = True
                if not uses_fp and "xmm" in insn.op_str:
                    uses_fp = True
                if not pointer_access:
                    for op in insn.operands:
                        if op.type == 3:  # memory
                            base = insn.reg_name(op.mem.base) if op.mem.base else None
                            if op.mem.index or (base and base not in (
                                    "esp", "rsp", "ebp", "rbp", "rip")):
                                pointer_access = True
                if mnem in ("ret", "retn", "retf"):
                    ret_pops.add(insn.operands[0].imm if insn.operands else 0)
    if n_insns > 20000:
        raise Unsupported(f"function too large ({n_insns} instructions)")

    # Call sites (direct, import, indirect) and tail calls out of the function.
    for part in parts:
        for site in part.get_call_sites():
            target = part.get_call_target(site)
            ins_addr = _last_insn_addr(insns_by_block.get(site))
            if target is not None and target in funcs and funcs[target].is_plt:
                pass
            ind = target is None or _is_indirect(insns_by_block.get(site))
            if ind:
                imp = _import_slot(binary, insns_by_block.get(site))
                if imp is not None:
                    calls.append(CallSite(ins_addr, imp, binary.imports[imp]))
                    continue
                hazards.append(f"indirect call at 0x{ins_addr:x}")
                calls.append(CallSite(ins_addr, None, "<indirect>"))
                continue
            calls.append(CallSite(ins_addr, target, callee_name(target)))
        for src, dst, data in part.transition_graph.edges(data=True):
            if data.get("outside") and data.get("type") == "transition":
                tgt = dst.addr
                if tgt in part_addrs:
                    continue
                last = _last_insn_addr(insns_by_block.get(src.addr))
                calls.append(CallSite(last, tgt, callee_name(tgt), tail=True))

    # Jump tables and unresolved indirect jumps inside the function.
    jump_tables: dict[int, list[int]] = {}
    block_starts = set(code)
    for ij_addr, ij in cfg.indirect_jumps.items():
        if ij.func_addr not in part_addrs:
            continue
        if ij.resolved_targets and not ij.jumpkind == "Ijk_Call":
            jump_tables[ij.ins_addr] = sorted(ij.resolved_targets)
        elif ij.jumpkind != "Ijk_Call":
            insns = insns_by_block.get(ij_addr)
            if _import_slot(binary, insns) is not None:
                continue  # import thunk / tail call through IAT
            hazards.append(f"unresolved indirect jump at 0x{ij.ins_addr:x}")

    widths: dict[str, int] = {}
    regs_first, max_arg, min_sp = _usage(binary, func, insns_by_block, code, notes, widths)

    if binary.fmt == "pe" and binary.arch == "x64":
        unwind = binary.pdata.get(entry)
        if unwind is not None:
            flags = (binary.read_int(unwind, 1) or 0) >> 3
            if flags & 0x3:
                hazards.append("function has a structured exception handler (x64 unwind info)")

    import networkx as nx
    has_loop = any(not nx.is_directed_acyclic_graph(p.graph) for p in parts)

    return FunctionInfo(entry, name, binary.arch, code, calls, ret_pops or {0}, jump_tables,
                        hazards, notes, constants, data_refs, regs_first, widths, max_arg,
                        min_sp, uses_x87, x87_arith, sets_cw, has_loop,
                        uses_fp or uses_x87, pointer_access)


def _last_insn_addr(insns) -> int:
    return insns[-1].insn.address if insns else 0


def _is_indirect(insns) -> bool:
    if not insns:
        return True
    op = insns[-1].insn.operands
    return not op or op[0].type != 2  # X86_OP_IMM == 2


def _import_slot(binary: Binary, insns) -> int | None:
    """If the block ends in call/jmp [slot] with slot an import, return the slot."""
    if not insns:
        return None
    insn = insns[-1].insn
    if not insn.operands or insn.operands[0].type != 3:  # X86_OP_MEM
        return None
    mem = insn.operands[0].mem
    if mem.index != 0:
        return None
    if mem.base == 0:
        slot = mem.disp & ((1 << (8 * binary.ptr_size)) - 1)
    elif insn.reg_name(mem.base) == "rip":
        slot = insn.address + insn.size + mem.disp
    else:
        return None
    return slot if slot in binary.imports else None


def _scan_insn(binary: Binary, insn, hazards, notes, constants, data_refs) -> None:
    """Collect constants/data references and flag hazards for one instruction."""
    mnem = insn.mnemonic
    if mnem in ("int", "int1", "into", "syscall", "sysenter", "in", "out", "ins", "outs",
                "hlt", "iret", "iretd", "iretq"):
        hazards.append(f"privileged/system instruction '{mnem}' at 0x{insn.address:x}")
    if mnem in ("rdtsc", "rdtscp", "rdrand", "rdseed", "cpuid"):
        hazards.append(f"non-deterministic instruction '{mnem}' at 0x{insn.address:x}")
    for i, op in enumerate(insn.operands):
        if op.type == 2:  # immediate
            constants.add(op.imm)
        elif op.type == 3:  # memory
            mem = op.mem
            seg = insn.reg_name(mem.segment) if mem.segment else None
            if seg == "fs" and binary.arch == "x86" and mem.base == 0 and mem.disp == 0:
                hazards.append(f"structured exception handling frame (fs:[0]) at 0x{insn.address:x}")
            addr = None
            if mem.base == 0 and mem.index == 0 and seg is None:
                addr = mem.disp & ((1 << (8 * binary.ptr_size)) - 1)
            elif mem.base and insn.reg_name(mem.base) == "rip" and mem.index == 0:
                addr = insn.address + insn.size + mem.disp
            if addr is not None:
                data_refs.add(addr)
                # A store to a constant address inside code is self-modifying code.
                written = i == 0 and mnem not in ("cmp", "test", "push", "call", "jmp", "ucomisd",
                                                  "ucomiss", "comisd", "comiss", "bt")
                if written and binary.is_exec(addr):
                    hazards.append(f"write into code at 0x{addr:x} (self-modifying code) "
                                   f"from 0x{insn.address:x}")


# Registers a call defines in the caller (union over the conventions we support).
CALL_CLOBBERS = {
    "x86": {"RAX", "RCX", "RDX"} | {f"XMM{i}" for i in range(8)},
    "x64": {"RAX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "R10", "R11"}
           | {f"XMM{i}" for i in range(16)},
}


def _signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _reg_bits(name: str) -> int:
    if name.startswith(("xmm", "ymm", "zmm")):
        return 128
    if name.startswith("r") and not name.endswith(("d", "w", "b")) and name != "r":
        return 64
    if name.startswith("e") or name.endswith("d"):
        return 32
    if name.endswith("l") or name.endswith("h") or name.endswith("b"):
        return 8
    return 16


def _usage(binary: Binary, func, insns_by_block, code, notes: list[str],
           widths: dict[str, int]) -> tuple[set[str], int, int]:
    """Registers read before written (any path), stack-argument extent, SP depth."""
    import angr  # noqa: F401  (analysis registry)

    arch = binary.arch
    word = binary.ptr_size
    # Must-defined dataflow over the function's own blocks.
    graph = func.graph
    order = sorted(insns_by_block)
    defined_in: dict[int, set[str] | None] = {a: None for a in order}
    defined_in[func.addr] = {"RSP"}
    read_first: set[str] = set()
    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        for addr in order:
            d = defined_in.get(addr)
            if d is None:
                continue
            cur = set(d)
            for ci in insns_by_block[addr]:
                insn = ci.insn
                try:
                    reads, writes = insn.regs_access()
                except Exception:  # capstone may not support every instruction
                    reads, writes = [], []
                if insn.mnemonic != "push" and not _is_zero_idiom(insn):
                    for r in reads:
                        nm = insn.reg_name(r)
                        c = CANON.get(nm)
                        if c and c not in cur and c != "RSP":
                            read_first.add(c)
                            widths[c] = max(widths.get(c, 0), _reg_bits(nm))
                for r in writes:
                    nm = insn.reg_name(r)
                    c = CANON.get(nm)
                    if c and _full_write(nm, arch):
                        cur.add(c)
                if insn.mnemonic == "call":
                    # The callee defines the return/caller-saved registers.
                    cur |= CALL_CLOBBERS[arch]
            node = func.get_node(addr)
            succs = list(graph.successors(node)) if node is not None and node in graph else []
            for s in succs:
                old = defined_in.get(s.addr)
                new = cur if old is None else (old & cur)
                if old is None or new != old:
                    defined_in[s.addr] = set(new)
                    changed = True

    # Stack usage via angr's stack-pointer tracker.
    max_arg, min_sp = 0, 0
    try:
        sp_name = "esp" if arch == "x86" else "rsp"
        bp_name = "ebp" if arch == "x86" else "rbp"
        proj = angr_project(binary)
        sptr = proj.analyses.StackPointerTracker(
            func, {proj.arch.registers[sp_name][0], proj.arch.registers[bp_name][0]})
        sp_off = proj.arch.registers[sp_name][0]
        bp_off = proj.arch.registers[bp_name][0]
        for addr in order:
            for ci in insns_by_block[addr]:
                insn = ci.insn
                sp = sptr.offset_before(insn.address, sp_off)
                bp = sptr.offset_before(insn.address, bp_off)
                sp = _signed(sp, 8 * word) if isinstance(sp, int) else None
                bp = _signed(bp, 8 * word) if isinstance(bp, int) else None
                if isinstance(sp, int):
                    min_sp = min(min_sp, sp)
                for op in insn.operands:
                    if op.type != 3 or not op.mem.base:
                        continue
                    base = insn.reg_name(op.mem.base)
                    if base in (sp_name,) and isinstance(sp, int):
                        off = sp + op.mem.disp
                    elif base in (bp_name,) and isinstance(bp, int):
                        off = bp + op.mem.disp
                    else:
                        continue
                    if off >= word:
                        max_arg = max(max_arg, off + op.size - word)
    except Exception as e:  # tracker failures only cost precision, never soundness
        max_arg = max(max_arg, 16 * word)
        min_sp = min(min_sp, -65536)
        notes.append(f"stack-pointer tracking failed ({type(e).__name__}); "
                     "using generous frame bounds")
    return read_first, max_arg, min_sp
