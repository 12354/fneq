"""Find a function's full extent and the facts later stages need.

Bounding is delegated to angr's CFGFast (tail calls, jump tables, padding). On top of
its result we collect what fneq needs: call sites and their callees, `ret N` values,
register/stack-argument usage for convention inference, comparison constants for the
fuzzer, and hazards that force an UNKNOWN verdict.
"""
from __future__ import annotations

import bisect
import contextlib
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
    indirect: bool = False  # call through a register/memory operand (target is a value)


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
    stack_read_sizes: dict[int, int] = dataclasses.field(default_factory=dict)
    call_defined: dict[int, set[str]] = dataclasses.field(default_factory=dict)
    call_sp: dict[int, int] = dataclasses.field(default_factory=dict)

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

_PROJECTS: dict[tuple, object] = {}


def angr_project(binary: Binary):
    import angr

    key = (binary.path, binary.arch, binary.image_base, _content_hash(binary))
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


# ------------------------------------------------------- function extent

def _chain_root(binary: Binary, unwind: int) -> int | None:
    """Follow UNW_FLAG_CHAININFO links; return the begin address of the primary entry."""
    root = None
    for _ in range(32):
        head = binary.read(unwind, 4)
        if head is None or not (head[0] >> 3) & 0x4:
            return root
        slot = unwind + 4 + 2 * ((head[2] + 1) & ~1)
        begin, unw = binary.read_int(slot, 4), binary.read_int(slot + 8, 4)
        if begin is None or unw is None:
            return None
        root, unwind = binary.image_base + begin, binary.image_base + unw
    return None


def _function_regions(binary: Binary, entry: int) -> list[tuple[int, int]]:
    """Code ranges CFGFast may explore for the function at `entry`.

    Giving CFGFast every executable section makes it recover the whole call graph
    reachable from the function, which takes many minutes on large images. On x64 PE
    the exception directory gives the function's extent: its own RUNTIME_FUNCTION plus
    the chained entries of split-off parts. A leaf function without unwind data
    extends to the next RUNTIME_FUNCTION.
    """
    exec_regions = [(s.addr, s.end) for s in binary.sections if s.executable]
    ranges = binary.pdata_ranges
    if binary.fmt != "pe" or not ranges:
        return exec_regions
    chains = getattr(binary, "_pdata_chains", None)
    if chains is None:
        chains = {}
        for begin, end, unwind in ranges:
            root = _chain_root(binary, unwind)
            if root is not None:
                chains.setdefault(root, []).append((begin, end))
        binary._pdata_chains = chains
    starts = [r[0] for r in ranges]
    i = bisect.bisect_right(starts, entry) - 1
    if i >= 0 and ranges[i][0] <= entry < ranges[i][1]:
        regions = [(entry, ranges[i][1])]
        regions += chains.get(ranges[i][0], [])
    else:
        sec = binary.section_at(entry)
        end = ranges[i + 1][0] if i + 1 < len(ranges) else sec.end
        regions = [(entry, min(end, sec.end))]
    return sorted(regions)


# ----------------------------------------------------------- the analysis

_BOUNDS: dict[tuple[int, int], tuple[Binary, object]] = {}


def bound(binary: Binary, entry: int) -> FunctionInfo:
    """Bounded function at entry; cached per binary (callees recur across pairs),
    and on disk when FNEQ_BOUND_CACHE names a directory (shared by batch workers
    and runs: bounding is deterministic for a binary file and this code)."""
    key = (id(binary), entry)
    cached = _BOUNDS.get(key)
    if cached is None or cached[0] is not binary:
        # Keep the owner alive: id(binary) alone can be reused after collection.
        cached = _BOUNDS[key] = (binary, _disk_bound(binary, entry))
    res = cached[1]
    if isinstance(res, Unsupported):
        raise Unsupported(str(res))
    return res


def _code_hash() -> str:
    global _CODE_HASH
    if _CODE_HASH is None:
        import hashlib
        import os
        import angr
        h = hashlib.sha1(angr.__version__.encode())     # CFGFast's behaviour
        for mod in ("bounding.py", "binary.py"):
            with open(os.path.join(os.path.dirname(__file__), mod), "rb") as f:
                h.update(f.read())
        _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH


_CODE_HASH: str | None = None
_CONTENT: dict = {}


def _content_hash(binary: Binary) -> str:
    """SHA-1 of the binary file, once per process (and file state)."""
    import hashlib
    import os
    st = os.stat(binary.path)
    key = (binary.path, st.st_size, st.st_mtime_ns)
    if key not in _CONTENT:
        h = hashlib.sha1()
        with open(binary.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _CONTENT[key] = h.hexdigest()
    return _CONTENT[key]


def _disk_bound(binary: Binary, entry: int):
    """_bound's result (a FunctionInfo, or the Unsupported it raised), cached on disk."""
    import hashlib
    import os
    import pickle
    root = os.environ.get("FNEQ_BOUND_CACHE")
    if not root:
        try:
            return _bound(binary, entry)
        except Unsupported as e:
            return e
    # The key is the binary's content (not its path or time stamp): any change to it --
    # the function itself, or what its bounding reads elsewhere (callee names, jump
    # tables, unwind data) -- gives a new key; an identical copy reuses the entries.
    try:
        ident = f"{_code_hash()}|{_content_hash(binary)}|{binary.fmt}|{binary.arch}|{binary.image_base}"
    except OSError:
        ident = None
    if ident is None:
        try:
            return _bound(binary, entry)
        except Unsupported as e:
            return e
    d = os.path.join(root, hashlib.sha1(ident.encode()).hexdigest()[:20])
    path = os.path.join(d, f"{entry:x}.pkl")
    try:
        with open(path, "rb") as f:
            kind, val = pickle.load(f)
        return val if kind == "ok" else Unsupported(val)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError):
        pass
    try:
        res = _bound(binary, entry)
        item = ("ok", res)
    except Unsupported as e:
        res, item = e, ("unsup", str(e))
    try:
        import tempfile
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(item, f)
        os.replace(tmp, path)       # atomic: workers may race on the same entry
    except OSError:
        pass
    return res


@contextlib.contextmanager
def _no_data_type_guessing():
    """CFGFast ends by classifying all the data it saw (strings, pointer arrays, ...)
    whatever its options -- most of its time on a large image. Bounding uses only
    functions, edges, call sites and jump tables, so that step is skipped."""
    from angr.analyses.cfg import cfg_fast
    from angr.knowledge_plugins.cfg.cfg_model import CFGModel
    orig = CFGModel.tidy_data_references
    orig_hints = cfg_fast.get_pointer_array_hints
    CFGModel.tidy_data_references = lambda self, *a, **k: False
    # (pointer-array hints depend on the loaded image only: computed once per image)
    cfg_fast.get_pointer_array_hints = _cached_hints(orig_hints)
    try:
        yield
    finally:
        CFGModel.tidy_data_references = orig
        cfg_fast.get_pointer_array_hints = orig_hints


_HINTS: dict = {}


def _cached_hints(orig):
    def hints(loader):
        key = id(loader)
        if key not in _HINTS:
            _HINTS[key] = (loader, orig(loader))   # (keeps the loader alive: id stays unique)
        return list(_HINTS[key][1])
    return hints


def _bound(binary: Binary, entry: int) -> FunctionInfo:
    proj = angr_project(binary)
    name = binary.symbol_at(entry) or f"sub_{entry:x}"
    if not binary.is_exec(entry):
        raise Unsupported(f"0x{entry:x} is not in an executable section")

    exec_regions = _function_regions(binary, entry)
    import angr
    # A fresh knowledge base per function: the project is shared by every analysis in
    # the process (batch workers), and CFGFast would otherwise accumulate functions.
    with _no_data_type_guessing():
        cfg = proj.analyses.CFGFast(
            kb=angr.KnowledgeBase(proj),
            function_starts=[entry], start_at_entry=False, force_complete_scan=False,
            force_smart_scan=False, treat_functions_as_complete=False,
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
                # Modelled as an uninterpreted call whose target is an argument.
                calls.append(CallSite(ins_addr, None, "<indirect>", indirect=True))
                continue
            calls.append(CallSite(ins_addr, target, callee_name(target)))
        for src, dst, data in part.transition_graph.edges(data=True):
            if data.get("outside") and data.get("type") == "transition":
                tgt = dst.addr
                if tgt in part_addrs:
                    continue
                last = _last_insn_addr(insns_by_block.get(src.addr))
                calls.append(CallSite(last, tgt, callee_name(tgt), tail=True))

    # This is a deliberately restricted CFG, not a complete image analysis. A
    # jump to a callee outside the permitted regions may have no destination
    # node/transition edge at all. Keep the machine-code tail call nevertheless;
    # otherwise argument-adjusting wrappers disappear from the call model.
    recorded = {(c.addr, c.target) for c in calls}
    for insns in insns_by_block.values():
        if not insns:
            continue
        last = insns[-1].insn
        if not last.operands:
            continue
        jump = last.mnemonic.startswith("j") or last.mnemonic.startswith("loop")
        if not jump:
            continue
        targets = [last.operands[0].imm] if last.operands[0].type == 2 else []
        if last.mnemonic != "jmp":
            targets.append(last.address + last.size)   # conditional fallthrough can be an import thunk
        for target in targets:
            if (last.address, target) not in recorded and binary.is_exec(target) \
                    and not any(lo <= target < hi for lo, hi in exec_regions):
                calls.append(CallSite(last.address, target, callee_name(target), tail=True))
                recorded.add((last.address, target))

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
    stack_reads: dict[int, int] = {}
    call_defined, call_sp = {}, {}
    regs_first, max_arg, min_sp = _usage(binary, func, insns_by_block, code, notes, widths, stack_reads,
                                        call_defined, call_sp)

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
                        uses_fp or uses_x87, pointer_access, stack_reads, call_defined, call_sp)


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
            elif mem.base == 0 and mem.index and seg is None and mem.disp:
                # Indexed table access [table + index*scale]: a data reference too.
                data_refs.add(mem.disp & ((1 << (8 * binary.ptr_size)) - 1))
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
           widths: dict[str, int], stack_reads: dict[int, int],
           call_defined: dict[int, set[str]], call_sp: dict[int, int]) -> tuple[set[str], int, int]:
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
                if insn.mnemonic == "call" or insn.mnemonic.startswith("j"):
                    previous = call_defined.get(insn.address)
                    call_defined[insn.address] = set(cur) if previous is None else previous & cur
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

    if changed:
        # Failure to reach the must-defined fixpoint cannot justify dropping an input.
        for at in call_defined:
            call_defined[at] = set()
        notes.append("register must-defined analysis did not converge; forwarded inputs kept")

    # Stack usage via angr's stack-pointer tracker.
    max_arg, min_sp = 0, 0
    try:
        sp_name = "esp" if arch == "x86" else "rsp"
        proj = angr_project(binary)
        tracked = {proj.arch.registers[r.lower()][0] for r in set(CANON.values())
                   if not r.startswith("XMM") and r.lower() in proj.arch.registers}
        tracked.add(proj.arch.registers[sp_name][0])
        sptr = proj.analyses.StackPointerTracker(
            func, tracked,
            # We query offsets at machine-instruction boundaries. VEX's
            # cross-instruction optimization can defer the RSP write from a SUB
            # until a later instruction, making a local buffer look like an
            # incoming stack argument at the intervening LEA/load/store.
            cross_insn_opt=False)
        sp_off = proj.arch.registers[sp_name][0]
        from angr.analyses.stack_pointer_tracker import OffsetVal

        def frame_offset(at, reg):
            # offset_before alone also returns offsets relative to the *incoming*
            # RBP/RCX/etc. Only values based on incoming SP address this frame.
            val = sptr._value_for(at, "pre", sptr.reg_values, sptr.reg_deltas, reg)
            return _signed(val.offset, 8 * word) if isinstance(val, OffsetVal) \
                and val.reg.offset == sp_off else None

        indexed_frame_read = False
        for addr in order:
            for ci in insns_by_block[addr]:
                insn = ci.insn
                sp = frame_offset(insn.address, sp_off)
                if isinstance(sp, int):
                    min_sp = min(min_sp, sp)
                    if insn.mnemonic == "call" or insn.mnemonic.startswith("j"):
                        call_sp[insn.address] = sp
                for op in insn.operands:
                    if op.type != 3 or not op.mem.base:
                        continue
                    base = insn.reg_name(op.mem.base)
                    if base not in proj.arch.registers:
                        continue
                    frame_base = frame_offset(insn.address, proj.arch.registers[base][0])
                    if frame_base is None:
                        continue
                    if op.mem.index:
                        if op.access & 1:
                            indexed_frame_read = True
                        continue
                    off = frame_base + op.mem.disp
                    if off >= word:
                        max_arg = max(max_arg, off + op.size - word)
                        # Stack arguments occupy words, but a callee reading an
                        # int32 consumes only four bytes of its x64 slot. The
                        # other bytes can legitimately be caller residue.
                        if insn.mnemonic != "lea" and op.access & 1:
                            for pos in range(off, off + op.size):
                                slot = word + (pos - word) // word * word
                                size = pos - slot + 1
                                size = next(n for n in (1, 2, 4, 8) if n >= size)
                                stack_reads[slot] = max(stack_reads.get(slot, 0), size)
        if indexed_frame_read:
            stack_reads.clear()     # unknown accessed slots: keep word-wide arguments
    except Exception as e:  # tracker failures only cost precision, never soundness
        stack_reads.clear()
        max_arg = max(max_arg, 16 * word)
        min_sp = min(min_sp, -65536)
        notes.append(f"stack-pointer tracking failed ({type(e).__name__}); "
                     "using generous frame bounds")
    return read_first, max_arg, min_sp
