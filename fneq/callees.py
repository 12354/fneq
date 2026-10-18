"""Build CalleeSpec models for a function's external calls."""
from __future__ import annotations

from fneq.binary import Binary
from fneq.bounding import FunctionInfo, bound
from fneq.errors import FneqError, Unsupported
from fneq.model import CALL_RESULTS, CalleeSpec
from fneq.signature import Convention

INDIRECT_NAME = "indirect_call"

# MSVC x64 __chkstk, verified in both Parsec images. It saves/restores R10/R11,
# never changes RAX (the allocation size), and preserves every GPR/XMM register.
# Match its entire body, not its name: an arbitrary function named __chkstk must
# retain the ordinary conservative model. Memory effects remain uninterpreted.
MSVC_X64_CHKSTK = bytes.fromhex(
    "4883ec104c8914244c895c24084d33db4c8d5424184c2bd04d0f42d3"
    "654c8b1c25100000004d3bd37316664181e200f04d8d9b00f0ffff41c60300"
    "4d3bd375f04c8b14244c8b5c24084883c410c3")


def callee_specs(binary: Binary, fi: FunctionInfo, caller_cc: Convention,
                 overrides: dict[str, dict] | None = None,
                 rename: dict[int, str] | None = None) -> tuple[dict[int, CalleeSpec], list[str]]:
    """Return {target address: spec} and notes about assumptions made."""
    overrides = overrides or {}
    rename = rename or {}
    word = binary.ptr_size
    specs: dict[int, CalleeSpec] = {}
    notes: list[str] = []
    for site in fi.calls:
        if site.indirect:
            # The target is a run-time value: an uninterpreted function of the target
            # and of every argument register of the convention (the callee is unknown).
            specs[site.addr] = CalleeSpec(
                addr=site.addr, name=INDIRECT_NAME,
                reg_args=list(caller_cc.int_args) + list(caller_cc.float_args), stack_args=0,
                pop=0, results=CALL_RESULTS[caller_cc.name], first_stack=word + caller_cc.shadow,
                indirect=True)
            if not any(n.startswith("indirect calls:") for n in notes):
                notes.append("indirect calls: modelled as one uninterpreted function of the target "
                             "and all argument registers; stack arguments are not compared")
            continue
        if site.target is None or site.target in specs:
            continue
        name = _callee_name(binary, site.target, rename, site.name)
        if site.target == fi.entry:
            name = "self"
        cc = caller_cc
        ov = overrides.get(name, {})
        info = None
        code_at = _through_thunks(binary, site.target)
        if binary.arch == "x64" and code_at is not None \
                and binary.read(code_at, len(MSVC_X64_CHKSTK)) == MSVC_X64_CHKSTK:
            specs[site.target] = CalleeSpec(
                addr=site.target, name=name, reg_args=["RAX"], stack_args=0,
                # Called before allocation, without an MS x64 home area.
                pop=0, results=(), first_stack=word)
            notes.append(f"callee {name}: verified MSVC x64 stack-probe code; "
                         "RAX is the size input and GPR/XMM registers are preserved")
            continue
        complete_signature = all(k in ov for k in ("reg_args", "stack_args", "pop"))
        if complete_signature:
            notes.append(f"callee {name}: declared complete register/stack signature; "
                         "code-based arity inference is replaced by this supplied contract")
        if not complete_signature and code_at is not None and binary.is_exec(code_at) and code_at not in binary.imports:
            try:
                # A successfully bounded tail wrapper still forwards inputs it
                # does not read itself. Infer its adjusted target signature,
                # rather than mistaking the wrapper for a zero-argument callee.
                info = _stub_info(binary, code_at)
                if info is None:
                    info = fi if code_at == fi.entry else bound(binary, code_at)
            except FneqError:
                info = _stub_info(binary, code_at)
            if info is not None:
                info = _tail_usage(binary, info)
        if info is not None:
            used = max(info.max_stack_arg - cc.shadow, 0)
            inferred_words = (used + word - 1) // word
            inferred_pop = max(info.ret_pops) if cc.callee_pops else 0
            if inferred_pop:
                inferred_words = max(inferred_words, inferred_pop // word)
            # Arity from the registers the callee actually reads: a prefix of the
            # convention's argument registers up to the highest one read.
            # Compare each argument at the width the callee reads (an int passed
            # in RDI only defines EDI; the upper half may legitimately differ).
            inferred_regs = [_with_width(r, info.read_widths, binary.arch)
                             for r in _prefix_used(cc.int_args, info.regs_read_first)]
            inferred_regs += _prefix_used(cc.float_args, info.regs_read_first)
        elif (api := _api_params(name, binary)) is not None:
            # an import declared in the Windows SDK: its parameters, by position (MS x64)
            inferred_regs, inferred_words, inferred_pop = _api_args(api, cc), max(len(api) - 4, 0), 0
        else:
            inferred_words, inferred_pop = 0, 0
            inferred_regs = list(cc.int_args) + list(cc.float_args)
            notes.append(f"callee {name}: code not available; assuming all argument registers "
                         f"and no stack arguments (override in the correspondence file)")
        stack_words = ov.get("stack_args", inferred_words)
        pop = ov.get("pop", inferred_pop)
        reg_args = list(ov.get("reg_args", inferred_regs))
        stack_sizes = []
        if info is not None:
            stack_sizes = [info.stack_read_sizes.get(word + cc.shadow + i * word, word)
                           for i in range(stack_words)]
        elif (api := _api_params(name, binary)) is not None:
            stack_sizes = [int(k[2:]) // 8 if k.startswith("i:") else
                           4 if k == "f" else word for k in api[4:4 + stack_words]]
        stack_sizes = list(ov.get("stack_arg_sizes", stack_sizes))
        specs[site.target] = CalleeSpec(
            addr=site.target, name=name, reg_args=reg_args, stack_args=stack_words,
            pop=pop, results=CALL_RESULTS[cc.name],
            first_stack=word + cc.shadow,
            variadic=info is not None and cc.name == "ms" and _spills_varargs(binary, code_at),
            stack_arg_sizes=stack_sizes, frame_args=list(ov.get("frame_args", [])))
        if ov.get("frame_args"):
            notes.append(f"callee {name}: declared stack-pointee extents {ov['frame_args']}; "
                         "assumes the callee accesses no bytes beyond these ranges")
    return specs, notes


def _stub_info(binary: Binary, entry: int, seen=frozenset()):
    """A stub that sets a few registers and tail-jumps to another function
    (`xor r8d, r8d; jmp f`): the target's model, with the registers the stub sets
    before the jump not read from the caller. None for anything else."""
    import dataclasses
    import re
    import capstone
    from fneq.bounding import CANON, _full_write
    if entry in seen or len(seen) >= 8:
        raise Unsupported("tail-wrapper argument inference encountered a cycle or depth limit")
    seen = seen | {entry}
    section = binary.section_at(entry)
    data = binary.read(entry, min(64, section.end - entry)) if section else None
    if not data or binary.arch != "x64":
        return None
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    written, read = set(), set()
    for i, insn in enumerate(md.disasm(data, entry)):
        if i >= 4:
            return None
        if insn.mnemonic == "jmp" and re.fullmatch(r"0x[0-9a-f]+", insn.op_str):
            try:
                dest = int(insn.op_str, 16)
                if dest in seen:
                    raise Unsupported("tail-wrapper argument inference encountered a cycle")
                _, slot = _thunk_end(binary, dest)
                if slot is not None:
                    target = _import_usage(binary, dest, slot)
                else:
                    target = _stub_info(binary, dest, seen)
                    if target is None:
                        target = bound(binary, dest)
                    target = _tail_usage(binary, target, seen)
            except Unsupported:
                raise
            except FneqError:
                return None
            first = (target.regs_read_first - written) | read
            return dataclasses.replace(target, entry=entry, regs_read_first=first, calls=[])
        ops = insn.op_str.split(", ")
        if insn.mnemonic not in ("xor", "mov") or len(ops) != 2 or ops[0] not in CANON:
            return None
        dst, src = ops
        if insn.mnemonic == "xor" and src != dst:
            return None
        if insn.mnemonic == "mov" and src in CANON:
            read.add(CANON[src])
        elif insn.mnemonic == "mov" and not re.fullmatch(r"0x[0-9a-f]+|\d+", src):
            return None
        if _full_write(dst, binary.arch):
            written.add(CANON[dst])
    return None


def _tail_usage(binary: Binary, info: FunctionInfo, seen=frozenset()) -> FunctionInfo:
    """Include incoming registers/slots consumed by direct tail targets.

    Conditional wrappers need this even when their own bounds are valid. An input
    can be dropped only if it is definitely fully defined before that tail jump.
    """
    import dataclasses
    tails = [site for site in info.calls if site.tail]
    if not tails:
        return info
    if info.entry in seen or len(seen) >= 8:
        raise Unsupported("tail-wrapper argument inference encountered a cycle or depth limit")
    seen = seen | {info.entry}
    regs, widths = set(info.regs_read_first), dict(info.read_widths)
    sizes, maximum, pops = dict(info.stack_read_sizes), info.max_stack_arg, set(info.ret_pops)
    word = binary.ptr_size
    for site in tails:
        if site.target is None:
            raise Unsupported("tail-wrapper argument inference has an unknown target")
        if info.call_sp.get(site.addr) != 0:
            raise Unsupported("tail-wrapper argument inference needs a restored, known stack pointer")
        end, slot = (None, site.target) if site.target in binary.imports else _thunk_end(binary, site.target)
        if slot is not None:
            target = _import_usage(binary, site.target, slot)
        elif end is not None:
            if end in seen:
                raise Unsupported("tail-wrapper argument inference encountered a cycle")
            target = _stub_info(binary, end, seen)
            if target is None:
                target = _tail_usage(binary, bound(binary, end), seen)
        else:
            raise Unsupported("tail-wrapper argument inference cannot resolve its target")
        defined = info.call_defined.get(site.addr, set())
        for name in target.regs_read_first - defined:
            regs.add(name)
            widths[name] = max(widths.get(name, 0), target.read_widths.get(name, 8 * word))
        for off in range(word, max(maximum, target.max_stack_arg) + word, word):
            reads = [source.stack_read_sizes.get(off, word) for source in (info, target)
                     if off < source.max_stack_arg + word]
            if reads:
                sizes[off] = max(sizes.get(off, 0), *reads)
        maximum = max(maximum, target.max_stack_arg)
        pops |= target.ret_pops
    if len(pops) > 1:
        raise Unsupported("tail-wrapper argument inference has path-dependent stack cleanup")
    return dataclasses.replace(info, regs_read_first=regs, read_widths=widths,
                               max_stack_arg=maximum, stack_read_sizes=sizes, ret_pops=pops)


def _spills_varargs(binary: Binary, entry: int) -> bool:
    """MS x64 va_start: the prolog stores R8 and R9 to their home slots."""
    import re
    data = binary.read(entry, 48)
    if not data:
        return False
    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    seen = set()
    for i, insn in enumerate(md.disasm(data, entry)):
        if i >= 8:
            break
        m = re.match(r"qword ptr \[rsp \+ (0x18|0x20)\], (r8|r9)$", insn.op_str)
        if insn.mnemonic == "mov" and m and {"0x18": "r8", "0x20": "r9"}[m.group(1)] == m.group(2):
            seen.add(m.group(2))
    return seen == {"r8", "r9"}


_API: dict | None = None


def _api_params(name: str, binary: Binary) -> list | None:
    """Parameter kinds of a Windows API import (data/win_arity.json, generated from the
    SDK headers by spike/gen_win_arity.py); None if unknown or variadic."""
    global _API
    if binary.fmt != "pe" or binary.arch != "x64" or "!" not in name:
        return None
    if _API is None:
        import json
        import os
        try:
            with open(os.path.join(os.path.dirname(__file__), "data", "win_arity.json")) as f:
                _API = json.load(f)
        except OSError:
            _API = {}
    return _API.get(name.rpartition("!")[2])


def _api_args(kinds: list, cc: Convention) -> list[str]:
    """MS x64: parameter i is in the i-th integer or the i-th XMM register."""
    out = []
    for i, k in enumerate(kinds[:4]):
        if k in ("f", "d"):
            out.append(cc.float_args[i])
        else:
            reg = cc.int_args[i]
            out.append(f"{reg}:{k[2:]}" if k.startswith("i:") else reg)
    return out


def _import_usage(binary: Binary, entry: int, slot: int) -> FunctionInfo:
    """Usage of an import behind a tail stub, from the same SDK contract as calls."""
    name = binary.imports.get(slot, "")
    params = _api_params(name, binary)
    if params is None:
        raise Unsupported(f"tail-wrapper argument inference has no contract for {name or hex(slot)}")
    from fneq.signature import CONVENTIONS
    cc, word = CONVENTIONS["ms"], binary.ptr_size
    regs, widths = set(), {}
    for r in _api_args(params, cc):
        reg, _, bits = r.partition(":")
        regs.add(reg)
        widths[reg] = int(bits) if bits else (128 if reg.startswith("XMM") else 8 * word)
    sizes = {word + cc.shadow + i * word: int(k[2:]) // 8 if k.startswith("i:") else
             4 if k == "f" else word for i, k in enumerate(params[4:])}
    return FunctionInfo(entry, name, binary.arch, {}, [], {0}, {}, [], [], set(), set(),
                        regs, widths, cc.shadow + max(len(params) - 4, 0) * word, 0,
                        False, False, False, stack_read_sizes=sizes)


def _thunk_end(binary: Binary, target: int) -> tuple[int | None, int | None]:
    """Follow jump thunks (`jmp rel`) from `target`: (address of the code reached,
    None) or (None, import slot) for a chain ending in a jump through memory."""
    for _ in range(8):
        head = binary.read(target, 7) if binary.is_exec(target) else None
        if not head:
            return target, None
        if head[0] == 0xE9:
            target = target + 5 + int.from_bytes(head[1:5], "little", signed=True)
        elif head[0] == 0xEB:
            target = target + 2 + int.from_bytes(head[1:2], "little", signed=True)
        elif head[:2] == bytes([0xFF, 0x25]) or head[:3] == bytes([0x48, 0xFF, 0x25]):
            n = 2 if head[0] == 0xFF else 3
            disp = int.from_bytes(head[n:n + 4], "little", signed=True)
            # x64: RIP-relative; x86: absolute address
            return None, (target + n + 4 + disp if binary.arch == "x64" else disp & 0xFFFFFFFF)
        else:
            return target, None
    return None, None


def _through_thunks(binary: Binary, target: int) -> int | None:
    """The code a call to `target` reaches through jump thunks; None for an import stub
    (a jump through memory), whose callee code is not available.

    A thunk reads no registers itself, so inferring the arity from it would give a
    callee without arguments.
    """
    return _thunk_end(binary, target)[0]


def _callee_name(binary: Binary, target: int, rename: dict[int, str], default: str) -> str:
    """The name a callee is compared by: thunks are named after what they reach (an
    import stub after its import, `calloc` -> `jmp _calloc_base` after the latter),
    so that a call through a thunk matches a direct call on the other side."""
    end, slot = _thunk_end(binary, target)
    if slot is not None and slot in binary.imports:
        return binary.imports[slot]
    if end is not None and end != target and end in rename:
        return rename[end]
    return rename.get(target, default)


def _with_width(reg: str, widths: dict[str, int], arch: str) -> str:
    from fneq.bounding import CANON
    canon = CANON.get(reg.lower(), reg)
    full = 32 if arch == "x86" else 64
    bits = widths.get(canon, full)
    return reg if bits >= full else f"{reg}:{max(bits, 8)}"


def _prefix_used(regs: tuple[str, ...], read: set[str]) -> list[str]:
    last = -1
    for i, r in enumerate(regs):
        if r in read or r.replace("E", "R", 1) in read:
            last = i
    return list(regs[:last + 1])
