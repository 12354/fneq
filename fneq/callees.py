"""Build CalleeSpec models for a function's external calls."""
from __future__ import annotations

from fneq.binary import Binary
from fneq.bounding import FunctionInfo, bound
from fneq.errors import FneqError
from fneq.model import CALL_RESULTS, CalleeSpec
from fneq.signature import Convention


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
        if site.target is None or site.target in specs:
            continue
        name = rename.get(site.target, site.name)
        if site.target == fi.entry:
            name = "self"
        cc = caller_cc
        ov = overrides.get(name, {})
        info = None
        if binary.is_exec(site.target) and site.target not in binary.imports:
            try:
                info = fi if site.target == fi.entry else bound(binary, site.target)
            except FneqError:
                info = None
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
        else:
            inferred_words, inferred_pop = 0, 0
            inferred_regs = list(cc.int_args) + list(cc.float_args)
            notes.append(f"callee {name}: code not available; assuming all argument registers "
                         f"and no stack arguments (override in the correspondence file)")
        stack_words = ov.get("stack_args", inferred_words)
        pop = ov.get("pop", inferred_pop)
        reg_args = list(ov.get("reg_args", inferred_regs))
        specs[site.target] = CalleeSpec(
            addr=site.target, name=name, reg_args=reg_args, stack_args=stack_words,
            pop=pop, results=CALL_RESULTS[cc.name],
            first_stack=word + cc.shadow)
    return specs, notes


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
