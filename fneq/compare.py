"""What 'equivalent' means, applied to two concrete runs."""
from __future__ import annotations

import dataclasses
from typing import Callable

from fneq.harness import RunResult
from fneq.signature import RET_KINDS, Signature

DEFINITION = (
    "Two functions are equivalent when, for every input (argument registers, stack "
    "arguments, all other registers, and the initial contents of all memory), they "
    "(1) return the same value in the return registers of the convention, "
    "(2) leave the same final memory state outside their own stack frame, "
    "(3) make the same external calls (after name mapping) with the same arguments in "
    "the same order, seeing the same memory, treating callees as uninterpreted "
    "functions, and (4) preserve the callee-saved registers, return to the same address "
    "and leave the same stack-pointer delta. A CPU fault is an outcome of its own: if only "
    "one function faults on an input they differ; if both fault they agree on that input "
    "(effects before the fault are not compared, since compilers may reorder them)."
)


@dataclasses.dataclass
class Checks:
    ret: bool = True
    memory: bool = True
    write_order: bool = False   # off by default: only the final state is observable
    calls: bool = True
    preserve: bool = True

    def disabled(self) -> list[str]:
        names = {"ret": "return value", "memory": "memory writes", "calls": "external calls",
                 "preserve": "callee-saved registers / stack delta"}
        out = [names[k] for k in names if not getattr(self, k)]
        if self.memory and not self.write_order:
            out.append("write order (final memory state is still compared)")
        return out


@dataclasses.dataclass
class Difference:
    check: str        # return | memory | write_order | calls | preserve | fault
    what: str
    a: object
    b: object

    def __str__(self) -> str:
        fmt = lambda v: hex(v) if isinstance(v, int) else str(v)  # noqa: E731
        return f"{self.check}: {self.what}: A={fmt(self.a)} B={fmt(self.b)}"


def compare(sig: Signature, ra: RunResult, rb: RunResult, checks: Checks,
            map_b: Callable[[int], int] = lambda x: x) -> list[Difference]:
    """Differences between run A and run B. Addresses of B are mapped into A's space."""
    diffs: list[Difference] = []
    ret_regs = set(RET_KINDS[sig.ret_kind][sig.arch])
    if ra.outputs["fault"] != rb.outputs["fault"]:
        diffs.append(Difference("fault", "CPU fault", ra.fault or "none", rb.fault or "none"))
        return diffs
    if ra.outputs["fault"]:
        return diffs  # both fault: equal (effects before a fault are not compared)
    for name in sig.outputs:
        va, vb = ra.outputs[name], rb.outputs[name]
        if va == vb:
            continue
        if name in ret_regs:
            if checks.ret:
                diffs.append(Difference("return", name, va, vb))
        elif checks.preserve:
            diffs.append(Difference("preserve", f"callee-saved {name}", va, vb))
    if checks.preserve:
        for k in ("sp_delta", "return_address"):
            if ra.outputs[k] != rb.outputs[k]:
                diffs.append(Difference("preserve", k, ra.outputs[k], rb.outputs[k]))
    if checks.calls:
        ca = list(ra.calls)
        cb = list(rb.calls)
        for i in range(max(len(ca), len(cb))):
            x = ca[i] if i < len(ca) else None
            y = cb[i] if i < len(cb) else None
            if x != y:
                diffs.append(Difference("calls", f"call #{i + 1}",
                                        _fmt_call(x), _fmt_call(y)))
                break
    if checks.memory:
        addrs = set(ra.final_bytes) | {map_b(a) for a in rb.final_bytes}
        inv = {map_b(a): a for a in rb.final_bytes}
        for addr in sorted(addrs):
            va = ra.read_byte(addr)
            vb = rb.read_byte(inv.get(addr, addr))
            if va != vb:
                diffs.append(Difference("memory", f"byte at 0x{addr:x}", va, vb))
                break
        if checks.write_order:
            wa = [(a, s, v) for a, s, v in ra.writes]
            wb = [(map_b(a), s, v) for a, s, v in rb.writes]
            if wa != wb:
                diffs.append(Difference("write_order", "sequence of memory writes",
                                        len(wa), len(wb)))
    return diffs


def _fmt_call(c) -> str:
    if c is None:
        return "<no call>"
    name, args = c
    return f"{name}({', '.join(hex(a) for a in args)})"
