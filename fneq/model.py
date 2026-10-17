"""Constants and callee descriptions shared by every backend.

All backends run a function in the same abstract environment:
  * the stack pointer at entry is STACK_TOP[arch]; the return address RET_ADDR[arch]
    is stored at [SP];
  * memory outside the image and the stack holds arbitrary (input) values;
  * the x87 control word and MXCSR are pinned; the x87 stack starts empty and the
    XMM registers that are not inputs start at zero;
  * external callees are uninterpreted functions of their arguments.
"""
from __future__ import annotations

import dataclasses

STACK_TOP = {"x86": 0x7FF00000, "x64": 0x7FFFFF000000}
RET_ADDR = {"x86": 0xDEAD0000, "x64": 0xDEADBEEF0000}

DEFAULT_X87_CW = 0x027F   # all exceptions masked, round-to-nearest, 53-bit precision
DEFAULT_MXCSR = 0x1F80    # all exceptions masked, round-to-nearest

# Results an external call produces, per calling convention: the return registers
# and every caller-saved (volatile) register, whose value becomes unknown.
_X86 = ("EAX", "EDX", "ECX")
CALL_RESULTS = {
    "cdecl": _X86, "stdcall": _X86, "fastcall": _X86, "thiscall": _X86,
    "sysv": ("RAX", "RDX", "RCX", "RSI", "RDI", "R8", "R9", "R10", "R11")
            + tuple(f"XMM{i}" for i in range(16)),
    "ms": ("RAX", "RCX", "RDX", "R8", "R9", "R10", "R11") + tuple(f"XMM{i}" for i in range(6)),
}


@dataclasses.dataclass
class CalleeSpec:
    """How a call to an external function is modelled."""
    addr: int                 # target address on this side
    name: str                 # canonical name shared by both sides
    reg_args: list[str]       # registers passed as arguments
    stack_args: int           # number of stack words passed (above the return address)
    pop: int                  # bytes the callee pops in addition to the return address
    results: tuple[str, ...]  # registers receiving results / clobbers
    first_stack: int = 0      # offset of the first stack argument from the callee's SP

    def stack_offsets(self, word: int) -> list[int]:
        base = self.first_stack or word
        return [base + word * i for i in range(self.stack_args)]

    def to_lifter(self, word: int) -> dict:
        args = [{"reg": r} for r in self.reg_args]
        args += [{"offset": off, "size": word} for off in self.stack_offsets(word)]
        return {"addr": self.addr, "name": self.name, "args": args,
                "results": list(self.results), "pop": self.pop}
