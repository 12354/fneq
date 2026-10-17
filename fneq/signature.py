"""Calling conventions, their inference, and the unified signature shared by all backends.

The unified signature fixes, for both functions, which machine state is an input and
which is an observable output, so that every backend (Alive2, angr, Unicorn) compares
exactly the same things.
"""
from __future__ import annotations

import dataclasses

from fneq.bounding import FunctionInfo
from fneq.errors import InputError


@dataclasses.dataclass(frozen=True)
class Convention:
    name: str
    arch: str
    int_args: tuple[str, ...]        # integer argument registers, in order
    float_args: tuple[str, ...]      # vector/float argument registers
    callee_saved: tuple[str, ...]
    callee_pops: bool                # `ret N` pops the stack arguments
    shadow: int = 0                  # bytes of caller-provided home space (MS x64)
    red_zone: int = 0                # bytes below SP the callee may use (SysV)


CONVENTIONS = {c.name: c for c in [
    Convention("cdecl", "x86", (), (), ("EBX", "ESI", "EDI", "EBP"), False),
    Convention("stdcall", "x86", (), (), ("EBX", "ESI", "EDI", "EBP"), True),
    Convention("fastcall", "x86", ("ECX", "EDX"), (), ("EBX", "ESI", "EDI", "EBP"), True),
    Convention("thiscall", "x86", ("ECX",), (), ("EBX", "ESI", "EDI", "EBP"), True),
    Convention("sysv", "x64", ("RDI", "RSI", "RDX", "RCX", "R8", "R9"),
               tuple(f"XMM{i}" for i in range(8)),
               ("RBX", "RBP", "R12", "R13", "R14", "R15"), False, red_zone=128),
    Convention("ms", "x64", ("RCX", "RDX", "R8", "R9"), tuple(f"XMM{i}" for i in range(4)),
               ("RBX", "RBP", "RDI", "RSI", "R12", "R13", "R14", "R15")
               + tuple(f"XMM{i}" for i in range(6, 16)), False, shadow=32),
]}
ALIASES = {"msx64": "ms", "win64": "ms", "microsoft": "ms", "systemv": "sysv", "sysv64": "sysv"}

X86_GPRS = ("EAX", "ECX", "EDX", "EBX", "ESI", "EDI", "EBP")
X64_GPRS = ("RAX", "RCX", "RDX", "RBX", "RSI", "RDI", "RBP",
            "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15")

# Return kinds -> registers compared (in order).
RET_KINDS = {
    "void": {"x86": (), "x64": ()},
    "i8": {"x86": ("AL",), "x64": ("AL",)},
    "i16": {"x86": ("AX",), "x64": ("AX",)},
    "i32": {"x86": ("EAX",), "x64": ("EAX",)},
    "i64": {"x86": ("EAX", "EDX"), "x64": ("RAX",)},
    "ptr": {"x86": ("EAX",), "x64": ("RAX",)},
    "f32": {"x86": ("ST0",), "x64": ("XMM0:32",)},
    "f64": {"x86": ("ST0",), "x64": ("XMM0:64",)},
    "f80": {"x86": ("ST0",), "x64": ("ST0",)},
}


def get_convention(name: str, arch: str) -> Convention:
    name = ALIASES.get(name.lower(), name.lower())
    if name not in CONVENTIONS:
        raise InputError(f"unknown calling convention {name!r}")
    cc = CONVENTIONS[name]
    if cc.arch != arch:
        raise InputError(f"convention {name} is not valid for {arch}")
    return cc


@dataclasses.dataclass
class Inference:
    convention: Convention
    evidence: list[str]
    inferred: bool          # False if the user specified it


def infer_convention(fi: FunctionInfo, fmt: str, user: str | None) -> Inference:
    """Infer from `ret N` and from argument registers read before written."""
    if user:
        return Inference(get_convention(user, fi.arch), ["specified by user"], False)
    ev = []
    reads = fi.regs_read_first
    if fi.arch == "x64":
        sysv_only = reads & {"RDI", "RSI"}
        ms_only = reads & {"R8", "R9"} - {"RDI", "RSI"}
        if sysv_only and not (reads & {"RCX"} and not sysv_only):
            ev.append(f"reads {sorted(sysv_only)} before writing them")
            name = "sysv"
        elif reads & {"RCX", "RDX", "R8", "R9"} and not sysv_only:
            ev.append(f"reads {sorted(reads & {'RCX', 'RDX', 'R8', 'R9'})} before writing them")
            name = "ms"
        else:
            name = "ms" if fmt == "pe" else "sysv"
            ev.append(f"no argument registers read; default for {fmt.upper()}")
        del ms_only
        return Inference(CONVENTIONS[name], ev, True)
    pops = max(fi.ret_pops)
    ev.append(f"returns with ret {pops}" if pops else "returns with plain ret")
    ecx, edx = "RCX" in reads, "RDX" in reads
    if ecx:
        ev.append("reads ECX before writing it")
    if edx:
        ev.append("reads EDX before writing it")
    if ecx and edx:
        name = "fastcall"
    elif ecx:
        name = "thiscall" if pops else "fastcall"
    elif pops:
        name = "stdcall"
    else:
        name = "cdecl"
    return Inference(CONVENTIONS[name], ev, True)


def infer_return(a: FunctionInfo, b: FunctionInfo, arch: str, user: str | None) -> tuple[str, str]:
    """Return kind and a description of how it was chosen."""
    if user:
        if user not in RET_KINDS:
            raise InputError(f"unknown return kind {user!r}; choose from {', '.join(RET_KINDS)}")
        return user, "specified by user"
    # Conservative default: compare the widest integer return; with x87 code on x86
    # the result is likely in ST0. Reported so the user can override with --ret.
    if arch == "x86" and (a.uses_x87 or b.uses_x87):
        return "f64", "guessed from x87 use (override with --ret)"
    return ("i32" if arch == "x86" else "ptr"), "default guess (override with --ret)"


@dataclasses.dataclass
class Slot:
    kind: str       # reg | stack
    name: str       # register name, or stk_<offset>
    size: int       # bytes
    offset: int = 0  # stack offset from entry SP (RA at 0)


@dataclasses.dataclass
class Signature:
    arch: str
    convention: Convention
    ret_kind: str
    inputs: list[Slot]
    outputs: list[str]          # register outputs, in order
    stack_bytes: int            # bytes of stack arguments considered
    frame_below: int
    frame_above: int

    @property
    def word(self) -> int:
        return 4 if self.arch == "x86" else 8

    def describe(self) -> dict:
        return {
            "convention": self.convention.name,
            "return": self.ret_kind,
            "inputs": [s.name for s in self.inputs],
            "outputs": self.outputs + ["sp_delta", "return_address", "fault"],
        }


def build_signature(arch: str, cc: Convention, ret_kind: str,
                    infos: list[FunctionInfo], extra_stack_words: int = 2) -> Signature:
    word = 4 if arch == "x86" else 8
    gprs = X86_GPRS if arch == "x86" else X64_GPRS
    xmms = [f"XMM{i}" for i in range(8 if arch == "x86" else 16)]
    inputs = [Slot("reg", r, word) for r in gprs]
    inputs += [Slot("reg", x, 16) for x in xmms]
    used = max(i.max_stack_arg for i in infos)
    stack_bytes = cc.shadow + ((used + word - 1) // word + extra_stack_words) * word
    for off in range(word, word + stack_bytes, word):
        inputs.append(Slot("stack", f"stk_{off}", word, off))
    outputs = list(RET_KINDS[ret_kind][arch])
    for r in cc.callee_saved:
        if r not in outputs:
            outputs.append(r)
    depth = max(-min(i.min_sp for i in infos), 0)
    frame_below = depth + cc.red_zone + 256
    frame_above = word + stack_bytes
    return Signature(arch, cc, ret_kind, inputs, outputs, stack_bytes, frame_below, frame_above)
