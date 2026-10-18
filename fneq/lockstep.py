"""Lockstep translation validation for near-identical functions.

A reimplementation that was written to compile into (nearly) the same machine code
as the original shares its control-flow graph, its calls and its memory accesses; the
two functions differ in register allocation, stack-slot offsets, addresses and local
scheduling. For that case equivalence is proven segment by segment instead of for the
whole function at once:

* both functions are cut into *segments* (cut-point product, as in Rêve / CHL /
  semantic program alignment): from a cut point -- the entry or a loop head (the
  target of a back edge) -- all paths are executed up to the next loop head or
  return (calls are followed). A path goes on through a loop head speculatively, so
  that a loop decided by constants runs to completion; at a real fork in the loop it
  falls back to leaving at the head. A segment pair with too many paths is cut where
  paths join as well;
* a segment pair (A at address a, B at address b) is executed symbolically from an
  abstract entry state: a set of *correspondences* (A location == B location, or
  A location == B location + d, where a location is a register, a stack byte or a
  stack word), per-side constants and stack addresses; everything else starts as a
  fresh, unrelated value. Memory outside the frames is one shared array (equal on
  entry);
* the exits are paired as a product: an A path and a B path whose conditions are
  provably equivalent go together (and with no other path), the remaining ones are
  paired whenever their conditions can hold together -- so branches need not
  correspond one to one. For every pair, Z3 must prove: the external calls match
  (name, arguments, memory and stack objects they see), both leave the same way, the
  memory outside the frames is equal, division faults happen together, and on a
  return the observable results are equal. The correspondences that still hold at a
  pair's exit (found by fingerprinting with random valuations, confirmed by Z3) become
  an entry state of the successor pair;
* the entry states are joined until a fixpoint is reached, so loops need no unrolling
  and the proof covers all executions.

Floating-point operations are uninterpreted functions of their operands (and rounding
mode). Callees without a name that are called at corresponding places are assumed to
correspond (one-to-one); a proof that needs this says so.

Soundness: every entry state is implied by every predecessor's exit (correspondences
are only kept when proven), so a completed fixpoint is a proof of equivalence. Any
failed or undecided obligation gives "differs here, or unknown" -- never a false
EQUIVALENT -- and is reported with the block addresses where it happened.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import time

import z3

from fneq import bounding
from fneq.model import RET_ADDR

_DIVMOD = re.compile(r"Iop_DivMod([SU])(\d+)to(\d+)$")
_INDIRECT = "indirect_call"            # (callees.INDIRECT_NAME)
# Values that are junk to a callee: uninitialised frame bytes, the callee home space
# an earlier call owned, and the registers an earlier call clobbered (its results
# other than the return registers, which _Checker.call_returns lists).
_UNINIT = re.compile(r"^(?:u[AB]\d+_\d+|x[AB]\d+_c\d+_\d+)$")
_CLOBBER = re.compile(r"^c[AB]\d+_\d+_(\w+)$")
# Return registers of a callee, per convention (the rest of CALL_RESULTS is clobbered)
_CALL_RETURNS = {"ms": {"rax", "xmm0"}, "sysv": {"rax", "rdx", "xmm0", "xmm1"}}


def _unpaired(calls_a, calls_b) -> tuple:
    """(side tag, call address) of the calls of a segment whose callee the other side
    does not call there: the candidates for inlining."""
    na = {_canon(c[2]) for c in calls_a}
    nb = {_canon(c[2]) for c in calls_b}
    return tuple([("A", c[6]) for c in calls_a if _canon(c[2]) not in nb]
                 + [("B", c[6]) for c in calls_b if _canon(c[2]) not in na])


def _same_ast(x, y) -> bool:
    return x is y or x.hash() == y.hash()


def _image_string(binary, addr: int) -> str | None:
    """A NUL-terminated ASCII or UTF-16 string in read-only image data, else None."""
    sec = binary.section_at(addr)
    if sec is None or sec.writable or sec.executable:
        return None
    blob = binary.read(addr, min(1024, sec.end - addr)) or b""
    n = blob.find(b"\0")
    if n > 0 and all(0x20 <= c < 0x7F or c in (9, 10, 13) for c in blob[:n]):
        return blob[:n].decode()
    if len(blob) >= 4 and blob[1] == 0:
        for i in range(0, len(blob) - 1, 2):
            if blob[i] == 0 and blob[i + 1] == 0:
                return blob[:i].decode("utf-16-le", "replace") if i else None
    return None


_CONV = re.compile(r"%([-+ #0]*)(\*|\d*)(?:\.(\*|\d*))?(hh|h|ll|l|L|j|z|t|I64|I32|I|w)?([diouxXeEfFgGaAcCsSpn%])")


def _conversions(fmt: str) -> int:
    """Arguments a printf format consumes (`%%` none, each `*` one more)."""
    n = 0
    for m in _CONV.finditer(fmt):
        if m.group(5) == "%":
            continue
        n += 1 + (m.group(2) == "*") + (m.group(3) == "*")
    return n
_DIV = re.compile(r"Iop_Div([SU])(\d+)E?$")
_UNNAMED = re.compile(r"^0x[0-9a-f]+$")


class _Unsup(Exception):
    """Outside the lockstep model: fall back to the other backends."""


class _TooManyPaths(_Unsup):
    """A region has more paths than allowed (retried with more cut points)."""


class _Differs(Exception):
    """A proof obligation failed (a real difference, or one the model cannot decide)."""

    def __init__(self, a: int | None, b: int | None, what: str, calls: tuple = ()):
        super().__init__(what)
        self.a, self.b, self.what = a, b, what
        self.calls = calls      # (side tag, call address): calls that did not line up


@dataclasses.dataclass
class Result:
    status: str          # equivalent | differs | unknown
    reason: str
    data: dict


# ------------------------------------------------------------- claripy -> z3

class _Conv:
    """claripy AST -> z3 term, memoised; side B maps addresses into A's space and
    substitutes unified symbols."""

    def __init__(self, word_bits: int, map_const=None, registry: dict | None = None):
        self.word_bits = word_bits
        self.map_const = map_const
        self.subst: dict[str, z3.ExprRef] = {}
        self.cache: dict = {}
        self.registry = registry if registry is not None else {}

    def set_subst(self, name: str, term) -> None:
        self.subst[name] = term
        self.cache = {}         # (a new dict: a clone may share the old one)

    def clone(self) -> "_Conv":
        """A converter with the same substitutions, for one exit pair's unification."""
        c = _Conv(self.word_bits, self.map_const, self.registry)
        c.subst = dict(self.subst)
        c.cache = self.cache    # shared until the clone substitutes something
        return c

    def __call__(self, root):
        cache = self.cache
        stack = [root]
        while stack:
            e = stack[-1]
            k = e.hash()
            if k in cache:
                stack.pop()
                continue
            pending = [a for a in e.args if hasattr(a, "op") and a.hash() not in cache]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            args = [cache[a.hash()] if hasattr(a, "op") else a for a in e.args]
            cache[k] = self._node(e.op, args, e)
        return cache[root.hash()]

    def _node(self, op: str, a: list, e=None):
        if op.startswith("fp") or op in ("FPV", "FPS", "bvToFP"):
            return _fp_node(op, a, e)
        if op == "BVV":
            v, n = a
            if self.map_const and n == self.word_bits:
                v = self.map_const(v)
            return z3.BitVecVal(v, n)
        if op == "BVS":
            name, n = a[0], a[1]
            if name in self.subst:
                return self.subst[name]
            t = z3.BitVec(name, n)
            self.registry[name] = t
            return t
        if op == "BoolV":
            return z3.BoolVal(a[0])
        if op == "BoolS":
            t = z3.Bool(a[0])
            self.registry[a[0]] = t
            return t
        f = _OPS.get(op)
        if f is None:
            raise _Unsup(f"operation {op} is not modelled by the lockstep checker")
        return f(a)


_RM = re.compile(r"^\((?:\(_ [^()]*\)|[\w.]+) (RNE|RNA|RTP|RTN|RTZ) ")


def _fp_node(op: str, a: list, e):
    """Floating point, uninterpreted: a value is its IEEE bit pattern, and every
    arithmetic operation, conversion and comparison is an uninterpreted function of
    (operation, rounding mode, widths). Equal operands give equal results, nothing
    else is assumed -- sound, and enough when both sides compute the same formula.
    Moves, sign flips and absolute values are exact bit operations."""
    size = None if type(e).__name__ == "Bool" else e.size()
    if op == "FPV":
        import struct
        if size == 32:
            return z3.BitVecVal(int.from_bytes(struct.pack("<f", a[0]), "little"), 32)
        if size == 64:
            return z3.BitVecVal(int.from_bytes(struct.pack("<d", a[0]), "little"), 64)
        raise _Unsup(f"{size}-bit floating-point constant")
    if op == "FPS":
        return z3.BitVec(a[0], size)
    if op in ("bvToFP", "fpToIEEEBV"):
        return a[0]
    if op == "fpNeg":
        return a[0] ^ z3.BitVecVal(1 << (size - 1), size)
    if op == "fpAbs":
        return a[0] & z3.BitVecVal((1 << (size - 1)) - 1, size)
    args = [x for x in a if isinstance(x, z3.ExprRef)]
    # The rounding mode is an argument of the arithmetic, a hidden field of the
    # conversions (only its SMT-LIB form shows it).
    rm = [str(x) for x in a if type(x).__name__ == "RM"]
    if not rm and op.startswith("fpTo"):
        m = _RM.match(e.smtlib())
        if m is None:
            raise _Unsup(f"rounding mode of {op} not found")
        rm = [m.group(1)]
    name = "|".join([op] + rm + [str(x.size()) for x in args] + [str(size)])
    f = z3.Function(name, *[x.sort() for x in args],
                    z3.BoolSort() if size is None else z3.BitVecSort(size))
    return f(*args)


def _reduce(fn):
    def go(a):
        r = a[0]
        for x in a[1:]:
            r = fn(r, x)
        return r
    return go


def _reverse(x):
    n = x.size() // 8
    return z3.Concat(*[z3.Extract(8 * i + 7, 8 * i, x) for i in range(n)]) if n > 1 else x


_OPS = {
    "__add__": _reduce(lambda x, y: x + y), "__sub__": _reduce(lambda x, y: x - y),
    "__mul__": _reduce(lambda x, y: x * y), "__and__": _reduce(lambda x, y: x & y),
    "__or__": _reduce(lambda x, y: x | y), "__xor__": _reduce(lambda x, y: x ^ y),
    "__invert__": lambda a: ~a[0], "__neg__": lambda a: -a[0],
    "__lshift__": lambda a: a[0] << a[1], "__rshift__": lambda a: a[0] >> a[1],
    "LShR": lambda a: z3.LShR(a[0], a[1]), "__floordiv__": lambda a: z3.UDiv(a[0], a[1]),
    "__mod__": lambda a: z3.URem(a[0], a[1]), "SDiv": lambda a: a[0] / a[1],
    "SMod": lambda a: z3.SRem(a[0], a[1]),
    "Extract": lambda a: z3.Extract(a[0], a[1], a[2]),
    "Concat": lambda a: z3.Concat(*a) if len(a) > 1 else a[0],
    "ZeroExt": lambda a: z3.ZeroExt(a[0], a[1]), "SignExt": lambda a: z3.SignExt(a[0], a[1]),
    "If": lambda a: z3.If(a[0], a[1], a[2]),
    "__eq__": lambda a: a[0] == a[1], "__ne__": lambda a: a[0] != a[1],
    "ULT": lambda a: z3.ULT(a[0], a[1]), "ULE": lambda a: z3.ULE(a[0], a[1]),
    "UGT": lambda a: z3.UGT(a[0], a[1]), "UGE": lambda a: z3.UGE(a[0], a[1]),
    "SLT": lambda a: a[0] < a[1], "SLE": lambda a: a[0] <= a[1],
    "SGT": lambda a: a[0] > a[1], "SGE": lambda a: a[0] >= a[1],
    "And": lambda a: z3.And(*a), "Or": lambda a: z3.Or(*a), "Not": lambda a: z3.Not(a[0]),
    "RotateLeft": lambda a: z3.RotateLeft(a[0], a[1]),
    "RotateRight": lambda a: z3.RotateRight(a[0], a[1]),
    "Reverse": lambda a: _reverse(a[0]),
}


# ------------------------------------------------------------ symbolic engine

def _claripy():
    from angr import claripy
    return claripy


def _make_engine_classes():
    from angr.engines.vex.heavy.heavy import HeavyVEXMixin
    from angr.state_plugins.plugin import SimStatePlugin
    claripy = _claripy()

    class LsMem(SimStatePlugin):
        """Memory as the lockstep model sees it: frame bytes, a log of accesses to the
        shared memory outside the frame, external calls and divisions."""

        def __init__(self, side=None, visit=0, stack=None, written=None, escaped=None,
                     heap=None, events=None, divs=None, ncalls=0, nreads=0, words=None, guard=None):
            super().__init__()
            self.side, self.visit = side, visit
            self.stack = dict(stack or {})      # offset from sp0 -> 8-bit value
            # offset -> (value, its bytes) of whole stores: reading the same bytes back
            # gives the stored expression (claripy rewrites byte extracts of `sp0 + c`).
            self.words = dict(words or {})
            self.escaped = set(escaped or ())   # frame offsets passed to callees
            self.heap = list(heap or [])        # ("store", addr, data, size) | ("havoc", name)
            self.events = list(events or [])    # ("read", ...) | ("call", ...)
            self.divs = list(divs or [])        # (op, dividend, divisor)
            self.ncalls, self.nreads = ncalls, nreads
            self.guard = claripy.true() if guard is None else guard

        @SimStatePlugin.memo
        def copy(self, memo):
            return LsMem(self.side, self.visit, self.stack, None, self.escaped, self.heap,
                         self.events, self.divs, self.ncalls, self.nreads, self.words, self.guard)

        # ---- frame
        def default_byte(self, off: int):
            side = self.side
            if 0 <= off < side.word:
                return claripy.BVV((RET_ADDR[side.arch] >> (8 * off)) & 0xFF, 8)
            if off >= side.word:
                return claripy.BVS(f"stk_in_{off}", 8, explicit_name=True)
            return claripy.BVS(f"u{side.tag}{self.visit}_{-off}", 8, explicit_name=True)

        def stack_read(self, off: int, size: int):
            w = self.words.get(off)
            if w is not None and len(w[1]) == size and all(
                    self.stack.get(off + i) is b for i, b in enumerate(w[1])):
                return w[0]
            parts = [self.stack.get(o) for o in range(off + size - 1, off - 1, -1)]
            parts = [p if p is not None else self.default_byte(o)
                     for p, o in zip(parts, range(off + size - 1, off - 1, -1))]
            return claripy.Concat(*parts) if len(parts) > 1 else parts[0]

        def stack_write(self, off: int, data):
            n = data.size() // 8
            for i in range(n):
                self.stack[off + i] = claripy.Extract(8 * i + 7, 8 * i, data)
            if n > 1:
                self.words[off] = (data, tuple(self.stack[off + i] for i in range(n)))

        # ---- classification
        def classify(self, addr):
            side = self.side
            if not addr.symbolic:
                v = addr.concrete_value
                if side.readonly(v):
                    return ("image", v)
                return ("heap", addr)
            if side.sp0_name in addr.variables:
                off = _sp_offset(addr, side.sp0_name, side.bits)
                if off is None:
                    raise _Unsup("memory access at a variable offset into the stack frame")
                return ("stack", off)
            return ("heap", addr)

        def load(self, addr, size: int):
            if addr.symbolic and self.side.sp0_name in addr.variables \
                    and _sp_offset(addr, self.side.sp0_name, self.side.bits) is None:
                offset, choices = self.stack_choices(addr, size)
                value = self.stack_read(choices[-1], size)
                for off in choices[:-1]:
                    value = claripy.If(offset == off, self.stack_read(off, size), value)
                return value
            kind, where = self.classify(addr)
            if kind == "image":
                data = self.side.binary.read(where, size)
                if data is None:
                    raise _Unsup(f"read outside the image at 0x{where:x}")
                return claripy.BVV(int.from_bytes(data, "little"), 8 * size)
            if kind == "stack":
                return self.stack_read(where, size)
            name = f"r{self.side.tag}{self.visit}_{self.nreads}"
            self.nreads += 1
            sym = claripy.BVS(name, 8 * size, explicit_name=True)
            self.events.append(("read", sym, addr, size, len(self.heap)))
            return sym

        def store(self, addr, data):
            if addr.symbolic and self.side.sp0_name in addr.variables \
                    and _sp_offset(addr, self.side.sp0_name, self.side.bits) is None:
                offset, choices = self.stack_choices(addr, data.size() // 8)
                for off in choices:
                    self.stack_write(off, claripy.If(offset == off, data,
                                                     self.stack_read(off, data.size() // 8)))
                return
            kind, where = self.classify(addr)
            if kind == "stack":
                self.stack_write(where, data)
                return
            if self.side.sp0_name in data.variables:
                raise _Unsup("a stack address is stored to memory outside the frame")
            self.heap.append(("store", addr, data, data.size() // 8))

        def stack_choices(self, addr, size):
            """Every possible offset, proved exhaustive under this block's entry guard.

            No sampled/concretised index: a final UNSAT query must establish coverage.
            Restrict this extension to the currently allocated frame below the root
            return slot; accesses beyond it keep the conservative UNKNOWN result.
            """
            side = self.side
            offset = claripy.simplify(addr - side.sp0)
            sp = _sp_offset(self.state.registers.load(side.sp), side.sp0_name, side.bits)
            if sp is None:
                raise _Unsup("variable frame access with a nonconstant stack pointer")
            solver = z3.Solver()
            solver.set("timeout", 250)
            solver.add(*_alignment(side), side.fconv(self.guard))
            term = side.fconv(offset)
            choices = []
            while True:
                result = solver.check()
                if result == z3.unsat:
                    if not choices:
                        raise _Unsup("infeasible variable frame access")
                    return offset, sorted(choices)
                if result != z3.sat or len(choices) >= _MAX_FRAME_CASES:
                    raise _Unsup("variable stack access has no proven small finite offset set")
                value = solver.model().eval(term, model_completion=True).as_long()
                off = _signed(value, side.bits)
                if off < sp or off + size > 0:
                    raise _Unsup("variable stack access can leave the allocated frame or reach the return slot")
                choices.append(off)
                solver.add(term != z3.BitVecVal(value, side.bits))

    class Engine(HeavyVEXMixin):
        def _perform_vex_expr_Load(self, addr, ty, endness, action=None, inspect=True,
                                   condition=None, **kwargs):
            return self.state.lsmem.load(addr, self._ty_to_bytes(ty))

        def _perform_vex_stmt_Store(self, addr, data, endness, action=None, inspect=True,
                                    condition=None):
            mem = self.state.lsmem
            if condition is not None and not (not condition.symbolic and condition.is_true()):
                if not condition.symbolic and condition.is_false():
                    return
                data = claripy.If(condition, data, mem.load(addr, data.size() // 8))
            mem.store(addr, data)

        def _perform_vex_expr_Op(self, op, args):
            if _DIVMOD.match(op) or _DIV.match(op):
                self.state.lsmem.divs.append((op, args[0], args[1]))
            return super()._perform_vex_expr_Op(op, args)

    return LsMem, Engine


_CLASSES = None


def _classes():
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = _make_engine_classes()
    return _CLASSES


def _sp_offset(e, name: str, bits: int) -> int | None:
    """c if `e` is `name + c` (a linear term in the stack-pointer symbol), else None."""
    def lin(x):
        if not x.symbolic:
            return 0, x.concrete_value
        if x.op == "BVS":
            return (1, 0) if x.args[0] == name else None
        if x.op in ("__add__", "__sub__"):
            co, c = 0, 0
            for i, a in enumerate(x.args):
                r = lin(a)
                if r is None:
                    return None
                sign = -1 if x.op == "__sub__" and i else 1
                co, c = co + sign * r[0], c + sign * r[1]
            return co, c
        if x.op == "__neg__":
            r = lin(x.args[0])
            return None if r is None else (-r[0], -r[1])
        return None
    r = lin(e)
    if r is None or r[0] != 1:
        return None
    return _signed(r[1], bits)


class _Flat(Exception):
    """An address outside the region model (--no-alias): use the flat memory."""


class _MemBuilder:
    """The memories after each heap operation of one path, built in program order
    as far as asked; in the region model, the aliases its reads establish are noted
    when they are passed."""

    def __init__(self, m: "_Match", side: int, mem):
        self.m, self.side, self.mem = m, side, mem
        regions = m._regions()
        h = _Heap(m.ck.A.bits, f"H{m.visit}", flat=None if regions else m.h0,
                  classify=m._classify if regions else None)
        self.out, self.n, self.ev, self.seen = [h], 0, 0, {}

    def at(self, i: int) -> "_Heap":
        evs, ops = self.mem.events, self.mem.heap
        while True:
            # the reads before operation n (they may alias pointers it uses)
            while self.ev < len(evs) and (evs[self.ev][0] != "read" or evs[self.ev][4] <= self.n):
                e = evs[self.ev]
                self.ev += 1
                if e[0] == "read" and self.out[-1].flat is None:
                    self.m._note_read(self.side, self.out[-1], e, self.seen)
            if self.n >= i:
                return self.out[i]
            h = self.out[-1].copy()
            self.m._apply(self.side, h, ops[self.n])
            self.out.append(h)
            self.n += 1


class _Heap:
    """Memory outside the frames at one point of a path: a flat byte array, or
    (--no-alias) one array per region -- the object a pointer base addresses,
    indexed by offset, and the globals at fixed addresses. `origin` names the
    initial contents (the entry memory, or a call's result); regions not written
    since then are `origin|region`."""

    def __init__(self, bits, origin, flat=None, classify=None):
        self.bits, self.origin, self.flat, self.classify = bits, origin, flat, classify
        self.arrays: dict = {}      # region -> array (written regions)
        self.stores: dict = {}      # region -> [(index, data, size)] since origin

    def _sort(self):
        return z3.BitVecSort(self.bits), z3.BitVecSort(8)

    def array(self, key):
        a = self.arrays.get(key)
        return a if a is not None else z3.Array(f"{self.origin}|{key}", *self._sort())

    def copy(self) -> "_Heap":
        h = _Heap(self.bits, self.origin, self.flat, self.classify)
        h.arrays, h.stores = dict(self.arrays), {k: list(v) for k, v in self.stores.items()}
        return h

    def store(self, a, data, size: int) -> None:
        if self.flat is not None:
            for k in range(size):
                self.flat = z3.Store(self.flat, a + k, z3.Extract(8 * k + 7, 8 * k, data))
            return
        key, idx = self.classify(a)
        arr = self.array(key)
        for k in range(size):
            arr = z3.Store(arr, idx + k, z3.Extract(8 * k + 7, 8 * k, data))
        self.arrays[key] = arr
        self.stores.setdefault(key, []).append((idx, data, size))

    def havoc(self, origin: str) -> None:
        if self.flat is not None:
            self.flat = z3.Array(origin, *self._sort())
            return
        self.origin, self.arrays, self.stores = origin, {}, {}

    def read(self, a, size: int):
        if self.flat is not None:
            arr, idx = self.flat, a
        else:
            key, idx = self.classify(a)
            arr = self.array(key)
        parts = [z3.Select(arr, idx + k) for k in range(size - 1, -1, -1)]
        return z3.Concat(*parts) if size > 1 else parts[0]

    def same(self, o: "_Heap") -> bool:
        if self.flat is not None:
            return self.flat.eq(o.flat)
        return self.origin == o.origin and set(self.arrays) == set(o.arrays) and \
            all(self.arrays[k].eq(o.arrays[k]) for k in self.arrays)

    def equal(self, o: "_Heap"):
        """The claim that both memories are equal."""
        if self.flat is not None:
            return self.flat == o.flat
        if self.origin != o.origin:
            return z3.BoolVal(False)    # contents of untouched regions differ
        keys = set(self.arrays) | set(o.arrays)
        return z3.And(*[self.array(k) == o.array(k) for k in keys]) if keys else z3.BoolVal(True)


def _alignment(s) -> list:
    """The ABI's stack alignment at entry (x64: SP + 8 is a multiple of 16)."""
    if s.arch != "x64":
        return []
    return [z3.Extract(3, 0, z3.BitVec(s.sp0_name, s.bits)) == 8]


def _signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


_REGS = {
    "x64": ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"] + [f"r{i}" for i in range(8, 16)]
           + ["cc_op", "cc_dep1", "cc_dep2", "cc_ndep", "d", "ac", "id", "sseround"]
           + [f"ymm{i}" for i in range(16)] + ["ftop", "fpreg", "fptag", "fpround", "fc3210"],
    "x86": ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
           + ["cc_op", "cc_dep1", "cc_dep2", "cc_ndep", "d", "ac", "id", "sseround"]
           + [f"xmm{i}" for i in range(8)] + ["ftop", "fpreg", "fptag", "fpround", "fc3210"],
}
# DF=0, round-to-nearest (pinned MXCSR/CW), status flags 0 as a COPY flags thunk.
_MAX_SPECULATION = 2048       # steps a path runs past a loop head before falling back
_MAX_FORKS = 48              # paths of one region before it is cut at joins as well
_MAX_TABLE = 1024            # cases of one resolved jump table
_MAX_FRAME_CASES = 256       # exhaustive cases of a variable stack load/store
_MAX_CONST_OFFSET = 64        # constants this close on both sides may be offset-related
_MAX_OFFSET = 1 << 16         # largest offset correspondence looked for
_FIXED = {"d": 1, "sseround": 0, "fpround": 0, "cc_op": 0, "cc_dep1": 0, "cc_dep2": 0, "cc_ndep": 0}


def _normalise_flags(st, flags=None) -> None:
    """Rewrite VEX's lazy flags thunk as COPY (cc_op 0, flags value in cc_dep1).

    angr's flag helpers need a concrete cc_op; in COPY form it always is, and the
    flags of both sides become comparable values.
    """
    if flags is None:
        if st.regs.cc_op.symbolic:
            raise _Unsup("flags thunk with a symbolic operation")
        flags = st.regs.flags
    bits = st.regs.cc_op.size()
    st.regs.cc_op = _claripy().BVV(0, bits)
    st.regs.cc_dep1 = flags
    st.regs.cc_dep2 = _claripy().BVV(0, bits)
    st.regs.cc_ndep = _claripy().BVV(0, bits)


def _canon(name: str) -> str:
    """Callee name as compared: imports may carry their DLL (kernel32.dll!Sleep)."""
    return name.rsplit("!", 1)[-1]


def _merge_arities(ca: dict, cb: dict) -> tuple[dict, dict]:
    """Give callees of the same name the same model on both sides.

    The arity is inferred from each side's copy of the callee, which may read a
    different prefix of the argument registers; comparing the larger set of
    arguments (at the wider width) is the stricter check.
    """
    def regs_key(r):
        name, _, w = r.partition(":")
        return name, int(w) if w else 1 << 30
    by_name: dict = {}
    for spec in list(ca.values()) + list(cb.values()):
        if not spec.indirect:
            by_name.setdefault(_canon(spec.name), []).append(spec)
    merged = {}
    for name, specs in by_name.items():
        regs: list = []
        for spec in specs:
            for i, r in enumerate(spec.reg_args):
                if i >= len(regs):
                    regs.append(r)
                elif regs_key(r)[0] == regs_key(regs[i])[0] and regs_key(r)[1] > regs_key(regs[i])[1]:
                    regs[i] = r
        words = max(s.stack_args for s in specs)
        sizes = []
        for i in range(words):
            read = [s.stack_arg_sizes[i] if i < len(s.stack_arg_sizes) else 0
                    for s in specs if i < s.stack_args]
            sizes.append(0 if 0 in read else max(read))
        merged[name] = (regs, words, sizes)
    out = []
    for c in (ca, cb):
        d = {}
        for k, spec in c.items():
            m = merged.get(_canon(spec.name))
            d[k] = spec if m is None else dataclasses.replace(spec, reg_args=list(m[0]),
                                                               stack_args=m[1], stack_arg_sizes=list(m[2]))
        out.append(d)
    return out[0], out[1]


def _unwind_ops(binary, unwind: int):
    """The codes of one x64 UNWIND_INFO in prolog order, as (code offset, kind, size)
    or (code offset, "save", offset, size), and the chained parent's (begin, unwind);
    None when the info is not understood (a frame register, an unknown code)."""
    head = binary.read(unwind, 4)
    if head is None or head[3] & 0xF:
        return None
    n = head[2]
    raw = binary.read(unwind + 4, 2 * n) if n else b""
    if raw is None:
        return None
    slots = [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]
    ops, i = [], 0
    try:
        while i < len(slots):
            at, op, info = slots[i] & 0xFF, (slots[i] >> 8) & 0xF, slots[i] >> 12
            if op == 0:                               # PUSH_NONVOL
                ops.append((at, "push", 8)); i += 1
            elif op == 1 and info == 0:               # ALLOC_LARGE, size/8 in one slot
                ops.append((at, "alloc", 8 * slots[i + 1])); i += 2
            elif op == 1:                             # ALLOC_LARGE, size in two slots
                ops.append((at, "alloc", slots[i + 1] | slots[i + 2] << 16)); i += 3
            elif op == 2:                             # ALLOC_SMALL
                ops.append((at, "alloc", 8 * info + 8)); i += 1
            elif op == 4:                             # SAVE_NONVOL, offset/8
                ops.append((at, "save", 8 * slots[i + 1], 8)); i += 2
            elif op == 5:                             # SAVE_NONVOL_FAR
                ops.append((at, "save", slots[i + 1] | slots[i + 2] << 16, 8)); i += 3
            elif op == 8:                             # SAVE_XMM128, offset/16
                ops.append((at, "save", 16 * slots[i + 1], 16)); i += 2
            elif op == 9:                             # SAVE_XMM128_FAR
                ops.append((at, "save", slots[i + 1] | slots[i + 2] << 16, 16)); i += 3
            else:
                return None
    except IndexError:
        return None
    parent = None
    if (head[0] >> 3) & 0x4:                          # UNW_FLAG_CHAININFO
        rec = unwind + 4 + 2 * ((n + 1) & ~1)
        begin, unw = binary.read_int(rec, 4), binary.read_int(rec + 8, 4)
        if begin is None or unw is None:
            return None
        parent = (binary.image_base + begin, binary.image_base + unw)
    return list(reversed(ops)), parent


def _save_slots(binary, entry: int, at: int) -> list[int]:
    """Frame offsets (from the stack pointer at entry) that hold callee-saved
    registers saved by the x64 PE function at `entry` when it executes `at`, from its
    unwind info: the primary entry's, and for code split off by shrink-wrapping the
    chained entries covering `at` -- of the entry covering `at`, only the codes already
    executed there. [] when there is no such information or it is not understood."""
    if binary.fmt != "pe" or binary.arch != "x64" or entry not in binary.pdata:
        return []
    cover = next(((b, u) for b, e, u in binary.pdata_ranges if b <= at < e), None)
    chain = []
    for _ in range(32):
        if cover is None:
            return []
        got = _unwind_ops(binary, cover[1])
        if got is None:
            return []
        chain.append((cover[0], got[0]))
        if got[1] is None:
            break
        cover = got[1]
    else:
        return []
    if chain[-1][0] != entry:
        return []
    cur, out, saves = 0, [], []
    for i, (begin, ops) in enumerate(chain):
        root = i == len(chain) - 1
        for off, kind, *v in ops:
            if i == 0 and begin + off > at:
                if root and kind != "save":
                    return []                         # (inside the prolog)
                continue                              # not executed yet at `at`
            if kind == "save":
                saves.append(v)
            elif not root:
                return []                             # (an allocation outside the prolog)
            else:
                cur -= v[0]
                if kind == "push":
                    out += range(cur, cur + 8)
    for off, size in saves:                           # relative to the stack pointer after the prolog
        out += range(cur + off, cur + off + size)
    return sorted(set(out))


class _Side:
    def __init__(self, tag: str, binary, info, callees: dict, make_spec=None):
        claripy = _claripy()
        self.tag, self.binary, self.info, self.callees = tag, binary, info, callees
        self.arch = binary.arch
        self.word = binary.ptr_size
        self.bits = 8 * self.word
        self.proj = bounding.angr_project(binary)
        self.engine = _classes()[1](self.proj)
        self.sp = "rsp" if self.arch == "x64" else "esp"
        self.ip = "rip" if self.arch == "x64" else "eip"
        self.sp0_name = f"sp0_{tag}"
        self.sp0 = claripy.BVS(self.sp0_name, self.bits, explicit_name=True)
        self.regs = _REGS[self.arch]
        self.reg_bits = {r: 8 * self.proj.arch.registers[r][1] for r in self.regs}
        self.calls = {c.addr: c for c in info.calls}
        self.make_spec = make_spec      # builds a callee model for an unrecorded tail jump
        self.ranges = info.code_ranges()
        self._ro = [(s.addr, s.end) for s in binary.sections if not s.writable]
        self.executed: set[int] = set()
        self.code = dict(info.code)     # this function's blocks and its inlined callees'
        self.heads, self.joins, self.bodies = self._loop_heads()
        self.fconv = _Conv(self.bits)     # for path feasibility (own symbols only)
        self._saves: dict[int, list[int]] = {}
        # Callees whose code runs in place of a model (entry -> FunctionInfo), and the
        # return sites of their calls: a `ret` to one goes back into this function.
        self.inlined: dict = {}
        self.inline_rets: dict[int, int] = {}       # return site -> inlined callee entry
        self.make_specs = None          # (binary, info) -> {target: CalleeSpec} (set by Checker)

    def saves(self, at: int) -> list[int]:
        """Frame offsets holding callee-saved registers at the instruction at `at`
        (inside an inlined callee: none of its own are known)."""
        if at not in self._saves:
            self._saves[at] = [] if self.inlined_at(at) else _save_slots(self.binary, self.info.entry, at)
        return self._saves[at]

    def frame_limits(self, mem, at: int) -> list[int]:
        """Frame offsets a stack object passed to the call at `at` must end at: the
        slots of saved registers and, inside inlined callees, of their return addresses
        and of their own saved registers. An inlined activation is found by its return
        address on the stack (its slot is the callee's stack pointer at entry)."""
        if not self.inlined_at(at):
            return self.saves(at)
        acts = []
        for o in sorted(mem.stack):
            ws = [mem.stack.get(o + i) for i in range(self.word)]
            if any(b is None or b.symbolic for b in ws):
                continue
            v = int.from_bytes(bytes(b.concrete_value for b in ws), "little")
            if v in self.inline_rets:
                acts.append((o, v))
        if not acts:
            return []
        out = [o for o, _ in acts]
        inner, outer = acts[0], acts[-1]
        out += [inner[0] + o for o in _save_slots(self.binary, self.inline_rets[inner[1]], at)]
        out += self.saves(outer[1])         # (the function's own, at the outermost call)
        return out

    def inlined_at(self, a: int) -> bool:
        return any(lo <= a < hi for fi in self.inlined.values() for lo, hi in fi.code_ranges())

    def can_inline(self, target: int) -> object | None:
        """The callee at `target` if its code can run in place of its model: code of
        this binary (not an import), without jump tables or calls to itself,
        and small. (Its loop heads become cut points.)"""
        if target in self.inlined or target == self.info.entry or self.make_specs is None:
            return None
        if not self.binary.is_exec(target) or target in self.binary.imports:
            return None
        try:
            fi = bounding.bound(self.binary, target)
        except Exception:      # (FneqError: not boundable)
            return None
        if sum(len(c) for c in fi.code.values()) > 2048 \
                or any(c.target == target for c in fi.calls):
            return None
        if fi.entry != target or not fi.code or fi.hazards or fi.jump_tables:
            return None
        return fi

    def inline(self, fi, other) -> None:
        """Run the callee `fi` in place of its model from now on."""
        self.inlined[fi.entry] = fi
        self.ranges = self.ranges + fi.code_ranges()
        self.code.update(fi.code)
        self.heads, self.joins, self.bodies = self._loop_heads()
        for c in fi.calls:
            self.calls.setdefault(c.addr, c)
        new = {k: v for k, v in self.make_specs(self.binary, fi).items() if k not in self.callees}
        mine, theirs = _merge_arities(new, {k: v for k, v in other.callees.items()
                                            if _canon(v.name) in {_canon(s.name) for s in new.values()}})
        self.callees.update(mine)
        other.callees.update(theirs)

    def _loop_heads(self) -> tuple[set[int], set[int], dict]:
        """Targets of the back edges of a depth-first search of the function's CFG
        (every cycle contains one, so code between them is acyclic), the blocks
        where paths join (two or more predecessors), and each head's loop: the
        blocks from which a back edge to it is reached without passing it."""
        succ: dict[int, set] = {}
        for a, data in self.code.items():
            try:
                irsb = self.proj.factory.block(a, size=len(data)).vex
            except Exception:
                continue
            out = {t for t in irsb.constant_jump_targets if self.in_code(t)}
            if irsb.jumpkind == "Ijk_Call":
                out = {a + len(data)} if self.in_code(a + len(data)) else set()
            succ[a] = out
        heads, state, back = set(), {}, []
        for root in [self.info.entry] + sorted(succ):
            if root in state:
                continue
            state[root] = 1
            stack = [(root, iter(sorted(succ.get(root, ()))))]
            while stack:
                n, it = stack[-1]
                for m in it:
                    if state.get(m) == 1:
                        heads.add(m)
                        back.append((n, m))
                    elif m not in state:
                        state[m] = 1
                        stack.append((m, iter(sorted(succ.get(m, ())))))
                        break
                else:
                    state[n] = 2
                    stack.pop()
        preds: dict = {}
        for a, out in succ.items():
            for t in out:
                preds.setdefault(t, set()).add(a)
        bodies: dict = {h: {h} for h in heads}
        for src, h in back:
            todo = [src]
            while todo:
                b = todo.pop()
                if b not in bodies[h]:
                    bodies[h].add(b)
                    todo.extend(preds.get(b, ()))
        return heads, {t for t, ps in preds.items() if len(ps) > 1}, bodies

    def in_loop(self, head: int, addr: int) -> bool:
        """Whether `addr` lies in the loop of `head` (in one of its blocks)."""
        code = self.code
        return any(b <= addr < b + len(code.get(b, b"")) or b == addr for b in self.bodies[head])

    def readonly(self, v: int) -> bool:
        return any(lo <= v < hi for lo, hi in self._ro)

    def in_image(self, v: int) -> bool:
        return any(sec.addr <= v < sec.end for sec in self.binary.sections)

    def in_code(self, a: int) -> bool:
        return any(lo <= a < hi for lo, hi in self.ranges)

    def spec(self, site):
        return self.callees.get(site.addr if site.indirect else site.target)


@dataclasses.dataclass
class _Exit:
    kind: str            # jump | ret | noreturn | trap
    guard: object        # claripy Bool
    target: int | None
    state: object        # angr SimState (registers + lsmem)
    at: int              # address of the instruction that leaves the segment


# ------------------------------------------------------------ abstract states

@dataclasses.dataclass
class _Abs:
    consts: tuple            # (dict loc->int, dict loc->int)
    stack_addr: tuple        # (dict loc->offset, dict loc->offset)
    pairs: frozenset         # {(locA, locB)}
    written: tuple           # (frozenset offsets, frozenset offsets)
    escaped: tuple           # (frozenset, frozenset)
    # {(locA, locB, d)}: A's value == B's value + d (d != 0, modulo the width), e.g.
    # a counter one ahead after loop rotation, a pointer to the next element
    offsets: frozenset = frozenset()

    @staticmethod
    def _const_offsets(ab: "_Abs") -> dict:
        """(locA, locB) -> d for locations holding constants a short distance apart."""
        out = {}
        for la, ca in ab.consts[0].items():
            if la[0] == "s":
                continue
            for lb, cb in ab.consts[1].items():
                if lb[0] != "s" and 0 < abs(ca - cb) <= _MAX_CONST_OFFSET:
                    out[la, lb] = ca - cb
        return out

    def join(self, o: "_Abs") -> "_Abs":
        def common(x, y):
            return {k: v for k, v in x.items() if y.get(k) == v}
        # A correspondence holds in a state if it is explicit there or if both
        # locations hold the same constant (a counter starting at 0 on both sides).
        def holds(ab, la, lb):
            if (la, lb) in ab.pairs:
                return True
            ca, cb = ab.consts[0].get(la), ab.consts[1].get(lb)
            return ca is not None and ca == cb
        def const_pairs(ab):
            by_value: dict = {}
            for lb, v in ab.consts[1].items():
                by_value.setdefault(v, []).append(lb)
            return {(la, lb) for la, v in ab.consts[0].items() for lb in by_value.get(v, ())}
        consts = (common(self.consts[0], o.consts[0]), common(self.consts[1], o.consts[1]))
        cands = (self.pairs | const_pairs(self)) & (o.pairs | const_pairs(o))
        # (a pair of locations that keep the same constant is implied by `consts`)
        pairs = frozenset(p for p in cands if holds(self, *p) and holds(o, *p)
                          and not (p[0] in consts[0] and p[1] in consts[1]))
        # Offset correspondences likewise: explicit, or between constants (i = 0 /
        # j = 1 on entry, i = 1 / j = 2 after one iteration: i == j - 1 at the head).
        def offs(ab):
            d = {(la, lb): v for la, lb, v in ab.offsets}
            for k, v in self._const_offsets(ab).items():
                d.setdefault(k, v)
            return d
        oa, ob = offs(self), offs(o)
        offsets = frozenset((la, lb, d) for (la, lb), d in oa.items() if ob.get((la, lb)) == d
                            and not (la in consts[0] and lb in consts[1]))
        return _Abs(consts,
                    (common(self.stack_addr[0], o.stack_addr[0]),
                     common(self.stack_addr[1], o.stack_addr[1])),
                    pairs,
                    (self.written[0] | o.written[0], self.written[1] | o.written[1]),
                    (self.escaped[0] | o.escaped[0], self.escaped[1] | o.escaped[1]),
                    offsets)


def _fp_value(name: str, rnd: int, bits: int) -> int:
    h = hashlib.sha256(f"{rnd}:{name}".encode()).digest()
    while len(h) * 8 < bits:
        h += hashlib.sha256(h).digest()
    return int.from_bytes(h, "little") & ((1 << bits) - 1)


# ------------------------------------------------------------------ checker

class Checker:
    def __init__(self, sig, side_a, side_b, checks, map_b, timeout_ms: int = 2000,
                 budget_s: float = 120.0, max_visits: int = 3000, make_spec=None,
                 no_alias: bool = False, make_specs=None):
        ret_regs = []
        from fneq.signature import RET_KINDS
        for r in RET_KINDS[sig.ret_kind][sig.arch]:
            if r.startswith("ST"):
                raise _Unsup("x87 return values are not modelled by the lockstep checker")
            ret_regs.append(r)
        self.sig, self.checks = sig, checks
        self.ret_regs = ret_regs
        self.saved = list(sig.convention.callee_saved)
        ca, cb = _merge_arities(side_a.callees, side_b.callees)
        self.A = _Side("A", side_a.binary, side_a.info, ca, make_spec and make_spec("A"))
        self.B = _Side("B", side_b.binary, side_b.info, cb, make_spec and make_spec("B"))
        self.timeout_ms = timeout_ms
        # The budget is CPU time of this process (Z3 runs in it): a verdict does not
        # depend on how busy the machine is (batch workers share the cores).
        self.cpu0 = time.process_time()
        self.deadline = self.cpu0 + budget_s
        self.max_visits = max_visits
        self.registry: dict = {}
        self.conv_a = _Conv(self.A.bits, registry=self.registry)
        self.conv_b = _Conv(self.B.bits, map_const=map_b, registry=self.registry)
        self.frame_pairs: dict[int, int] = {}       # escaped A frame offset -> B offset
        # Callees without a name (no symbol, no correspondence entry) that are called
        # at the same place on both sides: assumed to correspond, and reported.
        self.callee_pairs: dict = {}                # ("A"|"B", name) -> other side's name
        # --no-alias: distinct pointers address disjoint memory (see _Match.regions)
        self.no_alias = no_alias
        self.call_returns = _CALL_RETURNS.get(sig.convention.name, {"eax", "edx"})
        self.inlined_names: list[str] = []          # callees run in place of their model
        self.max_inlines = 8
        if make_specs is not None:
            self.A.make_specs, self.B.make_specs = make_specs("A"), make_specs("B")
        # callees whose original (side A) copy is variadic
        self.variadic = {_canon(s.name) for s in ca.values() if s.variadic}
        self.no_alias_used = 0                      # exit pairs proven with regions
        self.no_alias_flat = 0                      # ... that needed the flat memory
        self.visits = 0
        self.queries = 0
        self.proved = 0
        self.table: dict[tuple[int, int], _Abs] = {}
        self.cur: tuple[int, int] = (side_a.info.entry, side_b.info.entry)
        # Segment boundaries made path-sensitive after a failed obligation: the
        # predecessors execute straight through them (per side: block addresses).
        self.inline: tuple[set, set] = (set(), set())
        self.refined: list[tuple[int, int]] = []
        self.max_refinements = 8
        self.extended_joins: list[tuple[str, int]] = []

    # ---------------------------------------------------------------- driver
    def run(self) -> Result:
        res = self._refine()
        if self.callee_pairs and res.status == "equivalent":
            assumed = sorted(f"{a} = {b}" for (side, a), b in self.callee_pairs.items() if side == "A")
            res.data["assumed_callee_pairs"] = assumed
            res.reason += f", assuming {len(assumed)} unnamed callee pair(s) correspond: " \
                          + ", ".join(assumed[:6])
        if self.no_alias_used and res.status == "equivalent":
            res.data["no_alias_region_pairs"] = self.no_alias_used
            res.data["no_alias_flat_pairs"] = self.no_alias_flat
            res.reason += ", assuming pointers do not alias (--no-alias)"
        return res

    def _refine(self) -> Result:
        """The fixpoint, refined on failure (counterexample-guided, as in lockstep
        composition with refinement): an obligation that fails in a segment pair
        reached from a predecessor may fail only because the abstract entry state
        lost information at the join. The pair's boundary is then made
        path-sensitive -- its predecessors continue through it with their exact
        values -- and the fixpoint is recomputed. A real difference fails again (in
        the predecessor's segment, at the same instructions); a spurious one goes
        away. Re-segmenting never affects soundness.

        A failure that comes back unchanged (same instructions, same obligation) after
        a refinement ends the loop: refining further back would only re-run it."""
        t0 = time.monotonic()
        failures = set()
        while True:
            self.unmatched_calls = ()
            self.unmatched_join = None
            res, key = self._fixpoint(t0)
            fail = None
            if res.status == "differs":
                d = res.data["first_difference"]
                fail = (d["a"], d["b"], d["what"].split(" (unmatched memory reads")[0])
                # Calls that do not line up because one side inlined a helper the other
                # calls: run the helper's code on the calling side (exact) and retry.
                if time.process_time() <= self.deadline and self._inline_unmatched():
                    continue
                # A join used to limit path growth can cut one side before a
                # call the other side already reaches. Extend that side through
                # the join; loop heads remain cut points and every check reruns.
                if self.unmatched_join is not None and len(self.extended_joins) < 8 \
                        and time.process_time() <= self.deadline:
                    tag, target = self.unmatched_join
                    index = 0 if tag == "A" else 1
                    if target not in self.inline[index]:
                        self.inline[index].add(target)
                        self.extended_joins.append((tag, target))
                        continue
            repeated = fail in failures
            failures.add(fail)
            if res.status != "differs" or key is None or key == self._start() \
                    or key in self.refined or len(self.refined) >= self.max_refinements \
                    or repeated or time.process_time() > self.deadline:
                if self.refined:
                    res.data["refined_boundaries"] = [[hex(a), hex(b)] for a, b in self.refined]
                if self.inlined_names:
                    res.data["inlined_callees"] = self.inlined_names
                if self.extended_joins:
                    res.data["extended_join_cuts"] = [[tag, hex(target)] for tag, target in self.extended_joins]
                if repeated:
                    res.data["refinement_stopped"] = "the same obligation failed again"
                return res
            self.refined.append(key)
            self.inline[0].add(key[0])
            self.inline[1].add(key[1])

    def _start(self):
        return (self.A.info.entry, self.B.info.entry)

    def _hint_join_extension(self, ea, eb):
        for index, (tag, side, edge) in enumerate((("A", self.A, ea), ("B", self.B, eb))):
            if edge.kind == "jump" and edge.target in side.joins \
                    and edge.target not in side.heads and edge.target not in self.inline[index]:
                self.unmatched_join = (tag, edge.target)
                return

    def _inline_unmatched(self) -> bool:
        """Inline the callees of the last difference's unmatched calls that the other
        side never calls (it has their code inlined, or calls something else); True if
        any was inlined."""
        done = False
        unpaired = {"A": set(), "B": set()}
        for tag, at in self.unmatched_calls:
            s = self.A if tag == "A" else self.B
            site = s.calls.get(at)
            spec = s.spec(site) if site is not None else None
            if spec is not None:
                unpaired[tag].add(_canon(spec.name))
        for tag, at in self.unmatched_calls:
            if len(self.inlined_names) >= self.max_inlines:
                break
            s, other = (self.A, self.B) if tag == "A" else (self.B, self.A)
            site = s.calls.get(at)
            if site is None or site.indirect or site.target is None:
                continue
            spec = s.callees.get(site.target)
            if spec is None or any(_canon(v.name) == _canon(spec.name) for v in other.callees.values()):
                continue
            fi = s.can_inline(site.target)
            if fi is None:
                continue
            # Evidence that the other side has its code inlined: it has no unmatched
            # call at all (this side's call is extra), or its unmatched calls include
            # calls the callee makes (a call where the other side has the callee's).
            inner = {_canon(v.name) for v in s.make_specs(s.binary, fi).values()}
            theirs = unpaired["B" if tag == "A" else "A"]
            if theirs and not inner & theirs:
                continue
            s.inline(fi, other)
            self.inlined_names.append(f"{tag}:{spec.name}")
            done = True
        return done

    def format_args(self, args) -> int | None:
        """How many arguments a printf-like call uses: up to its first constant pointer
        to a string with a '%' in the original's image, plus one per conversion (and per
        `*`). None if no such format string is passed."""
        for j, v in enumerate(args):
            if v.symbolic or v.size() != self.A.bits:
                continue
            fmt = _image_string(self.A.binary, v.concrete_value)
            if fmt is not None and "%" in fmt:
                return j + 1 + _conversions(fmt)
        return None

    def junk(self, v) -> bool:
        """Is v made only of junk (see _UNINIT, _CLOBBER)? Constants are not."""
        names = v.variables
        if not names:
            return False
        for n in names:
            if _UNINIT.match(n):
                continue
            m = _CLOBBER.match(n)
            if m and m.group(1) not in self.call_returns:
                continue
            return False
        return True

    def kept_bytes(self, v) -> list[int]:
        """Indices of the bytes of v that are not junk."""
        if not self.junk(v) and not any(_UNINIT.match(n) or _CLOBBER.match(n) for n in v.variables):
            return list(range(v.size() // 8))
        return [i for i in range(v.size() // 8) if not self.junk(v[8 * i + 7:8 * i])]

    def _fixpoint(self, t0):
        start = self._start()
        self.table = {start: self._entry_abs()}
        self.frame_pairs = {}
        work = [start]
        try:
            while work:
                key = work.pop()
                if time.process_time() > self.deadline:
                    raise _Unsup("lockstep time budget exhausted")
                self.visits += 1
                if self.visits > self.max_visits:
                    raise _Unsup("lockstep fixpoint did not converge")
                self.cur = key
                for succ, new in self._visit(key, self.table[key]):
                    old = self.table.get(succ)
                    joined = new if old is None else old.join(new)
                    if old is None or joined != old:
                        self.table[succ] = joined
                        if succ not in work:
                            work.append(succ)
        except _Differs as e:
            self.unmatched_calls = e.calls
            return Result("differs", e.what, self._data(t0, {
                "a": hex(e.a) if e.a is not None else None,
                "b": hex(e.b) if e.b is not None else None,
                "segment": [hex(self.cur[0]), hex(self.cur[1])], "what": e.what})), self.cur
        except _Unsup as e:
            return Result("unknown", str(e), self._data(t0, None, segment=self.cur)), None
        except z3.Z3Exception as e:
            return Result("unknown", f"z3 error: {e}", self._data(t0, None, segment=self.cur)), None
        return Result("equivalent",
                      f"lockstep proof over {len(self.table)} segment pairs "
                      f"({self.visits} visits, {self.proved} obligations)",
                      self._data(t0, None)), None

    def _data(self, t0, diff, segment=None) -> dict:
        d = {"segment_pairs": len(self.table), "visits": self.visits, "queries": self.queries,
             "obligations_proved": self.proved, "seconds": round(time.monotonic() - t0, 2),
             "cpu_seconds": round(time.process_time() - self.cpu0, 2),
             "pairs": sorted([hex(a), hex(b)] for a, b in self.table)[:200]}
        if diff:
            d["first_difference"] = diff
        if segment:
            d["stopped_at"] = [hex(segment[0]), hex(segment[1])]
        for s in (self.A, self.B):
            missed = [a for a in sorted(s.info.code)
                      if not any(a <= x < a + len(s.info.code[a]) for x in s.executed)]
            d[f"unvisited_{s.tag}"] = [hex(a) for a in missed][:50]
        return d

    # ------------------------------------------------------------ entry
    def _entry_abs(self) -> _Abs:
        consts = {("r", r): v for r, v in _FIXED.items() if r in self.A.reg_bits}
        pairs = frozenset((("r", r), ("r", r)) for r in self.A.regs
                          if r not in _FIXED and r != self.A.sp)
        sa = {("r", self.A.sp): 0}
        return _Abs((dict(consts), dict(consts)), (sa, {("r", self.B.sp): 0}), pairs,
                    (frozenset(), frozenset()), (frozenset(), frozenset()))

    def _entry_values(self, ab: _Abs, visit: int):
        claripy = _claripy()
        parent: dict = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x
        for la, lb in ab.pairs:
            ra, rb = find(("A",) + la), find(("B",) + lb)
            if ra != rb:
                parent[ra] = rb
        members = {("A",) + la for la, _ in ab.pairs} | {("B",) + lb for _, lb in ab.pairs}
        names: dict = {}
        vals_a: dict = {}
        offset_b = {}
        for la, lb, d in sorted(ab.offsets):
            offset_b.setdefault(lb, (la, d))
        out = []
        for i, s in enumerate((self.A, self.B)):
            tag = s.tag
            regs, stack = {}, {}

            def value(loc, bits):
                v = value_of(loc, bits)
                if i == 0:
                    vals_a[loc] = v
                return v

            def value_of(loc, bits):
                if loc in ab.consts[i]:
                    return claripy.BVV(ab.consts[i][loc], bits)
                if loc in ab.stack_addr[i]:
                    return s.sp0 + ab.stack_addr[i][loc]
                key = (tag,) + loc
                if key in members:
                    root = find(key)
                    nm = names.setdefault(root, f"e{visit}_{len(names)}")
                    return claripy.BVS(nm, bits, explicit_name=True)
                if i == 1 and loc in offset_b:
                    la, d = offset_b[loc]
                    va = vals_a.get(la)
                    if va is not None and va.size() == bits:
                        return va - (d % (1 << bits))
                lname = loc[1] if loc[0] == "r" else f"s{loc[1]}"
                return claripy.BVS(f"{tag}{visit}_{lname}", bits, explicit_name=True)
            for r in s.regs:
                regs[r] = value(("r", r), s.reg_bits[r])
            for off in ab.written[i]:
                stack[off] = value(("s", off), 8)
            # Whole stack words (a saved register, a spilled pointer): their value
            # replaces the bytes', which only drops byte-level facts (sound).
            words = {}
            sw = {loc for loc in list(ab.consts[i]) + list(ab.stack_addr[i]) if loc[0] == "sw"}
            sw |= {p[i] for p in ab.pairs if p[i][0] == "sw"}
            sw |= {p[i] for p in ab.offsets if p[i][0] == "sw"}
            for loc in sorted(sw, key=lambda l: (l[1], -l[2])):
                _, off, n = loc
                v = value(loc, 8 * n)
                for k in range(n):
                    stack[off + k] = claripy.Extract(8 * k + 7, 8 * k, v)
                words[off] = (v, tuple(stack[off + k] for k in range(n)))
            out.append((regs, stack, words))
        return out

    # ------------------------------------------------------------ execution
    def _run_segment(self, s: _Side, start: int, regs: dict, stack: dict, words: dict, escaped,
                     visit, cuts=None):
        """All paths from `start` to the segment's exits; `cuts` are the blocks where
        paths leave the region (default: the loop heads)."""
        claripy = _claripy()
        from angr import sim_options as o
        LsMem, _ = _classes()
        # (Paths and their conditions are this checker's own, from all_successors: angr
        # need not keep constraints nor concretise symbolic jump targets -- the copies
        # it would make go only to flat_successors.)
        st = s.proj.factory.blank_state(
            addr=start, add_options={o.LAZY_SOLVES, o.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
                                     o.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
                                     o.NO_SYMBOLIC_JUMP_RESOLUTION, o.FAST_REGISTERS},
            remove_options={o.TRACK_CONSTRAINTS, o.TRACK_CONSTRAINT_ACTIONS})
        for r, v in regs.items():
            st.registers.store(r, v)
        st.register_plugin("lsmem", LsMem(s, visit, stack, None, escaped, words=words))
        exits: list[_Exit] = []
        inline = self.inline[0 if s is self.A else 1]
        cuts = s.heads if cuts is None else cuts
        stops = sorted(cuts)
        # (state, blocks on this path -> times entered, path condition, speculation)
        # Speculation: a path that reaches a loop head goes on, in
        # case the loop runs to completion by constants (a constant trip count, a
        # loop over constant data) -- lining up with an unrolled copy on the other
        # side. spec = (head, state there, guard there, jump address, steps there);
        # at the first real fork in the loop, or a cycle not decided by constants,
        # the path falls back to that state: it leaves the region at the head,
        # exactly as without speculation. A fork after the loop (the path left it
        # by constants) ends the speculation: the path goes on as any other.
        pending = [(st, {start: 1}, claripy.true(), None)]
        steps = forks = 0

        def fall_back(spec):
            head, ck, ck_guard, ck_at, _ = spec
            exits.append(_Exit("jump", ck_guard, head, ck, ck_at))
        while pending:
            st, seen, pc, spec = pending.pop()
            steps += 1
            if steps > 4096:
                raise _Unsup(f"segment at 0x{start:x} is too long")
            addr = None if st.regs.ip.symbolic else st.regs.ip.concrete_value
            if addr is None or not s.in_code(addr):
                raise _Unsup(f"execution left the function at {addr}")
            try:
                # A cut point always starts a block: straight-line code running into
                # one (no jump there) is cut at it too.
                st.lsmem.guard = pc
                succ = s.engine.process(st, extra_stop_points=stops)
            except (_Unsup, _Differs):
                raise
            except Exception as e:  # lifting / engine failure
                raise _Unsup(f"symbolic execution failed at 0x{addr:x}: {type(e).__name__}: {e}")
            insns = succ.artifacts.get("insn_addrs") or [addr]
            last = insns[-1]
            s.executed.update(insns)
            outs = [x for x in succ.all_successors
                    if x.scratch.guard.symbolic or not x.scratch.guard.is_false()]
            # VEX's alignment checks (movaps & co.): stack addresses are aligned by the
            # ABI, so a check on them cannot fail; one that can is a fault exit.
            outs = [x for x in outs if x.history.jumpkind != "Ijk_SigSEGV"
                    or self._satisfiable(s, x.scratch.guard)]
            outs = [y for x in outs for y in self._jump_table(s, x, pc, last)]
            if len(outs) == 1 and outs[0].scratch.guard.symbolic and \
                    not self._satisfiable(s, claripy.Not(outs[0].scratch.guard)):
                outs[0].scratch.guard = claripy.true()
            single = len(outs) == 1 and not outs[0].scratch.guard.symbolic
            if spec is not None and not single and not s.in_loop(spec[0], addr):
                spec = None             # the loop ran to completion: keep the path
            if spec is not None and (not single or steps - spec[4] > _MAX_SPECULATION):
                fall_back(spec)
                continue
            # A conditional branch decided by constants (a loop with a constant trip
            # count): followed again and again, so that the loop runs to completion
            # within the segment and lines up with an unrolled copy on the other side.
            decided = single and sum(1 for x in succ.all_successors
                                     if x.history.jumpkind == "Ijk_Boring") > 1
            # The exits of one block, in order: each is taken only if the earlier
            # ones are not (which makes the guards of the segment's paths disjoint).
            guards, rest = [], claripy.true()
            for x in outs:
                g = x.scratch.guard
                guards.append(g if rest.is_true() else claripy.And(rest, g))
                rest = claripy.Not(g) if rest.is_true() else claripy.And(rest, claripy.Not(g))
            for x, xg in zip(outs, guards):
                jk = x.history.jumpkind
                ip = x.regs.ip
                if jk == "Ijk_Call":
                    if not single:
                        raise _Unsup(f"conditional call at 0x{last:x}")
                    site = s.calls.get(last)
                    if site is None:
                        raise _Unsup(f"call at 0x{last:x} has no callee model")
                    ret_to = succ.artifacts.get("irsb_size")
                    ret_to = (succ.addr + ret_to) if ret_to else None
                    if not site.indirect and not site.tail and site.target in s.inlined \
                            and ret_to is not None and s.in_code(ret_to):
                        # An inlined callee: its code runs here (the call pushed the
                        # return address); its `ret` comes back to ret_to.
                        s.inline_rets[ret_to] = site.target
                        lo_hi = s.inlined[site.target].code_ranges()
                        fresh = {a: n for a, n in seen.items() if not any(lo <= a < hi for lo, hi in lo_hi)}
                        pending.append((x, {**fresh, site.target: 1}, pc, spec))
                        continue
                    self._do_call(s, x, site, last, pc)
                    if ret_to is None or not s.in_code(ret_to):
                        exits.append(_Exit("noreturn", claripy.true(), None, x, last))
                    else:
                        x.regs.ip = ret_to
                        pending.append((x, seen, pc, spec))
                    continue
                guard = xg if pc.is_true() else claripy.And(pc, xg)
                if jk == "Ijk_SigSEGV":
                    exits.append(_Exit("trap:" + jk, guard, None, x, last))
                    continue
                if ip.symbolic:
                    raise _Unsup(f"indirect jump at 0x{last:x} (target {str(ip)[:120]})")
                tgt = ip.concrete_value
                if jk == "Ijk_Ret":
                    if tgt in s.inline_rets and s.inlined_at(last):
                        # back from an inlined callee: its frame below the stack
                        # pointer is dead
                        sp_off = x.lsmem.classify(x.registers.load(s.sp))
                        if sp_off[0] == "stack":
                            for o in [o for o in x.lsmem.stack if o < sp_off[1]]:
                                del x.lsmem.stack[o]
                            for o in [o for o in x.lsmem.words if o < sp_off[1]]:
                                del x.lsmem.words[o]
                        pending.append((x, seen, pc, spec))
                        continue
                    if tgt != RET_ADDR[s.arch]:
                        raise _Unsup(f"return to a modified return address at 0x{last:x}")
                    exits.append(_Exit("ret", guard, None, x, last))
                elif jk == "Ijk_Boring":
                    if not s.in_code(tgt):
                        if s.inlined_at(last):
                            # an inlined callee's tail call: the callee it jumps to returns
                            # to the inlined call's return site
                            back = self._inline_tail(s, x, last, tgt, guard)
                            if back is None:
                                raise _Unsup(f"tail jump inside an inlined callee at 0x{last:x}")
                            if back == RET_ADDR[s.arch]:
                                exits.append(_Exit("ret", guard, None, x, last))
                            else:
                                pending.append((x, seen, pc, spec))
                            continue
                        site = s.calls.get(last)
                        # A conditional branch may have two external successors.
                        # Select the tail model of the edge actually taken.
                        site = next((c for c in s.info.calls if c.addr == last and c.tail
                                     and c.target == tgt), site)
                        if (site is None or not site.tail) and s.make_spec is not None:
                            site = self._tail_site(s, last, tgt)
                        if site is None or not site.tail:
                            raise _Unsup(f"jump out of the function at 0x{last:x}")
                        self._do_call(s, x, site, last, guard, tail=True)
                        exits.append(_Exit("ret", guard, None, x, last))
                    elif (decided or (spec is not None and single)) and seen.get(tgt, 0) < 256 \
                            and (pc.is_true() or spec is not None):
                        # (not after a fork unless speculating: paths unrolling on
                        # their own diverge. While speculating, a path without forks
                        # is deterministic, and every edge of it -- also one back to
                        # a loop head -- is followed, boundedly.)
                        pending.append((x, {**seen, tgt: seen.get(tgt, 0) + 1}, pc, spec))
                    elif tgt not in seen and not (tgt in inline and tgt in cuts) \
                            and (tgt not in cuts or tgt in s.heads):
                        # Acyclic code: both branches are followed (path-sensitively)
                        # up to the next loop head -- speculatively beyond it -- or return.
                        nxt = pc if single else guard
                        if not single:
                            if not self._feasible(s, guard):
                                continue
                            forks += 1
                            if forks > _MAX_FORKS:
                                raise _TooManyPaths(f"too many paths in the region at 0x{start:x}")
                        sp = spec
                        if tgt in s.heads and sp is None:
                            ck = x.copy()
                            _normalise_flags(ck)
                            sp = (tgt, ck, guard, last, steps)
                        pending.append((x, {**seen, tgt: 1}, nxt, sp))
                    elif tgt in inline and tgt not in seen:
                        # A refined boundary: the path continues through it.
                        pending.append((x, {**seen, tgt: 1}, guard, spec))
                    elif spec is not None:
                        fall_back(spec)        # a cycle not decided by constants
                    else:
                        _normalise_flags(x)
                        exits.append(_Exit("jump", guard, tgt, x, last))
                elif jk in ("Ijk_SigTRAP", "Ijk_SigILL", "Ijk_NoDecode"):
                    exits.append(_Exit("trap:" + jk, guard, None, x, last))
                else:
                    raise _Unsup(f"{jk} at 0x{last:x}")
        return exits

    def _inline_tail(self, s: _Side, x, at: int, tgt: int, guard):
        """Model the tail call at `at` (inside an inlined callee) and set the state to
        continue at the inlined call's return site; None if that is not possible."""
        mem = x.lsmem
        sp_off = mem.classify(x.registers.load(s.sp))
        if sp_off[0] != "stack":
            return None
        ret = mem.stack_read(sp_off[1], s.word)
        if ret.symbolic:
            return None
        back = ret.concrete_value
        if back not in s.inline_rets and not (back == RET_ADDR[s.arch] and sp_off[1] == 0):
            return None
        site = s.calls.get(at)
        if (site is None or not site.tail) and s.make_spec is not None:
            site = self._tail_site(s, at, tgt)
        if site is None or not site.tail:
            return None
        self._do_call(s, x, site, at, guard, tail=True)
        x.regs.ip = back
        return back

    def _jump_table(self, s: _Side, x, pc, at: int) -> list:
        """A jump whose target comes from one read of read-only image data at a
        symbolic address (a switch table): one successor per address the path allows,
        guarded by `addr == a`, the read value fixed to the table's contents there. The
        read leaves the memory model (it is of constant data, not of the shared memory).
        Other successors are returned unchanged."""
        claripy = _claripy()
        ip = x.regs.ip
        if x.history.jumpkind != "Ijk_Boring" or not ip.symbolic:
            return [x]
        reads = [e for e in x.lsmem.events if e[0] == "read" and e[1].variables <= ip.variables]
        if len(reads) != 1 or ip.variables != reads[0][1].variables or not reads[0][2].symbolic:
            return [x]
        _, sym, addr, size, _ = reads[0]
        g = x.scratch.guard if pc.is_true() else claripy.And(pc, x.scratch.guard)
        sol = z3.Solver()
        sol.set("timeout", self.timeout_ms)
        sol.add(*_alignment(s))
        if g.symbolic:
            sol.add(s.fconv(g))
        za, zs, ztgt = s.fconv(addr), s.fconv(sym), s.fconv(ip)
        cases = []
        while True:
            r = sol.check()
            if r == z3.unsat:
                break
            if r != z3.sat or len(cases) >= _MAX_TABLE:
                return [x]
            a = sol.model().eval(za, model_completion=True).as_long()
            data = s.binary.read(a, size) if s.readonly(a) and s.readonly(a + size - 1) else None
            if data is None:
                return [x]
            val = int.from_bytes(data, "little")
            tgt = z3.simplify(z3.substitute(ztgt, (zs, z3.BitVecVal(val, 8 * size))))
            if not z3.is_bv_value(tgt):
                return [x]
            cases.append((a, val, tgt.as_long()))
            sol.add(za != a)
        out = []
        for a, val, tgt in cases:
            y = x.copy()
            y.lsmem.events = [e for e in y.lsmem.events if e is not reads[0]]
            y.scratch.guard = claripy.And(x.scratch.guard, addr == a,
                                          sym == claripy.BVV(val, 8 * size))
            y.regs.ip = claripy.BVV(tgt, s.bits)
            out.append(y)
        return out

    def _satisfiable(self, s: _Side, cond) -> bool:
        """Whether `cond` (over side s's values) can hold when the stack is aligned as
        the ABI requires; True when undecided."""
        if not cond.symbolic:
            return cond.is_true()
        if s.sp0_name not in cond.variables:
            return True
        sol = z3.Solver()
        sol.set("timeout", self.timeout_ms)
        sol.add(*_alignment(s), _Conv(s.bits)(cond))
        return sol.check() != z3.unsat

    def _feasible(self, s: _Side, cond) -> bool:
        """Whether a path condition of side s can hold (over-approximated: memory
        reads are unconstrained); True when undecided."""
        if not cond.symbolic:
            return cond.is_true()
        sol = z3.Solver()
        sol.set("timeout", self.timeout_ms)
        sol.add(*_alignment(s), s.fconv(cond))
        return sol.check() != z3.unsat

    def _tail_site(self, s: _Side, at: int, tgt: int):
        """A jump to another function that bounding did not record as a tail call."""
        if not s.binary.is_exec(tgt):
            return None
        site, spec = s.make_spec(s.binary, s.info, at, tgt)
        if spec is None:
            return None
        other = self.B if s is self.A else self.A
        mine, theirs = _merge_arities({tgt: spec}, {k: v for k, v in other.callees.items()
                                                    if _canon(v.name) == _canon(spec.name)})
        s.callees[tgt] = mine[tgt]
        other.callees.update(theirs)    # (both sides keep one model of the callee)
        s.calls[at] = site
        return site

    def _do_call(self, s: _Side, st, site, at: int, pc, tail: bool = False) -> None:
        claripy = _claripy()
        spec = s.spec(site)
        if spec is None:
            raise _Unsup(f"no model for the call at 0x{at:x}")
        mem = st.lsmem
        sp = st.registers.load(s.sp)
        sp_off = mem.classify(sp)
        if sp_off[0] != "stack":
            raise _Unsup(f"stack pointer is not a frame address at 0x{at:x}")
        sp_off = sp_off[1]
        args = []
        if spec.indirect:
            args.append(st.regs.ip)
        for r in spec.reg_args:
            name, _, w = r.partition(":")
            v = st.registers.load(name.lower())
            args.append(v[int(w) - 1:0] if w else v)
        for off, size in spec.stack_operands(s.word):
            args.append(mem.stack_read(sp_off + off, size))
        k = mem.ncalls
        mem.ncalls += 1
        passed, unbounded, ends = set(), set(), {}
        contracts = {item["arg"]: item for item in spec.frame_args}
        if len(contracts) != len(spec.frame_args) or any(i >= len(args) for i in contracts):
            raise _Unsup(f"call {spec.name}: invalid stack-pointee contract argument")
        for j, a in enumerate(args):
            if s.sp0_name in a.variables:
                off = _sp_offset(a, s.sp0_name, s.bits)
                if off is None:
                    raise _Unsup(f"a variable stack address is passed to a call at 0x{at:x}")
                passed.add(off)
                contract = contracts.get(j)
                if contract is None:
                    unbounded.add(off)
                    continue
                if "size" in contract:
                    size = contract["size"]
                else:
                    index = contract["size_arg"]
                    if index >= len(args):
                        raise _Unsup(f"call {spec.name}: invalid stack-pointee size argument")
                    length = args[index]
                    if contract.get("indirect"):
                        loc = mem.classify(length)
                        if loc[0] != "stack":
                            raise _Unsup(f"call {spec.name}: pointee extent requires a stack size pointer")
                        length = mem.stack_read(loc[1], contract["indirect"])
                    if length.symbolic:
                        raise _Unsup(f"call {spec.name}: symbolic stack-pointee extent")
                    size = length.concrete_value * contract.get("scale", 1)
                if size > 1024 * 1024:
                    raise _Unsup(f"call {spec.name}: stack-pointee extent exceeds model limit")
                ends[off] = max(ends.get(off, off), off + size)
        mem.escaped |= passed
        # (--no-alias: callees keep no frame address after they return -- this call sees
        # only the objects passed to it and those reachable from them; else every object
        # whose address escaped so far.)
        visible = set(passed) if self.no_alias else set(mem.escaped)
        limits = s.frame_limits(mem, at)
        # Contents of the escaped frame objects the callee can see. A frame address
        # stored in one of them lets the callee reach that object too (transitively);
        # its bytes are compared as ("sp", object offset, byte) -- by frame correspondence.
        # An object runs up to the next escaped address, but never into the return-address
        # slot [0, word) or a register save slot of the prolog; on x64 one above the
        # return address (a home slot or stack argument whose address is passed) is that slot.
        while True:
            frames = {}
            bases = sorted(mem.escaped)      # (all escaped addresses bound the objects)
            for i, base in enumerate(bases):
                if base not in visible:
                    continue
                nxt = bases[i + 1] if i + 1 < len(bases) else None
                if base in ends and base not in unbounded:
                    end = ends[base]
                    # Never hide a return-address or saved-register overwrite by
                    # truncating an explicit range to the default frame boundary.
                    boundary = min([o for o in limits if o >= base] +
                                   ([0] if base < 0 else [s.word] if base < s.word else []),
                                   default=None)
                    if boundary is not None and end > boundary:
                        raise _Unsup(f"call {spec.name}: stack-pointee extent crosses a saved slot")
                elif base < 0:
                    end = 0 if nxt is None else min(nxt, 0)
                elif base < s.word:
                    raise _Unsup(f"the return address is passed to a call at 0x{at:x}")
                elif s.arch == "x64":       # one argument slot (one argument each)
                    end = (base // s.word + 1) * s.word
                    end = end if nxt is None else min(nxt, end)
                else:                       # (x86: structures are passed by value)
                    end = nxt if nxt is not None else                         max([base + s.word] + [o + 1 for o in mem.stack if o > base])
                end = min([end] + [o for o in limits if o > base])
                # (bytes never written: uninitialised, None -- below the return address,
                # and in x64's home space, which holds no argument at entry)
                home = s.word + 4 * s.word if s.arch == "x64" else s.word
                frames[base] = [mem.stack.get(o) if o < home or o in mem.stack else mem.default_byte(o)
                                for o in range(base, end)]
            reach = set()
            for base, contents in frames.items():
                for i, b in enumerate(contents):
                    if b is None or s.sp0_name not in b.variables:
                        continue
                    w = next(((o, v) for o, (v, bs) in mem.words.items()
                              if o <= base + i < o + len(bs) and mem.stack.get(base + i) is bs[base + i - o]),
                             None)
                    tgt = None if w is None else _sp_offset(w[1], s.sp0_name, s.bits)
                    if tgt is None:
                        raise _Unsup(f"a stack address escapes through a stack object at 0x{at:x}")
                    contents[i] = ("sp", tgt, base + i - w[0])
                    reach.add(tgt)
            if reach <= visible:
                break
            visible |= reach
            mem.escaped |= reach
        results = [(r.lower(), 8 * s.proj.arch.registers[r.lower()][1]) for r in spec.results]
        results.append(("cc_dep1", s.bits))
        # (pc: the path's condition when it calls -- facts the call may rely on)
        mem.events.append(("call", k, spec.name, args, len(mem.heap), frames, at, results, pc))
        # Effects: memory outside the frame and the escaped objects become callee
        # results, result registers are fresh.
        tag, v = s.tag, mem.visit
        mem.heap.append(("havoc", f"h{tag}{v}_c{k}"))
        for base, contents in frames.items():
            for i in range(len(contents)):
                mem.stack[base + i] = claripy.BVS(f"f{tag}{v}_c{k}_{i}_{base}", 8, explicit_name=True)
        # The callee owns its home space (MS x64 shadow area). Its stack-argument
        # slots are not clobbered: compiled callers never read them back, and an
        # over-estimated arity (varargs) would reach into the caller's own frame.
        for o in range(s.word, spec.first_stack):
            mem.stack[sp_off + o] = claripy.BVS(f"x{tag}{v}_c{k}_{o}", 8, explicit_name=True)
        for nm, bits in results:
            st.registers.store(nm, claripy.BVS(f"c{tag}{v}_{k}_{nm}", bits, explicit_name=True))
        _normalise_flags(st, st.regs.cc_dep1)
        st.registers.store(s.sp, sp + s.word + spec.pop)

    # ------------------------------------------------------------ one visit
    def _visit(self, key, ab: _Abs):
        visit = self.visits
        (ra, sa, wa), (rb, sb, wb) = self._entry_values(ab, visit)
        try:
            ex_a = self._run_segment(self.A, key[0], ra, sa, wa, ab.escaped[0], visit)
            ex_b = self._run_segment(self.B, key[1], rb, sb, wb, ab.escaped[1], visit)
        except _TooManyPaths:
            # Too many paths: the region is cut where paths join as well (on both
            # sides, so that the cuts can line up).
            ex_a = self._run_segment(self.A, key[0], ra, sa, wa, ab.escaped[0], visit,
                                     self.A.heads | self.A.joins)
            ex_b = self._run_segment(self.B, key[1], rb, sb, wb, ab.escaped[1], visit,
                                     self.B.heads | self.B.joins)
        if not ex_a or not ex_b:
            raise _Differs(key[0], key[1], "one side has no way out of this segment")
        if not self.no_alias:
            return self._match_visit(visit, key, ex_a, ex_b, flat=True)
        saved = (dict(self.frame_pairs), dict(self.callee_pairs), self.no_alias_used)
        try:
            return self._match_visit(visit, key, ex_a, ex_b, flat=False)
        except _Flat:
            # an address outside the region model: this visit with the flat memory
            self.frame_pairs, self.callee_pairs, self.no_alias_used = \
                saved[0], saved[1], saved[2]
            self.no_alias_flat += 1
            return self._match_visit(visit, key, ex_a, ex_b, flat=True)

    def _match_visit(self, visit, key, ex_a, ex_b, flat: bool) -> list:
        out = []
        for m, ea, eb in self._product(visit, key, ex_a, ex_b, flat):
            try:
                m.check_faults(ea, eb)
                if ea.kind == "jump":
                    m.check_heap(ea, eb, "after this segment")
                    out.append(((ea.target, eb.target), m.exit_abs(ea, eb)))
                elif ea.kind == "ret":
                    m.check_ret(ea, eb)
            except _Differs as e:
                if e.what == "memory outside the frame differs after this segment":
                    self._hint_join_extension(ea, eb)
                m.hint(e)
                raise
            m.commit()
        return out

    def _product(self, visit, key, ex_a, ex_b, flat: bool = True) -> list:
        """The exit pairs (A path, B path) both sides can take together: a pair whose
        guards are provably equivalent excludes every other pairing of its two paths
        (each side's paths are disjoint); the remaining paths are paired whenever their
        guards can hold together -- so branches need not correspond one to one (a
        condition split in two, if-conversion, a duplicated tail). Every pair of
        executions follows some pair of paths, so checking all feasible pairs covers
        them all."""
        cache = {}

        def pair(i, j):
            if (i, j) not in cache:
                m = _Match(self, visit, key, ex_a[i], ex_b[j], flat)
                err = None
                try:
                    m.unify()
                except (_Differs, _Unsup) as e:
                    err = e
                cache[i, j] = (m, err)
            return cache[i, j]
        out, used_b, rest_a = [], set(), []
        for i, ea in enumerate(ex_a):
            for j, eb in enumerate(ex_b):
                if j in used_b or eb.kind != ea.kind:
                    continue
                m, err = pair(i, j)
                if err is None and m.equal(m.z(0, ea.guard), m.z(1, eb.guard)):
                    used_b.add(j)
                    out.append((m, ea, eb))
                    break
            else:
                rest_a.append(i)
        rest_b = [j for j in range(len(ex_b)) if j not in used_b]
        for i in rest_a:
            for j in rest_b:
                ea, eb = ex_a[i], ex_b[j]
                m, err = pair(i, j)
                if not m.feasible(ea, eb):
                    continue
                if err is not None:
                    if isinstance(err, _Differs):
                        ca, cb = [[event for event in log.events if event[0] == "call"] for log in m.log]
                        if len(ca) != len(cb) and all(_canon(a[2]) == _canon(b[2]) for a, b in zip(ca, cb)):
                            tag, side, exit = ("A", self.A, ea) if len(ca) < len(cb) else ("B", self.B, eb)
                            if exit.kind == "jump" and exit.target in side.joins \
                                    and exit.target not in side.heads:
                                self.unmatched_join = (tag, exit.target)
                    if isinstance(err, _Differs):
                        m.hint(err)
                    raise err
                if ea.kind != eb.kind:
                    self._hint_join_extension(ea, eb)
                    e = _Differs(ea.at, eb.at, f"A leaves this segment by '{ea.kind}' at 0x{ea.at:x} "
                                               f"where B leaves by '{eb.kind}' at 0x{eb.at:x}")
                    m.hint(e)
                    raise e
                out.append((m, ea, eb))
        return out


class _Match:
    """Unification and proof obligations for one pair of paths (an A exit and a B
    exit) of a visit of a segment pair. Unifications (B reads and call results that
    are A's), frame-object and callee correspondences are the pair's own until the
    pair is accepted (`commit`)."""

    def __init__(self, ck: Checker, visit: int, key, ea, eb, flat: bool = True):
        self.ck, self.visit, self.key = ck, visit, key
        self.ex = ([ea], [eb])
        self.conv = (ck.conv_a, ck.conv_b.clone())
        self.h0 = z3.Array(f"H{visit}", z3.BitVecSort(ck.A.bits), z3.BitVecSort(8))
        self.log = [ea.state.lsmem, eb.state.lsmem]
        self.frame_pairs = dict(ck.frame_pairs)
        self.callee_pairs = dict(ck.callee_pairs)
        # the flat memory: no --no-alias, or an address outside the region model
        self.flat = flat or not ck.no_alias
        self._b_upto = None         # unifying: B's events before this one are matched

    def commit(self) -> None:
        ck = self.ck
        if ck.no_alias and not self.flat:
            ck.no_alias_used += 1
        for d, mine in ((ck.frame_pairs, self.frame_pairs), (ck.callee_pairs, self.callee_pairs)):
            for k, v in mine.items():
                if d.setdefault(k, v) != v:
                    raise _Unsup("stack objects or callees correspond differently on different paths")
        if len(set(ck.frame_pairs.values())) != len(ck.frame_pairs):
            raise _Unsup("stack objects correspond differently on different paths")

    def hint(self, e: _Differs) -> None:
        h = self.unmatched_reads()
        if h and h not in e.what:
            e.what = f"{e.what} ({h})"

    def guards(self, ea, eb) -> list:
        return [self.z(0, ea.guard), self.z(1, eb.guard)]

    def feasible(self, ea, eb) -> bool:
        """Whether both paths can be taken together (True when undecided)."""
        g = self.guards(ea, eb)
        s = z3.simplify(z3.And(*g))
        if z3.is_false(s):
            return False
        if z3.is_true(s):
            return True
        ck = self.ck
        ck.queries += 1
        sol = z3.Solver()
        sol.set("timeout", ck.timeout_ms)
        sol.add(*self.axioms(), *g)
        return sol.check() != z3.unsat

    # ---- z3 helpers
    def z(self, side: int, e):
        return self.conv[side](e)

    def heap_at(self, side: int, mem, i: int) -> _Heap:
        """Memory on `side` after the first i operations of mem.heap. (In the region
        model an address outside it raises _Flat: the visit is redone flat.)"""
        return self._builder(side, mem).at(i)

    def _regions(self) -> bool:
        return self.ck.no_alias and not self.flat

    def _mems(self, side: int) -> list:
        return list({id(e.state.lsmem): e.state.lsmem for e in self.ex[side]}.values())

    def _builder(self, side: int, mem) -> "_MemBuilder":
        """The memories of a path, built in program order and only as far as asked.

        A's terms never change; B's change when unification substitutes one of its
        symbols, which happens in program order for a symbol defined by the event
        being matched -- so what was built for earlier events stays valid, and B's
        memories are extended, never rebuilt."""
        bs = self.__dict__.setdefault("_builders", {})
        key = (side, id(mem), self.flat)
        b = bs.get(key)
        if b is None:
            if side == 1 and self._regions():
                for m in self._mems(0):     # B classifies with A's aliases
                    self._builder(0, m).at(len(m.heap))
            b = bs[key] = _MemBuilder(self, side, mem)
        return b

    def _apply(self, side: int, h: _Heap, op) -> None:
        if op[0] == "havoc":
            t = self.conv[side].subst.get(op[1]) if side else None
            h.havoc(str(t) if t is not None else op[1])
        else:
            _, addr, data, size = op
            h.store(self.z(side, addr), self.z(side, data), size)

    # ---- --no-alias: memory as regions
    #
    # A region is the object a pointer base addresses (`base + offset`, base a
    # symbolic value) or "G", the globals at fixed addresses (and global arrays,
    # `table + index`). Different bases are different objects -- that is the
    # assumption -- except where one pointer has two symbols, which _note_read finds:
    # the same location read twice, and a stored pointer read back. Any other
    # address form makes the pair use the flat memory (_Flat).

    def _region_state(self):
        st = self.__dict__.get("_rstate")
        if st is None:
            # alias: pointer symbol -> (region it addresses, offset);
            # spans: base -> [base term, lowest, highest constant offset];
            # memo: address term -> (region, index); vmemo: read -> simplified value
            st = self._rstate = {"alias": {}, "spans": {}, "memo": {}, "vmemo": {}}
        return st

    def _resolve(self, name: str) -> str:
        """A region name as it is now: B's symbol unified with A's is A's."""
        for _ in range(8):
            t = self.conv[1].subst.get(name)
            if t is None or not z3.is_const(t):
                break
            name = str(t)
        return name

    def _base_name(self, x):
        """A pointer base: a symbolic value (a register on entry, a read, a call
        result), or a pointer-wide part of a wider read (a 16-byte load)."""
        if x.size() != self.ck.A.bits:
            return None
        if z3.is_const(x) and x.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            return str(x)
        if z3.is_app(x) and x.decl().kind() == z3.Z3_OP_EXTRACT:
            y = x.arg(0)
            if z3.is_const(y) and y.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                return x.sexpr()
        return None

    def _classify(self, a):
        memo = self._region_state()["memo"]
        hit = memo.get(a)
        if hit is None:
            hit = memo[a] = self._classify1(a)
        return hit

    def _classify1(self, a):
        bits = self.ck.A.bits
        st = self._region_state()
        a = z3.simplify(a)
        if z3.is_bv_value(a):
            return "G", a
        parts = a.children() if z3.is_app(a) and a.decl().kind() == z3.Z3_OP_BADD else [a]
        bare = [x for x in parts if self._base_name(x)]
        rest = [x for x in parts if not any(x.eq(b) for b in bare)]
        if len(bare) == 1:
            name = self._base_name(bare[0])
            if all(z3.is_bv_value(x) for x in rest):
                c = _signed(sum(x.as_long() for x in rest), bits)
                sp = st["spans"].setdefault(name, [bare[0], c, c])
                sp[1], sp[2] = min(sp[1], c), max(sp[2], c)
            key, extra = st["alias"].get(name, (name, 0))
            idx = z3.BitVecVal(extra % (1 << bits), bits)
            for x in rest:
                idx = idx + x
            return self._resolve(key), z3.simplify(idx)
        if not bare and any(z3.is_bv_value(x) and self.ck.A.in_image(x.as_long()) for x in rest):
            return "G", a           # a global array: table + index
        raise _Flat()

    def _pointer(self, data):
        """(region, offset) if a value is a pointer base + constant."""
        try:
            key, idx = self._classify(data)
        except _Flat:
            return None
        if key == "G" or not z3.is_bv_value(idx):
            return None
        return key, _signed(idx.as_long(), self.ck.A.bits)

    def _note_read(self, side: int, h: _Heap, ev, seen: dict) -> None:
        """Pointer aliases a read establishes. The value read, simplified (read over
        write: stores at other offsets drop out, a store at this one gives its data):
        a stored pointer read back is that pointer, the same location read twice
        gives the same term -- for the read and each pointer-wide part of a wider one."""
        _, sym, addr, size, i = ev
        bits = self.ck.A.bits
        if 8 * size < bits:
            return                  # too narrow to hold a pointer
        zsym = self.z(side, sym)
        if not (z3.is_const(zsym) and zsym.decl().kind() == z3.Z3_OP_UNINTERPRETED):
            return
        st = self._region_state()
        alias, vmemo = st["alias"], st["vmemo"]
        za = self.z(side, addr)
        key, idx = self._classify(za)
        vkey = (h.array(key), idx, size)
        value = vmemo.get(vkey)
        if value is None:
            value = vmemo[vkey] = z3.simplify(h.read(za, size))
        if 8 * size == bits:
            parts = [(str(zsym), value)]
        elif (8 * size) % bits == 0:
            parts = [(z3.Extract(k + bits - 1, k, zsym).sexpr(),
                      z3.simplify(z3.Extract(k + bits - 1, k, value)))
                     for k in range(0, 8 * size, bits)]
        else:
            return
        for name, v in parts:
            if name in alias:
                continue
            ptr = self._pointer(v)
            if ptr is not None:
                alias[name] = ptr
                continue
            # (keyed by the term itself: an AST id is reused once the term is freed)
            first = seen.setdefault(v, name)
            if first != name:
                tgt, off = alias.get(first, (first, 0))
                alias[name] = (self._resolve(tgt), off)

    # ---- read axioms
    def _side_axioms(self, side: int, upto=None) -> list:
        """`sym == memory[addr]` for the reads of `side` (B: of its first `upto`
        events), extended in program order like the memories."""
        out = []
        cache = self.__dict__.setdefault("_axcache", {})
        for mem in self._mems(side):
            key = (side, id(mem), self.flat)
            ent = cache.get(key)
            if ent is None:
                ent = cache[key] = [[], 0, set()]       # axioms, events done, reads seen
            evs = mem.events
            limit = len(evs) if upto is None or side == 0 else min(upto, len(evs))
            while ent[1] < limit:
                ev = evs[ent[1]]
                ent[1] += 1
                if ev[0] != "read" or ev[1].hash() in ent[2]:
                    continue
                ent[2].add(ev[1].hash())
                _, sym, addr, size, i = ev
                ent[0].append(self.z(side, sym) == self.heap_at(side, mem, i).read(
                    self.z(side, addr), size))
            out += ent[0] if upto is None or side == 0 else ent[0][:]
        return out

    def axioms(self) -> list:
        upto = self._b_upto
        version = (len(self.conv[1].subst), self.flat, upto)
        if getattr(self, "_ax", None) is not None and self._ax[0] == version:
            return self._ax[1]
        out = _alignment(self.ck.A) + _alignment(self.ck.B)
        out += self._side_axioms(0) + self._side_axioms(1, upto)
        if self._regions():
            # (part of the assumption) an object does not wrap around the end of the
            # address space: the bytes accessed through a pointer are in order
            for base, lo, hi in self._region_state()["spans"].values():
                if lo < hi:
                    out.append(z3.ULT(base + lo, base + hi + 1))
        self._ax = (version, out)
        return out

    def prove(self, claim, extra=()) -> bool:
        ck = self.ck
        s = z3.simplify(claim)
        if z3.is_true(s):
            ck.proved += 1
            return True
        ck.queries += 1
        if time.process_time() > ck.deadline:
            raise _Unsup("lockstep time budget exhausted")
        sol = z3.Solver()
        sol.set("timeout", ck.timeout_ms)
        sol.add(*self.axioms(), *extra)
        sol.add(z3.Not(claim))
        r = sol.check()
        if r == z3.unsat:
            ck.proved += 1
            return True
        return False

    def equal(self, za, zb, extra=()) -> bool:
        if za.eq(zb):
            return True
        if za.sort() != zb.sort():
            return False
        return self.prove(za == zb, extra)

    # ---- unification of memory reads and calls
    def unify(self) -> None:
        ck = self.ck
        la, lb = self.log
        reads_a = [ev for ev in la.events if ev[0] == "read"]
        calls_a = [ev for ev in la.events if ev[0] == "call"]
        calls_b = [ev for ev in lb.events if ev[0] == "call"]
        epoch_a = {}
        n = 0
        for ev in la.events:
            if ev[0] == "call":
                n += 1
            else:
                epoch_a[ev[1].hash()] = n
        epoch = 0
        self._progress = (0, 0)      # (B events processed, epoch reached)
        try:
            self._unify(lb, la, reads_a, calls_a, epoch_a)
            # Matching prefix calls establishes relationships between their
            # results. The path-product feasibility check needs those facts even
            # when these two paths eventually make different numbers of calls:
            # e.g. f() == 0 on A paired with f() != 0 on B is impossible.
            if ck.checks.calls and len(calls_a) != len(calls_b):
                what = self._call_count_msg(calls_a, calls_b)
                raise _Differs(self.key[0], self.key[1], what, _unpaired(calls_a, calls_b))
        finally:
            self._b_upto = None

    def _unify(self, lb, la, reads_a, calls_a, epoch_a) -> None:
        epoch = 0
        for n_done, ev in enumerate(lb.events):
            self._progress = (n_done, epoch)
            # Queries about event n_done need B's reads before it only (a later
            # read's axiom just defines its symbol): B's terms then change only
            # beyond what was built.
            self._b_upto = n_done
            if ev[0] == "call":
                if epoch < len(calls_a):
                    self._match_call(calls_a[epoch], ev)
                epoch += 1
                continue
            _, sym, addr, size, i = ev
            zb_addr = z3.simplify(self.z(1, addr))
            cands = [r for r in reads_a if r[3] == size and epoch_a[r[1].hash()] == epoch]
            cands.sort(key=lambda r: not z3.simplify(self.z(0, r[2])).eq(zb_addr))
            for r in cands[:4]:
                za_addr = z3.simplify(self.z(0, r[2]))
                if not za_addr.eq(zb_addr) and not self.prove(za_addr == zb_addr):
                    continue
                ha, hb = self.heap_at(0, la, r[4]), self.heap_at(1, lb, i)
                if ha.same(hb) or self.prove(ha.read(za_addr, size) == hb.read(zb_addr, size)):
                    self.conv[1].set_subst(sym.args[0], self.z(0, r[1]))
                    break
        self._progress = (len(lb.events), epoch + 1)

    def unmatched_reads(self) -> str:
        """Memory reads of this segment without a partner on the other side: usually
        the cause of a later difference (e.g. a global missing from the correspondence)."""
        la, lb = self.log
        done, epoch = getattr(self, "_progress", (0, 0))
        b_reads = [ev for ev in lb.events[:done] if ev[0] == "read"]
        matched_a = {str(t) for t in self.conv[1].subst.values()}
        a_events, n = [], 0
        for ev in la.events:
            if ev[0] == "call":
                n += 1
                if n > epoch:
                    break
            else:
                a_events.append(ev)
        a_left = [ev for ev in a_events if ev[1].args[0] not in matched_a]
        b_left = [ev for ev in b_reads if ev[1].args[0] not in self.conv[1].subst]
        if not a_left and not b_left:
            return ""

        def show(evs):
            out = []
            for ev in evs[:4]:
                addr = ev[2]
                out.append(f"[0x{addr.concrete_value:x}]" if not addr.symbolic else f"[{addr}]"[:60])
            return ", ".join(out) + (" ..." if len(evs) > 4 else "")
        parts = []
        if a_left:
            parts.append(f"A reads {show(a_left)}")
        if b_left:
            parts.append(f"B reads {show(b_left)}")
        return "unmatched memory reads: " + "; ".join(parts)

    def _call_count_msg(self, ca, cb) -> str:
        na = [c[2] for c in ca]
        nb = [c[2] for c in cb]
        for i, (x, y) in enumerate(zip(na, nb)):
            if x != y:
                return f"A calls {x} (at 0x{ca[i][6]:x}) where B calls {y} (at 0x{cb[i][6]:x})"
        if len(na) > len(nb):
            c = ca[len(nb)]
            return f"A calls {c[2]} at 0x{c[6]:x}; B has no matching call in this segment"
        c = cb[len(na)]
        return f"B calls {c[2]} at 0x{c[6]:x}; A has no matching call in this segment"

    def _match_call(self, ca, cb) -> None:
        ck = self.ck
        _, k, name_a, args_a, hi_a, frames_a, at_a, results, pc_a = ca
        _, kb, name_b, args_b, hi_b, frames_b, at_b, _, pc_b = cb
        # Under the conditions of both paths up to the call (not beyond: those may
        # depend on the results, unified only because the calls match).
        g = [self.z(0, pc_a), self.z(1, pc_b)]
        if ck.checks.calls:
            if _canon(name_a) != _canon(name_b) and not self._callee_pair(name_a, name_b):
                la, lb = self.log
                raise _Differs(at_a, at_b, f"A calls {name_a} where B calls {name_b}",
                               _unpaired([e for e in la.events if e[0] == "call"],
                                         [e for e in lb.events if e[0] == "call"]))
            if len(args_a) != len(args_b):
                raise _Differs(at_a, at_b, f"call {name_a}: argument counts differ",
                               (("A", at_a), ("B", at_b)))
            target = args_a[0] if name_a == _INDIRECT else None
            # (--no-alias: a variadic callee with a constant format string reads the
            # arguments the format names, no more)
            used = ck.format_args(args_a) if ck.no_alias and _canon(name_a) in ck.variadic else None
            for j, (x, y) in enumerate(zip(args_a, args_b)):
                if used is not None and j >= used:
                    break
                if ck.no_alias:
                    # (--no-alias: the original passes no junk -- an uninitialised byte, a
                    # register an earlier call clobbered, `call reg`'s own target -- in
                    # an argument its callee reads: such bytes are not compared)
                    if target is not None and j > 0 and _same_ast(x, target):
                        continue
                    keep = ck.kept_bytes(x)
                    if not keep:
                        continue
                    if len(keep) < x.size() // 8:
                        claripy = _claripy()
                        x = claripy.Concat(*[x[8 * i + 7:8 * i] for i in reversed(keep)])
                        y = claripy.Concat(*[y[8 * i + 7:8 * i] for i in reversed(keep)])
                fa, fb = ck.A.sp0_name in x.variables, ck.B.sp0_name in y.variables
                if fa or fb:
                    if not (fa and fb):
                        raise _Differs(at_a, at_b, f"call {name_a}: argument {j + 1} is a stack "
                                                   f"address on one side only")
                    if not self._frame_pair(_sp_offset(x, ck.A.sp0_name, ck.A.bits),
                                            _sp_offset(y, ck.B.sp0_name, ck.B.bits)):
                        raise _Unsup(f"call {name_a}: stack objects passed inconsistently")
                    continue
                if not self.equal(self.z(0, x), self.z(1, y), g):
                    raise _Differs(at_a, at_b, f"call {name_a}: argument {j + 1} differs")
            la, lb = self.log
            ha, hb = self.heap_at(0, la, hi_a), self.heap_at(1, lb, hi_b)
            if ck.checks.memory and not (ha.same(hb) or self.prove(ha.equal(hb), g)):
                raise _Differs(at_a, at_b, f"call {name_a}: the memory it sees differs")
        # Escaped frame objects: the callee sees and changes them. Objects reached
        # through stack addresses stored in other objects correspond by those addresses.
        grown = True
        while grown:
            grown = False
            for base_a, cont_a in frames_a.items():
                cont_b = frames_b.get(self.frame_pairs.get(base_a), ())
                for x, y in zip(cont_a, cont_b):
                    if isinstance(x, tuple) and isinstance(y, tuple) and x[1] not in self.frame_pairs:
                        grown |= self._frame_pair(x[1], y[1])
        for base_a, cont_a in frames_a.items():
            base_b = self.frame_pairs.get(base_a)
            if base_b is None or base_b not in frames_b:
                raise _Unsup(f"call {name_a}: escaped stack objects do not correspond")
            cont_b = frames_b[base_b]
            n = min(len(cont_a), len(cont_b))
            for i in range(n):
                x, y = cont_a[i], cont_b[i]
                if x is None and y is None:
                    continue
                if ck.no_alias and (x is None or (not isinstance(x, tuple) and ck.junk(x))):
                    continue        # (--no-alias: junk in the original's object, see above)
                if isinstance(x, tuple) or isinstance(y, tuple):
                    if isinstance(x, tuple) and isinstance(y, tuple) and x[2] == y[2] \
                            and self._frame_pair(x[1], y[1]):
                        continue
                    raise _Differs(at_a, at_b, f"call {name_a}: stack object at A sp{base_a:+#x} / "
                                               f"B sp{base_b:+#x} differs at byte {i} (stack address)")
                if x is None or y is None or not self.equal(self.z(0, x), self.z(1, y), g):
                    raise _Differs(at_a, at_b, f"call {name_a}: stack object at A sp{base_a:+#x} / "
                                               f"B sp{base_b:+#x} differs at byte {i}")
            for i in range(n):
                self.conv[1].set_subst(f"fB{self.visit}_c{kb}_{i}_{base_b}",
                                       z3.BitVec(f"fA{self.visit}_c{k}_{i}_{base_a}", 8))
        # Unify the call's results.
        v = self.visit
        bits = self.ck.A.bits
        self.conv[1].set_subst(f"hB{v}_c{kb}", z3.Array(f"hA{v}_c{k}", z3.BitVecSort(bits),
                                                        z3.BitVecSort(8)))
        for reg, rb in results:
            self.conv[1].set_subst(f"cB{v}_{kb}_{reg}", z3.BitVec(f"cA{v}_{k}_{reg}", rb))

    def _callee_pair(self, na: str, nb: str) -> bool:
        """Two callees, at least one without a name, called at corresponding places:
        assume they correspond (as named callees do by name) if that is consistent
        with the assumptions made so far -- a B function corresponds to one A function,
        an unnamed A function to one B function (static copies of an A function in B
        may all correspond to it)."""
        if not (_UNNAMED.match(na) or _UNNAMED.match(nb)):
            return False
        cp = self.callee_pairs
        if cp.setdefault(("B", nb), na) != na:
            return False
        if _UNNAMED.match(na) and cp.setdefault(("A", na), nb) != nb:
            return False
        cp.setdefault(("A", na), nb)
        return True

    def _frame_pair(self, oa, ob) -> bool:
        """Record that A's frame object at offset oa corresponds to B's at ob; False if
        that contradicts an earlier correspondence (which must be one-to-one)."""
        fp = self.frame_pairs
        if oa is None or ob is None or fp.setdefault(oa, ob) != ob:
            return False
        return not any(a != oa and b == ob for a, b in fp.items())

    # ---- exits
    def check_faults(self, ea, eb) -> None:
        fa = self._fault(0, ea.state.lsmem)
        fb = self._fault(1, eb.state.lsmem)
        if fa is None and fb is None:
            return
        fa = fa if fa is not None else z3.BoolVal(False)
        fb = fb if fb is not None else z3.BoolVal(False)
        if not self.prove(fa == fb, self.guards(ea, eb)):
            raise _Differs(ea.at, eb.at, "a division may fault on one side only")

    def _fault(self, side, mem):
        conds = []
        for op, n, d in mem.divs:
            zn, zd = self.z(side, n), self.z(side, d)
            m = _DIVMOD.match(op)
            if m:
                signed, wide, narrow = m.group(1) == "S", int(m.group(2)), int(m.group(3))
                dd = (z3.SignExt if signed else z3.ZeroExt)(wide - narrow, zd)
                q = (zn / dd) if signed else z3.UDiv(zn, dd)
                lo = z3.Extract(narrow - 1, 0, q)
                fits = (z3.SignExt if signed else z3.ZeroExt)(wide - narrow, lo) == q
                conds.append(z3.Or(zd == 0, z3.Not(fits)))
            else:
                signed = _DIV.match(op).group(1) == "S"
                c = zd == 0
                if signed:
                    bits = zd.size()
                    c = z3.Or(c, z3.And(zn == z3.BitVecVal(1 << (bits - 1), bits), zd == -1))
                conds.append(c)
        return z3.Or(*conds) if conds else None

    def check_heap(self, ea, eb, when: str) -> None:
        if not self.ck.checks.memory:
            return
        ma, mb = ea.state.lsmem, eb.state.lsmem
        ha, hb = self.heap_at(0, ma, len(ma.heap)), self.heap_at(1, mb, len(mb.heap))
        if not (ha.same(hb) or self.prove(ha.equal(hb), self.guards(ea, eb))):
            raise _Differs(ea.at, eb.at, f"memory outside the frame differs {when}")

    def check_ret(self, ea, eb) -> None:
        ck = self.ck
        g = self.guards(ea, eb)
        self.check_heap(ea, eb, "at the return")
        sa, sb = ea.state, eb.state
        if ck.checks.preserve:
            da = sa.lsmem.classify(sa.registers.load(ck.A.sp))
            db = sb.lsmem.classify(sb.registers.load(ck.B.sp))
            if da != db:
                raise _Differs(ea.at, eb.at, f"stack-pointer delta at return differs ({da} vs {db})")
            for r in ck.saved:
                if not self.equal(self.z(0, self._reg(sa, r)), self.z(1, self._reg(sb, r)), g):
                    raise _Differs(ea.at, eb.at, f"callee-saved {r} differs at the return")
        if ck.checks.ret:
            for r in ck.ret_regs:
                if not self.equal(self.z(0, self._reg(sa, r)), self.z(1, self._reg(sb, r)), g):
                    raise _Differs(ea.at, eb.at, f"return value ({r}) differs")

    @staticmethod
    def _reg(st, r: str):
        name, _, w = r.partition(":")
        v = st.registers.load(name.lower())
        return v[int(w) - 1:0] if w else v

    # ---- successor abstract state
    def exit_abs(self, ea, eb) -> _Abs:
        ck = self.ck
        g = self.guards(ea, eb)
        locs = []       # per side: {loc: z3 term}
        consts, saddr, written = [], [], []
        for side, (e, s) in enumerate(((ea, ck.A), (eb, ck.B))):
            st, mem = e.state, e.state.lsmem
            vals = {}
            for r in s.regs:
                vals[("r", r)] = st.registers.load(r)
            for off, b in mem.stack.items():
                vals[("s", off)] = b
            for off, (v, bs) in mem.words.items():
                if all(mem.stack.get(off + k) is b for k, b in enumerate(bs)):
                    vals[("sw", off, len(bs))] = v
            c, sa, terms = {}, {}, {}
            for loc, v in vals.items():
                if s.sp0_name in v.variables:
                    if loc[0] in ("r", "sw"):
                        d = _sp_offset(v, s.sp0_name, s.bits)
                        if d is not None:
                            sa[loc] = d
                    continue
                t = z3.simplify(self.z(side, v))
                if z3.is_bv_value(t):
                    c[loc] = t.as_long()
                    continue
                terms[loc] = t
            if ("r", s.sp) not in sa:
                raise _Unsup(f"stack pointer is not a frame address after 0x{e.at:x}")
            consts.append(c)
            saddr.append(sa)
            written.append(frozenset(mem.stack))
            locs.append(terms)
        # Facts the branch taken implies (lifted into the successor's entry state):
        # locations it pins to a constant, and equalities that hold only under it.
        models = self._models(list(locs[0].values()) + list(locs[1].values()), z3.And(*g))
        for side in (0, 1):
            for loc, t in list(locs[side].items()):
                v = [m.eval(t, model_completion=True) for m in models]
                if v[0].eq(v[1]) and z3.is_bv_value(v[0]) and self.prove(t == v[0], g):
                    consts[side][loc] = v[0].as_long()
                    del locs[side][loc]
        pairs = self._pairs(locs[0], locs[1], g, models)
        offsets = self._offsets(locs[0], locs[1], g, models, pairs)
        esc = (frozenset(ea.state.lsmem.escaped), frozenset(eb.state.lsmem.escaped))
        return _Abs(tuple(consts), tuple(saddr), frozenset(pairs), tuple(written), esc,
                    frozenset(offsets))

    def _offsets(self, ta: dict, tb: dict, g, models, pairs, limit: int = 48) -> set:
        """(A loc, B loc, d) with A's value == B's value + d, d != 0 (under g): the
        values differ by the same amount in both valuations, then proven. Registers
        and stack words only."""
        def vals(t):
            return [m.eval(t, model_completion=True).as_long() for m in models]
        idx: dict = {}
        vb = {}
        for lb, u in tb.items():
            if lb[0] == "s":
                continue
            v = vb[lb] = vals(u)
            n = u.size()
            delta = (v[1] - v[0]) % (1 << n)
            if delta:
                idx.setdefault((n, delta), []).append(lb)
        paired = {(la, lb) for la, lb in pairs}
        out, tried = set(), 0
        for la, t in ta.items():
            if la[0] == "s":
                continue
            v = vals(t)
            n = t.size()
            for lb in idx.get((n, (v[1] - v[0]) % (1 << n)), ()):
                if (la, lb) in paired:
                    continue
                d = _signed(v[0] - vb[lb][0], n)
                if d == 0 or abs(d) > _MAX_OFFSET:
                    continue
                tried += 1
                if tried > limit:
                    return out
                if self.prove(t == tb[lb] + d, g):
                    out.add((la, lb, d))
        return out

    def _pairs(self, ta: dict, tb: dict, g, models) -> set:
        """(A loc, B loc) whose values are provably equal (under the exit's guard g):
        fingerprint with valuations satisfying g, then prove."""

        def fp(t):
            return (t.size(),) + tuple(m.eval(t, model_completion=True).as_long() for m in models)
        idx: dict = {}
        for loc, t in tb.items():
            idx.setdefault(fp(t), []).append(loc)
        out = set()
        proven: dict = {}
        for la, t in ta.items():
            for lb in idx.get(fp(t), ()):
                u = tb[lb]
                key = (t.get_id(), u.get_id())
                if key not in proven:
                    proven[key] = t.eq(u) or self.prove(t == u, g)
                if proven[key]:
                    out.add((la, lb))
        return out

    def _models(self, terms, guard=None):
        """Two deterministic random valuations of the variables of `terms` (a variable
        has the same value on both sides, so equal values give equal fingerprints),
        satisfying `guard`: variables the guard constrains take the values it forces,
        the others keep their random values."""
        consts, seen, todo = {}, set(), list(terms)
        while todo:
            t = todo.pop()
            if t.get_id() in seen:
                continue
            seen.add(t.get_id())
            if z3.is_const(t) and t.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                consts[str(t)] = t
            elif z3.is_app(t):
                todo.extend(t.children())
        if guard is not None and z3.is_true(z3.simplify(guard)):
            guard = None
        mods = []
        for rnd in range(2):
            want = {}
            for name, t in consts.items():
                if z3.is_bv(t):
                    want[name] = t == _fp_value(name, rnd, t.size())
                elif z3.is_bool(t):
                    want[name] = t == bool(_fp_value(name, rnd, 1))
            sol = z3.Solver()
            sol.set("timeout", self.ck.timeout_ms)
            if guard is not None:
                sol.add(guard)
            model = None
            for _ in range(12):
                lits = {z3.Bool(f"fp!{n}"): c for n, c in want.items()}
                for lit, c in lits.items():
                    sol.add(z3.Implies(lit, c))
                r = sol.check(*lits)
                if r == z3.sat:
                    model = sol.model()
                    break
                if r != z3.unsat:
                    break
                core = {str(x)[3:] for x in sol.unsat_core()}
                if not core:
                    break
                want = {n: c for n, c in want.items() if n not in core}
            if model is None:
                sol = z3.Solver()
                for c in want.values() if guard is None else ():
                    sol.add(c)
                sol.check()
                model = sol.model()
            mods.append(model)
        return mods


def check(sig, side_a, side_b, checks, map_b, timeout_ms: int = 2000,
          budget_s: float = 120.0, make_spec=None, no_alias: bool = False,
          make_specs=None) -> Result:
    if checks.write_order:
        return Result("unknown", "the order of memory writes is not checked by the lockstep proof", {})
    try:
        ck = Checker(sig, side_a, side_b, checks, map_b, timeout_ms, budget_s, make_spec=make_spec,
                     no_alias=no_alias, make_specs=make_specs)
    except _Unsup as e:
        return Result("unknown", str(e), {})
    return ck.run()
