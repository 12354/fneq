"""Secondary backend: angr symbolic execution of both functions, compared with Z3.

Model (the same one the other backends use):
  * inputs are symbolic variables shared by name between the two sides;
  * memory is a deterministic set of byte variables keyed by (epoch, side-A address),
    so both sides start from the same arbitrary memory; every external call starts a
    new epoch, i.e. memory the function touched becomes unknown after the call;
  * the i-th external call returns shared result variables; its arguments and the
    memory written so far are obligations that must be equal on both sides
    (uninterpreted-function reasoning);
  * loops are bounded by the unroll bound k: paths that exceed it are dropped and the
    result is reported as bounded.

Limitations, reported as UNKNOWN rather than approximated: memory accesses through
symbolic addresses (angr would concretise them), integer division (VEX has no fault
semantics for it), and x87 code (VEX models x87 as 64-bit doubles).
"""
from __future__ import annotations

import logging

from fneq.alive_backend import SolverResult
from fneq.compare import Checks
from fneq.model import RET_ADDR, STACK_TOP
from fneq.signature import RET_KINDS, Signature

for _n in ("angr", "cle", "pyvex", "claripy"):
    logging.getLogger(_n).setLevel(logging.ERROR)

MAX_FINAL_STATES = 256
MAX_STEPS = 4000
MAX_ADDRESS_VALUES = 64


class _Unsupported(Exception):
    pass


def _memory_class():
    import angr
    import angr.claripy as claripy

    class FneqMemory(angr.storage.DefaultMemory):
        """Uninitialised bytes are named by (epoch, shared address)."""

        def _default_value(self, addr, size, *args, **kwargs):
            if not isinstance(addr, int):
                if getattr(addr, "concrete", False):
                    addr = addr.concrete_value
                else:
                    raise _Unsupported("read of uninitialised memory at a symbolic address")
            g = self.state.globals
            epoch = g.get("fneq_epoch", 0)
            space = g["fneq_space"]
            parts = [claripy.BVS(f"m_e{epoch}_{space(addr + i):x}", 8, explicit_name=True)
                     for i in range(size)]
            touched = g.get("fneq_touched", frozenset())
            g["fneq_touched"] = touched | {addr + i for i in range(size)}
            return claripy.Concat(*parts) if len(parts) > 1 else parts[0]

    return FneqMemory


def _reg(state, name: str):
    reg, _, bits = name.partition(":")
    val = getattr(state.regs, reg.lower())
    return val[int(bits) - 1:0] if bits else val


def _callee_procedure(spec, word: int, frame: tuple[int, int]):
    import angr
    import angr.claripy as claripy

    class Callee(angr.SimProcedure):
        NO_RET = True

        def run(self):  # noqa: D401
            st = self.state
            g = st.globals
            sp = st.regs.sp
            args = [_reg(st, r) for r in spec.reg_args]
            for off, size in spec.stack_operands(word):
                args.append(st.memory.load(sp + off, size, endness="Iend_LE"))
            idx = len(g.get("fneq_calls", ()))
            written = _written_bytes(st, frame)
            g["fneq_calls"] = tuple(g.get("fneq_calls", ())) + ((spec.name, tuple(args), written),)
            for r in spec.results:
                cur = getattr(st.regs, r.lower())
                setattr(st.regs, r.lower(),
                        claripy.BVS(f"call{idx}_{spec.name}_{r}", cur.size(), explicit_name=True))
            # The callee may change any memory: everything touched so far becomes a
            # fresh variable of the next epoch.
            epoch = g.get("fneq_epoch", 0) + 1
            g["fneq_epoch"] = epoch
            space = g["fneq_space"]
            lo, hi = frame
            for a in sorted(g.get("fneq_touched", frozenset()) | set(g.get("fneq_written", ()))):
                if lo <= a < hi:
                    continue
                st.memory.store(a, claripy.BVS(f"m_e{epoch}_{space(a):x}", 8, explicit_name=True),
                                inspect=False)
            ret_to = st.memory.load(sp, word, endness="Iend_LE")
            st.regs.sp = sp + word + spec.pop
            self.jump(ret_to)

    return Callee


def _written_bytes(state, frame) -> tuple:
    lo, hi = frame
    out = []
    for a in sorted(state.globals.get("fneq_written", ())):
        if not (lo <= a < hi):
            out.append((a, state.memory.load(a, 1, inspect=False)))
    return tuple(out)


def _explore(side, sig: Signature, inputs: dict, unroll: int, has_loop: bool, timeout_ms: int):
    import angr
    import angr.claripy as claripy
    from fneq.bounding import angr_project

    proj = angr_project(side.binary)
    arch = sig.arch
    word = sig.word
    sp0 = STACK_TOP[arch]
    ra = RET_ADDR[arch]
    frame = (sp0 - sig.frame_below, sp0 + sig.frame_above)
    regions = side.regions

    def space(addr: int) -> int:
        for here, a, size in regions:
            if here <= addr < here + size:
                return a + (addr - here)
        return addr

    hooked = []
    try:
        proj.hook(ra, angr.SIM_PROCEDURES["stubs"]["PathTerminator"](), replace=True)
        hooked.append(ra)
        for addr, spec in side.callees.items():
            proj.hook(addr, _callee_procedure(spec, word, frame)(), replace=True)
            hooked.append(addr)

        # angr tracks a stack region; place it around our stack pointer.
        mem = _memory_class()(memory_id="mem", stack_end=sp0 + 0x10000, stack_size=0x100000,
                              cle_memory_backer=proj.loader.memory)
        opts = {angr.options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS}
        st = proj.factory.blank_state(addr=side.info.entry, plugins={"memory": mem},
                                      stack_end=sp0 + 0x10000, stack_size=0x100000,
                                      add_options=opts,
                                      remove_options={angr.options.SIMPLIFY_MEMORY_WRITES})
        st.globals["fneq_space"] = space
        st.globals["fneq_epoch"] = 0
        st.globals["fneq_written"] = frozenset()
        st.globals["fneq_calls"] = ()
        # A symbolic address is allowed only if all its possible values can be
        # enumerated (then every candidate is read/written: exact). Anything wider
        # fails, which makes the result UNKNOWN instead of concretising unsoundly.
        cs = angr.concretization_strategies
        exact = [cs.SimConcretizationStrategySingle(),
                 cs.SimConcretizationStrategyRange(MAX_ADDRESS_VALUES)]
        st.memory.read_strategies = list(exact)
        st.memory.write_strategies = list(exact)

        # Writable data is arbitrary input (as in the other backends), not the
        # image's initial contents.
        for sec in side.binary.sections:
            if sec.writable:
                if sec.vsize > 0x10000:
                    raise _Unsupported(f"writable section {sec.name} too large to model symbolically")
                for a in range(sec.addr, sec.end):
                    st.memory.store(a, claripy.BVS(f"m_e0_{space(a):x}", 8, explicit_name=True),
                                    inspect=False)
        for slot in sig.inputs:
            if slot.kind == "reg":
                setattr(st.regs, slot.name.lower(), inputs[slot.name])
        st.regs.sp = sp0
        st.memory.store(sp0, claripy.BVV(ra, 8 * word), endness="Iend_LE", inspect=False)
        for slot in sig.inputs:
            if slot.kind == "stack":
                st.memory.store(sp0 + slot.offset, inputs[slot.name], endness="Iend_LE", inspect=False)

        def on_write(state):
            addr = state.inspect.mem_write_address
            length = state.inspect.mem_write_length
            if addr is None:
                raise _Unsupported("memory write without an address")
            n = length if isinstance(length, int) else state.solver.eval_one(length)
            if addr.concrete:
                bases = [addr.concrete_value]
            else:
                bases = state.solver.eval_upto(addr, MAX_ADDRESS_VALUES + 1)
                if len(bases) > MAX_ADDRESS_VALUES:
                    raise _Unsupported("memory write through an unbounded symbolic address")
            new = {b + i for b in bases for i in range(n)}
            state.globals["fneq_written"] = state.globals["fneq_written"] | new

        st.inspect.b("mem_write", when=angr.BP_AFTER, action=on_write)

        simgr = proj.factory.simulation_manager(st)
        counts_key = "fneq_visits"
        steps = 0
        dropped = 0
        while simgr.active:
            steps += 1
            if steps > MAX_STEPS:
                raise _Unsupported("symbolic execution did not finish within the step limit")
            simgr.step()
            keep = []
            for s in simgr.active:
                visits = dict(s.globals.get(counts_key, {}))
                pc = s.addr
                visits[pc] = visits.get(pc, 0) + 1
                s.globals[counts_key] = visits
                if visits[pc] > unroll + 1:
                    dropped += 1
                    continue
                keep.append(s)
            simgr.stashes["active"] = keep
            if simgr.errored:
                err = simgr.errored[0].error
                raise _Unsupported(f"angr: {type(err).__name__}: {err}")
            if len(simgr.deadended) > MAX_FINAL_STATES:
                raise _Unsupported("too many paths")
        if simgr.stashes.get("unconstrained"):
            raise _Unsupported("unconstrained successor (indirect control flow)")
        finals = list(simgr.deadended)
        if not finals:
            raise _Unsupported("no path returned to the caller")
        return finals, dropped, frame
    finally:
        for a in hooked:
            proj.unhook(a)


def _path_outputs(state, sig: Signature, frame):
    out = {}
    for name in sig.outputs:
        if name.startswith("ST"):
            raise _Unsupported("x87 return value")
        out[name] = _reg(state, name)
    out["sp_delta"] = state.regs.sp - STACK_TOP[sig.arch]
    out["return_address"] = state.regs.ip
    return out


def check(sig: Signature, side_a, side_b, checks: Checks, env, timeout_ms: int, unroll: int,
          map_b) -> SolverResult:
    import angr.claripy as claripy

    for side, tag in ((side_a, "A"), (side_b, "B")):
        if side.info.uses_x87:
            return SolverResult("unknown", f"angr models x87 as 64-bit doubles ({tag} uses x87)")
    if _has_division(side_a) or _has_division(side_b):
        return SolverResult("unknown", "integer division (no fault semantics in VEX)")
    if any(c.indirect for side in (side_a, side_b) for c in side.callees.values()):
        return SolverResult("unknown", "indirect calls are not modelled by the angr backend")

    inputs = {s.name: claripy.BVS(f"in_{s.name}", 8 * s.size, explicit_name=True) for s in sig.inputs}
    has_loop = side_a.info.has_loop or side_b.info.has_loop
    try:
        pa, drop_a, frame_a = _explore(side_a, sig, inputs, unroll, has_loop, timeout_ms)
        pb, drop_b, frame_b = _explore(side_b, sig, inputs, unroll, has_loop, timeout_ms)
    except _Unsupported as e:
        return SolverResult("unknown", f"angr: {e}")
    except Exception as e:  # angr internals: never let them pass as a result
        return SolverResult("unknown", f"angr failed: {type(e).__name__}: {e}")

    ret_regs = set(RET_KINDS[sig.ret_kind][sig.arch])
    for sa in pa:
        for sb in pb:
            terms = []
            try:
                oa, ob = _path_outputs(sa, sig, frame_a), _path_outputs(sb, sig, frame_b)
            except _Unsupported as e:
                return SolverResult("unknown", f"angr: {e}")
            for name in sig.outputs:
                if (name in ret_regs and checks.ret) or (name not in ret_regs and checks.preserve):
                    terms.append(oa[name] != ob[name])
            if checks.preserve:
                terms += [oa["sp_delta"] != ob["sp_delta"], oa["return_address"] != ob["return_address"]]
            ca, cb = sa.globals.get("fneq_calls", ()), sb.globals.get("fneq_calls", ())
            if checks.calls:
                if [c[0] for c in ca] != [c[0] for c in cb] or any(len(x[1]) != len(y[1]) for x, y in zip(ca, cb)):
                    terms.append(claripy.true())
                else:
                    for x, y in zip(ca, cb):
                        terms += [u != v for u, v in zip(x[1], y[1])]
                        terms += _mem_terms(dict(x[2]), dict(y[2]), sa, sb, map_b, frame_a, frame_b)
            if checks.memory:
                wa = {a: sa.memory.load(a, 1, inspect=False) for a in sa.globals["fneq_written"]
                      if not (frame_a[0] <= a < frame_a[1])}
                wb = {a: sb.memory.load(a, 1, inspect=False) for a in sb.globals["fneq_written"]
                      if not (frame_b[0] <= a < frame_b[1])}
                terms += _mem_terms(wa, wb, sa, sb, map_b, frame_a, frame_b)
            if not terms:
                continue
            solver = claripy.Solver(timeout=timeout_ms)
            solver.add(sa.solver.constraints)
            solver.add(sb.solver.constraints)
            solver.add(claripy.Or(*terms))
            try:
                sat = solver.satisfiable()
            except Exception as e:
                return SolverResult("unknown", f"angr/Z3: {type(e).__name__}: {e}")
            if sat:
                return SolverResult("different", "angr/Z3 found inputs on which the effects differ",
                                    counterexample=_model(solver, inputs, sig))
    if drop_a or drop_b:
        return SolverResult("bounded", f"angr/Z3 proved all paths with loops bounded by {unroll} iterations",
                            bound=unroll)
    return SolverResult("equivalent", f"angr/Z3 proved all {len(pa)}x{len(pb)} path pairs equal")


def _mem_terms(wa: dict, wb: dict, sa, sb, map_b, frame_a, frame_b) -> list:
    terms = []
    inv = {map_b(a): a for a in wb}
    for addr in sorted(set(wa) | set(inv)):
        va = wa.get(addr)
        if va is None:
            va = sa.memory.load(addr, 1, inspect=False)
        b_addr = inv.get(addr, addr)
        vb = wb.get(b_addr)
        if vb is None:
            vb = sb.memory.load(b_addr, 1, inspect=False)
        terms.append(va != vb)
    return terms


def _model(solver, inputs: dict, sig: Signature) -> dict:
    vals = {}
    for name, bv in inputs.items():
        vals[name.lower()] = solver.eval(bv, 1)[0]
    memory = {}
    for var in solver.variables:
        if var.startswith("m_e0_"):
            import angr.claripy as claripy
            bv = claripy.BVS(var, 8, explicit_name=True)
            memory[int(var[5:], 16)] = solver.eval(bv, 1)[0]
    return {"inputs": vals, "memory": memory}


def _has_division(side) -> bool:
    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86,
                     capstone.CS_MODE_32 if side.binary.arch == "x86" else capstone.CS_MODE_64)
    for addr, data in side.info.code.items():
        for insn in md.disasm(data, addr):
            if insn.mnemonic in ("div", "idiv"):
                return True
    return False
