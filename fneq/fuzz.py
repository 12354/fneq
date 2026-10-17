"""Differential testing: deterministic boundary probes plus random inputs, with coverage."""
from __future__ import annotations

import dataclasses
import random
import struct

from fneq.binary import Binary
from fneq.bounding import FunctionInfo
from fneq.compare import Checks, Difference, compare
from fneq.harness import Environment, RunResult, run
from fneq.model import CalleeSpec
from fneq.signature import Signature


@dataclasses.dataclass
class Side:
    binary: Binary
    info: FunctionInfo
    callees: dict[int, CalleeSpec]
    regions: list[tuple[int, int, int]]


@dataclasses.dataclass
class FuzzResult:
    runs: int
    counterexample: dict | None           # {"inputs": {...}, "memory_seed": n}
    diffs: list[Difference]
    result_a: RunResult | None
    result_b: RunResult | None
    coverage_a: set[int]
    coverage_b: set[int]
    unmapped: list[dict]
    exhausted_runs: int


MIN_RANDOM_RUNS = 300


def _float_patterns() -> list[int]:
    d = lambda x: struct.unpack("<Q", struct.pack("<d", x))[0]  # noqa: E731
    f = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]  # noqa: E731
    vals = [d(0.0), d(-0.0), d(1.0), d(-1.0), d(0.5), d(2.0), d(1e308), d(-1e308),
            d(float("inf")), d(float("-inf")), 0x7FF8000000000000, 0x7FF0000000000001,
            0xFFF8000000000000, 0x0000000000000001, 0x000FFFFFFFFFFFFF, 0x0010000000000000,
            0x7FEFFFFFFFFFFFFF, d(3.0000000000000004), d(1.0000000000000002)]
    vals += [f(1.0), f(-1.0), f(float("inf")), 0x7FC00000, 0x00000001, 0x007FFFFF, 0x7F7FFFFF]
    return vals


def interesting_values(bits: int, constants: set[int]) -> list[int]:
    mask = (1 << bits) - 1
    vals = {0, 1, 2, 3, mask, mask - 1, 1 << (bits - 1), (1 << (bits - 1)) - 1,
            (1 << (bits - 1)) + 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF, 0x10000,
            0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100000000}
    for c in constants:
        for v in (c - 1, c, c + 1):
            vals.add(v & mask)
            vals.add(v & 0xFFFFFFFF)                 # also as a 32-bit value
            if v < 0:
                vals.add((v & 0xFFFFFFFF) | (mask & ~0xFFFFFFFF))   # sign-extended
    return sorted(v & mask for v in vals)


def fuzz(sig: Signature, a: Side, b: Side, checks: Checks, env: Environment,
         iterations: int = 2000, seed: int = 1, map_b=lambda x: x,
         time_budget: float = 60.0, float_pair_probes: bool = True,
         pointer_probes: bool = True, slot_probes: bool = True,
         uses_fp: bool = True) -> FuzzResult:
    rng = random.Random(seed)
    word_bits = 8 * sig.word
    # Read-only and executable image memory is outside the model (which treats all
    # memory that is not referenced constant data as arbitrary input), so argument
    # values must not point there.
    image = [(s.addr - 0x100, s.end + 0x100) for side in (a, b) for s in side.binary.sections
             if not s.writable]

    def outside_image(v: int) -> int:
        while any(lo <= v < hi for lo, hi in image):
            v = (v ^ (1 << (word_bits - 2))) if word_bits > 32 else (v + 0x10000000) & 0xFFFFFFFF
        return v

    constants = a.info.constants | b.info.constants
    ints = interesting_values(word_bits, constants)
    floats = _float_patterns()
    arg_slots = [s for s in sig.inputs
                 if s.kind == "stack" or s.name in sig.convention.int_args
                 or (uses_fp and s.name in sig.convention.float_args)]
    other_slots = [s for s in sig.inputs if s not in arg_slots]

    def random_value(slot) -> int:
        bits = 8 * slot.size
        r = rng.random()
        if slot.name.startswith("XMM"):
            return rng.choice(floats) if r < 0.6 else rng.getrandbits(bits)
        if r < 0.35:
            return rng.choice(ints)
        if r < 0.42 and slot.kind == "stack":
            return rng.choice(floats) & ((1 << bits) - 1)
        if r < 0.75:  # small values: loop counts, indices, flags
            return rng.randint(-16, 64) & ((1 << bits) - 1)
        return rng.getrandbits(bits)

    def base_inputs() -> dict[str, int]:
        vals = {s.name: rng.getrandbits(8 * s.size) for s in other_slots}
        vals.update({s.name: random_value(s) for s in arg_slots})
        return {k: outside_image(v) if not k.startswith("XMM") else v for k, v in vals.items()}

    # Deterministic probes: each argument slot over the interesting values (and the
    # float patterns), the others fixed; then random inputs.
    probes: list[dict[str, int]] = []
    fixed = base_inputs()
    for slot in (arg_slots if slot_probes else []):
        values = floats if slot.name.startswith("XMM") else ints + floats[:12]
        for v in values:
            p = dict(fixed)
            v &= (1 << (8 * slot.size)) - 1
            p[slot.name] = v if slot.name.startswith("XMM") else outside_image(v)
            probes.append(p)
    zero = {s.name: 0 for s in sig.inputs}
    probes.insert(0, zero)

    # Floating-point views of the arguments: on x86 a double spans two adjacent
    # 4-byte stack slots. Probe pairs of special values on the first arguments, so
    # that differences needing two special operands (e.g. overflow then cancel)
    # are found deterministically.
    views: list[list] = []
    if sig.arch == "x86":
        stack = [s for s in arg_slots if s.kind == "stack"]
        views = [stack[i:i + 2] for i in range(0, len(stack) - 1, 2)][:3]
    else:
        views = [[s] for s in arg_slots if s.name.startswith("XMM")][:3]
    fp_specials = floats[:19]
    if not float_pair_probes:
        views = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            for u in fp_specials:
                for v in fp_specials:
                    p = dict(fixed)
                    for view, val in ((views[i], u), (views[j], v)):
                        for k, slot in enumerate(view):
                            p[slot.name] = (val >> (8 * slot.size * k)) & ((1 << (8 * slot.size)) - 1)
                    probes.append(p)

    # Pointer-and-length probes: one argument just below the end of the user address
    # space, another a small count. Differences that only appear when a buffer
    # crosses the end of the address space (wrap-around vs. fault, e.g. a vectorised
    # loop against a scalar one) need both at once.
    top = (1 << 32) if sig.arch == "x86" else (1 << 47)
    near_top = [top - d for d in (1, 2, 3, 4, 7, 8, 12, 15, 16, 31, 32, 64)]
    counts = [1, 2, 7, 8, 9, 16, 17, 33, 64, 100]
    int_slots = [s for s in arg_slots if not s.name.startswith("XMM")][:3] if pointer_probes else []
    for si in int_slots:
        for sj in int_slots:
            if si is sj:
                continue
            for pv in near_top:
                for cv in counts:
                    p = dict(fixed)
                    p[si.name] = pv & ((1 << (8 * si.size)) - 1)
                    p[sj.name] = cv
                    probes.append(p)

    cov_a: set[int] = set()
    cov_b: set[int] = set()
    unmapped: list[dict] = []
    exhausted = 0
    total = len(probes) + iterations
    import time
    deadline = time.monotonic() + time_budget
    for i in range(total):
        # The budget only cuts the campaign after a minimum number of random runs,
        # so that results do not depend on machine load.
        if i >= len(probes) + min(iterations, MIN_RANDOM_RUNS) and time.monotonic() > deadline:
            total = i  # campaign budget used up; report the runs actually done
            break
        inputs = probes[i] if i < len(probes) else base_inputs()
        mem_seed = 0 if i < len(probes) else rng.getrandbits(32)
        env_a = dataclasses.replace(env, memory_seed=mem_seed, regions=a.regions)
        env_b = dataclasses.replace(env, memory_seed=mem_seed, regions=b.regions)
        ra = run(a.binary, a.info.entry, sig, inputs, a.callees, env_a, a.info.code)
        rb = run(b.binary, b.info.entry, sig, inputs, b.callees, env_b, b.info.code)
        cov_a |= ra.coverage
        cov_b |= rb.coverage
        if len(unmapped) < 50:
            unmapped += [dict(u, side="A") for u in ra.unmapped[:5]]
            unmapped += [dict(u, side="B") for u in rb.unmapped[:5]]
        if ra.exhausted or rb.exhausted:
            exhausted += 1
            continue
        diffs = compare(sig, ra, rb, checks, map_b)
        if diffs:
            return FuzzResult(i + 1, {"inputs": inputs, "memory_seed": mem_seed}, diffs,
                              ra, rb, cov_a, cov_b, unmapped, exhausted)
    return FuzzResult(total, None, [], None, None, cov_a, cov_b, unmapped, exhausted)


def coverage_report(info: FunctionInfo, covered: set[int]) -> dict:
    # Unicorn's translation blocks need not start where angr's blocks start: a block
    # counts as covered when any executed block start lies inside it.
    starts = sorted(covered)
    import bisect
    hit, missed = [], []
    for start, data in sorted(info.code.items()):
        i = bisect.bisect_left(starts, start)
        (hit if i < len(starts) and starts[i] < start + len(data) else missed).append(start)
    return {"blocks": len(info.code), "covered": len(hit),
            "percent": round(100.0 * len(hit) / max(len(info.code), 1), 1),
            "unreached": [hex(x) for x in missed]}
