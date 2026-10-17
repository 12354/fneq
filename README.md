# fneq — are these two compiled functions equivalent?

`fneq` compares two functions in x86 or x86-64 machine code (PE32, PE32+, ELF32, ELF64
or raw blobs) and answers with a **proof**, a **concrete counterexample**, or an honest
**"unknown"**. It never claims an equivalence it has not proven.

```
docker build -t fneq .
docker run --rm -v "$PWD:/work" fneq fneq orig.exe:sub_401000 patched.exe:sub_401000 --cc stdcall --ret i32
```

## Verdicts and exit codes

| verdict | meaning | exit |
|---|---|---|
| `EQUIVALENT` | proven for all inputs (Alive2, or angr+Z3) | 0 |
| `EQUIVALENT_BOUNDED(k)` | proven for loops of up to `k` iterations | 0 |
| `NOT_EQUIVALENT` | a concrete input on which the effects differ, **confirmed by concrete replay** in Unicorn | 1 |
| `NO_DIFFERENCE_FOUND` | no solver could decide; differential testing found nothing (coverage reported) | 2 |
| `UNKNOWN` | timeout, unsupported construct, unresolved indirect jump, ... (reason given) | 2 |
| tool error / detected tool bug | e.g. a solver counterexample that does not reproduce, backends that disagree | 3 |

## What "equivalent" means

For every input — argument registers, stack arguments, all other registers, and the
initial contents of all memory — both functions:

1. return the same value in the return registers (`--ret`: `i32` = EAX, `i64` = EDX:EAX on
   x86 / RAX on x64, `f64` = ST0 on x86 / low 64 bits of XMM0 on x64, ...);
2. leave the same final memory state outside their own stack frame (after mapping globals
   between binaries);
3. make the same external calls (after name mapping), with the same arguments, in the
   same order, seeing the same memory; callees are uninterpreted functions;
4. preserve the callee-saved registers, return to the same address, and leave the same
   stack-pointer delta.

A CPU fault (division by zero, access outside the address space) is an outcome of its
own: if only one function faults on an input they differ; if both fault they agree on
that input. Every check can be switched off (`--no-check-ret`, `--no-check-mem`,
`--no-check-calls`, `--no-check-preserve`); `--write-order` additionally requires the same
sequence of memory writes. The report always prints the definition and the disabled
checks.

## Pipeline

1. **Load and bound** (`binary.py`, `bounding.py`): angr's CFGFast finds the extent
   (tail calls, jump tables, GCC `.cold` parts). Unresolved indirect jumps, self-modifying
   code, SEH frames and system instructions give `UNKNOWN`.
2. **Conventions** (`signature.py`): cdecl/stdcall/fastcall/thiscall, System V, Microsoft
   x64 — inferred from `ret N` and argument registers read before being written
   (override with `--cc`, `--cc-a`, `--cc-b`). Callee arity is inferred from the callee's
   code (`callees.py`).
3. **Primary backend — remill + Alive2** (`lifter/fneq_lift.cpp`, `alive_backend.py`): a
   custom remill client lifts each function into a closed, poison-free LLVM function over
   a flat memory model; both sides are normalised with LLVM passes (a textual-identity
   fast path is reported) and `alive-tv` checks refinement in **both** directions.
4. **Secondary backend — angr + Z3** (`angr_backend.py`): symbolic execution with shared
   input/memory variables and uninterpreted calls; used when Alive2 cannot decide.
5. **Concrete confirmation and fallback — Unicorn** (`harness.py`, `fuzz.py`): every
   solver counterexample is replayed on the original machine code; a proof is also
   cross-checked by a short testing campaign (a disagreement is reported as a tool bug).
   When no solver decides, boundary probes (0, ±1, sign boundaries, min/max, every
   immediate in the code ±1, NaN/±inf/denormals, pairs of float specials) plus random
   inputs run under a pinned x87 control word and MXCSR, with an instruction limit; block
   coverage and every access to unmapped memory are reported.

Counterexamples are **shrunk** before they are reported: each input is set to 0, 1, its
low byte/word, or halved, as long as the difference still reproduces concretely (the
original values are kept in the JSON report as `unshrunk_inputs`).

A bounded proof (`EQUIVALENT_BOUNDED(k)`) only covers executions whose loops run at most
`k` times. If the concrete cross-check finds a difference in a longer execution, the
verdict is `NOT_EQUIVALENT` with that (replayed) counterexample; a *full* proof that
testing contradicts is reported as a tool bug instead.

## Batch mode

```
fneq batch pairs.toml --jobs 8 [--json]
```

compares every `[[pair]]` in the file (keys: `name`, `a`, `b`, and optionally `cc`, `cc_a`,
`cc_b`, `ret`, `corr`, `map_by_symbols`, `backend`, `arch`) and prints a summary table;
the exit code is the worst per-pair exit code. See `tests/batch_example.toml`.

## Modelling decisions (flags)

* `--unroll k` — loop bound; proofs that needed it are `EQUIVALENT_BOUNDED(k)`. Loops with a
  constant trip count are fully unrolled by normalisation and need no bound.
* Aliasing — pointer arguments may alias each other and globals (the flat memory model
  makes this exact). `--no-alias` discards counterexamples that rely on aliasing.
* Floating point — SSE is IEEE single/double. x87 is executed exactly (80-bit, precision
  control) by Unicorn; the solvers do not model x87 arithmetic, so such functions are
  decided by testing (`NO_DIFFERENCE_FOUND` at best) unless a difference is found.
* `--x87-cw` (default `0x027F`, 53-bit precision), `--mxcsr` (default `0x1F80`).
* Undefined behaviour — machine code has none. The lowering strips every poison-generating
  flag and UB-implying attribute, makes shifts total, turns faults into an explicit
  outcome and audits the IR; `tests/test_ub_lowering.py` checks exactly this.

## Comparing across binaries

`--map-by-symbols` maps writable globals present in both binaries by name. A
correspondence file (`--corr file.toml` or `.json`) maps globals and callees explicitly
and can override a callee's inferred arity:

```toml
[[globals]]
a = "g_table"        # side-A symbol or address
b = "0x804a010"      # side-B symbol or address
size = 64

[[callees]]
a = "ext_g2"
b = "my_g2"
name = "g2"
stack_args = 2
```

## Self-test corpus

`corpus/corpus.toml` lists function pairs with known answers (equivalent: -O0 vs -O2,
gcc vs clang vs mingw, hand-written variants, bounded loops; different: off-by-one,
signed vs unsigned, swapped call arguments, dropped write, truncated multiply, x87
precision, x87 vs SSE, pointer wraparound exploited by -O2 on 32-bit x86, a difference for
a single input value), expanded over all calling conventions on both architectures.

Two pairs are known limitations and expected to fail until the modelling improves:
`x87_equivalent` (the solvers do not model x87 arithmetic, so the best result is
`NO_DIFFERENCE_FOUND`) and `loop_vectorized` on x86-64 (remill's SSE vector IR exceeds the
solvers).

```
docker run --rm fneq fneq-corpus --jobs 8            # all cases
docker run --rm fneq fneq-corpus --target x64-sysv   # one target
```

A case passes only with the expected verdict class; skipped cases (missing toolchain,
e.g. MSVC) are listed separately and never count as passes. CI
(`.github/workflows/ci.yml`) runs the unit tests and the corpus on every change.

## Pinned toolchain

LLVM 21.1.8, remill v6.0.1, Alive2 v21.0, Z3 4.15.3 (Dockerfile); Python packages in
`requirements.lock`. `fneq --check-tools` verifies the versions. See `docs/SPIKE.md` for
the feasibility study and what had to be worked around.
