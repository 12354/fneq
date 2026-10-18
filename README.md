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
fneq batch pairs.toml --jobs 8 [--timeout 900] [--out-dir reports] [--jsonl results.jsonl] [--json]
```

compares every `[[pair]]` in the file (or every entry of a JSON list; keys: `name`, `a`, `b`,
and optionally `cc`, `cc_a`, `cc_b`, `ret`, `corr`, `map_by_symbols`, `backend`, `arch`, and
`args`, a list of further command-line flags) and prints a summary table; the exit code is
the worst per-pair exit code. See `tests/batch_example.toml`.

Pairs run in persistent worker processes that keep loaded binaries, angr projects,
bounded functions and the parsed correspondence file across pairs, so large images are
loaded once per worker rather than once per pair. A pair that exceeds `--timeout` is
stopped (its worker is replaced) and reported as `TIMEOUT`. `--out-dir` keeps each pair's
full JSON report; `--jsonl` appends one result line per pair as it finishes.

`--suggest-corr CSV` writes probable callee correspondences for callees the
correspondence file does not name. In each pair the direct calls of both sides are
aligned in address order, with named callees as anchors; an unnamed callee at the same
position as a callee on the other side gets a vote. A suggestion is `high` when both
sides vote consistently for each other (≥ 80 %) at three or more call sites. The list is
for review only — accepted lines go into the correspondence file as `[[callees]]`.
`fneq suggest-corr REPORT_DIR [-o CSV]` does the same for an earlier batch's `--out-dir`.

## Modelling decisions (flags)

* `--unroll k` — loop bound; proofs that needed it are `EQUIVALENT_BOUNDED(k)`. Loops with a
  constant trip count are fully unrolled by normalisation and need no bound.
* Aliasing — pointer arguments may alias each other and globals (the flat memory model
  makes this exact). `--no-alias` discards counterexamples that rely on aliasing.
  It is the fast mode for comparing a reimplementation with its original, and the lockstep
  proof then also assumes (each stated in the report):
  - callees keep no stack address after they return (not true of asynchronous I/O on a
    stack buffer);
  - the original (side A) passes no junk in an argument its callee reads. Bytes it leaves
    uninitialised or that an earlier call clobbered are not compared, and neither is the
    target register of `call reg`. A variadic callee with a constant format string reads
    only the arguments the format names.
* Callees — a callee called on one side whose code the other side has inlined is
  inlined by the lockstep proof as well (exact; its loop heads become cut points; listed
  as `inlined_callees`). Windows API imports get their parameters from the SDK headers
  (`fneq/data/win_arity.json`, generated by `spike/gen_win_arity.py`).
  Register and stack arguments are compared at their consumed widths; when the two
  copies read different widths, lockstep uses the wider requirement. A 32-bit stack
  parameter on x64 does not consume the unused high half of its eight-byte slot.
  The verified MSVC x64 `__chkstk` implementation has a byte-checked contract:
  allocation size in RAX, preserved general/vector registers, no caller home area;
  its memory effects remain uninterpreted and allocation-size differences are checked.
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

Lockstep stack objects can have explicit callee pointee extents in the correspondence
file. These are supplied contracts, never inferred from a function name. For example:

```json
{"callee_contracts": {"copy": {"frame_args": [
  {"arg": 0, "size_arg": 2}, {"arg": 1, "size_arg": 2}
]}}}
```

Indices are zero based in the callee's modelled argument list. `size` declares a fixed
byte count; `size_arg` selects a length argument, optionally multiplied by `scale`.
`indirect: 4` reads a UINT32 length through a stack pointer argument. Both reads and
writes remain conservative within each declared range. Other pointers retain the
default object model. Symbolic lengths or ranges crossing saved slots stay UNKNOWN.
Reports record the contracts and assumptions. Use them only after verifying the
callee's accessible ranges; an incorrect contract can invalidate a proof.
Contracts may also supply `reg_args`, `stack_args`, `stack_arg_sizes` and `pop`.
Supplying all of `reg_args`, `stack_args` and `pop` replaces code-based arity
inference with that complete declared signature, useful for a verified library
routine whose internal dispatch cannot be bounded. The report records this contract.

Variable stack loads and stores are supported when Z3 proves an exhaustive set of
at most 256 offsets under the block's entry path condition. Each access is modeled
with conditional byte updates/reads, retaining unselected bytes. Every possible
access must stay in the currently allocated frame below the root return slot.
Unbounded indices, incomplete enumeration, solver uncertainty and accesses that
can leave that frame remain UNKNOWN. This adds no input bounds or alignment assumptions.

Tail argument inference also follows conditional wrappers. Registers forwarded to a
tail target remain inputs unless the wrapper definitely defines the whole register
on every path reaching that jump; partial writes keep the incoming high bits live.
Forwarded stack slots retain their consumed widths. Unknown stack restoration,
unresolved targets and recursive/deep forwarding chains stay UNKNOWN.
Restricted function bounds include conditional jumps and fallthroughs to external
tail targets, including import thunks. Each taken edge selects its own call model;
call-site ordering does not select which target is observed.

When path-growth limits require cutting segments at control-flow joins, the checker
may extend either side through up to eight non-loop joins to align calls, effects
and exits. All proof obligations run again after each extension, and loop heads stay
cut points. Reports list `extended_join_cuts`; real call, argument and memory
differences remain failures.

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
e.g. MSVC) are listed separately and never count as passes. Run the unit tests with
`docker run --rm -w /opt/fneq fneq python3 -m pytest -q tests`.

## Pinned toolchain

LLVM 21.1.8, remill v6.0.1, Alive2 v21.0, Z3 4.15.3 (Dockerfile); Python packages in
`requirements.lock`. `fneq --check-tools` verifies the versions. See `docs/SPIKE.md` for
the feasibility study and what had to be worked around.
