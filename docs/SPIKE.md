# Feasibility spike (work-order step 1)

Date: 2026-10-17. Result: **the remill → LLVM → Alive2 path works** for the trivial pair on
both x86 (cdecl) and x86-64 (System V). Reproduce with
`docker run --rm -v "$PWD":/work fneq python3 spike/spike.py`.

| pair | x86 cdecl | x86-64 SysV |
|---|---|---|
| `a+b` vs `b+a` | EQUIVALENT (normalised IR textually identical) | EQUIVALENT (identical) |
| `a+b` vs `a+b+(a==0x1234567)` | NOT_EQUIVALENT, cex `stk_4 = 0x01234567` | NOT_EQUIVALENT, cex `rdi = 0x01234567` |

## Versions (all pinned in the Dockerfile)

| tool | version |
|---|---|
| Ubuntu base | 24.04 |
| LLVM / clang (apt.llvm.org) | 21.1.8 |
| remill | v6.0.1 (built against system LLVM 21) |
| Alive2 | v21.0 (`alive-tv`, built against system LLVM 21 with `BUILD_TV=1`) |
| Z3 (for Alive2) | 4.15.3 |
| Python stack | see `requirements.lock` (angr 10.0.4.post1, unicorn 2.1.4, capstone 5.0.9, z3-solver 5.1.0.0) |

## What did not work out of the box, and what we did instead

* **`remill-lift -signature`** only returns a single register, threads an `undef` memory
  token and runs remill's own O3 pipeline (with vectorizers). We wrote our own remill client,
  `lifter/fneq_lift.cpp`, which builds a closed function over a flat memory model and gives
  us control over every lowering step that matters for soundness.
* **`alive-tv` exit status is 0 even on `ERROR: Timeout`** and on failed proofs. The verdict
  must be parsed from its output; fneq treats anything it does not positively recognise as
  UNKNOWN.
* **Undef inputs make Z3 time out** even on `a+b` vs `b+a` (60 s). Machine registers and
  memory are never undef/poison, so we always run with `--disable-undef-input
  --disable-poison-input`. This is the semantically correct setting, not a workaround.
* **remill semantics carry UB-implying IR**: `noundef` on memory reads, `nuw nsw` flags, an
  `sret` signature for `__remill_undefined_f80`. `fneq-lift` strips all poison-generating
  flags, UB-implying attributes and metadata, makes shifts total, guards divisions (a #DE
  becomes an explicit output) and audits the final IR for `undef`/`poison`/`unreachable`.
* LLVM 21's CMake package needs libcurl headers to be found by downstream projects
  (`fneqlift` Docker stage installs them).
