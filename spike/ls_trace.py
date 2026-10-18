"""Trace the lockstep stage on one corpus case: segment exits and fixpoint failures.

usage: python spike/ls_trace.py CASE_ID_SUBSTRING
"""
import sys
import tomllib

from fneq import corpus, lockstep
from fneq.cli import main

orig_seg = lockstep.Checker._run_segment
orig_fix = lockstep.Checker._fixpoint


def seg(self, s, start, *a, **k):
    ex = orig_seg(self, s, start, *a, **k)
    print("SEG", "R" if self.regions else "S", s.tag, hex(start), "heads", sorted(hex(h) for h in s.heads),
          [(e.kind, hex(e.target) if e.target else None, hex(e.at)) for e in ex], file=sys.stderr)
    return ex


def fix(self, t0):
    res, key = orig_fix(self, t0)
    print(f"--- {'regions' if self.regions else 'segments'} refined={self.refined}: "
          f"{res.status}: {res.reason[:200]}", file=sys.stderr)
    if key in self.table:
        print("    entry consts", self.table[key].consts, file=sys.stderr)
    return res, key


lockstep.Checker._run_segment = seg
lockstep.Checker._fixpoint = fix
cases = corpus.expand(tomllib.loads((corpus.CORPUS / "corpus.toml").read_text()), list(corpus.TARGETS))
c = next(c for c in cases if sys.argv[1] in c.id)
arch, conv = corpus.TARGETS[c.target]
bins = [corpus.binary_path(c.target, corpus.BUILDS[b], s) for b, s, _ in (c.a, c.b)]
argv = [f"{bins[0]}:{c.a[2]}", f"{bins[1]}:{c.b[2]}", "--cc", conv, "--map-by-symbols",
        "--backend", "lockstep"] + (["--ret", c.ret] if c.ret else [])
print("CASE", c.id, argv, file=sys.stderr)
main(argv)
