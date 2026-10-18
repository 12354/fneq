"""Run every corpus case through the lockstep stage only and tabulate.

expect=different + EQUIVALENT would be a soundness bug; everything else is either a
proof (expect=equivalent) or an honest UNKNOWN.
"""
import collections
import concurrent.futures as cf
import os
import sys
import tomllib

from fneq import corpus

cases = corpus.expand(tomllib.loads((corpus.CORPUS / "corpus.toml").read_text()), list(corpus.TARGETS))
flt = sys.argv[1:]
if flt:
    cases = [c for c in cases if any(f in c.id for f in flt)]
cmd = [sys.executable, "-m", "fneq", "--backend", "lockstep", "--lockstep-time", "60"]
cmd += os.environ.get("LS_ARGS", "").split()      # e.g. LS_ARGS=--no-alias
todo = []
for c in cases:
    if all(corpus.binary_path(c.target, corpus.BUILDS[b], s).exists() for b, s, _ in (c.a, c.b)):
        todo.append(c)
with cf.ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda c: corpus.run_case(c, cmd, 200), todo))
stats = collections.Counter()
for c in todo:
    v = (c.verdict or "ERROR").split("(")[0]
    stats[(c.expect, v)] += 1
    bug = c.expect == "different" and v == "EQUIVALENT"
    if bug or v not in ("EQUIVALENT", "UNKNOWN") or c.status == "ERROR" or (c.expect == "equivalent" and v != "EQUIVALENT"):
        print(("SOUNDNESS BUG " if bug else "") + f"{c.id}: {v} {c.detail[:200]}")
print(f"\n{len(todo)} of {len(cases)} cases have binaries")
for k, n in sorted(stats.items()):
    print(f"  expect={k[0]:11} verdict={k[1]:20} {n}")
reasons = collections.Counter((c.detail or "").split(";")[0][:90] for c in todo
                              if (c.verdict or "").startswith("UNKNOWN"))
print("\nUNKNOWN reasons:")
for r, n in reasons.most_common(25):
    print(f"  {n:4}  {r}")
