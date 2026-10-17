"""Correspondence between two binaries: globals, callees, opaque callees.

File format (JSON or TOML):

    [[globals]]            # side-A address/symbol, side-B address/symbol, size in bytes
    a = "g_table"
    b = "0x804a010"
    size = 64

    [[callees]]            # callee on side A corresponds to callee on side B
    a = "ext_g2"
    b = "my_g2"
    name = "g2"            # optional shared name used in reports
    stack_args = 2         # optional overrides of the inferred callee model
    pop = 0

    opaque = ["memcpy"]    # callees always treated as uninterpreted (the default)

With --map-by-symbols, every data symbol present in both binaries is mapped
automatically (size = distance to the next symbol, the smaller of the two sides).
"""
from __future__ import annotations

import bisect
import dataclasses
import json
import tomllib
from pathlib import Path

from fneq.binary import Binary, undecorate
from fneq.errors import InputError


@dataclasses.dataclass
class Region:
    b: int
    a: int
    size: int


@dataclasses.dataclass
class Correspondence:
    regions: list[Region] = dataclasses.field(default_factory=list)   # B -> A
    names_a: dict[int, str] = dataclasses.field(default_factory=dict)  # callee addr -> name
    names_b: dict[int, str] = dataclasses.field(default_factory=dict)
    overrides: dict[str, dict] = dataclasses.field(default_factory=dict)
    opaque: set[str] = dataclasses.field(default_factory=set)
    notes: list[str] = dataclasses.field(default_factory=list)

    def finish(self) -> None:
        self.regions.sort(key=lambda r: r.b)
        self._starts = [r.b for r in self.regions]
        for x, y in zip(self.regions, self.regions[1:]):
            if x.b + x.size > y.b:
                raise InputError(f"overlapping global mappings at side-B 0x{y.b:x}")

    def map_b(self, addr: int) -> int:
        """Translate a side-B address into side-A space (identity if unmapped)."""
        i = bisect.bisect_right(self._starts, addr) - 1
        if i >= 0:
            r = self.regions[i]
            if r.b <= addr < r.b + r.size:
                return r.a + (addr - r.b)
        return addr

    def writable_only(self, b: Binary) -> "Correspondence":
        """Keep mappings of writable globals; read-only data is folded per side."""
        # Memory outside the image is ordinary (writable) memory as well.
        self.regions = [r for r in self.regions
                        if (sec := b.section_at(r.b)) is None or sec.writable]
        self.finish()
        return self

    def regions_a(self) -> list[tuple[int, int, int]]:
        return [(r.a, r.a, r.size) for r in self.regions]

    def regions_b(self) -> list[tuple[int, int, int]]:
        return [(r.b, r.a, r.size) for r in self.regions]

    def describe(self) -> dict:
        return {"globals": [{"a": hex(r.a), "b": hex(r.b), "size": r.size} for r in self.regions],
                "callees_a": {hex(k): v for k, v in self.names_a.items()},
                "callees_b": {hex(k): v for k, v in self.names_b.items()},
                "opaque": sorted(self.opaque)}


def _data_symbols(binary: Binary) -> dict[str, tuple[int, int]]:
    """Undecorated data symbol -> (address, size estimate)."""
    out = {}
    for name, addr in binary.symbols.items():
        if name in binary.func_symbols or binary.is_exec(addr):
            continue
        plain = undecorate(name)
        if plain.startswith("__") or (plain[:1] == "_" and plain[1:2].isupper())                 or plain.startswith((".", "$")):
            continue  # linker-defined / reserved names (section bounds, etc.)
        sec = binary.section_at(addr)
        if sec is None:
            continue
        nxt = binary.next_symbol_after(addr)
        end = min(nxt if nxt is not None else sec.end, sec.end)
        out[undecorate(name)] = (addr, max(end - addr, 1))
    return out


def load(path: str | None, a: Binary, b: Binary, by_symbols: bool) -> Correspondence:
    corr = Correspondence()
    if by_symbols:
        da, db = _data_symbols(a), _data_symbols(b)
        for name in sorted(set(da) & set(db)):
            (xa, sa), (xb, sb) = da[name], db[name]
            if sa != sb:
                corr.notes.append(f"global {name}: sizes differ ({sa} vs {sb}); mapping {min(sa, sb)} bytes")
            corr.regions.append(Region(xb, xa, min(sa, sb)))
        # Automatic mappings: keep one region per side-B address and truncate at the next.
        corr.regions.sort(key=lambda r: (r.b, -r.size))
        dedup: list[Region] = []
        for r in corr.regions:
            if dedup and r.b < dedup[-1].b + dedup[-1].size:
                if r.b == dedup[-1].b:
                    continue
                dedup[-1].size = r.b - dedup[-1].b
            dedup.append(r)
        corr.regions = dedup
    if path:
        text = Path(path).read_text()
        try:
            data = json.loads(text) if path.endswith(".json") else tomllib.loads(text)
        except (json.JSONDecodeError, tomllib.TOMLDecodeError) as e:
            raise InputError(f"correspondence file {path}: {e}") from e
        for g in data.get("globals", []):
            if "size" not in g:
                raise InputError(f"correspondence file: global {g} needs a size")
            xa, xb = a.resolve(str(g["a"])), b.resolve(str(g["b"]))
            corr.regions = [r for r in corr.regions if r.b != xb]
            corr.regions.append(Region(xb, xa, int(g["size"])))
        for c in data.get("callees", []):
            xa, xb = a.resolve(str(c["a"])), b.resolve(str(c["b"]))
            name = c.get("name", undecorate(str(c["a"])))
            corr.names_a[xa] = name
            corr.names_b[xb] = name
            ov = {k: c[k] for k in ("stack_args", "pop", "reg_args") if k in c}
            if ov:
                corr.overrides[name] = ov
        corr.opaque |= set(data.get("opaque", []))
    corr.finish()
    return corr
