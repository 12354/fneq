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
    readonly: bool = False   # constant data matched by content: only pointer values are
                             # translated; each side reads its own image


@dataclasses.dataclass
class Correspondence:
    regions: list[Region] = dataclasses.field(default_factory=list)   # B -> A
    names_a: dict[int, str] = dataclasses.field(default_factory=dict)  # callee addr -> name
    names_b: dict[int, str] = dataclasses.field(default_factory=dict)
    overrides: dict[str, dict] = dataclasses.field(default_factory=dict)
    opaque: set[str] = dataclasses.field(default_factory=set)
    notes: list[str] = dataclasses.field(default_factory=list)
    funcs: dict[int, int] = dataclasses.field(default_factory=dict)    # B entry -> A entry

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
        # a corresponding function's address (a callback, a thread procedure)
        return self.funcs.get(addr, addr)

    def writable_only(self, b: Binary) -> "Correspondence":
        """Keep mappings of writable globals; read-only data is folded per side."""
        # Memory outside the image is ordinary (writable) memory as well.
        self.regions = [r for r in self.regions
                        if (sec := b.section_at(r.b)) is None or sec.writable]
        self.finish()
        return self

    def restrict(self, refs_a: set[int], refs_b: set[int]) -> "Correspondence":
        """Keep the global mappings the two functions reference directly.

        Every mapped region costs a translation of each symbolic address on side B.
        A region neither function names can only be reached through an input
        pointer, which both sides then treat alike as ordinary memory.
        """
        def hit(lo: int, size: int, refs: set[int]) -> bool:
            return any(lo <= r < lo + size for r in refs)
        kept = [r for r in self.regions if hit(r.b, r.size, refs_b) or hit(r.a, r.size, refs_a)]
        if len(kept) != len(self.regions):
            self.notes.append(f"{len(self.regions) - len(kept)} global mappings not referenced "
                              f"by either function were dropped ({len(kept)} kept)")
        self.regions = kept
        self.finish()
        return self

    def add_readonly(self, ba: Binary, refs_a: set[int], bb: Binary, refs_b: set[int]) -> "Correspondence":
        """Map constant data both functions reference (strings, GUIDs) by content.

        The two images keep the same literal at different addresses; a pointer to it
        passed to a callee or returned is the same value once translated.
        """
        def objects(b: Binary, refs: set[int]) -> dict[bytes, list[int]]:
            out: dict[bytes, list[int]] = {}
            for r in refs:
                sec = b.section_at(r)
                if sec is None or sec.writable or sec.executable:
                    continue
                key = _content_key(b, r)
                if key:
                    out.setdefault(key, []).append(r)
            return out
        oa, ob = objects(ba, refs_a), objects(bb, refs_b)
        taken = [(r.b, r.b + r.size) for r in self.regions]
        added = 0
        for key, bs in ob.items():
            as_ = oa.get(key, [])
            if len(bs) != 1 or len(as_) != 1:
                continue
            lo, hi = bs[0], bs[0] + len(key)
            if any(lo < e and s < hi for s, e in taken):
                continue
            self.regions.append(Region(bs[0], as_[0], len(key), readonly=True))
            taken.append((lo, hi))
            added += 1
        if added:
            self.notes.append(f"{added} constant data objects (strings, GUIDs) matched by content")
        self.finish()
        return self

    def regions_a(self) -> list[tuple[int, int, int]]:
        return [(r.a, r.a, r.size) for r in self.regions]

    def regions_b(self, writable_only: bool = False) -> list[tuple[int, int, int]]:
        return [(r.b, r.a, r.size) for r in self.regions if not (writable_only and r.readonly)]

    def describe(self) -> dict:
        return {"globals": [{"a": hex(r.a), "b": hex(r.b), "size": r.size} for r in self.regions],
                "callees_a": {hex(k): v for k, v in self.names_a.items()},
                "callees_b": {hex(k): v for k, v in self.names_b.items()},
                "overrides": self.overrides,
                "opaque": sorted(self.opaque)}


def _content_key(b: Binary, addr: int) -> bytes | None:
    """Bytes identifying the constant object at addr: a NUL-terminated ASCII or UTF-16
    string, otherwise its first 16 bytes (GUIDs, small tables)."""
    data = b.read(addr, 16)
    if data is None:
        return None
    sec = b.section_at(addr)
    blob = b.read(addr, min(4096, sec.end - addr)) or data
    printable = lambda c: 0x20 <= c < 0x7F or c in (9, 10, 13)  # noqa: E731
    # Test UTF-16 first: the zero high byte of its first ASCII character would
    # otherwise terminate an apparent one-character ASCII string. That both
    # conflates different wide literals and prevents matching several literals
    # with the same first character within one function.
    if len(blob) >= 4 and printable(blob[0]) and blob[1] == 0:
        for i in range(0, len(blob) - 1, 2):
            if blob[i] == 0 and blob[i + 1] == 0:
                if i >= 4 and all(printable(blob[j]) and blob[j + 1] == 0 for j in range(0, i, 2)):
                    return blob[:i + 2]
                break
    n = blob.find(b"\0")
    if n >= 0 and all(printable(c) for c in blob[:n]):
        return blob[:n + 1]
    return data


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
            corr.funcs[xb] = xa
            ov = {k: c[k] for k in ("stack_args", "pop", "reg_args", "stack_arg_sizes", "frame_args") if k in c}
            if ov:
                corr.overrides[name] = ov
        corr.opaque |= set(data.get("opaque", []))
        for name, contract in data.get("callee_contracts", {}).items():
            if set(contract) - {"frame_args", "reg_args", "stack_args", "stack_arg_sizes", "pop"}:
                raise InputError(f"callee contract {name}: unsupported callee model field")
            corr.overrides.setdefault(name, {}).update(contract)
        for name, contract in corr.overrides.items():
            seen = set()
            for item in contract.get("frame_args", []):
                if set(item) - {"arg", "size", "size_arg", "indirect", "scale"} \
                        or "arg" not in item or (("size" in item) == ("size_arg" in item)) \
                        or any(type(v) is not int or v < 0 for v in item.values()) \
                        or item.get("indirect", 0) not in (0, 1, 2, 4, 8) \
                        or item["arg"] in seen:
                    raise InputError(f"callee contract {name}: invalid frame_args entry {item}")
                seen.add(item["arg"])
    corr.finish()
    return corr
