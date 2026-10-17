"""Load PE32/PE32+/ELF32/ELF64 images and raw code blobs into a flat address space."""
from __future__ import annotations

import bisect
import dataclasses
import re
import struct
from pathlib import Path

from fneq.errors import InputError


@dataclasses.dataclass
class Section:
    name: str
    addr: int
    data: bytes        # initialised bytes (may be shorter than vsize; rest is zero)
    vsize: int
    readable: bool = True
    writable: bool = False
    executable: bool = False

    @property
    def end(self) -> int:
        return self.addr + self.vsize

    def contains(self, addr: int, n: int = 1) -> bool:
        return self.addr <= addr and addr + n <= self.end


@dataclasses.dataclass
class Binary:
    path: str
    fmt: str                     # elf | pe | raw
    arch: str                    # x86 | x64
    os: str                      # linux | windows
    image_base: int
    sections: list[Section]
    symbols: dict[str, int]      # name -> address (functions and data)
    func_symbols: set[str]       # names known to be functions
    imports: dict[int, str]      # import slot (IAT/GOT) address -> "lib!name" or "name"
    pdata: dict[int, int] = dataclasses.field(default_factory=dict)  # x64 PE: func start -> unwind info RVA+base
    entry: int | None = None

    def __post_init__(self) -> None:
        self.sections.sort(key=lambda s: s.addr)
        self._starts = [s.addr for s in self.sections]
        self.by_addr: dict[int, list[str]] = {}
        for name, addr in self.symbols.items():
            self.by_addr.setdefault(addr, []).append(name)
        self._sym_addrs = sorted(self.by_addr)

    # ---------------------------------------------------------------- memory
    def section_at(self, addr: int) -> Section | None:
        i = bisect.bisect_right(self._starts, addr) - 1
        if i >= 0 and self.sections[i].contains(addr):
            return self.sections[i]
        return None

    def read(self, addr: int, n: int) -> bytes | None:
        """Read n bytes, or None if any byte is outside the image."""
        sec = self.section_at(addr)
        if sec is None or not sec.contains(addr, n):
            return None
        off = addr - sec.addr
        chunk = sec.data[off:off + n]
        return chunk + bytes(n - len(chunk))

    def read_int(self, addr: int, size: int, signed: bool = False) -> int | None:
        b = self.read(addr, size)
        return None if b is None else int.from_bytes(b, "little", signed=signed)

    def is_exec(self, addr: int) -> bool:
        sec = self.section_at(addr)
        return bool(sec and sec.executable)

    def is_readonly_data(self, addr: int) -> bool:
        sec = self.section_at(addr)
        return bool(sec and not sec.writable)

    @property
    def ptr_size(self) -> int:
        return 4 if self.arch == "x86" else 8

    # --------------------------------------------------------------- symbols
    def symbol_at(self, addr: int) -> str | None:
        names = self.by_addr.get(addr)
        if not names:
            return None
        funcs = [n for n in names if n in self.func_symbols]
        return min(funcs or names, key=len)

    def next_symbol_after(self, addr: int) -> int | None:
        i = bisect.bisect_right(self._sym_addrs, addr)
        return self._sym_addrs[i] if i < len(self._sym_addrs) else None

    def resolve(self, what: str) -> int:
        """Resolve a symbol name (decorations optional) or a hex/decimal address."""
        if re.fullmatch(r"(0x)?[0-9a-fA-F]+", what) and (what.startswith("0x") or what not in self.symbols):
            try:
                return int(what, 16) if what.startswith("0x") else int(what, 16 if re.search("[a-fA-F]", what) else 10)
            except ValueError:
                pass
        if what in self.symbols:
            return self.symbols[what]
        matches = {addr for name, addr in self.symbols.items() if undecorate(name) == what}
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise InputError(f"symbol {what!r} is ambiguous in {self.path}")
        raise InputError(f"symbol {what!r} not found in {self.path}")


def undecorate(name: str) -> str:
    """Strip PE32 C decorations: _f, _f@8 (stdcall), @f@8 (fastcall)."""
    m = re.fullmatch(r"[_@]?([A-Za-z_$?][\w$?.]*?)(@\d+)?", name)
    return m.group(1) if m else name


# ===================================================================== ELF

def _load_elf(path: str, data: bytes) -> Binary:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.relocation import RelocationSection
    from elftools.elf.sections import SymbolTableSection
    import io

    elf = ELFFile(io.BytesIO(data))
    machine = elf["e_machine"]
    if machine == "EM_386":
        arch = "x86"
    elif machine == "EM_X86_64":
        arch = "x64"
    else:
        raise InputError(f"{path}: unsupported ELF machine {machine}")
    sections, symbols, funcs, imports = [], {}, set(), {}
    for sec in elf.iter_sections():
        flags = sec["sh_flags"]
        if not flags & 0x2:  # SHF_ALLOC
            continue
        nobits = sec["sh_type"] == "SHT_NOBITS"
        sections.append(Section(sec.name, sec["sh_addr"], b"" if nobits else sec.data(),
                                sec["sh_size"], True, bool(flags & 0x1), bool(flags & 0x4)))
    if not sections:  # stripped section headers: fall back to segments
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_LOAD":
                f = seg["p_flags"]
                sections.append(Section("seg", seg["p_vaddr"], seg.data(), seg["p_memsz"],
                                        bool(f & 4), bool(f & 2), bool(f & 1)))
    for sec in elf.iter_sections():
        if isinstance(sec, SymbolTableSection):
            for sym in sec.iter_symbols():
                if sym.name and sym["st_shndx"] != "SHN_UNDEF" and sym["st_info"]["type"] in ("STT_FUNC", "STT_OBJECT", "STT_NOTYPE"):
                    symbols.setdefault(sym.name, sym["st_value"])
                    if sym["st_info"]["type"] == "STT_FUNC":
                        funcs.add(sym.name)
    # Dynamic imports: GOT slots named by their JUMP_SLOT / GLOB_DAT relocation.
    dynsym = elf.get_section_by_name(".dynsym")
    for sec in elf.iter_sections():
        if isinstance(sec, RelocationSection) and dynsym is not None:
            symtab = elf.get_section(sec["sh_link"])
            for rel in sec.iter_relocations():
                rtype = rel["r_info_type"]
                if rtype in (6, 7) and rel["r_info_sym"]:  # GLOB_DAT, JUMP_SLOT (same numbers on i386/x86-64)
                    imports[rel["r_offset"]] = symtab.get_symbol(rel["r_info_sym"]).name
    return Binary(path, "elf", arch, "linux", 0, sections, symbols, funcs, imports,
                  entry=elf["e_entry"])


# ====================================================================== PE

def _coff_symbols(pe, data: bytes, image_base: int) -> tuple[dict[str, int], set[str]]:
    """Parse the COFF symbol table that mingw keeps in PE images."""
    fh = pe.FILE_HEADER
    ptr, count = fh.PointerToSymbolTable, fh.NumberOfSymbols
    symbols, funcs = {}, set()
    if not ptr or not count or ptr + count * 18 > len(data):
        return symbols, funcs
    strtab = ptr + count * 18
    i = 0
    while i < count:
        ent = data[ptr + i * 18: ptr + (i + 1) * 18]
        name_raw, value, secno, typ, sclass, naux = struct.unpack("<8sIhHBB", ent)
        if name_raw[:4] == b"\0\0\0\0":
            off = struct.unpack("<I", name_raw[4:])[0]
            end = data.index(b"\0", strtab + off)
            name = data[strtab + off:end].decode("latin-1")
        else:
            name = name_raw.rstrip(b"\0").decode("latin-1")
        if secno > 0 and sclass in (2, 3, 6) and secno <= len(pe.sections):  # EXTERNAL, STATIC, LABEL
            sec = pe.sections[secno - 1]
            if not name.startswith("."):
                addr = image_base + sec.VirtualAddress + value
                symbols.setdefault(name, addr)
                if typ & 0x20:  # DTYPE_FUNCTION
                    funcs.add(name)
        i += 1 + naux
    return symbols, funcs


def _load_pe(path: str, data: bytes) -> Binary:
    import pefile

    try:
        pe = pefile.PE(data=data, fast_load=False)
    except pefile.PEFormatError as e:
        raise InputError(f"{path}: not a valid PE file: {e}") from e
    machine = pe.FILE_HEADER.Machine
    if machine == 0x14C:
        arch = "x86"
    elif machine == 0x8664:
        arch = "x64"
    else:
        raise InputError(f"{path}: unsupported PE machine 0x{machine:x}")
    base = pe.OPTIONAL_HEADER.ImageBase
    sections = []
    for s in pe.sections:
        ch = s.Characteristics
        name = s.Name.rstrip(b"\0").decode("latin-1")
        sections.append(Section(name, base + s.VirtualAddress, s.get_data()[:max(s.Misc_VirtualSize, 0) or None],
                                max(s.Misc_VirtualSize, s.SizeOfRawData),
                                bool(ch & 0x40000000), bool(ch & 0x80000000), bool(ch & 0x20000000)))
    symbols, funcs = _coff_symbols(pe, data, base)
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            if exp.name:
                symbols.setdefault(exp.name.decode("latin-1"), base + exp.address)
                funcs.add(exp.name.decode("latin-1"))
    imports = {}
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        dll = entry.dll.decode("latin-1").lower()
        for imp in entry.imports:
            nm = imp.name.decode("latin-1") if imp.name else f"ord{imp.ordinal}"
            imports[imp.address] = f"{dll}!{nm}"
    pdata = {}
    if arch == "x64" and hasattr(pe, "DIRECTORY_ENTRY_EXCEPTION"):
        for rf in pe.DIRECTORY_ENTRY_EXCEPTION:
            pdata[base + rf.struct.BeginAddress] = base + rf.struct.UnwindData
    return Binary(path, "pe", arch, "windows", base, sections, symbols, funcs, imports, pdata,
                  entry=base + pe.OPTIONAL_HEADER.AddressOfEntryPoint)


# ===================================================================== raw

def load_raw(path: str, base: int, arch: str, os_name: str = "linux") -> Binary:
    data = Path(path).read_bytes()
    if arch not in ("x86", "x64"):
        raise InputError(f"raw blob: unsupported architecture {arch!r}")
    sec = Section("raw", base, data, len(data), True, False, True)
    return Binary(path, "raw", arch, os_name, base, [sec], {}, set(), {})


def load(path: str, raw_base: int | None = None, raw_arch: str | None = None) -> Binary:
    data = Path(path).read_bytes()
    if raw_base is not None:
        if raw_arch is None:
            raise InputError("raw blobs need --arch")
        return load_raw(path, raw_base, raw_arch)
    if data[:4] == b"\x7fELF":
        return _load_elf(path, data)
    if data[:2] == b"MZ":
        return _load_pe(path, data)
    raise InputError(f"{path}: unrecognised format (use a raw base address for code blobs)")
