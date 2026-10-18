"""Generate fneq/data/win_arity.json: parameter lists of the Windows API functions that
the given PE files import, from the Windows SDK headers (run on a Windows host).

    python spike/gen_win_arity.py OUT.json A.dll [B.dll ...]

Each entry is the list of parameter kinds in order ("i" integer/pointer, "f" float,
"d" double) or null for a variadic function. Functions not found are left out (fneq
then assumes all argument registers).
"""
import glob
import json
import os
import re
import sys

import pefile

SDK = os.environ.get("WIN_SDK_INCLUDE", r"C:\Program Files (x86)\Windows Kits\10\Include")


def imported_names(paths):
    names = set()
    for p in paths:
        pe = pefile.PE(p, fast_load=True)
        pe.parse_data_directories([1])
        for d in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            for i in d.imports:
                if i.name:
                    names.add(i.name.decode())
    return names


def headers():
    ver = sorted(os.listdir(SDK))[-1]
    out = []
    for sub in ("um", "shared", "ucrt"):
        out += glob.glob(os.path.join(SDK, ver, sub, "*.h"))
    return out


def strip(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r"^\s*#.*$", " ", text, flags=re.M)


def params(body):
    """Split a parameter list at top-level commas."""
    depth, cur, out = 0, "", []
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [p.strip() for p in out]


NARROW = {32: r"DWORD|UINT|INT|BOOL|LONG|ULONG|HRESULT|LCID|int|long|unsigned|u_int|u_long|"
                r"NTSTATUS|COLORREF|LSTATUS|DEVINST|CONFIGRET|ACCESS_MASK|REGSAM|SECURITY_INFORMATION|"
                r"MMRESULT|UINT32|INT32|DWORD32",
          16: r"WORD|USHORT|SHORT|short|u_short|LANGID|ATOM|WCHAR|wchar_t",
          8: r"BYTE|UCHAR|CHAR|BOOLEAN|char"}


def kind(p):
    """'i' (a full-width integer or pointer), 'i:N' (an N-bit integer), 'f' or 'd'."""
    # SAL annotations (_In_, _Out_writes_(n), ...) carry no type
    p = re.sub(r"\b_[A-Z][A-Za-z_]*_(\([^)]*\))?", " ", p)
    if re.search(r"\*", p):
        return "i"
    if re.search(r"\b(FLOAT|float)\b", p):
        return "f"
    if re.search(r"\b(DOUBLE|double)\b", p):
        return "d"
    words = re.findall(r"[A-Za-z_]\w*", p)
    base = [w for w in words[:-1] if w not in ("const", "CONST", "unsigned", "signed", "struct", "enum")] \
        if len(words) > 1 else words
    if "unsigned" in words and not base:
        return "i:32"
    for bits, pat in NARROW.items():
        if base and re.fullmatch(pat, base[-1]):
            return f"i:{bits}"
    return "i"


def main():
    out_path, dlls = sys.argv[1], sys.argv[2:]
    want = imported_names(dlls)
    found = {}
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, want), key=len, reverse=True)) + r")\s*\(([^;{}]*?)\)\s*[;{]", re.S)
    for h in headers():
        try:
            text = strip(open(h, encoding="latin-1").read())
        except OSError:
            continue
        for m in pat.finditer(text):
            name, body = m.group(1), m.group(2)
            if name in found:
                continue
            # a declaration, not a call or a macro use: its parameters have types
            ps = params(body)
            if ps == [""] or [p.upper() for p in ps] == ["VOID"]:
                found[name] = []
            elif "..." in body:
                found[name] = None
            elif all(re.search(r"[A-Za-z_]\w*\W+[A-Za-z_]\w*\s*$|\*\s*\w*\s*$|^\w+$", p) for p in ps):
                found[name] = [kind(p) for p in ps]
    json.dump(dict(sorted(found.items())), open(out_path, "w"), indent=0)
    print(f"{len(found)} of {len(want)} imported names found -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
