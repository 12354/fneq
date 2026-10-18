"""Read-only literal correspondence uses complete UTF-16 strings."""
from __future__ import annotations

from fneq.binary import Binary, Section
from fneq.correspondence import Correspondence, _content_key


def literals(base, words):
    data = bytearray(0x40 * len(words))
    for i, word in enumerate(words):
        raw = (word + "\0").encode("utf-16-le")
        data[0x40 * i:0x40 * i + len(raw)] = raw
    return Binary("unused", "pe", "x64", "windows", 0,
                  [Section(".rdata", base, bytes(data), len(data))], {}, set(), {})


def test_complete_utf16_content_key():
    binary = literals(0x1000, ["Hello", "Howdy"])
    assert _content_key(binary, 0x1000) == "Hello\0".encode("utf-16-le")
    assert _content_key(binary, 0x1040) == "Howdy\0".encode("utf-16-le")


def test_same_first_character_does_not_map_different_literals():
    a, b = literals(0x1000, ["Hello"]), literals(0x2000, ["Howdy"])
    corr = Correspondence().add_readonly(a, {0x1000}, b, {0x2000})
    assert corr.map_b(0x2000) == 0x2000


def test_wide_literal_matching_with_shared_first_character():
    a = literals(0x1000, ["Hello", "Howdy"])
    b = literals(0x2000, ["Howdy", "Hello"])
    corr = Correspondence().add_readonly(a, {0x1000, 0x1040}, b, {0x2000, 0x2040})
    assert corr.map_b(0x2000) == 0x1040
    assert corr.map_b(0x2040) == 0x1000
