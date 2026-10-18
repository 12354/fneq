"""The on-disk bounding cache (FNEQ_BOUND_CACHE) is keyed by the binary's content.

Run inside the fneq image: pytest tests/test_bound_cache.py
"""
from __future__ import annotations

import os
import shutil

import pytest

pytest.importorskip("angr")

from fneq import binary as binmod, bounding  # noqa: E402

# lea eax,[rdi+rsi]; ret   /   lea eax,[rdi+rsi+1]; ret
ADD = bytes.fromhex("8d0437c3")
ADD_PLUS_1 = bytes.fromhex("8d443701c3")


def bound_fresh(path, calls):
    """Bound the function at 0x1000 in a fresh process state; count real analyses."""
    bounding._BOUNDS.clear()
    bounding._CONTENT.clear()
    orig = bounding._bound

    def counted(b, e):
        calls.append(path)
        return orig(b, e)
    bounding._bound = counted
    try:
        return bounding.bound(binmod.load(str(path), 0x1000, "x64"), 0x1000)
    finally:
        bounding._bound = orig


def test_cache_follows_content(tmp_path, monkeypatch):
    monkeypatch.setenv("FNEQ_BOUND_CACHE", str(tmp_path / "cache"))
    a = tmp_path / "a.bin"
    a.write_bytes(ADD)
    calls: list = []
    first = bound_fresh(a, calls)
    assert len(calls) == 1

    # an identical copy elsewhere, with a new time stamp: served from the cache
    b = tmp_path / "copy" / "b.bin"
    b.parent.mkdir()
    shutil.copy(a, b)
    os.utime(b, (1, 1))
    again = bound_fresh(b, calls)
    assert len(calls) == 1 and again.code == first.code

    # the function changed (same path): analysed again, with the new code
    a.write_bytes(ADD_PLUS_1)
    changed = bound_fresh(a, calls)
    assert len(calls) == 2 and changed.code != first.code


def test_no_cache_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FNEQ_BOUND_CACHE", raising=False)
    a = tmp_path / "a.bin"
    a.write_bytes(ADD)
    calls: list = []
    bound_fresh(a, calls)
    bound_fresh(a, calls)
    assert len(calls) == 2


def test_memory_cache_checks_owner_when_identity_key_is_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("FNEQ_BOUND_CACHE", raising=False)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(ADD)
    b.write_bytes(ADD_PLUS_1)
    # Reproduce the cache-key collision that Python id() reuse caused in the
    # combined bounding tests, without depending on the allocator's timing.
    monkeypatch.setattr(bounding, "id", lambda obj: -1, raising=False)
    first = bounding.bound(binmod.load(str(a), 0x1000, "x64"), 0x1000)
    second = bounding.bound(binmod.load(str(b), 0x1000, "x64"), 0x1000)
    assert b"".join(first.code.values()) == ADD
    assert b"".join(second.code.values()) == ADD_PLUS_1


def test_angr_project_cache_follows_rebuilt_file(tmp_path):
    path = tmp_path / "rebuild.bin"
    path.write_bytes(ADD)
    old = bounding.angr_project(binmod.load(str(path), 0x1000, "x64"))
    path.write_bytes(ADD_PLUS_1)
    new = bounding.angr_project(binmod.load(str(path), 0x1000, "x64"))
    assert new is not old
    assert new.factory.block(0x1000).bytes == ADD_PLUS_1
