"""Checked invocation of external tools with pinned versions."""
from __future__ import annotations

import shutil
import subprocess

from fneq.errors import ToolError, ToolTimeout

# Versions the Docker image pins; `fneq --check-tools` verifies them.
PINNED = {
    "alive-tv": "21.1",      # Alive2 v21.0 built against LLVM 21.1.x
    "opt": "21.1",
    "llvm-link": "21.1",
    "fneq-lift": None,       # built from lifter/ in the image
}


def which(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise ToolError(f"required tool {tool!r} not found on PATH (use the fneq Docker image)")
    return path


def run(cmd: list[str], timeout: float | None = None, ok_codes: tuple[int, ...] = (0,),
        input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run a tool; raise ToolError on a bad exit code or timeout. Never returns silently empty."""
    which(cmd[0])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input_text)
    except subprocess.TimeoutExpired as e:
        raise ToolTimeout(f"{cmd[0]} timed out after {timeout}s") from e
    if p.returncode not in ok_codes:
        tail = (p.stderr or p.stdout)[-1500:]
        raise ToolError(f"{cmd[0]} failed with exit code {p.returncode}: {tail}")
    return p


def versions() -> dict[str, str]:
    out = {}
    for tool in ("opt", "llvm-link", "alive-tv"):
        try:
            p = run([tool, "--version"])
            text = p.stdout + p.stderr
            ver = next((w for w in text.split() if w[:1].isdigit() and "." in w), "unknown")
            out[tool] = ver
        except ToolError as e:
            out[tool] = f"unavailable ({e})"
    return out


def check_pins() -> list[str]:
    problems = []
    for tool, ver in versions().items():
        want = PINNED.get(tool)
        if want and not ver.startswith(want):
            problems.append(f"{tool}: found {ver}, pinned {want}")
    return problems
