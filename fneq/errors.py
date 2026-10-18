"""Exception types; each maps to a verdict or an exit code."""


class FneqError(Exception):
    """Base class."""


class InputError(FneqError):
    """Bad user input (missing file, unknown symbol, ...). Exit code 3."""


class ToolError(FneqError):
    """An external tool failed or produced output we could not validate. Exit code 3."""


class Unsupported(FneqError):
    """The function uses something we cannot analyse soundly. Verdict UNKNOWN."""


class ToolTimeout(ToolError):
    """An external tool exceeded its wall-clock limit (no answer, not a malfunction)."""
