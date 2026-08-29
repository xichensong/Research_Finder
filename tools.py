"""
The tools our agent is allowed to use.

Every tool here is a plain Python function. The model never runs this code
itself — it only ever asks (via a `tool_use` block) for one of these
functions to be called with some arguments, and *we* run it and hand the
result back. That boundary is the whole safety model: the agent can only do
what we've given it a function for.

All file operations are sandboxed to ./sandbox so the agent can't wander
around your real filesystem.
"""

from pathlib import Path

SANDBOX = (Path(__file__).parent / "sandbox").resolve()
SANDBOX.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
    """Resolve a filename to a path inside SANDBOX, refusing to leave it."""
    p = (SANDBOX / filename).resolve()
    if not p.is_relative_to(SANDBOX):
        raise ValueError(f"Refusing to access path outside sandbox: {filename}")
    return p


def read_file(filename: str) -> str:
    """Read the contents of a text file from the sandbox directory."""
    path = _safe_path(filename)
    if not path.exists():
        return f"Error: {filename} does not exist."
    return path.read_text()


def write_file(filename: str, content: str) -> str:
    """Write (or overwrite) a text file in the sandbox directory."""
    path = _safe_path(filename)
    path.write_text(content)
    return f"Wrote {len(content)} characters to {filename}."


def list_files() -> str:
    """List all files currently in the sandbox directory."""
    files = sorted(p.name for p in SANDBOX.iterdir() if p.is_file())
    return "\n".join(files) if files else "(no files yet)"
