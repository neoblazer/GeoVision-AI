"""Fail fast when generated or heavyweight files leak into the repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {".venv", "venv", "node_modules", "__pycache__", "datasets", "models"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".pyc"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def tracked_files() -> list[Path]:
    """Return Git-tracked files, or a conservative source-only fallback."""

    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        return [ROOT / item for item in result.stdout.decode().split("\0") if item]

    ignored_parts = FORBIDDEN_DIRS | {".git", ".pytest_cache", ".ruff_cache"}
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in ignored_parts for part in path.relative_to(ROOT).parts)
    ]


def main() -> int:
    violations: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            violations.append(f"forbidden path: {relative}")
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden file type: {relative}")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            violations.append(f"file exceeds 20 MiB: {relative}")

    if violations:
        print("Repository hygiene check failed:")
        for violation in sorted(set(violations)):
            print(f"- {violation}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
