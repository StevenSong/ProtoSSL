#!/usr/bin/env python3
"""
Pre-commit hook: verify every file tracked by git has a CODEOWNERS entry.

Managed via pre-commit. To run manually:
    python3 .github/hooks/pre-commit
"""
import fnmatch
import os
import subprocess
import sys

CODEOWNERS_PATH = ".github/CODEOWNERS"


def parse_patterns(path: str) -> list[str]:
    """Return CODEOWNERS patterns in reverse order (last match wins)."""
    patterns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pattern = line.split()[0]
            patterns.append(pattern)
    return list(reversed(patterns))


def has_owner(file: str, patterns: list[str]) -> bool:
    """
    Return True if the file is matched by any CODEOWNERS pattern.

    Matching rules (mirrors GitHub's behaviour):
      - Patterns are tested in reverse order; the first match wins.
      - A leading slash anchors the pattern to the repo root.
      - A trailing slash matches a directory and everything beneath it.
      - Patterns without a slash match on the basename (any depth).
      - Otherwise fnmatch-style glob matching applies to the full path.
    """
    for pattern in patterns:
        if pattern in ("*", "/**"):
            return True

        anchored = pattern.startswith("/")
        norm = pattern.lstrip("/")

        if norm.endswith("/"):
            if file.startswith(norm) or fnmatch.fnmatch(file, norm + "*"):
                return True
            continue

        if not anchored and "/" not in norm:
            if fnmatch.fnmatch(os.path.basename(file), norm):
                return True
            continue

        if fnmatch.fnmatch(file, norm):
            return True

        if file.startswith(norm.rstrip("/") + "/"):
            return True

    return False


def tracked_files() -> list[str]:
    """All files currently tracked by git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.splitlines() if f]


def main() -> None:
    patterns = parse_patterns(CODEOWNERS_PATH)
    files = tracked_files()

    missing = [f for f in files if not has_owner(f, patterns)]

    if not missing:
        print("All tracked files have a CODEOWNERS entry.")
        return

    print("error: the following files have no CODEOWNERS entry:", file=sys.stderr)
    for f in missing:
        print(f"  {f}", file=sys.stderr)
    print(
        f"\nAdd a pattern covering each file to {CODEOWNERS_PATH}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
