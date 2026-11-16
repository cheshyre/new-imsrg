#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Generates the physics-symbol ignore patterns of .clang-tidy from tools/physics_symbols.txt.

Usage: tools/gen_physics_symbols.py [--check | --write] [--repo-root DIR]

  --write   rewrites the IgnoredRegexp lines of the variable-like naming kinds in .clang-tidy
  --check   exits 1 if .clang-tidy is not what --write would produce (default)

A name is accepted when its '_'-separated parts are lower-case words or listed symbols and at
least one part is a listed symbol (design/17-code-style.md section 3, exception 6). Private and
protected members additionally end in '_'; a parameter may end in '_' to avoid a member name.
Local constants also accept the kName form.
"""

import argparse
import re
import sys
from pathlib import Path

# naming kind -> required/optional trailing underscore of the pattern
KINDS = {
    "Variable": "_?",
    "LocalVariable": "_?",
    "Parameter": "_?",
    "PublicMember": "",
    "PrivateMember": "_",
    "ProtectedMember": "_",
    "LocalConstant": "_?",
}
K_NAME = "k[A-Z][A-Za-z0-9]*"


def load_symbols(path):
    """Returns the symbols of `path` (one per line, '#' comments), rejecting duplicates."""
    symbols = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", line):
                raise SystemExit(f"{path}: bad symbol '{line}'")
            symbols.append(line)
    if len(set(symbols)) != len(symbols):
        raise SystemExit(f"{path}: duplicate symbols")
    return symbols


def pattern(symbols, suffix):
    """The (unanchored) extended regular expression for `symbols` and a trailing-underscore rule."""
    sym = "|".join(sorted(symbols, key=lambda s: (-len(s), s)))
    part = f"([a-z0-9]+|{sym})"
    return f"({part}_)*({sym})(_{part})*{suffix}"


def rewrite(text, symbols):
    """`text` (the contents of .clang-tidy) with the IgnoredRegexp lines regenerated."""
    for kind, suffix in KINDS.items():
        pat = pattern(symbols, suffix)
        if kind == "LocalConstant":
            pat = f"{K_NAME}|{pat}"
        pat = f"^({pat})$"
        line = f"  readability-identifier-naming.{kind}IgnoredRegexp: '{pat}'"
        text, n = re.subn(
            rf"^  readability-identifier-naming\.{kind}IgnoredRegexp:.*$",
            lambda _m, line=line: line,
            text,
            flags=re.M,
        )
        if n != 1:
            raise SystemExit(f".clang-tidy: expected one {kind}IgnoredRegexp line, found {n}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = Path(args.repo_root)
    symbols = load_symbols(root / "tools" / "physics_symbols.txt")
    cfg = root / ".clang-tidy"
    old = cfg.read_text()
    new = rewrite(old, symbols)
    if args.write:
        cfg.write_text(new)
        print(f"gen_physics_symbols: {len(symbols)} symbols written to {cfg}")
        return 0
    if new != old:
        print("gen_physics_symbols: .clang-tidy is out of date with tools/physics_symbols.txt; "
              "run tools/gen_physics_symbols.py --write")
        return 1
    print(f"gen_physics_symbols: .clang-tidy matches the {len(symbols)} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
