#!/usr/bin/env python3
"""Enforce the imsrg module layering rules (design/10-architecture.md section 2).

Checks:
  (a) CMake-declared module deps (``--module-deps FILE``, JSON written by CMake) stay
      inside the allowed DAG and each module lives in ``src/imsrg/<name>``.
  (b) ``#include "imsrg/<t>/..."`` / ``<imsrg/<t>/...>`` in ``src/imsrg/<m>/`` only
      targets modules in the transitive closure of ``<m>`` (or ``<m>`` itself).
  (c) Third-party headers are confined, including transitively through repo headers:
      Kokkos only in ``src/imsrg/exec/`` and ``src/imsrg/*/kernels/``; ``mpi.h`` only in
      ``comm/``; toml++ only in ``driver/``; CLI11 nowhere under ``src/``.

Adding a module: update ``ALLOWED_DEPS`` below AND design/10-architecture.md section 2
in the same commit, then declare it in CMake.

Usage: check_layering.py [--repo-root DIR] [--module-deps FILE] [--no-git] [--self-test]
Exit status: 0 on success, 1 on violations or self-test failure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# Direct allowed edges. A module may use the transitive closure of its deps.
ALLOWED_DEPS: Dict[str, List[str]] = {
    "base": [],
    "exec": ["base"],
    "comm": ["base"],
    "linalg": ["base", "exec"],
    "angmom": ["base", "exec"],
    "basis": ["base", "exec", "angmom"],
    "operator": ["basis", "linalg", "angmom", "exec", "base"],
    "io": ["operator", "basis", "base"],
    "opslib": ["operator", "basis", "angmom"],
    "commutator": ["operator", "linalg", "angmom", "exec", "comm"],
    "commutator_ref": ["operator", "angmom"],  # test-only reference implementation
    "hf": ["operator", "linalg", "basis"],
    "mbpt": ["operator", "linalg", "basis"],
    "flow": ["commutator", "operator"],
    "driver": ["base", "exec", "comm", "linalg", "angmom", "basis", "operator", "io",
               "opslib", "commutator", "hf", "mbpt", "flow"],
}

CPP_EXTS = {".hpp", ".h", ".hh", ".hxx", ".cpp", ".cc", ".cxx", ".inl", ".ipp", ".tpp", ".cu"}
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')

# Third-party families: (regex on include target, allowed-zone regex on rel path, rule text).
THIRD_PARTY = {
    "Kokkos": (re.compile(r"^(?:[A-Za-z_]+/)?Kokkos\w*\.h(?:pp)?$"),
               re.compile(r"^src/imsrg/(?:exec/|[^/]+/kernels/)"),
               "Kokkos headers are allowed only in src/imsrg/exec/ and src/imsrg/*/kernels/"),
    "MPI": (re.compile(r"^mpi\.h$"), re.compile(r"^src/imsrg/comm/"),
            "MPI headers are allowed only in src/imsrg/comm/"),
    "toml++": (re.compile(r"^(?:toml\+\+/.*|toml\.hpp)$"), re.compile(r"^src/imsrg/driver/"),
               "toml++ headers are allowed only in src/imsrg/driver/"),
    "CLI11": (re.compile(r"^(?:CLI/.*|CLI11\.hpp)$"), re.compile(r"(?!)"),
              "CLI11 headers are not allowed under src/ (apps only)"),
}
KIND_OF = {"Kokkos": "kokkos", "MPI": "mpi", "toml++": "toml", "CLI11": "cli11"}


class Violation(NamedTuple):
    """One rule violation: `file` is repo-relative (or the module-deps file name)."""
    file: str
    kind: str
    message: str

    def __str__(self) -> str:
        return self.message


def find_cycle(dag: Dict[str, List[str]]) -> Optional[List[str]]:
    """Return a dependency cycle (names, first == last) or None if `dag` is acyclic."""
    state: Dict[str, int] = {}
    stack: List[str] = []

    def visit(n: str) -> Optional[List[str]]:
        state[n] = 1
        stack.append(n)
        for d in dag.get(n, []):
            if state.get(d, 0) == 1:
                return stack[stack.index(d):] + [d]
            if state.get(d, 0) == 0:
                c = visit(d)
                if c:
                    return c
        stack.pop()
        state[n] = 2
        return None

    for n in dag:
        if state.get(n, 0) == 0:
            c = visit(n)
            if c:
                return c
    return None


def closure(dag: Dict[str, List[str]], m: str) -> Set[str]:
    """Transitive dependency closure of `m` (excluding `m` itself unless cyclic)."""
    out: Set[str] = set()
    todo = list(dag.get(m, []))
    while todo:
        d = todo.pop()
        if d not in out:
            out.add(d)
            todo.extend(dag.get(d, []))
    return out


def check_module_deps(path: Path, dag: Dict[str, List[str]]) -> List[Violation]:
    """Validate the CMake-written module list against the allowed DAG."""
    vs: List[Violation] = []
    name = path.name
    data = json.loads(path.read_text())
    for mod in data.get("modules", []):
        m, deps, d = mod.get("name"), mod.get("deps", []), mod.get("dir")
        if m not in dag:
            vs.append(Violation(name, "cmake-unknown-module",
                                f"unknown module {m}: add it to design/10-architecture.md "
                                f"section 2 and tools/check_layering.py"))
            continue
        allowed = closure(dag, m)
        for dep in deps:
            if dep not in dag:
                vs.append(Violation(name, "cmake-dep",
                                    f"{m}: CMake dep '{dep}' is an unknown module"))
            elif dep not in allowed:
                vs.append(Violation(name, "cmake-dep",
                                    f"{m}: CMake dep '{dep}' not allowed "
                                    f"(allowed: {sorted(allowed)})"))
        if d != f"src/imsrg/{m}":
            vs.append(Violation(name, "cmake-dir",
                                f"{m}: dir is '{d}', expected 'src/imsrg/{m}'"))
    return vs


def list_files(root: Path, use_git: bool) -> List[str]:
    """Repo-relative posix paths of C/C++ files under src/: git-tracked files plus new files not
    yet added (never gitignored ones), so a check before `git add` sees new headers; else a walk."""
    names: Optional[List[str]] = None
    if use_git:
        try:
            r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                                "--exclude-standard", "src/"], cwd=root,
                               capture_output=True, check=True)
            names = [n for n in r.stdout.decode().split("\0") if n]
        except (OSError, subprocess.CalledProcessError):
            names = None
    if names is None:
        src = root / "src"
        names = ([p.relative_to(root).as_posix() for p in src.rglob("*") if p.is_file()]
                 if src.is_dir() else [])
    return sorted(n for n in names if Path(n).suffix in CPP_EXTS and (root / n).is_file())


def parse_includes(path: Path) -> List[Tuple[int, str, str]]:
    """(line number, '<' or '"', target) for each #include in the file."""
    out = []
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        mt = INCLUDE_RE.match(line)
        if mt:
            out.append((i, mt.group(1), mt.group(2).strip()))
    return out


def resolve(rel: str, delim: str, target: str, files: Set[str]) -> Optional[str]:
    """Map an include to a repo file path in `files`, or None if it is not a repo file."""
    if target.startswith("imsrg/"):
        cand = "src/" + target
    elif delim == '"':
        parts: List[str] = []
        for p in (Path(rel).parent / target).as_posix().split("/"):
            if p == "..":
                if parts:
                    parts.pop()
            elif p != ".":
                parts.append(p)
        cand = "/".join(parts)
    else:
        return None
    return cand if cand in files else None


def check_includes(root: Path, files: List[str], dag: Dict[str, List[str]]) -> List[Violation]:
    """Include-edge (rule b) and third-party confinement with taint propagation (rule c)."""
    vs: List[Violation] = []
    fset = set(files)
    # taint[f][X] = (chain string, next repo file or None when included directly)
    taint: Dict[str, Dict[str, Tuple[str, Optional[str]]]] = {f: {} for f in files}
    edges: Dict[str, List[Tuple[str, str]]] = {f: [] for f in files}
    for f in files:
        parts = f.split("/")
        mod: Optional[str] = None
        known = False
        if parts[:2] == ["src", "imsrg"]:
            if len(parts) == 3:
                vs.append(Violation(f, "file-in-root",
                                    f"{f}:1: file must live in a module dir src/imsrg/<module>/"))
            else:
                mod = parts[2]
                known = mod in dag
                if not known:
                    vs.append(Violation(f, "unknown-module",
                                        f"{f}:1: unknown module '{mod}': add it to "
                                        f"design/10-architecture.md section 2 and "
                                        f"tools/check_layering.py"))
        allowed = (closure(dag, mod) | {mod}) if known and mod else set()
        for ln, delim, tgt in parse_includes(root / f):
            mt = re.match(r"^imsrg/([^/]+)/", tgt)
            if mt and known and mt.group(1) not in allowed:
                vs.append(Violation(f, "include-edge",
                                    f"{f}:{ln}: {mod} may not include imsrg/{mt.group(1)}/... "
                                    f"(allowed: {sorted(allowed - {mod})})"))
            shown = f"<{tgt}>" if delim == "<" else f'"{tgt}"'
            for x, (rx, _zone, _txt) in THIRD_PARTY.items():
                if rx.match(tgt):
                    taint[f].setdefault(x, (shown, None))
            r = resolve(f, delim, tgt, fset)
            if r:
                edges[f].append((r, tgt))
    changed = True
    while changed:  # fixpoint over the include graph; cycle-safe since taint only grows
        changed = False
        for f in files:
            for r, tgt in edges[f]:
                for x, (chain, _n) in list(taint[r].items()):
                    if x not in taint[f]:
                        taint[f][x] = (f"{tgt} -> {chain}", r)
                        changed = True
    for f in files:
        for x, (chain, nxt) in taint[f].items():
            _rx, zone, rule = THIRD_PARTY[x]
            if zone.match(f):
                continue
            how = f"includes {chain} directly" if nxt is None else f"pulls in {x} via {chain}"
            vs.append(Violation(f, KIND_OF[x], f"{f}: {how}; {rule}"))
    return vs


def check(repo_root: Path, module_deps_path: Optional[Path], use_git: bool) -> List[Violation]:
    """Run all checks; returns the violations (empty means OK)."""
    vs: List[Violation] = []
    if module_deps_path is not None:
        vs += check_module_deps(module_deps_path, ALLOWED_DEPS)
    vs += check_includes(repo_root, list_files(repo_root, use_git), ALLOWED_DEPS)
    return vs


EXPECTED_BAD = {
    ("src/imsrg/basis/uses_operator.hpp", "include-edge"),
    ("src/imsrg/basis/angle_form.hpp", "include-edge"),
    ("src/imsrg/mystery/a.hpp", "unknown-module"),
    ("src/imsrg/loose.hpp", "file-in-root"),
    ("src/imsrg/operator/direct_kokkos.hpp", "kokkos"),
    ("src/imsrg/operator/transitive.hpp", "kokkos"),
    ("src/imsrg/hf/uses_mpi.hpp", "mpi"),
    ("src/imsrg/io/uses_toml.cpp", "toml"),
    ("src/imsrg/base/uses_cli.cpp", "cli11"),
    ("module_deps.json", "cmake-dep"),
    ("module_deps.json", "cmake-unknown-module"),
    ("module_deps.json", "cmake-dir"),
}


def self_test(repo_root: Path) -> int:
    """Run check() on the good/bad fixtures and the cycle detector; 0 on success."""
    base = repo_root / "tests" / "layering" / "fixtures"
    errs: List[str] = []
    if find_cycle(ALLOWED_DEPS) is not None:
        errs.append("built-in table is cyclic")
    if find_cycle({"a": ["b"], "b": ["c"], "c": ["a"]}) is None:
        errs.append("cyclic table not detected")
    if find_cycle({"a": ["b"], "b": []}) is not None:
        errs.append("acyclic table reported as cyclic")
    good = check(base / "good", base / "good" / "module_deps.json", False)
    errs += [f"good: unexpected violation: {v}" for v in good]
    bad = check(base / "bad", base / "bad" / "module_deps.json", False)
    got = {(v.file, v.kind) for v in bad}
    errs += [f"bad: missing expected violation {e}" for e in sorted(EXPECTED_BAD - got)]
    errs += [f"bad: unexpected violation {v}" for v in bad
             if (v.file, v.kind) not in EXPECTED_BAD]
    if errs:
        print("self-test: FAIL", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        return 1
    print("self-test: PASS")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Check imsrg module layering.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--module-deps", type=Path)
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    cyc = find_cycle(ALLOWED_DEPS)
    if cyc:
        print("check_layering: FAIL: allowed-dependency table has a cycle: " + " -> ".join(cyc),
              file=sys.stderr)
        return 1
    if a.self_test:
        return self_test(a.repo_root)
    vs = check(a.repo_root, a.module_deps, not a.no_git)
    if vs:
        for v in vs:
            print(v, file=sys.stderr)
        print(f"check_layering: FAIL ({len(vs)} violations)", file=sys.stderr)
        return 1
    nfiles = len(list_files(a.repo_root, not a.no_git))
    print(f"check_layering: OK ({len(ALLOWED_DEPS)} modules declared, {nfiles} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
