#!/usr/bin/env python3
"""After the D35 regeneration: restore every regenerated golden whose content is unchanged.

The patched harness writes its own version string (`<sha>+d35-patch:<hash>`) into the `source`
line of every dump it writes. A golden whose content equals the baseline once the two version
strings are masked (V2.dump, V3_NO2B.dump, random_H.dump, randomH_NO_Ne20.dump, ...) did not
depend on Tcm or Rp2, so the baseline file is copied back and keeps its old, correct version
string and a zero diff. Only files under hamiltonian/, hf/, operator/ and io/ are considered
(the e2e goldens carry wall times and are always rewritten).

Usage: d35_restore_unchanged.py [--baseline DIR] [--staging DIR]
  baseline default: tests/golden at git revision e291b07 (the goldens before the D35
                    regeneration), extracted with `git archive` into a temporary directory that
                    is removed afterwards. Nothing outside the repository is read.
  staging  default: tests/golden of the checkout that holds this script
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE_REV = "e291b07"
VER = re.compile(rb"[0-9a-f]{64}(\+d35-patch:[0-9a-f]+)?")


def extract_baseline(dest):
    """Extracts tests/golden of BASELINE_REV into dest and returns the path of dest/tests/golden."""
    archive = subprocess.run(
        ["git", "-C", REPO, "archive", "--format=tar", BASELINE_REV, "tests/golden"],
        check=True, stdout=subprocess.PIPE).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)
    return os.path.join(dest, "tests", "golden")


def restore(baseline, staging):
    kept, changed, new = [], [], []
    for sub in ("hamiltonian", "hf", "operator", "io"):
        for d, _, files in os.walk(os.path.join(staging, sub)):
            for f in sorted(files):
                p = os.path.join(d, f)
                rel = os.path.relpath(p, staging)
                if f == "README.md":
                    continue
                b = os.path.join(baseline, rel)
                if not os.path.exists(b):
                    new.append(rel)
                    continue
                mine, base = open(p, "rb").read(), open(b, "rb").read()
                if VER.sub(b"VER", mine) == VER.sub(b"VER", base):
                    if mine != base:
                        shutil.copyfile(b, p)
                    kept.append(rel)
                else:
                    changed.append(rel)
    print("unchanged (baseline restored): %d" % len(kept))
    for r in kept:
        print("   =", r)
    print("changed: %d" % len(changed))
    for r in changed:
        print("   *", r)
    for r in new:
        print("   + (no baseline)", r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None,
                    help="a tests/golden directory to compare with (default: git e291b07)")
    ap.add_argument("--staging", default=os.path.join(HERE, "..", "..", "tests", "golden"))
    a = ap.parse_args()
    tmp = None
    try:
        if a.baseline is None:
            tmp = tempfile.mkdtemp(prefix="d35-baseline-")
            a.baseline = extract_baseline(tmp)
        restore(a.baseline, a.staging)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
