#!/usr/bin/env python3
"""After the D35 regeneration: restore every regenerated golden whose content is unchanged.

The patched harness writes its own version string (`<sha>+d35-patch:<hash>`) into the `source`
line of every dump it writes. A golden whose content equals the baseline once the two version
strings are masked (V2.dump, V3_NO2B.dump, random_H.dump, randomH_NO_Ne20.dump, ...) did not
depend on Tcm or Rp2, so the baseline file is copied back and keeps its old, correct version
string and a zero diff. Only files under hamiltonian/, hf/, operator/ and io/ are considered
(the e2e goldens carry wall times and are always rewritten).

Usage: d35_restore_unchanged.py [--baseline DIR] [--staging DIR]
  baseline default: $IMSRG_REFERENCE_ROOT/golden-baseline (a copy of tests/golden at develop
                    c01d766, made before the regeneration)
  staging  default: tests/golden of the checkout that holds this script
"""
import argparse
import os
import re
import shutil
import sys

ROOT = os.environ.get("IMSRG_REFERENCE_ROOT", "/Users/cheshyre/Downloads/new-imsrg-reference")
HERE = os.path.dirname(os.path.abspath(__file__))
VER = re.compile(rb"[0-9a-f]{64}(\+d35-patch:[0-9a-f]+)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=os.path.join(ROOT, "golden-baseline"))
    ap.add_argument("--staging", default=os.path.join(HERE, "..", "..", "tests", "golden"))
    a = ap.parse_args()
    kept, changed, new = [], [], []
    for sub in ("hamiltonian", "hf", "operator", "io"):
        for d, _, files in os.walk(os.path.join(a.staging, sub)):
            for f in sorted(files):
                p = os.path.join(d, f)
                rel = os.path.relpath(p, a.staging)
                if f == "README.md":
                    continue
                b = os.path.join(a.baseline, rel)
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


if __name__ == "__main__":
    sys.exit(main())
