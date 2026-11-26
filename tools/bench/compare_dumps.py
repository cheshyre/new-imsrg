#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Compare two operator dumps of commutator_bench / tensor_commutator_bench --dump.

Usage: compare_dumps.py A B [--tol 1e-12]

Prints, per part (0B, 1B, 2B), max|A - B| and max|A|, and the relative deviation
max|A - B| / max(1, max|A|) (the tolerance form of design/30 and D59). Exits 1 if
any part exceeds --tol (default 1e-12) or the shapes differ. Used to check device
(HIP) results against host (OpenMP) results, M14.
"""
import argparse
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        n1, n2 = struct.unpack("<QQ", f.read(16))
        data = np.fromfile(f, dtype="<f8")
    if data.size != 1 + n1 + n2:
        sys.exit(f"{path}: expected {1 + n1 + n2} doubles, found {data.size}")
    return data[:1], data[1:1 + n1], data[1 + n1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=1.0e-12)
    args = ap.parse_args()
    a = load(args.a)
    b = load(args.b)
    ok = True
    for name, x, y in zip(("0B", "1B", "2B"), a, b):
        if x.shape != y.shape:
            print(f"{name}: shapes differ {x.shape} vs {y.shape}")
            ok = False
            continue
        d = float(np.max(np.abs(x - y))) if x.size else 0.0
        m = float(np.max(np.abs(x))) if x.size else 0.0
        rel = d / max(1.0, m)
        print(f"{name}: {x.size} values, max|A-B| {d:.3e}, max|A| {m:.3e}, "
              f"relative {rel:.3e}")
        ok = ok and rel <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
