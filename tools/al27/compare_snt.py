#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Elementwise comparison of two scalar KSHELL (tokyo) .snt files.

Orbits are matched by their (n, l, 2j, 2tz) labels, so the two files may
number them differently. Two-body elements are brought to a canonical form
(a <= b, c <= d, bra <= ket, with the antisymmetry phase
-(-1)^((j_a + j_b)/2 - J) for a swap within a pair and bra-ket symmetry), so
the orderings of the writers do not matter. Elements present in only one
file are compared against 0 (the 79e9330 writer drops 1b elements below
1e-7). Prints the zero-body terms, the maximum absolute deviation of the 1b
and 2b parts, the worst elements, and exits 1 if the maximum exceeds --tol.

usage: compare_snt.py <reference.snt> <test.snt> [--tol 1e-4] [--top 10]
"""

import argparse
import sys


def parse(path):
    lines = []
    zero = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("!"):
                if "Zero body term" in raw:
                    zero = float(raw.split(":")[1])
                continue
            s = raw.split("!")[0].strip()
            if s:
                lines.append(s.split())
    pos = 0
    nv_p, nv_n = int(lines[pos][0]), int(lines[pos][1])
    pos += 1
    orbit = {}
    for _ in range(nv_p + nv_n):
        k, n, l, j2, tz = (int(x) for x in lines[pos][:5])
        orbit[k] = (n, l, j2, tz)
        pos += 1
    n1 = int(lines[pos][0])
    pos += 1
    one = {}
    for _ in range(n1):
        a, b = int(lines[pos][0]), int(lines[pos][1])
        v = float(lines[pos][2])
        key = tuple(sorted((orbit[a], orbit[b])))
        one[key] = v
        pos += 1
    n2 = int(lines[pos][0])
    pos += 1
    two = {}
    for _ in range(n2):
        a, b, c, d, J = (int(x) for x in lines[pos][:5])
        v = float(lines[pos][5])
        key, phase = canonical(orbit[a], orbit[b], orbit[c], orbit[d], J)
        if key is not None:
            two[key] = phase * v
        pos += 1
    return zero, one, two


def canonical(a, b, c, d, J):
    phase = 1.0

    def swap_phase(x, y):
        return -((-1) ** ((x[2] + y[2]) // 2 - J))

    if a > b:
        phase *= swap_phase(a, b)
        a, b = b, a
    if c > d:
        phase *= swap_phase(c, d)
        c, d = d, c
    if a == b and J % 2 == 1:
        return None, 0.0
    if c == d and J % 2 == 1:
        return None, 0.0
    if (a, b) > (c, d):
        a, b, c, d = c, d, a, b
    return (a, b, c, d, J), phase


def compare(ref, test, label, top):
    keys = set(ref) | set(test)
    rows = []
    for k in keys:
        r, t = ref.get(k, 0.0), test.get(k, 0.0)
        rows.append((abs(r - t), k, r, t, k in ref, k in test))
    rows.sort(reverse=True)
    mx = rows[0][0] if rows else 0.0
    only_ref = sum(1 for x in rows if x[4] and not x[5])
    only_test = sum(1 for x in rows if x[5] and not x[4])
    print(f"{label}: {len(ref)} ref / {len(test)} test elements, "
          f"only in ref {only_ref}, only in test {only_test}, "
          f"max |diff| = {mx:.3e}")
    for d, k, r, t, _, _ in rows[:top]:
        print(f"   {k}: ref {r: .7e} test {t: .7e} diff {d:.2e}")
    return mx


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--tol", type=float, default=1.0e-4)
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()
    z_r, one_r, two_r = parse(a.ref)
    z_t, one_t, two_t = parse(a.test)
    print(f"zero body: ref {z_r} test {z_t}")
    m1 = compare(one_r, one_t, "1b", a.top)
    m2 = compare(two_r, two_t, "2b", a.top)
    mx = max(m1, m2)
    print(f"max |diff| = {mx:.3e} (tol {a.tol:.1e}): "
          f"{'PASS' if mx <= a.tol else 'FAIL'}")
    return 0 if mx <= a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
