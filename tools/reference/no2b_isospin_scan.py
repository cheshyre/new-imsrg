#!/usr/bin/env python3
"""Scan the start of a NO2B ".stream.bin" 3N file in the loop order of the imsrg++ reader.

Replicates ThreeBodyStorage_no2b::ReadFile (imsrg-devel src/ThreeBody/ThreeBodyStorage_no2b.cc,
lines 552-810): file orbits ordered by e, then l ascending, then j ascending; loops
iF1 >= iF2, iF3, iF1 >= iF4 >= iF5, iF6 (l6 = l3, j6 = j3), J, T12, T45, 2T; single-precision
floats. Reports, for the blocks with e1 <= emax_scan, statistics per total isospin (2T = 1, 3)
and Pauli violations (i1 == i2 with J + T12 even, which must be zero).

Diagnostic used in M1 (tests/golden/e2e/README.md): a regulated 3N contact (cE) term has a
totally symmetric spatial kernel, so its T = 3/2 matrix elements vanish identically; the
pi-exchange (c1, c3, c4) and cD terms do not.

Usage: python3 -I no2b_isospin_scan.py FILE [emax_file=6 e2max_file=12 e3max_file=18 emax_scan=2]
"""
import array
import sys


def main():
    path = sys.argv[1]
    opts = dict(a.split("=") for a in sys.argv[2:])
    emax_f = int(opts.get("emax_file", 6))
    e2max_f = int(opts.get("e2max_file", 12))
    e3max_f = int(opts.get("e3max_file", 18))
    lmax_f = int(opts.get("lmax_file", emax_f))
    emax_scan = int(opts.get("emax_scan", 2))

    orbits = []  # (e, l, twoj)
    for e in range(emax_f + 1):
        for l in range(e % 2, min(e, lmax_f) + 1, 2):
            for twoj in range(abs(2 * l - 1), 2 * l + 2, 2):
                orbits.append((e, l, twoj))
    norb = len(orbits)

    # Collect the (iF1..iF6, J, T12, T45, T3) labels of the elements we will read.
    labels = []
    for i1 in range(norb):
        e1, l1, j1 = orbits[i1]
        if e1 > emax_scan:
            break  # later blocks in the file all have e1 > emax_scan
        for i2 in range(i1 + 1):
            e2, l2, j2 = orbits[i2]
            if e2 > emax_f or e1 + e2 > e2max_f:
                continue
            for i3 in range(norb):
                e3, l3, j3 = orbits[i3]
                if e3 > emax_f or e2 + e3 > e2max_f or e1 + e3 > e2max_f or e1 + e2 + e3 > e3max_f:
                    continue
                for i4 in range(i1 + 1):
                    e4, l4, j4 = orbits[i4]
                    if e4 > emax_f:
                        continue
                    for i5 in range(i4 + 1):
                        e5, l5, j5 = orbits[i5]
                        if e5 > emax_f or e4 + e5 > e2max_f:
                            continue
                        for i6 in range(norb):
                            e6, l6, j6 = orbits[i6]
                            if j6 != j3 or l6 != l3 or e6 > emax_f:
                                continue
                            if e4 + e6 > e2max_f or e5 + e6 > e2max_f or e4 + e5 + e6 > e3max_f:
                                continue
                            if (l1 + l2 + l3) % 2 != (l4 + l5 + l6) % 2:
                                continue
                            jmin = max(abs(j1 - j2), abs(j4 - j5)) // 2
                            jmax = min(j1 + j2, j4 + j5) // 2
                            for J in range(jmin, jmax + 1):
                                for t12 in (0, 1):
                                    for t45 in (0, 1):
                                        for t3 in range(1, min(2 * t12 + 1, 2 * t45 + 1) + 1, 2):
                                            labels.append((i1, i2, i4, i5, J, t12, t45, t3))

    vals = array.array("f")
    with open(path, "rb") as f:
        vals.fromfile(f, len(labels))
    if sys.byteorder != "little":
        vals.byteswap()

    stats = {1: [0, 0.0, 0.0], 3: [0, 0.0, 0.0]}  # count, max|v|, sum v^2
    pauli_bad = 0
    for (i1, i2, i4, i5, J, t12, t45, t3), v in zip(labels, vals):
        s = stats[t3]
        s[0] += 1
        s[1] = max(s[1], abs(v))
        s[2] += v * v
        if ((i1 == i2 and (J + t12) % 2 == 0) or (i4 == i5 and (J + t45) % 2 == 0)) and abs(v) > 1e-6:
            pauli_bad += 1

    print(f"file orbits: {norb}; elements scanned (e1 <= {emax_scan}): {len(labels)}")
    print("first 8 floats:", " ".join(f"{v:.6g}" for v in vals[:8]))
    for t3 in (1, 3):
        n, vmax, ss = stats[t3]
        print(f"2T={t3}: count {n}  max|V| {vmax:.6g}  rms {((ss / n) ** 0.5 if n else 0):.6g}")
    print(f"Pauli-forbidden elements with |V| > 1e-6: {pauli_bad}")


if __name__ == "__main__":
    main()
