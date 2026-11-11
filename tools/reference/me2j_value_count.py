#!/usr/bin/env python3
"""Expected number of values in a Darmstadt me2j file (imsrg++ loop order).

Mirrors the loops of imsrg-devel ReadWrite::ReadBareTBME_Darmstadt_from_stream
(ReadWrite.cc:832ff): orbits ordered by e ascending, then l ascending, then j
ascending; loops nlj1 >= nlj2, nlj3 <= nlj1, nlj4 <= (nlj3 == nlj1 ? nlj2 : nlj3);
pair cut e1+e2 <= e2max; parity conservation; J from Jmin to Jmax; 4 floats per
J line (T=0 pn, nn, T=1 pn, pp).

Usage: me2j_value_count.py E1MAX E2MAX LMAX
Compare with:  gzip -dc FILE.me2j.gz | tail -n +2 | wc -w
Verified 2026-11-11: VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz holds
168066656 values == count(14, 28, 14).
"""
import sys


def count(e1max: int, e2max: int, lmax: int) -> int:
    orb = []
    for e in range(e1max + 1):
        for l in range(e % 2, min(e, lmax) + 1, 2):
            for j2 in range(abs(2 * l - 1), 2 * l + 2, 2):
                orb.append((e, l, j2))
    n = 0
    for i1, (e1, l1, j1) in enumerate(orb):
        for i2 in range(i1 + 1):
            e2, l2, j2 = orb[i2]
            if e1 + e2 > e2max:
                break
            for i3 in range(i1 + 1):
                e3, l3, j3 = orb[i3]
                for i4 in range((i2 if i3 == i1 else i3) + 1):
                    e4, l4, j4 = orb[i4]
                    if e3 + e4 > e2max:
                        break
                    if (l1 + l2 + l3 + l4) % 2:
                        continue
                    jmin = max(abs(j1 - j2), abs(j3 - j4)) // 2
                    jmax = min(j1 + j2, j3 + j4) // 2
                    if jmin <= jmax:
                        n += 4 * (jmax - jmin + 1)
    return n


if __name__ == "__main__":
    e1, e2, lm = (int(x) for x in sys.argv[1:4])
    print(count(e1, e2, lm))
