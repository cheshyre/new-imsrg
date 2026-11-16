"""Compares the imsrg++ one-body reduced matrix elements of E2 and GamowTeller (He4, HO basis,
tests/golden/tensor/emax2_He4/HO) with the Edmonds closed forms (sympy 3j/6j). Documents the
finding of design/15-operator-dump-format.md, "Tensor reference dumps": imsrg++ = (-1)^{j_bra-j_ket}
times Edmonds for sigma and E2.

usage: <venv>/bin/python -I tools/reference/tensor_convention_check.py [tests/golden/tensor]
"""
import sys

from sympy import N, Rational as R, pi, sqrt
from sympy.physics.wigner import wigner_3j, wigner_6j

root = sys.argv[1] if len(sys.argv) > 1 else "tests/golden/tensor"


def one_body(path):
    v = {}
    for line in open(path):
        t = line.split()
        if t and t[0] == "1b":
            v[(int(t[1]), int(t[2]))] = float(t[3])
    return v


# He4 emax 2 orbit indices: 2 = p 0p3/2, 3 = n 0p3/2, 4 = p 0p1/2, 5 = n 0p1/2
half = R(1, 2)


def edmonds_E2_angular(jp, j, lam=2):
    """<l' j'||Y_lam||l j> up to the radial integral and parity factor (Edmonds 5.4.5)."""
    return (-1) ** int(jp - half) * sqrt((2 * jp + 1) * (2 * j + 1) * (2 * lam + 1) / (4 * pi)) * wigner_3j(
        jp, lam, j, -half, 0, half)


def edmonds_sigma(jp, j, l):
    return (-1) ** int(l + half + j + 1) * sqrt((2 * j + 1) * (2 * jp + 1)) * wigner_6j(half, jp, l, j, half, 1) * sqrt(6)


e2 = one_body(root + "/emax2_He4/HO/E2.tdump")
gt = one_body(root + "/emax2_He4/HO/GamowTeller.tdump")
j = {2: R(3, 2), 3: R(3, 2), 4: R(1, 2), 5: R(1, 2)}
print("E2: code / Edmonds-angular (a common radial factor <0p|r^2|0p> b^2 cancels in the ratio of ratios)")
r0 = e2[(2, 2)] / float(N(edmonds_E2_angular(j[2], j[2])))
for a, b in [(2, 2), (2, 4), (4, 2)]:  # (4,4): 1/2 -> 1/2 has no quadrupole element
    ratio = e2[(a, b)] / float(N(edmonds_E2_angular(j[a], j[b]))) / r0
    print("  (%d,%d) j_bra-j_ket=%s ratio %+.12f  (-1)^(j_bra-j_ket) = %+d" % (a, b, j[a] - j[b], ratio, (-1) ** int(j[a] - j[b])))
print("GamowTeller (p<->n): code / Edmonds")
for a, b in [(2, 3), (2, 5), (4, 3), (4, 5)]:
    ratio = gt[(a, b)] / float(N(edmonds_sigma(j[a], j[b], 1)))
    print("  (%d,%d) j_bra-j_ket=%s ratio %+.12f  (-1)^(j_bra-j_ket) = %+d" % (a, b, j[a] - j[b], ratio, (-1) ** int(j[a] - j[b])))
