#!/usr/bin/env python3
"""Brute-force high-precision reference for the NREFT one-body tables (design/50 C2).

The fork's DM_NREFT formulas (DarkMatterNREFT.cc at 79e9330; reference/06 section 2.2) evaluated
term by term with mpmath at 50 digits and exact Wigner symbols (sympy), so the result is the exact
value of the formula for the double-precision y = q*q*(1/(hw*m_N))/4 the codes use. Needed because
in double precision the alternating radial sums lose up to 8 digits at emax 14: the fork's own
tables are off by up to ~1e-6 relative there (design/30 section 6), so they cannot validate the
port at the production basis size.

Writes tests/golden/nreft/mpref_emax14.txt (D42: regenerate only with this script):
  orbit <i> <n> <l> <j2> <tz2>                     (the orbits used, imsrg++ index at emax 14)
  ref <shorthand name> <a> <b> <value>             (<a||O||b>, shorthand prefactors included)
Sample: every shorthand family and L (FM, FPhipp, FPhitp L = 0/2/4 resp. 2/4; FSigmap, FSigmapp,
FDelta L = 1, 3, 5; protons), q = 15, 150, 600, 1050 MeV, and up to 12 orbit pairs per table with
shells e >= 10 (high radial n, where the sums cancel most), chosen deterministically.
Usage: .venv/bin/python -I tools/reference/nreft_mpref.py [OUT]
"""
import sys
import mpmath as mp
from functools import lru_cache
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
from sympy import Rational as R
mp.mp.dps = 50
H = R(1, 2)

def mpf(x): return mp.mpf(str(x.evalf(60))) if x != 0 else mp.mpf(0)
@lru_cache(None)
def w3(a, b, c): return mpf(wigner_3j(a, b, c, 0, 0, 0))
@lru_cache(None)
def w6(*a): return mpf(wigner_6j(*a))
@lru_cache(None)
def w9(*a): return mpf(wigner_9j(*a, prec=60))

def yof(q, hw=16.0):
    b2 = 1.0 / (hw * ((938.2720813 + 939.5654133) / 2.0))
    return mp.mpf(q * q * b2 / 4.0)

def G(x): return mp.gamma(x)
def common(na, la, nb, lb):
    return mp.sqrt(mp.factorial(nb-1)*mp.factorial(na-1)*G(na+la+mp.mpf(1)/2)*G(nb+lb+mp.mpf(1)/2))
def jho(na0, la, nb0, lb, L, y):
    na, nb = na0+1, nb0+1
    f1 = mp.mpf(2)**L/mp.fac2(2*L+1)*y**(mp.mpf(L)/2)*mp.e**(-y)*common(na, la, nb, lb)
    s = 0
    for k in range(nb):
        f2 = (-1)**k/mp.factorial(k)/mp.factorial(nb-1-k)/G(lb+k+mp.mpf(3)/2)
        f3 = 0
        for k1 in range(na):
            f3 += (-1)**k1/mp.factorial(k1)/mp.factorial(na-1-k1)*G(mp.mpf(L+lb+la+2*k+2*k1+3)/2)/G(la+k1+mp.mpf(3)/2)*mp.hyp1f1(mp.mpf(L-lb-la-2*k-2*k1)/2, L+mp.mpf(3)/2, y)
        s += f2*f3
    return f1*s
def jd(na0, la, nb0, lb, L, y, plus):
    na, nb = na0+1, nb0+1
    f1 = mp.mpf(2)**(L-1)/mp.fac2(2*L+1)*y**(mp.mpf(L-1)/2)*mp.e**(-y)*common(na, la, nb, lb)
    s = 0
    for k in range(nb):
        f2 = (-1)**k/mp.factorial(k)/mp.factorial(nb-1-k)/G(lb+k+mp.mpf(3)/2)
        f3 = 0
        for k1 in range(na):
            sec = (2*lb+2*k+1) if plus else 2*k
            f3 += (-1)**k1/mp.factorial(k1)/mp.factorial(na-1-k1)*G(mp.mpf(L+lb+la+2*k+2*k1+2)/2)/G(la+k1+mp.mpf(3)/2)*(
                -mp.mpf(L+lb+la+2*k+2*k1+2)/2*mp.hyp1f1(mp.mpf(L-lb-la-2*k-2*k1-1)/2, L+mp.mpf(3)/2, y)
                + sec*mp.hyp1f1(mp.mpf(L-lb-la-2*k-2*k1+1)/2, L+mp.mpf(3)/2, y))
        s += f2*f3
    return f1*s

def red0(v, J, par, j2a):
    return v/mp.sqrt(j2a+1) if (J == 0 and par == 0) else v
def M(J, a, b, y):
    (na, la, ja), (nb, lb, jb) = a, b
    v = 1/mp.sqrt(4*mp.pi)*(-1)**((2*J+jb+1)//2)*mp.sqrt((2*la+1)*(2*lb+1)*(ja+1)*(jb+1)*(2*J+1))*w6(la, R(ja, 2), H, R(jb, 2), lb, J)*w3(la, J, lb)*jho(na, la, nb, lb, J, y)
    return red0(v, J, J % 2, ja)
def Ms(J, L, a, b, y):
    (na, la, ja), (nb, lb, jb) = a, b
    if (la+lb+L) % 2: return mp.mpf(0)
    v = mp.sqrt(mp.mpf(3)/2/mp.pi)*(-1)**la*mp.sqrt((2*la+1)*(2*lb+1)*(ja+1)*(jb+1)*(2*L+1)*(2*J+1))*w9(la, lb, L, H, H, 1, R(ja, 2), R(jb, 2), J)*w3(la, L, lb)*jho(na, la, nb, lb, L, y)
    return red0(v, J, L % 2, ja)
def Mg(J, L, a, b, y):
    (na, la, ja), (nb, lb, jb) = a, b
    if la == 0 or lb == 0 or (la+lb+L) % 2 == 0: return mp.mpf(0)
    v = 1/mp.sqrt(4*mp.pi)*(-1)**((2*L+jb+1)//2)*mp.sqrt((2*la+1)*(ja+1)*(jb+1)*(2*L+1)*(2*J+1))*w6(la, R(ja, 2), H, R(jb, 2), lb, J)*(
        -mp.sqrt(lb+1)*mp.sqrt(2*lb+3)*w6(L, 1, J, lb, la, lb+1)*w3(la, L, lb+1)*jd(na, la, nb, lb, L, y, False)
        + mp.sqrt(lb)*mp.sqrt(2*lb-1)*w6(L, 1, J, lb, la, lb-1)*w3(la, L, lb-1)*jd(na, la, nb, lb, L, y, True))
    return red0(v, J, (L+1) % 2, ja)
def PhiS(a, b, J, K, plus_l, y):
    (na, la, ja), (nb, lb, jb) = a, b
    lb2 = lb+1 if plus_l else lb-1
    if lb2 < 0 or K < 0: return mp.mpf(0)
    lo = J if K > J else J-1
    s = 0
    for lam in (lo, lo+1):
        s += (-1)**(J-lam+(1 if plus_l else 0))*(2*lam+1)*w6(K, 1, lam, 1, J, 1)*w6(K, 1, lam, lb, la, lb2)*w9(la, lb, lam, H, H, 1, R(ja, 2), R(jb, 2), J)
    rest = mp.sqrt(2*lb+3)*mp.sqrt(lb+1) if plus_l else mp.sqrt(2*lb-1)*mp.sqrt(lb)
    rad = jd(na, la, nb, lb, K, y, not plus_l)
    return s*rest*w3(la, K, lb2)*rad
def Phi(J, a, b, y, pp):
    (na, la, ja), (nb, lb, jb) = a, b
    F = (-1)**(la+1)*6*mp.sqrt((2*la+1)*(ja+1)*(jb+1))/mp.sqrt(4*mp.pi)
    s12 = PhiS(a, b, J, J+1, True, y) + PhiS(a, b, J, J+1, False, y)
    s34 = (PhiS(a, b, J, J-1, True, y) + PhiS(a, b, J, J-1, False, y)) if J > 0 else 0
    if pp:
        v = F*(mp.sqrt(2*J+3)*mp.sqrt(J+1)*s12) + (F*mp.sqrt(2*J-1)*mp.sqrt(J)*s34 if J > 0 else 0)
    else:
        v = F*(-mp.sqrt(2*J+3)*mp.sqrt(J)*s12) + (F*mp.sqrt(2*J-1)*mp.sqrt(J+1)*s34 if J > 0 else 0)
    return red0(v, J, J % 2, ja)
def family(fam, J, a, b, y):
    if fam == 'FM': return mp.sqrt(4*mp.pi)*M(J, a, b, y)
    if fam == 'FPhipp': return mp.sqrt(4*mp.pi)*Phi(J, a, b, y, True)
    if fam == 'FPhitp': return mp.sqrt(4*mp.pi)*(Phi(J, a, b, y, False)+Ms(J, J, a, b, y)/2)
    if fam == 'FSigmap': return -mp.sqrt(mp.mpf(J)/(2*J+1))*Ms(J, J+1, a, b, y) + mp.sqrt(mp.mpf(J+1)/(2*J+1))*Ms(J, J-1, a, b, y)
    if fam == 'FSigmapp': return mp.sqrt(mp.mpf(J+1)/(2*J+1))*Ms(J, J+1, a, b, y) + mp.sqrt(mp.mpf(J)/(2*J+1))*Ms(J, J-1, a, b, y)
    if fam == 'FDelta': return Mg(J, J, a, b, y)

def orbits_emax(emax):
    orb = []
    for N in range(emax + 1):
        for l in range(N, -1, -2):
            n = (N - l) // 2
            for j2 in (2 * l + 1, 2 * l - 1):
                if j2 < 1:
                    continue
                for tz in (-1, 1):
                    orb.append((n, l, j2, tz))
    return orb


def in_channel(J, parity, a, b):
    if a[3] != b[3] or (a[1] + b[1] + parity) % 2 or abs(a[1] - b[1]) > J + 1:
        return False
    return abs(a[2] - b[2]) <= 2 * J


FAMILIES = [('FM', [0, 2, 4], 0), ('FPhipp', [0, 2, 4], 0), ('FPhitp', [2, 4], 0),
            ('FSigmap', [1, 3, 5], 1), ('FSigmapp', [1, 3, 5], 1), ('FDelta', [1, 3, 5], 1)]


if __name__ == '__main__':
    import os
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'golden', 'nreft',
        'mpref_emax14.txt')
    orb = orbits_emax(14)
    high = [i for i, o in enumerate(orb) if o[3] == -1 and 2 * o[0] + o[1] >= 10]
    lines, used = [], set()
    for fam, Ls, abnormal in FAMILIES:
        for L in Ls:
            parity = (L + abnormal) % 2  # operator parity code: always 0 here
            pairs = [(a, b) for a in high for b in high if in_channel(L, parity, orb[a], orb[b])]
            pick = pairs[::max(1, len(pairs) // 12)][:12]
            for q in (15, 150, 600, 1050):
                for a, b in pick:
                    oa, ob = orb[a][:3], orb[b][:3]
                    v = family(fam, L, oa, ob, yof(float(q)))
                    lines.append('ref %s%dp_%d %d %d %s' % (fam, L, q, a, b, mp.nstr(v, 20, min_fixed=1, max_fixed=0)))
                    used.update((a, b))
    with open(out, 'w') as f:
        f.write('# nreft_mpref.py: mpmath 50 digits, fork 79e9330 formulas, hw 16, emax 14\n')
        for i in sorted(used):
            f.write('orbit %d %d %d %d %d\n' % ((i,) + orb[i]))
        f.write('\n'.join(lines) + '\n')
    print('wrote', out, len(lines), 'elements')
