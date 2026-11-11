#!/usr/bin/env python3
"""Exact Talmi-Moshinsky brackets and Talmi B coefficients (reference data for angmom tests).

Brackets <N L, n l; lam | n1 l1, n2 l2; lam> for equal masses are computed by an algorithm that
is independent of the Buck et al. formula used in the C++ code (and in imsrg++): harmonic-
oscillator states are polynomials in creation operators acting on the vacuum, the two-particle
and the center-of-mass/relative coupled states are built explicitly, and the bracket is their
overlap. Everything is exact rational arithmetic (fractions.Fraction): with highest-weight
coupled states (M = lam) built from the J+ = 0 condition, no square roots appear before the final
normalization, so bracket^2 is an exact rational and the sign is exact; the value written is the
double nearest to the exact bracket (50-digit decimal square root).

Representation: per particle the spherical creation operators u = (ax+ + i ay+)/sqrt2,
v = (ax+ - i ay+)/sqrt2, w = az+ (independent boson modes, so <u^a v^b w^c|u^a v^b w^c> = a! b! c!).
The solid harmonic r^l Y_lm (Condon-Shortley) is a positive multiple of
S_lm = sum_{p-q=m, p+q+s=l} (-1)^p 2^-q u^p v^q w^s / (p! q! s!), and J+ S_lm = (l+m+1)/sqrt2
S_l,m+1, which gives the highest-weight coupling coefficients as a rational recursion. The HO
state (n, l, m) is (2uv + w^2)^n S_lm |0>, normalized; that radial function is positive at large
r, so the conventional (positive at the origin) phase multiplies each state by (-1)^n.
Center of mass and relative motion: R = (r1 + r2)/sqrt2, r = (r1 - r2)/sqrt2, i.e. U = (u1+u2)/sqrt2,
u_r = (u1-u2)/sqrt2 for every mode.

Phase conventions (--convention): "ladder" as constructed; "origin" with radial functions
positive at the origin, (-1)^(N + n + n1 + n2) times ladder; "moshinsky" (default), the imsrg++
convention: origin times (-1)^(L + l + lam), i.e. Moshinsky-Brody's relative-first coupling
<n l, N L; lam|. Checked against AngMom::Moshinsky for all 216 nonzero brackets with E <= 4
(magnitudes agree to 2e-15; this phase rule is the only simple one that fits) and the samples in
tests/unit/angmom/data/moshinsky_imsrgpp_samples.txt.

Talmi B (Brody-Moshinsky 1967): exact rational sum times the square root of an exact rational.

Usage: gen_moshinsky_exact.py --emax-e 8 --out FILE [--talmi-nmax 6 --talmi-lmax 12]
Output lines: `N L n l n1 l1 n2 l2 lam value` and `talmib na la nb lb p value`; '#' comments.
Only the Python standard library is needed.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache

getcontext().prec = 50

Poly = dict  # exponent tuple -> Fraction


def mul3(p: Poly, q: Poly) -> Poly:
    """Product of two one-particle polynomials (keys (a, b, c) for u^a v^b w^c)."""
    out: Poly = defaultdict(Fraction)
    for (a1, b1, c1), x in p.items():
        for (a2, b2, c2), y in q.items():
            out[(a1 + a2, b1 + b2, c1 + c2)] += x * y
    return {k: v for k, v in out.items() if v != 0}


@lru_cache(maxsize=None)
def solid(l: int, m: int) -> Poly:
    """S_lm as defined in the module docstring."""
    out: Poly = {}
    for q in range(0, l + 1):
        p = q + m
        s = l - p - q
        if p < 0 or s < 0:
            continue
        out[(p, q, s)] = Fraction((-1) ** p, 2**q * math.factorial(p) * math.factorial(q) *
                                  math.factorial(s))
    return out


@lru_cache(maxsize=None)
def r2_power(n: int) -> Poly:
    """(2uv + w^2)^n."""
    base: Poly = {(1, 1, 0): Fraction(2), (0, 0, 2): Fraction(1)}
    out: Poly = {(0, 0, 0): Fraction(1)}
    for _ in range(n):
        out = mul3(out, base)
    return out


@lru_cache(maxsize=None)
def ho_state(n: int, l: int, m: int) -> Poly:
    """Unnormalized HO state (2uv + w^2)^n S_lm (radial function positive at large r)."""
    return mul3(r2_power(n), solid(l, m))


def weight(key) -> int:
    w = 1
    for e in key:
        w *= math.factorial(e)
    return w


def inner(p: Poly, q: Poly) -> Fraction:
    if len(p) > len(q):
        p, q = q, p
    s = Fraction(0)
    for k, x in p.items():
        y = q.get(k)
        if y is not None:
            s += x * y * weight(k)
    return s


def coupled(n1: int, l1: int, n2: int, l2: int, lam: int) -> Poly:
    """Unnormalized highest-weight (M = lam) state of (n1 l1, n2 l2) coupled to lam; the
    coefficient of m1 = l1 is positive (Condon-Shortley). Keys: (a1, b1, c1, a2, b2, c2)."""
    coeff = {l1: Fraction(1)}
    k = l1
    while k - 1 >= max(-l1, lam - l2):
        coeff[k - 1] = -coeff[k] * Fraction(l2 + lam - k + 1, l1 + k)
        k -= 1
    out: Poly = defaultdict(Fraction)
    for m1, x in coeff.items():
        m2 = lam - m1
        if abs(m2) > l2:
            continue
        p1 = ho_state(n1, l1, m1)
        p2 = ho_state(n2, l2, m2)
        for k1, y1 in p1.items():
            for k2, y2 in p2.items():
                out[k1 + k2] += x * y1 * y2
    return {k: v for k, v in out.items() if v != 0}


@lru_cache(maxsize=None)
def mix(a: int, b: int):
    """Coefficients of x1^k x2^(a+b-k) in (x1 + x2)^a (x1 - x2)^b."""
    out = [0] * (a + b + 1)
    for i in range(a + 1):
        ci = math.comb(a, i)
        for j in range(b + 1):
            out[i + j] += ci * math.comb(b, j) * (-1) ** (b - j)
    return tuple(out)


def cm_rel_to_lab(phi: Poly) -> Poly:
    """Substitutes U = u1 + u2, u_r = u1 - u2 (same for v, w) in a state whose keys are
    (U, V, W, u_r, v_r, w_r) exponents; the common factor 2^(-E/2) is dropped (it cancels in the
    normalized overlap, where all norms are taken in the same representation)."""
    out: Poly = defaultdict(Fraction)
    for (au, av, aw, bu, bv, bw), c in phi.items():
        mu, mv, mw = mix(au, bu), mix(av, bv), mix(aw, bw)
        for ku, cu in enumerate(mu):
            if cu == 0:
                continue
            for kv, cv in enumerate(mv):
                if cv == 0:
                    continue
                cuv = c * cu * cv
                for kw, cw in enumerate(mw):
                    if cw == 0:
                        continue
                    out[(ku, kv, kw, au + bu - ku, av + bv - kv, aw + bw - kw)] += cuv * cw
    return {k: v for k, v in out.items() if v != 0}


def ho_tuples(e: int):
    """All (n1, l1, n2, l2) with 2 n1 + l1 + 2 n2 + l2 = e."""
    out = []
    for e1 in range(e + 1):
        for n1 in range(e1 // 2 + 1):
            l1 = e1 - 2 * n1
            e2 = e - e1
            for n2 in range(e2 // 2 + 1):
                out.append((n1, l1, n2, e2 - 2 * n2))
    return out


def signed_sqrt(num: Fraction, sq: Fraction) -> float:
    """sign(num) * sqrt(sq) as the nearest double (sq >= 0)."""
    if sq == 0:
        return 0.0
    v = (Decimal(sq.numerator) / Decimal(sq.denominator)).sqrt()
    return float(v) if num > 0 else -float(v)


def block(task):
    """All brackets of one (E, lam) block: (N, L, n, l, n1, l1, n2, l2, lam, value) for every
    parity- and triangle-allowed pair of states (exact zeros included)."""
    e, lam, convention = task
    rows = []
    labs = [t for t in ho_tuples(e) if abs(t[1] - t[3]) <= lam <= t[1] + t[3]]
    lab_states = {t: coupled(t[0], t[1], t[2], t[3], lam) for t in labs}
    lab_norm = {t: inner(s, s) for t, s in lab_states.items()}
    for cm in labs:  # same tuple set for (N, L, n, l)
        N, L, n, l = cm
        phi = cm_rel_to_lab(coupled(N, L, n, l, lam))
        phi_norm = inner(phi, phi)
        for t in labs:
            n1, l1, n2, l2 = t
            if (L + l - l1 - l2) % 2 != 0:
                continue
            ov = inner(phi, lab_states[t])
            value = signed_sqrt(ov, ov * ov / (phi_norm * lab_norm[t]))
            sign = 1
            if convention in ("moshinsky", "origin"):
                sign *= (-1) ** (N + n + n1 + n2)  # radial functions positive at r = 0
            if convention == "moshinsky":
                # Moshinsky-Brody couple the relative motion first, <n l, N L; lam|, which is
                # (-1)^(L + l - lam) times our (N L) (n l) order. Determined (uniquely among
                # simple phase rules) against imsrg++ for all brackets with E <= 4.
                sign *= (-1) ** (L + l + lam)
            rows.append((N, L, n, l, n1, l1, n2, l2, lam, sign * value))
    return rows


def brackets(emax_e: int, convention: str, processes: int = 1):
    """Yields the rows of `block` for every E <= emax_e and lam <= E, in a fixed order."""
    tasks = [(e, lam, convention) for e in range(emax_e + 1) for lam in range(e + 1)]
    if processes <= 1:
        for t in tasks:
            yield from block(t)
        return
    import multiprocessing

    with multiprocessing.Pool(processes) as pool:
        for rows in pool.imap(block, tasks, chunksize=1):
            yield from rows


def talmi_b(na: int, la: int, nb: int, lb: int, p: int) -> float:
    """Exact Brody-Moshinsky Talmi coefficient B(na la, nb lb; p) as the nearest double."""
    if (la + lb) % 2:
        return 0.0
    q = (la + lb) // 2
    f = math.factorial
    s = Fraction(0)
    for k in range(max(0, p - q - nb), min(na, p - q) + 1):
        s += Fraction(f(la + k) * f(p - (la - lb) // 2 - k),
                      f(k) * f(2 * la + 2 * k + 1) * f(na - k) * f(2 * p - la + lb - 2 * k + 1) *
                      f(nb - p + q + k) * f(p - q - k))
    if s == 0:
        return 0.0
    pref = Fraction(f(2 * p + 1), 2 ** (na + nb) * f(p)) * s * (-1) ** (p - q)
    rad = Fraction(f(na) * f(nb) * f(2 * na + 2 * la + 1) * f(2 * nb + 2 * lb + 1),
                   f(na + la) * f(nb + lb))
    return signed_sqrt(pref, pref * pref * rad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emax-e", type=int, default=8, help="max HO energy E of the brackets")
    ap.add_argument("--talmi-nmax", type=int, default=6)
    ap.add_argument("--talmi-lmax", type=int, default=12)
    ap.add_argument("--convention", choices=["moshinsky", "origin", "ladder"], default="moshinsky")
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=0, help="worker processes (0: all CPUs)")
    args = ap.parse_args()
    if args.processes <= 0:
        import os

        args.processes = os.cpu_count() or 1
    t0 = time.time()
    nb = nt = 0
    with open(args.out, "w") as fh:
        fh.write(f"# exact brackets, E <= {args.emax_e}, convention {args.convention}; "
                 f"generated by {sys.argv[0]}\n")
        for row in brackets(args.emax_e, args.convention, args.processes):
            fh.write(" ".join(str(x) for x in row[:9]) + f" {row[9]!r}\n")
            nb += 1
        for na in range(args.talmi_nmax + 1):
            for nbq in range(args.talmi_nmax + 1):
                for la in range(args.talmi_lmax + 1):
                    for lb in range(la % 2, args.talmi_lmax + 1, 2):
                        q = (la + lb) // 2
                        for p in range(q, q + na + nbq + 1):
                            fh.write(f"talmib {na} {la} {nbq} {lb} {p} "
                                     f"{talmi_b(na, la, nbq, lb, p)!r}\n")
                            nt += 1
    print(f"gen_moshinsky_exact: {nb} brackets, {nt} Talmi B values in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
