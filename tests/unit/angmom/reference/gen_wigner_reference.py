#!/usr/bin/env python3
"""Generates exact (sympy) reference values of Wigner symbols for tests/unit/angmom.

Usage: gen_wigner_reference.py --emax 8 --out FILE [--seed N] [--jobs N]

Every symbol is evaluated with sympy.physics.wigner on exact Rational arguments and converted to
the nearest double (float(value.evalf(40))). Output: one line per symbol,
    <family> <doubled args...> <value repr>
Families (sp = single-particle 2j in {1,3,...,2*emax+1}, l in [0,emax], 2B J in [0,2*emax+1]):
  cg_sp cg_ls cg_l0 3j_sp 6j_pandya 6j_3b 9j_lsjj 9j_sample
All arguments are doubled integers (2j, 2m). The 6j and 3j symbols are canonicalized under their
symmetry groups (24 and 12 elements) so that sympy is called once per symmetry class; the CG
coefficients are evaluated through sympy's clebsch_gordan once per exact argument set.
"""
import argparse
import itertools
import multiprocessing as mp
import random
import sys
import time

from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j


def h(t):
    return Rational(t, 2)


def tri(a, b, c):
    """Triangle condition for doubled arguments."""
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def to_float(v):
    return float(v.evalf(40))


# ---- canonical forms -------------------------------------------------------------------------

def canon_3j(j, m):
    """Returns (key, sign) with 3j(j; m) = sign * 3j(key); key = (j1 j2 j3 m1 m2 m3)."""
    s = -1 if ((j[0] + j[1] + j[2]) // 2) % 2 else 1
    best = None
    for perm in itertools.permutations(range(3)):
        odd = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b]) % 2
        for flip in (1, -1):
            jj = tuple(j[p] for p in perm)
            mm = tuple(flip * m[p] for p in perm)
            sign = 1
            if odd:
                sign *= s
            if flip == -1:
                sign *= s
            cand = (jj + mm, sign)
            if best is None or cand[0] < best[0]:
                best = cand
    return best


_COLPERM = list(itertools.permutations(range(3)))


def canon_6j(a):
    """Minimal representative of the 24 symmetry-equivalent arrangements of a 6j symbol."""
    top, bot = a[:3], a[3:]
    best = None
    for p in _COLPERM:
        t = [top[i] for i in p]
        b = [bot[i] for i in p]
        for swap in ((), (0, 1), (0, 2), (1, 2)):
            tt, bb = list(t), list(b)
            for i in swap:
                tt[i], bb[i] = bb[i], tt[i]
            cand = tuple(tt + bb)
            if best is None or cand < best:
                best = cand
    return best


# ---- workers ----------------------------------------------------------------------------------

def w3j(k):
    return to_float(wigner_3j(*[h(t) for t in k]))


def w6j(k):
    return to_float(wigner_6j(*[h(t) for t in k]))


def w9j(k):
    return to_float(wigner_9j(*[h(t) for t in k], prec=None))


def wcg(k):
    # key = (2j1, 2m1, 2j2, 2m2, 2J, 2M); sympy's order is (j1, j2, J, m1, m2, M).
    a, ma, b, mb, J, M = [h(t) for t in k]
    return to_float(clebsch_gordan(a, b, J, ma, mb, M))


def run_pool(pool, fn, keys):
    keys = sorted(keys)
    chunk = max(1, len(keys) // (pool._processes * 16))
    return dict(zip(keys, pool.imap(fn, keys, chunksize=chunk)))


# ---- enumeration ------------------------------------------------------------------------------

def mrange(tj):
    return range(-tj, tj + 1, 2)


def enum_cg(ja_list, jb_list, J_list):
    """All valid <ja ma jb mb | J M>: yields (ta, ma, tb, mb, tJ, tM), all doubled."""
    for ta in ja_list:
        for tb in jb_list:
            for tJ in J_list:
                if not tri(ta, tb, tJ):
                    continue
                for ma in mrange(ta):
                    for mb in mrange(tb):
                        M = ma + mb
                        if abs(M) <= tJ and (tJ - M) % 2 == 0:
                            yield (ta, ma, tb, mb, tJ, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emax", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20240607)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    e = args.emax
    sp = list(range(1, 2 * e + 2, 2))  # 2j of single-particle orbits
    J2B = [2 * J for J in range(0, 2 * e + 2)]  # 2B J integer 0..2emax+1, doubled
    fams = {}

    fams["cg_sp"] = list(enum_cg(sp, sp, J2B))
    # LS coupling <l ml 1/2 ms | j mj>, j = l +- 1/2
    ls = []
    for l in range(0, e + 1):
        for tj in (2 * l - 1, 2 * l + 1):
            if tj < 0:
                continue
            for ml in range(-l, l + 1):
                for ms in (-1, 1):
                    M = 2 * ml + ms
                    if abs(M) <= tj:
                        ls.append((2 * l, 2 * ml, 1, ms, tj, M))
    fams["cg_ls"] = ls
    l0 = []
    for la in range(0, 2 * e + 1):
        for lb in range(0, 2 * e + 1):
            for L in range(abs(la - lb), la + lb + 1):
                if (la + lb + L) % 2:
                    continue  # vanishes by parity
                l0.append((2 * la, 0, 2 * lb, 0, 2 * L, 0))
    fams["cg_l0"] = l0
    # 3j for the cg_sp arguments: (ja jb J; ma mb -M)
    fams["3j_sp"] = [(a, b, J, ma, mb, -M) for (a, ma, b, mb, J, M) in fams["cg_sp"]]
    p = []
    for ja, jb, jc, jd in itertools.product(sp, repeat=4):
        for J in J2B:
            if not tri(ja, jb, J):
                continue
            for Jp in J2B:
                # {ja jb J; jc jd J'}: triads (ja jb J)(ja jd J')(jc jb J')(jc jd J)
                if tri(ja, jd, Jp) and tri(jc, jb, Jp) and tri(jc, jd, J):
                    p.append((ja, jb, J, jc, jd, Jp))
    fams["6j_pandya"] = p
    q = []
    Jodd = list(range(1, 3 * (2 * e + 1) + 1, 2))
    for ja, jb, jc in itertools.product(sp, repeat=3):
        for Jab in J2B:
            if not tri(ja, jb, Jab):
                continue
            for Jbc in J2B:
                if not tri(jc, jb, Jbc):
                    continue
                for J in Jodd:
                    # {ja jb Jab; jc J Jbc}: (ja jb Jab)(ja J Jbc)(jc jb Jbc)(jc J Jab)
                    if tri(ja, J, Jbc) and tri(jc, J, Jab):
                        q.append((ja, jb, Jab, jc, J, Jbc))
    fams["6j_3b"] = q
    nj = []
    for la in range(0, e + 1):
        for lb in range(0, e + 1):
            for ja in (2 * la - 1, 2 * la + 1):
                for jb in (2 * lb - 1, 2 * lb + 1):
                    if ja < 0 or jb < 0:
                        continue
                    for L in range(abs(la - lb), la + lb + 1):
                        for S in (0, 2):
                            for J in J2B:
                                # {la 1/2 ja; lb 1/2 jb; L S J}
                                if (tri(2 * la, 1, ja) and tri(2 * lb, 1, jb) and tri(ja, jb, J)
                                        and tri(2 * L, S, J) and tri(2 * la, 2 * lb, 2 * L)
                                        and tri(1, 1, S)):
                                    nj.append((2 * la, 1, ja, 2 * lb, 1, jb, 2 * L, S, J))
    fams["9j_lsjj"] = nj
    rng = random.Random(args.seed)
    samp = set()
    while len(samp) < 3000:
        ja, jb, jc, jd = (rng.choice(sp) for _ in range(4))
        Jab, Jcd, Jac, Jbd, J = (rng.choice(J2B) for _ in range(5))
        # {ja jb Jab; jc jd Jcd; Jac Jbd J}
        if (tri(ja, jb, Jab) and tri(jc, jd, Jcd) and tri(Jac, Jbd, J) and tri(ja, jc, Jac)
                and tri(jb, jd, Jbd) and tri(Jab, Jcd, J)):
            samp.add((ja, jb, Jab, jc, jd, Jcd, Jac, Jbd, J))
    fams["9j_sample"] = sorted(samp)
    print("enumerated in %.1f s" % (time.time() - t0), file=sys.stderr)

    keys3, keys6 = set(), set()
    for k in fams["3j_sp"]:
        keys3.add(canon_3j(k[:3], k[3:])[0])
    for fam in ("6j_pandya", "6j_3b"):
        for k in fams[fam]:
            keys6.add(canon_6j(k))
    keys9 = set(fams["9j_lsjj"]) | set(fams["9j_sample"])
    keys_cg = set(fams["cg_sp"]) | set(fams["cg_ls"]) | set(fams["cg_l0"])

    jobs = args.jobs or mp.cpu_count()
    with mp.Pool(jobs) as pool:
        v3 = run_pool(pool, w3j, keys3)
        print("3j done %.1f s" % (time.time() - t0), file=sys.stderr)
        v6 = run_pool(pool, w6j, keys6)
        print("6j done %.1f s" % (time.time() - t0), file=sys.stderr)
        v9 = run_pool(pool, w9j, keys9)
        print("9j done %.1f s" % (time.time() - t0), file=sys.stderr)
        vcg = run_pool(pool, wcg, keys_cg)
        print("cg done %.1f s" % (time.time() - t0), file=sys.stderr)

    with open(args.out, "w") as f:
        for name in ("cg_sp", "cg_ls", "cg_l0"):
            for k in fams[name]:
                f.write("%s %s %r\n" % (name, " ".join(map(str, k)), vcg[k]))
        for k in fams["3j_sp"]:
            key, sign = canon_3j(k[:3], k[3:])
            f.write("3j_sp %s %r\n" % (" ".join(map(str, k)), sign * v3[key]))
        for name in ("6j_pandya", "6j_3b"):
            for k in fams[name]:
                f.write("%s %s %r\n" % (name, " ".join(map(str, k)), v6[canon_6j(k)]))
        for name in ("9j_lsjj", "9j_sample"):
            for k in fams[name]:
                f.write("%s %s %r\n" % (name, " ".join(map(str, k)), v9[k]))
    print("counts: " + ", ".join("%s=%d" % (n, len(fams[n])) for n in fams))
    print("unique sympy calls: 3j=%d 6j=%d 9j=%d cg=%d" % (len(v3), len(v6), len(v9), len(vcg)))
    print("generation time: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
