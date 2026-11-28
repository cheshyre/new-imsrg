#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Relative differences of the dominant elements of two 27Al farms.

  dominant_rel.py --ref-farm D --test-farm D [--part all|1b|2b]
      [--small 1e-6] [--dom-frac 0.01] [--qs 0,300,600,900] [--jobs N]
      [--md out.md]

Per operator column (all q pooled): an element of a file is *dominant* if
|ref| >= max(--dom-frac * (largest |ref| of the file), --small), so the
owner's rule (relative differences only for |ref| >= 1e-6) is never
violated. Reported per column: number of dominant elements, median, 90th
percentile and maximum of |test - ref| / |ref|; the largest |test - ref| over
the largest |ref| of a file (and its q); the median relative difference of
the dominant elements at the q values of --qs.

An operator that vanishes in REF (no element with |ref| >= --small, e.g.
every L > 0 multipole at q = 0) has no relative difference: it is reported
as "vanishes" (with the largest |test| if that is >= --small), never as 0.
Parts: `all` = one- and two-body elements (the zero-body part is excluded),
`1b` or `2b` restrict the dominant elements and the max-abs ratio to that
part. Complements diff_variants.py (which counts all elements).
"""

import argparse
import multiprocessing
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_farm as cf  # noqa: E402

_CS = None


def _cs():
    """compare_snt module (loaded once per process)."""
    global _CS
    if _CS is None:
        _CS = cf._load_compare_snt()
    return _CS


def load_elements(path):
    """Return {(part, *key): value} of the 1b and 2b elements of a file."""
    s = cf.parse_snt(path)
    if s.scalar:
        cf.canonicalize(s, _cs())
    d = {("1b",) + tuple(k): v for k, v in s.one.items()}
    d.update({("2b",) + tuple(k): v for k, v in s.two.items()})
    return d


def dominant_stats(ref, test, small, dom_frac, part="all"):
    """Statistics of one file pair; ref, test are {key: value} dicts.

    Keys are tuples whose first item is the part ("1b" or "2b"). Returns
    {"rels": [...], "vanishes": bool, "max_ref", "max_abs", "test_max"}:
    rels = |test - ref| / |ref| of the dominant elements (|ref| >= small and
    >= dom_frac * max_ref); max_ref and max_abs run over the selected part
    (all elements of ref and of ref union test); vanishes = no element of
    ref with |ref| >= small (then rels is empty and test_max is the largest
    |test| of the part).
    """
    sel = (lambda k: True) if part == "all" else (lambda k: k[0] == part)
    r = {k: v for k, v in ref.items() if sel(k)}
    t = {k: v for k, v in test.items() if sel(k)}
    max_ref = max((abs(v) for v in r.values()), default=0.0)
    test_max = max((abs(v) for v in t.values()), default=0.0)
    out = {"rels": [], "vanishes": max_ref < small, "max_ref": max_ref,
           "max_abs": 0.0, "test_max": test_max}
    if out["vanishes"]:
        out["max_abs"] = test_max
        return out
    cut = max(dom_frac * max_ref, small)
    out["rels"] = [abs(t.get(k, 0.0) - v) / abs(v) for k, v in r.items()
                   if abs(v) >= cut]
    out["max_abs"] = max(abs(t.get(k, 0.0) - r.get(k, 0.0))
                         for k in set(r) | set(t))
    return out


def _work(task):
    """Worker: one file pair -> (col, q, dominant_stats)."""
    col, q, a, b, small, dom_frac, part = task
    return col, q, dominant_stats(load_elements(a), load_elements(b), small,
                                  dom_frac, part)


def summarize(results):
    """Per column summary of [(col, q, stats)]: {col: dict}."""
    cols = {}
    for col, q, st in results:
        c = cols.setdefault(col, {"rels": [], "ratio": 0.0, "ratio_q": None,
                                  "byq": {}, "n_files": 0, "n_vanish": 0})
        c["n_files"] += 1
        c["rels"] += st["rels"]
        if st["vanishes"]:
            c["n_vanish"] += 1
            c["byq"][q] = ("vanishes", st["test_max"])
            continue
        c["byq"][q] = ("median", statistics.median(st["rels"]))
        ratio = st["max_abs"] / st["max_ref"]
        if ratio > c["ratio"]:
            c["ratio"], c["ratio_q"] = ratio, q
    return cols


def fmt_byq(c, q, small):
    """Cell text of the median at q ("vanishes" if the operator does)."""
    if q not in c["byq"]:
        return "n/a"
    kind, val = c["byq"][q]
    if kind == "median":
        return f"{val:.1e}"
    return ("vanishes" if val < small
            else f"vanishes in ref (test max {val:.1e})")


def table(cols, qs, small):
    """Markdown lines of the per-column table."""
    qtxt = " / ".join(str(q) for q in qs)
    lines = ["| column | files | dominant elements | median rel | 90% rel | "
             "max rel | max abs / max ref (q) | "
             f"median rel at q = {qtxt} |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for col in sorted(cols, key=lambda c: (cf.op_sort_key(c), c)):
        c = cols[col]
        r = sorted(c["rels"])
        qcells = " / ".join(fmt_byq(c, q, small) for q in qs)
        if not r:
            lines.append(f"| {col} | {c['n_files']} | 0 | vanishes at all "
                         f"{c['n_vanish']} q | | | | {qcells} |")
            continue
        lines.append(
            f"| {col} | {c['n_files']} | {len(r)} | "
            f"{statistics.median(r):.1e} | "
            f"{r[int(0.9 * (len(r) - 1))]:.1e} | {r[-1]:.1e} | "
            f"{c['ratio']:.1e} ({c['ratio_q']}) | {qcells} |")
    return lines


def parse_args(argv):
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-farm", required=True)
    ap.add_argument("--test-farm", required=True)
    ap.add_argument("--part", choices=("all", "1b", "2b"), default="all")
    ap.add_argument("--small", type=float, default=1.0e-6)
    ap.add_argument("--dom-frac", type=float, default=0.01)
    ap.add_argument("--qs", default="0,300,600,900")
    ap.add_argument("--jobs", type=int,
                    default=min(16, os.cpu_count() or 1))
    ap.add_argument("--md", help="write markdown here (default stdout)")
    return ap.parse_args(argv)


def run(argv=None):
    """Entry point; returns the exit status (2 if no file pair)."""
    args = parse_args(argv)
    qs = [int(x) for x in args.qs.split(",") if x]
    ra, _ = cf.scan_flat(args.ref_farm)
    rb, _ = cf.scan_flat(args.test_farm)
    tasks = [(k[0], k[1], ra[k], rb[k], args.small, args.dom_frac,
              args.part) for k in sorted(ra, key=lambda k: (k[0], k[1] or 0))
             if k in rb]
    if not tasks:
        print("no common (operator, q) file pairs", file=sys.stderr)
        return 2
    if args.jobs > 1 and len(tasks) > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            res = pool.map(_work, tasks, chunksize=8)
    else:
        res = [_work(t) for t in tasks]
    text = "\n".join(table(summarize(res), qs, args.small)) + "\n"
    if args.md:
        with open(args.md, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(run())
