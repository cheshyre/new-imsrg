#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Decomposition of the 27Al v2 - v1 operator differences by compat option.

  compat_decomp.py --none D --fork D [--only NAME=D ...] [--allbut NAME=D ...]
      [--v1 D] [--v2 D] [--compatop D] --label L [--md out.md] [--jobs N]
      [--top 10]

Every directory is a flat farm directory (compare_farm.scan_flat). The farms
--none, --fork, --only and --allbut transform the SAME Omegas (those of v2)
with run.compat = "none", the fork bundle, "none" plus one option switched on
(run.compat_enable) and the bundle minus one option (run.compat_disable).
Contributions, each a per-element difference in the direction v1 -> v2
(fork behaviour -> default behaviour):
  op         none - fork        the whole operator side
  X          none - only_X      fix X alone, at the default point
  X (loo)    allbut_X - fork    fix X alone, at the fork point (leave one out)
  resid      op - sum_X X       non-additivity (and writer rounding)
  flow       fork - v1          v1 -> v2 Omegas, both with fork behaviour
  total      v2 - v1
  v2-none    v2 - none          checks: same inputs, should vanish
  compatop-fork  compatop - fork
An element present in only one file counts as 0 there (as diff_variants.py).
Items are the (column, q) pairs present in --none and --fork. Output: a
markdown section with the per-column maxima of |contribution| (over the q of
the column), the per-rank shares ||X||_2 / ||op||_2 and the largest elements
of |op| with every contribution.
"""

import argparse
import heapq
import math
import multiprocessing
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_farm as cf  # noqa: E402

SHORT = {
    "tokyo_writer_fork_79e9330": "writer",
    "tensor_one_body_norm_imsrgpp": "norm (AL-A-Q6)",
    "tensor_no_fork_diagonal_bug": "NO diag (M15-1)",
    "tensor_ninej_threshold_1e8": "9j 1e-8",
    "tensor_fork_ph_missing_rank_phase": "ph phase (AL-F1)",
}
_CS = None


def _cs():
    """compare_snt module (loaded once per process)."""
    global _CS
    if _CS is None:
        _CS = cf._load_compare_snt()
    return _CS


def short(name):
    """Table header of a compat option (the name itself if unknown)."""
    return SHORT.get(name, name)


def load(path):
    """Return {(part, *key): value} with parts "0b", "1b", "2b"."""
    s = cf.parse_snt(path)
    if s.scalar:
        cf.canonicalize(s, _cs())
    d = {("0b",): s.zero}
    d.update({("1b",) + tuple(k): v for k, v in s.one.items()})
    d.update({("2b",) + tuple(k): v for k, v in s.two.items()})
    return d


def sub(a, b):
    """a - b over the union of the keys (missing = 0)."""
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in set(a) | set(b)}


def contributions(data, options):
    """{name: {key: value}} of every contribution available in `data`.

    data: {"none": d, "fork": d, ("only", X): d, ("allbut", X): d, "v1": d,
    "v2": d, "compatop": d} (only none and fork required); options: the
    option names in table order.
    """
    none, fork = data["none"], data["fork"]
    c = {"op": sub(none, fork)}
    have_all = True
    for x in options:
        if ("only", x) in data:
            c[x] = sub(none, data[("only", x)])
        else:
            have_all = False
        if ("allbut", x) in data:
            c[x + " (loo)"] = sub(data[("allbut", x)], fork)
    if have_all and options:
        resid = dict(c["op"])
        for x in options:
            for k, v in c[x].items():
                resid[k] = resid.get(k, 0.0) - v
        c["resid"] = resid
    if "v1" in data:
        c["flow"] = sub(fork, data["v1"])
        if "v2" in data:
            c["total"] = sub(data["v2"], data["v1"])
    if "v2" in data:
        c["v2-none"] = sub(data["v2"], none)
    if "compatop" in data:
        c["compatop-fork"] = sub(data["compatop"], fork)
    return c


def item_stats(task):
    """Worker: stats of one item; returns (col, q, stats, top) or an error."""
    col, q, paths, options, topk = task
    try:
        data = {tag: load(p) for tag, p in paths.items()}
    except (ValueError, OSError, IndexError) as e:
        return col, q, {"error": f"{type(e).__name__}: {e}"}, []
    c = contributions(data, options)
    stats = {}
    for name, d in c.items():
        mx, key, ss = 0.0, None, 0.0
        for k, v in d.items():
            a = abs(v)
            ss += a * a
            if a > mx or key is None:
                mx, key = a, k
        stats[name] = (mx, key, ss)
    top = heapq.nlargest(topk, c["op"].items(), key=lambda kv: abs(kv[1]))
    rows = [(abs(v), col, q, k, {n: c[n].get(k, 0.0) for n in c})
            for k, v in top]
    return col, q, stats, rows


def rank_of(col):
    """Tensor rank L of a column name F<family><L><p|n> (None if no match)."""
    m = re.fullmatch(r"F[A-Z][a-z]*([0-9])[pn]", col)
    return int(m.group(1)) if m else None


def e2(x):
    """Two-digit scientific format."""
    return f"{x:.2e}"


def fmt_key(key):
    """Readable element key: part, orbit labels (n l 2j 2tz), rest."""
    part, *rest = key
    out = []
    for r in rest:
        out.append("(" + " ".join(str(i) for i in r) + ")"
                   if isinstance(r, tuple) else str(r))
    return part + " " + " ".join(out)


def build(results, options, label, topk):
    """Markdown section of the decomposition."""
    cols = sorted({r[0] for r in results}, key=cf.op_sort_key)
    names = [n for n in ["total", "flow", "op", *options, "resid"]
             if any(n in r[2] for r in results)]
    loo = [x + " (loo)" for x in options
           if any(x + " (loo)" in r[2] for r in results)]
    checks = [n for n in ["v2-none", "compatop-fork"]
              if any(n in r[2] for r in results)]
    errors = [(r[0], r[1], r[2]["error"]) for r in results if "error" in r[2]]
    good = [r for r in results if "error" not in r[2]]
    out = [f"#### {label}", ""]
    if errors:
        out += ["Unreadable items: " + ", ".join(
            f"{c}_{q} ({e})" for c, q, e in errors), ""]

    def col_table(header_names, title):
        hdr = ["column", "items"] + [short(n.replace(" (loo)", "")) +
                                     (" (loo)" if n.endswith("(loo)") else "")
                                     for n in header_names]
        lines = [title, "", "| " + " | ".join(hdr) + " |",
                 "|" + "---|" * len(hdr)]
        for col in cols:
            rs = [r for r in good if r[0] == col]
            cells = [col, str(len(rs))]
            for n in header_names:
                vals = [(r[2][n][0], r[1]) for r in rs if n in r[2]]
                if not vals:
                    cells.append("")
                    continue
                v, q = max(vals)
                cells.append(f"{e2(v)} ({q})" if v > 0.0 else "0")
            lines.append("| " + " | ".join(cells) + " |")
        return lines + [""]

    out += col_table(names, "Max |contribution| per column (q of the max):")
    if loo or checks:
        out += col_table(loo + checks,
                         "Leave-one-out contributions and checks (max |.|):")
    # Shares per rank: sqrt(sum of squares) of each contribution over that of
    # op (flow and total: over total).
    ranks = sorted({rank_of(c) for c in cols if rank_of(c) is not None})
    share = [n for n in [*options, "resid"] if n in names]
    hdr = ["L", "\\|\\|op\\|\\|_2"] + [short(n) for n in share]
    if "total" in names and "flow" in names:
        hdr += ["\\|\\|flow\\|\\| / \\|\\|total\\|\\|",
                "\\|\\|op\\|\\| / \\|\\|total\\|\\|"]
    out += ["Shares per rank L: \\|\\|X\\|\\|_2 / \\|\\|op\\|\\|_2 over all "
            "elements of the rank's items:", "",
            "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for L in ranks:
        rs = [r for r in good if rank_of(r[0]) == L]

        def norm(n, rs=rs):
            return math.sqrt(sum(r[2][n][2] for r in rs if n in r[2]))
        op = norm("op")
        cells = [str(L), e2(op)]
        cells += [f"{norm(n) / op:.3f}" if op > 0.0 else "-" for n in share]
        if "total" in names and "flow" in names:
            t = norm("total")
            cells += [f"{norm('flow') / t:.3f}" if t > 0.0 else "-",
                      f"{op / t:.3f}" if t > 0.0 else "-"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    # Largest elements of |op|.
    rows = heapq.nlargest(topk, [row for r in good for row in r[3]],
                          key=lambda x: x[0])
    tn = [n for n in names if n != "op"]
    hdr = ["item", "element", "op"] + [short(n) for n in tn]
    out += [f"{topk} largest elements of \\|op\\|:", "",
            "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, col, q, key, vals in rows:
        cells = [f"{col}_{q}", fmt_key(key), f"{vals['op']:+.3e}"]
        cells += [f"{vals.get(n, 0.0):+.3e}" for n in tn]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--none", required=True)
    p.add_argument("--fork", required=True)
    p.add_argument("--only", action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--allbut", action="append", default=[],
                   metavar="NAME=DIR")
    p.add_argument("--v1")
    p.add_argument("--v2")
    p.add_argument("--compatop")
    p.add_argument("--label", default="compat decomposition")
    p.add_argument("--md")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--top", type=int, default=10)
    return p.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    dirs = {"none": args.none, "fork": args.fork}
    options = []
    for kind in ("only", "allbut"):
        for spec in getattr(args, kind):
            name, _, d = spec.partition("=")
            if not d:
                print(f"--{kind} needs NAME=DIR: {spec}", file=sys.stderr)
                return 2
            dirs[(kind, name)] = d
            if name not in options:
                options.append(name)
    for tag in ("v1", "v2", "compatop"):
        if getattr(args, tag):
            dirs[tag] = getattr(args, tag)
    scans = {}
    for tag, d in dirs.items():
        if not os.path.isdir(d):
            print(f"no directory {d}", file=sys.stderr)
            return 2
        scans[tag] = cf.scan_flat(d)[0]
    items = sorted(set(scans["none"]) & set(scans["fork"]))
    tasks = []
    for col, q in items:
        paths = {tag: s[(col, q)] for tag, s in scans.items() if (col, q) in s}
        tasks.append((col, q, paths, options, args.top))
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.map(item_stats, tasks)
    else:
        results = [item_stats(t) for t in tasks]
    text = build(results, options, args.label, args.top)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(run())
