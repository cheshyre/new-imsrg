#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Operators and valence H of Magnus runs against the ODE IMSRG(2) flow.

  ode_check.py --stem H --ode DIR [--items FM0p_150,FPhipp0p_0]
      NAME=FARM_DIR [NAME=FARM_DIR ...]

Reference = the ODE run directory DIR (`<H>_<item>.snt` of the operators it
carried, `<H>.snt` = valence H). Each NAME=FARM_DIR gives the operator files
`<H>_<item>.snt` in FARM_DIR and the valence H `<H>.snt` in FARM_DIR or, for
a `.../farm-native` directory, in its parent (the flow directory). Cells (as
al27-variants.md 5.2): median of |test - ref| / |ref| over the dominant
elements (dominant_rel.py rule: |ref| >= max(0.01 max|ref|, 1e-6), one- and
two-body parts) / largest |test - ref| over the largest |ref|; last row: the
core energy (zero-body term) of the valence H minus the ODE one in keV.
Missing files give an empty cell. Output: a markdown table on stdout.
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_farm as cf  # noqa: E402
import dominant_rel as dr  # noqa: E402


def cell(ref, test):
    """'median rel / max abs over max ref' of two files ('' if one lacks)."""
    if not (os.path.exists(ref) and os.path.exists(test)):
        return ""
    s = dr.dominant_stats(dr.load_elements(ref), dr.load_elements(test),
                          1.0e-6, 0.01)
    if s["vanishes"]:
        return "vanishes"
    return f"{statistics.median(s['rels']):.1e} / " \
           f"{s['max_abs'] / s['max_ref']:.1e}"


def flow_dir(d):
    """Directory of the valence H of a farm or flow directory."""
    d = os.path.normpath(d)
    return os.path.dirname(d) if os.path.basename(d) == "farm-native" else d


def table(stem, ode, others, items):
    """Markdown lines of the comparison."""
    names = [n for n, _ in others]
    out = ["| quantity (vs ODE) | " + " | ".join(names) + " |",
           "|---|" + "---|" * len(names)]
    for it in items:
        f = f"{stem}_{it}.snt"
        out.append(f"| {it} | " + " | ".join(
            cell(os.path.join(ode, f), os.path.join(d, f))
            for _, d in others) + " |")
    h = f"{stem}.snt"
    out.append("| valence H | " + " | ".join(
        cell(os.path.join(ode, h), os.path.join(flow_dir(d), h))
        for _, d in others) + " |")
    z0 = cf.parse_snt(os.path.join(ode, h)).zero
    zs = []
    for _, d in others:
        p = os.path.join(flow_dir(d), h)
        zs.append(f"{(cf.parse_snt(p).zero - z0) * 1000.0:+.1f}"
                  if os.path.exists(p) else "")
    out.append(f"| core energy - ODE (keV; ODE {z0:.6f} MeV) | " +
               " | ".join(zs) + " |")
    return out


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--ode", required=True)
    ap.add_argument("--items", default="FM0p_150,FPhipp0p_0")
    ap.add_argument("others", nargs="+", metavar="NAME=DIR")
    args = ap.parse_args(argv)
    others = []
    for spec in args.others:
        name, _, d = spec.partition("=")
        if not d:
            print(f"need NAME=DIR: {spec}", file=sys.stderr)
            return 2
        others.append((name, d))
    if not os.path.exists(os.path.join(args.ode, f"{args.stem}.snt")):
        print(f"no {args.stem}.snt in {args.ode}", file=sys.stderr)
        return 2
    items = [x for x in args.items.split(",") if x]
    print("\n".join(table(args.stem, args.ode, others, items)))
    return 0


if __name__ == "__main__":
    sys.exit(run())
