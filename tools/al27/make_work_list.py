#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Work list of the 27Al response farm (design/50 sections 2 and 6).

Writes one operator name per line for `transform.work_list`: the 17 NREFT
shorthand families (FM0/2/4, FPhipp0/2/4, FSigmap1/3/5, FSigmapp1/3/5,
FDelta1/3/5, FPhitp2/4), each for protons and neutrons, at q = 0, 15, ...,
1050 MeV (71 values): 34 x 71 = 2414 names, plus TCM (2415 lines). The names
are the fork's (`F<family><L><p|n>_<q>`), so the output files are
`<stem><name>.snt` / `<stem><name>_2b.snt` like the archive's.

Usage: make_work_list.py [--qmax 1050] [--dq 15] [--families FM0,FSigmap1]
                         [--no-tcm] > work.txt
"""
import argparse
import sys

FAMILIES = (["FM%d" % L for L in (0, 2, 4)] +
            ["FPhipp%d" % L for L in (0, 2, 4)] +
            ["FSigmap%d" % L for L in (1, 3, 5)] +
            ["FSigmapp%d" % L for L in (1, 3, 5)] +
            ["FDelta%d" % L for L in (1, 3, 5)] +
            ["FPhitp%d" % L for L in (2, 4)])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qmax", type=int, default=1050)
    p.add_argument("--dq", type=int, default=15)
    p.add_argument("--families", default=",".join(FAMILIES),
                   help="comma-separated subset, e.g. FM0,FSigmap1")
    p.add_argument("--no-tcm", action="store_true")
    a = p.parse_args()
    fams = [f for f in a.families.split(",") if f]
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        sys.exit("unknown families: " + ",".join(unknown))
    out = []
    for f in fams:
        for iso in ("p", "n"):
            for q in range(0, a.qmax + 1, a.dq):
                out.append("%s%s_%d" % (f, iso, q))
    if not a.no_tcm:
        out.append("TCM")
    sys.stdout.write("# 27Al response farm work list: %d operators\n" %
                     len(out))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
