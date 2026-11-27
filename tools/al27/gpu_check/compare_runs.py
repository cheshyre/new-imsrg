#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Compares two output directories of tools/al27/gpu_check/run_check.sh.

Checks (Brief A, acceptance item 4), each as max|a - b| / max|a| over the
compared values (a = reference directory):
  * flow file: every column but wall_s (relative to the column's
    maximum), and the final E0;
  * the Omegas (native .opb, full float64 payload) and the flow's .snt files
    (every number, as printed);         tolerance --tol-flow (1e-10)
  * the farm's native .opb dumps (transform.dump_native) and .snt files;
                                        tolerance --tol-farm (1e-12)
Native files are compared on their payload (zero body, one-body and
two-body buffers as stored, io/operator_binary.h) after checking that the
headers are identical. Exits 1 if a check fails or a file is missing.

usage: compare_runs.py REF_DIR TEST_DIR [--tol-flow 1e-10] [--tol-farm 1e-12]
"""

import argparse
import math
import os
import re
import struct
import sys

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def opb_payload(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"IMSRGOPB":
        raise ValueError(path + ": not a native operator file")
    hlen = struct.unpack_from("<I", data, 12)[0]
    header = data[16:16 + hlen]
    body = data[16 + hlen:-8]
    n = len(body) // 8
    return header, struct.unpack("<%dd" % n, body)


def rel_dev(a, b):
    """max|a-b| / max|a| (max|a-b| if a is all zero) and the worst index."""
    if len(a) != len(b):
        return math.inf, -1
    dmax, imax, amax = 0.0, -1, 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(x - y)
        if d > dmax:
            dmax, imax = d, i
        amax = max(amax, abs(x))
    return (dmax / amax if amax > 0 else dmax), imax


def text_numbers(path, skip_cols=()):
    vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith(("#", "!")):
                continue
            toks = NUM.findall(line)
            vals.extend(float(t) for i, t in enumerate(toks)
                        if i not in skip_cols)
    return vals


def flow_columns(path):
    """The flow file (CSV with a header line) as {column: values}, without
    the wall-clock column."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    names = lines[0].split(",")
    cols = {n: [] for n in names if n != "wall_s"}
    for ln in lines[1:]:
        for n, t in zip(names, ln.split(",")):
            if n in cols:
                cols[n].append(float(t))
    return cols


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--tol-flow", type=float, default=1.0e-10)
    p.add_argument("--tol-farm", type=float, default=1.0e-12)
    a = p.parse_args()
    fails = 0

    def report(name, dev, tol, extra=""):
        nonlocal fails
        ok = dev <= tol
        fails += 0 if ok else 1
        print("%-4s %-48s %.3e (tol %.0e)%s" %
              ("ok" if ok else "FAIL", name, dev, tol, extra))

    def files(sub, pred):
        d = os.path.join(a.ref, sub)
        return sorted(f for f in os.listdir(d) if pred(f)) if os.path.isdir(d) \
            else []

    # Flow file: every column (relative to the column's maximum), final E0.
    cr, ct = (flow_columns(os.path.join(x, "flow.txt"))
              for x in (a.ref, a.test))
    worst, wname = 0.0, ""
    for n, v in cr.items():
        d = rel_dev(v, ct.get(n, []))[0]
        if d > worst:
            worst, wname = d, n
    report("flow.txt (every column but wall_s)", worst, a.tol_flow,
           "  worst column %s, %d rows" % (wname, len(cr["E0"])))
    er, et = cr["E0"][-1], ct["E0"][-1]
    report("final E0", abs(er - et) / abs(er), a.tol_flow,
           "  %.14f vs %.14f" % (er, et))

    for sub, tol in (("", a.tol_flow), ("farm", a.tol_farm)):
        for f in files(sub, lambda f: f.endswith(".opb")):
            rp, tp = os.path.join(a.ref, sub, f), os.path.join(a.test, sub, f)
            if not os.path.exists(tp):
                report(os.path.join(sub, f) + " (missing)", math.inf, tol)
                continue
            hr, pr = opb_payload(rp)
            ht, pt = opb_payload(tp)
            if hr != ht:
                report(os.path.join(sub, f) + " (header differs)", math.inf,
                       tol)
                continue
            report(os.path.join(sub, f), rel_dev(pr, pt)[0], tol,
                   "  (%d values)" % len(pr))
        for f in files(sub, lambda f: f.endswith(".snt")):
            rp, tp = os.path.join(a.ref, sub, f), os.path.join(a.test, sub, f)
            if not os.path.exists(tp):
                report(os.path.join(sub, f) + " (missing)", math.inf, tol)
                continue
            # Printed values (6-9 digits): equal unless a rounding boundary
            # falls between the two results.
            report(os.path.join(sub, f) + " (printed)",
                   rel_dev(text_numbers(rp), text_numbers(tp))[0],
                   max(tol, 1.0e-10))
    print("RESULT", "PASS" if fails == 0 else "FAIL (%d)" % fails)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
