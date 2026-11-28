#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Configs of the 27Al v3-vs-v2 investigation scan (domega x omega_norm_max).

Each config is tools/al27/<h>_v3.toml (H_0 from the v2 checkpoint through
checkpoint.prepared_from, so no readers/HF/NAT) with another imsrg.step.domega
and imsrg.step.omega_norm_max, its own output/flow/checkpoint paths below
$IMSRG_WORK/runs/al27/scan/<H>-<label>, no Omega files and no operators (only
the flow and E0 are wanted). Run with tools/al27/flows_chain.sh; the "ode"
points (imsrg.method = "flow", no checkpoint) with tools/al27/flows_gpu.sh.

usage: make_scan_configs.py OUT_DIR
"""

import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
POINTS = [("s025-d0025", 0.025, 0.25), ("s025-d005", 0.05, 0.25),
          ("s1-d01", 0.1, 1.0), ("unsplit-d005", 0.05, 50000.0),
          ("s01-d01", 0.1, 0.1),
          # ODE reference (imsrg.method = "flow", adaptive dopri5): no Omegas,
          # no checkpoint (Magnus only); omega keys are ignored.
          ("ode", 0.1, 0.25)]
# Operator checks of the v3-vs-v2 differences (N2LOsat only): the same flows
# with native Omegas (for a small farm), and the ODE flow carrying two scalar
# NREFT operators alongside H (tensors would make it too slow).
OPS_POINTS = [("s1-d01-om", 0.1, 1.0), ("s01-d01-om", 0.1, 0.1),
              ("s025-d0025-om", 0.025, 0.25), ("ode-ops", 0.1, 0.25)]
ODE_OPERATORS = '["FM0p_150", "FPhipp0p_0"]'


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for H, h in [("N2LOsat", "n2losat"), ("DN2LOGO", "dn2logo")]:
        with open(f"{SRC}/{h}_v3.toml", encoding="utf-8") as f:
            src = f.read()
        points = POINTS + (OPS_POINTS if H == "N2LOsat" else [])
        for lab, dom, onm in points:
            s = re.sub(r"(?m)^domega = .*$", f"domega = {dom}", src)
            s = re.sub(r"(?m)^omega_norm_max = .*$",
                       f"omega_norm_max = {onm}", s)
            s = s.replace(f"/runs/al27/v3/{H}", f"/runs/al27/scan/{H}-{lab}")
            s = re.sub(r"(?m)^write_omega = true", "write_omega = false", s)
            s = re.sub(r'(?m)^list = \["TCM"\]', "list = []", s)
            if lab.endswith("-om"):
                s = re.sub(r"(?m)^write_omega = false", "write_omega = true",
                           s)
                s = re.sub(r"(?m)^omega_second_format = true",
                           "omega_second_format = false", s)
            if lab == "ode-ops":
                s = re.sub(r"(?m)^list = \[\]", f"list = {ODE_OPERATORS}", s)
            if lab.startswith("ode"):
                s = re.sub(r'(?m)^method = "magnus"', 'method = "flow"', s)
                s = re.sub(r'(?m)^(\[checkpoint\]\ndirectory = ).*$',
                           r'\1""', s)
                assert 'method = "flow"' in s and 'directory = ""' in s
            assert "/v3/" not in s
            with open(f"{out}/{H}-{lab}.toml", "w", encoding="utf-8") as f:
                f.write(s)


if __name__ == "__main__":
    main()
