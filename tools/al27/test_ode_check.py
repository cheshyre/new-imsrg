# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of ode_check.py with synthetic tiny files."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ode_check as oc  # noqa: E402
from test_compare_farm import ONE, TWO, write_snt  # noqa: E402


def scaled(rows, f):
    return [(k, v * f) for k, v in rows]


def test_table_cells(tmp_path, capsys):
    ode = tmp_path / "ode"
    flow = tmp_path / "v3"
    farm = flow / "farm-native"
    for d in (ode, farm):
        d.mkdir(parents=True)
    write_snt(ode / "H_FM0p_150.snt", ONE, TWO, zero=8.0)
    write_snt(ode / "H.snt", ONE, TWO, zero=-100.0)
    # Operator 2 % larger everywhere; valence H 1 % larger, core +2.5 keV.
    write_snt(farm / "H_FM0p_150.snt", scaled(ONE, 1.02), scaled(TWO, 1.02),
              zero=8.0)
    write_snt(flow / "H.snt", scaled(ONE, 1.01), scaled(TWO, 1.01),
              zero=-99.9975)
    assert oc.run(["--stem", "H", "--ode", str(ode), "--items",
                   "FM0p_150,FPhipp0p_0", f"v3={farm}"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "| quantity (vs ODE) | v3 |"
    assert lines[2] == "| FM0p_150 | 2.0e-02 / 2.0e-02 |"
    assert lines[3] == "| FPhipp0p_0 |  |"  # no files: empty cell
    assert lines[4] == "| valence H | 1.0e-02 / 1.0e-02 |"
    assert lines[5] == "| core energy - ODE (keV; ODE -100.000000 MeV) | +2.5 |"


def test_missing_ode_is_an_error(tmp_path):
    assert oc.run(["--stem", "H", "--ode", str(tmp_path), "x=y"]) == 2
