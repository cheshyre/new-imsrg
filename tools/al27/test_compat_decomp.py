# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of compat_decomp.py with synthetic tiny files."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_decomp as cd  # noqa: E402
from test_compare_farm import ONE, TENSOR_TWO, TWO, write_snt  # noqa: E402

A = "tensor_no_fork_diagonal_bug"
B = "tensor_fork_ph_missing_rank_phase"


def test_contributions_signs_and_residual():
    none = {"x": 1.0, "y": 2.0, "z": 0.5}
    fork = {"x": 0.9, "y": 2.2, "z": 0.5}   # A moves x by -0.1, B y by +0.2
    data = {"none": none, "fork": fork,
            ("only", A): {"x": 0.9, "y": 2.0, "z": 0.5},
            ("only", B): {"x": 1.0, "y": 2.2, "z": 0.5},
            ("allbut", A): {"x": 1.0, "y": 2.2, "z": 0.5},
            "v1": {"x": 0.9, "y": 2.2, "z": 0.25},
            "v2": dict(none)}
    c = cd.contributions(data, [A, B])
    assert c["op"]["x"] == pytest.approx(0.1)
    assert c["op"]["y"] == pytest.approx(-0.2)
    assert c[A]["x"] == pytest.approx(0.1) and c[A]["y"] == 0.0
    assert c[B]["y"] == pytest.approx(-0.2) and c[B]["x"] == 0.0
    assert c[A + " (loo)"]["x"] == pytest.approx(0.1)
    assert max(abs(v) for v in c["resid"].values()) < 1e-15
    assert c["flow"]["z"] == pytest.approx(0.25)
    assert c["total"]["z"] == pytest.approx(0.25)
    assert c["total"]["x"] == pytest.approx(0.1)
    assert all(v == 0.0 for v in c["v2-none"].values())


def test_missing_element_counts_as_zero_and_no_resid_without_all():
    data = {"none": {"x": 1.0, "w": 0.3}, "fork": {"x": 1.0},
            ("only", A): {"x": 1.0}}
    c = cd.contributions(data, [A, B])  # B has no farm: no residual
    assert c["op"]["w"] == pytest.approx(0.3)
    assert c[A]["w"] == pytest.approx(0.3)
    assert "resid" not in c and B not in c


def shifted(base, idx, delta):
    out = list(base)
    k, v = out[idx]
    out[idx] = (k, v + delta)
    return out


def make_farms(tmp_path):
    """Farms with one tensor and one scalar item; A acts on the 1b element
    (1,1) (-0.125 in the fork), B on the first 2b element (-0.0625), the flow
    on (3,3) (v1 = fork + 0.03125); binary-exact values."""
    spec = {"none": (ONE, 0), "fork": (shifted(ONE, 0, -0.125), -0.0625),
            "onlyA": (shifted(ONE, 0, -0.125), 0),
            "onlyB": (ONE, -0.0625),
            "allbutA": (ONE, -0.0625),
            "allbutB": (shifted(ONE, 0, -0.125), 0),
            "v1": (shifted(shifted(ONE, 0, -0.125), 1, 0.03125), -0.0625),
            "v2": (ONE, 0)}
    dirs = {}
    for tag, (one, d2) in spec.items():
        d = tmp_path / tag
        d.mkdir()
        write_snt(d / "X_FSigmap1p_0_2b.snt", one,
                  shifted(TENSOR_TWO, 0, d2), zero=0.0, tensor=True)
        write_snt(d / "X_FM0p_150.snt", one, TWO, zero=8.0)
        dirs[tag] = str(d)
    return dirs


def cells(text, start):
    line = [ln for ln in text.splitlines() if ln.startswith(start)][0]
    return [c.strip() for c in line.strip("|").split("|")]


def test_end_to_end_tables(tmp_path):
    d = make_farms(tmp_path)
    md = tmp_path / "out.md"
    code = cd.run(["--none", d["none"], "--fork", d["fork"],
                   "--only", f"{A}={d['onlyA']}", "--only", f"{B}={d['onlyB']}",
                   "--allbut", f"{A}={d['allbutA']}",
                   "--allbut", f"{B}={d['allbutB']}",
                   "--v1", d["v1"], "--v2", d["v2"], "--label", "t",
                   "--md", str(md), "--top", "3"])
    assert code == 0
    text = md.read_text()
    # Column order: column, items, total, flow, op, A, B, resid.
    row = cells(text, "| FSigmap1p |")
    assert row[1] == "1"
    assert row[2] == "1.25e-01 (0)"   # total = op + flow, max at (1,1)
    assert row[3] == "3.12e-02 (0)"   # flow
    assert row[4] == "1.25e-01 (0)"   # op
    assert row[5] == "1.25e-01 (0)"   # A
    assert row[6] == "6.25e-02 (0)"   # B
    assert row[7] == "0"              # resid
    # The scalar item shares the one-body pattern (A shifts (1,1)).
    row = cells(text, "| FM0p |")
    assert row[4] == "1.25e-01 (150)"
    # Leave-one-out table and checks: loo equals the single contributions,
    # v2 - none vanishes.
    loo = [ln for ln in text.splitlines() if ln.startswith("| FSigmap1p |")][1]
    loo_cells = [c.strip() for c in loo.strip("|").split("|")]
    assert loo_cells[2:5] == ["1.25e-01 (0)", "6.25e-02 (0)", "0"]
    # Per rank: L = 1 has A max 0.125, share 0.894 (0.125 / sqrt(0.125^2 +
    # 0.0625^2)), B 0.0625 / 0.447, resid 0.
    row = cells(text, "| 1 |")
    assert row[1] == "1.25e-01 / 1.40e-01"
    assert row[2:5] == ["1.25e-01 / 0.894", "6.25e-02 / 0.447",
                        "0.00e+00 / 0.000"]
    assert "+1.250e-01" in text  # top element of op


def test_missing_directory_is_an_error(tmp_path):
    assert cd.run(["--none", str(tmp_path / "x"), "--fork",
                   str(tmp_path)]) == 2
