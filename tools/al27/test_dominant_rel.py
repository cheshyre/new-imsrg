# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of dominant_rel.py with synthetic tiny files."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dominant_rel as dr  # noqa: E402
from test_compare_farm import ONE, TWO, write_snt  # noqa: E402


def test_dominant_cut_and_small_bound():
    ref = {("1b", 1): 1.0, ("1b", 2): 0.02, ("1b", 3): 0.005,
           ("2b", 1): 1.0e-3}
    test = {("1b", 1): 1.1, ("1b", 2): 0.021, ("1b", 3): 0.0, ("2b", 1): 0.0}
    st = dr.dominant_stats(ref, test, 1.0e-6, 0.01)
    # cut = 1e-2: elements 1.0 and 0.02 are dominant, 0.005 and 1e-3 not
    assert sorted(round(x, 6) for x in st["rels"]) == [0.05, 0.1]
    assert not st["vanishes"] and st["max_ref"] == 1.0
    assert abs(st["max_abs"] - 0.1) < 1e-12
    # a tiny file: the 1e-6 bound wins over the 1 % cut
    st = dr.dominant_stats({("1b", 1): 1.0e-7}, {("1b", 1): 2.0e-7}, 1.0e-6,
                           0.01)
    assert st["vanishes"] and st["rels"] == []
    assert abs(st["test_max"] - 2.0e-7) < 1e-20
    # an element just below the small bound is never dominant
    st = dr.dominant_stats({("1b", 1): 1.0e-5, ("1b", 2): 5.0e-7},
                           {("1b", 1): 1.0e-5, ("1b", 2): 1.0e-6}, 1.0e-6,
                           1.0e-3)
    assert st["rels"] == [0.0]


def test_part_selection():
    ref = {("1b", 1): 1.0, ("2b", 1): 0.5}
    test = {("1b", 1): 1.5, ("2b", 1): 0.5}
    assert dr.dominant_stats(ref, test, 1e-6, 0.01, "1b")["rels"] == [0.5]
    assert dr.dominant_stats(ref, test, 1e-6, 0.01, "2b")["rels"] == [0.0]
    assert len(dr.dominant_stats(ref, test, 1e-6, 0.01)["rels"]) == 2


def test_vanishing_reported_not_zero(tmp_path, capsys):
    ref, test = tmp_path / "ref", tmp_path / "test"
    ref.mkdir()
    test.mkdir()
    zero1 = [((1, 1), 0.0), ((3, 3), 0.0)]
    zero2 = [((1, 3, 1, 3, 0), 0.0)]
    write_snt(ref / "sFM2p_0.snt", zero1, zero2)
    write_snt(test / "sFM2p_0.snt", zero1, zero2)
    write_snt(ref / "sFM2p_15.snt", ONE, TWO)
    two = [(k, v * 1.1) for k, v in TWO]
    write_snt(test / "sFM2p_15.snt", ONE, two)
    code = dr.run(["--ref-farm", str(ref), "--test-farm", str(test),
                   "--jobs", "1", "--qs", "0,15"])
    out = capsys.readouterr().out
    assert code == 0
    row = [ln for ln in out.splitlines() if ln.startswith("| FM2p |")][0]
    cells = [c.strip() for c in row.split("|")]
    assert cells[-2].startswith("vanishes / ")  # q = 0, then q = 15
    assert "0.0e+00" not in cells[-2].split(" / ")[0]
    assert cells[2] == "2"  # files
    # q = 15 dominant (>= 1 % of max |ref| = 0.5): 1b 0.5, 0.25 and 2b 1e-2
    assert cells[3] == "3"
    ncell = out.splitlines()[0].count("|")
    for ln in out.splitlines():
        if ln.startswith("|"):
            assert ln.count("|") == ncell


def test_all_vanishing_column(tmp_path, capsys):
    ref, test = tmp_path / "ref", tmp_path / "test"
    ref.mkdir()
    test.mkdir()
    write_snt(ref / "sFM4p_0.snt", [((1, 1), 0.0)], [])
    write_snt(test / "sFM4p_0.snt", [((1, 1), 0.0)], [])
    assert dr.run(["--ref-farm", str(ref), "--test-farm", str(test),
                   "--jobs", "1", "--qs", "0"]) == 0
    assert "vanishes at all 1 q" in capsys.readouterr().out


def test_no_pairs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert dr.run(["--ref-farm", str(tmp_path / "a"), "--test-farm",
                   str(tmp_path / "b")]) == 2
