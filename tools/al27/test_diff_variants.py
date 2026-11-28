# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of diff_variants.py with synthetic tiny files."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_variants as dv  # noqa: E402
from test_compare_farm import ONE, TENSOR_TWO, TWO, write_snt  # noqa: E402


def mk(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def run(tmp_path, ref, test, *extra, flow=False):
    md = tmp_path / "out.md"
    args = ["--ref-farm", str(ref), "--test-farm", str(test), "--jobs", "1",
            "--md", str(md), "--label", "t", *extra]
    code = dv.run(args)
    return code, md.read_text()


def row(text, start):
    return [ln for ln in text.splitlines() if ln.startswith(start)][0]


def split_cells(line):
    """Cells of a markdown table row; an escaped pipe does not split."""
    body = line.strip()
    assert body.startswith("|") and body.endswith("|"), line
    body = body[1:-1].replace("\\|", "")
    return body.split("|")


def check_tables(text):
    """Every markdown table: header, delimiter and rows have equal cells."""
    block, ntab = [], 0
    for ln in text.splitlines() + [""]:
        if ln.startswith("|"):
            block.append(ln)
            continue
        if block:
            ntab += 1
            n = len(split_cells(block[0]))
            assert set(c.strip("- :") for c in split_cells(block[1])) == {""}
            for b in block:
                assert len(split_cells(b)) == n, (n, b)
            block = []
    return ntab


def test_both_naming_and_renumbered(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    write_snt(ref / "stemFM2p_15.snt", ONE, TWO)
    write_snt(test / "stem_FM2p_15.snt", ONE, TWO,
              perm={1: 3, 2: 1, 3: 4, 4: 2})
    write_snt(ref / "stemTCM.snt", ONE, TWO)
    write_snt(test / "stem_TCM.snt", ONE, TWO)
    code, text = run(tmp_path, ref, test)
    assert code == 0
    assert "2 file pairs compared" in text
    assert ("| 0.00e+00 (15) | 0.00e+00 (15) | 0 | 0.00e+00 | 0 of 6 | 0 |"
            in row(text, "| FM2p |"))
    check_tables(text)
    assert "| TCM | 1 |" in text


def test_big_statistic_ignores_moderately_small_refs():
    # |ref| = 5e-4 (>= --small, < BIG) enters max_rel but not max_rel_big.
    st = dv.diff_elements({"a": 5.0e-4, "b": 1.0}, {"a": 1.0e-3, "b": 1.002},
                          1.0e-6, 1.0e-3, 3)
    assert abs(st["max_rel"] - 1.0) < 1e-12
    assert abs(st["max_rel_big"] - 2.0e-3) < 1e-12
    assert st["n_flag"] == 2 and st["n_flag_big"] == 1


def test_small_excluded_and_rel_flag(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    one_r = [((1, 1), 5.0e-7), ((3, 3), 1.0)]
    one_t = [((1, 1), 5.0e-6), ((3, 3), 1.01)]
    write_snt(ref / "sFM0p_0.snt", one_r, [])
    write_snt(test / "sFM0p_0.snt", one_t, [])
    code, text = run(tmp_path, ref, test)
    r = row(text, "| FM0p |")
    assert "| 1.00e-02 (0) | 1 | 1.00e-02 | 1 of 2 | 1 |" in r
    st = dv.diff_elements({"a": 5.0e-7, "b": 1.0}, {"a": 5.0e-6, "b": 1.01},
                          1.0e-6, 1.0e-3, 3)
    assert st["n_small"] == 1 and st["n_flag"] == 1
    assert abs(st["max_rel"] - 0.01) < 1e-12
    assert st["abs_key"] == "b" and st["max_ref"] == 1.0


def test_only_in_one_file_and_zero_body(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    write_snt(ref / "sFM0p_0.snt", ONE, TWO, zero=1.5)
    write_snt(test / "sFM0p_0.snt", ONE, TWO[:2], zero=1.6)
    code, text = run(tmp_path, ref, test)
    assert "| FM0p | 1 |" in text
    assert " 1/0 |" in row(text, "| FM0p |")  # one 2b element only in ref
    assert "0b:-" in text
    assert "1.00e-01" in text  # zero-body abs diff 0.1


def test_missing_files_listed(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    for q in (0, 15, 30):
        write_snt(ref / f"sFM0p_{q}.snt", ONE, TWO)
    write_snt(test / "sFM0p_15.snt", ONE, TWO)
    write_snt(test / "sFM2n_45.snt", ONE, TWO)
    code, text = run(tmp_path, ref, test)
    assert code == 0
    assert "only in REF: 2 files" in text
    assert "1 columns, q = 0,30: FM0p" in text
    assert "only in TEST: 1 files" in text
    assert "1 columns, q = 45: FM2n" in text


def test_tensor_file(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    bad = [TENSOR_TWO[0], ((1, 2, 3, 4, 2, 1), -3.3e-3)]
    write_snt(ref / "sFM2p_15_2b.snt", ONE, TENSOR_TWO)
    write_snt(test / "sFM2p_15_2b.snt", ONE, bad)
    code, text = run(tmp_path, ref, test)
    assert "| FM2p | 1 |" in text
    assert "3.00e-04" in text
    assert "2b:(0, 2, 5, -1)" in text.replace("(", "(")


def test_flow_parsing_and_missing_dir(tmp_path):
    fl = {}
    for name, e0 in (("ref", -213.96), ("test", -213.95)):
        d = mk(tmp_path, name)
        write_snt(d / "N2LOsat.snt", ONE, TWO)
        write_snt(d / "N2LOsat_TCM.snt", ONE, TWO)
        (d / "results.json").write_text(json.dumps({
            "valence_interaction": str(d / "N2LOsat.snt"),
            "imsrg": {"e0": e0, "num_omegas": 2, "stages": [
                {"stage": 1, "steps": 3, "s": 1.0, "e0": e0}]}}))
        (d / "flow.txt").write_text(
            "stage,istep,s,ds,E0,norm_H,num_omegas\n"
            f"1,0,0,0,-50.0,1.0,1\n1,3,1.5,0.5,{e0},1.0,2\n")
        fl[name] = d
    md = tmp_path / "o.md"
    code = dv.run(["--ref-flow", str(fl["ref"]), "--test-flow",
                   str(fl["test"]), "--ref-farm", str(tmp_path / "none"),
                   "--test-farm", str(tmp_path / "none2"), "--md", str(md),
                   "--jobs", "1"])
    text = md.read_text()
    assert code == 2  # missing farm dirs
    assert "directory not found" in text
    assert "| final E0 (results.json) | -213.9600000000 | " in text
    assert "1.00e-02 |" in text
    assert "| stage 1 final step | 3 | 3 | 0 |" in text
    assert "| num_omegas (flow.txt, last row) | 2 | 2 | 0 |" in text
    assert "| H | 1 |" in text and "| TCM(flow) | 1 |" in text
    assert len(text) < 100000
    assert check_tables(text) >= 5


def test_per_column_and_q_counts(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    one_r = [((1, 1), 5.0e-7), ((3, 3), 1.0)]
    one_t = [((1, 1), 5.0e-6), ((3, 3), 1.01)]
    for q in (0, 15):
        write_snt(ref / f"sFM0p_{q}.snt", one_r, [])
        write_snt(test / f"sFM0p_{q}.snt", one_t, [])
    code, text = run(tmp_path, ref, test)
    # per file: 3 elements (zero-body, two 1b), 1 small -> 2 rel-eligible
    assert "| FM0p | 2 | 6 | 4 |" in text
    assert "| 0 | 1 | 3 | 2 | 1 |" in text
    assert "| 15 | 1 | 3 | 2 | 1 |" in text
    assert "\\|ref\\|>=0.001" in text
    assert check_tables(text) >= 5


def test_top_and_size(tmp_path):
    ref, test = mk(tmp_path, "ref"), mk(tmp_path, "test")
    for q in range(0, 300, 15):
        two = [(k, v * (1 + q * 1e-4)) for k, v in TWO]
        write_snt(ref / f"sFM0p_{q}.snt", ONE, TWO)
        write_snt(test / f"sFM0p_{q}.snt", ONE, two)
    code, text = run(tmp_path, ref, test, "--top", "3", "--rel-flag", "1e-3")
    assert code == 0
    assert text.count("sFM0p_285.snt") >= 2
    assert len(text.encode()) < 100000
