# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of compare_farm.py with synthetic tiny files."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_farm as cf  # noqa: E402

MODEL = [(1, 0, 2, 5, -1), (2, 0, 2, 3, -1), (3, 0, 2, 5, 1), (4, 0, 2, 3, 1)]


def write_snt(path, one, two, zero=1.5, tensor=False, perm=None,
              start="! interaction"):
    """Write a tiny tokyo file; `perm` renumbers the orbits (old -> new)."""
    perm = perm or {i: i for i in range(1, 5)}
    lines = ["! test file", f"! Zero body term: {zero}", "! ",
             "! model space", "   2   2   8   8"]
    for k, n, l, j2, tz in sorted(MODEL, key=lambda o: perm[o[0]]):
        lines.append(f"   {perm[k]}   {n}   {l}   {j2}  {tz}")
    lines += [start, f"{len(one)} 0 16"]
    for (a, b), v in one:
        lines.append(f"   {perm[a]}   {perm[b]}   {v:.6e}")
    lines.append(f"{len(two)} 0 16")
    for key, v in two:
        a, b, c, d = (perm[x] for x in key[:4])
        rest = " ".join(str(x) for x in key[4:])
        lines.append(f"   {a} {b} {c} {d} {rest} {v:.6e}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


ONE = [((1, 1), 0.5), ((3, 3), -0.25)]
TWO = [((1, 3, 1, 3, 0), 1.0e-2), ((1, 2, 1, 2, 2), -3.0e-3),
       ((2, 4, 2, 4, 1), 4.0e-3)]
TENSOR_TWO = [((1, 3, 1, 3, 0, 1), 1.0e-2), ((1, 2, 3, 4, 2, 1), -3.0e-3)]


def make_pair(tmp_path, ref_two=TWO, test_two=TWO, test_one=ONE, perm=None,
              tensor=False):
    ref = tmp_path / "ref"
    test = tmp_path / "test"
    ref.mkdir(exist_ok=True)
    test.mkdir(exist_ok=True)
    suffix = "_2b" if tensor else ""
    write_snt(ref / f"INTFILEFM2p_15{suffix}.snt", ONE, ref_two)
    write_snt(test / f"stemFM2p_15{suffix}.snt", test_one, test_two,
              perm=perm)
    return ref, test


def run(ref, test, *extra):
    return cf.run(["--ref-dir", str(ref), "--test", str(test), "--workers",
                   "1", "--tol", "1e-4", *extra])


def test_identical_scalar(tmp_path, capsys):
    ref, test = make_pair(tmp_path)
    assert run(ref, test) == 0
    assert "PASS" in capsys.readouterr().out


def test_identical_tensor(tmp_path):
    ref, test = make_pair(tmp_path, TENSOR_TWO, TENSOR_TWO, tensor=True)
    assert run(ref, test) == 0


def test_changed_row_fails(tmp_path, capsys):
    bad = [TWO[0], ((1, 2, 1, 2, 2), -3.0e-3 + 2.0e-4), TWO[2]]
    ref, test = make_pair(tmp_path, TWO, bad)
    tsv = tmp_path / "out.tsv"
    assert run(ref, test, "--tsv", str(tsv)) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "(0, 2, 5, -1), (0, 2, 3, -1)" in out.replace("), (", "), (")
    row = tsv.read_text().splitlines()[1].split("\t")
    assert abs(float(row[5]) - 2.0e-4) < 1e-9
    assert "ref -3.000000e-03" in row[8]


def test_row_missing_in_test(tmp_path, capsys):
    ref_two = TWO + [((3, 4, 3, 4, 1), 5.0e-5)]
    ref, test = make_pair(tmp_path, ref_two, TWO)
    assert run(ref, test) == 0
    res = cf.compare_files(("FM2p", 15, str(ref / "INTFILEFM2p_15.snt"),
                            str(test / "stemFM2p_15.snt"), False, False))
    assert res["only_ref"] == 1 and res["only_test"] == 0
    assert abs(res["max2"] - 5.0e-5) < 1e-12


def test_renumbering(tmp_path):
    ref, test = make_pair(tmp_path, perm={1: 3, 2: 1, 3: 4, 4: 2})
    assert run(ref, test) == 0


def test_missing_file_exit_code(tmp_path):
    ref, test = make_pair(tmp_path)
    (test / "stemFM2p_15.snt").unlink()
    assert run(ref, test) == 2
    assert run(ref, test, "--allow-missing") == 0


def test_canonical_swap(tmp_path):
    # (1,3 | 1,2) J=1 in ref versus swapped bra pair with the swap phase
    ref_two = [((1, 3, 1, 2, 1), 2.0e-3)]
    # swap a,b: phase -(-1)^((j_a+j_b)/2 - J) = -(-1)^(5-1) = -1
    test_two = [((3, 1, 1, 2, 1), -2.0e-3)]
    ref, test = make_pair(tmp_path, ref_two, test_two)
    assert run(ref, test) == 1
    assert run(ref, test, "--canonical") == 0


def test_md_replace_keeps_rest(tmp_path):
    ref, test = make_pair(tmp_path)
    md = tmp_path / "rep.md"
    md.write_text("# Report\n\nintro text\n")
    assert run(ref, test, "--md", str(md), "--title", "A") == 0
    assert run(ref, test, "--md", str(md), "--title", "B") == 0
    assert run(ref, test, "--md", str(md), "--title", "A") == 0
    text = md.read_text()
    assert text.startswith("# Report\n\nintro text\n")
    assert text.count("<!-- compare_farm:A begin -->") == 1
    assert text.count("<!-- compare_farm:B begin -->") == 1
    assert "FM2p" in text and "PASS" not in "" and "intro text" in text
    # replacement changes the section content
    write_snt(test / "stemFM2p_15.snt", ONE, TWO, zero=9.0)
    assert run(ref, test, "--md", str(md), "--title", "A") == 1
    text = md.read_text()
    assert text.count("<!-- compare_farm:A begin -->") == 1
    assert "FAIL" in text and "intro text" in text
