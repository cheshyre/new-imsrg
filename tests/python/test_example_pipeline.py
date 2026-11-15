"""examples/python/he4_pipeline.py runs end to end (He4, emax 2) and agrees with the C++
driver for the same configuration. Skips when the big input files are missing."""

import runpy
from pathlib import Path

import pytest

import imsrg
from imsrg import pipeline

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "python" / "he4_pipeline.py"

CONFIG = """
[basis]
emax = 2
hw = 16.0

[reference]
nucleus = "He4"

[interaction.two_body]
file = "{me2j}"
e1max = 14
e2max = 28
lmax = 14

[hartree_fock]
mixing = "linear"

[imsrg]
method = "magnus"

[imsrg.step]
smax = 30.0

[operators]
list = ["Rp2"]

[output]
directory = "{out}"
"""


def test_example_script_matches_the_driver(me2j_file, tmp_path, capsys):
    script = runpy.run_path(str(SCRIPT))
    mine = script["run"](me2j_file, smax=30.0, verbose=True)
    assert mine["e0"] < mine["e_hf"] - 1.0
    assert "E0 =" in capsys.readouterr().out

    cfg = imsrg.load_config(text=CONFIG.format(me2j=me2j_file, out=tmp_path / "driver"))
    ref = pipeline.run(cfg)
    assert mine["e_hf"] == pytest.approx(ref.e_hf, abs=1e-10)
    assert mine["e0"] == pytest.approx(ref.e0, abs=1e-10)
    assert mine["steps"] == ref.results["imsrg"]["stages"][0]["steps"]
    assert mine["rp2_reference"] == pytest.approx(ref.operators["Rp2"]["reference"], abs=1e-9)
    assert mine["rp2_imsrg"] == pytest.approx(ref.operators["Rp2"]["imsrg"], abs=1e-9)


def test_example_script_is_documented_and_importable():
    ns = runpy.run_path(str(SCRIPT))
    assert callable(ns["run"]) and callable(ns["main"]) and ns["__doc__"]
