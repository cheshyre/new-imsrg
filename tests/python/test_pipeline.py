"""pipeline: load_config / RunConfig -> run -> results, the way users run pipelines."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import imsrg
from imsrg import pipeline


def he4_toml(me2j_file, out_dir, task="imsrg", extra=""):
    return f"""
[run]
task = "{task}"

[basis]
emax = 2
hw = 16.0

[reference]
nucleus = "He4"

[interaction.two_body]
file = "{me2j_file}"
e1max = 14
e2max = 28
lmax = 14

[hartree_fock]
mixing = "linear"

[imsrg]
method = "magnus"

{extra}
[output]
directory = "{out_dir}"
"""


def test_run_config_surface(tmp_path):
    cfg = imsrg.load_config(text=he4_toml("/nonexistent/v.me2j", tmp_path / "out"))
    assert isinstance(cfg, imsrg.RunConfig)
    assert cfg.output_directory == str(tmp_path / "out")
    assert cfg.flow_file.startswith(str(tmp_path / "out"))
    assert cfg.to_dict()["basis"]["emax"] == 2
    # inputs are checked on request: the missing interaction file is a config error
    with pytest.raises(imsrg.ConfigError):
        imsrg.load_config(text=he4_toml("/nonexistent/v.me2j", tmp_path / "out"), check_files=True)
    # without the check, run() reports it as an input error and still leaves the resolved config
    with pytest.raises(imsrg.InputError):
        pipeline.run(cfg)
    assert (tmp_path / "out" / "resolved_config.toml").exists()


def test_hf_task(me2j_file, tmp_path):
    cfg = imsrg.load_config(text=he4_toml(me2j_file, tmp_path / "hf", task="hf"))
    res = pipeline.run(cfg)
    assert res.directory == tmp_path / "hf"
    assert res.e_hf == pytest.approx(res.e_reference)
    assert res.e0 == res.e_reference and res.results["imsrg"]["stages"] == []  # no flow was run
    assert res.results["hf"]["converged"] and res.valence_interaction is None
    assert res.operators == {}


def test_imsrg_run_results_and_flow_file(me2j_file, tmp_path, he4_emax2):
    cfg = imsrg.load_config(text=he4_toml(me2j_file, tmp_path / "run", extra='[operators]\nlist = ["Rp2"]\n'))
    res = pipeline.run(cfg)
    assert res.e_hf == pytest.approx(he4_emax2[2].e_hf, abs=1e-10)
    assert res.e_mp2 < 0 and res.e0 < res.e_reference
    assert res.results["imsrg"]["stages"][0]["stop"] in ("converged", "Converged", "converged ")
    assert set(res.operators) == {"Rp2"}
    assert res.operators["Rp2"]["imsrg"] > 0 and res.operators["Rp2"]["reference"] > 0
    flow = res.flow()
    assert {"s", "E0"} <= set(flow)
    assert flow["E0"][0] == pytest.approx(res.e_reference, abs=1e-8)
    assert flow["E0"][-1] == pytest.approx(res.e0, abs=1e-10)
    assert np.all(np.diff(flow["s"]) >= 0)
    assert (res.directory / "resolved_config.toml").exists()
    json.loads((res.directory / "results.json").read_text())
    # run() also accepts a path to a TOML file
    p = tmp_path / "cfg.toml"
    p.write_text(he4_toml(me2j_file, tmp_path / "from_file", task="hf"))
    assert pipeline.run(p).e_hf == pytest.approx(res.e_hf, abs=1e-12)


def test_python_m_imsrg_run(me2j_file, tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(he4_toml(me2j_file, tmp_path / "cli"))
    env = dict(os.environ)
    root = str(Path(imsrg.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-m", "imsrg", "run", str(p), "--set", "output.directory=" + str(tmp_path / "cli2")],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "E0" in out.stdout and (tmp_path / "cli2" / "results.json").exists()
