# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""pytest for flow_summary.py with synthetic run directories."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_summary as fs  # noqa: E402

HEADER = ("stage,istep,s,ds,E0,norm_H,norm_eta_1b,norm_eta_2b,norm_eta,"
          "norm_omega_1b,norm_omega_2b,norm_omega,num_omegas,E_MP2,wall_s\n")


def row(stage, istep, e0, wall):
    return (f"{stage},{istep},{istep * 0.5},0.5,{e0},100.0,0,0,0.01,0,0,"
            f"1.5,2,-3.0,{wall}\n")


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    # first process wrote steps 0..3 of stage 1 and 0..2 of stage 2 (killed
    # after step 2 of stage 2); the restart rewrites step 2 and continues.
    txt = HEADER
    for i in range(4):
        txt += row(1, i, -10.0 - i, 1.0 + i)
    for i in range(3):
        txt += row(2, i, -20.0 - i, 10.0 + i)
    for i in (2, 3, 4):
        txt += row(2, i, -30.0 - i, 5.0 + i)  # restarted: duplicate step 2
    (run / "flow.txt").write_text(txt)
    (run / "results.json").write_text(json.dumps({
        "e_reference": -5.0,
        "imsrg": {"e0": -34.0, "stages": [
            {"generator": "atan", "steps": 3, "stop": "converged"},
            {"generator": "shell_model_atan", "steps": 4,
             "stop": "converged"}]}}))
    (run / "X_omega_000.opb").write_text("x")
    (run / "X_omega_001.opb").write_text("x")
    (run / "X_omegas.toml").write_text("n = 2\n")
    (run / "flow-gpu.log").write_text(
        "link 1 job 11 start 2026-01-01T00:00:00 \n"
        "\tMaximum resident set size (kbytes): 2000000\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:00:00\n"
        "link 1 job 11 end 2026-01-01T01:00:00 status 75\n"
        "link 2 job 12 start 2026-01-01T02:00:00\n"
        "\tMaximum resident set size (kbytes): 3000000\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30:00\n"
        "link 2 job 12 end 2026-01-01T02:30:00 status 0\n")
    return run


def test_duplicates_keep_last(tmp_path):
    stages = fs.read_flow(str(make_run(tmp_path) / "flow.txt"))
    assert [len(stages[1]), len(stages[2])] == [4, 5]
    st2 = {int(r["istep"]): float(r["E0"]) for r in stages[2]}
    assert st2[2] == -32.0  # the restarted run's value, not -22.0
    assert float(stages[2][-1]["wall_s"]) == 9.0


def test_logs(tmp_path):
    links = fs.parse_logs(str(make_run(tmp_path)))
    assert [lk["link"] for lk in links] == ["1", "2"]
    assert [lk["status"] for lk in links] == ["75", "0"]
    assert links[1]["rss_kb"] == 3000000
    assert links[0]["elapsed"] == "1:00:00"


def test_summary_text(tmp_path):
    txt = fs.summarize(str(make_run(tmp_path)))
    assert "| 1 | 3 |" in txt  # rows istep 0..3: 3 steps
    assert "| 2 | 4 |" in txt  # rows istep 0..4: 4 steps
    assert "-34.000000" in txt
    assert "2 .opb file(s)" in txt
    assert "X_omegas.toml" in txt
    assert "| 75 |" in txt


def test_legacy_log_without_link(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    (run / "flow1.log").write_text(
        "start 2026-01-01T00:00:00-05:00 job 7\n"
        "end 2026-01-01T00:10:00-05:00 status 143\n")
    links = fs.parse_logs(str(run))
    assert len(links) == 1 and links[0]["status"] == "143"
    assert links[0]["job"] == "7"


def test_no_files(tmp_path):
    txt = fs.summarize(str(tmp_path))
    assert "no flow.txt" in txt and "not present" in txt


def test_steps_magnus_is_last_istep(tmp_path):
    stages = fs.read_flow(str(make_run(tmp_path) / "flow.txt"))
    assert fs.steps_text(stages[1]) == "3"  # 4 rows, istep 0..3
    assert fs.steps_text(stages[2]) == "4"  # 5 rows, istep 0..4


def test_ode_rows_not_merged(tmp_path):
    txt = HEADER
    for i in range(6):
        txt += row(1, 0, -10.0 - i, 1.0 + i)  # ODE: istep stays 0
    for i in range(4):
        txt += row(2, 0, -20.0 - i, 10.0 + i)
    path = tmp_path / "flow.txt"
    path.write_text(txt)
    stages = fs.read_flow(str(path))
    assert [len(stages[1]), len(stages[2])] == [6, 4]
    assert fs.steps_text(stages[1]) == "6 rows (ODE)"
    run = tmp_path / "ode"
    run.mkdir()
    (run / "flow.txt").write_text(txt)
    assert "| 2 | 4 rows (ODE) |" in fs.summarize(str(run))
