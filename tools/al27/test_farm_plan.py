# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Tests of tools/al27/farm_plan.py (run: .venv/bin/pytest tools/al27)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_plan as fp  # noqa: E402

CFG = fp.load_costs(fp.DEFAULT_COSTS)


def uniform_cfg(t=10.0, rpn=8, setup=60):
    sec = {"L%d" % i: t for i in range(6)}
    sec["TCM"] = t
    return {"model": {"weights": [1.0] * 6, "tcm_weight": 1.0},
            "cpu": {"ranks_per_node": rpn, "setup_s": setup, "seconds": sec}}


def test_assign_static_hand_example_with_ties():
    # weights 3, 5, 5, 2, 3 on 2 ranks; order (stable): 1 2 0 4 3
    # 1->r0 (5,0); 2->r1 (5,5); 0->r0 (8,5); 4->r1 (8,8); 3->r0 (10,8)
    assert fp.assign_static([3, 5, 5, 2, 3], 2) == [0, 0, 1, 0, 1]


def test_assign_static_single_rank_and_more_ranks_than_items():
    assert fp.assign_static([1, 2, 3], 1) == [0, 0, 0]
    assert fp.assign_static([1, 2, 3], 5) == [2, 1, 0]


def test_item_kind_and_weight():
    assert fp.item_kind("TCM") == ("TCM", "TCM")
    assert fp.item_kind("FSigmapp3n_45") == ("L3", "FSigmapp3")
    assert fp.item_kind("FM0p_0") == ("L0", "FM0")
    # The weights come from tools/al27/farm_costs.toml (refit to measured
    # item times); check the lookup, not the fitted values.
    model = CFG["model"]
    assert fp.item_weight("FSigmap5p_15", model) == model["weights"][5]
    assert fp.item_weight("FM0p_0", model) == model["weights"][0]
    assert fp.item_weight("TCM", model) == model["tcm_weight"]


def test_family_override():
    sec = {"L5": 1.0, "FSigmap5": 9.0}
    assert fp.item_seconds("FSigmap5p_0", sec) == 9.0
    assert fp.item_seconds("FDelta5p_0", sec) == 1.0


def test_uniform_makespan_is_ceil_items_over_ranks_times_t():
    cfg = uniform_cfg(10.0)
    for items in (1, 7, 8, 9, 100, 2415):
        names = ["FM0p_%d" % i for i in range(items)]
        for nodes in (1, 3, 40):
            s = fp.plan_step(names, nodes, cfg, "cpu")
            ranks = min(nodes * 8, items)
            assert s["ranks"] == ranks
            assert math.isclose(s["makespan_h"] * 3600,
                                math.ceil(items / ranks) * 10.0)


def test_target_hours_minimal_nodes():
    cfg = uniform_cfg(10.0)
    names = ["FM0p_%d" % i for i in range(100)]
    # makespan <= 50 s: ceil(100/ranks) <= 5 -> ranks >= 20 -> 3 nodes
    n, ok = fp.nodes_for_target(names, 50 / 3600.0, cfg, "cpu")
    assert ok and n == 3
    assert fp.plan_step(names, 2, cfg, "cpu")["makespan_h"] * 3600 > 50
    n, ok = fp.nodes_for_target(names, 5 / 3600.0, cfg, "cpu")
    assert not ok and n == 13  # ceil(100 / 8)
    n, ok = fp.nodes_for_target(names, 5 / 3600.0, cfg, "cpu", max_nodes=4)
    assert not ok and n == 4


def test_walltime_rounding_and_cap():
    assert fp.round_walltime(1) == (1800, False)
    assert fp.round_walltime(1800) == (1800, False)
    assert fp.round_walltime(1801) == (2400, False)
    assert fp.round_walltime(3600 * 12) == (43200, False)
    assert fp.round_walltime(3600 * 12 + 1) == (43200, True)
    assert fp.fmt_wall(5400) == "01:30:00"


def test_plan_step_walltime_is_twice_makespan_plus_setup():
    cfg = uniform_cfg(600.0, setup=60)
    names = ["FM0p_%d" % i for i in range(8)]
    s = fp.plan_step(names, 1, cfg, "cpu")
    # makespan 600 s: 2 x 660 = 1320 s -> minimum 30 min
    assert s["walltime"] == "00:30:00"
    assert math.isclose(s["idle_fraction"], 0.0, abs_tol=1e-12)


def test_plan_job_sums_nodes_and_takes_max_walltime():
    cfg = uniform_cfg(600.0)
    small = ["FM0p_%d" % i for i in range(8)]
    big = ["FM0p_%d" % i for i in range(80)]
    job = fp.plan_job([("a", small), ("b", big)], cfg, "cpu", nodes=1)
    assert job["nodes"] == 2
    assert job["walltime_s"] == job["steps"][1]["walltime_s"]
    assert job["steps"][1]["walltime_s"] >= job["steps"][0]["walltime_s"]
