#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Job sizing of the 27Al response farm (design/50 section 6).

Simulates the binary's static assignment (farm.cc AssignStatic: items longest
first by model weight, ties lower index first, each to the rank with the
smallest load, ties lower rank), evaluates the makespan with the measured
seconds of tools/al27/farm_costs.toml and recommends nodes and walltime.

  farm_plan.py --backend cpu --work-list W [--work-list W2 ...]
               (--nodes N | --target-hours H) [--max-nodes M] [--json]

One work list = one srun step. --nodes / --target-hours apply to each list;
several lists are the steps of one job: nodes add up, walltime is the max.
Needs Python >= 3.11 (tomllib): use $IMSRG_DEPS_PREFIX/venv/bin/python.
"""
import argparse
import json
import math
import os
import re
import sys

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

DEFAULT_COSTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "farm_costs.toml")
MAX_WALL_S = 12 * 3600
_NAME_RE = re.compile(r"^F([A-Za-z]+?)(\d)[pn]_\d+$")


def load_costs(path):
    """Returns the parsed cost file."""
    if tomllib is None:
        sys.exit("farm_plan.py needs Python >= 3.11 (tomllib); use the venv")
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_work_list(path):
    """Item names of a work list (first field; # lines and blanks skipped)."""
    names = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s.split()[0])
    return names


def item_kind(name):
    """Returns (kind, family): kind 'L0'..'L5' or 'TCM'; family 'FSigmap5'."""
    if name == "TCM":
        return "TCM", "TCM"
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError("unknown item name: " + name)
    return "L" + m.group(2), "F" + m.group(1) + m.group(2)


def item_weight(name, model):
    """Weight the binary assigns by (model weights; TCM is a scalar)."""
    kind, _ = item_kind(name)
    if kind == "TCM":
        return float(model["tcm_weight"])
    return float(model["weights"][int(kind[1:])])


def item_seconds(name, seconds):
    """Seconds of one item (family override, else its kind)."""
    kind, family = item_kind(name)
    return float(seconds.get(family, seconds[kind]))


def assign_static(weights, num_ranks):
    """Port of farm.cc AssignStatic: owner rank of each item.

    Longest first (stable: ties lower index first), each to the rank with the
    smallest load (ties lower rank).
    """
    order = sorted(range(len(weights)), key=lambda i: -weights[i])
    load = [0.0] * num_ranks
    owner = [0] * len(weights)
    for i in order:
        r = min(range(num_ranks), key=lambda k: load[k])  # first minimum
        owner[i] = r
        load[r] += weights[i]
    return owner


def round_walltime(seconds):
    """Rounds up to 10 min, at least 30 min. Returns (seconds, capped)."""
    s = max(math.ceil(seconds / 600.0) * 600, 1800)
    if s > MAX_WALL_S:
        return MAX_WALL_S, True
    return s, False


def fmt_wall(seconds):
    """HH:MM:SS of whole minutes."""
    return "%02d:%02d:00" % (seconds // 3600, (seconds % 3600) // 60)


def plan_step(names, nodes, cfg, backend):
    """Plans one step (one work list) on `nodes` nodes; returns a dict."""
    b = cfg[backend]
    rpn = int(b["ranks_per_node"])
    items = len(names)
    ranks = max(1, min(nodes * rpn, items))
    weights = [item_weight(n, cfg["model"]) for n in names]
    owner = assign_static(weights, ranks)
    load = [0.0] * ranks
    for n, r in zip(names, owner):
        load[r] += item_seconds(n, b["seconds"])
    makespan = max(load) if items else 0.0
    total = sum(load)
    allocated = nodes * rpn
    idle = 1.0 - total / (allocated * makespan) if makespan > 0 else 0.0
    setup = float(b["setup_s"])
    wall, capped = round_walltime(2.0 * (makespan + setup))
    return {
        "items": items, "nodes": nodes, "ranks": ranks,
        "makespan_h": makespan / 3600.0, "setup_s": setup,
        "walltime_s": wall, "walltime": fmt_wall(wall),
        "walltime_capped": capped,
        "node_hours_walltime": nodes * wall / 3600.0,
        "node_hours_makespan": nodes * makespan / 3600.0,
        "idle_fraction": idle,
    }


def nodes_for_target(names, target_h, cfg, backend, max_nodes=None):
    """Smallest node count with makespan <= target_h (binary search).

    Capped by max_nodes and by ceil(items / ranks_per_node) (more nodes than
    one item per rank cannot help). Returns (nodes, reached).
    """
    rpn = int(cfg[backend]["ranks_per_node"])
    hi = max(1, math.ceil(len(names) / rpn))
    if max_nodes is not None:
        hi = max(1, min(hi, max_nodes))

    def ok(n):
        return plan_step(names, n, cfg, backend)["makespan_h"] <= target_h

    if not ok(hi):
        return hi, False
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo, True


def plan_job(lists, cfg, backend, nodes=None, target_hours=None,
             max_nodes=None):
    """Plans a job of several steps; returns a dict with steps and totals."""
    steps, warnings = [], []
    for path, names in lists:
        if target_hours is not None:
            n, reached = nodes_for_target(names, target_hours, cfg, backend,
                                          max_nodes)
            if not reached:
                warnings.append("%s: target %.3f h not reachable with %d "
                                "nodes" % (path, target_hours, n))
        else:
            n = nodes
        s = plan_step(names, n, cfg, backend)
        s["work_list"] = path
        if s["walltime_capped"]:
            warnings.append("%s: walltime capped at 12:00:00 (needs more)"
                            % path)
        steps.append(s)
    wall = max(s["walltime_s"] for s in steps)
    total_nodes = sum(s["nodes"] for s in steps)
    return {
        "backend": backend, "steps": steps, "nodes": total_nodes,
        "walltime_s": wall, "walltime": fmt_wall(wall),
        "node_hours_walltime": total_nodes * wall / 3600.0,
        "node_hours_makespan": sum(s["node_hours_makespan"] for s in steps),
        "warnings": warnings,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=("cpu", "hip"), default="cpu")
    p.add_argument("--work-list", action="append", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--nodes", type=int)
    g.add_argument("--target-hours", type=float)
    p.add_argument("--max-nodes", type=int)
    p.add_argument("--costs", default=DEFAULT_COSTS)
    p.add_argument("--json", action="store_true", help="JSON line only")
    a = p.parse_args(argv)
    cfg = load_costs(a.costs)
    lists = [(w, read_work_list(w)) for w in a.work_list]
    job = plan_job(lists, cfg, a.backend, a.nodes, a.target_hours,
                   a.max_nodes)
    if not a.json:
        for s in job["steps"]:
            print("step %s: items %d nodes %d ranks %d makespan %.3f h "
                  "walltime %s idle %.1f%%" % (
                      s["work_list"], s["items"], s["nodes"], s["ranks"],
                      s["makespan_h"], s["walltime"],
                      100 * s["idle_fraction"]))
        print("job (%s): nodes %d walltime %s node-hours %.1f (walltime) "
              "%.1f (makespan)" % (
                  job["backend"], job["nodes"], job["walltime"],
                  job["node_hours_walltime"], job["node_hours_makespan"]))
        for w in job["warnings"]:
            print("WARNING: " + w)
    print(json.dumps(job, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
