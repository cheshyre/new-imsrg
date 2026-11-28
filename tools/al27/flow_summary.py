#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Markdown summary of one or more 27Al flow run directories.

usage: flow_summary.py RUN_DIR [RUN_DIR ...]

Reads, per RUN_DIR: flow.txt (CSV flow file; for a restarted run duplicate
(stage, istep) rows are resolved by keeping the last one; ODE flows, where
istep stays 0, keep every row), steps = last istep for Magnus, results.json (final
E0), the Omega files (<stem>_omegas.toml, omega_*.opb), and the job log
flow-gpu.log (or flow[0-9]*.log if there is none): per link the start/end lines,
"Maximum resident set size" and "Elapsed (wall clock) time" of /usr/bin/time -v.
The stage wall time is the wall_s of the stage's last row (wall_s counts from
the start of the process that wrote the row; after a restart it restarts, so
for a stage split over links the figure is that of the last link only).
Writes markdown to stdout.
"""
import csv
import glob
import json
import os
import re
import sys


def read_flow(path):
    """Return {stage: [row dicts]} with restart duplicates resolved.

    Magnus flows number their steps (istep 0 = initial state): rows are keyed
    by (stage, istep), later rows (a restarted run) overwrite earlier ones.
    ODE flows keep istep 0 in every row (accepted and trial steps alike):
    a stage whose rows all have the same istep is kept row by row.
    """
    raw = {}
    for r in _rows(path):
        raw.setdefault(r[0], []).append(r[1:])
    stages = {}
    for st, lst in sorted(raw.items()):
        if len({i for i, _ in lst}) == 1 and len(lst) > 1:
            stages[st] = [r for _, r in lst]
            continue
        keyed = {}
        for i, r in lst:
            keyed[i] = r
        stages[st] = [keyed[i] for i in sorted(keyed)]
    return stages


def _rows(path):
    """Yield (stage, istep, row) for every parsable row of a flow file."""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                yield int(r["stage"]), int(r["istep"]), r
            except (KeyError, TypeError, ValueError):
                continue


def is_ode_stage(rs):
    """True if all rows of the stage have the same istep (ODE flow)."""
    return len(rs) > 1 and len({r["istep"] for r in rs}) == 1


def steps_text(rs):
    """Number of steps of a stage as text.

    Magnus: the last step index (istep 0 is the initial state, so a stage
    with rows 0..54 has 54 steps). ODE: the flow file lists trial steps as
    well and has no step index, so only the row count is given.
    """
    if is_ode_stage(rs):
        return "{} rows (ODE)".format(len(rs))
    return str(max(int(r["istep"]) for r in rs))


def fnum(r, k, fmt="{:.6g}"):
    try:
        return fmt.format(float(r[k]))
    except (KeyError, TypeError, ValueError):
        return "-"


def stage_table(stages):
    out = ["| stage | steps | final s | final E0 (MeV) | norm_H | norm_eta |"
           " norm_omega | num_omegas | wall (s) |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for st, rs in stages.items():
        last = rs[-1]
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            st, steps_text(rs), fnum(last, "s", "{:.5f}"),
            fnum(last, "E0", "{:.6f}"), fnum(last, "norm_H", "{:.6f}"),
            fnum(last, "norm_eta", "{:.3e}"),
            fnum(last, "norm_omega", "{:.5f}"),
            fnum(last, "num_omegas", "{:.0f}"), fnum(last, "wall_s", "{:.1f}")))
    return out


def parse_logs(run):
    """Return a list of per-link dicts from the job log(s) of a run.

    A link starts at a line "link N job J start T" (flows_chain.sh) or
    "start T [job J]" (flows_gpu.sh, run_flow.sh) and ends with
    "[link N job J ]end T status S".
    """
    logs = [os.path.join(run, "flow-gpu.log")]
    if not os.path.exists(logs[0]):
        logs = sorted(glob.glob(os.path.join(run, "flow[0-9]*.log")))
    links = []
    for lg in logs:
        cur = None
        try:
            with open(lg, errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for ln in lines:
            s = ln.strip()
            m = re.match(r"link (\d+) job (\S+) start (\S+)", s)
            if not m:
                m = re.match(r"()start (\S+)(?: job (\S+))?", s)
                if m:
                    m = (None, m.group(3), m.group(2))
                else:
                    m = None
            if m:
                g = m.groups() if hasattr(m, "groups") else m
                cur = {"log": os.path.basename(lg), "link": g[0] or None,
                       "job": g[1], "start": g[2], "end": None,
                       "status": None, "rss_kb": None, "elapsed": None}
                links.append(cur)
                continue
            if cur is None:
                continue
            if s.startswith("Maximum resident set size"):
                cur["rss_kb"] = int(s.rsplit(":", 1)[1])
            elif s.startswith("Elapsed (wall clock)"):
                cur["elapsed"] = s.rsplit(": ", 1)[1]
            else:
                me = re.match(r"(?:link \d+ job \S+ )?end (\S+) status (\S+)", s)
                if me:
                    cur["end"], cur["status"] = me.groups()
    return links


def summarize(run):
    run = os.path.normpath(run)
    out = ["## " + run, ""]
    flow = os.path.join(run, "flow.txt")
    if os.path.exists(flow):
        out += stage_table(read_flow(flow)) + [""]
    else:
        out += ["(no flow.txt)", ""]
    res = os.path.join(run, "results.json")
    if os.path.exists(res):
        with open(res) as f:
            j = json.load(f)
        im = j.get("imsrg", {})
        out.append("- final E0 (results.json): {:.6f} MeV; E_ref {:.6f}; "
                   "stages: {}".format(
                       im.get("e0", float("nan")),
                       j.get("e_reference", float("nan")),
                       ", ".join("{} ({} steps, stop {})".format(
                           s.get("generator"), s.get("steps"), s.get("stop"))
                           for s in im.get("stages", []))))
    else:
        out.append("- results.json: not present (flow not finished)")
    nom = len(glob.glob(os.path.join(run, "*omega_*.opb")))
    tomls = sorted(glob.glob(os.path.join(run, "*_omegas.toml")))
    out.append("- Omegas written: {} .opb file(s) in the run dir; manifests: {}"
               .format(nom, ", ".join(os.path.basename(t) for t in tomls)
                       or "none"))
    links = parse_logs(run)
    if links:
        out += ["", "| log | link | job | start | end | status | max RSS (GB) |"
                " elapsed |", "|---|---:|---|---|---|---:|---:|---|"]
        for lk in links:
            rss = "-" if lk["rss_kb"] is None else "{:.1f}".format(
                lk["rss_kb"] / 1.0e6)
            out.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lk["log"], lk["link"] or "-", lk["job"] or "-",
                lk["start"] or "-", lk["end"] or "-",
                lk["status"] if lk["status"] is not None else "-", rss,
                lk["elapsed"] or "-"))
    for name in ("FLOW_DONE", "FLOW_FAILED", "FLOW_ERRORS"):
        if os.path.exists(os.path.join(run, name)):
            out.append("- marker {} present".format(name))
    out.append("")
    return "\n".join(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    print("# 27Al flow summary\n")
    for run in argv[1:]:
        print(summarize(run))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
