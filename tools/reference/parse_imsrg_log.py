#!/usr/bin/env python3
"""Parse an old-code `imsrg++` run directory into a JSON golden-data file.

Run with `python3 -I` (stdlib only):

    python3 -I parse_imsrg_log.py --rundir RUN --job IT-2 --omp-threads 8 --out result.json

The run directory holds what gen_golden_e2e.sh writes: stdout.log, stderr.log, flow.dat (flow
file, may be absent), cmd.txt, wall_time_s.txt, exit_code.txt and any int*.snt files.

All numbers are the values *as printed* by imsrg++ (6/8/9 decimals depending on the line), parsed
with float(); `*_str` keys keep the printed text.  Conventions of the flow file columns follow
design/reference/03-solvers-generators-driver.md section 5.2.  Timing/memory columns of the flow
file and the timer tables of stdout are deliberately not put in the JSON (they are not
reproducible); `wall_time_s` is the only timing.
"""
import argparse
import json
import os
import re
import sys

REF_SHA_FILE = "/Users/cheshyre/Downloads/new-imsrg-reference/REFERENCE_SRC_SHA256"
FLOW_COLS = ["istep", "s", "E0", "norm_H", "estim_err", "norm_Omega1", "norm_Omega2",
             "norm_Omega3", "norm_Eta1", "norm_Eta2", "norm_Eta3", "Ncomm", "E_MP2", "N_Ops",
             "walltime_s", "rss_MB", "slash", "max_MB"]
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def read(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m


def parse_params(lines):
    """`key => value` echo block between the two ==== lines (values kept as strings)."""
    params = {}
    for ln in lines:
        m = re.match(r"^(\S+) => (.*)$", ln)
        if m:
            params[m.group(1)] = m.group(2).strip()
        if ln.startswith("=====") and params and "Parameters" not in ln:
            break
    return params


def parse_spe_tables(lines):
    tables = []
    i = 0
    row = re.compile(r"^\s*(\d+):\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(" + NUM + r")\s+(" + NUM
                     + r")\s+(" + NUM + r")")
    while i < len(lines):
        if re.match(r"^\s*i:\s+n\s+l\s+2j\s+2tz\s+SPE", lines[i]):
            title = lines[i - 1].strip() if i > 0 else ""
            rows = []
            i += 1
            while i < len(lines):
                m = row.match(lines[i])
                if not m:
                    break
                g = m.groups()
                rows.append({"i": int(g[0]), "n": int(g[1]), "l": int(g[2]), "2j": int(g[3]),
                             "2tz": int(g[4]), "spe": float(g[5]), "occ": float(g[6]),
                             "occNAT": float(g[7]), "spe_str": g[5]})
                i += 1
            tables.append({"title": title, "rows": rows})
        else:
            i += 1
    return tables


FLOW_NAMES = FLOW_COLS[:13]  # istep .. N_Ops (timing columns are not kept)
_D9 = r"(-?\d+\.\d{9})"
# Columns are fixed-width in the writer but touch/overflow for large values (a diverged flow has
# |E0| ~ 1e9), so split by the known number formats instead: 5 decimals for s, 9 for the norms and
# energies, integers for Ncomm and N_Ops.
FLOW_RE = re.compile(r"^\s*(\d+)\s+(\d+\.\d{5})\s*" + r"\s*".join([_D9] * 9)
                     + r"\s*(\d+)\s*" + _D9 + r"\s*(\d+)\s")


def split_flow_line(ln):
    """Split one flow-file line into the 13 kept columns as strings; None if not a flow line."""
    m = FLOW_RE.match(ln)
    return list(m.groups()) if m else None


def parse_flow(path):
    text = read(path)
    if text is None:
        return None
    rows = []
    for ln in text.splitlines():
        t = split_flow_line(ln)
        if t is not None:
            rows.append(t)
    if not rows:
        return {"n_lines": 0}
    stages, prev = 1, -1
    for t in rows:
        if int(t[0]) <= prev:
            stages += 1
        prev = int(t[0])

    def conv(t):
        d = {}
        for name, v in zip(FLOW_NAMES, t):
            d[name] = int(v) if name in ("istep", "Ncomm", "N_Ops") else float(v)
        return d

    last = rows[-1]
    return {
        "n_lines": len(rows),
        "n_steps": sum(1 for t in rows if int(t[0]) >= 1),
        "n_stages": stages,
        "columns": FLOW_NAMES,
        "first": conv(rows[0]),
        "last": conv(last),
        "last_raw": " ".join(last[:13]),
        "final_s": float(last[1]),
        "final_E0": float(last[2]),
        "final_E0_str": last[2],
        "final_norm_Eta1": float(last[8]),
        "final_norm_Eta2": float(last[9]),
        "final_E_MP2": float(last[12]),
    }


def parse_snt(path):
    text = read(path)
    d = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
    m = re.search(r"Zero body term:\s*(" + NUM + ")", text or "")
    if m:
        d["zero_body_str"] = m.group(1)
        d["zero_body"] = float(m.group(1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--omp-threads", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = a.rundir

    out_text = read(os.path.join(rd, "stdout.log")) or ""
    err_text = read(os.path.join(rd, "stderr.log")) or ""
    lines = out_text.splitlines()
    res = {"job": a.job}

    sha = (read(REF_SHA_FILE) or "").strip()
    res["reference_version"] = "imsrg-devel src-sha256:" + sha if sha else None
    res["command"] = (read(os.path.join(rd, "cmd.txt")) or "").strip()
    res["omp_threads"] = a.omp_threads
    wt = (read(os.path.join(rd, "wall_time_s.txt")) or "").strip()
    res["wall_time_s"] = float(wt) if wt else None
    rc = (read(os.path.join(rd, "exit_code.txt")) or "").strip()
    res["exit_code"] = int(rc) if rc else None
    # perl alarm kill = SIGALRM = 128+14
    res["timed_out"] = res["exit_code"] in (142, 14, 137)
    res["completed"] = res["exit_code"] == 0

    res["parameters"] = parse_params(lines)

    hf = {}
    for key, pat in (("e1hf", r"^e1hf = (" + NUM + ")"), ("e2hf", r"^e2hf = (" + NUM + ")"),
                     ("e3hf", r"^e3hf = (" + NUM + ")"), ("EHF", r"^EHF = (" + NUM + ")"),
                     ("iterations", r"^HF converged after (\d+) iterations")):
        m = first(pat, out_text, re.M)
        if m:
            hf[key] = int(m.group(1)) if key == "iterations" else float(m.group(1))
            hf[key + "_str"] = m.group(1)
    res["hf"] = hf
    res["spe_tables"] = parse_spe_tables(lines)

    # Hbare 0b is the zero-body of the normal-ordered H in the final basis (HF or NAT)
    m = first(r"^Hbare 0b = (" + NUM + ")", out_text, re.M)
    if m:
        res["Hbare_0b"] = float(m.group(1))
        res["Hbare_0b_str"] = m.group(1)
    mbpt = {}
    for key, pat in (("EMP2", r"^EMP2 = (" + NUM + ")"), ("EMP2_3B", r"^EMP2_3B = (" + NUM + ")"),
                     ("E_2nd_order", r"^To 2nd order, E = (" + NUM + ")"),
                     ("E_3rd_order", r"^To 3rd order, E = (" + NUM + ")")):
        m = first(pat, out_text, re.M)
        if m:
            mbpt[key] = float(m.group(1))
            mbpt[key + "_str"] = m.group(1)
    m = first(r"E3_pp = (" + NUM + r")\s+E3_hh = (" + NUM + r")\s+E3_ph = (" + NUM
              + r")\s+EMP3 = (" + NUM + ")", out_text)
    if m:
        for k, v in zip(("E3_pp", "E3_hh", "E3_ph", "EMP3"), m.groups()):
            mbpt[k] = float(v)
            mbpt[k + "_str"] = v
    res["mbpt"] = mbpt

    res["generators"] = [m.group(1).strip() for m in
                         (re.match(r"^Setting generator to (.*)$", ln) for ln in lines) if m]
    pushes = [int(m.group(1)) for m in
              (re.match(r"^pushing back another Omega\. Omega\.size = (\d+)", ln) for ln in lines)
              if m]
    res["omega_pushed"] = len(pushes)
    res["omega_size_final"] = pushes[-1] if pushes else 0

    m = first(r"^Core Energy = (" + NUM + ")", out_text, re.M)
    if m:
        res["core_energy"] = float(m.group(1))
        res["core_energy_str"] = m.group(1)
    m = first(r"^Perturbative triples:\s+(\S+) -> (\S+)", out_text, re.M)
    if m:
        # inf/nan are not valid JSON numbers: keep only the printed string then
        def fin(t):
            v = float(t)
            return v if v - v == 0 else None
        res["perturbative_triples"] = {"dE": fin(m.group(1)), "dE_str": m.group(1),
                                       "E0_plus_dE": fin(m.group(2)),
                                       "E0_plus_dE_str": m.group(2)}
    norms3 = re.findall(r"^Norm of 3-body = (" + NUM + ")", out_text, re.M)
    if norms3:
        res["norm_of_3body_str"] = norms3

    # transformed operators: "<i>: <name> " followed by "   HF: <value>"
    ops = []
    for idx, ln in enumerate(lines):
        m = re.match(r"^(\d+): (\S+)\s*$", ln)
        if m and idx > 0:
            op = {"index": int(m.group(1)), "name": m.group(2)}
            for ln2 in lines[idx + 1: idx + 6]:
                m2 = re.match(r"^\s+HF: (" + NUM + ")", ln2)
                if m2:
                    op["HF_value"] = float(m2.group(1))
                    op["HF_value_str"] = m2.group(1)
                    break
            ops.append(op)
    if ops:
        res["operators"] = ops

    res["flow"] = parse_flow(os.path.join(rd, "flow.dat"))

    snts = sorted(f for f in os.listdir(rd) if f.endswith(".snt"))
    if snts:
        res["snt_files"] = [parse_snt(os.path.join(rd, f)) for f in snts]

    warn = [ln.strip() for ln in lines
            if re.search(r"warning|error|nan\b|unkown parameter|unknown parameter", ln, re.I)
            and "=>" not in ln]
    res["stdout_warning_lines"] = warn
    res["stderr_lines"] = [ln for ln in err_text.splitlines() if ln.strip()][:20]

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, sort_keys=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
