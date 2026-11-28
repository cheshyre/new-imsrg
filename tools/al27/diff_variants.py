#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Element-by-element comparison of two variants of the 27Al calculation.

  diff_variants.py --ref-flow D --ref-farm D --test-flow D --test-farm D
      --label "v2 vs v1, N2LOsat" [--md out.md] [--top 10] [--small 1e-6]
      [--jobs N] [--rel-flag 1e-3]

REF and TEST are each a flow directory (`<stem>.snt` = evolved valence H,
`<stem>[_]TCM.snt`, `results.json`, `flow.txt`) and a farm directory (flat
directory of tokyo .snt files, one per operator column and q, as read by
compare_farm.scan_flat). Any of the four directories may be omitted; what
is given is compared. Output: a markdown section (stdout, or --md file).

Conventions
* absolute difference |test - ref| for every element (zero-body, 1b, 2b);
* relative difference |test - ref| / |ref| only where |ref| >= --small
  (default 1e-6); elements with |ref| below it are counted as "small"
  and excluded from every relative statistic;
* --rel-flag (default 1e-3) counts elements with relative diff above it;
  the per-column table repeats max rel and the count for |ref| >= 1e-3 (BIG);
* orbits are matched by labels (n, l, 2j, 2tz), not by index; scalar files
  use canonical 2b keys (compare_snt.py), tensor files (`_2b.snt`, rows with
  more than five integer columns) keep their keys as written;
* an element present in only one file compares against 0 (counted as
  "only ref" / "only test");
* file names `<stem>F<Fam><L><p|n>_<q>[_2b].snt` and `<stem>TCM.snt` (v1) or
  `<stem>_F...` / `<stem>_TCM.snt` (v2/v3) are paired by (operator, q); the
  first TCM file of a farm directory is used (the number of copies is noted).

Exit status: 0 normally (a report, not pass/fail); 2 if inputs are missing
or unparseable (listed in the output).
"""

import argparse
import heapq
import json
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_farm as cf  # noqa: E402

SECTIONS = ("0b", "1b", "2b")
# Second relative statistic only over sizeable elements, |ref| >= BIG: the
# relative maxima over |ref| >= --small are dominated by elements near 1e-6.
BIG = 1.0e-3
_CS = None


def _cs():
    """compare_snt module (loaded once per process)."""
    global _CS
    if _CS is None:
        _CS = cf._load_compare_snt()
    return _CS


# ------------------------------------------------------------ file diffs --
def diff_elements(ref, test, small, rel_flag, topk):
    """Compare two {key: value} dicts; returns a stats dict.

    Keys: n, max_abs, abs_key, max_rel, rel_key, n_flag, n_small, only_ref,
    only_test, max_ref, top_abs / top_rel (lists of (value, key, ref, test)).
    """
    st = {"n": 0, "max_abs": 0.0, "abs_key": None, "max_rel": 0.0,
          "rel_key": None, "n_flag": 0, "n_small": 0, "only_ref": 0,
          "only_test": 0, "max_ref": 0.0, "max_rel_big": 0.0, "n_flag_big": 0, "n_big": 0}
    cand_abs, cand_rel = [], []
    for k in set(ref) | set(test):
        r, t = ref.get(k), test.get(k)
        if r is None:
            st["only_test"] += 1
            r = 0.0
        elif t is None:
            st["only_ref"] += 1
            t = 0.0
        st["n"] += 1
        ar = abs(r)
        d = abs(t - r)
        if ar > st["max_ref"]:
            st["max_ref"] = ar
        if d > st["max_abs"] or st["abs_key"] is None:
            st["max_abs"], st["abs_key"] = d, k
        cand_abs.append((d, k, r, t))
        if len(cand_abs) > 8 * topk + 64:
            cand_abs = heapq.nlargest(topk, cand_abs, key=lambda c: c[0])
        if ar < small:
            st["n_small"] += 1
            continue
        rel = d / ar
        if rel > st["max_rel"] or st["rel_key"] is None:
            st["max_rel"], st["rel_key"] = rel, k
        if rel > rel_flag:
            st["n_flag"] += 1
        if ar >= BIG:
            st["n_big"] += 1
            st["max_rel_big"] = max(st["max_rel_big"], rel)
            if rel > rel_flag:
                st["n_flag_big"] += 1
        cand_rel.append((rel, k, r, t))
        if len(cand_rel) > 8 * topk + 64:
            cand_rel = heapq.nlargest(topk, cand_rel, key=lambda c: c[0])
    st["top_abs"] = heapq.nlargest(topk, cand_abs, key=lambda c: c[0])
    st["top_rel"] = heapq.nlargest(topk, cand_rel, key=lambda c: c[0])
    return st


def compare_pair(task):
    """Worker: compare one file pair; returns {"col", "q", "secs": ...}."""
    col, q, rpath, tpath, small, rel_flag, topk = task
    res = {"col": col, "q": q, "ref": rpath, "test": tpath}
    try:
        r = cf.parse_snt(rpath)
        t = cf.parse_snt(tpath)
        if r.scalar and t.scalar:
            cf.canonicalize(r, _cs())
            cf.canonicalize(t, _cs())
    except (ValueError, OSError, IndexError) as e:
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    res["tensor"] = not (r.scalar and t.scalar)
    secs = {"0b": diff_elements({(): r.zero}, {(): t.zero}, small,
                                rel_flag, topk),
            "1b": diff_elements(r.one, t.one, small, rel_flag, topk),
            "2b": diff_elements(r.two, t.two, small, rel_flag, topk)}
    res["secs"] = secs
    return res


def merge_stats(secs):
    """Merge per-section stats dicts into one over all sections."""
    tot = {"n": 0, "max_abs": 0.0, "abs_key": None, "max_rel": 0.0,
           "rel_key": None, "n_flag": 0, "n_small": 0, "only_ref": 0,
           "only_test": 0, "max_ref": 0.0, "max_rel_big": 0.0,
           "n_flag_big": 0, "n_big": 0}
    for name, s in secs.items():
        for f in ("n", "n_flag", "n_small", "only_ref", "only_test"):
            tot[f] += s[f]
        tot["max_ref"] = max(tot["max_ref"], s["max_ref"])
        tot["max_rel_big"] = max(tot["max_rel_big"], s["max_rel_big"])
        tot["n_flag_big"] += s["n_flag_big"]
        tot["n_big"] += s["n_big"]
        if s["max_abs"] > tot["max_abs"] or tot["abs_key"] is None:
            tot["max_abs"], tot["abs_key"] = s["max_abs"], (name, s["abs_key"])
        if s["rel_key"] is not None and (s["max_rel"] > tot["max_rel"]
                                         or tot["rel_key"] is None):
            tot["max_rel"], tot["rel_key"] = s["max_rel"], (name,
                                                            s["rel_key"])
    return tot


# ---------------------------------------------------------------- inputs --
def scan_farm(directory):
    """Return ({(col, q): path}, n_tcm); TCM is (`TCM`, None), first copy."""
    ops, tcm = cf.scan_flat(directory)
    files = {(op, q): p for (op, q), p in ops.items()}
    if tcm:
        files[("TCM", None)] = tcm[0]
    return files, len(tcm)


def flow_files(directory):
    """Return {col: path} for the H and TCM files of a flow directory."""
    out = {}
    names = sorted(n for n in os.listdir(directory) if n.endswith(".snt"))
    stem = None
    try:
        with open(os.path.join(directory, "results.json")) as f:
            vi = json.load(f).get("valence_interaction")
        if vi and os.path.basename(vi) in names:
            stem = os.path.basename(vi)
    except (OSError, ValueError):
        pass
    for n in names:
        if cf.TCM_RE.search(n):
            out.setdefault("TCM(flow)", os.path.join(directory, n))
        elif stem is None and not cf.OP_RE.search(n):
            stem = n
    if stem:
        out["H"] = os.path.join(directory, stem)
    return out


def flow_info(directory):
    """Parse results.json and flow.txt; returns (info dict, [errors])."""
    info, errs = {}, []
    p = os.path.join(directory, "results.json")
    try:
        with open(p) as f:
            d = json.load(f)
        im = d.get("imsrg", {})
        info["e0"] = im.get("e0")
        info["num_omegas_json"] = im.get("num_omegas")
        info["stages_json"] = [(s.get("stage"), s.get("steps"), s.get("s"),
                                s.get("e0")) for s in im.get("stages", [])]
    except (OSError, ValueError) as e:
        errs.append(f"{p}: {e}")
    p = os.path.join(directory, "flow.txt")
    try:
        with open(p) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        head = lines[0].split(",")
        last = {}
        for ln in lines[1:]:
            row = dict(zip(head, ln.split(",")))
            last[int(row["stage"])] = row
        info["stages_txt"] = [(st, int(r["istep"]), float(r["s"]),
                               float(r["E0"])) for st, r in
                              sorted(last.items())]
        if last:
            info["num_omegas_txt"] = int(last[max(last)]["num_omegas"])
    except (OSError, ValueError, KeyError, IndexError) as e:
        errs.append(f"{p}: {type(e).__name__}: {e}")
    return info, errs


# ------------------------------------------------------------- reporting --
def e2(x):
    """Format a number as %.2e (n/a for None)."""
    return "n/a" if x is None else f"{x:.2e}"


def qs(q):
    """Format a q value (- for the q-independent TCM)."""
    return "-" if q is None else str(q)


def compress_q(qlist):
    """Compact text of a q list (complement if that is shorter)."""
    qlist = sorted(q for q in qlist if q is not None)
    if len(qlist) == len(cf.Q_ALL):
        return "all q"
    if not qlist:
        return "-"
    if len(qlist) <= 12:
        return "q = " + ",".join(str(q) for q in qlist)
    comp = [q for q in cf.Q_ALL if q not in qlist]
    if len(comp) < len(qlist) and len(comp) <= 12:
        return "all q except " + ",".join(str(q) for q in comp)
    return f"{len(qlist)} q ({qlist[0]}..{qlist[-1]})"


def missing_table(rfiles, tfiles):
    """Markdown lines listing files present on one side only.

    Columns with the same q pattern are grouped on one line.
    """
    lines = []
    for side, a, b in (("only in REF", rfiles, tfiles),
                       ("only in TEST", tfiles, rfiles)):
        by = {}
        for key in a:
            if key not in b:
                by.setdefault(key[0], []).append(key[1])
        if by:
            lines.append(f"* {side}: {sum(len(v) for v in by.values())} "
                         "files")
            groups = {}
            for col in sorted(by, key=cf.op_sort_key):
                groups.setdefault(compress_q(by[col]), []).append(col)
            for txt, cols in groups.items():
                lines.append(f"  * {len(cols)} columns, {txt}: "
                             f"{', '.join(cols)}")
    return lines


def fmt_key(sec, key):
    """Short text of a section and element key."""
    if not key:
        return f"{sec}:-"
    return f"{sec}:{','.join(str(x) for x in key)}"


def top_rows(results, field, topk):
    """Table rows of the worst elements over all results."""
    cands = []
    for r in results:
        if "secs" not in r:
            continue
        for sec, s in r["secs"].items():
            for c in s[field]:
                cands.append((c[0], r, sec, c))
    best = heapq.nlargest(topk, cands, key=lambda c: c[0])
    out = []
    for v, r, sec, c in best:
        out.append(f"| {os.path.basename(r['test'])} | "
                   f"{fmt_key(sec, c[1])} | {c[2]:.6e} | {c[3]:.6e} | "
                   f"{e2(v)} |")
    return out


def flow_section(rinfo, tinfo):
    """Markdown lines of the flow summary."""
    lines = ["### Flow (results.json, flow.txt)", "",
             "| quantity | ref | test | diff |", "|---|---|---|---|"]

    def row(name, a, b, fmt="{:.10f}"):
        if a is None or b is None:
            na = "n/a" if a is None else a
            nb = "n/a" if b is None else b
            lines.append(f"| {name} | {na} | {nb} | |")
            return
        diff = (e2(abs(b - a)) if isinstance(a, float)
                else ("0" if a == b else f"{b - a}"))
        lines.append(f"| {name} | {fmt.format(a)} | {fmt.format(b)} | "
                     f"{diff} |")
    row("final E0 (results.json)", rinfo.get("e0"), tinfo.get("e0"))
    row("num_omegas (results.json)", rinfo.get("num_omegas_json"),
        tinfo.get("num_omegas_json"), "{}")
    row("num_omegas (flow.txt, last row)", rinfo.get("num_omegas_txt"),
        tinfo.get("num_omegas_txt"), "{}")
    rs = {s[0]: s for s in rinfo.get("stages_txt", [])}
    ts = {s[0]: s for s in tinfo.get("stages_txt", [])}
    for st in sorted(set(rs) | set(ts)):
        a, b = rs.get(st), ts.get(st)
        row(f"stage {st} final E0", a and a[3], b and b[3])
        row(f"stage {st} final s", a and a[2], b and b[2], "{:.6f}")
        row(f"stage {st} final step", a and a[1], b and b[1], "{}")
    return lines


def table_cols(results, rel_flag):
    """Lines of the per-column summary table.

    "elements" is the number of compared elements (zero-, one- and two-body,
    union of the keys of both files, summed over the column's files),
    "rel-eligible" those with |ref| >= --small (n small is the rest).
    """
    cols = {}
    for r in results:
        if "secs" in r:
            cols.setdefault(r["col"], []).append(r)
    lines = ["| column | files | elements | rel-eligible | "
             "max abs (q) | max rel (q) | "
             f"n rel>{rel_flag:g} | max rel, \\|ref\\|>={BIG:g} | "
             f"n rel>{rel_flag:g} of n \\|ref\\|>={BIG:g} | n small | "
             "only ref/test | max abs ref |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for col in sorted(cols, key=lambda c: (cf.op_sort_key(c), c)):
        tots = [(r, merge_stats(r["secs"])) for r in cols[col]]
        ra, sa = max(tots, key=lambda x: x[1]["max_abs"])
        rr, sr = max(tots, key=lambda x: x[1]["max_rel"])
        nel = sum(t["n"] for _, t in tots)
        nsm = sum(t["n_small"] for _, t in tots)
        lines.append(
            f"| {col} | {len(tots)} | {nel} | {nel - nsm} | "
            f"{e2(sa['max_abs'])} ({qs(ra['q'])}) | "
            f"{e2(sr['max_rel'])} ({qs(rr['q'])}) | "
            f"{sum(t['n_flag'] for _, t in tots)} | "
            f"{e2(max(t['max_rel_big'] for _, t in tots))} | "
            f"{sum(t['n_flag_big'] for _, t in tots)} of "
            f"{sum(t['n_big'] for _, t in tots)} | "
            f"{nsm} | "
            f"{sum(t['only_ref'] for _, t in tots)}/"
            f"{sum(t['only_test'] for _, t in tots)} | "
            f"{e2(max(t['max_ref'] for _, t in tots))} |")
    return lines


def table_q(results, rel_flag):
    """Lines of the per-q table (farm columns only).

    Besides the maxima: the number of elements of the q, of those rel-eligible
    (|ref| >= --small) and of those with rel > rel_flag, summed over columns.
    """
    byq = {}
    for r in results:
        if "secs" in r and r["q"] is not None:
            byq.setdefault(r["q"], []).append((r, merge_stats(r["secs"])))
    lines = ["| q | files | elements | rel-eligible | "
             f"n rel>{rel_flag:g} | max abs | worst col (abs) | max rel | "
             "worst col (rel) |", "|---|---|---|---|---|---|---|---|---|"]
    for q in sorted(byq):
        lst = byq[q]
        ra, sa = max(lst, key=lambda x: x[1]["max_abs"])
        rr, sr = max(lst, key=lambda x: x[1]["max_rel"])
        nel = sum(t["n"] for _, t in lst)
        nsm = sum(t["n_small"] for _, t in lst)
        lines.append(f"| {q} | {len(lst)} | {nel} | {nel - nsm} | "
                     f"{sum(t['n_flag'] for _, t in lst)} | "
                     f"{e2(sa['max_abs'])} | "
                     f"{ra['col']} | {e2(sr['max_rel'])} | {rr['col']} |")
    return lines


def table_sections(results):
    """Lines of the 0b/1b/2b breakdown for a set of results."""
    lines = ["| part | elements | max abs | max rel | n rel>flag | n small | "
             "only ref/test |", "|---|---|---|---|---|---|---|"]
    for sec in SECTIONS:
        ss = [r["secs"][sec] for r in results if "secs" in r]
        if not ss:
            continue
        lines.append(
            f"| {sec} | {sum(s['n'] for s in ss)} | "
            f"{e2(max(s['max_abs'] for s in ss))} | "
            f"{e2(max(s['max_rel'] for s in ss))} | "
            f"{sum(s['n_flag'] for s in ss)} | "
            f"{sum(s['n_small'] for s in ss)} | "
            f"{sum(s['only_ref'] for s in ss)}/"
            f"{sum(s['only_test'] for s in ss)} |")
    return lines


def build_report(args, parts):
    """Assemble the markdown text from the collected parts."""
    out = [f"## {args.label}", "",
           "Absolute |test - ref| for all elements; relative only for "
           f"|ref| >= {args.small:g}; flag threshold rel > "
           f"{args.rel_flag:g}.", ""]
    for name in ("ref_flow", "ref_farm", "test_flow", "test_farm"):
        v = getattr(args, name)
        if v:
            out.append(f"* {name.replace('_', ' ')}: `{v}`")
    out.append("")
    if parts["errors"]:
        out += ["### Missing or unparseable inputs", ""]
        out += [f"* {e}" for e in parts["errors"]] + [""]
    if parts["flow_lines"]:
        out += parts["flow_lines"] + [""]
    allr = parts["results"]
    fl = [r for r in allr if r["col"] in ("H", "TCM(flow)")]
    if fl:
        out += ["### Flow directory files (valence H, TCM)", "",
                *table_cols(fl, args.rel_flag), "", "H by part:", "",
                *table_sections([r for r in fl if r["col"] == "H"]), ""]
    fr = [r for r in allr if r["col"] not in ("H", "TCM(flow)")]
    if fr or parts["notes"]:
        nf = sum(1 for r in fr if "secs" in r)
        out += [f"### Farm: {nf} file pairs compared", ""]
        out += [f"* {note}" for note in parts["notes"]]
        out += ["", "Missing files:", "",
                *(parts["missing"] or ["* none"]), ""]
    if fr:
        out += ["#### Per operator column", "",
                *table_cols(fr, args.rel_flag),
                "", "#### Per q", "", *table_q(fr, args.rel_flag), "",
                "#### By part", "", *table_sections(fr), ""]
    if allr:
        hdr = ["| file | key | ref | test | value |",
               "|---|---|---|---|---|"]
        out += [f"### {args.top} worst elements by relative diff", "",
                *hdr, *top_rows(allr, "top_rel", args.top), "",
                f"### {args.top} worst elements by absolute diff", "",
                *hdr, *top_rows(allr, "top_abs", args.top), ""]
    text = "\n".join(out)
    if len(text) > 99000:
        text = text[:99000] + "\n\n(truncated)\n"
    return text


# ----------------------------------------------------------------- driver --
def parse_args(argv):
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for n in ("ref-flow", "ref-farm", "test-flow", "test-farm"):
        ap.add_argument("--" + n)
    ap.add_argument("--label", default="variant comparison")
    ap.add_argument("--md", help="write markdown here (default stdout)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--small", type=float, default=1.0e-6)
    ap.add_argument("--rel-flag", type=float, default=1.0e-3)
    ap.add_argument("--jobs", type=int,
                    default=min(16, os.cpu_count() or 1))
    return ap.parse_args(argv)


def collect(args):
    """Scan inputs, run the comparisons, return the report parts."""
    errors, tasks, notes, missing = [], [], [], []
    rinfo = tinfo = None
    topk, small, flag = args.top, args.small, args.rel_flag
    for side in ("ref", "test"):
        for kind in ("flow", "farm"):
            d = getattr(args, f"{side}_{kind}")
            if d and not os.path.isdir(d):
                errors.append(f"{side} {kind} directory not found: {d}")
                setattr(args, f"{side}_{kind}", None)
    if args.ref_flow and args.test_flow:
        rf, tf = flow_files(args.ref_flow), flow_files(args.test_flow)
        for col in sorted(set(rf) | set(tf)):
            if col in rf and col in tf:
                tasks.append((col, None, rf[col], tf[col], small, flag,
                              topk))
            else:
                errors.append(f"flow file {col} only in "
                              f"{'ref' if col in rf else 'test'}")
        rinfo, e_r = flow_info(args.ref_flow)
        tinfo, e_t = flow_info(args.test_flow)
        errors += e_r + e_t
    elif args.ref_flow or args.test_flow:
        errors.append("flow comparison needs both --ref-flow and "
                      "--test-flow")
    if args.ref_farm and args.test_farm:
        rfiles, rn = scan_farm(args.ref_farm)
        tfiles, tn = scan_farm(args.test_farm)
        notes.append(f"files: ref {len(rfiles)}, test {len(tfiles)}; TCM "
                     f"copies: ref {rn}, test {tn} (first used)")
        missing = missing_table(rfiles, tfiles)
        for key in sorted(set(rfiles) & set(tfiles),
                          key=lambda k: (cf.op_sort_key(k[0]), k[1] or 0)):
            tasks.append((key[0], key[1], rfiles[key], tfiles[key], small,
                          flag, topk))
    elif args.ref_farm or args.test_farm:
        errors.append("farm comparison needs both --ref-farm and "
                      "--test-farm")
    if not tasks and not errors:
        errors.append("nothing to compare (no directories given)")
    jobs = max(1, args.jobs)
    if jobs > 1 and len(tasks) > 1:
        with multiprocessing.Pool(jobs) as pool:
            results = pool.map(compare_pair, tasks, chunksize=4)
    else:
        results = [compare_pair(t) for t in tasks]
    errors += [f"unparseable: {r['test']}: {r['error']}"
               for r in results if "error" in r][:20]
    flow_lines = flow_section(rinfo, tinfo) if rinfo and tinfo else []
    return {"results": results, "errors": errors, "notes": notes,
            "missing": missing, "flow_lines": flow_lines}


def run(argv=None):
    """Entry point; returns the exit status."""
    args = parse_args(argv)
    t0 = time.time()
    parts = collect(args)
    text = build_report(args, parts)
    text += f"\n(wall time {time.time() - t0:.1f} s, jobs {args.jobs})\n"
    if args.md:
        with open(args.md, "w") as f:
            f.write(text)
    else:
        print(text)
    return 2 if parts["errors"] else 0


if __name__ == "__main__":
    sys.exit(run())
