#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Compare a farm output directory of tokyo-format .snt files with a reference.

Two modes:
  compare_farm.py --archive RAWDIR --ham N2LOsat --test DIR [--ops A,B] [--tol 1e-4]
      RAWDIR holds one directory per job (2024 archive layout); the expected
      files are all q = 0, 15, ..., 1050 of every operator found for HAM.
      Archive files that are absent, empty or all zero are reported as
      "placeholder" (e.g. DN2LOGO FM2p q = 975..1050, job J00011500 died).
  compare_farm.py --ref-dir DIR1 --test DIR2 [--tol T]
      Two flat output directories of ours (no PASS/FAIL unless --tol given).

Flat files are matched by the suffix `F<Family><L><p|n>_<q>[_2b].snt` and
`TCM.snt` (fork naming `<stem>FM0p_15.snt`, devel naming `<stem>_FM0p_15.snt`).

Parsing follows the C++ reader ReadFork (tests/unit/driver/
test_al27_fork_golden.cc): data start after `! interaction`, `! reduced matrix
elements` or `! operator`; `<count> 0 <hw>` lines are section headers; rows are
keyed by their integer columns, the value is the last column; section 1 is
1b, section 2 is 2b. Orbit indices in keys are replaced by the model-space
labels (n, l, 2j, 2tz), so differently numbered files compare; keys are
otherwise kept as written. Rows present in only one file compare against 0.
--canonical brings scalar 2b rows to the canonical form of compare_snt.py
(phases for a/b and c/d swaps and bra-ket symmetry).
--isospin-average-test averages the pn 2b elements of the test files with
their isospin mirror (scalar files only; implies --canonical).

Exit status: 1 if a compared column exceeds the tolerance, else 2 if test
files are missing (unless --allow-missing), else 0.
"""

import argparse
import importlib.util
import multiprocessing
import os
import re
import sys
import time

FAMILIES = ["M", "Phipp", "Phitp", "Sigmap", "Sigmapp", "Delta"]
Q_ALL = list(range(0, 1051, 15))
OP_RE = re.compile(r"F([A-Z][a-z]*)([0-9])([pn])_([0-9]+)(_2b)?\.snt$")
TCM_RE = re.compile(r"TCM\.snt$")
START_MARKS = ("! interaction", "! reduced matrix elements", "! operator")


def _load_compare_snt():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "compare_snt.py")
    spec = importlib.util.spec_from_file_location("compare_snt_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def op_sort_key(op):
    """Sort key of an operator name `F<family><L><p|n>` (TCM last)."""
    m = re.fullmatch(r"F([A-Z][a-z]*)([0-9])([pn])", op)
    if not m:
        return (99, 0, "", op)
    fam = m.group(1)
    fi = FAMILIES.index(fam) if fam in FAMILIES else len(FAMILIES)
    return (fi, int(m.group(2)), m.group(3), op)


# ---------------------------------------------------------------- parsing --
class Snt:
    """Parsed .snt file: zero body term, 1b and 2b rows keyed by label."""

    def __init__(self):
        self.zero = 0.0
        self.orbits = {}
        self.one = {}
        self.two = {}
        self.scalar = True  # 2b rows have 5 integer columns


def parse_snt(path):
    """Parse a tokyo-format file (see module docstring)."""
    s = Snt()
    raw1, raw2 = [], []
    in_model = False
    model_n = None
    data = False
    nheader = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                if line.startswith("! Zero body term:"):
                    s.zero = float(line[17:].split()[0])
                elif line.startswith("! model space"):
                    in_model = True
                elif line.startswith(START_MARKS):
                    data = True
                    in_model = False
                continue
            tok = line.split()
            if not tok:
                continue
            if in_model:
                if model_n is None:
                    model_n = int(tok[0]) + int(tok[1])
                elif len(s.orbits) < model_n:
                    k, n, l, j2, tz = (int(x) for x in tok[:5])
                    s.orbits[k] = (n, l, j2, tz)
                continue
            if not data:
                continue
            if len(tok) == 3 and tok[1] == "0":
                nheader += 1
                continue
            ints = tuple(int(x) for x in tok[:-1])
            v = float(tok[-1])
            (raw1 if nheader <= 1 else raw2).append((ints, v))
    lab = s.orbits.get

    def mk(ints, n):
        return tuple(lab(i, i) for i in ints[:n]) + ints[n:]

    for ints, v in raw1:
        s.one[mk(ints, 2)] = v
    for ints, v in raw2:
        if len(ints) != 5:
            s.scalar = False
        s.two[mk(ints, 4)] = v
    return s


def canonicalize(s, cs):
    """Replace 2b keys by compare_snt canonical keys (scalar files only)."""
    if not s.scalar:
        return s
    two = {}
    for k, v in s.two.items():
        key, ph = cs.canonical(k[0], k[1], k[2], k[3], k[4])
        if key is not None:
            two[key] = ph * v
    s.two = two
    return s


def read_snt(path, canonical, iso_avg):
    """Parse `path`; optionally canonicalize and isospin-average (scalar)."""
    s = parse_snt(path)
    if canonical or iso_avg:
        cs = _load_compare_snt()
        canonicalize(s, cs)
        if iso_avg and s.scalar:
            s.two = cs.isospin_average(s.two)
    return s


def is_placeholder(path):
    """True if the file is empty, unparseable or contains only zeros."""
    try:
        if os.path.getsize(path) == 0:
            return True
        s = parse_snt(path)
    except (ValueError, OSError):
        return True
    if not s.one and not s.two:
        return True
    return (s.zero == 0.0 and all(v == 0.0 for v in s.one.values())
            and all(v == 0.0 for v in s.two.values()))


# ------------------------------------------------------------- comparison --
def diff_sections(ref, test):
    """Compare two dicts; returns (max, only_ref, only_test, worst)."""
    mx, worst = 0.0, None
    only_ref = only_test = 0
    for k in set(ref) | set(test):
        r, t = ref.get(k), test.get(k)
        if r is None:
            only_test += 1
            d = abs(t)
        elif t is None:
            only_ref += 1
            d = abs(r)
        else:
            d = abs(r - t)
        if worst is None or d > mx:
            mx, worst = d, k
    if worst is not None:
        worst = (worst, ref.get(worst, 0.0), test.get(worst, 0.0))
    return mx, only_ref, only_test, worst


def compare_files(task):
    """Worker: compare one file pair; returns a result dict."""
    col, q, rpath, tpath, canonical, iso = task
    res = {"col": col, "q": q, "ref": rpath, "test": tpath}
    try:
        r = read_snt(rpath, canonical, False)
        t = read_snt(tpath, canonical, iso)
    except (ValueError, OSError, IndexError) as e:
        res["error"] = str(e)
        return res
    m0 = abs(r.zero - t.zero)
    m1, or1, ot1, w1 = diff_sections(r.one, t.one)
    m2, or2, ot2, w2 = diff_sections(r.two, t.two)
    res.update(max0=m0, max1=m1, max2=m2, only_ref=or1 + or2,
               only_test=ot1 + ot2)
    best = max(((m0, "0b", None), (m1, "1b", w1), (m2, "2b", w2)),
               key=lambda x: x[0])
    res["max"] = best[0]
    if best[2]:
        res["worst"] = (f"{best[1]} {best[2][0]} ref {best[2][1]:.6e} "
                        f"test {best[2][2]:.6e}")
    else:
        res["worst"] = f"0b zero-body ref {r.zero:.6e} test {t.zero:.6e}"
    return res


def copy_task(task):
    """Worker: max |diff| of a TCM copy against the first copy."""
    first, path = task
    r, t = parse_snt(first), parse_snt(path)
    return path, max(abs(r.zero - t.zero),
                     diff_sections(r.one, t.one)[0],
                     diff_sections(r.two, t.two)[0])


# ------------------------------------------------------------- file scans --
def scan_flat(directory):
    """Return ({(op, q): path}, [tcm paths]) for a flat output directory."""
    ops, tcm = {}, []
    for name in sorted(os.listdir(directory)):
        m = OP_RE.search(name)
        if m:
            op = f"F{m.group(1)}{m.group(2)}{m.group(3)}"
            ops[(op, int(m.group(4)))] = os.path.join(directory, name)
        elif TCM_RE.search(name):
            tcm.append(os.path.join(directory, name))
    return ops, tcm


def scan_archive(raw, ham):
    """Return ({(op, q): path}, [tcm paths]) for one Hamiltonian."""
    ops, tcm = {}, []
    pat = re.compile(rf"_Al27_{re.escape(ham)}_")
    for d in sorted(os.listdir(raw)):
        if not d.startswith("output_") or not pat.search(d):
            continue
        sub = os.path.join(raw, d)
        if not os.path.isdir(sub):
            continue
        for name in sorted(os.listdir(sub)):
            m = OP_RE.search(name)
            if m and f"_INTFILEF{m.group(1)}" in name:
                op = f"F{m.group(1)}{m.group(2)}{m.group(3)}"
                ops[(op, int(m.group(4)))] = os.path.join(sub, name)
            elif name.endswith("INTFILETCM.snt"):
                tcm.append(os.path.join(sub, name))
    return ops, tcm


# ------------------------------------------------------------------ driver --
def build_tasks(args, ref_ops, ref_tcm, test_ops, test_tcm, archive_mode,
                notes):
    """Create the file-pair tasks and the missing / placeholder bookkeeping."""
    opset = set(op for op, _ in ref_ops)
    want_tcm = True
    if args.ops:
        wanted = [o.strip() for o in args.ops.split(",") if o.strip()]
        want_tcm = "TCM" in wanted
        opset = set(o for o in wanted if o != "TCM")
    cols = sorted(opset, key=op_sort_key)
    tasks, missing, placeholder, extra = [], {}, {}, {}
    for op in cols:
        in_ref = any(o == op for o, _ in ref_ops)
        qs = set(q for o, q in ref_ops if o == op)
        if archive_mode and in_ref:
            qs = set(Q_ALL) | qs
        for q in sorted(qs):
            rp = ref_ops.get((op, q))
            tp = test_ops.get((op, q))
            if rp is None or (archive_mode and is_placeholder(rp)):
                placeholder.setdefault(op, []).append(q)
                continue
            if tp is None:
                missing.setdefault(op, []).append(q)
                continue
            tasks.append((op, q, rp, tp, args.canonical, args.isospin))
        extra[op] = sorted(q for o, q in test_ops
                           if o == op and q not in qs)
    if want_tcm:
        cols.append("TCM")
        if ref_tcm:
            if test_tcm:
                tasks.append(("TCM", None, ref_tcm[0], test_tcm[0],
                              args.canonical, args.isospin))
            else:
                missing["TCM"] = [None]
        elif test_tcm:
            notes.append("TCM: no reference TCM file")
    return cols, tasks, missing, placeholder, extra


def summarize(cols, results, missing, placeholder, extra, tol):
    """Aggregate per-file results into per-column dicts."""
    out = []
    for col in cols:
        rs = [r for r in results if r["col"] == col and "error" not in r]
        errs = [r for r in results if r["col"] == col and "error" in r]
        c = {"col": col, "ncmp": len(rs),
             "missing": len(missing.get(col, [])),
             "placeholder": len(placeholder.get(col, [])),
             "extra": len(extra.get(col, [])), "errors": len(errs),
             "only_ref": sum(r["only_ref"] for r in rs),
             "only_test": sum(r["only_test"] for r in rs),
             "max": 0.0, "q": None, "max1": 0.0, "max2": 0.0, "max0": 0.0,
             "worst": ""}
        for r in sorted(rs, key=lambda r: -1 if r["q"] is None else r["q"]):
            if c["q"] is None and not c["worst"] or r["max"] > c["max"]:
                c["max"], c["q"], c["worst"] = r["max"], r["q"], r["worst"]
            c["max1"] = max(c["max1"], r["max1"])
            c["max2"] = max(c["max2"], r["max2"])
            c["max0"] = max(c["max0"], r["max0"])
        if errs:
            c["status"] = "ERROR"
        elif tol is None:
            c["status"] = "-" if rs else "n/a"
        elif not rs:
            c["status"] = "n/a"
        else:
            c["status"] = "PASS" if c["max"] <= tol else "FAIL"
        out.append(c)
    return out


def fmt_q(c):
    return "-" if c["q"] is None else str(c["q"])


def markdown_section(title, cols, mode_desc, tol, files_missing, tcm_note,
                     placeholders):
    """Compact markdown table (see module docstring)."""
    lines = [f"## {title}", "", mode_desc, ""]
    lines.append("| column | q cmp | missing | placeholder | max abs diff | at q"
                 " | max 1b | max 2b | only ref | only test | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c in cols:
        lines.append(
            f"| {c['col']} | {c['ncmp']} | {c['missing']} | "
            f"{c['placeholder']} | {c['max']:.2e} | {fmt_q(c)} | "
            f"{c['max1']:.2e} | {c['max2']:.2e} | {c['only_ref']} | "
            f"{c['only_test']} | {c['status']} |")
    comp = [c for c in cols if c["ncmp"]]
    allmax = max((c["max"] for c in comp), default=0.0)
    if tol is not None:
        npass = sum(1 for c in cols if c["status"] == "PASS")
        summ = (f"{npass}/{len(cols)} columns pass (tol {tol:.0e}); "
                f"overall max {allmax:.2e}; files missing {files_missing}")
    else:
        summ = (f"{len(comp)}/{len(cols)} columns compared; overall max "
                f"{allmax:.2e}; files missing {files_missing}")
    lines += ["", f"**Summary:** {summ}"]
    if tcm_note:
        lines += ["", tcm_note]
    if placeholders:
        txt = "; ".join(f"{k}: q={','.join(str(q) for q in v)}"
                        for k, v in placeholders.items())
        lines += ["", f"Placeholders (absent/empty/zero in reference): {txt}"]
    return "\n".join(lines) + "\n"


def write_md(path, title, body):
    """Write/replace the marked section `title` of the markdown file."""
    begin = f"<!-- compare_farm:{title} begin -->"
    end = f"<!-- compare_farm:{title} end -->"
    block = f"{begin}\n{body}{end}\n"
    text = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    i, j = text.find(begin), text.find(end)
    if i >= 0 and j > i:
        text = text[:i] + block + text[j + len(end):].lstrip("\n")
        if not text.endswith("\n"):
            text += "\n"
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += ("\n" if text else "") + block
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run(argv=None):
    """Command-line entry; returns the exit status."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--archive", help="archive raw/ directory")
    p.add_argument("--ham", help="Hamiltonian label (with --archive)")
    p.add_argument("--ref-dir", help="reference flat output directory")
    p.add_argument("--test", required=True, help="our flat output directory")
    p.add_argument("--ops", help="comma list of operators (TCM allowed)")
    p.add_argument("--tol", type=float, default=None,
                   help="tolerance (default 1e-4 with --archive)")
    p.add_argument("--md", help="markdown file (section replaced)")
    p.add_argument("--title", default="compare_farm")
    p.add_argument("--tsv", help="per-file TSV output")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--canonical", action="store_true")
    p.add_argument("--isospin-average-test", dest="isospin",
                   action="store_true")
    p.add_argument("--allow-missing", action="store_true")
    a = p.parse_args(argv)
    if bool(a.archive) == bool(a.ref_dir):
        p.error("give exactly one of --archive and --ref-dir")
    if a.archive and not a.ham:
        p.error("--archive needs --ham")
    archive_mode = bool(a.archive)
    tol = a.tol if a.tol is not None else (1.0e-4 if archive_mode else None)
    t0 = time.time()
    notes = []
    if archive_mode:
        ref_ops, ref_tcm = scan_archive(a.archive, a.ham)
        mode_desc = (f"Archive `{a.archive}` ({a.ham}) vs `{a.test}`. "
                     "Placeholder = archive file absent/empty/zero.")
    else:
        ref_ops, ref_tcm = scan_flat(a.ref_dir)
        mode_desc = f"`{a.ref_dir}` vs `{a.test}`."
    test_ops, test_tcm = scan_flat(a.test)
    cols_l, tasks, missing, placeholder, extra = build_tasks(
        a, ref_ops, ref_tcm, test_ops, test_tcm, archive_mode, notes)

    tcm_note = ""
    with multiprocessing.Pool(max(1, a.workers)) as pool:
        if len(ref_tcm) > 1 and any(t[0] == "TCM" for t in tasks):
            devs = dict(pool.imap_unordered(
                copy_task, [(ref_tcm[0], x) for x in ref_tcm[1:]], 8))
            worst = max(devs.values())
            ndiff = sum(1 for v in devs.values() if v > 0.0)
            tcm_note = (f"Reference TCM: {len(ref_tcm)} copies; max |diff| "
                        f"of any copy vs the first {worst:.2e} "
                        f"({ndiff} copies not bit-identical in value); "
                        "compared against the first copy.")
            if worst > (tol if tol is not None else 1.0e-4):
                tcm_note = "**" + tcm_note + "**"
            notes.append(tcm_note)
        results = pool.map(compare_files, tasks, chunksize=16)
    cols = summarize(cols_l, results, missing, placeholder, extra, tol)

    print(f"{'column':<12} {'qcmp':>4} {'miss':>4} {'plch':>4} "
          f"{'max|diff|':>10} {'at q':>5} {'max1b':>9} {'max2b':>9} "
          f"{'onlyRef':>7} {'onlyTest':>8} status")
    for c in cols:
        print(f"{c['col']:<12} {c['ncmp']:>4} {c['missing']:>4} "
              f"{c['placeholder']:>4} {c['max']:>10.2e} {fmt_q(c):>5} "
              f"{c['max1']:>9.2e} {c['max2']:>9.2e} {c['only_ref']:>7} "
              f"{c['only_test']:>8} {c['status']}")
        if c["worst"] and (tol is None or c["max"] > tol):
            print(f"    worst (q={fmt_q(c)}): {c['worst']}")
        if missing.get(c["col"]):
            print(f"    missing q: {missing[c['col']][:20]}")
        if placeholder.get(c["col"]):
            print(f"    placeholder q: {placeholder[c['col']]}")
        if extra.get(c["col"]):
            print(f"    in test only (no reference): q {extra[c['col']]}")
        if c["errors"]:
            print(f"    parse errors: {c['errors']}")
    for n in notes:
        print("note:", n)
    nmiss = sum(c["missing"] for c in cols)
    comp = [c for c in cols if c["ncmp"]]
    allmax = max((c["max"] for c in comp), default=0.0)
    print(f"{len(results)} files compared in {time.time() - t0:.1f} s; "
          f"overall max {allmax:.3e}; missing {nmiss}")

    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8") as f:
            f.write("column\tq\tmax\tmax0\tmax1\tmax2\tonly_ref\tonly_test"
                    "\tworst\tref\ttest\n")
            for r in sorted(results, key=lambda r: (
                    op_sort_key(r["col"]), -1 if r["q"] is None else r["q"])):
                q = "" if r["q"] is None else r["q"]
                if "error" in r:
                    f.write(f"{r['col']}\t{q}\tERROR\t\t\t\t\t\t"
                            f"{r['error']}\t{r['ref']}\t{r['test']}\n")
                    continue
                f.write(f"{r['col']}\t{q}\t"
                        f"{r['max']:.6e}\t{r['max0']:.6e}\t{r['max1']:.6e}\t"
                        f"{r['max2']:.6e}\t{r['only_ref']}\t{r['only_test']}"
                        f"\t{r['worst']}\t{r['ref']}\t{r['test']}\n")
    if a.md:
        body = markdown_section(a.title, cols, mode_desc, tol, nmiss,
                                tcm_note, placeholder)
        write_md(a.md, a.title, body)

    failed = tol is not None and any(c["status"] in ("FAIL", "ERROR")
                                     for c in cols)
    if failed:
        return 1
    if nmiss and not a.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(run())
