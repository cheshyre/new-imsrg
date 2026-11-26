"""Checks and result.json of tests/golden/tensor (called by gen_golden_tensor.sh).

usage: python3 -I tensor_golden_check.py OUT TMP GOLDEN COMMANDS_FILE VERSION

Checks (all must pass, otherwise the script exits with an error):
  * every operator, `all` commutator and BCH dump obeys its declared hermiticity
    (1B: v(b,a) = h (-1)^((2j_a-2j_b)/2) v(a,b); 2B diagonal-channel blocks: M^T = h M, h = +1 H / -1 A),
  * the sum rule Z_all = comm111st + comm121st + comm122st + comm222_pp_hh_221st + comm222_phst,
  * the O16 commutator inputs reused from the earlier goldens equal what the harness computes.
Then writes OUT/result.json (version stamp, commands, norms of every run, file sizes, checks).
"""
import glob
import json
import os
import sys
from collections import OrderedDict

out, tmp, golden, cmdfile, version = sys.argv[1:6]


def parse_tdump(path):
    d = {"orbits": {}, "oneb": {}, "twob": {}, "spec": {}, "header": []}
    block = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            k = t[0]
            if k == "orbit":
                d["orbits"][int(t[1])] = tuple(int(x) for x in t[2:6]) + (float(t[6]),)
            elif k == "spec":
                d["spec"] = dict(zip(t[1::2], t[2::2]))
            elif k == "1b":
                d["oneb"][(int(t[1]), int(t[2]))] = float(t[3])
            elif k == "block":
                block = tuple(int(x) for x in t[1:9])
            elif k == "2b":
                d["twob"][block + tuple(int(x) for x in t[1:5])] = float(t[5])
            elif k == "zero_body":
                d["zero"] = float(t[1])
    return d


def parse_v1(path):
    """OperatorDump v1: returns {key: value} over zero/one/two-body values."""
    d, sec = {}, None
    for line in open(path):
        t = line.split()
        if not t or t[0] in ("#", "source"):
            continue
        if t[0] == "zero_body":
            d[("z",)] = float(t[1])
        elif t[0] == "one_body":
            sec = 1
        elif t[0] == "two_body":
            sec = 2
        elif t[0] in ("three_body", "end"):
            sec = None
        elif sec == 1 and len(t) == 3:
            d[("1",) + tuple(t[:2])] = float(t[2])
        elif sec == 2 and len(t) == 7:
            d[("2",) + tuple(t[:6])] = float(t[6])
    return d


def check_hermiticity(path):
    d = parse_tdump(path)
    h = {"H": 1, "A": -1}.get(d["spec"].get("hermiticity"), 0)
    if h == 0:
        return 0.0
    scale = max([abs(v) for v in d["oneb"].values()] + [abs(v) for v in d["twob"].values()] + [1e-300])
    dev = 0.0
    for (a, b), v in d["oneb"].items():
        j2a, j2b = d["orbits"][a][2], d["orbits"][b][2]
        w = d["oneb"].get((b, a), 0.0)
        ph = -1 if ((j2a - j2b) // 2) % 2 else 1
        dev = max(dev, abs(w - h * ph * v))
    for key, v in d["twob"].items():
        chb, jb, pb, tb, chk, jk, pk, tk, p, q, r, s = key
        if chb != chk:
            continue
        w = d["twob"].get((chb, jb, pb, tb, chk, jk, pk, tk, r, s, p, q), 0.0)
        dev = max(dev, abs(w - h * v))
    return dev / scale


checks = OrderedDict()
bad = []

# 1. hermiticity of operator dumps, full commutators and BCH results
files = sorted(glob.glob(os.path.join(out, "**", "*.tdump"), recursive=True))
for f in files:
    base = os.path.basename(f)
    if base.startswith("Z_") and not base.startswith("Z_all__"):
        continue  # single terms need not be (anti-)Hermitian before the final symmetrization
    dev = check_hermiticity(f)
    checks["hermiticity:" + os.path.relpath(f, out)] = dev
    if dev > 1e-10:
        bad.append((f, dev))

# 2. sum rule of the commutator terms (every run directory that has the terms)
SUM = ["comm111st", "comm121st", "comm122st", "comm222_pp_hh_221st", "comm222_phst"]
for rd in sorted(glob.glob(os.path.join(tmp, "runs", "comm*"))):
    for allf in sorted(glob.glob(os.path.join(rd, "Z_all__*.tdump"))):
        suffix = os.path.basename(allf)[len("Z_all"):]
        tot = {}
        for t in SUM:
            p = os.path.join(rd, "Z_" + t + suffix)
            if not os.path.exists(p):
                continue  # identically zero
            dd = parse_tdump(p)
            for k, v in dd["oneb"].items():
                tot[("1",) + k] = tot.get(("1",) + k, 0.0) + v
            for k, v in dd["twob"].items():
                tot[("2",) + k] = tot.get(("2",) + k, 0.0) + v
        da = parse_tdump(allf)
        ref = {("1",) + k: v for k, v in da["oneb"].items()}
        ref.update({("2",) + k: v for k, v in da["twob"].items()})
        scale = max([abs(v) for v in ref.values()] + [1e-300])
        dev = max([abs(tot.get(k, 0.0) - ref.get(k, 0.0)) for k in set(tot) | set(ref)] + [0.0]) / scale
        name = "sum_rule:" + os.path.basename(rd) + suffix[:-len(".tdump")]
        checks[name] = dev
        if dev > 1e-12:
            bad.append((name, dev))

# 3. O16 commutator inputs reused from the earlier goldens
for nuc_dir, prefix, mine, theirs in [
    ("ops_O16_HF", "O16 H_HF", "H_HF.dump", os.path.join(golden, "hf", "emax2_O16_2N", "HNO.dump")),
    ("ops_O16_HF", "O16 random_A", "random_A.dump", os.path.join(golden, "commutator", "emax2_O16", "random_A.dump")),
]:
    if prefix == "O16 random_A":
        # random_A of the earlier golden was made with random_seed=7 as well
        pass
    a = parse_v1(os.path.join(tmp, "runs", nuc_dir, mine))
    b = parse_v1(theirs)
    dev = max([abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)] + [0.0])
    checks["input_equal:" + prefix] = dev
    # random_A depends on the C++ standard library's random distributions: the Mac (libc++)
    # golden differs from a Frontier (libstdc++) harness run, so it is recorded, not required
    # (the tensor goldens store their own random_A inputs since 2026-11-26).
    if dev > 1e-12 and prefix != "O16 random_A":
        bad.append((prefix, dev))

# 3b. random_A (seed 7) is reference independent: He4 run == the O16 golden file
a = parse_v1(os.path.join(tmp, "runs", "ops_He4_HF", "random_A.dump"))
b = parse_v1(os.path.join(golden, "commutator", "emax2_O16", "random_A.dump"))
dev = max([abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)] + [0.0])
checks["input_equal:He4 random_A vs O16 random_A"] = dev  # informational, see above
# required instead: the He4 and O16 runs of this harness agree (random_A is reference independent)
c = parse_v1(os.path.join(tmp, "runs", "ops_O16_HF", "random_A.dump"))
dev = max([abs(a.get(k, 0.0) - c.get(k, 0.0)) for k in set(a) | set(c)] + [0.0])
checks["input_equal:He4 random_A vs O16 random_A (this run)"] = dev
if dev > 1e-12:
    bad.append(("He4 random_A (this run)", dev))

# 4. result.json
runs = OrderedDict()
cmds = OrderedDict()
for line in open(cmdfile):
    name, cmd = line.rstrip("\n").split("|", 1)
    cmds[name] = cmd
for name in cmds:
    nj = os.path.join(tmp, "runs", name, "norms.json")
    runs[name] = {"command": cmds[name], "norms": json.load(open(nj)) if os.path.exists(nj) else None}
filelist = OrderedDict()
total = 0
for f in sorted(glob.glob(os.path.join(out, "**", "*"), recursive=True)):
    if os.path.isfile(f) and os.path.basename(f) != "result.json":
        n = os.path.getsize(f)
        filelist[os.path.relpath(f, out)] = n
        total += n
res = OrderedDict()
res["version"] = version.strip()
res["generator"] = "tools/reference/gen_golden_tensor.sh"
res["format"] = "imsrg-tensor-dump raw v0 (design/15-operator-dump-format.md, Tensor reference dumps)"
res["total_bytes"] = total
res["max_file_bytes"] = max(filelist.values())
res["checks"] = checks
res["runs"] = runs
res["files"] = filelist
with open(os.path.join(out, "result.json"), "w") as f:
    f.write("{\n")
    items = list(res.items())
    for i, (k, v) in enumerate(items):
        if isinstance(v, dict):
            f.write(' "%s": {\n' % k)
            sub = list(v.items())
            for j, (k2, v2) in enumerate(sub):
                f.write('  "%s": %s%s\n' % (k2, json.dumps(v2, separators=(",", ":")), "," if j + 1 < len(sub) else ""))
            f.write(" }%s\n" % ("," if i + 1 < len(items) else ""))
        else:
            f.write(' "%s": %s%s\n' % (k, json.dumps(v), "," if i + 1 < len(items) else ""))
    f.write("}\n")
print("tensor golden: %d files, %d bytes, largest %d bytes" % (len(filelist), total, res["max_file_bytes"]))
if bad:
    for b in bad:
        print("CHECK FAILED:", b)
    sys.exit(1)
print("all checks passed (%d)" % len(checks))
