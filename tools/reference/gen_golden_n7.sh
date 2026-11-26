#!/usr/bin/env bash
# IMSRG(3)-N7 reference data for milestone M10 step 3 (design/14-imsrg3-plan.md section 4 item 2):
#   tests/golden/n7/<case>/<TERM>.dump   Z = [X, Y] of one imsrg++ N7 term (refharness comm3), as an
#                                        OperatorDump v1 with a three_body section, at DUMP_E3MAX
#   tests/golden/n7/<case>/eta_white.dump  the imsrg++ white generator of a rank-3 H (refharness
#                                        generator3), at DUMP_E3MAX (and ETA_E3MAXS_EXTRA if small)
#   tests/golden/n7/<case>/values.json   norms of every output and of the inputs, per E3max
#
# Usage: tools/reference/gen_golden_n7.sh [all]
#
# Operators (tools/bench/n7_inputs.cc, `imsrg_n7_inputs`, RandomThreeBodyOperator of
# tests/support/three_body_test_support.h): X = anti-Hermitian seed 11, Y = Hermitian seed 12, H =
# Hermitian seed 13, one- and two-body scale 1, physical three-body part. The dumps of X, Y and H
# are NOT committed (the tests regenerate them in process from the seeds; values.json holds their
# imsrg++ norms as a drift check). TERM = comm330ss comm331ss comm231ss comm132ss comm232ss
# comm133ss comm223ss and n7 (full Commutator::Commutator with IMSRG(3) and the N7 switches).
# Cases: He4_emax2 and O16_emax2, hw 16, valence = reference, E3max as in E3MAXS below (dumps only
# at DUMP_E3MAX = 3; the larger E3max give norms only, they are where the hhh-ppp terms live).
#
# Env: IMSRG_REFERENCE_ROOT (default: see ref_env.sh; build outputs go to IMSRG_REFERENCE_BUILD),
#      HARNESS_BUILD_DIR_D35 (default $ROOT/harness-build-d35; must contain a refharness built from
#        this checkout: `HARNESS_BUILD_DIR_D35=<dir> tools/reference/build_imsrg_devel.sh patched-harness`),
#      BENCH_BUILD_DIR (default build/golden-n7 of this checkout: release-openmp preset with
#        -DIMSRG_BUILD_BENCH=ON, configured and built on demand), OMP_NUM_THREADS (default 8),
#      E3MAXS (default "3 6"), DUMP_E3MAX (default 3), ETA_DUMP_E3MAXS (E3max whose eta dump is
#        also kept as eta_white_e3max<M>.dump; default "6"; skipped above 1 MB).
# The harness is the D35 patched build; IMSRG_REFERENCE_SHA_FILE selects its version string.
# The terms are parallel sums: results differ in the last digits with the thread count and from run
# to run (e.g. n7, He4 E3max 3, 2B entry 1 1: 3.4421068778002937 vs 3.4421068778002946); compare to
# 1e-12 relative. Dump files must stay below 1 MB (golden rule 7): the script fails otherwise.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ref_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${REF_BUILD}"
RH="${HARNESS_BUILD_DIR_D35:-${ROOT}/harness-build-d35}/refharness"
BENCH_DIR="${BENCH_BUILD_DIR:-${REPO}/build/golden-n7}"
INPUTS="${BENCH_DIR}/tools/bench/imsrg_n7_inputs"
export IMSRG_REFERENCE_SHA_FILE="${IMSRG_REFERENCE_SHA_FILE:-${REF_SHA_D35}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
OUT="${REPO}/tests/golden/n7"
DUMP_E3MAX="${DUMP_E3MAX:-3}"
E3MAXS="${E3MAXS:-3 6}"
ETA_DUMP_E3MAXS="${ETA_DUMP_E3MAXS:-6}"
TERMS="comm330ss comm331ss comm231ss comm132ss comm232ss comm133ss comm223ss n7"
MAX_BYTES=1000000
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# The harness prints noise (allocation messages) before its JSON; keep from the first '{' line on.
json_of() { awk '/^\{/{p=1} p' "$1"; }

# one_case <case> <reference> <emax>
one_case() {
  local name="$1" ref="$2" emax="$3"
  local d="${OUT}/${name}" w="${tmp}/${name}"
  mkdir -p "${d}" "${w}"
  rm -f "${d}"/*.dump "${d}/values.json"
  local m t
  for m in ${E3MAXS}; do
    echo "== ${name}: inputs e3max=${m}"
    "${INPUTS}" --emax "${emax}" --reference "${ref}" --e3max "${m}" --seed 11 --herm A --out "${w}/X_${m}.dump" 2> /dev/null
    "${INPUTS}" --emax "${emax}" --reference "${ref}" --e3max "${m}" --seed 12 --herm H --out "${w}/Y_${m}.dump" 2> /dev/null
    "${INPUTS}" --emax "${emax}" --reference "${ref}" --e3max "${m}" --seed 13 --herm H --out "${w}/H_${m}.dump" 2> /dev/null
    local o
    for o in X Y H; do
      "${RH}" norms "${w}/${o}_${m}.dump" e3max="${m}" > "${w}/raw_${o}_${m}.txt" 2> /dev/null
      json_of "${w}/raw_${o}_${m}.txt" > "${w}/norm_${o}_${m}.txt"
    done
    for t in ${TERMS}; do
      echo "== ${name}: comm3 ${t} e3max=${m}"
      "${RH}" comm3 "${t}" "${w}/X_${m}.dump" "${w}/Y_${m}.dump" "${w}/Z_${t}_${m}.dump" e3max="${m}" \
          > "${w}/out_${t}_${m}.txt" 2> "${w}/err_${t}_${m}.txt"
      json_of "${w}/out_${t}_${m}.txt" > "${w}/norm_${t}_${m}.txt"
      if [[ "${m}" == "${DUMP_E3MAX}" ]]; then
        local sz
        sz="$(wc -c < "${w}/Z_${t}_${m}.dump" | tr -d ' ')"
        if (( sz >= MAX_BYTES )); then
          echo "ERROR: ${name}/${t}.dump is ${sz} bytes (>= 1 MB); use a smaller case" >&2
          exit 1
        fi
        cp "${w}/Z_${t}_${m}.dump" "${d}/${t}.dump"
      fi
    done
    echo "== ${name}: generator3 white e3max=${m}"
    "${RH}" generator3 white "${w}/H_${m}.dump" "${w}/eta_${m}.dump" e3max="${m}" \
        > "${w}/out_eta_${m}.txt" 2> "${w}/err_eta_${m}.txt"
    json_of "${w}/out_eta_${m}.txt" > "${w}/norm_eta_${m}.txt"
    local esz
    esz="$(wc -c < "${w}/eta_${m}.dump" | tr -d ' ')"
    if [[ "${m}" == "${DUMP_E3MAX}" ]]; then
      (( esz < MAX_BYTES )) || { echo "ERROR: ${name}/eta_white.dump is ${esz} bytes" >&2; exit 1; }
      cp "${w}/eta_${m}.dump" "${d}/eta_white.dump"
    elif [[ " ${ETA_DUMP_E3MAXS} " == *" ${m} "* ]]; then
      if (( esz < MAX_BYTES )); then cp "${w}/eta_${m}.dump" "${d}/eta_white_e3max${m}.dump"
      else echo "   eta dump at e3max ${m} is ${esz} bytes: not kept"; fi
    fi
  done
  python3 -I - "${d}/values.json" "${name}" "${ref}" "${emax}" "${w}" "${E3MAXS}" "${DUMP_E3MAX}" \
      "${TERMS}" "${d}" <<'EOF'
import json, os, sys
path, name, ref, emax, w, e3maxs, dump_e3, terms, d = sys.argv[1:10]
e3maxs = [int(x) for x in e3maxs.split()]
terms = terms.split()

def load(fn):
    # keep the %.17g text next to the value
    raw = json.load(open(fn), parse_float=str, parse_int=str)
    out = {}
    for k, v in raw.items():
        if k == "norm":
            continue
        out[k] = float(v)
        out[k + "_str"] = v
    return out

def version(fn):
    for ln in open(fn):
        if ln.startswith("source "):
            return ln[7:].split(" refharness")[0].strip()
    raise RuntimeError("no source line in " + fn)

doc = {
    "case": name, "reference": ref, "emax": int(emax),
    "reference_version": version(os.path.join(w, "eta_%d.dump" % e3maxs[0])),
    "operators": ("RandomThreeBodyOperator(ThreeBodyContext(MakeThreeBodySpace(emax, reference, "
                  "{e3max})), h, seed) of tests/support/three_body_test_support.h, hw 16, "
                  "valence = reference: X = anti-Hermitian seed 11, Y = Hermitian seed 12, "
                  "H = Hermitian seed 13 (tools/bench/n7_inputs.cc). The dumps of X, Y, H are not "
                  "committed; 'inputs' holds their imsrg++ norms (drift check)"),
    "commands": {
        "inputs": "imsrg_n7_inputs --emax %s --reference %s --e3max <e3max> --seed <s> --herm <A|H> --out F" % (emax, ref),
        "term": "refharness comm3 <TERM> X.dump Y.dump Z.dump e3max=<e3max>",
        "generator": "refharness generator3 white H.dump ETA.dump e3max=<e3max>",
    },
    "terms": terms,
    "dump_e3max": int(dump_e3),
    "norm_fields": "zero_body, one_body_norm, two_body_norm: imsrg++ OneBodyNorm/TwoBodyNorm; "
                   "three_body_norm: imsrg++ ThreeBodyNorm (canonical pn Frobenius norm); *_str: %.17g text",
    "cases": [],
}
for m in e3maxs:
    entry = {"e3max": m,
             "inputs": {o: load(os.path.join(w, "norm_%s_%d.txt" % (o, m))) for o in "XYH"},
             "terms": {}}
    for t in terms:
        v = load(os.path.join(w, "norm_%s_%d.txt" % (t, m)))
        v["dump"] = (t + ".dump") if m == int(dump_e3) else None
        entry["terms"][t] = v
    g = load(os.path.join(w, "norm_eta_%d.txt" % m))
    if m == int(dump_e3):
        g["dump"] = "eta_white.dump"
    elif os.path.exists(os.path.join(d, "eta_white_e3max%d.dump" % m)):
        g["dump"] = "eta_white_e3max%d.dump" % m
    else:
        g["dump"] = None
    entry["generator_white"] = g
    doc["cases"].append(entry)
with open(path, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
EOF
}

if [[ ! -x "${INPUTS}" ]]; then
  cmake --preset release-openmp -B "${BENCH_DIR}" -DIMSRG_BUILD_BENCH=ON > /dev/null
  cmake --build "${BENCH_DIR}" --target imsrg_n7_inputs
fi
one_case He4_emax2 He4 2
one_case O16_emax2 O16 2
