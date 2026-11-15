#!/usr/bin/env bash
# IMSRG(3) reference data for milestone M10 (comm223ss and perturbative triples of imsrg++):
#   tests/golden/imsrg3/<case>/        random-operator [X, Y]_3 (full-mode comm223ss) goldens
#   tests/golden/e2e/IT-8/<case>/harness_triples.json   refharness flow --perturbative-triples
#
# Usage: tools/reference/gen_golden_imsrg3.sh [comm223|triples|all]     (default: all)
#
#   comm223  For He4 and O16 at emax 2 and O16 at emax 3: X = RandomOperator(Scalar(2, kAntiHermitian),
#            seed 1), Y = the same with kHermitian and seed 2 (scales 1), as the layout benchmark makes
#            them (tools/bench/imsrg3_layout, `--write-dumps`, which also writes epsilon.txt); then
#              refharness comm223 emax=E e3max=M reference=R x=X.dump y=Y.dump epsilon=epsilon.txt
#                                 out=Z3.dump reps=1
#            Z3.dump (OperatorDump v1 with a three_body section) is written for DUMP_E3MAX = 3 at
#            emax 2 (the largest E3max whose dump stays below 1 MB; E3max 4 is 1.6 MB). values.json
#            holds the printed 3B norm and perturbative-triples energy (%.17g) for every E3max listed
#            in CASES below (no dump for those). Cases: He4_emax2, O16_emax2 (E3max 3 with dump, 6),
#            O16_emax3 (E3max 9; no dump).
#   triples  refharness flow --perturbative-triples with the settings of the IT-8 jobs of
#            gen_golden_e2e.sh (method magnus = harness defaults; the same 2N file) for He4 emax 2 and
#            O16 emax 3 -> tests/golden/e2e/IT-8/{He4_emax2,emax3}/harness_triples.json.
#
# Env: IMSRG_REFERENCE_ROOT (default /Users/cheshyre/Downloads/new-imsrg-reference),
#      HARNESS_BUILD_DIR_D35 (default $ROOT/harness-build-d35; must contain a refharness built from
#        this checkout: `HARNESS_BUILD_DIR_D35=<dir> tools/reference/build_imsrg_devel.sh patched-harness`),
#      BENCH_BUILD_DIR (default build/golden-imsrg3 of this checkout: release-openmp preset with
#        -DIMSRG_BUILD_BENCH=ON, configured and built on demand), OMP_NUM_THREADS (default 8).
# The harness is the D35 patched build; IMSRG_REFERENCE_SHA_FILE selects its version string.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
RH="${HARNESS_BUILD_DIR_D35:-${ROOT}/harness-build-d35}/refharness"
BENCH_DIR="${BENCH_BUILD_DIR:-${REPO}/build/golden-imsrg3}"
BENCH="${BENCH_DIR}/tools/bench/imsrg3_layout/imsrg3_layout_bench"
export IMSRG_REFERENCE_SHA_FILE="${IMSRG_REFERENCE_SHA_FILE:-${ROOT}/REFERENCE_SRC_SHA256_D35}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
F2="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
OUT="${REPO}/tests/golden/imsrg3"
E2E="${REPO}/tests/golden/e2e"
DUMP_E3MAX=3
what="${1:-all}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# one_case <case> <reference> <emax> <dump_e3max|-> <e3max...>
one_case() {
  local name="$1" ref="$2" emax="$3" dump_e3="$4"; shift 4
  local d="${OUT}/${name}" w="${tmp}/${name}"
  mkdir -p "${d}" "${w}"
  "${BENCH}" --emax "${emax}" --reference "${ref}" --seed 1 --kernels pack --reps 1 \
      --write-dumps "${w}" > /dev/null
  cp "${w}/epsilon.txt" "${d}/epsilon.txt"
  if [[ "${dump_e3}" != "-" ]]; then cp "${w}/X.dump" "${w}/Y.dump" "${d}/"; fi
  : > "${w}/lines.txt"
  local m
  for m in "$@"; do
    local extra=()
    [[ "${m}" == "${dump_e3}" ]] && extra=("out=${d}/Z3.dump")
    echo "== ${name}: comm223 emax=${emax} e3max=${m}"
    "${RH}" comm223 emax="${emax}" e3max="${m}" reference="${ref}" x="${w}/X.dump" y="${w}/Y.dump" \
        epsilon="${w}/epsilon.txt" reps=1 "${extra[@]+"${extra[@]}"}" > "${w}/out_${m}.txt"
    local nrm tri
    nrm="$(awk '$1=="comm223ss_three_body_norm"{print $2}' "${w}/out_${m}.txt")"
    tri="$(awk '$1=="triples_energy"{print $2}' "${w}/out_${m}.txt")"
    echo "${m} ${nrm} ${tri}" >> "${w}/lines.txt"
    echo "   3B norm ${nrm}, triples energy ${tri}"
  done
  python3 -I - "${d}/values.json" "${name}" "${ref}" "${emax}" "${dump_e3}" "${w}/lines.txt" \
      "$(head -n 2 "${w}/out_$1.txt" | tail -n 1 | sed 's/.* version //')" <<'EOF'
import json, sys
path, name, ref, emax, dump_e3, lines, version = sys.argv[1:8]
cases = []
for ln in open(lines):
    m, nrm, tri = ln.split()
    cases.append({"e3max": int(m), "three_body_norm": float(nrm), "three_body_norm_str": nrm,
                  "triples_energy": float(tri), "triples_energy_str": tri,
                  "dump": "Z3.dump" if m == dump_e3 else None})
doc = {
    "case": name, "reference": ref, "emax": int(emax),
    "reference_version": version,
    "operators": ("X = RandomOperator(Scalar(2, kAntiHermitian), {seed 1, one_body_scale 1, "
                  "two_body_scale 1}); Y = the same with kHermitian and seed 2; epsilon.txt as "
                  "tools/bench/imsrg3_layout/bench_main.cc --write-dumps"),
    "command": ("refharness comm223 emax=%s e3max=<e3max> reference=%s x=X.dump y=Y.dump "
                "epsilon=epsilon.txt out=Z3.dump reps=1" % (emax, ref)),
    "cases": cases,
}
with open(path, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
EOF
}

# flow_case <job dir name> <reference> <emax>
flow_case() {
  local job="$1" ref="$2" emax="$3" w="${tmp}/flow_$1"
  mkdir -p "${w}" "${E2E}/IT-8/${job}"
  local -a args=(emax="${emax}" reference="${ref}" valence_space="${ref}" 2bme="${F2}" file2e1max=14
                 file2e2max=28 file2lmax=14 hw=16)
  echo "== IT-8/${job}: refharness flow ${args[*]} --perturbative-triples"
  "${RH}" flow "${args[@]}" out="${w}" --perturbative-triples > "${w}/stdout.txt" 2> "${w}/stderr.txt"
  python3 -I - "${w}/result.json" "${E2E}/IT-8/${job}/harness_triples.json" "IT-8/${job}" \
      "refharness flow ${args[*]} --perturbative-triples" <<'EOF'
import json, sys
res, out, job, cmd = sys.argv[1:5]
r = json.load(open(res))
doc = {"job": job, "reference_version": r["source"].replace(" refharness flow", ""),
       "E0": r["E0"], "dE": r["perturbative_triples"], "E0_plus_dE": r["e0_plus_triples"],
       "steps": r["steps"], "command": cmd,
       "note": "refharness flow of the D35 patched reference, defaults = imsrg++ method=magnus; "
               "values with %.17g precision, compare with result.json (imsrg++ log, 8 decimals)"}
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
print("   dE %.17g  E0 %.17g  E0+dE %.17g  steps %d" % (doc["dE"], doc["E0"], doc["E0_plus_dE"], doc["steps"]))
EOF
}

if [[ "${what}" == "comm223" || "${what}" == "all" ]]; then
  if [[ ! -x "${BENCH}" ]]; then
    cmake --preset release-openmp -B "${BENCH_DIR}" -DIMSRG_BUILD_BENCH=ON > /dev/null
    cmake --build "${BENCH_DIR}" --target imsrg3_layout_bench
  fi
  one_case He4_emax2 He4 2 "${DUMP_E3MAX}" "${DUMP_E3MAX}" 6
  one_case O16_emax2 O16 2 "${DUMP_E3MAX}" "${DUMP_E3MAX}" 6
  one_case O16_emax3 O16 3 - 9
fi
if [[ "${what}" == "triples" || "${what}" == "all" ]]; then
  flow_case He4_emax2 He4 2
  flow_case emax3 O16 3
fi
