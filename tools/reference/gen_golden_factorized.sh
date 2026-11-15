#!/usr/bin/env bash
# Golden data for M10 track 3f2 (IMSRG(3f2), design decision D32): the factorized double
# commutator [Eta,[Eta,Gamma]_3b]_1b,2b and the BCH transform/product with and without the
# factorized correction, from the OLD imsrg++ built from the D35-patched reference sources
# (default REFHARNESS: harness-build-3f2).
#   tests/golden/factorized/emax2_He6/      Z_d32__<ETA>__<GAMMA>.dump, random inputs, norms.json
#   tests/golden/factorized/emax2_O16/      Z_d32__<ETA>__<GAMMA>.dump (inputs are read in place
#                                           from tests/golden/commutator/emax2_O16), norms.json
#   tests/golden/factorized/bch_emax2_O16/  BCH transform/product dumps, bch_scaled_norms.txt,
#                                           norms.json
# Self-checks (all vs direct; d32 vs chi1b_231 + chi1b_232) are printed, not kept.
#
# Usage: tools/reference/gen_golden_factorized.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
R="/Users/cheshyre/Downloads/new-imsrg-reference"
RH="${REFHARNESS:-${R}/harness-build-3f2/refharness}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# The dumps' source line names the D35-patched reference build.
mkdir -p "${TMP}/refroot"
cp "${R}/REFERENCE_SRC_SHA256_D35" "${TMP}/refroot/REFERENCE_SRC_SHA256"
export IMSRG_REFERENCE_ROOT="${TMP}/refroot"
export IMSRG_REFERENCE_SHA_FILE="${R}/REFERENCE_SRC_SHA256_D35"

GOLD="${REPO}/tests/golden/factorized"
CIN="${REPO}/tests/golden/commutator/emax2_O16"
BME2=(2bme="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
      file2e1max=14 file2e2max=28 file2lmax=14)
HE6=(emax=2 reference=He6 valence_space=He6 hw=16 "${BME2[@]}")

# Python helper: parse dump elements (one_body and two_body lines) into {key: value}.
cat > "${TMP}/dumpcmp.py" << 'PY'
import sys
def parse(p):
    sec = None; v = {}
    for line in open(p):
        t = line.split()
        if not t:
            continue
        if t[0] in ("one_body", "two_body", "three_body", "end"):
            sec = t[0]; continue
        if sec in ("one_body", "two_body"):
            v[(sec,) + tuple(t[:-1])] = float(t[-1])
    return v
def maxdiff(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cmp":      # cmp A B: max abs diff over all elements / max |A|
        a, b = parse(sys.argv[2]), parse(sys.argv[3])
        m = max(abs(x) for x in a.values())
        d = maxdiff(a, b)
        print("n=%d max|Z|=%.6g maxdiff=%.6g rel=%.3g" % (len(set(a) | set(b)), m, d, d / m))
    else:                  # sum X Y Z: max abs (X + Y - Z)
        x, y, z = parse(sys.argv[2]), parse(sys.argv[3]), parse(sys.argv[4])
        s = {k: x.get(k, 0.0) + y.get(k, 0.0) for k in set(x) | set(y)}
        m = max(abs(v) for v in z.values())
        print("n=%d max|Z|=%.6g maxdiff=%.3g" % (len(set(s) | set(z)), m, maxdiff(s, z)))
PY

# make_input STEM OPS SEED: dump-hamiltonian into TMP, keep as OUT/STEM.dump
make_input() {
  local stem="$1" ops="$2" seed="$3"
  local sub="${TMP}/in_${stem}"
  mkdir -p "${sub}"
  "${RH}" dump-hamiltonian "${HE6[@]}" ops="${ops}" random_seed="${seed}" out="${sub}" \
      > "${sub}.log" 2>&1
  cp "${sub}/${ops}.dump" "${OUT}/${stem}.dump"
}

# norms_dir: imsrg++ norms of every dump in OUT, keyed by file stem, into OUT/norms.json
norms_dir() {
  local f base
  rm -rf "${TMP}/norms"
  mkdir -p "${TMP}/norms"
  for f in "${OUT}"/*.dump; do
    base="$(basename "${f}" .dump)"
    "${RH}" norms "${f}" out="${TMP}/norms/${base}.norms.json" > "${TMP}/n.log" 2>&1
  done
  python3 -I - "${TMP}/norms" "${OUT}/norms.json" << 'PY'
import glob, json, os, sys
from collections import OrderedDict
d = OrderedDict()
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.norms.json"))):
    d[os.path.basename(f)[:-len(".norms.json")]] = json.load(open(f))
json.dump(d, open(sys.argv[2], "w"), indent=2)
open(sys.argv[2], "a").write("\n")
PY
}

# pair ETA_PATH GAMMA_PATH ETA_NAME GAMMA_NAME: the d32 golden into OUT
pair() {
  "${RH}" factorized d32 "$1" "$2" "${OUT}/Z_d32__$3__$4.dump" > "${TMP}/f.log" 2>&1
}

# selfcheck LABEL ETA GAMMA
selfcheck() {
  local label="$1" e="$2" g="$3" p
  for p in all direct chi1b_231 chi1b_232 d32; do
    "${RH}" factorized "${p}" "${e}" "${g}" "${TMP}/sc_${p}.dump" > "${TMP}/f.log" 2>&1
  done
  echo "selfcheck ${label}: all vs direct: $(python3 -I "${TMP}/dumpcmp.py" cmp "${TMP}/sc_all.dump" "${TMP}/sc_direct.dump")"
  echo "selfcheck ${label}: d32 - (chi1b_231 + chi1b_232): $(python3 -I "${TMP}/dumpcmp.py" sum "${TMP}/sc_chi1b_231.dump" "${TMP}/sc_chi1b_232.dump" "${TMP}/sc_d32.dump")"
}

# --- emax2_He6
OUT="${GOLD}/emax2_He6"
mkdir -p "${OUT}"
make_input random_A1 random_A 41
make_input random_A2 random_A 42
make_input random_H1 random_H 43
pair "${OUT}/random_A1.dump" "${OUT}/random_H1.dump" random_A1 random_H1
pair "${OUT}/random_A1.dump" "${OUT}/random_A2.dump" random_A1 random_A2
selfcheck "He6 (random_A1, random_H1)" "${OUT}/random_A1.dump" "${OUT}/random_H1.dump"
selfcheck "He6 (random_A1, random_A2)" "${OUT}/random_A1.dump" "${OUT}/random_A2.dump"
norms_dir
du -sh "${OUT}"

# --- emax2_O16 (inputs in place)
OUT="${GOLD}/emax2_O16"
mkdir -p "${OUT}"
pair "${CIN}/eta_white.dump" "${CIN}/H.dump" eta_white H
pair "${CIN}/random_A.dump" "${CIN}/random_H.dump" random_A random_H
pair "${CIN}/eta_white.dump" "${CIN}/random_A.dump" eta_white random_A
selfcheck "O16 (eta_white, H)" "${CIN}/eta_white.dump" "${CIN}/H.dump"
norms_dir
du -sh "${OUT}"

# --- bch_emax2_O16
# Scales have 3 significant digits. ||eta_white|| = 1.0764164, ||random_A|| = 241.13063,
# ||H|| = 330.92558 (tests/golden/commutator/emax2_O16/norms.json).
OUT="${GOLD}/bch_emax2_O16"
mkdir -p "${OUT}"
: > "${OUT}/bch_scaled_norms.txt"
FACT=(--factorized --factorized-1b-intermediates=1 --factorized-2b-intermediates=0)
# bch NAME MODE A B SCALE_A SCALE_B [OPTS...]
bch() {
  local name="$1" mode="$2" a="$3" b="$4" sa="$5" sb="$6"
  shift 6
  echo "== ${name}: bch ${mode} ${a} ${b} scale_a=${sa} scale_b=${sb} $*" >> "${OUT}/bch_scaled_norms.txt"
  "${RH}" bch "${mode}" "${CIN}/${a}.dump" "${CIN}/${b}.dump" "${OUT}/${name}.dump" \
      scale_a="${sa}" scale_b="${sb}" "$@" >> "${OUT}/bch_scaled_norms.txt" 2> "${TMP}/b.log" \
      || { cat "${TMP}/b.log"; exit 1; }
}
bch transform_H__eta_white_x0.0929__d32 transform H eta_white 1 0.0929 "${FACT[@]}"
bch transform_H__eta_white_x1.86__d32 transform H eta_white 1 1.86 "${FACT[@]}"
bch transform_H__eta_white_x1.86__none transform H eta_white 1 1.86
bch product_eta_white_x0.0464__random_A_x0.000415__d32 product eta_white random_A 0.0464 0.000415 "${FACT[@]}"
bch product_eta_white_x0.0929__random_A_x0.00829__d32 product eta_white random_A 0.0929 0.00829 "${FACT[@]}"
bch product_eta_white_x0.0929__random_A_x0.00829__none product eta_white random_A 0.0929 0.00829
norms_dir
du -sh "${OUT}"
du -sh "${GOLD}"
