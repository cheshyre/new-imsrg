#!/usr/bin/env bash
# Extra golden data for M7 (commutators): term-by-term IMSRG(2) commutators of the OLD imsrg++ at
# emax 1, for hermiticity pairs and references missing from gen_golden_commutator.sh:
#   tests/golden/commutator/emax1_He4/  closed-shell He4 reference: (A,H), (H,A), (H,H), (A,A)
#   tests/golden/commutator/emax1_He6/  open-shell He6 reference (neutron 0p3/2 occupation 0.5):
#                                       (A,H), (H,H)
# Each directory holds
#   random_A*.dump, random_H*.dump                                     (inputs)
#   Z_<TERM>__<X>__<Y>.dump                                           (TERM in the list below)
#   norms.json                                                        (imsrg++ norms of every dump)
# Dumps that the reader cannot read back (non-hermitian rounding noise) are dropped and listed.
#
# Usage: tools/reference/gen_golden_commutator_extra.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
RH="${REFHARNESS:-${ROOT}/harness-build/refharness}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

TERMS=(comm110ss comm220ss comm111ss comm121ss comm221ss comm122ss
       comm222_pp_hhss comm222_phss)

BME2=(2bme="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
      file2e1max=14 file2e2max=28 file2lmax=14)
HE4=(emax=1 reference=He4 valence_space=He4 hw=16 "${BME2[@]}")
HE6=(emax=1 reference=He6 valence_space=He6 hw=16 "${BME2[@]}")

# make_input DIR STEM OPS SEED REFARGS...: dump-hamiltonian into TMP, keep it as DIR/STEM.dump
make_input() {
  local dir="$1" stem="$2" ops="$3" seed="$4"
  shift 4
  local sub="${TMP}/${dir}/${stem}"
  mkdir -p "${sub}"
  "${RH}" dump-hamiltonian "$@" ops="${ops}" random_seed="${seed}" out="${sub}" \
      > "${sub}.log" 2>&1
  cp "${sub}/${ops}.dump" "${OUT}/${stem}.dump"
}

# make_pair DIR X Y: the 8 single-term commutators [X, Y] into DIR/Z_<TERM>__X__Y.dump
make_pair() {
  local x="$1" y="$2"
  local t
  for t in "${TERMS[@]}"; do
    "${RH}" commutator "${t}" "${OUT}/${x}.dump" "${OUT}/${y}.dump" \
        "${OUT}/Z_${t}__${x}__${y}.dump" > "${TMP}/c.log" 2>&1
  done
}

# norms_dir DIR: imsrg++ norms of every dump in OUT, unreadable dumps dropped
norms_dir() {
  local f base
  mkdir -p "${TMP}/norms"
  for f in "${OUT}"/*.dump; do
    base="$(basename "${f}" .dump)"
    # A single term of a hermitian Z need not satisfy the dump's hermiticity check when its
    # entries are rounding noise; such an unreadable dump is not kept.
    if ! "${RH}" norms "${f}" out="${TMP}/norms/${base}.norms.json" > "${TMP}/n.log" 2>&1; then
      echo "note: dropping unreadable ${base}.dump: $(tail -n 1 "${TMP}/n.log")"
      rm -f "${f}"
    fi
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
  rm -rf "${TMP}/norms"
}

# --- emax1_He4 (closed shell)
OUT="${REPO}/tests/golden/commutator/emax1_He4"
mkdir -p "${OUT}"
make_input he4 random_A1 random_A 21 "${HE4[@]}"
make_input he4 random_A2 random_A 22 "${HE4[@]}"
make_input he4 random_H1 random_H 23 "${HE4[@]}"
make_input he4 random_H2 random_H 24 "${HE4[@]}"
make_pair random_A1 random_H1
make_pair random_H1 random_A1
make_pair random_H1 random_H2
make_pair random_A1 random_A2
norms_dir
du -sh "${OUT}"

# --- emax1_He6 (open shell: neutron 0p3/2 filled with occupation 0.5 by imsrg++)
OUT="${REPO}/tests/golden/commutator/emax1_He6"
mkdir -p "${OUT}"
make_input he6 random_A1 random_A 31 "${HE6[@]}"
make_input he6 random_H1 random_H 33 "${HE6[@]}"
make_input he6 random_H2 random_H 34 "${HE6[@]}"
make_pair random_A1 random_H1
make_pair random_H1 random_H2
norms_dir
du -sh "${OUT}"
