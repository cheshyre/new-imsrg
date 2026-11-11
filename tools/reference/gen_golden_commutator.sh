#!/usr/bin/env bash
# Golden data for M7 (commutators): term-by-term IMSRG(2) commutators of the OLD imsrg++ at
# emax 2 (O16), written to tests/golden/commutator/emax2_O16/:
#   H.dump, eta_white.dump, eta_atan.dump, random_A.dump, random_H.dump   (inputs)
#   Z_<TERM>__eta_white__H.dump, Z_<TERM>__random_A__random_H.dump        (TERM in the list below + all)
#   norms.json                                                            (imsrg++ norms of everything)
#
# Usage: tools/reference/gen_golden_commutator.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
RH="${REFHARNESS:-${ROOT}/harness-build/refharness}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OUT="${REPO}/tests/golden/commutator/emax2_O16"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${OUT}" "${TMP}/a" "${TMP}/h"

TERMS=(comm110ss comm220ss comm111ss comm121ss comm221ss comm122ss
       comm222_pp_hhss comm222_phss comm222_pp_hh_221ss all)

# --- inputs
cp "${REPO}/tests/golden/hf/emax2_O16_2N/HNO.dump" "${OUT}/H.dump"
common=(emax=2 reference=O16 valence_space=O16 hw=16
        2bme="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
        file2e1max=14 file2e2max=28 file2lmax=14)
"${RH}" dump-hamiltonian "${common[@]}" ops=random_A random_seed=7 out="${TMP}/a" > "${TMP}/a.log" 2>&1
cp "${TMP}/a/random_A.dump" "${OUT}/random_A.dump"
"${RH}" dump-hamiltonian "${common[@]}" ops=random_H random_seed=11 out="${TMP}/h" > "${TMP}/h.log" 2>&1
cp "${TMP}/h/random_H.dump" "${OUT}/random_H.dump"
"${RH}" generator white "${OUT}/H.dump" "${OUT}/eta_white.dump" > "${TMP}/g1.log" 2>&1
"${RH}" generator atan "${OUT}/H.dump" "${OUT}/eta_atan.dump" > "${TMP}/g2.log" 2>&1

# --- term-by-term commutators
for t in "${TERMS[@]}"; do
  "${RH}" commutator "${t}" "${OUT}/eta_white.dump" "${OUT}/H.dump" \
      "${OUT}/Z_${t}__eta_white__H.dump" > "${TMP}/c.log" 2>&1
  "${RH}" commutator "${t}" "${OUT}/random_A.dump" "${OUT}/random_H.dump" \
      "${OUT}/Z_${t}__random_A__random_H.dump" > "${TMP}/c.log" 2>&1
done

# --- norms.json (imsrg++ norms of every dump)
for f in "${OUT}"/*.dump; do
  # A single term of a hermitian Z need not satisfy the dump's hermiticity check when its entries
  # are rounding noise (comm111ss on eta_white,H); such an unreadable dump is not kept.
  if ! "${RH}" norms "${f}" out="${TMP}/$(basename "${f}" .dump).norms.json" > "${TMP}/n.log" 2>&1; then
    echo "note: dropping unreadable $(basename "${f}"): $(tail -n 1 "${TMP}/n.log")"
    rm -f "${f}"
  fi
done
python3 -I - "${TMP}" "${OUT}/norms.json" << 'PY'
import glob, json, os, sys
from collections import OrderedDict
d = OrderedDict()
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.norms.json"))):
    d[os.path.basename(f)[:-len(".norms.json")]] = json.load(open(f))
json.dump(d, open(sys.argv[2], "w"), indent=2)
open(sys.argv[2], "a").write("\n")
PY
du -sh "${OUT}"
