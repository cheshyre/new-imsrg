#!/usr/bin/env bash
# Golden data for M4 (operator), M5 (io), M6 (hf) and MBPT from `refharness`:
#   tests/golden/hamiltonian/emax2_O16/   T, V2, H (= T + V2, bare, HO basis), V3_NO2B (3N
#                                          normal ordered w.r.t. O16, HO basis), Rp2, random_H
#                                          OperatorDumps + norms.json
#   tests/golden/hf/emax{2,4}_O16_{2N,2N3N}/hf.json   E_HF, SPEs, iterations, C, MP2, MP3
#   tests/golden/hf/emax2_O16_{2N,2N3N}/HNO.dump      HF-basis normal-ordered H (NO2B)
#
# Usage: tools/reference/gen_golden_hamiltonian.sh [--generated]
#   --generated  additionally writes the emax-4 dumps (several MB, not committed) into
#                tests/golden/generated/ (gitignored)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
RH="${REFHARNESS:-${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}/harness-build/refharness}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
F2="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
F3="${DATA}/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
G="${REPO}/tests/golden"

common=(hw=16 2bme="${F2}" file2e1max=14 file2e2max=28 file2lmax=14
        file3e1max=6 file3e2max=12 file3e3max=18 3bme_type=no2b)

run() {  # run <dir> <args...>; logs stdout/stderr of the old code to <dir>/../<name>.log (not kept)
  local dir="$1"; shift
  mkdir -p "${dir}"
  local t0=$SECONDS
  "${RH}" "$@" out="${dir}" > "${dir}.log" 2>&1
  echo "wrote ${dir} ($((SECONDS - t0)) s)"
  rm -f "${dir}.log"
}

# --- bare Hamiltonian pieces at emax 2 (committed)
run "${G}/hamiltonian/emax2_O16" dump-hamiltonian emax=2 reference=O16 valence_space=O16 \
    "${common[@]}" 3bme="${F3}" ops=Rp2,random_H random_seed=1

# --- HF + MBPT
for emax in 2 4; do
  dump=false; [[ ${emax} == 2 ]] && dump=true
  run "${G}/hf/emax${emax}_O16_2N" hf emax=${emax} reference=O16 valence_space=O16 \
      "${common[@]}" dump_hno=${dump}
  run "${G}/hf/emax${emax}_O16_2N3N" hf emax=${emax} reference=O16 valence_space=O16 \
      "${common[@]}" 3bme="${F3}" dump_hno=${dump}
done

if [[ "${1:-}" == "--generated" ]]; then
  GEN="${G}/generated"
  run "${GEN}/hamiltonian/emax4_O16" dump-hamiltonian emax=4 reference=O16 valence_space=O16 \
      "${common[@]}" 3bme="${F3}" ops=Rp2
  run "${GEN}/hf/emax4_O16_2N" hf emax=4 reference=O16 valence_space=O16 "${common[@]}" dump_hno=true
  run "${GEN}/hf/emax4_O16_2N3N" hf emax=4 reference=O16 valence_space=O16 "${common[@]}" \
      3bme="${F3}" dump_hno=true
fi
