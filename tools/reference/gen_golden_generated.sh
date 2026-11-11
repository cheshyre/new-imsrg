#!/usr/bin/env bash
# Regenerates the emax-4 golden dumps that are too large to commit (several MB each) into a
# directory outside the source tree (default: build/golden-generated, gitignored). Tests find it
# through the CMake cache variable IMSRG_GENERATED_GOLDEN_DIR and are skipped when it is absent.
#
# Usage: tools/reference/gen_golden_generated.sh [OUT_DIR]
# Needs the reference harness (tools/reference/build_imsrg_devel.sh harness) and the big inputs.
# Writes OUT_DIR/hamiltonian/emax4_O16/{T,V2,H,V3_NO2B,Rp2}.dump + norms.json and
# OUT_DIR/hf/emax4_O16_{2N,2N3N}/{hf.json,HNO.dump}. Takes about 2 s.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO}/build/golden-generated}"
RH="${REFHARNESS:-${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}/harness-build/refharness}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
F2="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
F3="${DATA}/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
common=(hw=16 2bme="${F2}" file2e1max=14 file2e2max=28 file2lmax=14
        file3e1max=6 file3e2max=12 file3e3max=18 3bme_type=no2b)

run() {  # run <dir> <args...>
  local dir="$1"
  shift
  mkdir -p "${dir}"
  "${RH}" "$@" out="${dir}" > "${dir}.log" 2>&1 || { cat "${dir}.log"; exit 1; }
  rm -f "${dir}.log"
  echo "wrote ${dir}"
}

run "${OUT}/hamiltonian/emax4_O16" dump-hamiltonian emax=4 reference=O16 valence_space=O16 \
    "${common[@]}" 3bme="${F3}" ops=Rp2
run "${OUT}/hf/emax4_O16_2N" hf emax=4 reference=O16 valence_space=O16 "${common[@]}" dump_hno=true
run "${OUT}/hf/emax4_O16_2N3N" hf emax=4 reference=O16 valence_space=O16 "${common[@]}" \
    3bme="${F3}" dump_hno=true
