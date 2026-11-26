#!/usr/bin/env bash
# Owner decision D35 / D42: regenerates the goldens that contain T_int = T - Tcm or Rp2 from the
# PATCHED imsrg-devel build (tools/reference/patches/imsrg-devel-d35.patch), into this checkout's
# tests/golden/. Prerequisite: `tools/reference/build_imsrg_devel.sh patched`.
#
# Usage: tools/reference/regen_golden_d35.sh [--check]
#   --check   run only the pre-flight (patched binaries present, version file) and exit
#
# Steps, in dependency order (each is the ordinary gen script with environment overrides; no
# golden number is edited by hand):
#   1. gen_golden_hamiltonian.sh  -> hamiltonian/emax2_O16 (T, H, Rp2 change; V2, V3_NO2B,
#                                    random_H do not), hf/* (hf.json, HNO.dump)
#   2. gen_golden_operator.sh     -> operator/emax2 (everything derived from H.dump / hf.json)
#   3. gen_golden_io.sh           -> io/ (valence writers on the new HNO.dump)
#   4. gen_golden_e2e.sh IT-1..IT-7 and IT-8/emax2 (the Hamiltonian of flow/test_magnus.cpp);
#      IT-8 (other jobs), IT-9 and IT-10 are left to the M10 tracks
#   5. gen_golden_flow.sh (IT-2 trajectory only)
#   6. gen_golden_generated.sh    -> $ROOT/golden-generated-d35 (the gitignored emax-4 dumps that
#                                    opslib/hf/io tests read through IMSRG_GENERATED_GOLDEN_DIR)
# Afterwards `tools/reference/d35_restore_unchanged.py` puts back every regenerated dump whose
# content equals the old one (only the version string in its `source` line differs).
# Environment: IMSRG_REFERENCE_ROOT, IMSRG_TEST_DATA_DIR (as in the gen scripts), OMP_NUM_THREADS.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ref_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${REF_BUILD}"
export REFHARNESS="${ROOT}/harness-build-d35/refharness"
export IMSRG_REFERENCE_EXE="${ROOT}/build-d35/imsrg++"
export IMSRG_REFERENCE_RUNS="${ROOT}/runs-d35/e2e"
export IMSRG_REFERENCE_SHA_FILE="${REF_SHA_D35}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for f in "${REFHARNESS}" "${IMSRG_REFERENCE_EXE}" "${IMSRG_REFERENCE_SHA_FILE}"; do
  [[ -e "$f" ]] || { echo "missing $f: run tools/reference/build_imsrg_devel.sh patched" >&2; exit 1; }
done
echo "patched reference: $(cat "${IMSRG_REFERENCE_SHA_FILE}")"
[[ "${1:-}" == "--check" ]] && exit 0

bash "${HERE}/gen_golden_hamiltonian.sh"
bash "${HERE}/gen_golden_operator.sh"
bash "${HERE}/gen_golden_io.sh"
bash "${HERE}/gen_golden_e2e.sh" IT-1 IT-2 IT-3 IT-4 IT-5 IT-6 IT-7 IT-8/emax2
FLOW_JOBS="IT-2" bash "${HERE}/gen_golden_flow.sh" "${ROOT}/harness-build-d35/refharness"
bash "${HERE}/gen_golden_generated.sh" "${ROOT}/golden-generated-d35"
python3 -I "${HERE}/d35_restore_unchanged.py"
