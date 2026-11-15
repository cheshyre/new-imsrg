#!/usr/bin/env bash
# Regenerates the refharness-based flow goldens (M1): the IT-2 trajectory, the hunter-gatherer
# cases (IT-2-HG) and IT-10 (IMSRG(3f2) with the owner's recipe, D32).
# Usage: tools/reference/gen_golden_flow.sh [refharness-binary]
# Env: FLOW_JOBS (default "IT-2 IT-2-HG IT-10"), IMSRG_GOLDEN_E2E_OUT (default tests/golden/e2e of
#      the checkout this script lives in), IMSRG_REFERENCE_SHA_FILE (default: the D35 patched build's).
# Needs the same input as gen_golden_e2e.sh (default me2j path in rh_common). OMP_NUM_THREADS=8.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
RH="${1:-${ROOT}/harness-build-d35/refharness}"
export IMSRG_REFERENCE_SHA_FILE="${IMSRG_REFERENCE_SHA_FILE:-${ROOT}/REFERENCE_SRC_SHA256_D35}"
OUT="${IMSRG_GOLDEN_E2E_OUT:-${REPO}/tests/golden/e2e}"
FLOW_JOBS="${FLOW_JOBS:-IT-2 IT-2-HG IT-10}"
export OMP_NUM_THREADS=8
CAP="${IT_TIMEOUT_S:-2700}"   # perl alarm = portable wall-time cap

run() {  # run <outdir> <args...>
  local d="$1"; shift
  mkdir -p "${d}"
  echo "== refharness flow $* -> ${d}"
  perl -e 'alarm shift; exec @ARGV' "${CAP}" "${RH}" flow "$@" out="${d}" 2> "${d}/stderr.log" >/dev/null
  rm -f "${d}/stderr.log"
}

if [[ " ${FLOW_JOBS} " == *" IT-2 "* ]]; then
# IT-2: full %.17g trajectory in a scratch dir; commit only steps 0-9 and the last step.
tmp="$(mktemp -d)"
run "${tmp}" emax=4 reference=O16 valence_space=O16 hw=16
{ sed -n 1,11p "${tmp}/trajectory.csv"; tail -n 1 "${tmp}/trajectory.csv"; } > "${OUT}/IT-2/trajectory.csv"
cp "${tmp}/result.json" "${OUT}/IT-2/refharness_result.json"
fi

if [[ " ${FLOW_JOBS} " == *" IT-2-HG "* ]]; then
  # IMSRG(2) + hunter-gatherer Magnus (`--hunter-gatherer`: GatherOmega instead of NewOmega, and the
  # final gather of IMSRGSolver::Solve). Full %.17g trajectories (every step) plus result.json.
  HG="${OUT}/IT-2/hunter_gatherer"
  run "${HG}/omax_0.25" emax=4 reference=O16 valence_space=O16 hw=16 --hunter-gatherer --omega-norm-max 0.25
  run "${HG}/omax_0.10" emax=4 reference=O16 valence_space=O16 hw=16 --hunter-gatherer --omega-norm-max 0.1
  # small and fast, for unit-level tests
  run "${HG}/He4_emax2_omax_0.10" emax=2 reference=He4 valence_space=He4 hw=16 --hunter-gatherer --omega-norm-max 0.1
fi

if [[ " ${FLOW_JOBS} " == *" IT-10 "* ]]; then
  # IT-10: IMSRG(3f2) as defined by owner decision D32: IMSRG(2) with hunter-gatherer Magnus,
  # omega_norm_max 0.1, and the factorized double-commutator correction in BCH with 1b intermediates
  # on and 2b intermediates off (pyIMSRG: SetHunterGatherer(True), SetOmegaNormMax(0.1),
  # BCH.SetUseFactorizedCorrection(True), SetUse_1b_Intermediates(True), SetUse_2b_Intermediates(False)).
  RECIPE=(--factorized --hunter-gatherer --omega-norm-max 0.1
          --factorized-1b-intermediates 1 --factorized-2b-intermediates 0)
  run "${OUT}/IT-10/O16_emax2" emax=2 reference=O16 valence_space=O16 hw=16 "${RECIPE[@]}"
  run "${OUT}/IT-10/He4_emax2" emax=2 reference=He4 valence_space=He4 hw=16 "${RECIPE[@]}"
  run "${OUT}/IT-10/O16_emax3" emax=3 reference=O16 valence_space=O16 hw=16 "${RECIPE[@]}"
fi
