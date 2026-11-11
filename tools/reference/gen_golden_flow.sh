#!/usr/bin/env bash
# Regenerates the refharness-based flow goldens (M1): the IT-2 trajectory and IT-10 (IMSRG(3f2)).
# Usage: tools/reference/gen_golden_flow.sh [refharness-binary]
# Needs the same input as gen_golden_e2e.sh (default me2j path in rh_common). OMP_NUM_THREADS=8.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
RH="${1:-${ROOT}/harness-build/refharness}"
OUT="${REPO}/tests/golden/e2e"
export OMP_NUM_THREADS=8
CAP="${IT_TIMEOUT_S:-2700}"   # perl alarm = portable wall-time cap

run() {  # run <outdir> <args...>
  local d="$1"; shift
  mkdir -p "${d}"
  echo "== refharness flow $* -> ${d}"
  perl -e 'alarm shift; exec @ARGV' "${CAP}" "${RH}" flow "$@" out="${d}" 2> "${d}/stderr.log" >/dev/null
  rm -f "${d}/stderr.log"
}

# IT-2: full %.17g trajectory in a scratch dir; commit only steps 0-9 and the last step.
tmp="$(mktemp -d)"
run "${tmp}" emax=4 reference=O16 valence_space=O16 hw=16
{ sed -n 1,11p "${tmp}/trajectory.csv"; tail -n 1 "${tmp}/trajectory.csv"; } > "${OUT}/IT-2/trajectory.csv"
cp "${tmp}/result.json" "${OUT}/IT-2/refharness_result.json"

# IT-10: IMSRG(3f2) (IMSRG(2) + factorized triples correction), full trajectories.
run "${OUT}/IT-10/O16_emax2" emax=2 reference=O16 valence_space=O16 hw=16 --factorized
run "${OUT}/IT-10/He4_emax2" emax=2 reference=He4 valence_space=He4 hw=16 --factorized
run "${OUT}/IT-10/O16_emax3" emax=3 reference=O16 valence_space=O16 hw=16 --factorized
