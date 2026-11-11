#!/usr/bin/env bash
# Golden data for M5 (io): tokyo .snt and NuShellX .int/.sp writers of the old code.
#   tests/golden/io/O16_sdshell.{snt,int,sp}
#     from tests/golden/hf/emax2_O16_2N/HNO.dump (core O16) with valence_space=sd-shell
#
# Usage: tools/reference/gen_golden_io.sh
# Environment: REFHARNESS (default $IMSRG_REFERENCE_ROOT/harness-build/refharness); the harness
#   must contain the write-valence subcommand.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
RH="${REFHARNESS:-${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}/harness-build/refharness}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
G="${REPO}/tests/golden"
OUT="${G}/io"
mkdir -p "${OUT}"

"${RH}" write-valence "${G}/hf/emax2_O16_2N/HNO.dump" valence_space=sd-shell out="${OUT}/O16_sdshell"
"${RH}" write-valence "${G}/hf/emax2_O16_2N/HNO.dump" valence_space=sd-shell mode=op out="${OUT}/O16_sdshell_op"
