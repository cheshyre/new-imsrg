#!/usr/bin/env bash
# Generates tests/golden/basis/*.txt ("imsrg-basis-dump v1", see tests/golden/README.md) with
# `refharness dump-basis` from the old imsrg++ library. Golden data for M3 (basis).
#
# Usage: tools/reference/gen_golden_basis.sh [refharness-binary]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
RH="${1:-${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}/harness-build/refharness}"
OUT="${REPO}/tests/golden/basis"
mkdir -p "${OUT}"

# name                 reference  valence_space
cases=(
  "O16                 O16        O16"
  "He4                 He4        He4"
  "Ne20                Ne20       Ne20"
  "sd-shell            default    sd-shell"
  "p-shell             default    p-shell"
  "Ne20_sd-shell       Ne20       sd-shell"
)

for emax in 2 4; do
  for c in "${cases[@]}"; do
    read -r name ref val <<<"${c}"
    f="${OUT}/emax${emax}_${name}.txt"
    "${RH}" dump-basis emax="${emax}" reference="${ref}" valence_space="${val}" hw=16 out="${f}" \
      > /dev/null
    echo "wrote ${f} ($(wc -l < "${f}") lines)"
  done
done

# Cases with active cuts (added in M3): an E3max cut, and an lmax cut (truncated orbit indexing).
extra=(
  "emax4_O16_e3max7                  emax=4 e3max=7 reference=O16 valence_space=O16"
  "emax4_Ne20_sd-shell_lmax2_e3max8  emax=4 e3max=8 lmax=2 reference=Ne20 valence_space=sd-shell"
)
for c in "${extra[@]}"; do
  read -r name args <<<"${c}"
  f="${OUT}/${name}.txt"
  # shellcheck disable=SC2086
  "${RH}" dump-basis ${args} hw=16 out="${f}" > /dev/null
  echo "wrote ${f} ($(wc -l < "${f}") lines)"
done
