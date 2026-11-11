#!/usr/bin/env bash
# Round-trip check of the OperatorDump reader/writer of refharness (M1 acceptance): every
# committed tests/golden/**/*.dump is read by `refharness roundtrip` into the old imsrg++
# Operator and written again; the output must be byte-identical to the input.
#
# Usage: tools/reference/check_roundtrip.sh [refharness-binary]
# Output files go to $IMSRG_REFERENCE_ROOT/roundtrip-check/ (outside the repo, overwritten).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
RH="${1:-${ROOT}/harness-build/refharness}"
WORK="${ROOT}/roundtrip-check"
mkdir -p "${WORK}"

n=0; bad=0
while IFS= read -r f; do
  out="${WORK}/$(echo "${f#"${REPO}/"}" | tr '/' '_')"
  if "${RH}" roundtrip "${f}" "${out}" > "${out}.log" 2>&1 && cmp -s "${f}" "${out}"; then
    n=$((n + 1))
  else
    echo "FAIL: ${f} (see ${out}.log)"; bad=$((bad + 1))
  fi
done < <(find "${REPO}/tests/golden" -name '*.dump' -not -path '*/generated/*' | sort)

if [[ ${bad} -eq 0 ]]; then
  echo "PASS: ${n} OperatorDumps round-trip byte-identically"
else
  echo "FAIL: ${bad} of $((n + bad)) OperatorDumps differ"; exit 1
fi
