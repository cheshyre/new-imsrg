#!/usr/bin/env bash
# Golden data for M4 (operator): normal ordering and basis transformation by the old code.
#   tests/golden/operator/emax2/
#     H_NO_O16.dump          Operator::DoNormalOrdering of hamiltonian/emax2_O16/H.dump (ref O16)
#     H_NO_O16_undo.dump     UndoNormalOrdering of H_NO_O16.dump (must reproduce H.dump)
#     randomH_NO_Ne20.dump   random_H normal ordered w.r.t. Ne20 (0d5/2 occupation 1/3)
#     randomA_NO_Ne20.dump   commutator/emax2_O16/random_A.dump w.r.t. Ne20 (anti-Hermitian)
#     C_HF_O16_2N.txt        C of hf/emax2_O16_2N/hf.json (row = HO orbit), %.17g
#     H_HFbasis.dump         HartreeFock::TransformToHFBasis(H.dump) with that C
#     norms.json             refharness norms of every dump
#
# Usage: tools/reference/gen_golden_operator.sh
# Environment: REFHARNESS (default $IMSRG_REFERENCE_ROOT/harness-build/refharness); the
#   harness must contain the normal-order and basis-transform subcommands.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
RH="${REFHARNESS:-${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}/harness-build/refharness}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
G="${REPO}/tests/golden"
OUT="${G}/operator/emax2"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${OUT}"

H="${G}/hamiltonian/emax2_O16/H.dump"
"${RH}" normal-order "${H}" "${OUT}/H_NO_O16.dump" reference=O16 > "${TMP}/a.log" 2>&1
"${RH}" normal-order "${OUT}/H_NO_O16.dump" "${OUT}/H_NO_O16_undo.dump" reference=O16 undo=true \
    > "${TMP}/b.log" 2>&1
"${RH}" normal-order "${G}/hamiltonian/emax2_O16/random_H.dump" "${OUT}/randomH_NO_Ne20.dump" \
    reference=Ne20 > "${TMP}/c.log" 2>&1
"${RH}" normal-order "${G}/commutator/emax2_O16/random_A.dump" "${OUT}/randomA_NO_Ne20.dump" \
    reference=Ne20 > "${TMP}/d.log" 2>&1

python3 -I - "${G}/hf/emax2_O16_2N/hf.json" "${OUT}/C_HF_O16_2N.txt" << 'PY'
import json, sys
C = json.load(open(sys.argv[1]))["C"]
with open(sys.argv[2], "w") as f:
    for row in C:
        f.write(" ".join("%.17g" % x for x in row) + "\n")
PY
"${RH}" basis-transform "${H}" "${OUT}/C_HF_O16_2N.txt" "${OUT}/H_HFbasis.dump" > "${TMP}/e.log" 2>&1

# --- undo(DO(H)) vs H: largest absolute deviation over zero/one/two-body values
python3 -I - "${H}" "${OUT}/H_NO_O16_undo.dump" << 'PY'
import sys
def load(p):
    d, sec = {}, None
    for line in open(p):
        t = line.split()
        if not t or t[0] in ("#", "source"): continue
        if t[0] == "zero_body": d[("z",)] = float(t[1])
        elif len(t) == 3 and t[0].isdigit() and sec == 1: d[("1",) + tuple(t[:2])] = float(t[2])
        elif len(t) == 7 and sec == 2: d[("2",) + tuple(t[:6])] = float(t[6])
        if t[0] == "one_body": sec = 1
        elif t[0] == "two_body": sec = 2
        elif t[0] in ("three_body", "end"): sec = None
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = set(a) | set(b)
dev = max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
print("undo deviation: max|undo(DO(H)) - H| = %.3e over %d entries" % (dev, len(keys)))
PY

# --- norms.json
for f in "${OUT}"/*.dump; do
  "${RH}" norms "${f}" out="${TMP}/$(basename "${f}" .dump).norms.json" > "${TMP}/n.log" 2>&1
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
