#!/usr/bin/env bash
# Golden data of the scalar-tensor commutator terms of the owner's imsrg++ fork at 79e9330 (the
# code of the archived 27Al NREFT responses, design/50), for the fork-compat path of the new code
# (CommutatorOptions::tensor_fork_ph_missing_rank_phase, QUESTIONS.md AL-F1). Written to
#
#   tests/golden/tensor/fork-79e9330/emax2_He4/
#     inputs/random_A.dump                       X: random anti-Hermitian scalar (OperatorDump v1)
#     inputs/rand_<J>_<dTz>_<par>_H_<seed>.tdump Y: random Hermitian tensors, 1B + 2B (rank 1, 3,
#                                                1 with odd parity, and rank 2 as an even control)
#     commutator/Z_<term>__A__<Y>.tdump          the fork's comm111st, comm121st, comm122st,
#                                                comm222_pp_hh_221st, comm222_phst and `all`
#
# (raw tensor dump v0, design/15; identically zero terms are not written). The directory
# tests/golden/tensor/fork-79e9330 belongs to this script (gen_golden_tensor.sh keeps it).
#
# Usage: tools/reference/gen_golden_tensor_fork.sh   (on a compute node)
# Environment: FORKHARNESS (default $REF_BUILD/fork-79e9330/build-harness/forkharness, built by
#   tools/reference/build_fork.sh plain; it must have the commutator-tensor subcommand).
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ref_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
FH="${FORKHARNESS:-${REF_BUILD}/fork-79e9330/build-harness/forkharness}"
VERSION="$(cat "${REF_BUILD}/FORK_79E9330_SRC_SHA256")"
# one thread: the fork marks its 9j cache complete per rank only (tensor_transform_first_pass),
# so a second parity of the same rank would fill the cache from parallel threads
export OMP_NUM_THREADS=1
OUT="${REPO}/tests/golden/tensor/fork-79e9330"
rm -rf "${OUT}"
for case in "2 He4"; do
  read -r emax ref <<< "${case}"
  d="${OUT}/emax${emax}_${ref}"
  mkdir -p "${d}/inputs" "${d}/commutator"
  "${FH}" commutator-tensor "${emax}" "${ref}" "${d}" "${VERSION}" > /dev/null
done
du -sh "${OUT}"
