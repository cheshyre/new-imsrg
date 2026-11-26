#!/usr/bin/env bash
# Golden data from the owner's imsrg++ fork at 79e9330 for the 27Al NREFT task
# (design/50-al27-responses.md C1/C2; reference/06-al27-fork-pipeline.md). Build the fork first:
# `tools/reference/build_fork.sh` (both variants). D42: these goldens are regenerated ONLY through
# this script; a regeneration commit states the reason and the old vs new numbers.
#
# Usage: tools/reference/gen_golden_nreft.sh [bare] [flow] [large]  (default: bare flow)
#   bare   forkharness (plain 79e9330 build) dumps of the bare one-body tables, HO basis, hw 16:
#            tests/golden/nreft/bare_emax2.txt             shorthands + raw building blocks
#            tests/golden/nreft/bare_emax4_shorthands.txt  shorthands only
#          and jho/jdmho/jdpho samples in $REF_BUILD/runs-fork/radial.txt (not committed: the
#          tables test the integrals; kept for debugging, golden budget < 1 MB)
#          (format: header of tools/reference/fork/forkharness.cpp)
#   flow   the fork's imsrg++ itself, configuration of the archived runs (provenance README 3.1)
#          scaled down: reference Al27, valence sd-shell, basis NAT with use_HF_reference_in_NAT,
#          NSmagnus (atan then shell-model-atan), the small test interaction of AGENTS.md, and a
#          set of NREFT operators at q = 150 MeV (scalars and tensors) plus TCM:
#            nat_emax2         emax 2  (= emax_imsrg)
#            nat_emax4_imsrg3  emax 4, emax_imsrg 3 (the emax -> emax_imsrg truncation, C5)
#          each run twice: with the plain build and with the phipfix build (which adds the raw
#          tilde-Phi' DMNREFT_Phitpp_150_2, Hermitian there). Outputs per run in
#          tests/golden/nreft/flow/<variant>/<run>/: the .snt files, stdout.log up to the
#          profiler tables (HF/NAT/flow numbers, <op>_HF and <op>_IMSRG zero bodies with 24
#          digits), occupation files, intC.mat (C_HO2NAT, armadillo binary "ARMA_MAT_BIN_FN008",
#          column major: with the bare tables it gives the NAT-basis one-body parts C^T O C),
#          cmd.txt. For the phipfix variant only the files that
#          differ: the tilde-Phi' .snt and stdout.log.
#          The run directories (with the Omega scratch files) stay in
#          $REF_BUILD/runs-fork/<variant>/<run>.
#   large  the shorthands at the production emax 14 (HF/NAT basis size of the archived runs),
#          13 MB, NOT committed: $GEN_OUT/nreft/bare_emax14_shorthands.txt with GEN_OUT =
#          IMSRG_GENERATED_GOLDEN_DIR, else $IMSRG_WORK/golden-generated (the frontier presets'
#          IMSRG_GENERATED_GOLDEN_DIR); test opslib_nreft skips it when absent.
# Environment: OMP_NUM_THREADS (default 8). Frontier: run on the compute node
# (`imsrg_compute tools/reference/gen_golden_nreft.sh`).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
. "${HERE}/ref_env.sh"
if [[ "${LMOD_SYSTEM_NAME:-}" == frontier ]]; then
  # shellcheck disable=SC1091
  . "${HERE}/../lib/frontier_env.sh"
  imsrg_frontier_modules cpu
  module load gsl/2.8 boost/1.88.0 >/dev/null 2>&1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OUT="${IMSRG_GOLDEN_NREFT_OUT:-${REPO}/tests/golden/nreft}"
F2="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
F3="${DATA}/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"

version_of() { cat "${REF_BUILD}/$1"; }

gen_bare() {
  local fh="${REF_BUILD}/fork-79e9330/build-harness/forkharness"
  local v; v="fork-79e9330:$(version_of FORK_79E9330_SRC_SHA256)"
  mkdir -p "${OUT}"
  "${fh}" nreft-bare 2 16 all "$v" > "${OUT}/bare_emax2.txt" 2> /dev/null
  "${fh}" nreft-bare 4 16 shorthands "$v" > "${OUT}/bare_emax4_shorthands.txt" 2> /dev/null
  mkdir -p "${REF_BUILD}/runs-fork"
  "${fh}" nreft-radial "$v" > "${REF_BUILD}/runs-fork/radial.txt" 2> /dev/null
  ls -l "${OUT}"/bare_emax2.txt "${OUT}"/bare_emax4_shorthands.txt
}

OPS="FM0p_150,FM0n_150,FPhipp0p_150,FPhipp0n_150,FM2p_150,FPhipp2n_150,FSigmap1p_150"
OPS="${OPS},FSigmapp1n_150,FDelta1p_150,FSigmap3n_150,TCM"
COMMON=(2bme="${F2}" file2e1max=14 file2e2max=28 file2lmax=14
        3bme="${F3}" 3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18 no2b_precision=single
        hw=16 A=27 reference=Al27 valence_space=sd-shell BetaCM=0 hwBetaCM=16 basis=NAT
        use_HF_reference_in_NAT=true denominator_delta=0 IMSRG3=false method=NSmagnus smax=100
        dsmax=0.5 ds_0=0.01 omega_norm_max=0.25 threebody_threshold=0 eta_criterion=0.001
        write_omega=true valence_file_format=tokyo)
run_args() {
  case "$1" in
    nat_emax2) echo emax=2 e3max=6 ;;
    nat_emax4_imsrg3) echo emax=4 e3max=12 emax_imsrg=3 e3max_imsrg=9 ;;
  esac
}

run_flow() {   # run_flow <variant dir> <run>
  local variant="$1" run="$2"
  local exe="${REF_BUILD}/${variant}/build/imsrg++"
  local rd="${REF_BUILD}/runs-fork/${variant}/${run}" gd="${OUT}/flow/${variant}/${run}"
  local ops="${OPS}"
  [[ "${variant}" == *phipfix ]] && ops="${ops},DMNREFT_Phitpp_150_2"
  rm -rf "${rd}" "${gd}"; mkdir -p "${rd}/scratch" "${gd}"
  local -a args; args=($(run_args "${run}"))
  local -a cmd=("${exe}" "${COMMON[@]}" "${args[@]}" "Operators=${ops}"
                "scratch=${rd}/scratch" "flowfile=${rd}/flow.dat" "intfile=${rd}/int")
  echo "${cmd[*]}" > "${rd}/cmd.txt"
  echo "== ${variant}/${run}"
  ( cd "${rd}" && "${cmd[@]}" > stdout.log 2> stderr.log ) || echo "   EXIT $?"
  local f
  local -a keep=("${rd}"/*.snt "${rd}"/*occ*.dat "${rd}/intC.mat" "${rd}/cmd.txt")
  [[ "${variant}" == *phipfix ]] && keep=("${rd}"/intDMNREFT*.snt "${rd}/cmd.txt")
  for f in "${keep[@]}"; do
    [[ -f "$f" ]] && cp "$f" "${gd}/"
  done
  # The command line holds absolute paths of this machine; keep only the file names.
  sed -i -e "s|${REF_BUILD}/||g" -e "s|${DATA}/||g" "${gd}/cmd.txt"
  # stdout up to the profiler tables (counters, timings, memory: machine-dependent).
  sed '/^===================== COUNTERS/,$d' "${rd}/stdout.log" > "${gd}/stdout.log"
  { echo "fork-79e9330 variant ${variant}"
    if [[ "${variant}" == *phipfix ]]; then version_of FORK_79E9330_PHIPFIX_SRC_SHA256
    else version_of FORK_79E9330_SRC_SHA256; fi; } > "${gd}/VERSION"
}

gen_flow() {
  local v r
  for v in fork-79e9330 fork-79e9330-phipfix; do
    for r in nat_emax2 nat_emax4_imsrg3; do run_flow "$v" "$r"; done
  done
  du -sh "${OUT}/flow"
}

gen_large() {
  local fh="${REF_BUILD}/fork-79e9330/build-harness/forkharness"
  local v; v="fork-79e9330:$(version_of FORK_79E9330_SRC_SHA256)"
  local gen="${IMSRG_GENERATED_GOLDEN_DIR:-${IMSRG_WORK:?}/golden-generated}"
  mkdir -p "${gen}/nreft"
  "${fh}" nreft-bare 14 16 shorthands "$v" > "${gen}/nreft/bare_emax14_shorthands.txt" 2> /dev/null
  ls -l "${gen}/nreft/bare_emax14_shorthands.txt"
}

sel=("$@"); [[ ${#sel[@]} -eq 0 ]] && sel=(bare flow)
for s in "${sel[@]}"; do
  case "$s" in
    bare) gen_bare ;;
    flow) gen_flow ;;
    large) gen_large ;;
    *) echo "usage: $0 [bare] [flow] [large]" >&2; exit 2 ;;
  esac
done
