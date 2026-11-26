#!/usr/bin/env bash
# End-to-end golden data (IT-1..IT-9 of design/30-testing-and-validation.md section 4) from the OLD
# `imsrg++` executable.
#
# Usage: tools/reference/gen_golden_e2e.sh [JOB ...]
#   JOB is an IT id (IT-2) or a sub-job (IT-8/emax3); an IT id selects all of its sub-jobs.
#   Default: all jobs, in the order below (the slow emax-3 jobs last).
#   Env: IMSRG_REFERENCE_ROOT (default: see ref_env.sh; build outputs go to IMSRG_REFERENCE_BUILD),
#        IMSRG_TEST_DATA_DIR (default: see ref_env.sh),
#        IT_TIMEOUT_S (default 3600: wall-time cap per run; killed runs are recorded as timed out),
#        OMP_NUM_THREADS (default 8),
#        IMSRG_REFERENCE_EXE (default $ROOT/build-d35/imsrg++), IMSRG_REFERENCE_RUNS (default
#        $ROOT/runs-d35/e2e), IMSRG_GOLDEN_E2E_OUT (default tests/golden/e2e of this checkout) and
#        IMSRG_REFERENCE_SHA_FILE (default $ROOT/REFERENCE_SRC_SHA256_D35, read by parse_imsrg_log.py):
#        the defaults are the D35 patched build (see tests/golden/README.md); the pristine build is
#        selected by setting these variables to its paths.
# Run directories (outside the repo): ${IMSRG_REFERENCE_RUNS}/<job>/ (default ${IMSRG_REFERENCE_ROOT}/runs-d35/e2e/)
# Results (committed, small): tests/golden/e2e/<job>/{result.json,flow.dat,int*.snt} (stdout.log is copied too but gitignored, `*.log`)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
. "${HERE}/ref_env.sh"
ROOT="${REF_BUILD}"
EXE="${IMSRG_REFERENCE_EXE:-${ROOT}/build-d35/imsrg++}"   # D35: the patched build (default)
F2="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
F3="${DATA}/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
CAP="${IT_TIMEOUT_S:-3600}"
RUNS="${IMSRG_REFERENCE_RUNS:-${ROOT}/runs-d35/e2e}"   # D35: the patched runs (keeps the pristine runs)
export IMSRG_REFERENCE_SHA_FILE="${IMSRG_REFERENCE_SHA_FILE:-${REF_SHA_D35}}"
OUT="${IMSRG_GOLDEN_E2E_OUT:-${REPO}/tests/golden/e2e}"

# common 2N arguments (C) and 3N arguments (N) of the brief
C=(2bme="${F2}" file2e1max=14 file2e2max=28 file2lmax=14 hw=16)
N=(3bme="${F3}" 3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18)

# jobs in run order (slow emax-3 jobs last); job_args prints the job-specific arguments
# IT-4/smax10, IT-5/smax10: the full IT-4/IT-5 flows diverge (cE-only 3N file, see
# tests/golden/e2e/README.md), so the comparable golden is the flow stopped at s = 10, well before
# the runaway (s ~ 25). IT-8/He4_emax2: O16 at emax 2 is unstable in IMSRG(2), He4 converges.
# IT-14 (F2b, D42): the ODE flow methods `flow_RK4` (fixed-step RK4) and `flow` (odeint dopri5),
# with Rp2 flowing alongside H. flow.dat of `flow` has one line per right-hand-side evaluation.
# IT-15 (M8 two-stage ODE flow, D42): He4 emax 2 with the p-shell valence space (O16 emax 2 diverges), `flow_RK4` for both
# decoupling stages (core atan, then the valence-space generator); the golden holds the flow file of
# both stages, the final E0 and the valence-space .snt. The adaptive `flow` variant restarts s at 0 in
# stage 2 (M8-Q40): compared by its final quantities only.
# IT-15 *_smax10_eta (D47): smax=10 eta_criterion=0.014, so that stage 1 CONVERGES at s = 7.399 and stage 2 ends at
# s1 + smax = 17.399 (the doubled smax of pristine imsrg++ would run to 20): the only job that tells the two Magnus rules apart.
# IT-15/He4_emax2_pshell_hg (D48): two-stage hunter-gatherer, stage 2 starts from the H rebuilt with the gathered Omega.
# IT-15 *_smax10 (D47): smax=10, so that stage 1 stops at smax and stage 2 integrates [s1, s1 + smax] for every
# method (the D35 patch; pristine imsrg++ leaves the flow_RK4 stage 2 empty, doubles smax for Magnus only).
declare -a JOBS=(IT-1 IT-2 IT-3 IT-4 IT-4/smax10 IT-5 IT-5/smax10 IT-6 IT-7 IT-8/emax2
                 IT-8/He4_emax2 IT-9/He4_emax2 IT-9/O16_emax2
                 IT-8/emax3 IT-9/He4_emax3 IT-9/O16_emax3
                 IT-14/He4_emax2_rk4 IT-14/He4_emax2_flow IT-14/O16_emax3_rk4
                 IT-14/O16_emax3_flow IT-14/O16_emax4_rk4 IT-14/O16_emax4_flow
                 IT-15/He4_emax2_pshell_rk4 IT-15/He4_emax2_pshell_flow
                 IT-15/He4_emax2_pshell_rk4_smax10 IT-15/He4_emax2_pshell_flow_smax10
                 IT-15/He4_emax2_pshell_magnus_smax10
                 IT-15/He4_emax2_pshell_magnus_smax10_eta IT-15/He4_emax2_pshell_hg)
job_args() {
  case "$1" in
    IT-1) echo emax=4 reference=O16 valence_space=O16 method=MP3 ;;
    IT-2) echo emax=4 reference=O16 valence_space=O16 method=magnus ;;
    IT-3) echo emax=4 reference=O16 valence_space=sd-shell method=magnus ;;
    IT-4) echo emax=4 reference=O16 valence_space=O16 method=magnus "${N[@]}" ;;
    IT-4/smax10) echo emax=4 reference=O16 valence_space=O16 method=magnus smax=10 "${N[@]}" ;;
    IT-5) echo emax=4 reference=O16 valence_space=O16 basis=NAT method=magnus "${N[@]}" ;;
    IT-5/smax10) echo emax=4 reference=O16 valence_space=O16 basis=NAT method=magnus smax=10 "${N[@]}" ;;
    IT-6) echo emax=4 reference=O16 valence_space=O16 method=magnus Operators=Rp2 ;;
    IT-7) echo emax=4 reference=Ne20 valence_space=sd-shell method=magnus ;;
    IT-8/emax2) echo emax=2 reference=O16 valence_space=O16 method=magnus perturbative_triples=true ;;
    IT-8/He4_emax2) echo emax=2 reference=He4 valence_space=He4 method=magnus perturbative_triples=true ;;
    IT-8/emax3) echo emax=3 reference=O16 valence_space=O16 method=magnus perturbative_triples=true ;;
    IT-9/He4_emax2) echo emax=2 IMSRG3=true imsrg3_n7=true method=magnus reference=He4 valence_space=He4 Operators=Rp2 ;;
    IT-9/O16_emax2) echo emax=2 IMSRG3=true imsrg3_n7=true method=magnus reference=O16 valence_space=O16 Operators=Rp2 ;;
    IT-9/He4_emax3) echo emax=3 IMSRG3=true imsrg3_n7=true method=magnus reference=He4 valence_space=He4 ;;
    IT-9/O16_emax3) echo emax=3 IMSRG3=true imsrg3_n7=true method=magnus reference=O16 valence_space=O16 ;;
    IT-14/He4_emax2_rk4) echo emax=2 reference=He4 valence_space=He4 method=flow_RK4 Operators=Rp2 ;;
    IT-14/He4_emax2_flow) echo emax=2 reference=He4 valence_space=He4 method=flow Operators=Rp2 ;;
    IT-14/O16_emax3_rk4) echo emax=3 reference=O16 valence_space=O16 method=flow_RK4 Operators=Rp2 ;;
    IT-14/O16_emax3_flow) echo emax=3 reference=O16 valence_space=O16 method=flow Operators=Rp2 ;;
    IT-14/O16_emax4_rk4) echo emax=4 reference=O16 valence_space=O16 method=flow_RK4 Operators=Rp2 ;;
    IT-14/O16_emax4_flow) echo emax=4 reference=O16 valence_space=O16 method=flow Operators=Rp2 ;;
    IT-15/He4_emax2_pshell_rk4) echo emax=2 reference=He4 valence_space=p-shell method=flow_RK4 ;;
    IT-15/He4_emax2_pshell_flow) echo emax=2 reference=He4 valence_space=p-shell method=flow ;;
    IT-15/He4_emax2_pshell_rk4_smax10) echo emax=2 reference=He4 valence_space=p-shell method=flow_RK4 smax=10 ;;
    IT-15/He4_emax2_pshell_flow_smax10) echo emax=2 reference=He4 valence_space=p-shell method=flow smax=10 ;;
    IT-15/He4_emax2_pshell_magnus_smax10) echo emax=2 reference=He4 valence_space=p-shell method=magnus smax=10 ;;
    IT-15/He4_emax2_pshell_magnus_smax10_eta) echo emax=2 reference=He4 valence_space=p-shell method=magnus smax=10 eta_criterion=0.014 ;;
    IT-15/He4_emax2_pshell_hg) echo emax=2 reference=He4 valence_space=p-shell method=magnus hunter_gatherer=true omega_norm_max=0.1 ;;
  esac
}

run_job() {
  local job="$1" rd="${RUNS}/$1" gd="${OUT}/$1"
  local -a args
  # shellcheck disable=SC2207
  args=($(job_args "${job}"))   # arguments contain no spaces by construction
  rm -rf "${rd}"; mkdir -p "${rd}" "${gd}"
  local -a cmd=("${EXE}" "${C[@]}" "${args[@]}" "flowfile=${rd}/flow.dat" "intfile=${rd}/int")
  echo "${cmd[*]}" > "${rd}/cmd.txt"
  echo "== ${job}: ${cmd[*]}"
  local rc t0 t1
  t0=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
  # perl alarm = portable wall-time cap (macOS has no coreutils timeout); alarm survives exec
  ( cd "${rd}" && perl -e 'alarm shift; exec @ARGV' "${CAP}" "${cmd[@]}" > stdout.log 2> stderr.log )
  rc=$?
  t1=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
  echo "${rc}" > "${rd}/exit_code.txt"
  perl -e 'printf "%.3f\n", $ARGV[1] - $ARGV[0]' "${t0}" "${t1}" > "${rd}/wall_time_s.txt"
  echo "   exit ${rc}, $(cat "${rd}/wall_time_s.txt") s"
  python3 -I "${HERE}/parse_imsrg_log.py" --rundir "${rd}" --job "${job}" \
      --omp-threads "${OMP_NUM_THREADS}" --out "${gd}/result.json" || echo "   PARSE FAILED"
  [[ -f "${rd}/flow.dat" ]] && cp "${rd}/flow.dat" "${gd}/flow.dat"
  local f
  for f in "${rd}"/*.snt "${rd}"/*.op; do
    [[ -f "$f" && $(wc -c < "$f") -lt 1000000 ]] && cp "$f" "${gd}/"
  done
  [[ $(wc -c < "${rd}/stdout.log") -lt 200000 ]] && cp "${rd}/stdout.log" "${gd}/stdout.log"
  return 0
}

sel=("$@"); [[ ${#sel[@]} -eq 0 ]] && sel=("${JOBS[@]}")
for job in "${JOBS[@]}"; do
  for s in "${sel[@]}"; do
    if [[ "${job}" == "${s}" || "${job}" == "${s}/"* ]]; then run_job "${job}"; break; fi
  done
done
