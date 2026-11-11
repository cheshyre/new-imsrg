#!/usr/bin/env bash
# End-to-end golden data (IT-1..IT-9 of design/30-testing-and-validation.md section 4) from the OLD
# `imsrg++` executable.
#
# Usage: tools/reference/gen_golden_e2e.sh [JOB ...]
#   JOB is an IT id (IT-2) or a sub-job (IT-8/emax3); an IT id selects all of its sub-jobs.
#   Default: all jobs, in the order below (the slow emax-3 jobs last).
#   Env: IMSRG_REFERENCE_ROOT (default /Users/cheshyre/Downloads/new-imsrg-reference),
#        IMSRG_TEST_DATA_DIR (default /Users/cheshyre/Downloads/new-imsrg),
#        IT_TIMEOUT_S (default 3600: wall-time cap per run; killed runs are recorded as timed out),
#        OMP_NUM_THREADS (default 8).
# Run directories (outside the repo): ${IMSRG_REFERENCE_ROOT}/runs/e2e/<job>/
# Results (committed, small): tests/golden/e2e/<job>/{result.json,flow.dat,int*.snt,stdout.log}
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
DATA="${IMSRG_TEST_DATA_DIR:-/Users/cheshyre/Downloads/new-imsrg}"
EXE="${ROOT}/build/imsrg++"
F2="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
F3="${DATA}/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
CAP="${IT_TIMEOUT_S:-3600}"
RUNS="${ROOT}/runs/e2e"
OUT="${REPO}/tests/golden/e2e"

# common 2N arguments (C) and 3N arguments (N) of the brief
C=(2bme="${F2}" file2e1max=14 file2e2max=28 file2lmax=14 hw=16)
N=(3bme="${F3}" 3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18)

# jobs in run order (slow emax-3 jobs last); job_args prints the job-specific arguments
declare -a JOBS=(IT-1 IT-2 IT-3 IT-4 IT-5 IT-6 IT-7 IT-8/emax2 IT-9/He4_emax2 IT-9/O16_emax2
                 IT-8/emax3 IT-9/He4_emax3 IT-9/O16_emax3)
job_args() {
  case "$1" in
    IT-1) echo emax=4 reference=O16 valence_space=O16 method=MP3 ;;
    IT-2) echo emax=4 reference=O16 valence_space=O16 method=magnus ;;
    IT-3) echo emax=4 reference=O16 valence_space=sd-shell method=magnus ;;
    IT-4) echo emax=4 reference=O16 valence_space=O16 method=magnus "${N[@]}" ;;
    IT-5) echo emax=4 reference=O16 valence_space=O16 basis=NAT method=magnus "${N[@]}" ;;
    IT-6) echo emax=4 reference=O16 valence_space=O16 method=magnus Operators=Rp2 ;;
    IT-7) echo emax=4 reference=Ne20 valence_space=sd-shell method=magnus ;;
    IT-8/emax2) echo emax=2 reference=O16 valence_space=O16 method=magnus perturbative_triples=true ;;
    IT-8/emax3) echo emax=3 reference=O16 valence_space=O16 method=magnus perturbative_triples=true ;;
    IT-9/He4_emax2) echo emax=2 IMSRG3=true imsrg3_n7=true method=magnus reference=He4 valence_space=He4 ;;
    IT-9/O16_emax2) echo emax=2 IMSRG3=true imsrg3_n7=true method=magnus reference=O16 valence_space=O16 ;;
    IT-9/He4_emax3) echo emax=3 IMSRG3=true imsrg3_n7=true method=magnus reference=He4 valence_space=He4 ;;
    IT-9/O16_emax3) echo emax=3 IMSRG3=true imsrg3_n7=true method=magnus reference=O16 valence_space=O16 ;;
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
