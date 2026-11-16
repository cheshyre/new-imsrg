#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# mpi_scaling.sh -- MPI strong-scaling measurement of the IMSRG(2) commutator
# (M12b phase 1; design/benchmarks/mpi-scaling.md).
#
# Runs the release `commutator_bench` under `mpiexec -n P` with OMP_NUM_THREADS
# = T for (P, T) in (1,16), (2,8), (4,4) at emax 6 and 8 (O16 reference), RUNS
# (default 5) invocations per configuration of REPS repetitions each. Per
# invocation the bench reports, per function, the minimum over repetitions of
# the maximum over ranks; this script takes the best invocation (smallest full
# commutator) and prints a Markdown table: emax, P x T, full commutator time,
# the slowest terms, the combine time (reduce + broadcast alone), the speedup
# versus one rank at the same emax and the 1-minute load average before each
# run. A run started at a 1-minute load above LOAD_MAX (6) waits for the load
# to drop (polling every 30 s, giving up after 40 min and marking the row
# "load>6"). Needs MPI; run it when the machine is otherwise idle.
#
# Usage: tools/bench/mpi_scaling.sh [build-dir]   (default build/release-openmp)
#   environment: RUNS=5 REPS=3 EMAXES="6 8" CONFIGS="1x16 2x8 4x4" LOAD_MAX=6

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${1:-${root}/build/release-openmp}"
bench="${build}/tools/bench/commutator_bench"
runs="${RUNS:-5}"
reps="${REPS:-3}"
emaxes="${EMAXES:-6 8}"
configs="${CONFIGS:-1x16 2x8 4x4}"
load_max="${LOAD_MAX:-6}"
poll_s=30
give_up_s=2400

[[ -x "${bench}" ]] || { echo "missing ${bench}: build target commutator_bench" >&2; exit 1; }

# Launcher flags of the build (IMSRG_MPIEXEC_PREFLAGS), default for Open MPI.
flags="--bind-to none"
if [[ -f "${build}/CMakeCache.txt" ]]; then
  cached="$(sed -n 's/^IMSRG_MPIEXEC_PREFLAGS:[A-Z]*=//p' "${build}/CMakeCache.txt")"
  [[ -n "${cached}" ]] && flags="${cached}"
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/mpi_scaling.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

# One-minute load average as a number.
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
above() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a > b)}'; }

echo "# mpi_scaling: ${flags}, runs ${runs}, reps ${reps}, $(date '+%Y-%m-%d %H:%M')"
echo "# $(sysctl -n machdep.cpu.brand_string), bench $(git -C "${root}" rev-parse --short HEAD)"
echo
echo "| emax | P x T | full commutator ms | slowest terms (ms) | combine ms | speedup | load before runs |"
echo "|---|---|---|---|---|---|---|"

for emax in ${emaxes}; do
  for cfg in ${configs}; do
    p="${cfg%x*}"
    t="${cfg#*x}"
    best_full=""
    loads=""
    marked=""
    for ((r = 1; r <= runs; ++r)); do
      waited=0
      load="$(load1)"
      while above "${load}" "${load_max}"; do
        if ((waited >= give_up_s)); then
          marked="load>${load_max}"
          break
        fi
        sleep "${poll_s}"
        waited=$((waited + poll_s))
        load="$(load1)"
      done
      loads="${loads} ${load}"
      out="${work}/run.txt"
      OMP_NUM_THREADS="${t}" OMP_PROC_BIND=false \
        mpiexec -n "${p}" ${flags} "${bench}" --emax "${emax}" --reps "${reps}" \
        --reference O16 >"${out}" 2>/dev/null
      # Table rows: name followed by 2 (or 3 with several ranks) numbers; keep
      # "name min_ms" (the second number).
      awk '/^term /{on=1; next} on {n=NF; while (n > 0 && $n ~ /^[0-9.]+$/) n--;
           name=$1; for (i = 2; i <= n; ++i) name = name " " $i;
           print name "\t" $(n + 2)}' "${out}" >"${work}/rows.tsv"
      full="$(awk -F'\t' '$1 == "commutator (full)" {print $2}' "${work}/rows.tsv")"
      if [[ -z "${best_full}" ]] || awk -v a="${full}" -v b="${best_full}" 'BEGIN {exit !(a < b)}'; then
        best_full="${full}"
        cp "${work}/rows.tsv" "${work}/best.tsv"
      fi
    done
    slow="$(awk -F'\t' '$1 != "commutator (full)" && $1 != "combine (reduce+bcast)"' \
      "${work}/best.tsv" | sort -t$'\t' -k2,2 -gr | head -3 |
      awk -F'\t' '{printf "%s%s %s", (NR > 1 ? ", " : ""), $1, $2}')"
    comb="$(awk -F'\t' '$1 == "combine (reduce+bcast)" {print $2}' "${work}/best.tsv")"
    [[ -n "${comb}" ]] || comb="-"
    if [[ "${p}" == "1" ]]; then
      echo "${best_full}" >"${work}/base_${emax}"
    fi
    speed="-"
    if [[ -f "${work}/base_${emax}" ]]; then
      speed="$(awk -v b="$(cat "${work}/base_${emax}")" -v f="${best_full}" 'BEGIN {printf "%.2f", b / f}')"
    fi
    echo "| ${emax} | ${p} x ${t} | ${best_full} | ${slow} | ${comb} | ${speed} | ${loads# } ${marked} |"
  done
done
