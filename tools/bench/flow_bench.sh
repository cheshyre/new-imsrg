#!/usr/bin/env bash
# flow_bench.sh - time a full IMSRG(2) Magnus flow (O16, hw 16, 2N only, the IT-2
# settings at a chosen emax) in this code and in imsrg++ on the same machine.
#
# Usage:
#   tools/bench/flow_bench.sh --emax E --threads T [--code ours|ref|both] [--reps N]
#                             [--ours-bin PATH] [--ref-bin PATH] [--out DIR]
#
# Defaults: --code both, --reps 1, --ours-bin <repo>/build/rel/apps/imsrg/imsrg
# (release-openmp), --ref-bin <reference>/build-d35/imsrg++, --out a fresh dir under
# ${TMPDIR:-/tmp} (never inside the repo).  Environment: OMP_NUM_THREADS=T,
# OMP_PROC_BIND=false.  Override of the 2N file: IMSRG_BENCH_INPUT.
#
# Columns of the table (one row per run, also appended to <out>/flow_bench.md):
#   total   wall time of the whole process, /usr/bin/time -l "real" (s).
#   flow    time of the IMSRG flow alone (s):
#             ours: last "wall_s" of the flow file (clock starts when the flow starts,
#                   after 2N read, Moshinsky, HF and MBPT; covers the Magnus loop with
#                   BCH/commutators and the per-step norms/MP2 evaluation).
#             ref : (time column of the last flow-file row) - (time column of row 0).
#                   imsrg++ prints that column as the time since its global timer
#                   started; row 0 is written after HF and MP2, so the difference
#                   is the Magnus loop (same content as ours).
#   pre     time before the flow (2N read + HF + MP2 + setup) (s):
#             ours: total - flow (upper bound: also contains output writing and exit);
#             ref : time column of flow-file row 0 (excludes process start-up).
#   steps   flow steps (last istep);  E0: final flow-file E0 (MeV);
#   RSS     peak resident set size (MiB) from /usr/bin/time -l.
# With --reps N all reps are printed, then the best rep (minimum total) per code.
# The two codes should agree on steps and E0 (to ~1e-6 MeV); a warning is printed if not.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

emax=""
threads=""
code=both
reps=1
ours_bin="$repo_root/build/rel/apps/imsrg/imsrg"
ref_bin=/Users/cheshyre/Downloads/new-imsrg-reference/build-d35/imsrg++
out=""
input=${IMSRG_BENCH_INPUT:-/Users/cheshyre/Downloads/new-imsrg/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz}

die() { echo "flow_bench: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --emax) emax=${2:?}; shift 2 ;;
    --threads) threads=${2:?}; shift 2 ;;
    --code) code=${2:?}; shift 2 ;;
    --reps) reps=${2:?}; shift 2 ;;
    --ours-bin) ours_bin=${2:?}; shift 2 ;;
    --ref-bin) ref_bin=${2:?}; shift 2 ;;
    --out) out=${2:?}; shift 2 ;;
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done

[[ -n $emax && $emax =~ ^[0-9]+$ ]] || die "--emax E (integer) is required"
[[ -n $threads && $threads =~ ^[0-9]+$ ]] || die "--threads T (integer) is required"
[[ $reps =~ ^[0-9]+$ && $reps -ge 1 ]] || die "--reps must be a positive integer"
case "$code" in ours|ref|both) ;; *) die "--code must be ours, ref or both" ;; esac
[[ -r $input ]] || die "input file not readable: $input"
[[ -x /usr/bin/time ]] || die "/usr/bin/time not found"
if [[ $code != ref && ! -x $ours_bin ]]; then
  die "our binary missing: $ours_bin (cmake --preset release-openmp -B build/rel -DIMSRG_BUILD_TESTS=OFF && cmake --build build/rel --target imsrg)"
fi
if [[ $code != ours && ! -x $ref_bin ]]; then die "imsrg++ binary missing: $ref_bin"; fi
if [[ -z $out ]]; then out=$(mktemp -d "${TMPDIR:-/tmp}/flow_bench.XXXXXX"); fi
mkdir -p "$out"
out=$(cd "$out" && pwd)
case "$out" in
  "$repo_root"|"$repo_root"/*) die "--out must not be inside the repository" ;;
  *) ;;
esac
table="$out/flow_bench.md"

common=("2bme=$input" file2e1max=14 file2e2max=28 file2lmax=14 hw=16
        "emax=$emax" reference=O16 valence_space=O16 method=magnus)

header="| code | emax | thr | rep | total [s] | flow [s] | pre [s] | steps | E0 [MeV] | peak RSS [MiB] | load avg (1/5/15) |"
if [[ ! -s $table ]]; then
  printf '%s\n|---|---|---|---|---|---|---|---|---|---|---|\n' "$header" >> "$table"
fi
echo "$header"
echo "|---|---|---|---|---|---|---|---|---|---|---|"

# run_one CODE REP: runs, prints/appends one table row, records "code total steps E0".
run_one() {
  local c=$1 rep=$2 dir load rc=0 bin
  dir="$out/$c-e$emax-t$threads-$rep"
  mkdir -p "$dir"
  load=$(uptime | sed 's/.*load averages*: *//; s/,//g; s/ /\//g')
  if [[ $c == ours ]]; then bin=$ours_bin; else bin=$ref_bin; fi
  (
    cd "$dir"
    export OMP_NUM_THREADS=$threads OMP_PROC_BIND=false
    /usr/bin/time -l "$bin" "${common[@]}" "flowfile=$dir/flow.dat" "intfile=$dir/int" \
      > stdout.log 2> stderr.log
  ) || rc=$?
  [[ $rc -eq 0 ]] || die "$c run failed (exit $rc), see $dir/stderr.log"
  [[ -s $dir/flow.dat ]] || die "$c run produced no flow file in $dir"

  local total rss
  total=$(awk '/ real /{print $1; exit}' "$dir/stderr.log")
  rss=$(awk '/maximum resident set size/{printf "%.1f", $1/1048576; exit}' "$dir/stderr.log")
  [[ -n $total && -n $rss ]] || die "cannot parse /usr/bin/time output in $dir/stderr.log"

  local flow pre steps e0
  if [[ $c == ours ]]; then
    # CSV: stage,istep,s,ds,E0,...,wall_s
    read -r steps e0 flow < <(awk -F, 'NR>1{i=$2; e=$5; w=$NF} END{printf "%d %.9f %.3f\n", i, e, w}' "$dir/flow.dat")
    pre=$(awk -v t="$total" -v f="$flow" 'BEGIN{printf "%.3f", t-f}')
  else
    # columns: istep s E0 ... N_Ops time "mem / mem"; the time is field NF-3
    read -r steps e0 flow pre < <(awk 'NR==1{t0=$(NF-3)} {i=$1; e=$3; t=$(NF-3)} END{printf "%d %.9f %.3f %.3f\n", i, e, t-t0, t0}' "$dir/flow.dat")
  fi
  local row="| $c | $emax | $threads | $rep | $total | $flow | $pre | $steps | $e0 | $rss | $load |"
  echo "$row" | tee -a "$table"
  echo "$c $total $steps $e0" >> "$out/.results"
}

: > "$out/.results"
codes=("$code")
if [[ $code == both ]]; then codes=(ours ref); fi
for ((r = 1; r <= reps; r++)); do
  for c in "${codes[@]}"; do run_one "$c" "$r"; done
done

if [[ $reps -gt 1 ]]; then
  for c in "${codes[@]}"; do
    best=$(awk -v c="$c" '$1==c{n++; if(m==""||$2<m){m=$2; l=n}} END{print l}' "$out/.results")
    echo "best $c (minimum total): rep $best" | tee -a "$table"
  done
fi

if [[ $code == both ]]; then
  read -r _ _ s1 e1 < <(awk '$1=="ours"{print; exit}' "$out/.results")
  read -r _ _ s2 e2 < <(awk '$1=="ref"{print; exit}' "$out/.results")
  if [[ $s1 != "$s2" ]] || ! awk -v a="$e1" -v b="$e2" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=1e-6)}'; then
    echo "flow_bench: WARNING: codes disagree: steps $s1 vs $s2, E0 $e1 vs $e2" >&2
  fi
fi
echo "results in $out" >&2
