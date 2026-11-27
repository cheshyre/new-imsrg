#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Self-resubmitting one-node development allocation (D77, 27Al weekend).
#
# usage (login node):  tools/al27/alloc_chain.sh NAME [--after JOBID]
#   submits a 2-h one-node job named NAME that only sleeps; agents use it
#   through `srun --overlap` (IMSRG_SLURM_JOBNAME=NAME). About 10 minutes
#   before its end the job submits its own successor under the same name with
#   --dependency=afterany:<this job>, so the chain survives orchestrator and
#   lead stops. --after JOBID makes the first job wait for JOBID (to take over
#   from an existing allocation without a gap).
# stop a chain:  touch $IMSRG_WORK/alloc/NAME.stop; scancel -n NAME -u $USER
#   (the stop file prevents further successors; remove it to restart a chain).
# Logs: $IMSRG_WORK/alloc/NAME-<jobid>.out
#SBATCH -A nph123
#SBATCH -N 1
#SBATCH -t 02:00:00
set -euo pipefail

work=${IMSRG_WORK:?IMSRG_WORK is not set}
dir=$work/alloc
mkdir -p "$dir"
self=$(realpath "${BASH_SOURCE[0]}")

submit() {  # submit NAME [dependency job]
  local name=$1 dep=${2:-} args=()
  [[ -n $dep ]] && args+=(--dependency="afterany:$dep")
  sbatch --parsable -A nph123 -N 1 -t 02:00:00 -J "$name" \
    -o "$dir/$name-%j.out" "${args[@]}" "$self" "$name"
}

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  # Login node: start a chain.
  name=${1:?usage: alloc_chain.sh NAME [--after JOBID]}
  dep=""
  if [[ ${2:-} == --after ]]; then dep=${3:?--after needs a job id}; fi
  rm -f "$dir/$name.stop"
  # The script runs from the hub-independent path at submit time; keep a
  # private copy so later checkouts or edits do not change running chains.
  cp "$self" "$dir/alloc_chain.sh"
  self=$dir/alloc_chain.sh
  submit "$name" "$dep"
  exit 0
fi

# Inside the job (BASH_SOURCE is Slurm's spool copy; resubmit the private copy).
self=$dir/alloc_chain.sh
name=${1:-$SLURM_JOB_NAME}
echo "chain $name job $SLURM_JOB_ID on $(hostname) start $(date -Is)"
lead=600   # seconds before the end at which the successor is submitted
remaining() {  # seconds left in this job
  local l
  l=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
  # formats: MM:SS, HH:MM:SS, D-HH:MM:SS
  local d=0 h=0 m=0 s=0
  if [[ $l == *-* ]]; then d=${l%%-*}; l=${l#*-}; fi
  IFS=: read -r -a p <<< "$l"
  case ${#p[@]} in
    3) h=${p[0]}; m=${p[1]}; s=${p[2]} ;;
    2) m=${p[0]}; s=${p[1]} ;;
    1) s=${p[0]} ;;
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
submitted=0
while :; do
  if [[ -e $dir/$name.stop ]]; then
    echo "stop file found $(date -Is); no successor"; break
  fi
  r=$(remaining || echo 999999)
  if (( submitted == 0 && r <= lead )); then
    # Avoid duplicates: only submit if no other pending job of this name.
    if [[ -z $(squeue -h -u "$USER" -n "$name" -t PD -o %i) ]]; then
      succ=$(submit "$name" "$SLURM_JOB_ID")
      echo "submitted successor $succ $(date -Is)"
    else
      echo "pending successor already exists $(date -Is)"
    fi
    submitted=1
  fi
  sleep 60
done
# Stopped: sleep out the remaining time so running work is not killed.
while [[ -e $dir/$name.stop ]] && (( $(remaining || echo 0) > 60 )); do
  sleep 60
done
