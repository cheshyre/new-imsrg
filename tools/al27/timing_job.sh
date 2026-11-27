#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Brief A item 5: emax-10 GPU timings on ONE dedicated node (sbatch). Five
# srun steps side by side, each on its own GCD with its own cores (--exact):
#   GCD-steps 1-4: the farm (run.task = "transform", one rank) over the v1
#     Omegas of N2LOsat and DN2LOGO, native float64 and ME2JP float32, one
#     item per rank L = 0..5 at q = 150 MeV and two at q = 1050 MeV (plus TCM
#     for N2LOsat native), 7 cores each;
#   step 5: the N2LOsat v1 flow restarted on a GCD (14 cores) from a COPY of
#     its last checkpoint (stage 2, step 219): steps 220-272 time the emax-10
#     flow step and check the GPU flow against the CPU flow's flow.txt.
# Outputs go to $IMSRG_WORK/runs/al27/v1/<H>/timing-<tag>/ (never into the v1
# run directories themselves).
#
# usage: sbatch tools/al27/timing_job.sh     (from the repository root)
#SBATCH -A nph123
#SBATCH -J al27-timing
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH --gpus-per-node=8
#SBATCH -o /lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/timing-%j.out
set -uo pipefail
repo=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$repo/tools/lib/frontier_env.sh"
imsrg_frontier_modules hip
runs=$IMSRG_WORK/runs/al27/v1
bin=$repo/build/frontier-hip/apps/imsrg/imsrg
tag=$SLURM_JOB_ID
export OMP_PROC_BIND=spread OMP_PLACES=threads
items=(FM0p_150 FSigmap1p_150 FM2p_150 FSigmap3p_150 FM4p_150 FSigmap5p_150
       FM0p_1050 FSigmap5p_1050)

echo "start $(date -Is) $(hostname) $(uptime)"
rocm-smi --showuse | grep -E "^GPU" || true

farm() {  # <H> <manifest file> <format tag> [extra items]
  local h=$1 man=$2 fmt=$3
  shift 3
  local out=$runs/$h/timing-$fmt-$tag
  mkdir -p "$out"
  cp "$bin" "$out/imsrg.timing"
  printf '%s\n' "${items[@]}" "$@" > "$out/work.txt"
  OMP_NUM_THREADS=7 srun -N1 -n1 -c 7 --gpus=1 --gpu-bind=closest --exact \
    /usr/bin/time -v "$out/imsrg.timing" \
    --config "$runs/$h/resolved_config.toml" \
    --set run.task=transform --set "operators.list=[]" \
    --set "output.directory=$out" \
    --set "transform.manifest=$runs/$h/$man" \
    --set "transform.work_list=$out/work.txt" > "$out/farm.log" 2>&1
  echo "farm $h $fmt status $? $(date -Is)"
}

flow_restart() {
  local out=$runs/N2LOsat/timing-flow-gpu-$tag
  mkdir -p "$out"
  cp -r "$runs/N2LOsat/checkpoint" "$out/checkpoint"
  cp "$runs/N2LOsat/flow.txt" "$out/flow.txt"
  cp "$bin" "$out/imsrg.timing"
  OMP_NUM_THREADS=14 srun -N1 -n1 -c 14 --gpus=1 --gpu-bind=closest --exact \
    timeout 75m /usr/bin/time -v "$out/imsrg.timing" \
    --config "$runs/N2LOsat/resolved_config.toml" \
    --set "output.directory=$out" \
    --set "output.flow_file=$out/flow.txt" \
    --set "checkpoint.directory=$out/checkpoint" \
    --set checkpoint.restart=true > "$out/flow.log" 2>&1
  echo "flow restart status $? $(date -Is)"
}

farm N2LOsat N2LOsat_omegas_no.toml native TCM &
farm N2LOsat N2LOsat_me2jp_omegas_no.toml me2jp &
farm DN2LOGO DN2LOGO_omegas_no.toml native &
farm DN2LOGO DN2LOGO_me2jp_omegas_no.toml me2jp &
flow_restart &
wait
echo "end $(date -Is) $(uptime)"
