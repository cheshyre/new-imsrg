#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Runs the fork reference build (imsrg++ 79e9330, $IMSRG_WORK/reference-build)
# with the N2LOsat inputs of the 27Al archive, basis=NAT with
# use_HF_reference_in_NAT, method=HF, at a reduced emax/e3max, on the node of
# the Slurm job $IMSRG_SLURM_JOBNAME (default imsrg-c) with 16 threads. Used to
# localize differences of the HF/NAT step (lead D, design/benchmarks/
# al27-production.md).
#
# usage: tools/al27/fork_hf.sh <emax> <e3max> <tag> [extra key=value ...]
# output: $IMSRG_WORK/runs/al27/nat-debug/fork_<tag>/stdout.txt
set -uo pipefail
e=$1; e3=$2; tag=$3; shift 3
M=/lustre/orion/nph123/proj-shared/nuccor_interactions/me_matthias/MEs/NNLOsat
d=$IMSRG_WORK/runs/al27/nat-debug/fork_$tag
mkdir -p "$d/scratch"
cd "$d" || exit 1
job=$(squeue -h -u "$USER" -n "${IMSRG_SLURM_JOBNAME:-imsrg-c}" -t R -o %i |
  head -1)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
srun --jobid="$job" --overlap -N1 -n1 -c "$OMP_NUM_THREADS" --cpu-bind=none \
  "$IMSRG_WORK/reference-build/fork-79e9330/build/imsrg++" \
  scratch="$d/scratch" hw=16 A=27 reference=Al27 valence_space=sd-shell \
  BetaCM=0.0 hwBetaCM=16 \
  2bme=$M/TwBME-HO_NN-only_N2LO_sat_bare_hw16.0_emax16_e2max32.me2j.gz \
  file2e1max=16 file2e2max=32 file2lmax=16 \
  3bme=$M/NO2B_ThBME_N2LOsat_3NFJmax15_IS_hw16.0_ms16_32_24.stream.bin \
  file3e1max=16 file3e2max=32 file3e3max=24 3bme_type=no2b \
  no2b_precision=single basis=NAT use_HF_reference_in_NAT=true \
  emax="$e" e3max="$e3" denominator_delta=0.0 method=HF intfile="$d/int" \
  "$@" > "$d/stdout.txt" 2>&1
echo "done $?" >> "$d/stdout.txt"
