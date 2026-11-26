#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Builds the imsrg binary with the frontier-cpu-release preset on the compute
# node of the Slurm job $IMSRG_SLURM_JOBNAME (default imsrg-c), using
# ${IMSRG_COMPUTE_CPUS:-16} cores (a production run may share the node).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
export IMSRG_SLURM_JOBNAME=${IMSRG_SLURM_JOBNAME:-imsrg-c}
export IMSRG_COMPUTE_CPUS=${IMSRG_COMPUTE_CPUS:-16}
# shellcheck source=/dev/null
source "$root/tools/lib/frontier_env.sh"
imsrg_frontier_modules cpu >/dev/null 2>&1
cd "$root"
imsrg_compute bash -c "cmake --preset frontier-cpu-release >/dev/null && \
  cmake --build --preset frontier-cpu-release --target imsrg \
  -j $IMSRG_COMPUTE_CPUS 2>&1 | tail -3"
