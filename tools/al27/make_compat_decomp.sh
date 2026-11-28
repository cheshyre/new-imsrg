#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Creates the flow directories of the compat sub-option decomposition of the
# 27Al v2 - v1 operator differences (al27-variants.md section 4.3): the v2
# Omegas of each Hamiltonian, transformed with
#   ctl-none        run.compat = "none" (control: the v2 operator behaviour)
#   ctl-fork        run.compat = "imsrg_fork_79e9330" (the whole bundle)
#   only-<opt>      "none" + run.compat_enable = ["<opt>"]
#   allbut-<opt>    "imsrg_fork_79e9330" + run.compat_disable = ["<opt>"]
# for the operator-side options below (fork_step_defaults acts on the flow
# only, fork_operator_file_names only on the file names). Each directory gets
# symlinks to the v2 flow outputs the transform reads (basis, occupations,
# Omegas, manifest) and its own resolved_config.toml in which EVERY writable
# path (output.directory, output.flow_file, checkpoint.directory) points into
# the directory itself; the v2 run directories are only read.
#
# usage: tools/al27/make_compat_decomp.sh [OUT_ROOT]
#   OUT_ROOT default $IMSRG_WORK/runs/al27/investigate/compat
# Then: BACKEND=hip tools/al27/farm_sweep.sh with one SPEC per directory,
#   SPEC = DIR:<H>_omegas.toml:native:OUT_ROOT/work136.txt
set -euo pipefail
src=$IMSRG_WORK/runs/al27/v2
out=${1:-$IMSRG_WORK/runs/al27/investigate/compat}
opts=(tokyo_writer_fork_79e9330 tensor_no_fork_diagonal_bug
  tensor_ninej_threshold_1e8 tensor_fork_ph_missing_rank_phase
  tensor_one_body_norm_imsrgpp)
mkdir -p "$out"
# The 136 items of the earlier decomposition farm (job 6016893): all 34
# operator columns at q = 0, 150, 450, 900 MeV.
cp "$src/N2LOsat-compatop/farm-native/work.txt" "$out/work136.txt"

make_dir() {  # H VARIANT COMPAT ENABLE DISABLE
  local H=$1 v=$2 compat=$3 enable=$4 disable=$5
  local d=$out/$H-$v
  mkdir -p "$d"
  local f
  for f in "${H}_basis.bin" "${H}_occ.dat" "${H}_occ_no.dat" \
    "${H}_omega_000.opb" "${H}_omega_001.opb" "${H}_omegas.toml"; do
    ln -sfn "$src/$H/$f" "$d/$f"
  done
  # The provenance command line (of the v2 flow) is cleared: it is a record
  # of that run, not an input.
  sed -e "s|^compat = .*|compat = \"$compat\"\ncompat_enable = [$enable]\ncompat_disable = [$disable]|" \
    -e "s|^directory = \"$src/$H/checkpoint\"|directory = \"$d/checkpoint\"|" \
    -e "s|^flow_file = \"$src/$H/flow.txt\"|flow_file = \"$d/flow.txt\"|" \
    -e "s|^directory = \"$src/$H\"|directory = \"$d\"|" \
    -e "s|^command_line = .*|command_line = \"\"|" \
    "$src/$H/resolved_config.toml" > "$d/resolved_config.toml"
  if grep -q "runs/al27/v[123]" "$d/resolved_config.toml"; then
    echo "make_compat_decomp: $d/resolved_config.toml still names a v1/v2/v3 path" >&2
    exit 1
  fi
  echo "$d"
}

for H in N2LOsat DN2LOGO; do
  make_dir "$H" ctl-none none "" ""
  make_dir "$H" ctl-fork imsrg_fork_79e9330 "" ""
  for o in "${opts[@]}"; do
    make_dir "$H" "only-$o" none "\"$o\"" ""
    make_dir "$H" "allbut-$o" imsrg_fork_79e9330 "" "\"$o\""
  done
done
