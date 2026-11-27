#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# The v1 flows (2026-11-26) wrote their Omega manifests before the keys
# normal_ordering_emax / normal_ordering_occupations existed (design/50 C10);
# run.task = "transform" with [imsrg.space] needs them. This writes a copy
# <manifest stem>_no.toml next to the manifest with the two keys added. The
# emax-14 normal-ordering occupations equal the IMSRG-space ones of
# `occupations` (Al27 fills only orbitals up to the sd shell, so the orbitals
# above imsrg.space.emax = 10 are empty, and OccupationsInBasis sets missing
# orbitals to 0); the archive comparison of the farm output validates it.
#
# usage: tools/al27/manifest_add_no.sh MANIFEST [NO_EMAX=14]
set -euo pipefail
in=$1
emax=${2:-14}
out=${in%.toml}_no.toml
occ=$(sed -n 's/^occupations = "\(.*\)"/\1/p' "$in")
[[ -n $occ ]] || { echo "no occupations key in $in" >&2; exit 2; }
if grep -q '^normal_ordering_occupations' "$in"; then
  echo "$in already has normal_ordering_occupations" >&2; exit 2
fi
{
  sed '/^files = \[/,$d' "$in"
  echo "# added by tools/al27/manifest_add_no.sh ($(date -I))"
  echo "normal_ordering_emax = $emax"
  echo "normal_ordering_occupations = \"$occ\""
  sed -n '/^files = \[/,$p' "$in"
} > "$out"
echo "$out"
