#!/usr/bin/env bash
# Golden data for M15 (tensor operators), run-3 reference preparation (owner decision D53):
# tensor operator dumps, scalar-tensor commutator terms and a Magnus step of a tensor operator
# from the D35-patched imsrg++. Written to tests/golden/tensor/ in the RAW tensor dump format
# (`# imsrg-tensor-dump raw v0`, design/15-operator-dump-format.md, "Tensor reference dumps").
#
#   tests/golden/tensor/
#     emax2_He4/{HO,HF}/<op>.tdump      E1 E2 E3 M1 Fermi GamowTeller, 0nubb: M0nu_{GT,F,T,C}, DGT
#     emax2_O16/{HO,HF}/<op>.tdump      (HO: E2 M1 GamowTeller Fermi M0nu_GT M0nu_F only)
#     emax2_Ne20/HF/<op>.tdump          open shell (0d5/2 occupation 1/3): E2 M1 GamowTeller Fermi
#     emax2_He4/inputs/{H_HF,random_A}.dump   X inputs of the commutators (OperatorDump v1)
#     emax2_*/commutator/Z_<term>__<X>__<Y>.tdump   CommutatorScalarTensor terms (X = H | A)
#     emax1_He4/commutator/...          the same for seeded random tensor Y (all 2B terms nonzero)
#     emax{1,2}_He4/bch/Y_<Y>_transformed.tdump   BCH_Transform(Y, Omega)
#     result.json                       patch stamp, commands, norms, checks
#
# Usage: tools/reference/gen_golden_tensor.sh
# Environment: REFHARNESS (default $IMSRG_REFERENCE_ROOT/harness-build-d35/refharness), which must
#   contain the dump-tensor / commutator-tensor / bch-tensor subcommands (rh_tensor.cpp).
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ref_env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROOT="${REF_BUILD}"
RH="${REFHARNESS:-${ROOT}/harness-build-d35/refharness}"
export IMSRG_REFERENCE_SHA_FILE="${IMSRG_REFERENCE_SHA_FILE:-${REF_SHA_D35}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
G="${REPO}/tests/golden"
OUT="${G}/tensor"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/runs"
# the script owns tests/golden/tensor: start from an empty directory (no stale files)
rm -rf "${OUT}"
mkdir -p "${OUT}"
INT=(2bme="${DATA}/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
     file2e1max=14 file2e2max=28 file2lmax=14)
CMDLOG="${TMP}/commands.txt"
: > "${CMDLOG}"

# run_rh <run-name> <subcommand> <args...>: runs the harness into ${TMP}/runs/<run-name>, logs the
# command (without the interaction-file arguments), keeps the harness's norms.json.
run_rh() {
  local name="$1" sub="$2"; shift 2
  local d="${TMP}/runs/${name}"
  mkdir -p "${d}"
  "${RH}" "${sub}" "$@" "${INT[@]}" out="${d}" > "${TMP}/${name}.log" 2>&1 \
    || { echo "FAILED: ${sub} $*" >&2; tail -n 20 "${TMP}/${name}.log" >&2; exit 1; }
  echo "${name}|refharness ${sub} $* <interaction args> out=<dir>" >> "${CMDLOG}"
}
# put_existing: as put, but silently skips files the run did not write (identically zero terms)
put_existing() {
  local name="$1" dest="$2"; shift 2
  mkdir -p "${OUT}/${dest}"
  for f in "$@"; do [[ -f "${TMP}/runs/${name}/${f}" ]] && cp "${TMP}/runs/${name}/${f}" "${OUT}/${dest}/${f}"; done
  return 0
}
# put <run-name> <dest-dir> <file-glob...>: copy the listed files of a run into the golden tree
put() {
  local name="$1" dest="$2"; shift 2
  mkdir -p "${OUT}/${dest}"
  for f in "$@"; do cp "${TMP}/runs/${name}/${f}" "${OUT}/${dest}/${f}"; done
}

OPS_1B=E1,E2,E3,M1,Fermi,GamowTeller
OPS_0NBB=M0nu_GT_7.72_none,M0nu_F_7.72_none,M0nu_T_7.72_none,M0nu_C_500_3,DGT
tdumps() { local IFS=,; for o in $1; do echo "${o}.tdump"; done; }

# --- 1. operator dumps ---------------------------------------------------------------------
for nuc in He4 O16; do
  common=(emax=2 reference=${nuc} valence_space=${nuc} hw=16)
  if [[ ${nuc} == He4 ]]; then ho_ops="${OPS_1B},M0nu_GT_7.72_none,M0nu_F_7.72_none,DGT"; else ho_ops=E2,M1,Fermi,GamowTeller,M0nu_GT_7.72_none,M0nu_F_7.72_none; fi
  run_rh "ops_${nuc}_HO" dump-tensor "${common[@]}" basis=HO ops="${ho_ops}"
  run_rh "ops_${nuc}_HF" dump-tensor "${common[@]}" basis=HF ops="${OPS_1B},${OPS_0NBB}" write_h=true
  put "ops_${nuc}_HO" "emax2_${nuc}/HO" $(tdumps "${ho_ops}")
  put "ops_${nuc}_HF" "emax2_${nuc}/HF" $(tdumps "${OPS_1B},${OPS_0NBB}")
done
run_rh ops_Ne20_HF dump-tensor emax=2 reference=Ne20 valence_space=Ne20 hw=16 basis=HF ops=E2,M1,Fermi,GamowTeller
put ops_Ne20_HF emax2_Ne20/HF E2.tdump M1.tdump Fermi.tdump GamowTeller.tdump
# X inputs: He4 here; O16 reuses hf/emax2_O16_2N/HNO.dump and commutator/emax2_O16/random_A.dump
# (checked below to agree with what the harness computes)
put ops_He4_HF emax2_He4/inputs H_HF.dump random_A.dump
# random_A (seed 7) is reference independent, but its values depend on the C++ standard library
# (libc++ on the Mac, libstdc++ on Frontier), so the X input of the X=A goldens is stored here

# --- 2. scalar-tensor commutator terms -------------------------------------------------------
ALLTERMS=comm111st,comm121st,comm122st,comm221st,comm222_pp_hhst,comm222_pp_hh_221st,comm222_phst,all
# emax 2 physical operators. For a one-body Y only 111, 121 (1B) and 122 (2B) are nonzero and
# `all` is their sum (checked), so the big comm122st dump is not stored at emax 2 (norms.json has it).
for nuc in He4 O16; do
  common=(emax=2 reference=${nuc} valence_space=${nuc} hw=16)
  run_rh "comm_${nuc}_H" commutator-tensor "${common[@]}" X=H Y=E2,GamowTeller terms="${ALLTERMS}"
  files=()
  for y in E2 GamowTeller; do
    for t in comm111st comm121st all; do files+=("Z_${t}__H__${y}.tdump"); done
  done
  # identically zero terms are not written (see norms.json)
  put_existing "comm_${nuc}_H" "emax2_${nuc}/commutator" "${files[@]}"
done
run_rh comm_He4_A commutator-tensor emax=2 reference=He4 valence_space=He4 hw=16 X=A Y=E2 terms=comm111st,comm121st,comm122st,comm222_pp_hh_221st,comm222_phst,all
put_existing comm_He4_A emax2_He4/commutator Z_all__A__E2.tdump Z_comm111st__A__E2.tdump Z_comm121st__A__E2.tdump

# emax 1, random tensor Y (nonzero 2B parts: every term is exercised). One run per (X, Y) pair
# so that the four hermiticity combinations (H,H) (H,A) (A,H) (A,A) are all covered.
combos=("H rand_2_0_0_H_3" "H rand_1_0_1_A_5" "A rand_1_1_0_H_4" "A rand_1_0_1_A_5")
i=0
for c in "${combos[@]}"; do
  set -- ${c}; x="$1"; y="$2"
  run_rh "comm1_${i}" commutator-tensor emax=1 reference=He4 valence_space=He4 hw=16 X="${x}" Y="${y}" terms="${ALLTERMS}"
  mkdir -p "${OUT}/emax1_He4/commutator"
  cp "${TMP}/runs/comm1_${i}"/Z_*.tdump "${OUT}/emax1_He4/commutator/"
  i=$((i + 1))
done
# emax 1 inputs (X = H in the He4 HF basis, X = A): needed by readers of the emax-1 files
# and the Y inputs of the emax-1 commutator and BCH goldens (E2 in the HF basis; the random tensors
# are written exactly as commutator-tensor builds them: make_tensor_op returns rand_* before the HF
# step), added in the Al27 night (2026-11-26) for the M15 acceptance tests
Y1_OPS=E2,rand_2_0_0_H_3,rand_1_0_1_A_5,rand_1_1_0_H_4
run_rh inputs1 dump-tensor emax=1 reference=He4 valence_space=He4 hw=16 basis=HF ops="${Y1_OPS}" write_h=true
put inputs1 emax1_He4/inputs H_HF.dump random_A.dump $(tdumps "${Y1_OPS}")

# --- 3. Magnus step: BCH_Transform(Y, Omega), Omega = s * random_A (seed 7), |Omega| = 0.3 ---
run_rh bch2 bch-tensor emax=2 reference=He4 valence_space=He4 hw=16 Y=E2 omega_norm=0.3
put bch2 emax2_He4/bch Y_E2_transformed.tdump
run_rh bch1 bch-tensor emax=1 reference=He4 valence_space=He4 hw=16 Y=E2,rand_2_0_0_H_3,rand_1_1_0_H_4 omega_norm=0.3
put bch1 emax1_He4/bch Y_E2_transformed.tdump Y_rand_2_0_0_H_3_transformed.tdump Y_rand_1_1_0_H_4_transformed.tdump

# --- 4. checks and result.json ---------------------------------------------------------------
python3 -I "${HERE}/tensor_golden_check.py" "${OUT}" "${TMP}" "${G}" "${CMDLOG}" "$(cat "${IMSRG_REFERENCE_SHA_FILE}")"
du -sh "${OUT}"
