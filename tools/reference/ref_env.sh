# tools/reference/ref_env.sh -- paths shared by the reference tooling. Source it; do not execute.
# design/20-build-and-tooling.md section 5.
#
# Defines (each overridable from the environment; the Mac defaults keep the old behaviour):
#   REF_ROOT   IMSRG_REFERENCE_ROOT, default /Users/cheshyre/Downloads/new-imsrg-reference.
#              Holds the SHA files (REFERENCE_SRC_SHA256[_D35]); on Frontier read-only.
#   REF_BUILD  IMSRG_REFERENCE_BUILD, else $IMSRG_WORK/reference-build when IMSRG_WORK is set,
#              else REF_ROOT. Every build output (builds, copies, runs) lives here.
#   REF_SRC    IMSRG_REFERENCE_SOURCE, else $REF_ROOT/imsrg-devel if it exists, else the Mac
#              path /Users/cheshyre/Downloads/new-imsrg/imsrg-devel. Pristine, read-only.
#   DATA       IMSRG_TEST_DATA_DIR, else $IMSRG_WORK/inputs when IMSRG_WORK is set, else
#              /Users/cheshyre/Downloads/new-imsrg. The big test inputs.
#   REF_SHA, REF_SHA_D35
#              the version files: $REF_ROOT/REFERENCE_SRC_SHA256[_D35] if present, else the one
#              in REF_BUILD (where build_imsrg_devel.sh writes them when REF_ROOT is read-only).
# IMSRG_REFERENCE_ROOT is exported as REF_ROOT, for refharness (it reads the SHA file there).
REF_ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
if [[ -n "${IMSRG_REFERENCE_BUILD:-}" ]]; then
  REF_BUILD="${IMSRG_REFERENCE_BUILD}"
elif [[ -n "${IMSRG_WORK:-}" ]]; then
  REF_BUILD="${IMSRG_WORK}/reference-build"
else
  REF_BUILD="${REF_ROOT}"
fi
if [[ -n "${IMSRG_REFERENCE_SOURCE:-}" ]]; then
  REF_SRC="${IMSRG_REFERENCE_SOURCE}"
elif [[ -d "${REF_ROOT}/imsrg-devel" ]]; then
  REF_SRC="${REF_ROOT}/imsrg-devel"
else
  REF_SRC="/Users/cheshyre/Downloads/new-imsrg/imsrg-devel"
fi
if [[ -n "${IMSRG_TEST_DATA_DIR:-}" ]]; then
  DATA="${IMSRG_TEST_DATA_DIR}"
elif [[ -n "${IMSRG_WORK:-}" ]]; then
  DATA="${IMSRG_WORK}/inputs"
else
  DATA="/Users/cheshyre/Downloads/new-imsrg"
fi
REF_SHA="${REF_ROOT}/REFERENCE_SRC_SHA256"
[[ -f "${REF_SHA}" ]] || REF_SHA="${REF_BUILD}/REFERENCE_SRC_SHA256"
REF_SHA_D35="${REF_ROOT}/REFERENCE_SRC_SHA256_D35"
[[ -f "${REF_SHA_D35}" ]] || REF_SHA_D35="${REF_BUILD}/REFERENCE_SRC_SHA256_D35"
export IMSRG_REFERENCE_ROOT="${REF_ROOT}"
