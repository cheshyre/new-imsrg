#!/usr/bin/env bash
# tools/lint.sh -- the clang-tidy naming check (design/17-code-style.md section 4).
#
# Usage: tools/lint.sh [file...]
#   Runs clang-tidy with only readability-identifier-naming (configured in .clang-tidy) on the
#   given translation units, or on every translation unit of the compile database that lies in
#   src/ apps/ tests/ python/ tools/bench/; headers are checked through the units that include
#   them. Prints each finding once and exits nonzero if there is any. It is not part of
#   tools/ci.sh, which must stay under 5 minutes (D42); run it after renaming or adding names.
#
# Environment:
#   IMSRG_BUILD_DIR   build tree with compile_commands.json (default <repo>/build/dev-openmp).
#                     Code that this build does not compile (communicator_serial.cc without MPI,
#                     tools/bench without IMSRG_BUILD_BENCH, CUDA/HIP branches) is not checked.
#   CLANG_TIDY        clang-tidy binary (default: clang-tidy on PATH, else Homebrew LLVM's)
#   IMSRG_LINT_JOBS   parallel jobs (default: number of CPUs)
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/imsrg_env.sh
. "$script_dir/lib/imsrg_env.sh"
repo_root=$(imsrg_repo_root "$script_dir") || { echo "lint: not inside a git checkout"; exit 2; }
build_dir=${IMSRG_BUILD_DIR:-$repo_root/build/dev-openmp}
db=$build_dir/compile_commands.json
if [ ! -f "$db" ]; then
  echo "lint: $db not found (configure a build first: cmake --preset dev-openmp)"; exit 2
fi
tidy=${CLANG_TIDY:-}
if [ -z "$tidy" ]; then
  tidy=$(command -v clang-tidy || true)
  if [ -z "$tidy" ] && [ -x /opt/homebrew/opt/llvm/bin/clang-tidy ]; then
    tidy=/opt/homebrew/opt/llvm/bin/clang-tidy
  fi
fi
if [ -z "$tidy" ]; then echo "lint: clang-tidy not found (brew install llvm)"; exit 2; fi
# The physics-symbol patterns of .clang-tidy must match tools/physics_symbols.txt (D50).
"${IMSRG_VENV}/bin/python" "$script_dir/gen_physics_symbols.py" --check --repo-root "$repo_root" \
  || python3 "$script_dir/gen_physics_symbols.py" --check --repo-root "$repo_root" || exit 2
jobs=${IMSRG_LINT_JOBS:-$(imsrg_ncpu)}

if [ $# -gt 0 ]; then
  files=$(printf '%s\n' "$@")
else
  # Every translation unit of the compile database inside our source directories.
  files=$("${IMSRG_VENV}/bin/python" - "$db" "$repo_root" <<'EOF'
import json, os, sys
db, root = sys.argv[1], os.path.realpath(sys.argv[2])
dirs = ("src/", "apps/", "tests/", "python/", "tools/bench/")
seen = set()
for e in json.load(open(db)):
    f = os.path.realpath(os.path.join(e["directory"], e["file"]))
    rel = os.path.relpath(f, root)
    if rel.startswith(dirs) and not rel.startswith("tests/golden/") and rel not in seen:
        seen.add(rel)
        print(rel)
EOF
)
fi
n_files=$(echo "$files" | grep -c .)
echo "lint: $("$tidy" --version | grep -i 'version' | head -n 1 | sed 's/^ *//'), $n_files units"
out=$(cd "$repo_root" && echo "$files" | tr '\n' '\0' | xargs -0 -n 4 -P "$jobs" "$tidy" \
  -p "$build_dir" --quiet --checks='-*,readability-identifier-naming' \
  --extra-arg=-Wno-error 2>/dev/null)
findings=$(echo "$out" | grep -E 'warning: .*\[readability-identifier-naming\]' | sort -u)
if [ -z "$findings" ]; then
  echo "lint: no naming findings"
  exit 0
fi
echo "$findings"
echo "lint: $(echo "$findings" | grep -c .) naming findings"
exit 1
