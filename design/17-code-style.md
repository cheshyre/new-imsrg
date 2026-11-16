# 17 — Code style: the Google C++ Style Guide

**Decision D38** *(user, 2026-11-15)*: the code follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) (which targets C++20).
clang-format applies the formatting rules automatically, and cpplint lints the code, but the
linter is **not followed blindly** (§4). Naming is enforced with clang-tidy, because cpplint does not
check names.

**Status:** done. The existing code was converted in run 2 (2026-11-15, branch
`feature/style-google`, §6); `tools/ci.sh` enforces clang-format and cpplint, `tools/lint.sh` the
naming. The structural items of §2.D were done in part in run 2 and by the structural pass MS2 in
run 3 (2026-11-16, branch `feature/style-structural`, §6); `AGENTS.md` states the conventions.

## 1. Rules in one paragraph

Google style everywhere in `src/`, `apps/`, `tests/`, `python/` and `tools/bench/`, with the
exceptions of §3. Formatting: clang-format (Google base, **80 columns**). Files: `snake_case.h` /
`snake_case.cc`, `#define` guards `IMSRG_<PATH>_H_`, license boilerplate first. Names: types and
**functions in PascalCase** (accessors, mutators and STL-analog members such as `begin`, `size`,
`data` may stay snake_case), variables snake_case, class members with a trailing `_`, struct members
without, **`kName` constants** (static storage duration) and **`kName` enumerators**, namespaces
snake_case, macros `IMSRG_UPPER_CASE`. Code: no using-directives, no non-trivially-destructible
static objects, `int64_t` instead of `long`, C++ casts, include what you use, explicit copy/move
declarations for classes, structs only for passive data, inline functions of about 10 lines at most,
floating-point literals with a radix point (`1.0e-10`).

## 2. What changes (measured on `develop` 61adb16, 2026-11-15)

Scope: 304 C++ files (113 headers, 186 `.cpp`, 5 `.cc`), 41,077 lines; `tools/reference/` (verbatim
imsrg++ copies linked against the old library), `tests/golden/` and `extern/` excluded. Counts marked
~ come from a regex scan and are approximate; the clang-format and cpplint numbers come from runs
(clang-format 23.1.5, cpplint 2.0.2 in the shared venv).

### A. Mechanical (clang-format, scripts)

| Change | Today | Scale |
|--------|-------|-------|
| 80 columns (now 100) | 5,547 lines > 80 in 281 files | clang-format reflows 281 files, ~11,100 changed lines; cpplint `whitespace/line_length` 5,513 |
| File extensions `.h` / `.cc` | `.hpp` / `.cpp` (and `.cc` in `python/src`) | 113 + 186 renames, 782 `#include` lines, CMake lists, layering check |
| `#define` header guards | `#pragma once` in all 113 headers (cpplint 2.0 accepts it, the guide does not) | 113 headers |
| License boilerplate | none (cpplint `legal/copyright` 304) | `// SPDX-License-Identifier: GPL-3.0-only` + `// Copyright 2026 The imsrg authors` in 304 files, plus an `AUTHORS` file |
| Floating-point literals | 325 like `1e-10`, 12 like `1.` | `1.0e-10`, `1.0` (values identical) |
| Include order: related, C system, C++ std, other libraries, project | C system headers grouped with third-party | ~14 lines |

### B. Naming (clang-tidy `readability-identifier-naming --fix`, then enforced)

| Change | Scale |
|--------|-------|
| Functions and methods to PascalCase (`hf::Solve` → `hf::Solve`, `mp2_energy` → `Mp2Energy`); accessors/mutators (`norm()`, `set_norm()`) and STL analogs may stay | ~490 function names declared in headers, plus all call sites |
| Enumerators to `kName` (`Mixing::Linear` → `Mixing::kLinear`) | 193 enumerators in 41 enums |
| Static-storage constants to `kName` (`inv_sqrt2` → `kInvSqrt2`, `moshinsky_max_energy` → `kMoshinskyMaxEnergy`) | ~27 at namespace scope, ~17 static members/locals |
| Internal namespaces: unique names containing `internal` instead of `detail` (optional) | 8 |

Python names do not change: the bindings name every function explicitly (snake_case, PEP 8).

### C. cpplint findings to fix

| Category | Count | Fix |
|----------|-------|-----|
| `build/namespaces` | 86 | replace using-directives (mostly tests) with using-declarations or qualified names |
| `build/include_what_you_use` | 109 | add the missing standard headers |
| `runtime/int` | 41 | `int64_t` etc. instead of `long`, `long long`, `short` |
| `build/include_subdir` | 41 | include local headers by their path from the source root |
| `readability/casting` | 6 | `static_cast` |
| `readability/braces`, `runtime/string`, `build/explicit_make_pair` | 2, 1, 1 | as reported |

### D. Structural rules (manual review)

| Rule | Today |
|------|-------|
| Static-storage objects must be trivially destructible | ~14 (e.g. `static const std::vector<std::string>` tables in `driver/config/resolve.cc`, `static const DdTables` in `angmom/moshinsky.cc`) → `constexpr std::array<std::string_view, N>` or `static const auto& x = *new T(...)` |
| Classes declare their copy/move operations explicitly | 38 of 48 classes in headers do not |
| Structs only for passive data | 36 of 213 structs have member functions (review), 1 has private members |
| Inline function bodies in headers ≤ ~10 lines | 24 longer (templates excepted) → `.cc` |
| Functions ≤ ~40 lines (guideline) | 61 longer, 11 longer than 100 (many are binding and schema tables) |
| Concepts sparingly, internal, `requires` syntax | `ConfigEnum` in a public header; 5 uses of `template <ConfigEnum E>` |
| Namespace aliases in headers only in internal namespaces | 3 (e.g. `namespace nb = nanobind;`) |
| Avoid forward declarations | ~10 in headers (pimpl declarations stay) |
| Function comments descriptive ("Returns …") | 14 imperative `///` comments |
| Unsigned types only for bit patterns | 125 `unsigned`/`uintN_t` (review); `size_t` (775) stays for container sizes |

### E. Already compliant

C++20; type, variable and namespace names; trailing `_` on class members; `IMSRG_` prefix on exported
macros (the guide allows exported macros with the project prefix); 2-space indentation, `char* p`,
unindented namespaces and access specifiers (clang-format Google base); project headers in quotes
(Q21); non-member binary operators (friends); no RTTI, no user-defined literals, no inline namespaces;
1 TODO (a quote of imsrg++) and 1 non-ASCII line.

## 3. Documented exceptions to the guide

1. **C++ exceptions** are used (D16): typed exceptions for user and input errors, mapped to Python
   exceptions by nanobind; never in device code. *(user, 2026-11-15)*
2. **`<filesystem>`** is allowed in `apps/`, `driver`, `io` and tests, never in kernels or numerics;
   cpplint's `build/c++17` check is filtered. *(user, 2026-11-15)*
3. **Kokkos device lambdas:** `IMSRG_LAMBDA` expands to `[=]`, as Kokkos requires (30 uses); the
   device-code rules of `12` §5 already restrict captures to values. Host lambdas follow the guide
   (explicit captures when a lambda may escape; `[&]` only for short-lived lambdas such as the 88
   immediately invoked initializers).
4. **`///` doc comments** stay for API documentation (Doxygen, M13); they are `//` comments, used
   consistently.
5. `tools/reference/` and `extern/` are not converted.
6. **Physics notation in variable names** *(run 2, MS-Q1)*: variables, parameters and data members
   keep the capitals of the physics symbols they name (`X`, `Y`, `Z` for commutator operands, `H`,
   `A`/`Z` for nucleon numbers, `J`, `Jp`, `Tz`, `hw_MeV`, `twoJ`), as in imsrg++ and the
   literature; lower-casing them would merge distinct quantities (`j` and `J`) and break the
   element-by-element comparison with imsrg++. **Rule (D50, run 3):** the naming check accepts a
   variable, parameter or data-member name only if its `_`-separated parts are lower-case words
   (`[a-z0-9]+`) or symbols from the curated list `tools/physics_symbols.txt`, with at least one
   symbol; private and protected members end in `_` as usual, a parameter may end in `_`, and
   local constants may also be `kName`. So `twoJ`, `ket_J_offsets`, `delta_Tz`, `hw_MeV`,
   `A_core_`, `Z_fast` pass; camelCase such as `numOrbits` or `bigA` does not (a descriptive
   suffix is a separate part: `A_big`). The list (159 symbols, collected with clang-tidy from the
   names the code used on 2026-11-16; 158 since MS2 renamed `Jbig` to `J_big`) holds single letters (`A`..`Z`), numbered or labelled
   forms (`J12`, `T45`, `Hh`, `Zg`), angular-momentum and isospin quantities (`Jab`, `Jpmax`,
   `Tz`, `twoJ`, `twoTz`), index/count/double forms (`iJ`, `nA`, `tJ`, `dZ`), nucleon numbers and
   reference labels (`Aref`, `Zcore`), `Om`/`Omega` and `MeV`. A new symbol is added to the file,
   `tools/gen_physics_symbols.py --write` regenerates the `IgnoredRegexp` lines of `.clang-tidy`
   (Variable, LocalVariable, Parameter, PublicMember, PrivateMember, ProtectedMember,
   LocalConstant), and `tools/lint.sh` fails if the two are out of sync. Physics names were not
   renamed; the 18 non-physics names found (all locals) were: see §6.
7. **Accessors** *(run 2, MS-Q2)*: the guide lets accessors and mutators be named like variables.
   Here an accessor is a const method that returns a property or a part of the object without
   side effects, possibly selected by an index or key (`num_orbits()`, `orbit(i)`, `channel(c)`,
   `spec()`, `zero_body()`, `norm()`, `get_normalized(...)`); mutators are `set_*`; STL analogs
   (`begin`, `size`, `data`, `find`, `value`, `has_value`, `error`, and `test`/`set`/`reset` of
   the bitset-like `TermMask`) keep the STL spelling. Static member functions are not accessors,
   except `ThreeBodyChannels::channel_index` and the internal `DeviceBufferAccess::view`, which
   share the names of accessors. `.clang-tidy` holds the allowlist (199 names, collected from the 349 flagged method
   names during the conversion; 224 after the M10 three-body pieces); a new accessor name is added
   there.

## 4. Tooling (added by the conversion)

`.clang-format`: as today, with `ColumnLimit: 80` and the guide's include order:

```yaml
IncludeCategories:
  - Regex: '^<(mpi|omp|zlib|Kokkos_[A-Za-z_]+|nanobind/.*|gtest/.*|gmock/.*|fmt/.*|CLI/.*|toml\+\+/.*)(\.h|\.hpp)?>$'
    Priority: 3          # other libraries (before the C-system rule, which would catch <mpi.h>)
  - Regex: '^<[a-z_0-9/]+\.h>$'
    Priority: 1          # C system headers
  - Regex: '^<[a-z_0-9]+>$'
    Priority: 2          # C++ standard library
  - Regex: '^("|<imsrg/)'
    Priority: 4          # project headers: every quoted include, and <imsrg/...>
  - Regex: '.*'
    Priority: 3
```

As built, every quoted include is a project header (the sketch had only `"imsrg/`): test- and
binding-local headers (`"tests/unit/hf/hf_test_support.h"`, `"python/src/bindings.h"`) would
otherwise sort before `<gtest/gtest.h>` and violate cpplint's include order.

`CPPLINT.cfg` at the repository root (cpplint is a lint, not an oracle; every filter has a reason):

```
set noparent
linelength=80
root=src
extensions=h,cc
headers=h
includeorder=standardcfirst
filter=-whitespace/indent_namespace,-whitespace/braces,-whitespace/parens,-build/c++17
```

* `includeorder=standardcfirst`: without it cpplint treats every `<x.h>` (nanobind, gtest, MPI,
  zlib) as a C system header; it removed all 113 `build/include_order` findings on 2026-11-15.
* `whitespace/indent_namespace`, `whitespace/braces`, `whitespace/parens`: their findings (65, 6, 2)
  contradict clang-format's own output (continuation lines inside namespaces, `std::errc{}`, macro
  parameter lists); clang-format is the formatting authority.
* `build/c++17`: the `<filesystem>` exception (§3).
* Header guards outside `src/` need no extra `CPPLINT.cfg`: cpplint falls back to the path from the
  repository root (`TESTS_UNIT_HF_HF_TEST_SUPPORT_H_`, `PYTHON_SRC_BINDINGS_H_`; MS-Q3), and
  `.clang-tidy` exempts `*_H_` from the `IMSRG_` macro prefix. Justified exceptions use
  `// NOLINT(<category>): <reason>`; after the conversion there are three (§6).

`.clang-tidy`: add `readability-identifier-naming` with the Google naming of §1 (classes, structs,
enums, functions, methods, type aliases and template parameters CamelCase; variables, parameters and
members lower_case; private members suffix `_`; constexpr/static constants and enumerators prefix `k`
plus CamelCase; namespaces lower_case; macros UPPER_CASE with prefix `IMSRG_`). STL-analog members and
the accessor/mutator allowlist collected during the conversion go into the ignore regexps. Verify the
option names against the installed clang-tidy. As built (clang-tidy 23.1.5): `MethodIgnoredRegexp`
is the allowlist of §3 item 7; variables, parameters and members accept names with capitals (§3
item 6); local constants accept `kName` as well as snake_case (the guide makes `kName` optional for
automatic variables); STL protocol aliases (`value_type`, ...) are exempt; header guards are exempt
from the macro prefix. `tools/lint.sh` runs the check.

`tools/ci.sh`: a cpplint step (fails on findings) next to the clang-format check; the clang-tidy naming
check runs on the changed files (or in a separate `tools/lint.sh`) so that `ci.sh` stays under 5 minutes
(D42).

## 5. Conversion plan (first task of run 2)

One branch, `feature/style-google`, cut from `develop` before any other run-2 work; no other run-2
branch starts until it is merged (D38). One commit per step; after every step the build and
`tools/ci.sh dev-openmp` pass and the test results stay **bitwise identical** (record the IT
`results.json` numbers before step 1 and compare after each step).

1. Tooling: `.clang-format` (80 columns, include order), `CPPLINT.cfg`, `.clang-tidy` naming, the
   cpplint step in `ci.sh` (advisory until step 8), `AUTHORS`.
2. Renames `.hpp` → `.h`, `.cpp` → `.cc` with `git mv`; `#include` lines, CMake, the layering check
   and its fixtures, docs that cite paths.
3. `#define` guards (script).
4. License boilerplate (script).
5. clang-format at 80 columns (a pure reformat commit, listed in `.git-blame-ignore-revs`).
6. Naming with `clang-tidy --fix` from `compile_commands.json` (functions, methods, constants,
   enumerators), then check that every Python binding name is unchanged; update examples and docs.
7. Floating-point literals (script).
8. cpplint fixes (§2 C); the cpplint step becomes blocking.
9. Structural items (§2 D) as far as time allows; leftovers go to `STATUS.md`.
10. `AGENTS.md` code conventions, `10-architecture.md` §1, `20-build-and-tooling.md` and this
    document's status.

Acceptance: `ci.sh` passes with the blocking cpplint step and zero findings (apart from NOLINTs with
reasons); clang-format clean at 80 columns; clang-tidy naming clean; Python tests unchanged;
bitwise-identical test results.

## 6. As converted (run 2, 2026-11-15)

One commit per step on `feature/style-google` (cut from `develop` c01d766); after every step
`tools/ci.sh dev-openmp` passed with the same test counts (851 unit, 34 integration, 3 MPI, 23
Python) and the integration-test outputs (results.json, flow CSVs without `wall_s`, `.snt`,
resolved configs without date and git hash, and the printed IT numbers) were identical to the
pre-conversion baseline; the Python export list never changed.

| Step | Commit | Result |
|------|--------|--------|
| 1 tooling | `de6fbf6` | `CPPLINT.cfg`, `.clang-tidy` naming, advisory cpplint step, `tools/lint.sh`, `AUTHORS`; the `.clang-format` change moved to step 5 so that every commit passes the format check |
| 2 renames | `00f16d8` | 299 files `.hpp/.cpp` -> `.h/.cc`, references by basename (code, CMake, layering fixtures, living docs) |
| 3 guards | `46e6f63` | 113 headers |
| 4 license | `aa8d342` | 304 files |
| 5 reformat | `1c59fec` (+ `728cfaa` blame-ignore) | 80 columns, include order; long comment paragraphs re-wrapped with hanging indentation before clang-format, 14 formula/pseudo-code comment blocks laid out by hand |
| 6 naming | `4ee5688` | 1,205 declarations (612 functions, 182 methods, 164 enumerators, 38 constants, ...); naming check clean |
| 7 literals | `6185891` | 340 literals |
| 8 cpplint | `ca9812e`, `7990487` | 6,139 findings after step 2 (5,512 line length, 304 copyright, 109 IWYU, 86 using-directives, 42 runtime/int, 41 include_subdir, ...) -> 0; step blocking |
| 9 structural | `c9cee57`, `c97f04c`, `da651d3`, `e0eec54`, `9f704e8` | see below |

NOLINTs (all with reasons in the code): the layering fixture's relative include
(`build/include_subdir`, it is what the fixture tests) and two `build/include_order` lines in
`tests/smoke/kokkos_smoke_test.cc` (cpplint counts `<Kokkos_Core.hpp>` as a project header because
of its extension).

Structural items (§2.D) done: static-storage objects trivially destructible (11 statics in src and
tests now leaked `new` objects); explicit copy/move declarations for all 37 classes in headers
(trait-checked: the supported set is unchanged); `ConfigEnum` internal (`driver::detail`) with
requires-clauses; no namespace aliases at namespace scope in headers; the one struct with a private
section; one redundant forward declaration. Not done (left as guidelines for new code, MS-Q5):
inline functions longer than ~10 lines in headers (the remaining ones are device-code lookups,
templates or header-only test helpers; moving the former out of line could cost performance),
the review of the 36 structs that have member functions, functions longer than ~40 lines, the
review of unsigned types, and descriptive wording of the remaining imperative function comments.

**Owner, 2026-11-16 (MS-Q5, D50): the structural pass is scheduled with priority, first in run 3,
alone and time-boxed** (like MS): structs that hold invariants become classes (private data,
accessors; passive structs stay), unsigned types become signed except sizes and bit patterns,
functions over about 40 lines are split, long inline functions in headers move to `.cc` files
(device-code lookups and templates excepted, with a measured reason where performance argues for
inline), imperative function comments become descriptive; one commit per item, bitwise-identical
results, `tools/ci.sh` after each. Header guards outside `src/` stay as they are (MS-Q3). The
license line is `GPL-3.0-only` since the interview (D46).

### Physics-symbol list (run 3, D50, MS-Q1)

The ignore pattern that accepted any name containing a capital was replaced by the curated list of
exception 6 (§3). Removing the exemption flagged 215 distinct variable, parameter and member
names (not counting `kName` constants); 197 are physics notation (159 distinct whole parts after
splitting at `_`, in `tools/physics_symbols.txt`); 18 were not and were renamed in one commit (locals only, no behavior change): `bigA`/`bigB`/`bigC`
-> `A_big`/`B_big`/`C_big`, `chiY` -> `chi_Y`, `etaME` -> `eta_me`, `hEta`/`hZ` -> `h_eta`/`h_Z`
(the names `factorized.h` documents), `mOm` -> `neg_Om`, `MEs` -> `me_sum`, `Acopy` -> `A_copy`,
`Zfast`/`Zok`/`Zbad`/`Zswap`/`Zwrong`/`Zwrong2` -> `Z_fast`/..., `Cbad` -> `C_bad`, and the lambda
local `busy_` -> `busy`. Borderline, kept as symbols: the one-letter-prefixed forms `iJ`, `nA`,
`tJ`, `dZ`, ... (index, number, twice, change of a symbol) and the labelled matrices `Hh`, `Fk`,
`Ct`, `EB`, `Lam`, `PL`, `Tde`, which are physics notation of the formulas they implement. No
other camelCase names exist in variables, parameters or members (public API methods and types are
checked by the other naming kinds). MS2 renamed `Jbig` to `J_big` and dropped it from the list.
The method allowlist of `.clang-tidy` gained 12 accessors that run-3 code added without a clang-tidy
run (`flowing_operator`, `num_omegas`, `num_gathers`, `num_operators`, `is_magnus`,
`magnus_solver`, `three_body_context`, `chi1`, `chia`, `row`, `mutex`, `ws`).

### Structural pass (run 3, MS2, D50)

Branch `feature/style-structural`, cut from `develop` 0cc1efb (2026-11-16, 16:10-19:00 JST); the
lead (Opus) and seven Sonnet workers (one per module group, at most two at a time, each in its own
worktree, one commit per item and module, cherry-picked in item order; a second round of splits
followed the documentation commit). After every item `tools/ci.sh dev-openmp` passed
with the baseline counts (1073 unit, 67 integration, 3 MPI, 150 Python + 1 skipped) and the outputs
were identical to the baseline taken before item 1: the `ctest -L integration -V` logs with every
printed number (wall-time columns and `(N ms)` / `N s` timings masked), every file the driver tests
write (`results.json`, flow CSVs without `wall_s`, `.snt`, resolved configs without date, git hash
and temp paths), the `ctest -N` list and the Python API surface (every `_core` name with its
docstring and signature, the `__all__` lists). Two baseline effects needed care (MS2-Q3, MS2-Q4):
the perturbative-triples dE is not bitwise reproducible between two runs of one binary (compared to
13 digits), and 53 gtest names embed heap pointers (masked). `commutator_bench --checksum` (FNV hash
of [X, Y] at emax 6) is identical before and after; `tools/lint.sh` is clean.

| Item | Commits | Result |
|------|---------|--------|
| 1 structs with invariants -> classes | `9b89054` python, `42cc14f` driver, `fa20e80` allowlist, `7b7eb2c` angmom + bench | 12 classes: `PyModelSpace`, `PyOperator`, `PyMagnus`, `PyOde` (virtual functions, bases, a destructor that purges caches); `LineWriter`, `KeyCollector`, `SchemaLists` (stateful helpers); `SixJView`, `NineJLSjjView`, `MoshinskyView`, `SixJ3View` (lookups that rely on the pointer matching the size fields; still trivially copyable, `IMSRG_INLINE_FUNCTION` constructors and accessors `data()`, `emax()`, `lmax()`, `e2max()`); the bench's `TriplesBody`. Kept as structs: passive aggregates (configs, options, results, keys, `Orbit`, `OperatorSpec`, ...), pimpl `Impl` structs, kernel functors, and the device views and kernel data bundles listed in MS2-Q1 |
| 2 unsigned -> signed | `91a8db6` python, `eccde3c` | 14 scoped enums `std::uint8_t` -> `std::int8_t`, the NO2B monopole orbit indices -> `std::int16_t`, two test counters; what stays unsigned and why: MS2-Q2 |
| 3 functions over ~40 lines split | round 1: `39b337a` python, `f80c0a0` driver, `732f45d` operator, `2a86576` io, `eb78d09` hf, `0300529` mbpt, `e3e1b32` opslib, `4000ac8` flow, `c8fc09a` basis, `4ccb6f2` angmom, `f07a7fe` commutator_ref, `623d308` + `2336af6` + `0a373e2` commutator; round 2: `9799513` driver, `8976c7c` io, `a1b9c67` python, `e996ad4` flow, `c16d92d` angmom, `d5656e7` commutator, `6eec4bf` hf, `8ff32ef` flow | about 85 functions split into named helpers (anonymous namespaces; code moved verbatim, accumulators passed by reference, no expression split across a helper boundary, no reordering of floating-point operations). Function bodies (clang-tidy `readability-function-size`): over 40 lines 147 -> 125 (splits create helpers of 41-60 lines), over 60 lines 84 -> 30, over 100 lines 35 -> 11; host code outside kernels and tests: over 40 lines 97 -> 76, over 60 lines 62 -> 8, over 100 lines 25 -> 1 (the `VisitSchema` table; `driver::Run` 217 -> under 60) |
| 4 long inline header functions -> `.cc` | `92a9e3b` | `linalg::detail::SyevdLower` and 13 test helpers (`ReadSnt`, `CompareSnt`, the mini-JSON parsers, `Execute`, `ReadFlow`, `TomlText`, `FreshDir`, `LoadGolden`, `CompareToGolden`, `BuildNat`) moved into 5 new `.cc` files; header functions over 10 lines 35 -> 22, all templates (`VisitSchema`, `GuardRead`, the `PairEngine` adders, `DeviceBuffer` members) or device-code lookups (`n7_views.h`, `device_views.h`, `three_body_recoupling.h`, the angmom views, the constexpr `half.h` converters). No function stayed inline for timing reasons |
| 5 descriptive function comments | `68afe0a` driver, `b597472` | 2 imperative function comments and 5 field comments reworded; the 14 imperative `///` comments counted in run 2 had been reworded since |
| 6 leftovers | `a355e54`, `6122b0c` | `Jbig` -> `J_big`, `Jbig` dropped from `tools/physics_symbols.txt` (158 symbols); `tools/lint.sh` clean |

Not split (guidelines for later): the kernels (23 functions in `*/kernels/*.cc`, e.g. `TeamTiles`
285 lines, `Comm222Phss` 233; splitting them is left to the N7 tiling work, where register pressure
and inlining can be measured), test bodies and test tables (26), data tables (`VisitSchema`,
`AddP1Rows`, `AddPlannedRows`), 68 host functions of 41-60 lines (many of them new helpers) and
five of 61-80 lines where a split would mostly add parameters or need header changes
(`apps/imsrg` `main` 80, `MagnusSolver::Solve` 79, the `PairThirdSpace` and `DeviceTables`
constructors 77 and 75, `Moshinsky` 66).

Timing (release-openmp with `IMSRG_BUILD_BENCH`, separate build trees for 0cc1efb and the final
branch head 8ff32ef, `OMP_NUM_THREADS=6`, `OMP_PROC_BIND=false`, runs interleaved, best of 5; machine
shared with other worktrees, loads 4.6 / 8.0 / 8.7 at the start, 5.2 / 6.6 / 7.9 at the end). Both
differences are within the run-to-run spread (commutator 4.63-4.75 ms, IT-9 24.15-24.60 s; the same
measurement after round 1 gave -0.3 % and +0.1 %); the commutator checksums are identical:

| Benchmark | before (0cc1efb) | after (MS2) | change |
|-----------|-----------------:|------------:|-------:|
| IMSRG(2) commutator, emax 6 (`commutator_bench --emax 6 --reps 10 --only-full`, mean ms per call) | 4.643 | 4.626 | -0.4 % |
| IT-9 He4 emax 2 (`driver_imsrg3_n7 --gtest_filter=Imsrg3N7.He4Emax2`, s) | 24.28 | 24.15 | -0.5 % |
