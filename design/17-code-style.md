# 17 — Code style: the Google C++ Style Guide

**Decision D38** *(user, 2026-11-15)*: the code follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) (which targets C++20).
clang-format applies the formatting rules automatically, and cpplint lints the code, but the
linter is **not followed blindly** (§4). Naming is enforced with clang-tidy, because cpplint does not
check names.

**Status:** done. The existing code was converted in run 2 (2026-11-15, branch
`feature/style-google`, §6); `tools/ci.sh` enforces clang-format and cpplint, `tools/lint.sh` the
naming. The structural items of §2.D are done in part (§6); `AGENTS.md` states the conventions.

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
   element-by-element comparison with imsrg++. The naming check accepts any name with a capital
   for these kinds; everything else follows the guide.
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
