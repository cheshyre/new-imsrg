# 10 — Architecture

## 1. Repository layout

```
CMakeLists.txt            top level: options, deps, add_subdirectory(src apps python tests)
CMakePresets.json         dev-serial, dev-openmp (default on macOS), release-openmp, asan, cuda, hip
cmake/                    helper modules (warnings, sanitizers, deps lookup, layering check)
extern/                   git submodules (pinned release tags): kokkos, googletest, fmt, CLI11,
                          tomlplusplus, nanobind   [kokkos-kernels: optional, later]
src/imsrg/<module>/       library code; headers included as "imsrg/<module>/x.hpp"
src/imsrg/<module>/kernels/   .cpp files that contain Kokkos kernels (only these include Kokkos)
apps/imsrg/               the `imsrg` executable (CLI11, thin wrapper around imsrg::driver)
python/src/               nanobind bindings (module imsrg._core)
python/imsrg/             pure-Python package layered on _core
tests/unit/<module>/      GoogleTest unit tests, one executable per module
tests/integration/        pytest end-to-end tests driving `imsrg` and comparing to golden data
tests/python/             pytest tests of the bindings
tests/data/               small committed input files (synthetic, < 1 MB each)
tests/golden/             small committed reference outputs produced by imsrg-devel
tests/smoke/              toolchain smoke tests (Kokkos backend, MPI launcher)
tests/layering/           fixtures for the layering check's self-test
tools/                    scripts: bootstrap_deps.sh, ci.sh, check_layering.py, lib/ (shared
                          shell settings), reference/ (build old code, generate golden data)
docs/                     user documentation (later: Doxygen + Sphinx)
design/                   these design documents, STATUS.md, QUESTIONS.md
```

## 2. Module DAG

Each module is one CMake library target `imsrg_<module>` (alias `imsrg::<module>`). A module may
include headers **only** from the modules it lists as dependencies. `tools/check_layering.py`
enforces this. It parses `#include "imsrg/..."` (and `<imsrg/...>`) lines against the table below
and the module deps recorded by CMake, and runs as a CTest
test, so a violation fails the build's test suite.

```
Layer 0  base        strong types, errors/assertions, logging, timers/profiler, physical constants,
                     small math (factorials, log-factorials, double factorials), half->float
         exec        Kokkos facade: Runtime init/finalize, DeviceBuffer<T>, memory spaces,
                     parallel patterns, IMSRG_FUNCTION macros                     deps: base (+Kokkos)
         comm        MPI facade: Communicator, collectives, serial stub           deps: base (+MPI)
Layer 1  linalg      Matrix views, BlockSparseMatrix storage, gemm/grouped_gemm, eigh, solve
                                                                                  deps: base, exec
         angmom      Wigner 3j/6j/9j, Clebsch-Gordan, Moshinsky brackets, Talmi integrals,
                     immutable precomputed tables (device-copyable)               deps: base, exec
Layer 2  basis       Orbit, SingleParticleBasis, Nucleus/shell names, Reference (occupations),
                     Partition (core/valence/qspace), TwoBodyChannels, ThreeBodyChannels,
                     ParticleHoleChannels (Pandya maps), ModelSpace bundle        deps: base, exec, angmom
Layer 3  operator    OperatorSpec, OneBody, TwoBody, ThreeBody, Operator, arithmetic, norms,
                     normal ordering, unitary basis transformation, OperatorDump (text format)
                                                                                  deps: basis, linalg, angmom, exec, base
Layer 4  io          readers (me2j, NO2B, me3j, ...), writers (tokyo .snt, nushellx .int/.sps),
                     gzip streams                                                 deps: operator, basis, base (+zlib)
         opslib      operator library: kinetic, Tcm, HCM (Lawson), Rp2/Rn2/Rm2, ... (scalar first)
                                                                                  deps: operator, basis, angmom
         commutator  IMSRG(2) and IMSRG(3) terms, CommutatorOptions, dispatch, Workspace
                                                                                  deps: operator, linalg, angmom, exec, comm
         commutator_ref  slow, transparent reference implementations (tests only)
                                                                                  deps: operator, angmom
Layer 5  hf          Hartree-Fock (2N + 3N via NO2B / monopole), HF basis transformation
                                                                                  deps: operator, linalg, basis
         mbpt        MBPT(2)/(3) energies, MBPT(2) density matrix, natural orbitals
                                                                                  deps: operator, linalg, basis
         flow        generators, BCH, Magnus solver, ODE flow solver, step control,
                     Omega stack, checkpoints, perturbative triples driver        deps: commutator, operator
Layer 6  driver      Config schema (TOML), LegacyParameters translation, Pipeline (SR/VS, 1/2 step),
                     results summary                                              deps: everything below (+toml++)
apps     imsrg       main(): CLI11 parsing -> driver                              deps: driver (+CLI11)
python   _core       nanobind module                                              deps: driver and below
```

Rules:

* Dependencies are **transitive**: a module may include headers from its listed dependencies and
  everything they depend on. `base` is available everywhere.
* Input 3N containers (NO2B, isospin me3j storage) are data types in `operator`; `io` fills them,
  and `hf` and normal ordering consume them.
* `flow` does **not** depend on `hf`, `io` or `opslib`. The driver wires them together.
* `commutator_ref` is never linked into production targets, only into tests and the Python
  test helpers.
* A module that needs a new dependency updates this table and `tools/check_layering.py` in the
  same commit and explains why in the commit message.

## 3. Core types (sketch)

All names are tentative; implementers may refine them and record the change in the module's
docstring and here.

### 3.1 Quantum numbers and indices (module `base`, header `imsrg/base/types.hpp`)

```cpp
template <class Tag, class Rep = std::int32_t>
class StrongIndex {            // trivially copyable, constexpr, usable in device code
 public:
  constexpr explicit StrongIndex(Rep v) noexcept;
  constexpr Rep value() const noexcept;
  friend constexpr auto operator<=>(StrongIndex, StrongIndex) = default;
  constexpr StrongIndex& operator++() noexcept;   // no other arithmetic with raw ints
};
using OrbitIndex      = StrongIndex<struct OrbitTag>;
using ChannelIndex    = StrongIndex<struct ChannelTag>;     // 2B channel (J, P, Tz)
using KetIndex        = StrongIndex<struct KetTag>;         // global 2B ket |pq> (p<=q)
using LocalKetIndex   = StrongIndex<struct LocalKetTag>;    // row/col inside one channel block
using Channel3Index   = StrongIndex<struct Channel3Tag>;
using Ket3Index       = StrongIndex<struct Ket3Tag>;

struct TwoJ   { std::int32_t twice; };   // 2j or 2J; explicit construction only
struct TwoTz  { std::int32_t twice; };   // proton: -1 (imsrg++ convention), neutron: +1
enum class Parity : std::uint8_t { Even = 0, Odd = 1 };
enum class Hermiticity : std::uint8_t { Hermitian, AntiHermitian, NonHermitian };
```

### 3.2 Model space (module `basis`, as implemented in M3)

Headers in `src/imsrg/basis/`; the docstrings are authoritative. Orderings are imsrg++'s (D11),
verified record by record against the 14 golden basis dumps (`tests/golden/basis/`).

```cpp
struct Orbit { int32 n, l; TwoJ j2; TwoTz tz2; e(); parity(); };     // trivially copyable
struct BasisTruncation { int32 emax, lmax = -1, emax_unocc = -1, lmax_unocc = -1; };
class SingleParticleBasis {      // immutable; orbits in imsrg++ index order
  static SingleParticleBasis harmonic_oscillator(const BasisTruncation&, double hw_MeV);
  std::span<const Orbit> orbits() const;  std::optional<OrbitIndex> find(const Orbit&) const;
};
struct Nucleus { double A, Z; static Nucleus parse(std::string_view); };   // "O16", "mix_A4.5_Z2.5"
std::vector<OrbitOccupation> naive_filling(const Nucleus&, int32 emax, std::span<const Orbit> skip);
class Reference {                // occupations in [0,1]; holes/particles; A, Z; e_fermi
  static Reference from_nucleus(const SingleParticleBasis&, const Nucleus&);
  static Reference from_occupations(const SingleParticleBasis&, std::vector<double>, std::string);
  static Reference from_orbit_occupations(const SingleParticleBasis&, span<const OrbitOccupation>, ...);
};
ValenceSpaceDefinition define_valence_space(std::string_view spec, const SingleParticleBasis&,
                                            const Nucleus& reference);  // imsrg++ valence strings
class Partition {                // OrbitClass Core/Valence/QSpace per orbit (imsrg++ cvq)
  static Partition from_definition(const SingleParticleBasis&, const ValenceSpaceDefinition&);
  static Partition with_holes_as_core(const Partition&, const Reference&, const SingleParticleBasis&);
};
class TwoBodyChannels;           // kets p<=q (Index2), channels 6J+2(Tz+1)+P, global<->local maps
class TwoBodyKetSubsets;         // per channel: pp hh ph cc vc qc vv qv qq (+ occupation products)
class ParticleHoleChannels;      // cross-coupled channels (Tz = |tz2_p-tz2_q|/2), pair_index
class ThreeBodyChannels;         // (2J, P, 2Tz) channels, kets |pqr;Jpq> with E3max + Pauli cuts
class DeviceTables;              // device mirrors (raw-pointer POD views) of orbit/2B/subset tables

class ModelSpace {               // immutable bundle, shared via std::shared_ptr<const ModelSpace>
  static std::shared_ptr<const ModelSpace> create(SingleParticleBasis, Reference, Partition);
  static std::shared_ptr<const ModelSpace> from_names(const BasisTruncation&, double hw,
                                                      std::string_view reference, std::string_view valence);
  const SingleParticleBasis& basis() const;  const Reference& reference() const;
  const Partition& partition() const;        const TwoBodyChannels& two_body() const;
  const TwoBodyKetSubsets& two_body_subsets() const;
  const ParticleHoleChannels& particle_hole() const;
  const ThreeBodyChannels* three_body() const;   // only after with_three_body(cuts)
  std::shared_ptr<const ModelSpace> with_reference(Reference) const;   // shares basis/channels
  std::shared_ptr<const ModelSpace> with_reference_and_partition(Reference, Partition) const;
  std::shared_ptr<const ModelSpace> with_three_body(const ThreeBodyCuts&) const;
  // later: const angmom::Tables& angmom() const;
};
```

* Heavy, occupation-independent pieces (basis, two-body, ph and 3B channels) are held by
  `shared_ptr<const ...>`, so `with_reference` is cheap.
* Copy, move and assign are trivially correct because there are no raw back-pointers (imsrg++'s
  `ModelSpace` copy is unsound; see `reference/01` §9).
* `DeviceTables` is not owned by the ModelSpace (building a ModelSpace never needs Kokkos;
  QUESTIONS M3-Q7). The 6j-weighted Pandya maps belong to the commutator module (M3-Q6).

### 3.3 Operators (module `operator`)

```cpp
struct OperatorSpec {
  int particle_rank;          // 1, 2 or 3 (highest stored part)
  int tensor_rank_J = 0;      // 0 for scalars
  Parity parity = Parity::Even;
  int delta_Tz = 0;           // isospin projection change (imsrg++ calls |dTz| "rank_T")
  Hermiticity hermiticity = Hermiticity::Hermitian;
  int legs = 4;               // later: dagger (odd-particle-number) operators
};

class Operator {               // regular value type: copy = deep copy, move = cheap
 public:
  Operator(std::shared_ptr<const ModelSpace>, OperatorSpec);
  double zero_body;
  OneBody one_body;            // dense norb x norb (norb = 240 at emax=14, so tiny)
  TwoBody two_body;            // BlockSparseMatrix over allowed (ch_bra, ch_ket) pairs
  std::optional<ThreeBody> three_body;   // layout chosen by benchmark (14-imsrg3-plan.md)
  const OperatorSpec& spec() const;
  const ModelSpace& model_space() const;
};
// Free functions instead of a god-class:
Operator& axpy(double a, const Operator& x, Operator& y);   // y += a x (checks model-space compatibility)
double norm(const Operator&);                                // imsrg++ convention (D12)
Operator normal_order(const Operator& op_vacuum, const ModelSpace& ms);  // vacuum -> reference
Operator undo_normal_order(const Operator&);
Operator transform_basis(const Operator&, const UnitaryTransform& C);    // e.g. HO -> HF
```

* `TwoBody` stores **all** blocks of a scalar operator densely (both triangles) in one contiguous
  `DeviceBuffer<double>` with an offset table. That makes the whole-operator operations (axpy,
  scale, norms, generators) single flat kernels, and the per-channel products (grouped) GEMMs.
  For tensor operators we will decide between storing only `ch_bra <= ch_ket` (as imsrg++ does)
  and storing both; we will decide when tensor operators are implemented.
* Combining operators checks that their `ModelSpace` pointers are identical or that the spaces
  are structurally equal (fast hash). An operator normal-ordered with respect to another
  reference cannot be mixed in by accident.
* No `Operator` member allocates unused blocks. imsrg++'s always-allocated ThreeLeg block costs
  about 0.14 GB at emax = 12.
* **As implemented (M4):** the C++ namespace is `imsrg::op` (`operator` is a keyword; the CMake
  target stays `imsrg::operator`). Parts are reached through accessors (`zero_body()`,
  `one_body()`, `two_body()`, `three_body()`); `OperatorSpec` adds `delta_particle_number` for
  later dagger operators and `validate()` rejects what is not implemented yet. A scalar
  `TwoBody` holds one block per channel, all channels in order (empty ones 0 x 0), so block k is
  channel k. Element accessors use host mirrors (`sync_to_host/device` on GPU backends).
  Headers: `imsrg/operator/{operator_spec,one_body,two_body,three_body,operator,arithmetic,norms,
  compare,operator_dump}.hpp`.

### 3.4 Algorithms as free functions or small value types

* `commutator::commutator(X, Y, Z, opts, ws)`: `opts` is a `CommutatorOptions` value (truncation
  enum, term mask as an enum bitset, 3B cuts, thresholds). `ws` is a `Workspace` of preallocated
  scratch buffers reused across calls (no allocation in the flow loop). Individual terms
  (`terms::comm222_ph(X, Y, Z, ws)`) are public for tests and Python.
* `hf::solve(H, ms, HFOptions) -> HFResult {energy, C, spe, occupations, converged, iterations}`.
* `flow::MagnusSolver` holds the state (s, Omega stack, H(s)) plus a `SolverOptions` value. The
  generator is `flow::Generator`, an enum `GeneratorKind` plus parameters, evaluated by a free
  function with a `switch`. There is no generator class hierarchy.
* ODE integrators are function templates over a state type that satisfies a small `VectorSpace`
  concept (axpy, scale, norm), so the same Euler/RK4/DP45 code serves operators and test vectors.

## 4. Runtime, configuration and error handling

* `imsrg::Runtime` (module `exec` plus `comm`) is an RAII guard that initializes and finalizes
  Kokkos and MPI. Exactly one exists per process: in `main()`, or created on import in Python.
  The communicator is passed **explicitly** to code that needs it (through `CommutatorOptions` /
  pipeline context). Nothing calls `MPI_COMM_WORLD` implicitly.
* Logging: `imsrg::log::info(...)` and friends, using {fmt}. Only rank 0 prints by default.
  Verbosity is set by config.
* Profiling: `IMSRG_SCOPED_TIMER("name")` feeds a thread-safe registry and also emits
  `Kokkos::Profiling::pushRegion`, so NVTX/rocTX tools see the same regions. It is never called
  from constructors or destructors of data types.
* Errors: see D16. Every input-file reader validates headers, sizes and truncations, and reports
  the file name and position on failure.

## 5. Header and compile-time policy

* Public headers stay light: no Kokkos, no MPI, no toml/CLI headers. `exec/device_buffer.hpp`
  exposes `DeviceBuffer<T>` (pimpl) with explicit instantiations for `double`, `float`,
  `std::int32_t`, `std::uint16_t`. Only `src/imsrg/*/kernels/*.cpp` and `exec/*.cpp` include
  `<Kokkos_Core.hpp>` through `imsrg/exec/kokkos_backend.hpp`.
* One commutator term family per `.cpp` file, so editing one kernel recompiles one file.
* Templates are explicitly instantiated in `.cpp` files where possible.
* `ccache` is picked up automatically if present. Unity builds are optional
  (`IMSRG_UNITY_BUILD`).
* Warnings: `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` (the last one can be relaxed in
  kernels). `-Werror` in the CI preset.

## 6. Documentation policy

* Every public class and function gets a Doxygen `///` docstring: what it computes, its
  conventions (normalization, phases, units), complexity for kernels, and preconditions.
* Physics conventions are stated once in `docs/conventions.md` (written in M3/M4) and referenced
  from the docstrings.
* Python bindings carry docstrings, and nanobind stubgen produces `.pyi` stubs.
