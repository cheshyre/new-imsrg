# 18 - Python API (nanobind bindings and the `imsrg` package)

> **Precedence:** where this document disagrees with `10-architecture.md`, `16-config-and-cli.md` or
> `20-build-and-tooling.md`, those documents win (module/target names, test locations, build options,
> dependencies via submodules). Source files use `.hpp`/`.cpp`.

Status: proposal. Scope: `imsrg._core` (nanobind extension) and the pure-Python package `imsrg`.
Old reference: `imsrg-devel/src/pyIMSRG.cc` (pybind11, 961 lines, ~150 individually exposed commutator terms, `arma::mat` as an opaque class, no numpy, no GIL release).

## 1. Goals, non-goals, design rules

### 1.1 Goals
1. **Prototyping.** A user can write a new generator, a new flow driver, or a new operator in numpy and plug it into the C++ solver without recompiling (callbacks, section 2.7).
2. **Testing.** Every individual commutator term (`comm222_pp_hh`, ...) is callable from Python for term-by-term validation against (a) the old code's `pyIMSRG`, (b) brute-force m-scheme references, (c) the in-tree reference implementations. This is the main consumer of `imsrg.commutators.terms`.
3. **Scripting full pipelines.** Read interaction, build model space, HF, normal order, flow, transform observables, write output. Equivalent to the `imsrg++` command-line driver, in about 15 lines.

### 1.2 Non-goals
- No Python-side physics: no Python reimplementation of the commutators for production use (reference commutators in numpy live in `tests/`, not in the package).
- No exposure of internal caches or hash tables (`NineJHash`, `Vmon3_keys`, factorial lists, scratch-file handling). No RPA/TDA, no atomic systems (dropped project-wide).
- No stable ABI promise for `imsrg._core`; the stable surface is the documented `imsrg.*` Python names.
- No pickling of operators (use `imsrg.io.write_operator`/`read_operator`, the C++ binary format). `Operator` supports `copy.copy`/`copy.deepcopy` (both deep, see below).

### 1.3 Design rules
- **Names**: PEP 8. `snake_case` functions/methods/attributes, `CamelCase` classes. No `Get`/`Set` prefixes: properties (`ms.emax`, `op.zero_body`) and methods named by what they do. No Hungarian leftovers (`TBME`, `OneBody` -> `one_body`, `two_body`).
- **Keyword arguments** for everything beyond the first two obvious positionals; all option structs are keyword-only dataclass-like nanobind classes (`CommutatorOptions(use_imsrg3=True)`), with `repr`, `==`, `copy.replace`, and `to_dict()`/`from_dict()` (used by `imsrg.config`).
- **Enums, not strings**: C++ enums become `enum.Enum` subclasses (`nb::enum_`), e.g. `Parity`, `Generator`, `Hermiticity`, `Method`. A convenience layer accepts the lowercase string for scripting (`generator="white"`); `imsrg.config` is the only place where strings are parsed, and it does so once, into enums.
- **Docstrings on everything** exposed. Each `.def` carries a numpy-style docstring (Parameters/Returns/Notes with the equation or reference for physics routines). A CI check fails if any public symbol of `imsrg._core` has an empty `__doc__`.
- **Type stubs**: `python -m nanobind.stubgen -m imsrg._core -o python/imsrg/_core.pyi` run at build time and shipped (`py.typed` included). The pure-Python layer is fully annotated; `mypy --strict` runs in CI against stubs.
- **No object slicing, no inheritance in the bound types**. `Operator` is a single value type (C++ has no `Operator` hierarchy); `Generator`, `Observable` factories etc. return plain values/enums. Where C++ has a small interface (`GeneratorFn`, flow callbacks) Python subclassing is not supported; callables are used instead.
- **Value semantics**. `Operator.copy()`, `copy.copy`, `copy.deepcopy` and `Operator(other)` are all deep copies. `a + b`, `a * 2.0` return new operators; `a += b` is in-place. There are no aliasing surprises: slices of an operator's storage are numpy views, never `Operator` objects.
- **Lifetimes**: `ModelSpace` is an immutable object held by `std::shared_ptr<const ModelSpace>`. Every `Operator`, `Channel` accessor, `Solver`, `HFResult` etc. holds its own `shared_ptr` to the model space, so a Python `ModelSpace` may be garbage collected while operators live. Numpy views returned from an operator use `nb::rv_policy::reference_internal` plus an ndarray owner capsule that keeps the **Operator's storage** (not just the Operator wrapper) alive. A view therefore remains valid after `del op`, but not after `op.resize/clear` (those do not exist; there is nothing that reallocates storage in-place except `op = op.with_model_space(...)`, which returns a new object).
- **Errors**: C++ exceptions map to `imsrg.ImsrgError` subclasses (`ModelSpaceError`, `SymmetryError`, `ConvergenceError`, `FileFormatError`, `StaleHostError`); `std::invalid_argument` maps to `ValueError`; index out-of-range to `IndexError`. No `exit()` anywhere in the core (old `UnitTest` calls `exit(0)`).
- **Strong index types** are exposed as thin int-like classes (`OrbitIndex`, `ChannelIndex`, `KetIndex`), implicitly convertible *from* Python `int` at API boundaries but returned as their own type (hashable, `int()`-able, `repr` shows `OrbitIndex(7)`). Passing a `KetIndex` where a `ChannelIndex` is expected raises `TypeError` (catches the commonest old-API bug class). Bulk accessors return `numpy.ndarray[int32]` of raw values for speed.

## 2. Proposed Python API by module

Layout: `imsrg/__init__.py` re-exports the names used in section 2.8's example; submodules mirror the C++ library targets.

```
imsrg/
  __init__.py       Runtime, ModelSpace, Operator, units, enums, __version__
  _core.*.so        nanobind extension (all heavy code)
  _core.pyi         generated stubs
  basis.py          thin re-exports/helpers around _core.basis
  operators.py      constructors of physical operators + helpers
  io.py             readers / writers
  hf.py  mbpt.py  commutators.py  flow.py
  pipeline.py  config.py     pure Python
  testing/          reference implementations, comparison helpers (not imported by default)
```

### 2.0 `imsrg.Runtime` (see also section 4)
```python
class Runtime:
    def __init__(self, *, num_threads: int | None = None, device_id: int = 0,
                 mpi: bool = False, kokkos_args: Sequence[str] = ()) -> None: ...
    def __enter__(self) -> "Runtime": ...
    def __exit__(self, *exc) -> None: ...
    @property
    def execution_space(self) -> str          # "OpenMP" | "Serial" | "CUDA" | "HIP" | ...
    @property
    def mpi_rank(self) -> int; mpi_size: int
    @staticmethod
    def current() -> "Runtime | None"
    @staticmethod
    def ensure() -> "Runtime"                  # lazy default init, used internally
def init(**kw) -> Runtime                      # same as Runtime(**kw) + keep until exit
```
**Decision**: `import imsrg` does **not** initialize Kokkos/MPI (import side effects break thread/device configuration and `multiprocessing`). The runtime is created (a) explicitly by `imsrg.init(...)` or `with imsrg.Runtime(...):`, or (b) lazily with defaults on the first call that allocates device memory (`ModelSpace.__init__` and `Operator.__init__` call `Runtime.ensure()`). Finalization happens at interpreter exit through `atexit` (after all `Operator`s are collected: see 4.3), or at `__exit__` of an explicit context manager, which first raises `RuntimeError` if live device-resident objects still exist.

### 2.1 `imsrg.basis`
```python
class Orbit:                       # immutable
    n: int; l: int; j2: int; tz2: int; index: OrbitIndex
    energy_label: int              # 2n + l
    occupation: float              # reference occupation (from ModelSpace)
    role: OrbitRole                # CORE | VALENCE | QSPACE | UNOCCUPIED (replaces cvq)
    def __repr__(self) -> str      # "Orbit(0p3/2, proton)"

class ModelSpace:                  # immutable; built via classmethods
    @classmethod
    def from_reference(cls, *, emax: int, reference: str | Mapping[OrbitKey, float],
                       hw: float, e2max: int | None = None, e3max: int = 0,
                       lmax: int | None = None, valence: str | None = None,
                       core: str | None = None, emax_unocc: int | None = None,
                       target: str | tuple[int, int] | None = None) -> "ModelSpace": ...
    # reference: "O16", "He6", or {(0,1,3,-1): 1.0, ...} occupations, or via ModelSpace.Builder
    @classmethod
    def from_orbits(cls, orbits: Iterable[Orbit], *, hw, e2max=None, e3max=0, ...) -> "ModelSpace": ...
    def with_reference(self, occupations) -> "ModelSpace"   # new object, shares basis storage
    def with_natural_occupations(self, occ) -> "ModelSpace"

    # scalars (read-only properties)
    emax, e2max, e3max, lmax: int;  hw: float;  a_ref: float; z_ref: float
    n_orbits: int;  n_two_body_channels: int;  n_three_body_channels: int
    # collections (numpy int32 arrays of OrbitIndex values, or tuples of Orbit if requested)
    holes, particles, core, valence, qspace, all_orbits: np.ndarray
    def orbit(self, i: OrbitIndex | int) -> Orbit
    def orbit_index(self, *, n: int, l: int, j2: int, tz2: int) -> OrbitIndex
    def orbit_from_label(self, label: str) -> OrbitIndex            # "p0p3/2"
    def two_body_channel(self, ch: ChannelIndex | int) -> TwoBodyChannel
    def find_two_body_channel(self, *, j: int, parity: Parity, tz: int) -> ChannelIndex
    def three_body_channel(...)  / find_three_body_channel(...)
    def one_body_blocks(self) -> list[OneBodyBlock]                 # (l, j2, tz2) -> orbit list
    def find_fermi_energy(...)  # dropped (see section 3)

class TwoBodyChannel:               # lightweight view, holds the model space
    j: int; parity: Parity; tz: int; n_kets: int
    kets: np.ndarray                # shape (n_kets, 2) of OrbitIndex values
    def local_index(self, p, q) -> KetIndex | None
    def ket_subset(self, kind: KetClass) -> np.ndarray   # KetClass.PP, HH, PH, CC, VC, QC, VV, QV, QQ

class ThreeBodyChannel: twoj, parity, twotz, n_kets, kets (shape (n,3)+Jpq), local_index(...)
```
`ModelSpace` has no setters (old `SetHbarOmega`, `SetE3max`, `SetReference`, ...): derived models spaces are made with `with_*`. Equality (`==`) compares the defining parameters. Angular-momentum helpers live in `imsrg.angmom` (`cg`, `three_j`, `six_j`, `nine_j`, `moshinsky`), vectorized over numpy inputs; caches are an implementation detail inside the core (no `precalculate_*` calls from Python).

### 2.2 `imsrg.operators`
```python
class Operator:
    def __init__(self, ms: ModelSpace, *, j_rank: int = 0, tz_rank: int = 0,
                 parity: Parity = Parity.EVEN, particle_rank: int = 2,
                 hermiticity: Hermiticity = Hermiticity.HERMITIAN,
                 reduced: bool = True) -> None          # zero operator
    # metadata
    model_space: ModelSpace; j_rank, tz_rank: int; parity: Parity; particle_rank: int
    hermiticity: Hermiticity; is_reduced: bool; normal_ordered_wrt: NormalOrdering   # NONE | REFERENCE | CORE | FILLED_VALENCE
    # storage
    zero_body: float                                    # read/write property
    one_body: OneBody                                   # accessor object (below)
    two_body: TwoBody
    three_body: ThreeBody                               # only if particle_rank == 3
    # residency (section 4.2)
    def sync_to_host(self) -> Operator                  # returns self
    def sync_to_device(self) -> Operator
    residency: Residency                                # HOST_CURRENT | DEVICE_CURRENT | BOTH
    # arithmetic (value semantics; Python-visible results are new Operators)
    __add__, __sub__, __mul__ (float), __rmul__, __truediv__, __neg__, __iadd__, __isub__, __imul__, __itruediv__
    def copy(self) -> Operator
    def norm(self, *, parts: Parts = Parts.ALL) -> float     # Parts: ONE | TWO | THREE | ALL; ALL = sqrt(n1^2+n2^2+n3^2), zero-body EXCLUDED (imsrg++, D12)
    def to_reduced(self) / to_unreduced(self) -> Operator
    # transformations
    def normal_order(self, *, wrt: NormalOrdering = NormalOrdering.REFERENCE) -> Operator
    def undo_normal_order(self) -> Operator
    def renormal_order(self, *, wrt: NormalOrdering, keep_three_body: bool = False) -> Operator
    def truncate(self, *, particle_rank: int) -> Operator    # replaces Erase*Body
    def average_isospin(self) -> Operator
    def with_model_space(self, ms: ModelSpace) -> Operator   # embedding / truncation to another basis
```
Normal ordering etc. return new operators (value semantics, no hidden in-place state mutation as in old `DoNormalOrdering`, which mutates and returns `Operator`). An in-place spelling is available where memory matters: `op.normal_order_(wrt=...)`.

**Storage accessors and the host/device policy (decision).** Operators may live in Kokkos device memory. Policy: **views of the host mirror, explicit synchronisation, tracked staleness**.
- `op.one_body.matrix` -> `ndarray[float64]` of shape `(n_orbits, n_orbits)`, F-order, zero-copy view of the host mirror (one-body is small; this is the dense reduced matrix).
- `op.two_body.block(ch_bra, ch_ket=None, *, writable=False)` -> `ndarray` of shape `(n_kets_bra, n_kets_ket)`, F-order, zero-copy view of the host mirror, normalised antisymmetrised J-coupled elements in the operator's current reduced/normalisation convention. `ch_ket` defaults to `ch_bra`. Also `op.two_body.diagonal_blocks()` -> iterator of `(ChannelIndex, ndarray)`.
- `op.two_body.element(ch_bra, ch_ket, bra, ket)`, `op.two_body.get(j, a, b, c, d, *, tz=..., parity=...)` -> scalar convenience readers that return unnormalised/normalised physics values via `imsrg.tbme.get_j(...)`; slow path, for tests.
- `op.three_body.block(ch_bra, ch_ket)` same convention; stored half-ladder only (documented); `op.three_body.element(...)` returns the full antisymmetrised element.
- **Staleness rule**: each operator carries a state `{HOST_CURRENT, DEVICE_CURRENT, BOTH}`. Obtaining a host view when the state is `DEVICE_CURRENT` raises `StaleHostError("call op.sync_to_host() first")`. We chose *raise* over *implicit sync* because an implicit device-to-host copy of a 3-body operator (tens of GB) hidden inside an innocent-looking `block()` call is a performance bug that should be visible. We chose *views over copies* because blocks are accessed in loops (prototype generators touch every block) and 2-body block sizes (up to ~10^3 x 10^3) make copies cost more than the view. Every compute entry point (`commutator`, `normal_order`, solver) accepts any state: if the state is `HOST_CURRENT` and the op was handed out as writable, it syncs to device automatically (documented; cheap check). Read-only views (default, numpy `writeable=False`) never invalidate the device copy; a view obtained with `writable=True` (or `op.edit()` context) marks the state `HOST_CURRENT` (device stale) immediately. A host-only build (Serial/OpenMP space where host == device mirror) makes `sync_*` no-ops and the state always `BOTH`, so scripts written on a laptop run unchanged on GPU nodes.
- A numpy view of a block remains valid as long as the operator lives; calling `sync_to_host()` while views exist is safe (host mirror storage is never reallocated).

```python
with op.edit() as e:                  # optional sugar: sync_to_host on enter, sync_to_device on exit
    e.one_body.matrix[:] = h1
    e.two_body.block(ch)[:] = v_ch
```

Operator factory (replaces `OperatorFromString`, string parsing happens only in `imsrg.config`):
```python
ops.kinetic_energy_cm(ms) ; ops.kinetic_energy_rel(ms) ; ops.t_rel / hcm ; ops.number(ms, kind=...)
ops.r2(ms, *, species: Species = Species.MATTER) ; ops.rp2(ms)   # point proton radius squared incl. CM and spin-orbit corrections
ops.electric_multipole(ms, k) ; ops.magnetic_multipole(ms, k) ; ops.isospin2(ms) ; ops.l_dot_s(ms)
ops.from_name(ms, name: str, **params) -> Operator             # string front-end for config/CLI only
```
(Phase 1: `kinetic_energy_*`, `hcm`, `r2`, `rp2`, `number`; the rest follows the operator-library roadmap. Tensor/dagger/0vbb/DM later.)

### 2.3 `imsrg.io`
```python
io.read_me2j(path, ms, *, file_e1max: int, file_e2max: int | None = None, file_e3max: int = 0,
             hw: float | None = None, fmt: Me2jFormat = Me2jFormat.AUTO) -> Operator    # .me2j, .me2j.gz, binary variants
io.read_three_body(path, ms, *, fmt: ThreeBodyFormat, file_e1max, file_e2max, file_e3max,
                   no2b_path: str | None = None) -> Operator       # phase 2: me3j, no2b, hdf5, Navratil
io.read_no2b(path, ms, ...) -> Operator
io.write_me2j(op, path, *, gz=True) ; io.write_tokyo(op, path_stem) ; io.write_nushellx(op, *, stem, fmt=NushellxFmt.INT|SPS|OPERATOR)
io.write_operator(op, path) / io.read_operator(path, ms) -> Operator    # native binary, versioned header
io.read_one_body_text / write_one_body_text(op, path)                    # simple human-readable
```
All readers take `ms` and return an `Operator` (no output-parameter style as in `ReadWrite.ReadBareTBME_Darmstadt(file, Operator&)`). Readers/writers release the GIL. Where the old signature had a `(file2e1max, file2e2max, file2lmax)` triple, these are keyword-only `file_e1max`, `file_e2max`, `file_lmax`. Missing files raise `FileNotFoundError`, malformed content `FileFormatError` with the line/offset.

### 2.4 `imsrg.hf`
```python
hf.solve(h: Operator, *, options: HFOptions = HFOptions()) -> HFResult
@dataclass HFOptions: max_iterations=1000, tolerance=1e-8, accelerator="diis"|"mixing", mixing=0.2 (imsrg++ default, HartreeFock.cc:59-60), freeze_occupations=False,
                      use_three_body=True, fill_lowest: bool=False
class HFResult:
    energy: float                       # E_HF
    single_particle_energies: np.ndarray ; orbit labels via .ms.orbit
    c: np.ndarray                        # (n_orbits, n_orbits) transformation, one block per (l, j, tz) embedded dense
    rho: np.ndarray ; fock: np.ndarray ; n_iterations: int ; converged: bool
    def transform(self, op: Operator) -> Operator          # to the HF basis (generic, any rank)
    def normal_ordered_hamiltonian(self, h: Operator) -> Operator   # transform + normal order wrt reference, discards 3N residual per options
    def basis_change(self) -> BasisChange                   # reusable object (also from natural orbitals)
```
Non-converged HF raises `ConvergenceError` unless `options.raise_on_failure=False`. HF is a function returning an immutable result; there is no stateful `HartreeFock` class with `Solve()`.

### 2.5 `imsrg.mbpt`
```python
mbpt.mp2(h: Operator) -> float ; mbpt.mp3(h) -> MP3Terms(pp, hh, ph, total)
mbpt.natural_orbitals(h: Operator, *, order: int = 2) -> NaturalOrbitals   # phase 2
NaturalOrbitals.basis_change: BasisChange ; .occupations: np.ndarray ; .transform(op)
```
`BasisChange` (shared with hf): `.apply(op)`, `.compose(other)`, `.matrix` (read-only numpy). Van Vleck/MBPT4 diagnostics stay out of Phase 1.

### 2.6 `imsrg.commutators`
```python
commutators.commutator(x: Operator, y: Operator, *, options: CommutatorOptions = CommutatorOptions()) -> Operator
commutators.double_commutator_factorized(eta, h, *, options) -> Operator   # factorized 223-231/232, phase 3
class CommutatorOptions:  # explicit replacement of the global static flags
    imsrg3: Imsrg3Mode = Imsrg3Mode.OFF      # OFF | N7 | FULL (FULL later)
    only_2b_eta: bool = False ; three_body_threshold: float = 0.0
    enabled_terms: frozenset[Term] | None = None            # None = all valid for the chosen mode
    discard_three_body_output: bool = False
    factorized: FactorizedOptions = FactorizedOptions()      # goose-tank, type II/III intermediates ...
    perturbative_triples: bool = False
    deterministic: bool = False                              # fixed reduction order for term tests
```
Term-by-term (testing/debugging) submodule, **separate from the stable API**:
```python
from imsrg.commutators import terms        # also imsrg.commutators.Term enum
terms.comm222_pp_hh(x, y) -> Operator        # one function per term; returns a fresh Operator containing only that term's contribution
terms.apply(Term.COMM_222_PP_HH, x, y, *, options=...) -> Operator       # same by enum
terms.TERMS: Mapping[str, TermInfo]          # name -> TermInfo(bra_legs, ket_legs, scalar/tensor, requires_3b, doc)
terms.reference.comm222_pp_hh(x, y)          # straightforward J-coupled reference (loops), only for small spaces
```
Rules: a term function accumulates into a **zero-initialised output of the correct J/Tz/parity/rank/hermiticity/particle rank** (the logic that the old `Test_against_ref_impl` hand-rolls), so `terms.comm222_pp_hh(x, y)` is self-contained. Names equal the old term names minus the `ss`/`st` suffix; tensor handling is by rank of the operands (scalar-scalar vs scalar-tensor dispatched on ranks), so `comm121ss`/`comm121st` collapse to `comm121`. Enabling/disabling by `TurnOnTerm("...")` becomes `CommutatorOptions(enabled_terms=...)`, no global state.

### 2.7 `imsrg.flow`
```python
flow.Generator            # enum: WHITE, ATAN, WEGNER, IMTIME, SHELL_MODEL, SHELL_MODEL_ATAN, WEGNER_ATAN, ...  (names kept in C++ enum)
flow.Method               # enum: MAGNUS_EULER, MAGNUS_MODIFIED_EULER, MAGNUS_ADAPTIVE (later), BCH_RK4 ...
flow.Denominator          # enum: MOLLER_PLESSET, EPSTEIN_NESBET, ...

@dataclass SolverOptions: method=Method.MAGNUS_EULER, generator=Generator.WHITE, denominator=Denominator.EPSTEIN_NESBET,
                          ds=0.5, ds_max=0.5, s_max=500.0, eta_criterion=1e-6, omega_norm_max=0.25,
                          ode_tolerance=1e-6, commutator=CommutatorOptions(), bch=BchOptions(),
                          flow_file: str | None = None, scratch_dir: str | None = None
class Solver:
    def __init__(self, h: Operator, *, options: SolverOptions = SolverOptions(),
                 generator: GeneratorFn | None = None, observer: Callable[[FlowState], bool | None] | None = None)
    def solve(self) -> FlowResult                    # releases the GIL; observer may return True to stop
    def add_observable(self, name: str, op: Operator) -> None      # evolved alongside H (Magnus: tracked via Omega)
    def transform(self, op: Operator, *, inverse: bool = False) -> Operator   # apply accumulated Omega to any operator
    omega_list: Sequence[Operator]                   # copies, host-resident, lazy
    def state(self) -> FlowState                    # s, E0, norm(eta), n_omega, h (current Hamiltonian, device-resident)

@dataclass FlowResult: s: float, e0: float, h: Operator, omega: Operator|None, n_steps: int, history: pandas-free list[FlowRecord], converged: bool
```
Generator extension point (prototyping use case 1): a Python generator is any callable

```python
GeneratorFn = Callable[[FlowState], Operator]      # returns eta (anti-Hermitian, same ranks as H)
```
The C++ solver calls it with the GIL re-acquired (`nb::gil_scoped_acquire` inside the trampoline). The `FlowState` passes `h` (the current Hamiltonian, host synced on demand: the trampoline syncs before the call and after the return syncs the returned eta to device) plus the denominators helper `FlowState.denominators()` returning numpy views of `Delta_ai`, `Delta_abij`. Example, numpy White generator in ~12 lines in 2.8b. Likewise `observer(FlowState)` is called after every accepted step (for printing, early stopping, saving). The built-in generators run entirely in C++; the callable path costs one host-device synchronisation per step and is documented as a prototyping tool.

BCH/Magnus utilities for testing: `flow.bch_transform(omega, op, *, options) -> Operator`, `flow.bch_product(omega1, omega2, *, options) -> Operator`, `flow.magnus_omega_step(...)`.

### 2.8 `imsrg.pipeline` and `imsrg.config`
```python
config.Config                  # dataclass tree mirroring the old CLI parameters, but typed (enums), versioned
config.from_toml(path) / from_dict(d) / to_dict(); Config.validate()      # unknown keys are errors
pipeline.run(cfg: Config) -> PipelineResult             # the library equivalent of the imsrg++ driver
pipeline.steps: read_hamiltonian, build_model_space, hartree_fock, normal_order, flow, transform_observables, write_outputs
```
`pipeline.run(config)` calls the C++ `driver::run` through the bindings, so there is exactly one pipeline implementation; the step functions above remain available for interactive use. `python -m imsrg run config.toml` calls `pipeline.run`. There is **no** console-script entry point named `imsrg` (that name belongs to the C++ executable). Observables are a list of `ObservableSpec(name, kwargs)` resolved through `ops.from_name`. Results: `PipelineResult.e_hf, .e0, .observables: dict[str, float|np.ndarray], .log`.

### 2.9 Full example script

```python
import numpy as np
import imsrg
from imsrg import ModelSpace, io, hf, flow, operators as ops
from imsrg.commutators import CommutatorOptions, Imsrg3Mode

imsrg.init(num_threads=8)                       # optional; lazy default otherwise

emax, hw = 4, 16.0
ms = ModelSpace.from_reference(emax=emax, e2max=2 * emax, reference="O16", hw=hw)

# 2N interaction (file e1max = 14, 2N-only: Hamiltonian = T_rel + V)
h_bare = io.read_me2j("VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz", ms,
                      file_e1max=14, file_e2max=28, file_lmax=14)  # 2N interaction ONLY; add the kinetic term separately, as the driver does
rp2 = ops.rp2(ms)                                      # observable, built in the HO basis

res = hf.solve(h_bare)
print(f"E_HF = {res.energy:.6f} MeV  (converged in {res.n_iterations} iterations)")
h = res.normal_ordered_hamiltonian(h_bare)
rp2_hf = res.transform(rp2).normal_order()

opts = flow.SolverOptions(method=flow.Method.MAGNUS_EULER, generator=flow.Generator.WHITE,
                          denominator=flow.Denominator.EPSTEIN_NESBET, s_max=500.0, ds=0.5,
                          commutator=CommutatorOptions(imsrg3=Imsrg3Mode.OFF))
solver = flow.Solver(h, options=opts, observer=lambda st: print(f"s={st.s:8.3f} E0={st.e0:.6f}"))
flow_result = solver.solve()
print(f"E0 = {flow_result.e0:.6f} MeV   (HF {res.energy:.6f})")

rp2_evolved = solver.transform(rp2_hf)                  # apply Omega
r2 = rp2_evolved.zero_body + 0.0                        # <Rp^2> (add 3/2 hbar/(mw)-type corrections inside ops.rp2)
print(f"<Rp^2> = {rp2_evolved.zero_body:.5f} fm^2")
```
(b) Prototype generator in numpy (White, Epstein-Nesbet, 1b+2b):
```python
def white(st: flow.FlowState) -> imsrg.Operator:
    h, eta = st.h.sync_to_host(), imsrg.Operator(st.h.model_space, hermiticity=imsrg.Hermiticity.ANTI_HERMITIAN)
    d1 = st.denominators(rank=1)
    for ch, vals in st.off_diagonal_blocks(rank=2):       # (ChannelIndex, ndarray block indices of ph/hp)
        ...
    h1 = h.one_body.matrix; e1 = eta.one_body.matrix_writable()
    mask = st.one_body_offdiag_mask                       # numpy bool array
    e1[mask] = h1[mask] / d1[mask]; e1[:] = e1 - e1.T
    return eta
solver = flow.Solver(h, options=opts, generator=white)
```
(c) Term test against a stored old-code result: see section 5.

## 3. Mapping: old `pyIMSRG` -> new API

| Old (pyIMSRG) | New | Note |
|---|---|---|
| `ArmaMat` (all 15 methods) | `numpy.ndarray` | zero-copy F-order views; `Set(i,j,x)` -> `a[i,j] = x`; `Schur_Prod` -> `a * b`; `Norm` -> `np.linalg.norm`; `save/load` -> `np.save/load` |
| `Orbit` (`n,l,j2,tz2,occ,cvq,index`) | `basis.Orbit` | immutable; `cvq` -> `role: OrbitRole`; `occ` -> `occupation` |
| `Ket`, `Ket3` | dropped as classes | `kets` ndarray of indices |
| `TwoBodyChannel`, `ThreeBodyChannel` | `basis.TwoBodyChannel/ThreeBodyChannel` | read-only; `GetKetIndex_pp,...` -> `ket_subset(KetClass.PP)`; `GetLocalIndex` -> `local_index`; `TwoBodyChannel_CC` dropped (copy/paste bug; CC channels are internal) |
| `ModelSpace` ctors (5 overloads) | `ModelSpace.from_reference(...)`, `from_orbits`, `ModelSpace.Builder` | immutable, no setters |
| `ModelSpace` setters (`SetHbarOmega, SetEmax, SetE2max, SetE3max, SetdE3max, SetLmax, SetEmaxUnocc, SetEmax3Body, SetTargetMass/Z, SetReference, SetOcc, SetOccNAT, AddOrbit, SetupKets, Setup3bKets, ResetFirstPass, SetScalar*FirstPass, Init_occ_from_file, InitSingleSpecies`) | constructor kwargs / `with_reference`, `with_natural_occupations` | `*FirstPass`, `SetupKets`, `Setup3bKets`, `ClearVectors`: dropped (internal lifecycle) |
| `ModelSpace` getters | properties (`emax, e2max, e3max, hw, a_ref, z_ref, n_orbits, ...`) and `orbit`, `two_body_channel`, `find_two_body_channel`, `orbit_index`, `orbit_from_label` | `GetOneBodyChannels` -> `one_body_blocks()`; `Index2String` -> `Orbit.__str__` |
| `PreCalculateSixJ/NineJ/Moshinsky`, `GetMoshinsky/SixJ/NineJ`, `NineJHash/UnHash` | `imsrg.angmom.*` free functions; precalc dropped | caches are internal and lazily filled |
| `FindEFermi`, `holes/particles/core/valence/qspace/all_orbits` attrs | `find_fermi_energy` later; sets -> ndarray properties | read-only; no per-access copy semantics hazard |
| `Operator` ctors (default, `(ms)`, copy, `(ms,J,T,P,rank)`) | `Operator(ms, *, j_rank, tz_rank, parity, particle_rank)`, `copy()` | default ctor dropped (no model-space-less operators) |
| `Operator` arithmetic, `ZeroBody`, `OneBody`, `TwoBody`, `ThreeBody` | `+ - * /`, `zero_body`, `one_body`, `two_body`, `three_body` | returns new objects; no mutable internal `TwoBodyME` handle |
| `GetOneBody/SetOneBody/SetOneBodyME/GetOneBodyChannel/ScaleOneBody/ScaleTwoBody` | `one_body.matrix` view, `one_body.element`; scaling by numpy | |
| `GetTwoBody/SetTwoBody/GetTwoBodyDimension` | `two_body.block`, `two_body.element(...)`, `two_body.n_kets(ch)` | |
| `EraseZeroBody/OneBody/TwoBody/ThreeBody` | `truncate(particle_rank=)`, `op.zero_body = 0.0`, `op - op` | |
| `DoNormalOrdering`, `DoNormalOrderingCore`, `DoNormalOrderingFilledValence`, `UndoNormalOrdering(Core)`, `ReNormalOrderCore` | `normal_order(wrt=...)`, `undo_normal_order()`, `renormal_order(wrt=..., keep_three_body=)` | `_` suffixed in-place variants |
| `Norm, OneBodyNorm, TwoBodyNorm, ThreeBodyNorm` | `norm(parts=...)` | |
| `Set/IsHermitian, SetAntiHermitian, IsAntiHermitian, SetNonHermitian` | `hermiticity` property (enum) | set only at construction (`with_hermiticity` unsafe-flag later for tests) |
| `IsReduced, MakeReduced, MakeNotReduced, MakeNormalized, MakeUnNormalized` | `is_reduced`, `to_reduced()/to_unreduced()`; normalisation is fixed internal convention | `MakeNormalized` dropped |
| `GetJRank, GetTRank, GetParity, GetParticleRank, SetParticleRank, GetNumberLegs, Size` | properties; `SetParticleRank` -> `with_particle_rank` (later); `GetNumberLegs` dropped (dagger ops later); `Size` -> `nbytes` | |
| `Truncate, DoIsospinAveraging, SetModelSpace` | `with_model_space`, `average_isospin` | |
| `Print*`, `PrintTimes` | `__repr__`/`__str__`, `op.summary()`, `imsrg.profiler.report()` | no stdout printing from library |
| `GetMP2_Energy, GetMP2_3BEnergy, GetMP3_Energy, GetPPHH_Ladders` | `mbpt.mp2`, `mbpt.mp3`; ladders later | |
| `ReadBinary/WriteBinary` | `io.read_operator/write_operator` | |
| `TwoBodyME` (14 methods) | `op.two_body` accessor | `GetTBME_J(Jbra,Jket,a,b,c,d)`, `GetTBMEmonopole` -> `two_body.get_j`, `two_body.monopole`; `GetChannelMatrix` -> `block`; `PrintAll/PrintMatrix` dropped; `Erase` dropped |
| `ThreeBodyME` (`GetME_iso`, `GetME_pn*`, `SetME_pn_ch`, `RecouplingCoefficient`, `TransformToPN`, `SwitchToPN_and_discard`, `SetMode`, `IsAllocated`, ...) | `op.three_body.element/block`; iso/pn mode handled internally | `GetME_iso(13 ints)` later (3N input phase); mode switches dropped |
| `ReadWrite` `ReadBareTBME_Darmstadt/Navratil/Jason`, `ReadTBME_Oslo/OakRidge` | `io.read_me2j(..., fmt=)` (Darmstadt), Navratil later; Oslo/OakRidge/Jason: dropped (unused formats; revisit on request) | |
| `Read_Darmstadt_3body`, `Read3bodyHDF5`, `ReadJacobi3NFiles`, `Set3NFormat`, `ReadDarmstadt_2bodyRel`, `Jacobi3BME` | `io.read_three_body` (me3j/hdf5), Jacobi path later | phase 2 |
| `Write_me2j(_gz)`, `Write_me3j`, `WriteTBME_Navratil` | `io.write_me2j`; 3N write later | |
| `WriteNuShellX_{sps,int,op}`, `ReadNuShellX_int(_iso)`, `WriteAntoine_{int,input}`, `WriteValence3body` | `io.write_nushellx(fmt=)`; reads and Antoine later/dropped | |
| `WriteTokyo`, `ReadTokyo`, `WriteTensor*`, `ReadTensorTokyo`, `Write/ReadOneBody/TwoBody_Takayuki`, `WriteOneBody_Oslo`, `WriteTwoBody_Oslo`, `WriteOneBody_Simple` | `io.write_tokyo`, `io.write_one_body_text`; tensor variants later; Takayuki/Oslo dropped | |
| `WriteOperator(Human)`, `ReadOperator(Human)`, `CompareOperators` | `io.write_operator/read_operator`; human form -> `repr`; comparison -> `imsrg.testing.compare` | |
| `ReadOperator2b_Miyagi`, `ReadTwoBodyEngel`, `ReadOperator_Nathan`, `ReadTensorOperator_Nathan`, `ReadRelCMOpFromJavier`, `Read2bCurrent_Navratil`, `ReadH2_2body`, `WriteDaggerOperator` | later (tensor/dagger/currents) or dropped; each needs a user request | |
| `SetCoMCorr`, `SetScratchDir/GetScratchDir`, `CopyFile` | `SolverOptions.scratch_dir`; others dropped | |
| `HartreeFock`: `Solve, TransformToHFBasis, GetNormalOrderedH(_Cin), GetHbare, GetOmega, CalcEHF, FillLowestOrbits, FreezeOccupations, DiscardNO2Bfrom3N` | `hf.solve`, `HFResult.transform`, `.normal_ordered_hamiltonian`, `HFOptions` | stateless |
| `HartreeFock`: `PrintSPE(andWF), GetRadialWF_r, GetHFPotential, GetAverageHFPotential, UpdateDensityMatrix, UpdateF, BuildMonopoleV, FreeVmon, GetValence3B, Vmon3*`, attrs `EHF, F, rho, C` | results: `energy, fock, rho, c, single_particle_energies`; `radial_wavefunction(...)` later; the rest dropped (internals) | |
| `HFMBPT`: `GetNaturalOrbitals, TransformHOToNATBasis, TransformHFToNATBasis, GetNormalOrderedHNAT, UseNATOccupations, C_HO2NAT, C_HF2NAT` | `mbpt.natural_orbitals`, `BasisChange`, `ModelSpace.with_natural_occupations` | phase 2 |
| `HFMBPT`: `GetMP2/3_Energy, GetMP3_*`, `GetMP4_*, SolveVanVleck, VV_*, SetVV*` | `mbpt.mp2/mp3`; MP4, Van Vleck: later/dropped | |
| `IMSRGSolver`: `Solve, Transform, InverseTransform, GetOmega, GetH_s, GetEta, GetS, AddOperator, GetOperator` | `Solver.solve/transform(inverse=)/omega_list/state()/add_observable` | |
| `IMSRGSolver` setters (`SetMethod, SetGenerator, SetEtaCriterion, SetDs, SetdOmega, SetOmegaNormMax, SetSmax, SetDsmax, SetODETolerance, SetDenominator*, SetOnly2bEta, SetOnly1bEta, SetHin, Reset, SetFlowFile, SetOmega, SetH_s`) | `SolverOptions` fields; `SetHin/Reset/SetOmega/SetH_s` -> construct a new `Solver` | no mutable setters; options immutable after construction |
| `IMSRGSolver`: `SetMagnusAdaptive, SetReadWrite, SetHunterGatherer, FlushOmegaToScratch, GetNOmegaWritten, GetOmegaSize, EstimateBCHError, UpdateEta, GetSystemDimension, CalculatePerturbativeTriples` | `Method.MAGNUS_ADAPTIVE`, `scratch_dir`; `estimate_bch_error` and `perturbative_triples()` (phase 3); rest dropped/internal | |
| `IMSRGSolverPV`, `GeneratorPV`, `Etapv`, `Solve_RK4`, `Solve_magnus_euler`, `AddOperatorPV`, ... | later (parity-violating / tensor) | |
| `Generator` (`SetType, SetDenominatorPartitioning, SetUseIsospinAveraging, Update, UpdateGeneral, GetHod*`) | `flow.Generator` enum, `GeneratorFn` callable, `Denominator`; `update` -> `st.generator(...)`; `GetHod` -> `FlowState.off_diagonal()` | `UpdateGeneral` dropped |
| `IMSRGProfiler` | `imsrg.profiler` (`report()`, `counters()`, `clear()`) | module-level, thread-safe |
| `Commutator.Commutator/CommutatorScalarScalar/CommutatorScalarTensor` | `commutators.commutator` | one entry point, dispatch on ranks |
| `Commutator.SetUseIMSRG3(N7)(_Tensor)`, `SetThreebodyThreshold`, `SetSingleThread`, `SetVerbose`, `SetPerturbativeTriples`, `PrintSettings`, `Set*Noqqq/Onlyvvv/valence2b/PertTripNovvv`, `Discard{0,1,2}bFrom3b`, `TurnOnTerm/TurnOffTerm` | `CommutatorOptions` fields (`imsrg3`, `three_body_threshold`, `enabled_terms`, `discard_three_body_output`, ...); `SingleThread` -> `Runtime(num_threads=1)`; `SetVerbose` -> logging; valence/qqq variants later | no global flags |
| `Commutator.comm110ss ... comm333_pph_hhpss` (24 scalar-scalar terms) and `comm111st ... comm333_pph_hhpst` (18 scalar-tensor terms) | `commutators.terms.<name>` (`comm110`, `comm220`, `comm111`, `comm121`, `comm221`, `comm122`, `comm222_pp_hh`, `comm222_ph`, `comm222_pp_hh_221`, `comm330`, `comm331`, `comm231`, `comm132`, `comm232`, `comm223`, `comm133`, `comm332_ppph_hhhp`, `comm332_pphh`, `comm233_pp_hh`, `comm233_ph`, `comm333_ppp_hhh`, `comm333_pph_hhp`) | `ss`/`st` merged by rank dispatch; terms involving 3b operands are 3N phase; `comm_pandya_single_channel` -> `terms.pandya_transform_single_channel` (testing) |
| `Commutator.FactorizedDoubleCommutator.*` (`comm223_231, comm223_232, *_chi{2b,1b}`, `SetUse_*` flags, `SetUse_*Intermediates`) | `commutators.double_commutator_factorized`, `FactorizedOptions` | phase 3; `_chi` variants are options |
| `BCH.BCH_Transform/BCH_Product`, `SetUseGooseTank`, `SetUseFactorized*`, `SetOnly2bOmega`, `SetComm223_231/232`, `Set_BCH_*_Threshold`, `SetBCHSkipiEq1` | `flow.bch_transform/bch_product`, `BchOptions` | |
| `ReferenceImplementations.*` (~60 terms, `diagram_CIa ... diagram_DIVb`, `comm223_231_BruteForce`, `TriplesGuess`, `TestMBPT6*`) | `imsrg.commutators.terms.reference.*` (same naming as terms); brute-force m-scheme refs in `imsrg.testing.mscheme`; diagram_* later; `TestMBPT6*` dropped | optional, built only with `IMSRG_BUILD_REFERENCE` |
| `UnitTest` (~130 methods) | pytest suite (section 5) plus `imsrg.testing.random_operator`, `imsrg.testing.mscheme` | no test code in the library; `SetRandomSeed` -> `random_operator(..., seed=)` |
| `RPA` | dropped | non-goal |
| operator library free functions (`TCM_Op, Trel_Op, R2CM_Op, HCM_Op, NumberOp, RSquaredOp, RpSpinOrbitCorrection, E0Op, ElectricMultipoleOp, MagneticMultipoleOp, Isospin2_Op, LdotS_Op, OperatorFromString`) | `operators.*` (names in 2.2) | `E0Op/Schiff/Allowed*/Sigma/SerberTypePotential/...` later; `OperatorFromString` -> `operators.from_name` |
| `RPA_resummed_1b, FirstOrderCorr_1b, MBPT2_SpectroscopicFactor, Single_Ref_1B_Density_Matrix, Get_Charge_Density, Embed1BodyIn2Body, Calculate_p1p2_all` | `Embed1BodyIn2Body` -> `Operator.with_particle_rank`; density-matrix routines later; rest dropped | |
| numerics: `HO_density, HO_Radial_psi, RadialIntegral*, FrequencyConversionCoeff, TalmiI/B, Tcoeff, GetOccupationsHF, GetDensity` | `imsrg.angmom` and `imsrg.hopot` (vectorized numpy-in/numpy-out), those used by tests only | |
| `CG, ThreeJ, SixJ, NineJ, NormNineJ, Moshinsky, FillFactorialLists, factorial, double_fact, AngMomJmin/Jmax` | `imsrg.angmom.cg/three_j/six_j/nine_j/moshinsky/jmin/jmax`; factorial lists dropped | needed by tests as oracle |
| `BuildVersion`, `SetUseGooseTank/IMSRG3/IMSRG3N7` (module-level duplicates) | `imsrg.__version__`, `imsrg.build_info()` (git hash, compiler, Kokkos backend); duplicates dropped | |

## 4. Host/device, threading and runtime policy

### 4.1 GIL
- Every call that can take more than ~10 ms releases the GIL with `nb::call_guard<nb::gil_scoped_release>()`: `commutator`, all `terms.*`, `normal_order*`, `hf.solve`, `mbpt.*`, `Solver.solve`, `Solver.transform`, `BasisChange.apply`, all `io.*` readers/writers, `Operator` arithmetic on large operators (threshold on size: arithmetic releases always; cost is negligible).
- Accessors, property getters and numpy-view creation hold the GIL (nothing blocks).
- Callbacks (`GeneratorFn`, `observer`) re-acquire the GIL inside the trampoline (`nb::gil_scoped_acquire`) and must not be called from a Kokkos parallel region: the solver invokes them from its main control thread only. If the callback raises, the exception is captured, the solver stops at a step boundary, and the Python exception is re-raised from `solve()` (state remains consistent; `FlowResult` partial available via `exc.partial`).
- Releasing the GIL means two Python threads can call into the library concurrently. Documented rule: concurrent calls are safe iff they work on **distinct** `Operator` objects (the `ModelSpace` is immutable and shareable); the same `Operator` used mutably from two threads is undefined, and the core guards it with a per-operator `std::atomic<bool> busy` that raises `RuntimeError("operator in use by another thread")` rather than corrupting.
- Kokkos is run in the process-wide execution space; two Python threads launching Kokkos kernels in the same instance serialise through the stream. The supported production concurrency model is "one Python thread drives the library, Kokkos/OpenMP parallelises inside".

### 4.2 Host/device residency
As defined in 2.2: operator state `{HOST_CURRENT, DEVICE_CURRENT, BOTH}`; read-only views of the host mirror; `StaleHostError` instead of implicit sync; automatic host->device push before any compute call if a writable view was handed out. Additional rules:
- `Operator.copy()` copies the device side (device-to-device) when device is current, otherwise the host side; the copy inherits the state.
- `io.read_*` return operators `HOST_CURRENT` (filled on host, pushed lazily); `io.write_*` and `__repr__`/`norm` on a `DEVICE_CURRENT` operator reduce on device and do not require sync (`norm` is a device reduction; `write_*` performs an internal temporary sync with no state change).
- `numpy` views are marked non-owning and carry a Python-side `__array_interface__` readonly flag; the C++ side never hands out device pointers. (DLPack export of the **device** buffer for CuPy/PyTorch interop is a possible later addition: `op.two_body.block_dlpack(ch)`; not in scope.)
- Memory: `Operator.nbytes` (host and device) is exposed so scripts can budget; `Operator.release_host_mirror()` frees the host mirror of a device-resident operator (view creation afterwards re-creates it via `sync_to_host()`; existing views keep a reference and thus block release: `RuntimeError`).

### 4.3 Kokkos and MPI initialisation / finalisation
- `Runtime` object owns `Kokkos::initialize` (via `Kokkos::ScopeGuard` semantics) and MPI init. `Runtime` is a process singleton: a second `Runtime(...)` while one is active raises `RuntimeError` unless arguments are identical (then returns the existing one).
- Lazy default init on first allocation uses `Runtime.ensure()`: `num_threads` from `IMSRG_NUM_THREADS` or `OMP_NUM_THREADS` or hardware concurrency; device id from `IMSRG_DEVICE_ID`/local MPI rank.
- Finalization: `Kokkos::finalize` must run *after* every Kokkos `View` is destroyed or it aborts. Rules: (1) all device-owning C++ objects register with the runtime (counted); (2) `atexit` handler registered at first init calls `gc.collect()` then finalizes if the live count is zero; if live objects remain (e.g. user held a global operator), it logs a warning and **skips** `Kokkos::finalize` (process exit reclaims memory; avoids the abort at the cost of GPU-profiler tools not getting a clean finalize); (3) explicit `Runtime.__exit__`/`imsrg.finalize()` with live objects raises `RuntimeError` and leaves the runtime active. `fork()` after init is unsupported (documented; `multiprocessing` must use `spawn`, and workers call `init` themselves).
- Re-initialisation after finalisation is not supported by Kokkos: `Runtime(...)` after a clean `__exit__` raises with that explanation. For test suites, one session-scoped runtime fixture.
- **MPI**: off by default (`mpi=False`; no `MPI_Init` side effects and no mpi4py requirement). `Runtime(mpi=True)` calls `MPI_Init_thread(MPI_THREAD_FUNNELED)` unless `mpi4py.MPI.Is_initialized()` (then it adopts it, and never finalizes what it did not initialize; if mpi4py initialized MPI the finalization stays with mpi4py's own atexit, ordering: imsrg's handler is registered later and so runs first). Under `mpirun python script.py` every rank runs the same script; I/O readers read on rank 0 and broadcast if the build has MPI. `Runtime.mpi_comm()` returns an `mpi4py` communicator if available. The library functions that are MPI-collective are documented as such (all ranks must call `solve`). Python-level callbacks run on every rank; the user is responsible for rank-guarding prints (`rt.mpi_rank == 0`).
- Device selection under MPI: `device_id = local_rank % n_devices` by default (read from the MPI shared-memory communicator).

## 5. Testing the bindings (pytest)

Layout: `tests/python/` for binding tests (run against the installed wheel in CI), `tests/data/` for small reference arrays. C++ unit tests live in the C++ test tree; Python does *not* host the physics validation of the core, but it does host the comparisons that need an external oracle.

### 5.1 Tiers
1. **Binding hygiene** (`test_api_surface.py`): every public symbol has a docstring; stubs are in sync (regenerate and diff); enum round trips; `repr` stability; index-type strictness (`TypeError`); error mapping (each `ImsrgError` subclass is raised by a minimal failing call); lifetime tests (`del ms` while operator alive; view survives `del op`; `StaleHostError` path); `copy` and `deepcopy` are deep (modify the copy's block, original unchanged); GIL release verified by a thread that increments a counter while `commutator` runs on a medium operator.
2. **Numpy interop** (`test_views.py`): `block(ch)` is `F_CONTIGUOUS`, `writeable=False` unless asked, `np.shares_memory` between two views of the same block, `edit()` context sync round trip, shapes against `TwoBodyChannel.n_kets`.
3. **Term tests** (`test_terms_vs_reference.py`, parametrised over `terms.TERMS`): random operators from `imsrg.testing.random_operator(ms, j_rank, tz_rank, parity, particle_rank, hermiticity, seed)`; `terms.<t>(x, y)` vs `terms.reference.<t>(x, y)`, assert `abs(norm(diff)) < 1e-10 * max(1, norm(ref))` (relative; the old absolute 1e-6 is retained as a loose upper bound only). Seeds are explicit (fixing the old order-dependent global seed); three-body random generation is fully independent (fixes the old shadowed-variable bug). Small model spaces (emax 1-2, `He4/He6`) so the suite runs in seconds; large 3N-3N terms marked `@pytest.mark.slow`.
4. **Brute-force m-scheme** (`imsrg.testing.mscheme`, `test_mscheme.py`): numpy/python implementation of the textbook uncoupled formula for the 0-2 body terms (e.g. `Z0 = sum_ab n_a (1-n_b)(X_ab Y_ba - Y_ab X_ba)`), applied to the m-scheme expansion of `x`, `y` (`mscheme.expand(op)` builds dense m-scheme tensors via Clebsch-Gordan for emax <= 1, A <= 6) and compared with the J-scheme result expanded the same way. Used for normal ordering (`test_normal_ordering_mscheme`) and each commutator term at 2-body level; 3-body terms limited to emax=1.
5. **Physics regression** (`test_pipeline.py`): HF + IMSRG(2) Magnus for a tiny model (emax 2, He4, small synthetic interaction constructed in the test, plus `@pytest.mark.data` tests using the real VNN file if `IMSRG_TEST_DATA` is set): E_HF and E0 against stored old-code values (section 5.2), tolerance 1e-6 MeV; `pipeline.run(config)` equals the hand-written script.
6. **Prototyping path** (`test_callbacks.py`): numpy White generator equals the built-in `Generator.WHITE` flow to 1e-10 on a small space; observer early stop; exception propagation from callback.

### 5.2 Reference data from the old pyIMSRG
The old code stays buildable (read-only tree) and is the oracle for the first milestone; the process is scripted in `tools/reference/`:
1. `tools/reference/gen_old.py` (run with the **old** `pyIMSRG` on `PYTHONPATH`, never imported by the new package): for a grid of small model spaces (`He4`, `He6`, `C12`-p-shell, emax 1-2) and seeds, builds a deterministic operator by *numerical content* rather than by the old RNG: it fills one-body and 2-body channel matrices and 3-body elements from a fixed formula or from a stored npz (a deterministic function of `(channel, ibra, iket)` such as a hash-based pseudo-random number, symmetrised for hermiticity, implemented identically in the new `imsrg.testing`), runs `Commutator.comm*ss/st` terms via `pyIMSRG.Commutator`, and the HF/IMSRG(2) chain, and writes `tests/data/old_pyimsrg_<case>.npz` (operator inputs as flat arrays in a documented ordering, outputs, energies, metadata: pyIMSRG build hash, tolerances).
2. The new test suite loads the npz, builds `x`, `y` with `imsrg.testing.from_flat(...)` (maps the old channel/ket ordering to the new one through `(J, parity, Tz, p, q)` labels, never through raw indices), applies the new term and compares. Ordering conventions (old `index_t` hashing, normalisation, reduced vs. unreduced storage) are normalised in `from_flat` only, in one place.
3. Because the old RNG is order-dependent and has the 3-body shadowing bug, we do **not** try to reproduce the old random operators; the data files carry the actual numbers.
4. The npz files are committed (small, a few MB total; larger cases fetched by `tools/reference/fetch.py` into `IMSRG_TEST_DATA`). Regeneration is a manual step (needs the old build), and the generator script is pinned to the old commit hash recorded in the npz metadata.
5. Direct side-by-side mode (developer machine only): `tools/reference/compare_live.py` imports both `pyIMSRG` (old) and `imsrg` (new) in one process, builds the same inputs through both, and prints per-term max deviations; this is the debugging tool when a term test fails.

### 5.3 CI
`pytest -q -m "not slow and not data"` on every PR (CPU Serial/OpenMP backends); `slow`/`data` nightly; one GPU runner builds with the CUDA backend and runs tier 1-3 (checks host/device policy). Session-scoped `runtime` fixture; tests never call `Runtime` themselves.

## 6. Build and packaging

- **Python package build**: `pyproject.toml` with `build-backend = "scikit_build_core.build"`, `requires = ["scikit-build-core>=0.10"]` (nanobind comes from the `extern/nanobind` submodule); `pip install .` builds the C++ library as part of the wheel. `[tool.scikit-build] cmake.args = ["-DIMSRG_BUILD_PYTHON=ON"]`, `wheel.packages = ["python/imsrg"]`, `wheel.install-dir = "imsrg"`, `cmake.build-type = "Release"`, `minimum-version`, and `sdist.include` listing the third-party sources we vendor. Kokkos is found with `find_package(Kokkos)` (deps prefix or cluster module) or built from the `extern/kokkos` submodule as a fallback (no FetchContent); backend selection through `-DKokkos_ENABLE_OPENMP/CUDA/HIP` forwarded via `SKBUILD_CMAKE_ARGS`/`[tool.scikit-build.cmake.define]` and documented per cluster in `docs/install.md`. Editable dev install: `pip install -e . --no-build-isolation -Ceditable.rebuild=true`.
- **Option**: top-level CMake option `IMSRG_BUILD_PYTHON` (default `ON` per `20-build-and-tooling.md`; forced `ON` by scikit-build-core). When `OFF` the C++ library, drivers and C++ tests build with no Python/nanobind dependency.
- **Target layout** (bindings are a consumer of the public library, never part of it):
  ```
  python/
    CMakeLists.txt         # if(IMSRG_BUILD_PYTHON) find_package(Python ... Development.Module) + nanobind
    pyproject.toml (at repo root)
    src/
      module.cc            # NB_MODULE(_core, m): calls the bind_* functions
      bind_runtime.cc  bind_basis.cc  bind_operator.cc  bind_io.cc
      bind_hf.cc  bind_mbpt.cc  bind_commutators.cc  bind_terms.cc  bind_flow.cc  bind_angmom.cc
      ndarray_util.hh      # host view creation, owner capsules, residency checks
      trampolines.hh       # GeneratorFn / observer GIL trampolines
    imsrg/                 # pure Python package (+ py.typed, _core.pyi generated)
    tests/
  ```
  `nanobind_add_module(_core NB_STATIC STABLE_ABI LTO python/src/*.cc)` (stable ABI only where nanobind supports it with `nb::ndarray`: requires Python >= 3.12; otherwise per-version wheels), `target_link_libraries(_core PRIVATE imsrg::driver imsrg::operator imsrg::hf imsrg::mbpt imsrg::commutator imsrg::flow imsrg::io imsrg::basis imsrg::exec)`. **Only public library targets** (exported with `imsrg::` aliases, public headers only) are linked: no internal include directories and no `-I src/...`. Anything the bindings need that is not in a public header is a signal to add it to the public API (or to a `imsrg::testing` target), not to include private headers. `commutators::terms` is exposed from the public `imsrg::commutators_terms` target, compiled in by default but removable with `-DIMSRG_EXPOSE_TERMS=OFF` for slim wheels; the reference implementations require `-DIMSRG_BUILD_REFERENCE=ON` (on in CI, in dev wheels, off in release wheels; `terms.reference` is absent when off and tests skip).
- **Library linkage**: for wheels, the C++ libraries are static (`BUILD_SHARED_LIBS=OFF`, `-fPIC`) and linked into `_core`, so a single `.so` carries everything and avoids rpath issues; the Kokkos backend libs are linked statically as well. For development installs against a shared `libimsrg` (to avoid the old "remove .so before relink to prevent SIGBUS" hack), the `imsrg::*` targets can be shared and the module gets `INSTALL_RPATH $ORIGIN`.
- **Stubs**: `nanobind_add_stub(_core_stub MODULE imsrg._core OUTPUT python/imsrg/_core.pyi PYTHON_PATH $<TARGET_FILE_DIR:_core> DEPENDS _core)` in the build; the `.pyi` is installed into the wheel.
- **Versioning**: `imsrg.__version__` from `scikit-build-core` metadata (`dynamic = ["version"]` read from `CMakeLists.txt` project version); `imsrg.build_info()` returns git hash, compiler, Kokkos backend/arch, MPI on/off, `IMSRG_EXPOSE_TERMS`, `IMSRG_BUILD_REFERENCE` flags. Replaces old `BuildVersion`.
- **Optional extras** (`pyproject` extras): `imsrg[config]` (`tomli` for Python < 3.11), `imsrg[mpi]` (mpi4py), `imsrg[test]` (pytest, numpy, hypothesis for index-type property tests), `imsrg[dev]` (mypy, ruff).
- **CI/packaging risks recorded**: (a) wheels with CUDA/HIP are not published to PyPI (site-specific `pip install .` from source is the supported route); CPU OpenMP manylinux/macOS wheels are possible later; (b) nanobind requires C++17 and our C++20 core compiles in separate translation units from the bindings, so no C++20-only constructs may appear in public headers that `python/src` includes unless the bindings TU also builds as C++20 (we set `CXX_STANDARD 20` on `_core` too; this is the plan, MSVC not supported); (c) `nb::ndarray` zero-copy of Kokkos host mirrors requires `Kokkos::LayoutLeft` host views (F-order, matches Armadillo/old layout), asserted at compile time in `ndarray_util.hh`.

## 7. Open questions for a human decision
1. `import imsrg` is passive and Kokkos initializes lazily (2.0, 4.3); acceptable, or should `import` init with defaults (simpler for novices, hostile to GPU selection)?
2. `StaleHostError` (raise) vs implicit sync policy for host views (2.2): raise is proposed.
3. Should `Operator` expose `block()` as *writable by default* (numpy semantics expectations) with an automatic dirty mark, instead of read-only by default? Proposed: read-only default.
4. Python generator callbacks cost a host sync per step; accept as prototyping-only, or provide a Kokkos-aware GeneratorFn in C++ later?
5. Old reader formats dropped (Oslo, OakRidge, Jason, Takayuki, Antoine): confirm with users before removal.
6. Stable-ABI wheels need Python >= 3.12; confirm minimum supported Python version (proposal: 3.10 with per-version wheels, stable ABI optional).

## 8. Implementation status (M11 phase 1, 2026-11-12)

Built: `imsrg._core` with `Runtime` (lazy `ensure()`, context manager, atexit finalize through
`imsrg/__init__.py`), `Orbit`, `Channel`, `ModelSpace` (shared-pointer wrapper, identity
equality), `Operator`/`OperatorSpec` (zero-copy F-ordered numpy views of the one-body matrix and
two-body blocks, read-only unless `writable=True`; arithmetic, norms, normal ordering),
`random_operator`, operator dump read/write, `RunConfig` and `load_config` (to_toml, to_dict,
warnings), and the exception hierarchy (`ImsrgError` > `InputError` > `ConfigError`, plus
`InvariantError`, `NumericalError`, `RuntimeError`, `NotImplementedError`). The GIL is released in
arithmetic, normal ordering and dump I/O. Sources: `python/src/{module,runtime,basis,operator,config}.cc`.
Not yet bound: hf, mbpt, commutators, flow, pipeline run, angmom (later phases).

Notes: nanobind comes from the submodule; `python/CMakeLists.txt` must call `find_package(Python
... Development.Module)` itself (the include dirs must be visible in that directory scope).
Operators must be released before `Runtime` finalizes; the atexit hook runs `gc.collect()` first.
Golden basis files store pair J and Tz (not 2J, 2Tz) in `channel` lines. Tests: `tests/python`
(basis golden, H/T/V2 dump norms at 1e-10, numpy views, config examples); `tools/ci.sh` runs them.
