# 18 - Python API (nanobind bindings and the `imsrg` package)

> **Precedence:** where this document disagrees with `10-architecture.md`, `16-config-and-cli.md` or
> `20-build-and-tooling.md`, those documents win (module/target names, test locations, build options,
> dependencies via submodules). Source files use `.h`/`.cc` (Google style, D38).

Status: **as bound** (rewritten 2026-11-16 for D51 / M11-Q14 from `python/src`, `python/imsrg` and
`tests/python/test_api_surface.py`; the proposal it replaces is in the history of this file). Scope:
`imsrg._core` (nanobind extension, stable ABI) and the pure-Python package `imsrg`. Old reference:
`imsrg-devel/src/pyIMSRG.cc` (pybind11, ~150 individually exposed commutator terms, `arma::mat` as an
opaque class, no numpy, no GIL release).

Sections 1-2 give the goals and rules, 3 the module-by-module API, 4 residency, threading and runtime
policy, 5 what is not implemented (with the run-3 filler tasks), 6 testing, 7 build and packaging.

## 1. Goals, non-goals, design rules

### 1.1 Goals
1. **Scripting full pipelines.** Read interaction, build model space, HF, normal order, flow, transform
   observables, write output: the `imsrg` executable in about 15 lines (`examples/python/he4_pipeline.py`),
   or `pipeline.run(config)` for the driver itself.
2. **Operator algebra and testing.** Build operators, take commutators with chosen terms
   (`commutators.commutator(x, y, terms=...)`), BCH products and transformations, and compare with numpy
   views of the stored matrices.
3. **Prototyping** a flow with the C++ solvers driven from Python (`MagnusSolver`/`OdeSolver` with a status
   callback; a Python *generator* callback is a planned addition, section 5).

Main users are the owner's group, for pipelines (scripted runs, parameter scans, post-processing) and
operator algebra; notebooks and external users are secondary (D41).

### 1.2 Non-goals
- No Python-side physics for production use (reference implementations in numpy live in `tests/`).
- No exposure of internal caches, hash tables or scratch handling. No RPA/TDA, no atomic systems.
- No stable ABI promise for `imsrg._core`; the stable surface is the documented `imsrg.*` names.
- No pickling of operators; use `io.write_operator` / `io.read_operator` (the operator dump, section 3.3).

### 1.3 Design rules (as implemented)
- **Module-shaped API.** The Python modules follow the C++ library modules (`imsrg.io`, `imsrg.hf`,
  `imsrg.mbpt`, `imsrg.commutators`, `imsrg.flow`, `imsrg.operators`, `imsrg.pipeline`) rather than a
  dataclass tree of options (owner, M11-Q14: no binding churn). Options are small nanobind classes with
  keyword constructors (`flow.SolverOptions(smax=..., ds_0=...)`, `hf.HFOptions(...)`), mirroring the C++
  option structs; most functions also take the options as keyword arguments.
- **Names**: PEP 8. `snake_case` functions, methods, attributes and keyword arguments; classes
  `CamelCase`; enum members `CamelCase` as in the C++ enums (`GeneratorKind.Atan`, `Mixing.Diis`,
  `Term.Comm222PpHhss`). No `Get`/`Set` prefixes except the C++ mutators of the solvers
  (`set_solver_options`, `set_generator`, `set_smax`).
- **Enums, not strings** - with one convenience: the Python layer converts strings to enums at the
  entry points that scripts use (`flow.generator("shell-model-atan")`, `flow.ode_solver("flow_rk4", h0)`,
  `hf.solve(h, mixing="diis")`, `commutators.commutator(x, y, terms=["Comm110ss"])`) through
  `imsrg._util.to_enum`; TOML configs are parsed once, in C++.
- **Docstrings on everything** exposed; `test_api_surface.py` fails when a public symbol of `_core` or
  an `__all__` name of a Python module has an empty docstring. Docstrings are numpy style in the Python
  layer, short sentences with units (MeV, fm^2) in the C++ bindings.
- **No inheritance, no slicing.** `Operator` is one value type (C++ has no `Operator` hierarchy); there
  are no Python subclasses of bound types.
- **Value semantics.** `Operator.copy()`, `copy.copy` and `copy.deepcopy` are deep copies. `a + b`,
  `a - b`, `-a`, `a * c`, `c * a`, `a / c` return new operators; `+=`, `-=`, `*=`, `/=` work in place.
  Arithmetic requires the same model space (`ModelSpaceError`).
- **Lifetimes.** A `ModelSpace` is held by `std::shared_ptr`; every `Operator`, solver and result holds its
  own pointer, so a Python `ModelSpace` may be collected while operators live. A numpy view keeps the
  operator's storage alive (it remains valid after `del op`, but see finalization in 4.3).
- **Errors.** One hierarchy under `imsrg.ImsrgError` (section 3.9); standard `ValueError`/`TypeError`
  come from nanobind argument checking. No `exit()` in the core.
- **Index types.** Orbits, channels and kets are plain `int`s at the Python boundary (the strong index
  types of the C++ API are not exposed); bulk accessors return `numpy.ndarray[int32]`.

## 2. Package layout

```
imsrg/                  python/imsrg/
  __init__.py           re-exports the core names of section 3.1; residency scope helpers; the exit hook
  _core.abi3.so         nanobind extension (python/src/*.cc): all heavy code
  _util.py              to_enum(enum_cls, value)
  operators.py          operator library (re-export of the _core functions)
  io.py  hf.py  mbpt.py  commutators.py  flow.py     thin layers over _core (strings -> enums, defaults)
  pipeline.py           run(config) -> RunResult, load_flow
  __main__.py           python -m imsrg run CONFIG.toml [--set key=value ...]
  py.typed              (no generated stubs yet, section 5)
```

Extension sources: `module.cc` (`NB_MODULE`), `runtime.cc`, `residency.cc`, `basis.cc`, `operator.cc`,
`opslib.cc`, `io.cc`, `hf.cc`, `commutator.cc`, `flow.cc`, `config.cc`, and `bindings.h` (shared
declarations: `PyOperator`, `Residency`, `Releasable`, `BusyFlag`, exception translation). There is no
console-script named `imsrg` (that name belongs to the C++ executable).

## 3. The API by module

### 3.1 `imsrg` (top level)

| Name | Meaning |
|---|---|
| `Runtime` | process-wide Kokkos (and optional MPI) runtime, section 4.3 |
| `Orbit(n, l, two_j, two_tz)` | one-body orbit: `n, l, two_j, two_tz, e, is_proton, label` |
| `Channel` | two-body channel: `index, two_j, parity, two_tz, num_kets` |
| `ModelSpace(emax, reference="default", valence_space="default", hw=0.0, lmax=-1, emax_unocc=-1, lmax_unocc=-1)` | built as the imsrg++ driver does (reference nucleus name such as `"O16"`, valence-space name, `hw` in MeV); read-only: `emax, lmax, hbar_omega, num_orbits, num_kets, num_channels, reference_name`, `orbits()`, `occupations()`, `channels()`, `channel(i)`, `find_channel(two_j, parity, two_tz)`, `channel_kets(i)` and `kets()` (int32 arrays of orbit pairs) |
| `Operator(model_space, spec=OperatorSpec())` | zero operator; section 3.2 |
| `OperatorSpec(particle_rank=2, rank_J=0, parity=Even, delta_Tz=0, hermiticity=...)` | symmetry and rank labels; `scalar`, `is_scalar` |
| `Parity` (`Even`, `Odd`), `Hermiticity` (`Hermitian`, `AntiHermitian`, `NonHermitian`) | enums |
| `random_operator(model_space, spec=..., seed=1, zero_body_scale=0.0, one_body_scale=1.0, two_body_scale=1.0)` | seeded random operator for tests (library result: device-current) |
| `read_operator_dump(path, model_space=None, check_occupations=True)`, `write_operator_dump(path, op, source="imsrg", reference="")`, `read_operator_dump_header(path)` | the text dump format of the test goldens; also `io.read_operator`/`io.write_operator` |
| `load_config(text=None, file=None, set=[], check_files=False) -> RunConfig` | TOML config as in `16`; `RunConfig.to_toml(include_provenance=True)`, `to_dict()`, `warnings`, `output_directory`, `flow_file` |
| `Residency`, `ResidencyPolicy`, `residency_policy()`, `set_residency_policy()`, `residency_enforced()`, `residency_policy_scope(policy)`, `strict_residency()` | section 4.2 |
| exceptions | section 3.9 |

### 3.2 `imsrg.Operator`

- Metadata: `model_space`, `spec`, `residency`; scalars `zero_body` (read/write property, MeV).
- Numpy views of the **host mirror** (section 4.2): `one_body` (read-only `(num_orbits, num_orbits)`,
  F-ordered, zero copy), `one_body_view(writable=False)`, `two_body_block(channel, writable=False)`
  (`(num_kets, num_kets)` F-ordered J-coupled normalized antisymmetrized elements in the operator's
  stored convention). A writable view marks the host side current.
- Norms (device reductions, never need a sync): `norm()`, `one_body_norm()`, `two_body_norm()`,
  `three_body_norm()`. The zero-body part is excluded from `norm()` (imsrg++ convention, D12).
- Transformations returning new operators: `normal_ordered()` (with respect to the reference of the model
  space), `undo_normal_order()`, `copy()`; in place: `set_zero()`.
- Sync: `sync_to_host()` and `sync_to_device()` return the operator; `with op.edit() as e:` syncs to the
  host on enter and to the device on exit.
- Not on `Operator`: an HF-basis transformation (`hf.to_hf_basis`), a transformation by Omega
  (`MagnusSolver.transform`, `flow.bch_transform`).

### 3.3 `imsrg.operators`
`kinetic_energy(ms)`, `t_rel(ms, mass_number)`, `t_cm(ms, mass_number)`, `h_cm(ms, mass_number,
hbar_omega_trap=None)`, `r2_cm(ms, mass_number)`, `rp2(ms, mass_number, proton_number)`, `rn2(ms,
mass_number, proton_number)`, `rm2(ms, mass_number)`: the operator library (`opslib`), each returning an
`Operator` on `ms`. Unlike the old `OperatorFromString` there is no string front end here; names are
resolved by the driver (`16`).

### 3.4 `imsrg.io`
- `read_me2j(path, model_space, *, file_e1max, file_e2max=-1, file_lmax=-1, read_whole_file=False)
  -> Operator` (two-body interaction only, `.me2j` and `.me2j.gz`; add the kinetic term separately as the
  driver does, `h += operators.t_rel(ms, A)`); `me2j_value_count(...)`.
- `read_no2b(path, model_space, *, file_e1max, file_e2max, file_e3max, e3max, file_lmax=-1,
  precision=No2bPrecision.Single, lmax=-1) -> No2bThreeBody` (NO2B three-nucleon input, passed as `v3` to
  HF; `Half`/`Single` precision); `no2b_value_count(...)`.
- `write_tokyo(path, op, options)`, `write_nushellx_int(path, op, options)`, `write_nushellx_sps(path,
  op)` with `ValenceWriteOptions(file_2n, file_3n, e1max, e2max, e3max, hbar_omega, a_ref, z_ref,
  a_target, version, mode)` and `ValenceWriteMode`.
- `read_operator` / `write_operator`: the dump format of 3.1.
Readers and writers release the GIL. A missing file raises `InputFileNotFoundError` (also a
`FileNotFoundError`), malformed content `FileFormatError`. Readers return library results (device-current, so a script that
looks at their numpy views syncs first); writers never need a sync.

### 3.5 `imsrg.hf` and `imsrg.mbpt`
- `hf.solve(h, v3=None, *, options=None, raise_on_failure=False, **kwargs) -> HFResult`; keyword
  arguments are those of `HFOptions(mixing=Mixing.Linear, density_mixing=0.2, diis_start=2,
  diis_subspace=8, tolerance=1e-8, max_iterations=1000, freeze_occupations=False, e3max=12)` (pass
  either `options` or keywords). The Python default `mixing` is linear (the C++ and imsrg++ scheme the
  goldens use); TOML pipelines default to DIIS (D37). `raise_on_failure=True` raises `ConvergenceError`
  with the result as `exc.result`.
- `HFResult`: `e_hf` (= `e1 + e2 + e3`), `e1`, `e2`, `e3`, `spe`, `c` (HO -> HF coefficients, column-major
  copy), `rho`, `fock`, `occupations`, `iterations`, `converged`, `ediff_history`; methods
  `transform(op)` (to the HF basis, no normal ordering) and `normal_ordered_hamiltonian(h, v3=None)`
  (transform, normal order with respect to the HF reference, three-body part truncated at two-body
  level). Module-level equivalents: `hf.to_hf_basis`, `hf.normal_ordered_hamiltonian`,
  `hf.solution_in_basis(h, v3, c_ho2basis, occupations)` (mean-field state of a given basis).
- `mbpt.mp2(h)`, `mbpt.mp3(h) -> Mp3Energy(pp, hh, ph, total)`, `mbpt.natural_orbitals(h,
  model_space=None) -> NaturalOrbitals(c_hf2nat, occupations)`, `mbpt.to_nat_basis(h, c_hf2nat)`,
  `mbpt.mbpt2_density_matrix(h)`, `mbpt.reference_density_ho(c_ho2nat, model_space)`. The Hamiltonian
  must be normal ordered in the HF basis.

### 3.6 `imsrg.commutators`
- `commutator(x, y, *, terms=None, options=None, workspace=None) -> Operator`: Z = [X, Y] for scalar or
  tensor, Hermitian or general operands, rank 2 (and rank 3 with the N7 terms). `terms` is a `TermMask`
  or an iterable of `Term` (or their names); `None` is `TermMask.imsrg2()`. `CommutatorOptions(terms=...)`
  carries the same choice; a `CommutatorWorkspace(model_space)` reuses the channel tables across calls
  (a cache keyed by model space otherwise; `clear_workspace_cache()`).
- `Term` (21 enum members, `Comm110ss` ... `Comm333PppHhhss`, `ss` = scalar-scalar names of imsrg++),
  `term_name(term)`, `term_from_name(name)`; `TermMask` (`none()`, `imsrg2()`, `imsrg3_n7()`, `only(t)`,
  `of(iterable)`, `set/reset/test`, `terms`, `bits`, `empty`).
- `factorized_double_commutator(eta, gamma, *, options=FactorizedOptions(), workspace=None)` with
  `FactorizedOptions(one_body_intermediates, two_body_intermediates)`.
- Calls release the GIL. There is **no per-term function** (`terms.comm222_pp_hh(x, y)`); a single term
  is selected with `terms=[Term.Comm222PpHhss]`, which is what the term tests use (section 5).

### 3.7 `imsrg.flow`
- Generators: `GeneratorKind` (`White`, `Atan`, `Wegner`, `ImaginaryTime`, `ShellModel*`,
  `QtransferAtan`, `HartreeFock`, `OnePa`), `Partitioning` (`EpsteinNesbet`, `MollerPlesset`,
  `MpIsospin`), `GeneratorOptions(kind, partitioning, only_1b_eta, denominator_cutoff,
  denominator_delta, denominator_delta_all, denominator_delta_orbit, qtransfer_n)`; the helper
  `flow.generator(kind="atan", *, partitioning="epstein-nesbet", **kwargs)` accepts strings.
  `compute_generator(h, options) -> Operator`, `generator_function(kind, h_od, denominator,
  qtransfer_n=1)`, `floor_denominator(delta, cutoff)`, `Denominators`.
- Solvers: `SolverOptions(smax=200.0, ds_0=0.5, dsmax=0.5, domega=0.1, omega_norm_max=0.25,
  eta_criterion=1e-6, magnus_adaptive=True, hunter_gatherer=False)`.
  `MagnusSolver(h0, *, solver=SolverOptions(), generator=GeneratorOptions(), bch=..., commutator=...)`:
  `solve(on_status=None) -> SolveResult`, `transform(op)` (applies the accumulated Omega stack),
  `omegas` (copies, oldest first), `hamiltonian`, `eta`, `s`, `num_gathers`, `solver_options`,
  `generator_options`, `set_solver_options`, `set_generator`, `set_smax`. `OdeSolver(method, h0, *,
  operators=[], solver=..., ode_tolerance=..., generator=...)` (`OdeMethod.Rk4` = imsrg++ `flow_rk4`,
  `Dopri5` = `flow`; `flow.ode_solver(method, h0, **kw)` accepts the imsrg++ names): `solve(on_status)`,
  `hamiltonian`, `flowing_operator(i)`, `num_operators`, `eta`, `s`, `set_solver_options`,
  `set_generator`. The ODE solver evolves the listed operators with H; the Magnus solver transforms
  them afterwards with `transform`.
- `SolveResult(e0, norm_eta, s, steps, num_gathers, reason)` with `StopReason` (`Converged`,
  `SmaxReached`, `Diverged`).
- Status callback: `on_status(FlowStatus)` is called after every accepted step from the thread that called
  `solve` (with the GIL); `FlowStatus` holds `stage, istep, s, ds, e0, norm_eta(_1b/_2b), norm_h,
  norm_omega(_1b/_2b), num_omegas, wall_seconds`. An exception raised by the callback stops the solve at
  a step boundary and is re-raised from `solve`; the solver stays consistent. A pending Ctrl-C raises
  `KeyboardInterrupt` from `solve` at the next step. The callback's *return value* does not stop the
  flow (section 5).
- BCH: `bch_transform(op, omega, *, bch=BchOptions(), commutator=..., workspace=None)` (Z = e^Omega X
  e^-Omega), `bch_transform_with_report(...) -> (Z, BchReport)`, `bch_product(x, y, ...)`;
  `BchOptions(max_iterations, warn_iterations, transform_threshold, product_threshold, factorized,
  factorized_terms)` (`factorized=True` adds the IMSRG(3f2) correction).

### 3.8 `imsrg.pipeline`
`run(config) -> RunResult` where `config` is a `RunConfig` or the path of a TOML file; it calls the C++
driver (`driver::Run`), so there is exactly one pipeline implementation, and parses the `results.json`
the driver wrote. `RunResult` has `config`, `directory`, `results` (the parsed dict, schema in `16`
section 2.10), and the properties `e_reference`, `e_mp2`, `e_hf` (None without HF), `e0` (None without a
flow), `operators` (`{name: {"reference", "imsrg", "file"}}`), `valence_interaction`, and `flow()` (the
flow CSV as `{column: ndarray}`, also `pipeline.load_flow(path)`). `python -m imsrg run config.toml [--set
key=value]` prints `E_ref` and `E0`. Errors: `ConfigError`, `InputError`, `ConvergenceError` (the flow
diverged; `results.json` is still written). Ctrl-C while the driver runs stops the process (the driver
cannot be cancelled from inside; section 5), whereas the solvers raise `KeyboardInterrupt`.

### 3.9 Exceptions
`ImsrgError` is the root (`imsrg.ImsrgError`; derived classes live in `imsrg` and are the C++ error
categories of `base/error.h`):
`InputError` (`ConfigError`, `InputFileNotFoundError` which is also `FileNotFoundError`,
`FileFormatError`), `InvariantError`, `NumericalError` (`ConvergenceError`), `RuntimeError` (note:
`imsrg.RuntimeError`, not the builtin), `NotImplementedError`, `ModelSpaceError`, `SymmetryError`,
`StaleHostError`. Argument-type problems raise the builtin `TypeError`/`ValueError` from nanobind.

## 4. Host/device, threading and runtime policy

### 4.1 GIL and threads
- Every call that can take more than ~10 ms releases the GIL: commutators, BCH, solver `solve`/`transform`,
  HF, MBPT, all readers and writers, `pipeline.run`, operator arithmetic and normal ordering. Accessors,
  property getters and numpy-view creation hold it.
- Callbacks (`on_status`) re-acquire the GIL in the trampoline and are called from the thread that called
  `solve`, never from a Kokkos parallel region.
- Concurrent calls from two Python threads are safe when they work on **distinct** objects (the
  `ModelSpace` is immutable and shareable). The same operator or solver used from two threads at once
  raises `imsrg.RuntimeError("object in use by another thread")` (`BusyFlag`), it does not corrupt data.
  The supported production model is one Python thread driving the library with Kokkos/OpenMP inside
  (`tests/python/test_threading.py`).

### 4.2 Host/device residency
**Policy (owner, 2026-11-16, D29, D51, M11-Q11).** Views of the host mirror, explicit synchronization,
tracked staleness. An operator carries a state `Residency.HostCurrent | DeviceCurrent | Both`; library
results (`random_operator`, readers, arithmetic, `normal_ordered`, HF and flow results) are
`DeviceCurrent`. Obtaining a host view (`one_body`, `one_body_view`, `two_body_block`) of a
`DeviceCurrent` operator raises `StaleHostError("... call op.sync_to_host() first")` instead of copying
silently: an implicit device-to-host copy of a large operator inside an innocent accessor is a
performance bug that should be visible. Rules:
- `sync_to_host()` makes the state `Both`; `sync_to_device()` pushes host edits (`HostCurrent` -> `Both`);
  both return the operator and are cheap when nothing is to do.
- A read-only view never invalidates the device data; a writable view (`writable=True`, `op.edit()`)
  marks the state `HostCurrent` at once. Every compute entry point accepts any state and pushes pending
  host edits itself (`sync_to_device()`; see `Residency` in `python/src/bindings.h` for the exact
  transitions).
- `copy()` inherits the state; norms and writers use the device side and need no sync.
- Numpy views count themselves; `sync_to_host()` with views alive is safe (the host mirror is never
  reallocated).

**Enforcement.** `ResidencyPolicy.Auto` (the user default): the exception is raised only where the memory
is not host accessible (GPU builds); on Serial/OpenMP the host mirror aliases the device data, so CPU scripts
need no sync calls. `Strict` (`with imsrg.strict_residency():`, `imsrg.set_residency_policy(...)` or the
environment variable `IMSRG_RESIDENCY=strict`) raises everywhere; `Implicit` (`IMSRG_RESIDENCY=implicit`)
never raises and syncs by itself. **CI:** `tools/ci.sh` runs the whole Python suite and the example under
`IMSRG_RESIDENCY=strict`, so every test and example calls `sync_to_host()` wherever it reads a library
result, and stays portable to GPU nodes; the tests of the `Auto` and `Implicit` policies set them
explicitly with `residency_policy_scope`. `residency_enforced()` reports whether the current policy and
build raise.

### 4.3 Kokkos and MPI initialization and finalization
- `import imsrg` is passive. The runtime is created explicitly (`imsrg.Runtime(num_threads=-1,
  device_id=-1, mpi=False, print_configuration=False)`, usable as a context manager; a second explicit
  construction raises) or lazily with defaults by the first `ModelSpace`/`Operator` (`Runtime.ensure()`).
  Static helpers: `Runtime.is_initialized()`, `execution_space()` (`"OpenMP"`, `"Serial"`, `"Cuda"`, ...),
  `concurrency()`, `mpi_enabled()`.
- Finalization (`Runtime.finalize(force=False)`, `__exit__`, and an `atexit` hook that runs `gc.collect()`
  first): refused with `imsrg.RuntimeError` while a call is in flight or while operators or solvers are
  alive; `force=True` frees the device data of every live operator and solver (later use raises
  `StaleHostError`) but refuses, all or nothing, while numpy views of an operator are alive. At exit a
  surviving global view skips Kokkos finalization (process exit reclaims memory). Kokkos cannot be
  re-initialized after finalization (M11-Q12). `fork()` after initialization is unsupported (use `spawn`).
- MPI is off by default (`mpi=False`: no `MPI_Init` side effects, no mpi4py requirement); under `mpirun`
  every rank runs the same script and the collective library calls are documented as such. GPU device
  selection by local rank follows the C++ runtime.

## 5. Not implemented (and planned)

Run-3 filler tasks (owner, D51, M11-Q13/Q15):
- **A stop request from status callbacks:** `on_status` returning `True` stops the flow at a step boundary
  (needs a C++ stop flag in the solvers); today only an exception stops it.
- **Python generator callbacks** (`generator=callable(FlowStatus)->Operator`): needs a C++ generator hook
  in `flow`; built-in generators run entirely in C++.
- **A cancel flag in `driver::Run`** so that `pipeline.run` raises `KeyboardInterrupt` cleanly instead of
  Ctrl-C stopping the process.

Not planned for the current milestones (the owner's group does not need them yet; each was in the proposal):
- term-by-term functions `commutators.terms.comm222_pp_hh(x, y)` and `terms.reference` (use `terms=`);
- `ModelSpace.from_reference`/`from_orbits` builders and immutable `with_*` derivations (the constructor
  takes the driver's names); `OneBody`/`TwoBody` accessor objects, `three_body` accessors, `truncate`,
  `average_isospin`, `to_reduced`, `with_model_space` on `Operator`;
- `imsrg.angmom` (CG, 3j/6j/9j, Moshinsky), `imsrg.testing` (m-scheme brute force), `imsrg.profiler`,
  `imsrg.build_info()`;
- `hf.HFOptions(use_three_body=...)`, `mbpt` MP4/Van Vleck, tensor and dagger operators, MPI collectives
  from Python, DLPack export;
- generated type stubs (`nanobind_add_stub`): `test_api_surface.py` has a skipped test for them.
The old `pyIMSRG` -> new mapping table of the proposal (about 70 rows) is in the git history of this
file; most of its entries are covered by 3.1-3.8 or dropped project-wide (`D4` and the dropped formats).

## 6. Testing the bindings

`tests/python` (pytest, run by `tools/ci.sh` against the build tree under `IMSRG_RESIDENCY=strict`):
`test_api_surface.py` (docstrings, enum round trips, `repr`, exception hierarchy, error mapping),
`test_basis.py`, `test_operator.py` (numpy views, arithmetic, dumps), `test_residency.py` (the state
machine, force-finalize, failed constructors), `test_commutator.py`, `test_hf_mbpt.py`, `test_flow.py`
(solvers, callbacks, BCH), `test_io.py`, `test_pipeline.py`, `test_config.py`, `test_threading.py`
(GIL release, busy flags, concurrent distinct objects; timing tolerant), `test_example_pipeline.py`
(`examples/python/he4_pipeline.py`). Goldens are those of imsrg++ (`tests/golden`); big input files
come from `IMSRG_TEST_DATA_DIR` and their tests skip when absent. The extension must exist when `tools/ci.sh`
built it (`IMSRG_REQUIRE_PYTHON_EXT=1`: a missing `_core` fails instead of skipping). Physics validation
of the kernels is in the C++ tests; Python hosts the binding hygiene and the end-to-end comparisons.

## 7. Build and packaging

- **Build.** `python/CMakeLists.txt` (option `IMSRG_BUILD_PYTHON`, default ON per `20`; when OFF the C++
  library, drivers and tests build without Python or nanobind): `find_package(Python ...
  Development.Module)` in that directory scope, nanobind from the `extern/nanobind` submodule,
  `nanobind_add_module(_core NB_STATIC STABLE_ABI ...)` over `python/src/*.cc`, C++20, linking only the
  public `imsrg::` targets (no private include directories: a needed symbol that is not public is a signal
  to extend the public API). The module is `_core.abi3.so` (CMake >= 3.26; with 3.25 a version-specific
  build); configure removes stale `_core.*` files with another suffix.
- **Python version and wheels** (D29, D41): Python >= 3.12, so the stable ABI is always available;
  `pyproject.toml` (scikit-build-core, `wheel.py-api = "cp312"`) gives one `cp312-abi3` wheel per platform,
  built with cibuildwheel for Linux x86_64/aarch64 and macOS arm64 with Kokkos (OpenMP) inside; GPU and MPI
  builds stay source builds on the clusters. The C++ libraries are static in the wheel.
- **Runtime paths.** For development the extension is found through `PYTHONPATH=<build>/python`; an
  editable pip install in a shared venv is avoided because it shadows in-tree builds of other worktrees.
- `__version__` is `"0.1.0"` today; the git hash and Kokkos configuration are in the driver's
  `results.json` and the printed configuration (`Runtime(print_configuration=True)`).
- Risks recorded: CUDA/HIP wheels are not published (site-specific `pip install .`); `nb::ndarray`
  zero-copy of Kokkos host mirrors requires `LayoutLeft` host views (F-order, asserted at compile time);
  the C++20 core and the bindings translation units both build as C++20, MSVC is not supported.

## 8. Decisions and history

1. `import imsrg` is passive; explicit `Runtime()` plus lazy default (owner, 2026-11-13; Q15, D29).
2. `StaleHostError` instead of implicit sync for host views (Q16, D29); default policy `Auto`, CI under
   `Strict` (M11-Q11, D51).
3. Python >= 3.12, stable-ABI wheels (Q17, D29).
4. Module-shaped API, no dataclass tree, `opslib` bound (M11-Q14, D51); this document rewritten (D51).
5. Force-finalize frees live objects but refuses while views are alive (M11-Q12).
6. Solver observers: exception-based early exit today; stop flag, generator hook and driver cancel flag
   are run-3 filler tasks (M11-Q13, M11-Q15, D51).
Phase history: M11 phase 1 (2026-11-12) bound runtime, basis, operators, dumps, config; phase 2
(2026-11-15) residency, io, hf/mbpt, commutator, flow, opslib and the pipeline run.
