# 16. Configuration and command line (`imsrg` executable)

Status: design. Replaces `src/Parameters.hh` (four static maps, 85 keys) and the argument handling of `src/imsrg++.cc`.
References: `design/reference/03-solvers-generators-driver.md` (cited as R03, items P1-P16 and 8.2.x), `design/reference/05-operators-bindings-tests-build.md` section 1 (operator catalog, cited as R05).
Legacy source (read-only): `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel/src/{Parameters.hh,imsrg++.cc}`.

---

## 1. Goals and principles

1. **Typed config.** Every option is a field of a plain C++20 struct. Strings with a closed set of values are `enum class`; "-1 means default" sentinels become `std::optional<T>`; booleans are `bool` (old code: only the literal `"true"` was true, anything else silently false).
2. **No globals.** No static maps, no `BCH::`/`Commutator::` switches set from config. The driver builds one immutable `RunConfig`; pipeline stages receive `const RunConfig&` (or a narrow sub-struct such as `const ImsrgConfig&`) explicitly. Lower libraries never include the config header: they get their own small option structs (`SolverOptions`, `CommutatorPolicy`, ...) built from the config by the driver.
3. **One resolved object.** Sources (TOML file, legacy `key=value`, `--set`) are merged into a raw tree, defaults applied, derived values computed (effective `file_e2max`, mass number, number of steps, output stem), validation run, and the result frozen as `RunConfig`. The same object is serialised to `<output.directory>/resolved_config.toml`. Feeding that file back reproduces the run (provenance table is ignored on input).
4. **Validate everything up front, report everything at once.** All errors are collected (not first-fail) before any expensive work. Each error has: source (file:line:col, or `legacy arg 'emax=abc'`, or `--set`), key path, what was expected (type/range/allowed values), and a "did you mean" suggestion for unknown keys/enum values. Input files are checked for existence/readability in the same preflight.
5. **Unknown keys are hard errors** (TOML and legacy). Old behaviour (print "Unkown parameter", continue) hid typos.
6. **Policy for old quirks:** fix genuine bugs and document; keep behaviour that defines the numerics (Magnus step-size rule, norm conventions, BCH thresholds). Exception: the generator denominator floor keeps the sign of the denominator (D21, owner decision). Where the old code *tolerated* something harmless/no-op, the TOML schema is strict (error) but the legacy layer downgrades to a **warning** and reproduces the old numerics, so existing scripts keep running (section 3.4).
7. **Provenance.** `resolved_config.toml` is written at the start of the run (after validation, before the HF step) so failed runs still leave it. It contains all keys with defaults filled, canonical enum spellings, and a `[provenance]` table (version, git hash, date, original command line, legacy args verbatim, translation warnings, applied rules).

Terminology: "legacy layer" = `LegacyParameters`; "TOML schema" = the new canonical config.

---

## 2. TOML schema

Conventions: keys are `snake_case`; enum values are lower `snake_case` strings (legacy spellings such as `shell-model-atan`, `Epstein_Nesbet`, `NAT` are accepted **only** by the legacy layer). "req" = required, no default (the legacy layer supplies old defaults where they existed). "opt" = `std::optional`, absent means the stated derived behaviour. Phase column: P1 = implemented in first phase, PL = planned later (value parsed, validation fails with `E-PLANNED: not available in this build`), NS = not supported.

### 2.1 `[run]`

| key | type | default | meaning |
|---|---|---|---|
| `task` | enum `imsrg`, `hf`, `mp3`, `fci_export` | `imsrg` | `hf`: stop after HF/basis + normal ordering (old `method=HF`); `mp3`: stop after MBPT(3) diagnostics (old `method=MP3`); `fci_export`: vacuum HF + write FCI interaction (old `method=FCI`, PL). |
| `scratch_dir` | string | `""` | directory for Omega spill files; used only when `imsrg.omega.storage="scratch"`. Probed for writability at preflight. |
| `log_level` | enum `quiet`, `normal`, `debug` | `normal` | stdout verbosity (new key; no legacy equivalent). |

### 2.2 `[basis]` (model space and single-particle basis)

| key | type | default | meaning |
|---|---|---|---|
| `emax` | int >= 0 | req | single-particle truncation e = 2n + l. |
| `hw` | double > 0 | req | oscillator frequency (MeV). |
| `e3max` | int >= 0 | `12` | three-body truncation; effective value `min(e3max, 3*emax)`. |
| `lmax` | opt int >= 0 | none | l cut for the whole calculation. |
| `lmax3` | opt int >= 1 | none | l cut for the three-body interaction. |
| `emax_unocc` | opt int >= 1 | none | separate emax for (l,j) orbits that are unoccupied in the reference. |
| `mass_number` | opt int >= 1 | A of the reference nucleus (or core, see 5.4) | `A` used in the intrinsic kinetic energy and radius operators. |
| `kind` | enum `hf`, `oscillator`, `nat` | `hf` | basis the Hamiltonian is normal ordered in. `nat` = natural orbitals from MBPT(2) density. |

### 2.3 `[reference]` and `[valence]`

| key | type | default | meaning |
|---|---|---|---|
| `reference.nucleus` | opt string (e.g. `"O16"`) | none = core of the valence space | nucleus defining HF reference / occupations for normal ordering. |
| `reference.occupations_file` | opt path | none | per-orbit occupations (ensemble references); rows `n l 2j 2tz occ`. |
| `valence.space` | opt string | none | named valence space (`sd-shell`, `0hw-shell`, `jj-shell`, `FCI`, comma lists; names exactly as in `ModelSpace`'s table). Absent = single reference. |
| `valence.custom` | opt string | none | custom comma-separated orbit lists (core orbits then valence orbits). Mutually exclusive with `valence.space`. |

For single-reference runs set only `reference.nucleus`.

### 2.4 `[hamiltonian]` (one- and two-body terms added to the input interaction)

| key | type | default | meaning |
|---|---|---|---|
| `kinetic` | enum `intrinsic`, `lab`, `none` | `intrinsic` | `intrinsic`: add `Trel` (depends on `mass_number`); `lab`: `Tlab` only; `none`: input already contains kinetic energy (NuShellX files). |
| `trap_hw` | opt double > 0 | none | add lab-frame HO trap `1/2 M w^2 r^2` with this frequency; requires `kinetic="lab"`. |
| `beta_cm` | double | `0.0` | Lawson CM term strength `beta * HCM`; 0 = off. |
| `beta_cm_hw` | opt double > 0 | `basis.hw` | frequency in the Lawson term. Constant `-1.5*hw*beta` is subtracted at zero-body level. |
| `nucleon_mass_correction` | bool | `false` | proton-neutron mass splitting correction (`TrelMassCorrection`). |
| `relativistic_correction` | bool | `false` | p^4 relativistic kinetic correction. |

### 2.5 `[interaction.two_body]`, `[interaction.three_body]`

`[interaction.two_body]`

| key | type | default | meaning |
|---|---|---|---|
| `file` | string or array of 2 strings | req | path (oakridge: `[sp.dat, vnn.dat]`; schematic: potential name). |
| `format` | enum, see below | `me2j` | file format. |
| `e1max` | int | req | single-particle emax stored in the file. |
| `e2max` | opt int | `2*e1max` | file e2max cut. (Old default was `2*file2e1max`, not `2*emax`; see Q4.) |
| `lmax` | opt int >= 0 | `e1max` | file l cut. |
| `lecs` | opt string | none | LEC preset name; only meaningful for hdf5 formats (PL); else error if set. |

`format` values: `me2j` (P1; Darmstadt `.me2j[.gz]`), `navratil` (PL), `oslo` (PL), `oakridge` / `oakridge_bin` (PL; old `fmt2=oakridge[bin]`), `takayuki` (PL), `nushellx` (PL), `schematic_minnesota` (PL; old `fmt2=schematic 2bme=Minnesota`).

`[interaction.three_body]` (absent table = no 3N; presence of `file` makes the Hamiltonian rank 3)

| key | type | default | meaning |
|---|---|---|---|
| `file` | path | req if table present | 3N file. |
| `type` | enum `full`, `no2b`, `mono` | `full` | `full`: all 3N matrix elements (me3j); `no2b`: file already normal ordered to 2B in HO basis (stream.bin); `mono`: monopole-only (PL). |
| `format` | enum `me3j`, `navratil` (PL), (HDF5 dropped, D24) | `me3j` | file family for `type=full`. |
| `no2b_precision` | enum `single`, `half` | `single` | storage precision of a `no2b` file. |
| `e1max`, `e3max` | int | req | file e1max, e3max. |
| `e2max` | opt int | none = no extra e2 cut | file e2max cut. |
| `store_pn` | bool | `false` | keep 3N in proton-neutron instead of isospin formalism. |
| `discard_no2b_after_hf` | bool | `false` | diagnostic: use 3N in HF but drop its normal-ordered 2B contribution. |

### 2.6 `[hartree_fock]`, `[natural_orbitals]`

| key | type | default | meaning |
|---|---|---|---|
| `hartree_fock.freeze_occupations` | bool | `false` | keep reference occupations fixed (default: fractional/ensemble occupations may follow HF energies). Ignored for `basis.kind="oscillator"`. |
| `natural_orbitals.use_occupations` | bool | `false` | use NAT occupations for the reference. |
| `natural_orbitals.order` | enum `occupation`, `energy` | `occupation` | orbit labelling order. Only valid with `basis.kind="nat"`. |

### 2.7 `[imsrg]`

| key | type | default | meaning |
|---|---|---|---|
| `method` | enum `magnus`, `magnus_backoff` (PL), `magnus_modified_euler` (PL), `magnus_adaptive` (PL), `flow` (P1, DP45), `flow_euler` (PL), `flow_rk4` (P1), `restore_4th_order` (PL) | `magnus` | solver. Legacy `magnus_euler`->`magnus`, `flow_adaptive`->`flow`. |
| `steps` | opt int in {1,2} | auto: 2 if valence orbits exist, else 1 | decoupling stages. |
| `use_brueckner_bch` | bool | `false` | Brueckner-style BCH (separate 1-body then 2-body transform). |
| `goose_tank` | bool | `false` | IMSRG(2)* goose-tank correction in commutators (PL). |
| `perturbative_triples` | bool | `false` | after stage 1, add perturbative [2,2]->3 energy shift; requires `method="magnus"`. |

`[imsrg.generator]`

| key | type | default | meaning |
|---|---|---|---|
| `core` | enum generator | `atan` | generator of stage 1 (the only stage if `steps=1`). |
| `valence` | enum generator | `shell_model_atan` | generator of stage 2. |
| `qtransfer_n` | int >= 1 | `1` | exponent index for `qtransfer_atan` (`0.5*n`). |

Generator enum: `white`, `atan`, `imaginary_time`, `wegner`, `qtransfer_atan`, `shell_model`, `shell_model_atan`, `shell_model_wegner`, `shell_model_imaginary_time`, `shell_model_atan_npnh`, `hartree_fock`, `one_pa`. P1 (aligned with `13-physics-modules.md` §7): `white`, `atan`, `imaginary_time`, `wegner`, `qtransfer_atan`, `shell_model`, `shell_model_atan`, `shell_model_imaginary_time`, `hartree_fock`; the rest PL. (Legacy spellings: `shell-model-atan`, `imaginary-time`, `qtransfer-atan_<n>`, `hartree-fock`, `1PA`, ...)

`[imsrg.denominator]`

| key | type | default | meaning |
|---|---|---|---|
| `partitioning` | enum `epstein_nesbet`, `moller_plesset`, `mp_isospin` | `epstein_nesbet` | denominator definition (R03 4.5). |
| `delta` | double | `0.0` | offset added to the denominator; requires `delta_orbit`. |
| `delta_orbit` | opt string (`"all"` or an orbit label) | none | where `delta` applies; for 2-body denominators only `"all"` has effect. |

`[imsrg.step]` (step control; numerics-defining, R03 2.1)

| key | type | default | meaning |
|---|---|---|---|
| `smax` | double > 0 | `200.0` | max flow parameter per stage (Magnus: doubled for stage 2, kept; see Q12). |
| `dsmax` | double > 0 | `0.5` | max step size. |
| `ds_0` | double > 0 | `0.5` | initial step; used only by `magnus_backoff`/flow methods and as the gate of the imaginary-time/Wegner rule (Q2). |
| `domega` | double > 0 | `0.1` | max `\|\|eta ds\|\|`. |
| `omega_norm_max` | double > 0 | `0.25` | norm at which Omega is split. |
| `eta_criterion` | double > 0 | `1e-6` | convergence threshold on `\|\|eta\|\|` (old code stored this as `float`; now double). |
| `ode_tolerance` | double > 0 | `1e-6` | ODE solver tolerance (flow methods, PL). |

`[imsrg.omega]`

| key | type | default | meaning |
|---|---|---|---|
| `storage` | enum `memory`, `scratch`, `discard` | `memory` | where split Omegas live. `discard` = old `scratch=/dev/null` (only valid when no operators are requested). |
| `hunter_gatherer` | bool | `false` | merge Omega pieces into one product-form Omega. |

`[imsrg.space]` (IMSRG-internal truncation; any key present activates truncation, absent keys take defaults)

| key | type | default | meaning |
|---|---|---|---|
| `emax` | opt int | `basis.emax` | |
| `e2max` | opt int | `2*imsrg.space.emax` | |
| `e3max` | opt int | `min(basis.e3max, 3*imsrg.space.emax)` | |
| `emax_3body` | opt int | `imsrg.space.emax` | emax for 3-body operators in the IMSRG part. |

### 2.8 `[imsrg3]`

Whole table absent or all-default = IMSRG(2). Non-default keys without `in_flow` or `at_end` are errors in TOML (except `factorized`, which works with IMSRG(2) storage).

| key | type | default | meaning |
|---|---|---|---|
| `in_flow` | bool | `false` | include 3-body terms in commutators during the flow (old `IMSRG3`). |
| `at_end` | bool | `false` | after the IMSRG(2) flow, replay Omega at IMSRG(3) level (old `imsrg3_at_end`); requires `method="magnus"`. |
| `variant` | enum `full`, `n7`, `mp4` | `full` | which 3-body commutator terms (old `imsrg3_n7`, `imsrg3_mp4`; both set = error). |
| `factorized` | bool | `false` | IMSRG(3f2): factorized double-commutator correction inside the BCH transform. imsrg++ exposes it only as `BCH::SetUseFactorizedCorrection` (pyIMSRG), with no executable parameter; here it maps to `BchOptions::factorized`, not to the commutator TermMask. |
| `no_qqq` | bool | `false` | drop `qqq` terms (old `imsrg3_no_qqq`). |
| `only_2b_eta` | bool | `false` | generator stays 2-body. |
| `only_2b_omega` | bool | `false` | Omega stays 2-body. |
| `discard_residual_input_3n` | bool | `false` | drop input 3N after normal ordering (keep induced only). |
| `threebody_threshold` | double >= 0 | `0.0` | skip 3-body terms when `\|\|A\|\|` or `\|\|B\|\|` is below. |
| `de3max` | opt double | none (old `99`) | energy cut on 3-body states in generator/commutators. |
| `occ_nat3_cut` | opt double > 0 | none (old `-1`) | natural-orbital occupation cut `n(1-n)` on 3-body states; also triggers the NAT-occupation calculation (Q13). |

### 2.9 `[operators]`

| key | type | default | meaning |
|---|---|---|---|
| `list` | array of operator specs | `[]` | operators transformed with the flow. Each element is either a string in the legacy grammar (`"Rp2"`, `"rhop_3.0"`, `"OBD_..."`) or an inline table `{ name = "rhop", r = 3.0 }`; both normalise to the same `OperatorRequest` (section 4). |
| `first_order` | array of operator specs | `[]` | old `OperatorsPT1` (first-order perturbative correction of the 1-body part); PL. |
| `from_file` | array of tables | `[]` | operators read from file: `{ name, rank, dtz, parity, particle_rank, file_2b, file_3b?, format }`, `format` in `navratil`, `miyagi` (old `input_op_fmt`). PL for `rank>0`. |

Old `OperatorsRPA`/`OperatorsTDA`: not supported (no schema key).

### 2.10 `[output]`

| key | type | default | meaning |
|---|---|---|---|
| `directory` | path | `"output"` | created if missing (old code required it to pre-exist). |
| `stem` | opt string | `<method>_<reference>_<valence>_hw<hw>_e<emax>_A<A>` using resolved values, `%g` formatting | base name for interaction/operator files. |
| `flow_file` | opt path | `<directory>/<stem>.flow` | flow table. |
| `valence_file_format` | enum `tokyo`, `nushellx`, `antoine` (PL, old code marks it buggy) | `tokyo` | format of the valence-space interaction/operator files. |
| `write_omega` | bool | `false` | write Omega operators (and occupations, coefficient matrix) next to the interaction. |
| `write_ho_ops`, `write_hf_ops` | bool | `false` | write operators before the HF transformation / before the IMSRG flow (PL). |
| `spwf` | array of strings | `[]` | orbits whose HF/NAT single-particle wave functions are written; `"valence"` = all valence orbits. |

### 2.11 Example configs

(a) Single-reference IMSRG(2), O16, 2N only.

```toml
[run]
task = "imsrg"

[basis]
emax = 4
hw = 16.0
kind = "hf"

[reference]
nucleus = "O16"

[interaction.two_body]
file   = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
format = "me2j"
e1max  = 14
e2max  = 28
lmax   = 14

[imsrg]
method = "magnus"
[imsrg.generator]
core = "atan"

[operators]
list = ["Rp2"]

[output]
directory = "output"
```

(b) As (a) plus a NO2B 3N file.

```toml
[basis]
emax = 4
hw = 16.0
e3max = 12                      # <= file e3max (18)

[reference]
nucleus = "O16"

[interaction.two_body]
file  = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
e1max = 14
e2max = 28
lmax  = 14

[interaction.three_body]
file  = "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
type  = "no2b"
e1max = 6
e2max = 12
e3max = 18

[imsrg]
method = "magnus"
```

(c) Valence-space sd-shell run (two stages, `tokyo` output).

```toml
[basis]
emax = 6
hw = 20.0

[reference]
nucleus = "O16"                 # HF reference; core of sd-shell is also O16

[valence]
space = "sd-shell"

[interaction.two_body]
file  = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"   # hw of file need not equal basis.hw only if the file is hw-independent; user responsibility
e1max = 14
e2max = 28
lmax  = 14

[imsrg]
method = "magnus"
# steps omitted: auto -> 2 because valence orbits exist
[imsrg.generator]
core    = "atan"
valence = "shell_model_atan"

[operators]
list = ["Rp2"]

[output]
valence_file_format = "tokyo"
```

(d) IMSRG(3)-N7 at emax=2 (NO2B input: the 3N residual is dropped after normal ordering, induced 3-body is kept; use `type="full"` + `format="me3j"` for the full 3N).

```toml
[basis]
emax = 2
hw = 16.0
e3max = 6

[reference]
nucleus = "O16"

[interaction.two_body]
file  = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
e1max = 14
e2max = 28
lmax  = 14

[interaction.three_body]
file  = "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
type  = "no2b"
e1max = 6
e2max = 12
e3max = 18

[imsrg]
method = "magnus"

[imsrg3]
in_flow = true
variant = "n7"
threebody_threshold = 0.0
```

Equivalent legacy command for (a): `imsrg 2bme=VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz file2e1max=14 file2e2max=28 file2lmax=14 emax=4 hw=16 reference=O16 valence_space=O16 Operators=Rp2` -- must resolve to exactly the struct of (a) (tested, section 6).

### 2.12 CLI

```
imsrg --config run.toml [--set section.key=value ...] [--validate] [--print-config] [--list-operators] [--legacy-keys]
imsrg 2bme=... emax=6 hw=16 reference=O16 valence_space=O16 ...      # legacy form
```

* `--set a.b=v`: `v` parsed as a TOML value; if that fails, as a bare string (`--set imsrg.method=magnus` works). Arrays: `--set operators.list='["Rp2","E2"]'`. Applied after the file; repeated key = last wins.
* `--validate`: resolve and validate, write nothing, exit 0/2. `--print-config`: print the resolved TOML to stdout and exit.
* Legacy mode is selected when no `--config` is given and at least one positional token contains `=`. Mixing `--config` with legacy tokens is an error (`E-CLI-MIX`) so provenance stays unambiguous. Legacy `help`, `-help`, `--help` print the CLI11 help plus a pointer to `--legacy-keys` (table of old keys with new names).
* Exit codes: 0 ok, 2 configuration/validation error, 3 input file error, 1 runtime failure.
* Legacy quirks removed in the parser: `key=` (empty value) is an error rather than "keep default"; a token without `=` is an error; scalar keys given twice: last wins plus warning; vector keys given twice append (old behaviour, kept); integers must parse completely (`emax=4.5` is an error, old truncated); booleans must be `true`/`false` (old: anything else silently false).

---

## 3. Legacy mapping table

Columns: old key (type) | new key | type / enum | old default | status | notes. Status: **S** supported, **C** changed semantics (documented), **NS** not supported (legacy layer rejects with a message), **PL** planned later (accepted by the schema, fails `E-PLANNED` when requested). Rows marked (Qn) refer to the quirk list in 3.4.

### 3.1 Strings

| old key | new key | type/enum | old default | status | notes |
|---|---|---|---|---|---|
| `2bme` | `interaction.two_body.file` | path | `none` | S | `none` + `fmt2` not atomic = no input; new: file required unless `format=schematic_minnesota`. Oakridge `sp,vnn` comma form -> 2-element array. |
| `3bme` | `interaction.three_body.file` | path | `none` | S | `none` = table absent. |
| `3bme_type` | `interaction.three_body.type` | `full`,`no2b`,`mono` | `full` | S (`mono` PL) | unknown value = error (old: fell through). |
| `no2b_precision` | `interaction.three_body.no2b_precision` | `single`,`half` | `single` | S | |
| `core_generator` | `imsrg.generator.core` | generator enum | `atan` | S | hyphenated names mapped; `qtransfer-atan_<n>` -> `qtransfer_atan` + `qtransfer_n=n`. |
| `valence_generator` | `imsrg.generator.valence` | generator enum | `shell-model-atan` | S | unused when `steps=1` (Q1). |
| `flowfile` | `output.flow_file` | path | `default` | C | `default` -> unset (Q5). Explicit path used verbatim. |
| `intfile` | `output.directory` + `output.stem` | path | `default` | C | explicit `dir/stem` split at the last `/`; `default` -> unset (Q5). |
| `fmt2` | `interaction.two_body.format` | `me2j`,`navratil`,`oslo`,`oakridge`,`oakridge_bin`,`takayuki`,`nushellx`,`schematic_minnesota` | `me2j` | S (`me2j`), PL (rest) | `Navratil` capital accepted; `oakridge` substring test replaced by exact values; `fmt2=nushellx` also sets `hamiltonian.kinetic="none"` (old: skipped Trel). `fmt2=schematic` only valid with `2bme=Minnesota`. |
| `fmt3` | `interaction.three_body.format` | `me3j`,`navratil`,`hdf5` | `me3j` | S (`me3j`), PL | |
| `input_op_fmt` | `operators.from_file[i].format` | `navratil`,`miyagi` | `navratil` | PL | global old key is applied to every from-file operator. |
| `reference` | `reference.nucleus` | string | `default` | C | `default` -> absent (3.5). |
| `valence_space` | `valence.space` **or** `reference.nucleus` | string | `""` | C | if value is a known valence space (table in `ModelSpace`) -> `valence.space`; otherwise it is a nucleus -> single reference, `reference.nucleus` set if `reference` was `default` (3.5). `FCI`, `0hw-shell`, `jj-shell`, comma lists pass through. |
| `custom_valence_space` | `valence.custom` | string | `""` | S | old: overrides `valence_space` silently; legacy: same override plus warning; TOML: both set = error. |
| `basis` | `basis.kind` | `HF`,`oscillator`,`NAT` -> `hf`,`oscillator`,`nat` | `HF` | S | case-sensitive as old; others error. |
| `method` | `imsrg.method` / `run.task` / shorthands | see 3.5 | `magnus` | C | driver pseudo-methods `HF`, `MP3`, `FCI`, `NSmagnus`, `brueckner*` handled in 3.5. |
| `denominator_delta_orbit` | `imsrg.denominator.delta_orbit` | `all` or orbit label | `none` | S | `none` -> absent. |
| `LECs` | `interaction.two_body.lecs` | string | `EM2.0_2.0` | PL | only used by the hdf5 readers; legacy: default silently dropped, explicit value with a non-hdf5 format = warning "ignored". |
| `scratch` | `run.scratch_dir`, `imsrg.omega.storage` | path | `""` | C | `""` -> `storage=memory`; `/dev/null` -> `discard`; otherwise `scratch` + `scratch_dir`. |
| `use_brueckner_bch` | `imsrg.use_brueckner_bch` | bool | `false` | S | |
| `valence_file_format` | `output.valence_file_format` | `tokyo`,`nushellx`,`antoine` | `tokyo` | S (`tokyo`), PL | `antoine` PL (old code buggy, also writes NuShellX files). |
| `occ_file` | `reference.occupations_file` | path | `none` | S | `none`/`""` -> absent. |
| `goose_tank` | `imsrg.goose_tank` | bool | `false` | PL | |
| `write_omega` | `output.write_omega` | bool | `false` | C | old silently wrote nothing unless `scratch` was set (R03 8.2.9); new always writes (Q16). |
| `nucleon_mass_correction` | `hamiltonian.nucleon_mass_correction` | bool | `false` | S | |
| `hunter_gatherer` | `imsrg.omega.hunter_gatherer` | bool | `false` | S | |
| `relativistic_correction` | `hamiltonian.relativistic_correction` | bool | `false` | PL | |
| `IMSRG3` | `imsrg3.in_flow` | bool | `false` | S | |
| `imsrg3_n7` | `imsrg3.variant="n7"` | bool | `false` | S | old: no effect unless `IMSRG3=true`; legacy warns (3.4 rule L2). |
| `imsrg3_mp4` | `imsrg3.variant="mp4"` | bool | `false` | PL | both `n7` and `mp4` true = error. |
| `imsrg3_at_end` | `imsrg3.at_end` | bool | `false` | S | Q11. Requires `method` containing `magnus`. |
| `imsrg3_no_qqq` | `imsrg3.no_qqq` | bool | `false` | PL | |
| `physical_system` | -- | `nuclear`,`atomic` | `nuclear` | NS for `atomic` | `atomic` rejected: "atomic systems are not supported by imsrg"; `nuclear` accepted and dropped. |
| `freeze_occupations` | `hartree_fock.freeze_occupations` | bool | `false` | S | |
| `discard_no2b_from_3n` | `interaction.three_body.discard_no2b_after_hf` | bool | `false` | PL | diagnostic. |
| `use_NAT_occupations` | `natural_orbitals.use_occupations` | bool | `false` | S | |
| `order_NAT_by_energy` | `natural_orbitals.order="energy"` | bool | `false` | C | overrides `NAT_order` when true (old behaviour, kept in legacy only). |
| `NAT_order` | `natural_orbitals.order` | `occupation`,`energy` | `occupation` | S | |
| `store_3bme_pn` | `interaction.three_body.store_pn` | bool | `false` | S | |
| `discard_residual_input3N` | `imsrg3.discard_residual_input_3n` | bool | `false` | S | old: also forces normal-ordering rank 2 even without IMSRG3; same here. |
| `only_2b_eta` | `imsrg3.only_2b_eta` | bool | `false` | S | no-op without IMSRG(3); rule L2. |
| `only_2b_omega` | `imsrg3.only_2b_omega` | bool | `false` | S | no-op without IMSRG(3); rule L2. |
| `perturbative_triples` | `imsrg.perturbative_triples` | bool | `false` | S | runs after stage 1 (kept, P10). |
| `write_HO_ops` | `output.write_ho_ops` | bool | `false` | PL | |
| `write_HF_ops` | `output.write_hf_ops` | bool | `false` | PL | |
| `denominator_partitioning` | `imsrg.denominator.partitioning` | `Epstein_Nesbet`,`Moller_Plesset`,`MP_isospin` | `Epstein_Nesbet` | S (`mp_isospin` PL) | `Parameters.hh` comment lists two values; code also has `MP_isospin`. Invalid value = error. |

### 3.2 Doubles

| old key | new key | old default | status | notes |
|---|---|---|---|---|
| `hw` | `basis.hw` | `20.0` | S | |
| `smax` | `imsrg.step.smax` | `200.0` | S | `smax=0` = no flow (as in `goUniversal.py`); schema allows `>= 0`. |
| `dsmax` | `imsrg.step.dsmax` | `0.5` | S | see imaginary-time rule (Q2). |
| `ds_0` | `imsrg.step.ds_0` | `0.5` | C | unused by `magnus` (Q2). |
| `domega` | `imsrg.step.domega` | `0.1` | S | |
| `omega_norm_max` | `imsrg.step.omega_norm_max` | `0.25` | S | shorthands may override (3.5). |
| `ode_tolerance` | `imsrg.step.ode_tolerance` | `1e-6` | PL (flow methods) | stored as double (old `float`, Q14). |
| `denominator_delta` | `imsrg.denominator.delta` | `0` | C | silently ignored in old code without an orbit (Q3). |
| `BetaCM` | `hamiltonian.beta_cm` | `0` | S | active iff `\|BetaCM\| > 1e-6` (old threshold kept). |
| `hwBetaCM` | `hamiltonian.beta_cm_hw` | `-1` | S | negative -> absent (= basis `hw`). |
| `eta_criterion` | `imsrg.step.eta_criterion` | `1e-6` | S | |
| `hw_trap` | `hamiltonian.trap_hw` | `-1` | C | `> 0` -> `trap_hw` and `kinetic="lab"`; `<= 0` (old: `<0`) -> absent. |
| `dE3max` | `imsrg3.de3max` | `99` | S | `99` kept as value in legacy; absent in TOML = no cut. Effective only with IMSRG(3) or perturbative triples. |
| `OccNat3Cut` | `imsrg3.occ_nat3_cut` | `-1` | S | `<= 0` -> absent. `> 0` also triggers NAT occupations (Q13). |
| `threebody_threshold` | `imsrg3.threebody_threshold` | `0` | S | |

### 3.3 Integers

| old key | new key | old default | status | notes |
|---|---|---|---|---|
| `A` | `basis.mass_number` | `-1` | S | `<= 0` -> absent (old: `> 0` applied). |
| `e3max` | `basis.e3max` | `12` | S | effective value clamped to `3*emax` (no physical effect). |
| `emax` | `basis.emax` | `6` | S | |
| `lmax` | `basis.lmax` | `99999` | S | `>= emax` or `99999` -> absent. |
| `lmax3` | `basis.lmax3` | `-1` | S | applied only if `> 0` (kept). |
| `nsteps` | `imsrg.steps` | `-1` | C | `-1` -> auto; values other than 1, 2 error. Q1, Q9. |
| `file2e1max` | `interaction.two_body.e1max` | `12` | S | |
| `file2e2max` | `interaction.two_body.e2max` | `-1` | C | `<= 0` -> absent = `2*file2e1max` (Q4). |
| `file2lmax` | `interaction.two_body.lmax` | `-1` | C | `<= 0` -> absent = `file2e1max`. Explicit `0` was treated as default by old code; legacy keeps that (warning). |
| `file3e1max` | `interaction.three_body.e1max` | `12` | S | |
| `file3e2max` | `interaction.three_body.e2max` | `-1` | S | `-1` -> absent. |
| `file3e3max` | `interaction.three_body.e3max` | `12` | S | |
| `atomicZ` | -- | `-1` | NS | explicit use rejected ("atomic systems are not supported"). |
| `emax_unocc` | `basis.emax_unocc` | `-1` | S | applied only if `> 0` (kept). |
| `emax_imsrg` | `imsrg.space.emax` | `-1` | S | any of the four `*_imsrg` set -> `[imsrg.space]` present; unset ones are left absent and take the derived defaults of 2.7 (same as old P8). |
| `e2max_imsrg` | `imsrg.space.e2max` | `-1` | S | |
| `e3max_imsrg` | `imsrg.space.e3max` | `-1` | S | |
| `emax_3body_imsrg` | `imsrg.space.emax_3body` | `-1` | S | |

### 3.4 Vectors (comma-separated, repeated keys append)

| old key | new key | status | notes |
|---|---|---|---|
| `Operators` | `operators.list` | S (scalars P1, rest PL) | macros `rhop_all`, `rhon_all` (r = 0.0..10.0 step 0.2), `DaggerHF_valence`, `DaggerAlln_valence` expanded by the parser (section 4.3). |
| `OperatorsFromFile` | `operators.from_file` | PL (P1 for scalars) | tag `Name^J_T_P_R^file2b[^file3b]` parsed into a table: `J`=rank, `T`=dtz, `P`=parity, `R`=particle rank. |
| `OperatorsPT1` | `operators.first_order` | PL | |
| `OperatorsRPA` | -- | NS | "RPA-resummed operator corrections are not supported". |
| `OperatorsTDA` | -- | NS | "TDA-resummed operator corrections are not supported". |
| `SPWF` | `output.spwf` | S | `valence` expands to all valence orbits. |

### 3.5 Driver-only magic values

| legacy usage | translation | status |
|---|---|---|
| `method=HF` | `run.task="hf"` | S |
| `method=MP3` | `run.task="mp3"` | S |
| `method=FCI` | `run.task="fci_export"` | PL |
| `method=NSmagnus` | `imsrg.method="magnus"`, `imsrg.step.omega_norm_max=50000` (no Omega splitting); overrides a user `omega_norm_max` with warning | S |
| `method=brueckner` | `method="magnus"`, `use_brueckner_bch=true`, `omega_norm_max=500` | S |
| `method=brueckner1step` | as `brueckner` plus `steps=1`, `generator.core := core_generator <- valence_generator` (old overwrote `core_generator`) | S |
| `method=brueckner2` | as `brueckner` plus HF-basis rotation restart | PL |
| `method=magnus_euler` / `flow_adaptive` | `magnus` / `flow` | S / PL |
| `method=<other>` | error with did-you-mean (old: "I don't know method", continued) | -- |
| `scratch=/dev/null` | `imsrg.omega.storage="discard"`; error if operators requested (old check applied only to `method=magnus`) | S |
| `reference=default` | `reference.nucleus` absent | S |
| `valence_space=FCI`, `0hw-shell`, `jj-shell`, comma lists | `valence.space` verbatim | S (FCI PL) |
| `2bme=none` | no 2N input; only valid for `schematic`/atomic in old code | C (error otherwise) |
| `fmt2=oakridge[bin]` with `2bme=sp.dat,vnn.dat` | `format`, 2-element `file` | PL |
| `core_generator` containing `imaginary` or `wegner` | solver rule, not a config rewrite: before stage 1 and before stage 2, if the active generator is `imaginary_time`/`wegner` and `ds_0 > 1e-2`, set `ds_0=1e-4, dsmax=1e-2` (persists once triggered). Kept (Q2) | S (PL generators) |
| `Operators=Rp2` | additionally prints point-proton and charge radius with `PROTON_RCH2`, `NEUTRON_RCH2`, `DARWIN_FOLDY` constants | S (post-processing owned by the output stage) |
| `HCM_<hw>` built internally for `BetaCM` | done by the Hamiltonian-assembly stage from `hamiltonian.beta_cm*`, not via the operator list | S |
| `help`, `-help`, `--help` | CLI11 help | S |

### 3.6 Quirk decisions (Q1-Q16)

Default = keep numerics; "fix" = behaviour changed deliberately. "Legacy" column = what the legacy layer does. Items marked **HUMAN** want a decision.

| # | quirk | decision | TOML behaviour | legacy behaviour |
|---|---|---|---|---|
| Q1 | `nsteps=1` with a valence space only ever uses `core_generator`; `valence_generator` ignored although `Parameters.hh` says it is used (P1) | keep numerics, document: `steps=1` uses `generator.core` | error if `generator.valence` explicitly set while `steps=1`; warning if a valence space exists and `generator.core` is not `shell_model*` | warning W-NSTEPS1; same numerics. **HUMAN**: should `steps=1` rather use `generator.valence`? |
| Q2 | `ds_0` dead for `magnus`, yet gates the imaginary-time/Wegner override (`ds_0>1e-2` => `ds_0=1e-4, dsmax=1e-2`) (P3) | keep, document as above | accepted; documented; no error | warning only if explicitly set with `method=magnus` and no imaginary/wegner generator |
| Q3 | `denominator_delta` ignored unless `denominator_delta_orbit` set (P5) | keep numerics | error if `delta != 0` and no `delta_orbit` | warning, `delta` dropped |
| Q4 | `file2e2max=-1` means `2*file2e1max` (code), not `2*emax` as the comment says; `file2lmax<=0` means `file2e1max`; so `file2lmax=0` can't be requested | fix: optional keys, derived defaults documented; explicit `0` honoured | `lmax=0` means s-wave cut | `0` treated as default + warning |
| Q5 | default `flowfile`/`intfile` use `setprecision(0)` (`hw=16` -> `2e+01`), embed `A=-1` when unset, require existing `output/` | fix | `%g` formatting, resolved `A`, directory created | same (stem differs from old default names; explicit names honoured) |
| Q6 | silent parsing: `key=` keeps default, `int` truncates, bool not `true` -> false, no `=` token | fix | n/a (typed TOML) | errors (section 2.12) |
| Q7 | unknown method/generator/format: print and continue, producing garbage output | fix | error + suggestion | error + suggestion |
| Q8 | `scratch=/dev/null` check only for `method=magnus` | fix | `discard` + operators = error for any method | same |
| Q9 | second stage condition is the string compare `valence_space != reference` (R03 1.3) | fix | stage 2 runs iff `steps=2` and valence orbits exist | `nsteps=2` without valence orbits -> warning, stage skipped (old ran a useless stage) |
| Q10 | `perturbative_triples` runs after stage 1, only for magnus; solver flag unused | keep | requires `method="magnus"` | warning if other method (old: silently ignored) |
| Q11 | `imsrg3_at_end` replays Omega on `HNO` which has no 3N unless `IMSRG3=true` (P11) | keep numerics, document | `at_end` without `in_flow`: warning "input 3N is not included at IMSRG(3) level" | same warning. **HUMAN**: probably should normal order 3N at rank 3 whenever `at_end` is set |
| Q12 | Magnus `smax` doubled for stage 2 (flow methods not) | keep | documented | -- |
| Q13 | `OccNat3Cut>0` (or `basis=NAT`) triggers natural-orbital computation even with `basis=HF` (P16) | keep, make explicit | derived internally; `occ_nat3_cut` requires 3N-related IMSRG(3) or perturbative triples | same |
| Q14 | `eta_criterion`, `ode_tolerance` pass through `float` | fix (double); effect on thresholds ~1e-8 relative | -- | -- |
| Q15 | `NSmagnus`/`brueckner*` overwrite `omega_norm_max` | keep | n/a | override + warning if user value differs |
| Q16 | `write_omega` writes nothing without `scratch` | fix | always writes | same |

Legacy warning rules: **L1** an explicitly given key that the old code ignored in this context (e.g. `valence_generator` with `nsteps=1`) -> warning, key dropped or kept inert; **L2** an explicitly given `imsrg3_*`/`only_2b_*`/`dE3max` key while no IMSRG(3)/perturbative-triples is active -> warning, key dropped; **L3** deprecated-but-working spellings -> no warning. Warnings are printed and stored in `provenance.warnings`.

---

## 4. Operator-name grammar

The old factory `OperatorFromString` is a 480-line if/else chain, splits on `_`, indexes tokens without bounds checks, and returns an empty operator for unknown names (R05 1.1). The new parser is table driven and total.

### 4.1 Grammar

```
spec        := name ( "_" param )*
name        := [A-Za-z][A-Za-z0-9]*          // dispatch token, matched case-sensitively against the catalog
param       := number | orbit | word
number      := C locale floating point or integer, fully consumed (std::from_chars)
orbit       := orbit label as accepted by the model-space orbit parser (ModelSpace String2Index)
word        := [A-Za-z0-9+\-]+              // enumerated tokens (e.g. AA, GT, "+" and "-")
```

Because some names contain `_` internally (e.g. `R2_p1`, `Rp2Z_<Z>`, `M0nuHeavy_GT_...`) the catalog entry, not the grammar, decides how many leading tokens belong to the name: the parser looks up the longest catalog prefix of the token list, then converts the remaining tokens with the entry's parameter schema.

### 4.2 Typed result

```cpp
using OpParam = std::variant<int, double, OrbitLabel, std::string>;
struct OperatorRequest {
  OperatorKind kind;                 // enum over the catalog
  std::string  canonical_name;       // e.g. "rhop_3"
  std::vector<OpParam> params;       // validated against the catalog schema
  OperatorShape shape;               // rank J, dTz, parity, particle rank (from the catalog)
  Phase phase;                       // P1 / PL / NS
};
Expected<std::vector<OperatorRequest>, ConfigErrors> parse_operator_list(span<const OperatorSpec>, const ModelSpaceInfo&);
```

Catalog entry (static table, one row per operator family): `{ name, n_params, param_types[], shape, phase, builder_id, doc }`. The same table feeds `--list-operators`. TOML inline-table form `{ name = "rhop", r = 3.0 }` uses the parameter names of the entry (`r`, `q`, `hw`, `z`, `a`, `orbit1`, ...); both forms normalise to the same `OperatorRequest`.

### 4.3 Macros and special cases

* `rhop_all`, `rhon_all` -> `rhop_r`/`rhon_r` for r = 0.0, 0.2, ..., 10.0 (51 requests), expanded in the parser (old: driver).
* `DaggerHF_valence`, `DaggerAlln_valence` -> one request per valence orbit (needs the valence space; PL).
* `HCM` without parameter uses `basis.hw`; `HCM_<hw>` uses the explicit value.
* Duplicates in the list are an error (old: silently computed twice and overwrote output files).

### 4.4 Errors

* Unknown name: `E-OP-UNKNOWN: operator 'Rp3' is not known; did you mean 'Rp2'?` (edit distance over catalog names).
* Wrong parameter count or type: `E-OP-PARAMS: 'rhop' expects 1 parameter (r: number >= 0), got 0`, `'rhop_x': 'x' is not a number`.
* Orbit not in the model space: `E-OP-ORBIT`.
* Phase: `E-PLANNED: operator 'E2' (rank-2 tensor) is not available in this build` for PL, `E-OP-NS` for not supported (atomic `hfsNMS`, `hfsSMS`).
* Name from a removed feature (`OperatorsRPA`/`OperatorsTDA` keys) is rejected at the key level (section 3.4).

### 4.5 Catalog phasing (names from R05 1.2)

| phase | operators |
|---|---|
| **P1** (scalar, J=0, dTz=0, parity 0, rank <= 2) | kinetic/CM: `Trel`, `TCM`, `Tlab`, `TrelMassCorrection`, `HCM`, `HCM_<hw>`, `VCM_<hw>`, `R2CM`; radii: `Rp2`, `Rn2`, `Rm2`, `Rm2lab`, `Rp2Z_<Z>`, `Rp2AZ_<A>_<Z>`, `Rn2Z_<Z>`, `R2_p1`, `R2_n1`, `R2_p2`, `R2_n2`, `Rso`; monopole `E0`; number/density: `OccRef`, `OneOcc_<orb>`, `AllOcc_<orb>`, `OBD_<orb>_<orb>`, `rhop_<r>`, `rhon_<r>`, `rhop_all`, `rhon_all`, `rhocentralp_<rmax>`, `rhocentraln_<rmax>`; isospin: `Iso2`, `Tz2`; `LdotS`. |
| **PL, later scalar potentials** | `VCoul`, `VCentralCoul`, `VMinnesota`, `VGaus_<s>`, `VGausCSB_<s>`, `VBareDelta`, `VSDI_<V0>_<R>`, `VWS_<V0>_<R0>_<a0>`, `HOtrap_<hw>`, `QdotQ*`, `VQQ`, `L2rel`, `FFp_<q>`, `FFn_<q>`, `protonFBC_<nu>`, `neutronFBC_<nu>`. |
| **PL, tensor** | `E1`..`E6`, `E2int`, `nE2`, `M1`..`M5`, `M1p`, `M1n`, `M1S`, `M1L`, `Sigma`, `Sigma_p`, `Sigma_n`, `ISM`, `IVM`, `ISQ`, `IVQ`, `ISO`, `IVO`, `IVD`, `ISD_<Rms>`, `Schiff_<R>`, `Anapole`, `GamowTeller`. |
| **PL, charge changing / parity odd** | `Fermi`, `AxialCharge`, `RadialOverlap`, `VPT_<...>`. |
| **PL, dagger** | `Dagger_<orb>`, `DaggerHF_<orb>`, `DaggerAlln_<orb>`. |
| **PL, 0vbb and double GT** | `M0nu_GT_<Ecl>_<src>[_<ff>]`, `M0nu_F_*`, `M0nu_T_*`, `M0nu_C_*`, `M0nuHeavy_*`, `M0nuIntegrand_*`, `M0nuR_*`, `DGT`, `DGTR_*`, `DGTRLocal_*`. |
| **PL, dark matter** | `DMNREFT_<Op><IsoSV>_<q>_<J>`. |
| **NS** | atomic `hfsNMS`, `hfsSMS`; any `RPA`/`TDA`-resummed variant. |

The `src` parameter of the `M0nu` family is accepted by the old code but never read (R05); the new grammar keeps the token for compatibility and ignores it with a warning when `M0nu` becomes available.

---

## 5. Validation rules and cross-parameter constraints

Errors are collected; each rule has an id used in tests. TOML = strict; "legacy: warn" = downgraded per 3.6 rules.

### 5.1 Types and ranges

* V-01 all enums exact; unknown keys/tables/values = error with suggestion (Damerau-Levenshtein <= 2 or common prefix, over keys in that table / values of that enum).
* V-02 `emax >= 0`, `hw > 0`, `e3max >= 0`, step-control doubles `> 0` (`smax >= 0`), `threebody_threshold >= 0`, `steps in {1,2}`.
* V-03 required keys present (`basis.emax`, `basis.hw`, two-body `file`/`e1max`, three-body `file`/`e1max`/`e3max` when the table exists). Legacy supplies old defaults, so missing keys cannot occur there.

### 5.2 Interaction files versus model space (Q4)

* V-10 `basis.emax <= two_body.e1max` (old: silently read fewer orbits, leaving states at higher e with zero matrix elements). Error.
* V-11 effective `two_body.e2max >= 2*basis.emax`. Error.
* V-12 if `basis.lmax` set: `basis.lmax <= two_body.lmax` (effective). Error.
* V-13 3N: `min(basis.emax, basis.e3max) <= three_body.e1max`, `effective basis.e3max <= three_body.e3max`, `three_body.e2max` (if given) `>= min(2*basis.emax, basis.e3max)`. Error.
* V-14 all referenced input files exist and are readable (preflight, before any computation); `scratch_dir` and `output.directory` creatable/writable (old code probed scratch with `_this_is_a_test_delete_me`).
* Escape hatch for V-10/V-11/V-13: none in the first phase (legacy users get the same errors with the message naming the old keys, e.g. `emax=8 > file2e1max=6`).

### 5.3 Hamiltonian and basis

* V-20 `hamiltonian.trap_hw` set requires `kinetic="lab"`.
* V-21 `natural_orbitals.order`/`use_occupations` set requires `basis.kind="nat"` (TOML). Legacy: ignored with warning when `basis != NAT` (old behaviour).
* V-22 `basis.kind="oscillator"` with `hartree_fock.freeze_occupations` set: warning.
* V-23 `mass_number` default needs a resolvable nucleus (`reference.nucleus` or valence-space core); else error "set basis.mass_number".

### 5.4 Reference and valence space

* V-30 `valence.space` and `valence.custom` mutually exclusive (TOML); legacy: custom wins + warning.
* V-31 `reference.nucleus`, if given, must be a known nucleus label `<Element><A>`; `reference.nucleus` and `valence.space` may differ (targeted/ensemble normal ordering): then the final output is re-normal-ordered with respect to the core (old behaviour, driver step "renormal_order").
* V-32 `reference.occupations_file` requires a nucleus or valence space to be present.
* V-33 resolved `mass_number` and number of valence orbits drive `steps` auto (`2` iff valence orbits exist).

### 5.5 Flow

* V-40 `steps=2` requires valence orbits (TOML error; legacy warn and run single stage, Q9).
* V-41 `steps=1` + valence space + `generator.valence` explicitly set: TOML error, legacy warning (Q1). `steps=1` + valence orbits + non-`shell_model*` core generator: warning in both.
* V-42 `shell_model*` generator in a stage with no valence orbits: error (TOML), warning (legacy; old produced a degenerate decoupling).
* V-43 `delta != 0` requires `delta_orbit` (Q3). `delta_orbit` other than `all` or a valid orbit label: error.
* V-44 `imsrg.omega.storage="discard"` with non-empty `operators.*`: error. `storage="scratch"` requires `run.scratch_dir` non-empty.
* V-45 `perturbative_triples` requires `method="magnus"`.
* V-46 methods other than `magnus` are PL; additionally flow-type methods (`flow*`, `restore_4th_order`) reject IMSRG(3) (ODE algebra has no 3-body part, R03 2.3) and `brueckner2` is PL.
* V-47 `domega > omega_norm_max` is allowed (old behaviour); `|omega_norm_max - domega| < 1e-6` bump is a solver detail (R03 P4), not validated here.
* V-48 `imsrg.space`: each given key `<=` its parent (`emax <= basis.emax`, `e2max <= 2*emax_imsrg`, `e3max <= basis.e3max`, `emax_3body <= emax`).

### 5.6 IMSRG(3)

* V-50 any non-default `[imsrg3]` key other than `in_flow`/`at_end` while both are false: error (TOML), warning (legacy rule L2).
* V-51 `variant` in {`n7`,`mp4`} requires `in_flow` or `at_end`; legacy: both `imsrg3_n7` and `imsrg3_mp4` true = error.
* V-52 `at_end` requires `method="magnus"` (old: warned and ignored for other methods). `at_end` without a 3N interaction: warning (nothing to add beyond induced).
* V-53 `discard_residual_input_3n` or `three_body.type="no2b"` force normal-ordering rank 2: no error; documented. `in_flow` with `type="no2b"` is valid (induced 3-body only).
* V-54 `de3max`, `occ_nat3_cut`, `only_2b_*` only with `in_flow`/`at_end`/`perturbative_triples`.
* V-55 IMSRG(3) with `basis.e3max` smaller than 3-body states needed: warning with `CountThreeBodyStatesInsideCut`-style estimate (memory).

### 5.7 Operators and output

* V-60 operator grammar errors (4.4). Duplicate operators error.
* V-61 `output.valence_file_format` with non-valence run (single reference): ignored; the single-reference output is `.int`-less, only operator values and `flow_file` (as old). No error.
* V-62 `operators.from_file[i]`: `rank >= 0`, `dtz in {-2..2}`, `parity in {0,1}`, `particle_rank in {1,2,3}`; files exist.
* V-63 `output.stem` must not contain path separators; `output.directory` is where `resolved_config.toml` goes.

---

## 6. Implementation notes

### 6.1 Module placement

* Library `imsrg_driver` (depends on `imsrg_flow`, `imsrg_hf`, `imsrg_io`, ... downward only; nothing depends on it except the app and Python bindings):
  * `config/schema.hpp`: the `RunConfig` struct tree and enums (generator/method enums live in the solver library and are reused; config-only enums live here). No toml++ types in public headers.
  * `config/toml_reader.cpp` (toml++ private): TOML -> raw tree with source locations; unknown-key detection by walking the schema description.
  * `config/schema_table.cpp`: single static description of every key (path, type, default, enum values, docs, validator) used for reading, defaults, `--legacy-keys`/docs generation, did-you-mean candidates and the resolved-config writer, so there is exactly one place per key.
  * `config/resolve.cpp`: defaults, derived values (effective e2max/lmax, mass number, steps, output stem, `hamiltonian.kinetic`), then `validate.cpp` (rules V-xx, accumulating `ConfigErrors`).
  * `config/writer.cpp`: `RunConfig` -> `resolved_config.toml` (deterministic key order, 17-digit doubles, provenance table).
  * `legacy/legacy_parameters.cpp`: static table `{ old_key, kind (string/double/int/vector), old_default, apply(RawTree&, value, Context&) }` covering all 85 keys of `Parameters.hh`; the translator (`translate(std::span<std::string_view>) -> Expected<RawTree + warnings>`) plus the shorthand logic of 3.5 and the sentinel conversions of 3.2/3.3. No parsing of operator names here (delegated to `operators/operator_request.cpp`).
  * `operators/operator_request.cpp` and its catalog table (section 4); `--list-operators` prints it.
* App `apps/imsrg/main.cpp`: the only place that includes CLI11. It collects `--config`, `--set`, positional tokens, flags; calls `driver::load_config(sources) -> Expected<RunConfig>`; prints errors with exit code 2; calls `driver::run(const RunConfig&)`. Python bindings call the same `load_config` from a dict/TOML string, never the legacy path.
* `--set` is applied on the raw tree before resolution, so `resolved_config.toml` always reflects the effective values.

### 6.2 Error type

`struct ConfigIssue { Severity sev; std::string code; std::string key_path; std::string message; std::optional<std::string> suggestion; SourceRef where; }`; the loader returns `Expected<RunConfig, std::vector<ConfigIssue>>` plus a list of warnings on success. Codes (`E-PLANNED`, `E-OP-UNKNOWN`, `E-KEY-UNKNOWN`, `V-10`, ...) are stable and tested.

### 6.3 Test plan for the translation layer (table-driven, `tests/driver/legacy_parameters_test.cpp`)

1. **Key coverage test:** a fixed list of all 85 old key names (copied from `Parameters.hh`, checked in as test data) must each appear in the legacy table with a status; adding or removing a row without updating the list fails the test. Statuses `NS` must produce the documented message.
2. **Per-key table:** rows `{args, expected path=value | expected issue code}`: default values (`emax` absent -> 6, `hw` -> 20, `file2e1max` -> 12, `valence_generator` -> `shell_model_atan`, ...); every enum spelling (hyphen, capital) mapping to the canonical enum; sentinel conversions (`file2e2max=-1`, `lmax=99999`, `OccNat3Cut=-1`, `hwBetaCM=-1`, `hw_trap=-1`, `A=-1`); bad values (`emax=4.5`, `IMSRG3=True`, `emax=`, bare token, `method=magnus_eular` with suggestion `magnus_euler`).
3. **Shorthand rows:** `method=NSmagnus`, `brueckner`, `brueckner1step`, `brueckner2`, `HF`, `MP3`, `FCI`; `scratch=/dev/null` with and without operators; `valence_space=O16` (nucleus) versus `sd-shell`; `custom_valence_space`; `intfile=dir/stem`; `OperatorsFromFile` tag parsing; macros.
4. **Warning rules:** `nsteps=1 valence_space=sd-shell valence_generator=...` -> W-NSTEPS1; `denominator_delta=1` without orbit -> warning and key dropped; `imsrg3_n7=true` without `IMSRG3` -> L2.
5. **Rejections:** `physical_system=atomic`, `atomicZ=8`, `OperatorsRPA=...`, `OperatorsTDA=...`; unknown key `emaxx=4` with suggestion `emax`.
6. **Golden equivalence:** the legacy command of 2.11(a) must resolve to a `RunConfig` equal (field by field) to the one loaded from TOML example (a); same for (b) with `3bme=... 3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18`, (c) `valence_space=sd-shell reference=O16`, (d) `IMSRG3=true imsrg3_n7=true`. Doc snippets in this file are extracted and run as tests so the examples cannot rot.
7. **Round trip:** `RunConfig -> resolved_config.toml -> RunConfig` is the identity (property test over randomised valid configs); every enum value round-trips through its string.
8. **Validation table:** one row per rule V-xx with a minimal failing config and the expected code/message fragment; error accumulation test (three independent errors reported together).
9. **Operator parser table:** every P1 catalog name valid; wrong arity/type; unknown name with suggestion; macro expansion count (`rhop_all` -> 51); PL names produce `E-PLANNED`.
10. **CLI tests** (app level, CLI11): `--set` overrides, TOML-vs-bare-string parsing, `--config` + legacy tokens = `E-CLI-MIX`, exit codes, `--validate` writes nothing, `resolved_config.toml` present after a failed-later run.

### 6.4 Open items for a human decision

* Q1: `steps=1` with a valence space: keep "uses `generator.core`" or switch to `generator.valence` (intended meaning in `Parameters.hh`).
* Q11: `imsrg3_at_end` without `IMSRG3` drops the input 3N; fix by normal ordering at rank 3 whenever `at_end` is set?
* TOML requiredness of `basis.emax`, `basis.hw`, `file e1max` (safer) versus old defaults (6, 20.0, 12).
* Strictness split (TOML strict, legacy lenient with warnings) versus uniform strictness.
* Whether `ds_0` should be rejected under `method=magnus` once imaginary-time/Wegner generators are implemented (currently it gates the step-limit override).
