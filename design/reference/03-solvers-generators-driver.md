# imsrg++ reference analysis 03: solvers, generators, BCH, main driver

Scope: `src/imsrg++.cc`, `src/Parameters.hh`, `src/IMSRGSolver.{hh,cc}`, `src/IMSRGSolverPV.{hh,cc}`,
`src/BCH.{hh,cc}`, `src/Generator.{hh,cc}`, `src/GeneratorPV.{hh,cc}`, plus `work/compiled/*.cc`, `work/scripts/*`.
Source root: `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only). Line numbers refer to that tree.
All paths below are relative to `src/` unless noted. "Claim (unverified at runtime)" marks things I derived from reading
code only; nothing was built or run.

Terminology used by the code: orbit label `cvq` = 0 core, 1 valence, 2 q-space (`ModelSpace`). "Reference" = the
determinant (and occupations) used for normal ordering; "core" = orbits in `modelspace.core` (the decoupled
inert core); for a single-reference calculation `valence` is empty, `core == holes`, `qspace == particles`.
Kets of a two-body channel are classified `cc, vc, qc, vv, qv, qq` (`GetKetIndex_*`).

---------------------------------------------------------------------------------------------------------

## 0. One-page summary of the design-relevant facts

1. Everything is run through ONE executable driven by `key=value` command-line args stored in four static
   `std::map`s (string/double/int/vector) in `Parameters.hh`. There is no input-file format. Unknown keys are only warned about.
2. The default and only production path is `method=magnus`: Euler steps on Omega, `Omega <- BCH_Product(eta*ds, Omega)`,
   `H(s) = BCH_Transform(H_0 or H_saved, Omega)`, Omega split into a stack when `||Omega|| > omega_norm_max`.
   Operators are transformed AFTER the flow by replaying the Omega stack (in memory or from scratch files).
3. Other `method`s: `magnus_euler` (= magnus), `magnus_backoff`, `magnus_modified_euler`, `magnus_adaptive` (boost RK4 on Omega),
   `flow`/`flow_adaptive` (boost dopri5 on H), `flow_euler`, `flow_RK4` (hand-rolled), `restore_4th_order`, plus driver-level
   pseudo-methods `NSmagnus`, `brueckner*`, `HF`, `MP3`, `FCI`.
4. Two-step decoupling = same solver object, `SetGenerator(core_generator)`, Solve, then `SetGenerator(valence_generator)` (which pushes a
   NEW Omega on the stack), Solve again with `smax*2` (magnus only).
5. BCH_Transform series: nested commutators `1/i!`, termination `||nested|| < eps`, `eps` grows by factor `(i+1)`, hard max 40 iterations.
   BCH_Product: truncated Zassenhaus/BCH series using Bernoulli numbers up to k=8 with threshold 1e-4.
6. Generators: `white`, `atan` (default core), `imaginary-time`, `wegner`, `qtransfer-atan[_n]`, `shell-model*` (default valence = `shell-model-atan`),
   `hartree-fock`, `1PA`, `shell-model-atan-npnh`. Selection of decoupled blocks is done via ket classes `cc/vc/vv/qv/qq`.
7. "PV" = parity-violating. `IMSRGSolverPV`/`GeneratorPV`/`BCH_*PV` evolve a second (parity-odd, perturbative) operator alongside H,
   and are reachable ONLY from the Python bindings, never from `imsrg++.cc`.
8. Huge amount of hidden global state: `BCH::`, `Commutator::` namespace-level flags, static `IMSRGProfiler` maps, static `Parameters` maps,
   uninitialised `Generator::H` pointer, PID-named scratch files.
9. Several parameters silently do nothing or do something other than documented (Section 8).

---------------------------------------------------------------------------------------------------------

## 1. Main driver pipeline (`imsrg++.cc`)

### 1.1 Parameter parsing (`Parameters.hh`)

* `Parameters(argc, argv)` (Parameters.hh:~125-170) loops over args, splits at first `=`, looks the key up in
  `string_par`, `double_par`, `int_par`, `vec_par` in that order, parses with `istringstream >>` (so a string value cannot contain
  whitespace, an `int` given `1.5` silently reads `1`). Empty value (`key=`) keeps the default but prints it. Vector values are comma split.
  Unknown keys: prints "Unkown parameter" and continues. An arg without `=` makes `arg.substr(pos+1)` with `pos==npos` (== whole string
  after wraparound: `substr(0)`), so `var==val`; just an "unknown parameter" print.
* `help`/`-help`/`--help` prints all options and returns (`help_mode`).
* Static data members are defined (not just declared) inside the header -> only valid in a single translation unit.
* `flowfile=default`/`intfile=default` are replaced AFTER parsing by `DefaultFlowFile()` =
  `output/BCH_<method>_<reference>_<valence_space>_hw<hw %.0f>_e<emax>_A<A>` and `DefaultIntFile()` =
  `output/<method>_<reference>_<valence_space>_hw..._e..._A...`. The `output/` directory must pre-exist; nothing creates it.
  Note `std::setprecision(0)` without `std::fixed` on `hw`: 20.0 prints as `2e+01`? (default float format with precision 0 is treated as 1 significant
  digit, so `hw=20` prints `2e+01`, `hw=16` prints `2e+01`). Do not rely on default names.
* Complete defaults (Parameters.hh:44-120):

| group | key : default |
|---|---|
| files/formats | `2bme:none` `3bme:none` `3bme_type:full` (also `no2b`, `mono`) `no2b_precision:single` (`half`) `fmt2:me2j` (also `navratil`, `oslo`, `oakridge[bin]`, `takayuki`, `nushellx`, `schematic`) `fmt3:me3j` `input_op_fmt:navratil` (`miyagi`) `LECs:EM2.0_2.0` `scratch:""` `occ_file:none` `valence_file_format:tokyo` (`nushellx`, `antoine`) `flowfile/intfile:default` |
| model space | `reference:default` `valence_space:""` `custom_valence_space:""` `emax:6` `e3max:12` `lmax:99999` `lmax3:-1` `A:-1` `hw:20.0` `file2e1max:12` `file2e2max:-1` `file2lmax:-1` `file3e1max:12` `file3e2max:-1` `file3e3max:12` `emax_unocc:-1` `emax_imsrg/e2max_imsrg/e3max_imsrg/emax_3body_imsrg:-1` `atomicZ:-1` `physical_system:nuclear` (`atomic`) |
| basis | `basis:HF` (`oscillator`, `NAT`) `freeze_occupations:false` `use_NAT_occupations:false` `order_NAT_by_energy:false` `NAT_order:occupation` `discard_no2b_from_3n:false` `store_3bme_pn:false` `dE3max:99` `OccNat3Cut:-1` |
| Hamiltonian extras | `BetaCM:0` `hwBetaCM:-1` `hw_trap:-1` `nucleon_mass_correction:false` `relativistic_correction:false` |
| flow | `method:magnus` `core_generator:atan` `valence_generator:shell-model-atan` `nsteps:-1` `smax:200` `dsmax:0.5` `ds_0:0.5` `domega:0.1` `omega_norm_max:0.25` `ode_tolerance:1e-6` `eta_criterion:1e-6` `denominator_partitioning:Epstein_Nesbet` `denominator_delta:0` `denominator_delta_orbit:none` `use_brueckner_bch:false` `goose_tank:false` `hunter_gatherer:false` |
| IMSRG(3) | `IMSRG3:false` `imsrg3_n7` `imsrg3_mp4` `imsrg3_at_end` `imsrg3_no_qqq` (all false) `only_2b_eta` `only_2b_omega` `discard_residual_input3N` `threebody_threshold:0` `perturbative_triples:false` |
| operators/outputs | `Operators:` `OperatorsFromFile:` `OperatorsPT1:` `OperatorsRPA:` `OperatorsTDA:` `SPWF:` (vector) `write_omega:false` `write_HO_ops` `write_HF_ops` |

### 1.2 Step by step (line numbers in `imsrg++.cc`)

1. **Preflight** (141-158): 2bme (unless `fmt2` contains `oakridge` or is `schematic`) and 3bme must be readable (`return 1` otherwise).
   `OperatorsFromFile` tags are unpacked (160-215): `Name^J_T_P_R^file2b[^file3b]`, i.e. `J` rank, `T`=dTz, `P`=parity, `R`=particle rank. Files checked.
2. `ReadWrite rw` created; `SetLECs_preset`, `SetScratchDir(scratch)`, `Set3NFormat(fmt3)` (222-225).
3. **Method shorthands** (267-283): `NSmagnus` -> `omega_norm_max=50000; method=magnus`. Any method containing `brueckner`:
   `brueckner2` sets `brueckner_restart`; `brueckner1step` sets `nsteps=1` and `core_generator=valence_generator`; all set
   `use_brueckner_bch=true`, `omega_norm_max=500`, `method="magnus"`.
4. **Scratch directory test** (287-318): only if `method=="magnus"` AND operators were requested: `scratch=="/dev/null"` is a fatal error
   (needs Omega on disk or in memory); non-empty scratch is probed by writing/reading `_this_is_a_test_delete_me`.
5. **Model space** (323-364):
   * `custom_valence_space` non-empty -> it replaces `valence_space` (name becomes `custom`; flow/int file defaults recomputed).
   * `reference=="default"` -> `ModelSpace(emax, valence_space)`: if the string is a known valence space (`ValenceSpaces` table, e.g. `sd-shell`, `0hw-shell`, `jj-shell`)
     the reference/core is that space's core and valence its orbits; otherwise the string is a nucleus (`O16`) -> single reference, no valence.
     Else `ModelSpace(emax, reference, valence_space)` (reference nucleus for HF/normal ordering may differ from the core; core = the core of the valence space or the nucleus itself).
     Valence strings may also be `FCI` (all orbits valence), comma lists (core orbits then valence orbits), `0hw-shell`, `jj-shell` (ModelSpace.cc:224-290).
   * `SetE3max`, `SetLmax`, `SetdE3max`, `SetOccNat3Cut`, optional `SetEmaxUnocc` (only if >0), atomic: `InitSingleSpecies`, `occ_file`: `Init_occ_from_file`.
   * **`nsteps` default (358)**: `-1` -> `valence.size()>0 ? 2 : 1`.
   * `SetHbarOmega(hw)`, `SetTargetMass(A)` if `A>0` (A=-1 => A of the reference is used for the kinetic energy), `SetLmax3` if `lmax3>0`.
   * Operator-name macro expansion (390-425): `SPWF=valence`; `rhop_all`/`rhon_all` -> `rhop_0.0 ... rhop_10.0` step 0.2; `DaggerHF_valence`, `DaggerAlln_valence`.
6. **Hamiltonian** (430-557): `Hbare = Operator(ms,0,0,0, rank)` with rank 3 iff `3bme != none`.
   * 2N read (440-472): `me2j` -> `ReadBareTBME_Darmstadt(file, H, e1max,e2max,lmax)`; `navratil`; `oslo`; `oakridge` ("sp.dat,vnn.dat", optional `bin`); `takayuki`;
     `nushellx` (`ReadNuShellX_int`); `schematic` (only `Minnesota`).
   * 3N read (473-495): `3bme_type=full` -> `Read_Darmstadt_3body(file,H,e1max,e2max,e3max)` (me3j; also `fmt3` navratil/hdf5 handled in ReadWrite);
     `no2b` -> `ThreeBody.SetMode("no2b"|"no2bhalf")` then `ReadFile`; `mono` -> mode `mono`. NOTE: the `if(full)` and `else if(mono)` are in a chain broken by a plain `if`
     so `no2b` and `mono` both go through the final `else if` chain after the first `if`; harmless but the structure is `if(full){}  if(no2b){} else if(mono){}`.
   * `store_3bme_pn` -> `ThreeBody.TransformToPN()`.
   * Atomic (509-516): no 2N file: `-Z*VCentralCoulomb + VCoulomb + KineticEnergy`, rescaled by `sqrt(M_e/M_N)`, converted to Hartree.
   * **Kinetic energy / CM** (518-557):
     * `fmt2 != nushellx and not atomic and hw_trap<0` -> `Hbare += Trel_Op(modelspace)` (relative kinetic energy `(1-1/A) T` style one-body + `-p_i.p_j/(mA)` two-body; depends on `A`).
       Aborts if the one-body part has NaN.
     * `hw_trap>0` -> `+ 0.5 M hw_trap^2/(hbar c)^2 r^2` and lab-frame `KineticEnergy_Op` (no Trel).
     * `nucleon_mass_correction` -> `+Trel_Masscorrection_Op`; `relativistic_correction` -> `+KineticEnergy_RelativisticCorr`.
     * **Lawson term** `|BetaCM|>1e-6`: `hwBetaCM<0 -> basis hw`; `Hbare += BetaCM * HCM_<hwBetaCM>` (operator string `HCM_<freq>`). The constant
       `3/2 hw` is subtracted later at zero-body level: `HNO -= BetaCM*1.5*hwBetaCM` (733). Nothing is removed in the 1b/2b parts.
7. **HF / basis** (559-640): `HFMBPT hf(Hbare)` (HFMBPT derives from HartreeFock). Unless `freeze_occupations`, occupations are unfrozen (`UnFreezeOccupations`; fractional
   occupations for open-shell "ensemble" references e.g. `reference=O18` or `occ_file`). `discard_no2b_from_3n` flag. `basis!="oscillator"` -> `hf.Solve()`.
   `basis=="NAT"` or `OccNat3Cut>0` -> natural-orbital machinery: `UseNATOccupations`, `OrderNATBy(NAT_order)` (`order_NAT_by_energy` overrides), `GetNaturalOrbitals()`
   (density matrix to MBPT2, diagonalise). HF/NAT prints SPEs and wave functions (`PrintSPEandWF`). `SPWF` list -> `imsrg_util::WriteSPWaveFunctions(spwf,hf,intfile)`.
   `method=="HF"` -> print timers, return 0 (HF only; note HF energy is printed inside `HartreeFock`).
8. **IMSRG model-space truncation** (598-643): `modelspace_imsrg = modelspace` copy. If any of `emax_imsrg, e2max_imsrg, e3max_imsrg, emax_3body_imsrg != -1`:
   defaults `emax_imsrg=emax`, `e2max_imsrg=2*emax_imsrg`, `e3max_imsrg=min(e3max,3*emax_imsrg)`, `emax_3body_imsrg=emax_imsrg`; re-`Init(emax_imsrg, reference, valence_space)`,
   re-apply atomic/occ_file, rebuild holes map from the HF space's occupations via `GetOrbitIndex(n,l,j2,tz2)` and `SetReference(hole_map)` (this is how fractional
   occupations survive). Warns when a hole has `occ<1e-8` and `cvq!=1`. `hf.SetModelspaceForOutput3N(modelspace_imsrg)` so the 3N normal ordering writes directly in the truncated space.
   The truncation of 2N/1N occurs later at (926-1000).
9. **Normal ordering** (650-690): `hno_particle_rank = 2`, but `3` if `IMSRG3 && ||Hbare_3||>1e-5`; forced to 2 if `discard_residual_input3N` or `3bme_type==no2b`.
   `basis==HF`: `HNO = hf.GetNormalOrderedH(rank)` (HF transformation + NO2B w.r.t. reference). `basis==NAT`: `hf.GetNormalOrderedHNAT(rank)`.
   `basis==oscillator`: `HNO = Hbare.DoNormalOrdering()`. `HNO` is a reference to `Hbare` (the bare H is overwritten, to save memory; later `Hbare.ThreeBodyNorm()`
   prints still use the overwritten object).
10. `perturbative_triples` / `IMSRG3` banners with `CountThreeBodyStatesInsideCut()`; with IMSRG3 and rank<3, `HNO.ThreeBody.SetMode("pn"); SetParticleRank(3)` (allocates empty 3-body, "IMSRG(3) always in pn mode").
11. **Perturbative diagnostics** (733-753): `HNO -= BetaCM*1.5*hwBetaCM` then prints `Hbare 0b`, `EMP2`, `EMP2_3B`, `To 2nd order, E = ...`, `E3_pp E3_hh E3_ph EMP3`, `To 3rd order...`.
12. **Operators other than via flow** (754-852): `if method != "magnus"`: build all `Operators` via `imsrg_util::OperatorFromString(modelspace, name)`, `OperatorsPT1` (`FirstOrderCorr_1b`), `OperatorsTDA/RPA`
    (`RPA_resummed_1b`), `OperatorsFromFile` (navratil 2b current or miyagi 2b/3b). Each is made non-reduced if `J=0` with non-zero dTz/parity, transformed to the HF (`TransformToHFBasis`) or NAT
    (`TransformHOToNATBasis`) basis (not for names containing `DaggerHF`), `DoNormalOrdering()`, `<basis> expectation value <name> = ZeroBody` printed; `method==MP3`: `MP1_Eval`; `Rp2` prints point-proton and charge radius
    (`r_ch^2 = Rp2 + R_p^2 + R_n^2 (N/Z)... + DF`, `PhysConst::PROTON_RCH2` etc.).
    For `method=="magnus"` all of this is deferred until after the flow (operators are processed in Section 1.4).
13. `method=="FCI"` (854-922): undo normal ordering, `SetReference("vacuum")`, write tokyo `.snt` (and `_<op>.snt`) or NuShellX after a vacuum HF; return 0. FCI operator output is partial for tokyo.
14. `only_2b_omega` -> `BCH::SetOnly2bOmega(true)`. Truncation of HNO to `modelspace_imsrg` (926-1000); re-print perturbative energies if `emax_imsrg != emax`. `method=="MP3"` -> return.
15. **Solver setup** (1026-1081): `IMSRGSolver imsrgsolver(HNO)`; `SetReadWrite(rw)` (only copies scratch dir), `SetMethod`, `SetDenominatorPartitioning`, `SetEtaCriterion`, `GetGenerator().SetOnly2bEta`,
    `max_omega_written=500`, `SetHunterGatherer`, `SetPerturbativeTriples` (stored but unused by the solver, the driver calls `CalculatePerturbativeTriples` itself), `SetSmax`, `SetFlowFile` (truncates the file),
    `SetDs(ds_0)`, `SetDsmax`, `SetDenominatorDelta`, `SetdOmega(domega)`, `SetOmegaNormMax`, `SetODETolerance`, `SetDenominatorDeltaOrbit` if != none.
    Global flags: `BCH::SetUseBruecknerBCH`, `BCH::SetUseGooseTank` (set earlier, line 436), `Commutator::SetThreebodyThreshold`, `SetUseIMSRG3`, `SetUseIMSRG3N7`, `SetUseIMSRG3_MP4`, `SetIMSRG3Noqqq`.
    For `method` in {`flow`,`flow_RK4`} all `ops` are appended to `FlowingOps` (index 0 is H).
16. `imsrgsolver.SetGenerator(core_generator)`; if `core_generator` contains `imaginary` or `wegner` and `ds_0>1e-2`: `ds_0=1e-4`, `dsmax=1e-2` (1080-1090).
17. `imsrgsolver.Solve()` (step 1). Prints `Norm of 3-body` if IMSRG3.
18. **Perturbative triples** (1096-1109): only if `perturbative_triples && method=="magnus"`: `dE = CalculatePerturbativeTriples()`; prints `Perturbative triples: dE -> E0+dE`. NOTE this runs right after the FIRST solve
    (core decoupling), before any valence step.
19. **brueckner2 restart** (1112-1120): `C_new = hf.C * expm(-Omega(0).OneBody)`, `HNO = hf.GetNormalOrderedH(C_new)`, `SetHin(HNO)`, `s=0`, `Solve()` again.
20. **Second step** (1122-1143): `if nsteps>1 and valence_space != reference` (string compare!): `if method=="magnus" smax*=2`; `SetGenerator(valence_generator)` (starts a new Omega, resets `ds=ds_0`);
    `modelspace_imsrg.ResetFirstPass()`; same imaginary/wegner ds override; `SetSmax(smax)`; `Solve()`.
21. **IMSRG(3) at end** (1145-1186): only if `imsrg3_at_end` and method contains `magnus`: set `dE3max`, `OccNat3Cut` on `modelspace_imsrg`, `new_E3max = min(E3max, ceil(3*max(efermi_p, efermi_n)+dE3max))`,
    build `H3` with 3-body storage (`SwitchToPN_and_discard`), copy `HNO`'s 0/1/2-body parts (HNO is the *input* normal-ordered H, so H3 has NO 3N from the interaction!), `SetUseIMSRG3(true)`,
    `H_with_3 = imsrgsolver.Transform(HNO)` (replay all Omega at IMSRG(3) level), drop 3-body (`SetNumberLegs(4)`, `SetParticleRank(2)`), `FlowingOps[0] = H_with_3`.
    Non-magnus -> warning, ignored. (The 3N part of the original H is therefore only used if `HNO` itself had rank 3, i.e. IMSRG3 with 3N.)
22. **Re-normal ordering** (1221-1277): `renormal_order` true if valence space present and the NO reference (holes with occupations) differs from the core (different size or any core orbit with occ != 1 / not a hole).
    Then `HNO = FlowingOps[0]` (dropping 3N if IMSRG3), `UndoNormalOrdering()` wrt reference, `DoNormalOrderingCore()` (normal order wrt the core), `FlowingOps[0] = HNO`. (Ensemble / targeted NO.)
23. **Output** (1280-1370): see Section 5.
24. **Operator transformation for magnus** (1386-1560): see 1.4.
25. `write_omega` (1637-1700), final `Norm of 3-body`, `Hbare.PrintTimes()` (static profiler), `return 0`.

### 1.3 Single reference vs valence-space; 1-step vs 2-step

* Single-reference (`valence_space` is a nucleus or `reference == valence_space`): `nsteps=1`, solver runs with `core_generator` (default `atan`) decoupling ph. Output: `Core Energy = <E0>` (1352) and
  operator zero-body values.
* Valence space with `nsteps=2` (default): step 1 `core_generator` (atan, single-ref type, decouples the CORE ph/pp-hh using the core/(valence+q) partition), step 2 `valence_generator` (`shell-model-atan`) decouples valence from q-space and core-valence mixing.
* `nsteps=1` with a valence space: ONLY `core_generator` is ever applied (`valence_generator` is not used). Despite the comment in `Parameters.hh:27` ("valence_generator: used for valence decoupling and 1-step (also single-ref)"),
  users must set `core_generator=shell-model-atan` (or use `method=brueckner1step`, which does exactly this assignment) to get a true 1-step valence decoupling. See pitfall P1.
* Second-step guard `valence_space != reference` compares strings, not model spaces: with `reference=default` and `valence_space=O16` (single-ref) and a user-supplied `nsteps=2`, the second step still runs
  (with the `shell-model-atan` generator in a space that has no valence orbits -> generates nothing harmful but does work).

### 1.4 Operator transformation (magnus path)

After the flow(s), for each name in `Operators` (+ `OperatorsFromFile` as `<name>_FROMFILE`; `PT1/RPA/TDA` are NOT supported with magnus because they are only appended in the `method != "magnus"` branch):
1. optional `write_HO_ops`: reduce, normal-order wrt core, write tokyo tensor/`.op` files `<intfile><name>_HO_2b.snt` ... then restore;
2. make non-reduced if J=0 and (dTz != 0 or parity != 0);
3. basis change: `oscillator` or name `OccRef` -> `DoNormalOrdering()`; `HF` -> `TransformToHFBasis(op).DoNormalOrdering()`; `NAT` -> `TransformHOToNATBasis`;
4. print `   HF: <zero body>`; truncate to `modelspace_imsrg` if any `*max_imsrg` set (the condition tests `e3Max_imsrg != -1` etc. AFTER defaults were assigned, so it is true whenever truncation is on);
5. optional `write_HF_ops` (same as 1 with `_HF_` infix);
6. `op = imsrgsolver.Transform(op)` = replay Omega stack (Section 3.4); for IMSRG(3) runs the 3-body-aware BCH only if `use_imsrg3` is still on;
7. if `renormal_order`: discard 3-body residual, `UndoNormalOrdering()`, `DoNormalOrderingCore()`;
8. write (scalar -> `.snt` (tokyo, mode "op") / `.int`; odd legs -> `WriteDaggerOperator(.dag)`; tensor -> `_2b.snt` (tokyo, made reduced if J=0 & dTz/parity) or `_1b.op`+`_2b.op`).
No IMSRG-value print for magnus-transformed operators (the print is commented out); only `HF:` values and the contents of the output files. For `flow`/`flow_RK4`, operators were flowed with H
(`FlowingOps[i+1]`) and are re-normal-ordered `op.SetModelSpace(ms2); DoNormalOrdering()` (note: different re-NO procedure than magnus) and written the same way.

---------------------------------------------------------------------------------------------------------

## 2. Methods and ODE integration (`IMSRGSolver.cc`)

`Solve()` (207-246) dispatches on `method` and prints the stdout flow header if `s < 1e-4`. At the end, `if hunter_gatherer` -> `GatherOmega()`.
Solver constructor defaults (20-40): `ds=0.1, ds_max=0.5, norm_domega=0.1, omega_norm_max=2.0, eta_criterion=1e-6, smax=2.0, method="magnus_euler", magnus_adaptive=true,
max_omega_written=500, ode_e_abs=ode_e_rel=1e-6 (float)`. imsrg++ overrides smax, ds, dsmax, domega, omega_norm_max, eta_criterion, tol.

| `method` string | function | what it does |
|---|---|---|
| `magnus`, `magnus_euler` | `Solve_magnus_euler` (280) | production default; first-order step on Omega through BCH_Product, see 2.1 |
| `magnus_backoff` | `Solve_magnus_backoff` (368) | same, but ds controlled by back-off / growth heuristics and a "soft landing" phase (added by M. Heinz), see 2.2 |
| `magnus_modified_euler` | `Solve_magnus_modified_euler` (493) | midpoint-like predictor/corrector on eta: `H_temp = H + ds [eta,H]`, `eta_mid = gen(H_temp)`, `Omega <- BCH_Product(0.5 ds eta_mid, Omega)`. Uses `ds = min(domega/||eta||, domega/||eta||/(||Omega||+1e-9), ds_max)`; no eta_criterion, no hunter/gatherer, no divergence guard; runs until `s>=smax`. Note the half-factor: looks like a bug or at least unusual (a midpoint rule would use full `ds*eta_mid`; this effectively halves the step). |
| `magnus_adaptive` | `Solve_ode_magnus` (934) | `odeint::runge_kutta4` with `integrate_const` on the Omega deque, step = `ds`. `dOmega/ds = eta - 1/2[Omega,eta]` (truncated, see Sec. 2.4). Marked "doesn't seem to work so well". |
| `flow`, `flow_adaptive` | `Solve_ode_adaptive` (819) | `odeint::integrate_adaptive(make_controlled<runge_kutta_dopri5>(abs_tol,rel_tol), system, FlowingOps, s, smax, ds, monitor)` on the deque of flowing operators (H plus extra operators); `dO/ds = [eta(H), O]` for every flowing operator |
| `flow_euler` | `Solve_flow_euler` (250) | explicit Euler with FIXED `ds` (initial member `ds`, never adapted); stop when `||dH/ds|| < 1e-6` or `s>=smax` |
| `flow_RK4` | `Solve_flow_RK4` (550) | hand-rolled classical RK4 with fixed `ds = min(ds_max, smax-s)`; eta recomputed at every stage from the stage Hamiltonian; stop if `||eta||<eta_criterion`; supports goose-tank chi (IMSRG(2)*) |
| `restore_4th_order` | via `Solve_ode_adaptive`, `ode_mode="Restored"` | appends a one-body operator chi to `FlowingOps` (index 1, which collides with user operators) and integrates `dH/ds=[eta,H+chi]`, `dchi/ds` = one-body `comm221ss(eta,H)` with ph block zeroed (goose-tank mock-up in flow form) |
| anything else | prints "I don't know method" | and does nothing (the driver then continues with an unsolved H!) |

Driver-level pseudo methods (imsrg++.cc): `NSmagnus` (no split), `brueckner` / `brueckner1step` / `brueckner2` (Section 1.2 #3), `HF`, `MP3`, `FCI` (exit before solving).
`method=="flow_adaptive"` is accepted by `Solve()` but imsrg++ only registers extra operators for `flow` and `flow_RK4` (1062); with `flow_euler`/`flow_adaptive`/`magnus_*` and `ops` non-empty (non-magnus), operators are never evolved and are written un-transformed.

### 2.1 `magnus` algorithm (280-366)

```
generator.Update(H_s, Eta)                      // eta(s)
norm_domega bumped by +1e-4 if |omega_norm_max - norm_domega| < 1e-6   (avoids machine-dependent splitting)
loop istep=1.., while s < smax:
   if ||Eta|| < eta_criterion: break            // convergence
   if ||Eta|| > 1e12 or |E0|>1e9: print "!!!!" banner; H*=1.0/0.0 (inf); break
   if ||Omega.back()|| > omega_norm_max: GatherOmega() if hunter_gatherer else NewOmega(); norm_omega=0
   if magnus_adaptive:  ds = min{ domega/||eta||,  domega/||eta||/(||Omega||+1e-9),  (omega_norm_max+1e-5)/||eta||,  ds_max }
   ds = min(ds, smax - s);  s += ds
   Eta *= ds
   Omega.back() = BCH_Product(Eta, Omega.back())            // exp(Omega_new) = exp(eta ds) exp(Omega_old)
   H_s = BCH_Transform( (Omega.size()+n_omega_written < 2) ? H_0 : H_saved , Omega.back() )
   if ||eta||<1 and generator=="shell-model-atan": denominator_cutoff = 1e-6   (no-op in magnus: default cutoff is already 1e-6)
   generator.Update(H_s, Eta)
   write flow status to flowfile and stdout; Elast = E0
```
Key properties:
* `||eta||` and `||Omega||` use `Operator::Norm()` (1b+2b+3b mixed, see Section 5.3), so thresholds depend on that convention.
* `ds` is regenerated from scratch every step; initial `ds_0` is never used in magnus (only as the value reset to by `SetGenerator`, which is overwritten at the next step). `magnus_adaptive=true` can only be changed from Python.
* Step size caps: (a) `||eta ds|| <= domega`, (b) `||eta ds|| ||Omega|| <= domega`, (c) `||eta ds|| <= omega_norm_max+1e-5` (never over-fill a stack element), (d) `ds <= dsmax`. `ds<1` implicit "should never be more than 1".
  `Eta` is a single Euler (frozen eta) step but inside a BCH_Product, so unitarity of the accumulated transformation is preserved exactly (up to BCH truncation), not the Euler flow of H.
* The Omega stack: the new `Omega.back()` starts at zero; H is always recomputed from `H_saved` (H at the time the last stack element was opened) via `BCH_Transform(H_saved, Omega.back())`, so the nested-commutator series
  never sees a big Omega. `H_saved` is set in `NewOmega()`. Condition `(Omega.size()+n_omega_written) < 2` means "this is the very first Omega" -> use `*H_0` (not a copy, `H_0` is the pointer to the user's `HNO`).
* `*H_0` is the original unevolved H; it is never modified by the solver.
* Splitting (`NewOmega`, 43-79): `H_saved = FlowingOps[0]`; prints `pushing back another Omega. Omega.size = .., operator size = MB, memory usage = GB`; if `scratchdir != ""` (and not containing `/dev/null`) writes `Omega.back()` binary to
  `<scratch>/OMEGA_<pid %06d><index %03d>` (index = `n_omega_written`), and terminates via `std::terminate()` after `max_omega_written` (500) files; with scratch the stack does NOT grow (it stays size 1 and
  `n_omega_written++`); without scratch `Omega.emplace_back(Eta)` (a copy of the Eta structure) and then `Erase()` the new back.
  `/dev/null` scratch: nothing written, nothing pushed (stack stays at size 1, `n_omega_written++`): all but the last Omega are LOST (valid only if no operator is to be transformed; enforced at imsrg++.cc:292).
* `GatherOmega` (hunter-gatherer, 98-122): keeps exactly two stack entries: `Omega[0]` = gatherer, `Omega[1]` = hunter. At split: `gatherer = BCH_Product(hunter, gatherer)` (if `||hunter||>1e-12`), `hunter.Erase()`,
  `H_saved = BCH_Transform` of `*H_0` through all but the last stack entries. So the final result is a single product-form Omega (cheaper operator transformation, higher BCH_Product error). `Solve()` calls `GatherOmega()` once more
  at the end so the hunter is merged. `SetGenerator`/second step also gather (`SetGenerator` calls `GatherOmega()` if hunter_gatherer else `NewOmega()` when `||Omega.back()||>1e-6`).
  Not implemented for IMSRGSolverPV (exits).
* Divergence guard writes `FlowingOps[0] *= 1.0/0.0`, i.e. +inf everywhere, then `break`; the driver then goes on (outputs inf files).

### 2.2 `magnus_backoff` (368-491)
* Soft-landing: if `||eta|| < 20*eta_criterion` set `in_soft_landing_phase_` (no more growth).
* `while (magnus_adaptive && ds*||eta|| > pi/4) ds *= 0.5` (print "Backing off ds...").
* Split on `||Omega||>omega_norm_max` (no domega/omega cap on ds otherwise); `ds=min(ds, smax-s)`.
* After the update: if `||eta_new|| > ||eta_old||` `ds *= 0.5` ("Backing off ds because of Eta norm growth"), else if not soft-landing `ds = min(ds_max, 1.2*ds)` (`ds_max_growth_factor_`=1.2).
* This is the only method where `ds_0` and the initial member `ds` matter in magnus family. No divergence guard.

### 2.3 Boost.odeint usage (`#ifndef NO_ODE`)
* `typedef runge_kutta_dopri5<std::deque<Operator>, double, std::deque<Operator>, double, vector_space_algebra> stepper`; controlled stepper with `abs=rel=ode_tolerance` (the setter takes `float`!).
* Required algebra for `std::deque<Operator>`: global overloads `operator/` (element-wise on zero/one/two-body incl. TwoBody matrices), `operator*(double,deque)`, `operator+(deque,deque)`,
  `operator+(double,deque)` (adds a scalar to every element: odeint's eps-offset), `abs(deque)`; and specialisation `vector_space_norm_inf<std::deque<Operator>>` which returns the SUM of `Operator::Norm()` over the deque (a 2-norm sum, despite the name). Old-boost variant (`OLD_BOOST`) specialises `vector_space_reduce` instead.
  Three-body parts are not part of these element-wise ops: **the ODE (`flow`) methods do not support IMSRG(3) or 3-body operators**; the `/` operator even indexes `TwoBody.MatEl` only.
* `operator()(x, dxdt, t)` (the ODE right-hand side, 736-815) sets `s=t`, `FlowingOps[0]=x[0]`, regenerates eta from `x[0]` (`generator.Update`), returns `dxdt[i]=[eta,x[i]]` or all zeros if `||eta||<eta_criterion`
  (makes the controller free-run to `smax`), and writes a flow-status line to flowfile AND stdout for every RHS call (so dopri5's 6 stages per step each print a line; dopri5 FSAL still evaluates RHS >=6 times per accepted step, and rejected steps print too).
* The ODE monitor stores `(t, E0, ||eta_1||, ||eta_2||)` per accepted step and `report()` prints them at the end.
* `auto system = *this;` (822, 934) copies the whole solver (including all operators and the profiler reference) before handing it to odeint. State lives in `FlowingOps` (passed by reference), but `s`, `Eta`, `generator` mutations
  happen on the copy. Claim (unverified at runtime): consequently after `Solve_ode_adaptive` the original solver's `s` and `Eta` are stale (`s` still 0) and `ODE_Monitor`'s `eta1/eta2` read the ORIGINAL (never updated) `Eta`. If the flow
  is followed by a second `Solve()` (nsteps=2) the time restarts at 0 and `smax` is not doubled (that doubling is magnus-only). A rewrite should not copy this.
* Dense output / step-size control: dopri5 adaptive with initial dt = `ds` member (= `ds_0`); `dsmax` is NOT enforced for `flow`; `domega`, `omega_norm_max` unused.

### 2.4 `magnus_adaptive` (Omega ODE)
`ode_mode="Omega"`: `norm_omega>omega_norm_max` -> `NewOmega()` (inside the RHS! side effects in RK stages); `Omega.back()=x.back()`; `H_s = BCH_Transform(H_saved or H_0, Omega_s)`; `eta = gen(H_s)`;
`dOmega/ds = eta - 1/2 [Omega, eta]` (the two-term truncation of the Magnus expansion derivative; full series is `sum B_n/n! ad_Omega^n eta`). Fixed step `ds`; `integrate_const`. Experimental.

### 2.5 Convergence / termination criteria (summary)
* `s >= smax` (default 200, in magnus doubled for step 2). Note: with `domega`/`dsmax` defaults the flow typically ends by `||eta|| < eta_criterion = 1e-6` long before.
* `||eta|| < eta_criterion` (magnus, backoff, RK4; flow methods zero the RHS).
* `||dH/ds|| < 1e-6` (flow_euler only).
* Divergence: `||eta||>1e12` or `|E0|>1e9` (magnus family euler and PV only; RK4 has no guard).
* `n_omega_written > 500` -> `CleanupScratch(); std::terminate()`.
* `cumulative_error` (printed as "estim. err") is always 0: `EstimateStepError()` is only called from commented-out code and applies only to `qtransfer` generators.

### 2.6 Checkpointing / restart
There is no checkpoint/restart of the flow. Persistence exists only as: (a) flushing the Omega stack as binary operator files `OMEGA_<pid><idx>` in `scratch`, which are PID-named and deleted by `~IMSRGSolver()`
(`CleanupScratch()`) so they do not survive the process unless copied; (b) `write_omega=true` driver option that `FlushOmegaToScratch()` and copies them to `<intfile>_Omega_<i>`; then `rw.CopyFile`. A python driver can reload through `SetOmega`.
`CleanupScratch` is invoked from the destructor, which can also run on a *copy* (the `auto system = *this` copy above) -> could delete files owned by the live solver (copy shares `n_omega_written` and `scratchdir`). Claim (unverified at runtime).

---------------------------------------------------------------------------------------------------------

## 3. BCH and transformation machinery (`BCH.cc`)

All in namespace `BCH`; file-scope flags (BCH.cc:7-18): `use_goose_tank_correction, use_brueckner_bch, bch_skip_ieq1, use_factorized_correction, use_factorized_correction_BCH_Product,
use_factorized_correct_ZBterm (unused), only_2b_omega, TurnOn_comm223_231, TurnOn_comm223_232` (all false except the last two = true), thresholds `bch_transform_threshold = 1e-9`, `bch_product_threshold = 1e-4`.
imsrg++ sets only `use_brueckner_bch`, `use_goose_tank_correction`, `only_2b_omega`, `bch_skip_ieq1` (internally for triples). Factorised corrections and thresholds are Python-only.

### 3.1 `BCH_Transform(X, Omega)` = `e^Omega X e^-Omega` (`Standard_BCH_Transform`, 101-218)

```
Z = X + [Om,X] + 1/2! [Om,[Om,X]] + ...   with  Z += (1/i!) * nested_i,  nested_i = Commutator(Omega, nested_{i-1}),  nested_0 = X
```
* Only executed if `||X|| > bch_transform_threshold (1e-9)`; otherwise returns a copy of X.
* `factorial_denom /= i` each non-skipped iteration (1/i!).
* Convergence: `epsilon0 = ||X|| exp(-2||Omega||) thr/(2||Omega||)`, then `epsilon *= (i+1)` each iteration BEFORE test; stop when `||nested_i|| < epsilon`. (Comment in source: "this should probably be explained somewhere".)
  Equivalent to an a-priori bound on the remainder of the exponential series. NaN `||Omega||=0` gives division by zero: epsilon=inf -> stops after the first commutator (fine, it is zero then).
* `max_iter=40`, warn at `i==12` ("BCH_Transform not converged after 12 nested commutators") and at 40. Prints go to stdout.
* Preprocessing of OpOut: if number of legs even and particle rank <2 -> `SetParticleRank(2)`; J=0 with dTz or parity !=0 and not reduced -> `MakeReduced()`; if `Commutator::use_imsrg3` and OpOut is not already a rank-3 allocated pn-mode operator -> `ThreeBody.SetMode("pn")`, `SetParticleRank(3)`
  (printing "Allocating Three Body with pn mode"). (Comment documents a Feb-May 2022 bug where the 3N inherited from OpIn was erased.)
* Tensor operators (`rank_J>0`) print "Tensor BCH, i=.. Norm = ..." diagnostics every iteration (noisy; uses `GetOrbitIndex(0,2,5,+1)` for an unused variable, which throws/misbehaves on small spaces).
* `commutators` are whatever `Commutator::Commutator(Omega, Op)` dispatches to (scalar-scalar, scalar-tensor, dagger...). IMSRG(2) vs IMSRG(3) is decided by the global `Commutator::use_imsrg3`
  (3-body terms evaluated only if `||X||,||Y||>threebody_threshold`).
* `bch_skip_ieq1` (used by perturbative triples): iteration `i==1` is skipped via `continue`; since `factorial_denom` is updated only on non-skipped iterations and `OpNested` is not advanced at the skip, iteration 2 adds `[Om,X]/2`: i.e. the series
  `1/(i+1)! ad^i X`, as required to build `Htilde` with `W = [Omega, Htilde]_3b` (comment at 130-132, 157-160). Careful: the initial `epsilon` bookkeeping `epsilon *= i+1` is also skipped for i=1 (continue precedes it).
* Goose-tank (`use_goose_tank_correction`, IMSRG(2)* / "removing goose tanks"): maintain a one-body auxiliary `chi`; per iteration `chi = GooseTankUpdate(Omega, OpNested)` = one-body `comm221ss(Omega,OpNested)` (scalar) or
  `comm222_pp_hh_221st` (tensor), symmetrised, each element multiplied by `n_i n_j + (1-n_i)(1-n_j)`; then `OpNested.OneBody += chi_last` (chi from previous iteration). Mimics the 4th-order quadruples (disconnected) contribution.
  Also used inside `Solve_flow_RK4` as `goosetank_chi` added to H in the commutator argument.
* `use_factorized_correction`: stores `chi2` (nested commutator two steps ago) and adds `FactorizedDoubleCommutator::comm223_231/232(Omega, chi2, ..)` terms to OpNested for `i > i_min_factorized` ("factorized" IMSRG(3)-lite, B.C. He). Python-only. `use_factorized_correct_ZBterm` is declared but never read.
* Time bookkeeping: `OpIn.profiler.timer["Standard_BCH_Transform"]` (name from `__func__`).

### 3.2 `Brueckner_BCH_Transform` (240-250)
`e^{O1+O2} X e^{-O1-O2} -> e^{O2} e^{O1} X e^{-O1} e^{-O2}`: Omega split into one-body part `Omega1` (`SetParticleRank(1)`, `EraseTwoBody`) and 2-body part `Omega2` (`EraseOneBody`; note any 3-body part of Omega stays in Omega2),
two sequential Standard transforms (one-body first). Selected by `BCH_Transform` through the global `use_brueckner_bch`. Used with `omega_norm_max=500` (i.e. effectively no stack splitting) and (for `brueckner2`) the HF-basis rotation by `expm(-Omega1)`.

### 3.3 `BCH_Product(X,Y)` : Z with `e^Z = e^X e^Y` (261-352)

```
bernoulli = {1,-1/2,1/6,0,-1/30,0,1/42,0,-1/30};  factorial = {1,1,2,6,24,...}
Z = X + Y;  Nested = [Y,X];  nxy = ||Nested||
if nxy*||X|| > 1e-4:  Z += (1/12) [Nested, X]  (= [[Y,X],X]);   (+ factorized correction)
k=1;  while nxy > bch_product_threshold:
    if k<2 or k even:  Z += bernoulli[k]/factorial[k] * Nested     // k=1 -> -1/2 [Y,X] = +1/2 [X,Y]; k=2 -> +1/12 [Y,[Y,X]]; k=4 -> -1/720 ad_Y^4 X ...
    k++;  if k >= 9: break
    if 2*||Y||*nxy < threshold: break
    Nested = [Y, Nested]
    nxy = ||Nested||
```
* Assumes `X` is small (it is `eta*ds`) and `Y` the accumulated Omega: the series is in powers of ad_Y (to 8th order) with only a single `[X,[X,Y]]` term. So it is NOT a general BCH product and is not symmetric in X,Y.
* The `1/12 [[Y,X],X]` term is the same as `1/12 [X,[X,Y]]`. The `-1/24 [Y,[X,[X,Y]]]` term of the exact series is NOT included (source comment lists it, code does not).
* Thresholds: default `bch_product_threshold=1e-4` -> product-form error of order 1e-4 relative to unit norm, controlling accuracy of Omega (and of H after the transform).
* `only_2b_omega`: `Z.ThreeBody.Erase()` and commutators evaluated with `use_imsrg3=false` (restored afterwards); so Omega stays a 2b operator in IMSRG(3) runs.
* Takes non-const refs `Operator &X, Operator &Y` (non-const though not modified).
* Profile timers keyed by `__func__`.

### 3.4 How operator transformation reuses Omega (`IMSRGSolver::Transform*`, 937-1081)
* `Transform(Op)` = `Transform_Partial(Op,0)`: `OpOut=Op` (if particle rank 1 -> 2). Order: first the Omegas that were flushed to scratch (indices `n..n_omega_written-1`, read back as binary into `Operator omega(Eta)`; `ReadBinary`),
  printing "Transforming using <file>", "norm of omega", "op zero body" (lvalue overload only), then the in-memory stack `max(n-n_omega_written,0)..Omega.size()-1`; each step `OpOut = BCH_Transform(OpOut, Omega[i])`. Order = chronological, matching `H = e^{Om_n}...e^{Om_1} H_0 e^{-Om_1}...e^{-Om_n}`.
* `InverseTransform` applies `-Omega` in reverse order (`BCH_Transform(op, -omega)`), without scratch support (in-memory stack only).
* `Transform_Partial(op, n)` is the hook for "apply only the Omegas from index n" (used by the old Universal driver for re-normal ordering with extra flow).
* Because transformed operators are computed with the same truncated BCH series, `Transform(H_0)` reproduces `H_s` only up to BCH truncation + the BCH_Product error. The perturbative-triples code explicitly
  calls `Transform(*H_0)` again with `bch_skip_ieq1`.
* Scratch file location/naming is PID based, so Transform must be called in the same process as Solve.

### 3.5 IMSRG(3) variants (as seen from this layer)
* Controlled entirely by `Commutator::` globals: `use_imsrg3`, `use_imsrg3_n7` (only n^7 terms), `use_imsrg3_mp4` (only terms needed to 4th order), `imsrg3_no_qqq`, `threebody_threshold`, individual terms via
  `comm_term_on[...]` map (`comm133ss, comm330ss, comm223ss, comm232ss, comm331ss, comm231ss, comm132ss, comm233_pp_hhss, ...`), plus tensors via `SetUseIMSRG3_Tensor` and `...N7_Tensor`.
  `CommutatorScalarScalar` (Commutator.cc:322) allocates Z with particle rank 3 and pn mode when `use_imsrg3`, evaluates 0-2 body terms always and 3-body terms if both operators exceed the threshold.
* `perturbative triples`: `Commutator::perturbative_triples=true` makes `comm223ss` accumulate only a zero-body energy `sum |W|^2/Delta` instead of storing 3N.
* `imsrg3_at_end`: replay existing Omega at IMSRG(3) level once (Sec. 1.2 #21).
* `only_2b_eta` (generator keeps eta 2-body), `only_2b_omega` (BCH_Product drops 3-body part) are the two knobs for "IMSRG(3) in H but 2b generator".
* IMSRG(3) with scratch: the Omega files are written with `WriteBinary`; 3-body part of Omega (if not `only_2b_omega`) is included.

### 3.6 Perturbative triples (`CalculatePerturbativeTriples`, 1172-1214)
`omega = product(Omega stack)` through `BCH_Product` (i.e. NOT a sum), `Htilde = Transform(*H_0)` with `bch_skip_ieq1` on (so `Htilde = sum_{i>=1} ad^i/(i+1)! H`... i.e. first nested term at i=2 weight 1/2 of `[Om,H]`),
`Wbar` is a 3-body-rank operator constructed with `H_s` one and two-body (for denominators), `modelspace.SetE3max(3*Emax)` temporarily (so all 3-body states are included; the 3-body storage is never allocated),
`comm223ss(omega, Htilde, Wbar)` with `perturbative_triples=true`; returns `Wbar.ZeroBody`. The variant taking an operator computes the corresponding correction for `Op_0` (used by Python). Commented-out "doubling" for `[O,[O,H]_3]_3`.
This uses the whole product-form omega including its 3-body part if `IMSRG3` (no explicit check).

---------------------------------------------------------------------------------------------------------

## 4. Generators (`Generator.cc`, `GeneratorPV.cc`)

### 4.1 Interface
`Generator` fields: `generator_type` (default "white" in the ctor; imsrg++ always sets one), `denominator_cutoff` (ctor default 1e-6), `denominator_delta` (0), `denominator_delta_index` (-1),
`denominator_partitioning` (Epstein_Nesbet), `only_1b_eta`, `only_2b_eta`, `use_isospin_averaging` (default false, Python-only; replaces eta by `UndoNO -> DoIsospinAveraging -> DoNO` of eta).
Raw pointers `H, H_denom, Eta` (set in `AddToEta`, uninitialised before).
* `Update(H_s, Eta_s)` = `Eta.Erase(); AddToEta(H_s,H_s,Eta_s)` (denominators from the same H). `UpdateGeneral(H_s,H_d,Eta)` uses `H_d` for denominators (used by HFMBPT Van Vleck code).
  `AddToEta` (public) can accumulate on top of an existing eta (used by `magnus_modified_euler`).
* Eta structure comes from the Eta operator's allocated blocks (loops `for iter in Eta->TwoBody.MatEl`): the 2-body blocks that exist in `Eta` determine channel pairs (for tensor operators `ch_bra != ch_ket`).
* `GetHod(H)` returns the off-diagonal part ("what the generator is trying to kill") for `wegner, white, atan, imaginary-time, qtransfer-atan` (single-ref blocks) and `shell-model, shell-model-atan, shell-model-wegner, shell-model-imaginary-time` (shell-model blocks);
  for others prints and returns zero. Used by HFMBPT Van Vleck.

### 4.2 Catalog of generator types (`AddToEta`, 45-76)

Functions `eta = f(H_od, Delta)`:
| name | f | notes |
|---|---|---|
| `wegner_func` | `H_od * Delta` | "never tested, probably doesn't work" (comment). Note: this is not Wegner's canonical eta=[H_d,H_od] either: here eta_ij = H_ij (e_i-e_j) (the diag-1b version) |
| `white_func` | `H_od / Delta` | Moller-Plesset/EN White; stiff for small Delta |
| `atan_func` | `0.5 atan(2 H_od / Delta)` | default for core; bounded; the common production choice |
| `imaginarytime_func` | `H_od * sign(Delta)` (`(Delta>0)-(Delta<0)`; 0 at exactly 0, but Delta is floored positive) | imaginary-time generator |
| `qtransferatan1_func` | `(|Delta| M_N/hbarc^2)^{1/2} atan(...)` | momentum-transfer weighted; `qtransfer-atan_n` -> exponent `0.5*n`, `n` parsed from the substring after the first `_` |

Names -> `AddToEta` dispatch:
| `generator_type` | construct | blocks |
|---|---|---|
| `wegner`, `white`, `atan`, `imaginary-time`, `qtransfer-atan`, `qtransfer-atan_<n>` | `ConstructGenerator_SingleRef(f)` | core vs (valence+q) |
| `shell-model-wegner`, `shell-model`, `shell-model-atan`, `shell-model-imaginary-time` | `ConstructGenerator_ShellModel(f)` | valence-space decoupling |
| `shell-model-atan-npnh` | `ConstructGenerator_ShellModel_NpNh` | shell-model-atan + extra |
| `hartree-fock` | `ConstructGenerator_HartreeFock` | full one-body eta |
| `1PA` | `ConstructGenerator_1PA(atan)` | shell-model-like, particle-number-adjusted? (one-body and valence-core terms, no `vv|qq` second part) |
| anything else | prints "Error. Unkown generator_type" | eta = 0 (so `||eta||<criterion` and flow stops immediately) |

Note on dispatch order: the generic `qtransfer-atan_n` branch is last and `generator_type.find("qtransfer-atan")` also matches `qtransfer-atan` itself, but the exact `==` branch comes earlier.

### 4.3 Single-reference blocks (`ConstructGenerator_SingleRef`, 214-290)
* One-body: `a in core`, `i in valence U qspace` (cvq>0): `eta(i,a) = f(H(i,a), Delta_1b(i,a))`, `eta(a,i) = f(H(a,i), -Delta_1b(i,a))` (the second line instead of `-eta(i,a)` so that non-scalar (reduced/tensor) operators get the right phase). I.e. decouple p-h (ph with respect to the CORE, so valence orbits count as particles in step 1).
* `only_1b_eta` -> return here.
* Two-body: for every stored channel pair (bra,ket): `iket in cc` kets (both orbits core), `ibra in qq U vv U qv` (no core, i.e. both orbits non-core): `eta(ibra,iket) = f(H(ibra,iket), Delta_2b)` and, if `ch_bra==ch_ket`, `eta(iket,ibra) = -eta(ibra,iket)`.
  If `ch_bra != ch_ket` (tensor operators) also the converse block (`ibra in cc`, `iket in qq U vv U qv`) is filled, without antisymmetrisation (only one ordering is stored).
  So the blocks: **pp'|hh' with all of {qq,vv,qv} on the particle side, cc on the hole side**. `vc`/`qc` kets (one core orbit) are not decoupled in this step (those are the 2-body ph-ish parts that get handled via the one-body piece and induced).
* Three-body (only if `eta` and `H` have particle rank>2 and not `only_2b_eta`): `ConstructGenerator_SingleRef_3body`: for all (ch_bra, ch_ket, ibra) triples, in parallel (`#pragma omp parallel for schedule(dynamic,1)`), only `ppp|hhh` and `hhh|ppp` blocks
  where 'p' means `cvq>0` and 'h' means `cvq==0` (all three orbits); for `ch_bra==ch_ket` bra with any core orbit skipped; cuts: `sum_i |2n+l - e_fermi(tz)| > dE3max` skip, `n_a(1-n_a) n_b(1-n_b) n_c(1-n_c) < OccNat3Cut` skip (uses `occ_nat`). Denominator `Get3bDenominator` (see below).
  `Eta->ThreeBody.SetME_pn_ch(...)` (Hermitian conjugate is set automatically). Requires pn-mode three-body storage.

### 4.4 Shell-model blocks (`ConstructGenerator_ShellModel`, 295-380)
* One-body: `a in core U valence`, `i in valence U qspace`, `i != a`: `eta(i,a)=f(H(i,a),Delta)`, `eta(a,i)=f(H(a,i),-Delta)`. This removes core-valence, q-valence, q-core one-body mixing AND off-diagonal elements within the valence space (`i,a` both valence) so the one-body valence Hamiltonian becomes diagonal (comment: "make sure the valence one-body part is diagonal").
* Two-body: (bra, ket) kets in
  * bra `vv U qv U qq`, ket `cc U vc`  (decouples core and core-valence-pair excitations from valence/q-pair states), plus
  * bra `qv U qq`, ket `vv`  (decouples valence-pair states from q-space: "eliminate ppvh and pqvv").
  Antisymmetric partner set for `ch_bra==ch_ket`; converse blocks for `ch_bra!=ch_ket`.
  Not decoupled: `qc` kets (q-core) and `vv|vv`, `vv|vc`... `vc`/`cc` kets as bras. `vv|vv` stays (this is the effective valence Hamiltonian).
* Three-body (`ConstructGenerator_ShellModel_3body`, 520-): requires `||H_denom 1b|| > 1e-6` else warns. Blocks: `ppp|ccc, ppp|ccv, ppp|cvv, ppp|vvv` (`bra all cvq>0`, `ket all cvq<2`) and Hermitian conjugates; `vvv|vvv` excluded explicitly; for same channel only `iket>=ibra`; same `dE3max` / `OccNat3Cut` cuts; parallel over (ch_bra,ch_ket,ibra).
* `shell-model-atan-npnh` (`...NpNh`, ~700-): runs `ConstructGenerator_ShellModel(atan_func)` and adds core-core (c<c') one-body rotations `eta(c,c')`, and 2-body `vc|qc` and `cc|(vc,qc)` kets that share no orbit between bra and ket (to also decouple core excitations "np-nh").
* `hartree-fock`: sets `Eta` particle rank to 1; for all orbit pairs `j<=i`, `eta(i,j)=H(i,j)/Delta_1b(i,j)` (always the white form, regardless of any f), antisymmetric; basically a one-body rotation to diagonalise the one-body part.
* `1PA`: `ConstructGenerator_1PA(atan)`: one-body shell-model terms with `eta(a,i) = -eta(i,a)` and the 2-body `(vv,qv,qq) | (cc,vc)` block only (omits the `(qv,qq)|vv` valence block). Intended for one-particle-adding (single-nucleon dagger) targeted decoupling.

### 4.5 Denominators
`H_denom` (same as H unless `UpdateGeneral`). All floors: if `|Delta| < denominator_cutoff` (default 1e-6) then `Delta = +denominator_cutoff` (SIGN IS LOST: negative tiny denominators become positive; commented-out earlier code preserved the sign).
* `Get1bDenominator(i,j)`: `Delta = f_ii - f_jj`; Epstein-Nesbet adds `(n_i - n_j) * Gamma_monopole(i,j,i,j)` (the `<ij|V|ij>` J-averaged monopole);
  Moller-Plesset: nothing else; `MP_isospin`: average with the isospin-flipped orbits' SPEs (`(Delta + f_i'i' - f_j'j')/2`). Delta added when `delta_index == -12345` or `i/j == delta_index`.
* `Get2bDenominator(ch_bra, ch_ket, ibra, iket)`: `Delta = f_ii + f_jj - f_kk - f_ll` (bra = ij, ket = kl; bra is the "particle" side) plus (EN only), using monopoles `V̄_abab` with occupation numbers n of bra/ket orbits:
```
Delta += (1-n_i-n_j) V̄_ijij - (1-n_k-n_l) V̄_klkl + (n_i-n_k) V̄_ikik + (n_i-n_l) V̄_ilil + (n_j-n_k) V̄_jkjk + (n_j-n_l) V̄_jljl
```
  MP_isospin averages with the (tz -> -tz) partner orbits (SPE only; EN monopole terms are not added in this mode). `denominator_delta` is added only if `delta_index == -12345` ("all").
  NOTE: `Get2bDenominator` uses `occ` for bra/ket orbits `n`, so fractional occupations (ensemble references) flow into the denominators.
* `Get2bDenominator_Jdep` (unused in imsrg++): same but uses the J-dependent `V_ijij(J)` and `V_klkl(J)` in the pp'/hh' terms.
* `Get3bDenominator(i,j,k,l,m,n)`: `f_ii+f_jj+f_kk-f_ll-f_mm-f_nn` only, with no EN monopole terms and **no floor / cutoff applied** (division by zero is possible; the comment says the EN terms are "unpleasant and almost certainly not necessary"). Delta and partitioning options ignored.
* `denominator_delta` and `denominator_delta_orbit`: the delta is only active if an index is set; imsrg++ only calls `SetDenominatorDeltaOrbit` when the parameter is not `none`; `"all"` -> index `-12345`; a specific orbit string -> `modelspace->GetOrbitIndex`. (Old testing feature, "Not typically used".)
  `SetDenominatorDeltaOrbit(orb)` with a specific orbit dereferences `H` which is uninitialised if no `Update` has happened (it is called right after constructing the solver), i.e. undefined behaviour.

### 4.6 Which block is decoupled when (cheat-sheet)
| Step | generator | one-body | two-body | three-body |
|---|---|---|---|---|
| single-ref/core step | `atan` etc. | core x (v,q) ph | (qq,vv,qv) x cc | ppp x hhh |
| valence step | `shell-model-atan` | (core,valence) x (valence,q), i!=a | (vv,qv,qq) x (cc,vc); (qv,qq) x vv | ppp x (ccc,ccv,cvv,vvv), no vvv x vvv |
| HF-like | `hartree-fock` | all i,j | n/a | n/a |

### 4.7 "PV": parity violating (`GeneratorPV.*`, `IMSRGSolverPV.*`, `BCH::*PV`)
* PV stands for **parity-violating** (a parity-odd perturbation `VPT`, e.g. weak/anapole-like interaction, added "to first order": evolve `H` and `H_PV` together with `H_tot = H + lambda V_PV`, keeping only terms linear in `V`).
  Classes: `GeneratorPV : Generator` (adds `Update(H, HPV, Eta, EtaPV)`), `IMSRGSolverPV : IMSRGSolver` (adds `FlowingOpsPV`, `OmegaPV`, `Etapv`, `VPT_0`, `VPT_saved`), and `BCH::BCH_ProductPV`, `BCH_TransformPV`, `BCH_TransformHPV`, `Standard_BCH_TransformPV/HPV`.
* Algebra: operators are pairs (PC part, PV part) with first-order rules `[(A,A'),(B,B')] = ([A,B] + [A',B'], [A',B] + [A,B'])` for the PC component: since PV x PV is second order in the small parameter... **but the code keeps PV x PV in the PC part** (`Commutator(Y,X) + Commutator(YPV,XPV)`), i.e. it treats PV operators like components of a full "parity-mixed" algebra (with `Etapv` Hermitian/anti-Hermitian flipped relative to `VPT`), not a strict first-order expansion.
  The parity-odd operators are tensors of rank J=0 with parity=1 (reduced/non-reduced handling via `MakeNotReduced()/MakeReduced()` in `AddToEtaPV`).
* `GeneratorPV::ConstructGeneratorPV_SingleRef/ShellModel`: PC eta as usual; PV eta uses the SAME denominators (from H) with `V` as numerator: `Etapv(i,a) = f(V(i,a), Delta(i,a))`, `Etapv(a,i) = herm*Etapv(i,a)` with `herm = V.IsHermitian() ? -1 : +1`; two-body only fills `Etapv` on `(qq,vv,qv) x cc` kets and does NOT set the antisymmetric partner (`iket,ibra` block) -- sets a different sign convention than the PC one; the ShellModel-PV supports the same shell-model blocks. Generators `hartree-fock`, `1PA`, `npnh` not supported (commented out); `qtransfer` in PV is parsed but the dispatch for `qtransfer-atan` exact name is identical.
  Many `timer[...]` assignments sit inside inner loops (copy-paste diagnostics) -> measurable overhead.
* `IMSRGSolverPV`: `Solve_flow_RK4_PV` (RK4 with the coupled system: `K_H = [eta,H] + [eta_pv, V]`, `K_V = [eta,V] + [eta_pv,H]`, `eta`s recomputed at every stage) and `Solve_magnus_euler_PV` (same as magnus Euler but the norms combine as `sqrt(||eta||^2 + ||eta_pv||^2)`,
  `BCH_ProductPV`, `BCH_TransformPV`; hunter-gatherer not implemented -> `exit(0)`), `NewOmega_PV` (stack + scratch files `OMEGAPV_<pid><idx>`), `SetGeneratorPV`, `Transform(Op, OpPV)`, status writers with a different column set (`WriteFlowStatusPV`).
  Bugs: `BCH_ProductPV`'s while-condition is `sqrt(nxy*nxy + nxyPV + nxyPV)` (missing squares on the PV terms); `Solve_*_PV` calls `exit(0)` on error (returns success). `IMSRGSolverPV::method` shadows the base member `method`.
* **Used by:** only `pyIMSRG.cc` (class `IMSRGSolverPV`, `Solve_RK4`, `Solve_magnus_euler`, `AddOperatorPV`, `SetGeneratorPV`, ...). Never by `imsrg++.cc`. Rewrite: treat as an optional extension (a "coupled operator pair" flow) after core features are solid; do not shape the core API around it.

---------------------------------------------------------------------------------------------------------

## 5. Outputs

### 5.1 Files written by `imsrg++`
| file | when | contents |
|---|---|---|
| `flowfile` (default `output/BCH_<method>_<ref>_<val>_hw.._e.._A..`, no extension) | every flow | text, one line per step (and per RHS call for `flow`), no header (header only goes to stdout); appended by both stages; truncated when `SetFlowFile` is called |
| `<intfile>.snt` | valence space, `valence_file_format=tokyo` (default) | KSHELL/Tokyo format: comments `! shell model effective interaction generated by IMSRG version <ver>`, `! input 2N: <file>`, `! input 3N: <file>`, `! e1max .. e2max .. e3max .. hw .. Aref .. Zref .. A_for_kinetic_energy`, `! Zero body term: E0`, model space block (`n_valence_protons n_valence_neutrons Zcore Ncore`), orbit list (index n l 2j 2tz), one-body (`cnt  0  hw`; lines `a b value`, only `a>=b`), two-body (`cnt 0 hw`; lines `a b c d J value`), `1e-7*norm` threshold for dropping elements. Implementation `ReadWrite::WriteTokyo` (ReadWrite.cc:5841). |
| `<intfile>.int`, `<intfile>.sp` | `valence_file_format=nushellx` | NuShellX interaction + single-particle space |
| `<intfile>.ant`, `<intfile>.inp` | `valence_file_format=antoine` | "still being tested"; written in addition to the NuShellX files (the code writes `.ant/.inp` and then falls to the else-branch writing `.int/.sp`) |
| `<intfile><opname>.snt` / `.int` | scalar operators (magnus, or flow* after renormal-order) | same Tokyo format ("op" mode isospin-averages pn-mixed tbmes: `tbme = (tbme(a,b,c,d) + tbme(a-tz,...))/2` when `tz2_a != tz2_b`) |
| `<intfile><opname>_2b.snt`, or `_1b.op` + `_2b.op` | tensor operators | `WriteTensorTokyo` / `WriteTensorOneBody` + `WriteTensorTwoBody` |
| `<intfile><opname>.dag` | operators with an odd number of legs (dagger operators `DaggerHF_*`, `DaggerAlln_*`) | `WriteDaggerOperator` |
| `<intfile><opname>.op` | single-ref non-magnus, rank>0 | `WriteOperatorHuman` |
| `<intfile><op>_HO_*`, `_HF_*` | `write_HO_ops`, `write_HF_ops` | pre-flow operators in core-normal-ordered form (A. Belley) |
| `SPWF` files | `SPWF` list | `WriteSPWaveFunctions(spwf, hf, intfile)` |
| `<intfile>_Omega_<i>` | `write_omega=true` | copies of the Omega scratch binaries (`ReadWrite::CopyFile`) |
| `<intfile>_occ.dat`, `_occ_nat.dat` | `write_omega` | `n l 2j 2tz occ` rows (scientific, width 26, precision 16) for orbits with `|occ|>1e-6` |
| `<intfile>C.mat` | `write_omega` | `hf.C` (HF) or `hf.C_HO2NAT` (NAT) as armadillo binary (`mat.save`) |
| `OMEGA_<pid><idx>` in `scratch` | magnus with scratch | binary Omega operators, deleted on solver destruction |
| FCI: `<intfile>.snt`, `<intfile>_<op>.snt` or NuShellX (`.int/.sp`, `_op.int`), `<intfile><op>_1b.op/_2b.op` | `method=FCI` | |

### 5.2 Flow line format (`WriteFlowStatus`, 1253-1288; header 1295)
`istep` (`%5d`), `s` (`%12.5f`), then fixed width 16 / 9 decimals: `E0 (ZeroBody)`, `||H||`, `estim. err` (cumulative_error, always 0), `||Omega_1||`, `||Omega_2||`, `||Omega_3||` (current `Omega.back()`), `||Eta_1||`, `||Eta_2||`, `||Eta_3||`;
`Ncomm` (`%7.0f`: `N_ScalarCommutators + N_TensorCommutators` from static profiler counter), `E(MP2)` (`H_s.GetMP2_Energy()` -> computed every line!), `N_Ops` (`profiler.counter["N_Operators"]`), `Walltime (s)` (`%12.3f`), `RSS MB / max MB`.
Header: `i s E0 ||H|| estim. err ||Omega_1|| ||Omega_2|| ||Omega_3|| ||Eta_1|| ||Eta_2|| ||Eta_3|| Ncomm E(MP2) N_Ops Walltime (s) Memory (MB)` plus 175 dashes. The step-0 line is printed before the loop (istep=0, s=0).
`istep` is the loop counter that restarts at 1 for each `Solve()` call and `s` continues (second stage lines start from the first stage's final `s`).
Quantities useful for equivalence tests: `E0(s)` per step, `||Eta||` components, `||Omega||` components, `E(MP2)`.

### 5.3 Stdout lines usable for integration tests
In order: Parameters echo (`key => value`), `Hbare 0b = <E>` (after HF/NO, with 8 digits), HF energies from `HartreeFock` (`EHF`, SPEs), `EMP2`, `EMP2_3B`, `To 2nd order, E`, `E3_pp E3_hh E3_ph EMP3`, `To 3rd order, E`, flow lines (final E0), `Setting generator to ...`,
`pushing back another Omega...` (count of Omega stack elements; matches `n_omega_written`/`Omega.size()`), `Core Energy = <E0>` (single-ref, 6 decimals), `Perturbative triples: dE -> E0+dE`, `<basis> expectation value <op>  <value>`, `   HF: <value>`, `HF point proton radius`, `IMSRG point proton radius/charge radius`, timing table from `Hbare.PrintTimes()` (static `IMSRGProfiler::timer`; keys include `BCH_Transform`/`Standard_BCH_Transform`, `BCH_Product`, `UpdateEta`/`Generator::AddToEta`, comm terms).
`Operator::Norm()` convention (needed to reproduce `||eta||`, `||Omega||` numerically): `sqrt(n1^2 + n2^2 + n3^2)` with `n1 = sqrt(sum_pq O_pq^2 (2j_p+1))` for scalars (OneBodyNorm, Operator.cc:1666: degeneracy `(j2+1) * ((min(j2q, j2p+J) - max(-j2q, j2p-J))/2+1)` which equals `(2j+1)` for scalars,
and a different weight for tensors); `n2 = sqrt( sum_channels w * (||M||_F * (2J_ket+1))^2 )` (non-reduced: weight `(2J+1)` applied to the Frobenius norm and then squared, i.e. `(2J+1)^2` weighting; reduced: `1/(2J_rank+1)`), with `w=2` for off-diagonal channel pairs (TwoBodyME.cc:781);
note the one-body `(2j+1)` and two-body `(2J+1)^2` weights are not the same convention. `Norm()` returns the norm of the NON-reduced/reduced storage convention as stored. Thresholds (`eta_criterion`, `omega_norm_max`, `domega`) are defined relative to this, so a rewrite must replicate it to reproduce step sequences.

---------------------------------------------------------------------------------------------------------

## 6. Global/static state, OpenMP, memory, error handling

### 6.1 Global / static state
* `Parameters` : static maps (and defined in a header -> one TU only). `Parameters::string_par["valence_space"]` is mutated by main for custom spaces.
* `IMSRGProfiler::timer`, `counter`, `start_time`: static maps shared by every operator and solver (`profiler.counter["N_Operators"]` etc.); all `Operator`s hold an `IMSRGProfiler profiler` member that references the static maps.
  Not thread-safe (`timer[...] +=` inside OpenMP regions in e.g. PV generator loops or `Eta->profiler.timer[__func__]` after parallel regions).
* `BCH::` file-scope booleans/thresholds listed in Section 3. `Commutator::` globals: `use_imsrg3, use_imsrg3_n7, use_imsrg3_mp4, imsrg3_no_qqq, threebody_threshold, single_thread, verbose, perturbative_triples, comm_term_on` map (per-term toggles),
  `TurnOn_*`. `Generator` has `static std::function` objects (wegner/white/atan/..., initialised at static-init time).
* `ModelSpace` carries mutable "first-pass" flags (`scalar_transform_first_pass`, `scalar3b_transform_first_pass`, `tensor_transform_first_pass[40]`) used by the commutator code to decide whether to precompute and to force single-thread on first pass;
  imsrg++ resets via `modelspace_imsrg.ResetFirstPass()` before the second stage. Also `PreCalculateSixJ()` caches (lazy, in ModelSpace), `sixj_has_been_precalculated` flags.
* The solver holds `Operator* H_0` (raw pointer to caller's `HNO`) and `ModelSpace* modelspace`; if the caller's `HNO` is later reassigned (as imsrg++ does for `brueckner_restart` and re-normal-ordering) the solver's `H_0` still points at the same object by address (stays valid since `HNO` is a reference to `Hbare`), but contents change.
* Hidden coupling: `Generator::H` raw pointers are set only during `AddToEta`; `Generator::SetDenominatorDeltaOrbit` uses them before they are set (UB).
* Scratch file names depend on `getpid()`.
* `IMSRGSolver` holds a copy of the profiler and `ode_monitor` containing references to its own members (copying the solver, as `auto system = *this` does, copies references that still point at the original).

### 6.2 OpenMP
* The solver code itself is serial; parallelism comes from the commutators, the generator, and HF/transformations.
* In `Generator`: only the 3-body generator constructors have `#pragma omp parallel for schedule(dynamic,1)` over a flattened `(ch3bra,ch3ket,ibra)` list (comment: "roll ch3 and ibra into a single index to improve load balancing"), and `GetHod_SingleRef` 3-body (over ch3).
  The 2-body generator loops are serial over channels (a rewrite can parallelise over channel pairs trivially).
* `Commutator.cc` uses `#pragma omp parallel for` extensively and `single_thread` global (forced true during first pass of the 3-body scalar transform since the first pass fills caches that are not thread-safe).
* Elapsed times via `omp_get_wtime()`; thread count from `OMP_NUM_THREADS` only (`goUniversal.py` sets `NTHREADS=48`).
* No MPI anywhere in the analysed files (so MPI is a pure addition for the rewrite).

### 6.3 Memory-heavy parts
* `FlowingOps` (H + any flowing operators), `Eta`, `Omega` stack (each a full `Operator` incl. optional 3-body; "operator size = X MB" is printed at every `NewOmega`), `H_saved`, `H_0` (user's HNO), plus temporaries in every commutator
  (`Commutator::Commutator` returns by value, BCH keeps `OpNested`, `OpOut`; `BCH_Product` holds `Z`, `Nested`, and with factorisation up to `chi`, `chi2`). Peak is about 5-6 full operators for IMSRG(2) and similarly in IMSRG(3) with 3-body storage (which dominates by orders of magnitude).
* The `std::vector<Operator> K1..K4,Ktmp` in `Solve_flow_RK4` allocates `5*nops` full operators per step.
* The ODE solvers deep-copy operators through the `operator+/*` overloads for the vector-space algebra (each stage creates new deques).
* Scratch: writing Omega to disk (`scratch=...`) reduces stack memory to 1 Omega.
* `modelspace_imsrg` is a full second copy of the model space (kets, 3-body kets, lookups) when truncation is used; `hf` (HFMBPT) object (HF matrices, density, `C`) lives to the end of main.
* HNO references Hbare: the 3-body bare interaction storage of Hbare is freed when `HNO = hf.GetNormalOrderedH(...)` assigns (the old storage is overwritten).

### 6.4 Error handling
* No exceptions are used for numerical issues. Errors are: print + `return 1` (missing files), `exit(EXIT_FAILURE)` (NaN in Trel, scratch problems), `exit(0)` (PV solver mismatches, returning success!), `std::terminate()` (too many Omega files), print-and-continue (unknown method/generator/parameter),
  divergence guard multiplies H by `1.0/0.0`.
* Unknown `method`: the code prints a message but the driver continues to outputs (producing meaningless files).
* Output directories are never created; file open failures in writers are not checked (`ofstream` silently fails).

---------------------------------------------------------------------------------------------------------

## 7. Dependency graph

```
imsrg++.cc
 ├─ Parameters.hh                (standalone; static maps)
 ├─ IMSRG.hh  (umbrella: ReadWrite, ModelSpace, TwoBodyME, ThreeBodyME, Operator, HartreeFock, Commutator, BCH,
 │             Generator, GeneratorPV, IMSRGSolver, IMSRGSolverPV, imsrg_util, AngMom, IMSRGProfiler, Jacobi3BME,
 │             HFMBPT, UnitTest, PhysicalConstants, RPA, ReferenceImplementations)
 ├─ ReadWrite  (all readers/writers; also Set3NFormat, scratch dir, CopyFile)
 ├─ ModelSpace (orbits, kets, channels, holes/core/valence/q, e-max settings, SetReference)
 ├─ HFMBPT <- HartreeFock           (HF, natural orbitals, normal ordering; HFMBPT also uses Generator (GetHod, UpdateGeneral) for Van Vleck / MBPT)
 ├─ imsrg_util (OperatorFromString, Trel_Op, HCM, RPA_resummed, FirstOrderCorr_1b, WriteSPWaveFunctions, VCoulomb ...)
 ├─ IMSRGSolver
 │    ├─ Operator (ZeroBody/OneBody/TwoBody/ThreeBody, Norm*, Binary I/O, Truncate, Do/UndoNormalOrdering, MP2/MP3 energies)
 │    ├─ Generator (holds raw pointers to Operator; reads ModelSpace partitions, TwoBodyME monopoles, ThreeBodyME)
 │    ├─ BCH (Commutator:: for all nested commutators; Operator)
 │    ├─ Commutator (scalar/tensor/dagger/IMSRG3/Factorized/232)
 │    ├─ IMSRGProfiler (static)
 │    ├─ ReadWrite (only for `SetReadWrite` to fetch scratch dir: "for backwards compatibility")
 │    └─ boost::odeint (only flow/ODE methods; `NO_ODE` compile switch)
 ├─ IMSRGSolverPV -> IMSRGSolver, GeneratorPV -> Generator, BCH::*PV (python only)
 └─ pyIMSRG.cc (pybind11 bindings of all of the above)
```
Awkward/circular points to avoid in a rewrite:
1. `Solver -> Generator -> (ModelSpace, Operator)` with raw pointers kept in Generator across calls; make generators stateless functions `eta = gen(H, H_denom, params)`.
2. `BCH` is not independent of `Commutator` global state (`use_imsrg3`, thresholds): BCH_Product flips `Commutator::use_imsrg3` temporarily (save/restore of a global); `BCH_Transform` allocates 3-body storage as a side effect depending on global state.
   Pass a `CommutatorPolicy` (IMSRG level, enabled terms, thresholds) explicitly.
3. `IMSRGSolver` public data members (everything is public with `//  private:` commented out) are touched by main (`imsrgsolver.FlowingOps[0] = H`, `imsrgsolver.s = 0`, `max_omega_written`, `n_omega_written`) and bindings (`def_readwrite`).
4. `ReadWrite` is a god-class (readers, writers, scratch, CopyFile, LEC presets); the solver only needs a scratch path.
5. The ODE code mutates solver state from inside the RHS and copies the solver (Sec. 2.3); keep RHS pure.
6. `Parameters` as global static maps + `Parameters.hh` defining statics: replace with a typed immutable config struct that is passed everywhere (and serialised to JSON/YAML or Python kwargs).
7. `imsrg++.cc` is a monolithic script: model-space building, HF, truncation, solver orchestration, operator transform and I/O interleaved with `if (method...)` branches. The rewrite should use a pipeline of stages with explicit data flow
   (`ModelSpace -> Hamiltonian -> BasisTransform -> NormalOrder -> Truncate -> Solve(core) -> Solve(valence) -> Transform ops -> Output`).
8. Hbare/HNO aliasing (reference to overwrite) makes data flow implicit.

### 7.1 Legacy example drivers (`work/`)
* `work/compiled/SingleRef.cc`, `Universal.cc`, `Universal_tensor_debug.cc`, `ValenceSpace.cc`, `ValenceSpace_targeted.cc`: older hand-parsed (`var=val`) executables that predate `imsrg++.cc` (they use removed APIs: `SetN3max`, `CalculateKineticEnergy`, `TCM_Op`, `Hbare -= Tcm`, `SetHin` after construction).
  They show the historic pipeline: read 2N/3N -> kinetic energy -> HF (`hf.Solve(); hf.GetNormalOrderedH()`) -> optional Brueckner -> `IMSRGSolver` -> `SetGenerator(core_generator); Solve(); SetGenerator(valence_generator); SetSmax; Solve()` -> `Transform` of `Rp2`/operators.
  `ValenceSpace_targeted.cc` hard-codes orbit lists for `sd-shell`, `fp-shell`, `d5-shell`, ... and references (`O16, O22, O24, O28, Si28, Si34, S32, C12, Ca40, Ca48`) -- an ancestor of the targeted/ensemble normal ordering (`reference` != core) now done via `ModelSpace(emax, reference, valence_space)`.
  `SingleRef.cc` charge radius uses `R2p = 0.770, R2n = -0.1149, DF = 0.033` (hard-coded; now `PhysConst`). `Makefile` builds `Universal` only, linking `-lIMSRG -lopenblas -lgsl -lz -lhdf5 -lhdf5_cpp`, `-std=c++11 -fopenmp -O3`.
  `work/compiled/Parameters.hh` is a copy of `src/Parameters.hh`, without the license header (diverged slightly).
  Verdict: not useful as references for the new code except as examples of the intended API call order; do not port.
* `work/scripts/goUniversal.py`: Python 2 launcher (batch-system detection PBS/SLURM, `NTHREADS=48`, `exe` hard-coded path, builds `ARGS` dictionary -> command line `key=value`; default `basis=oscillator`, `smax=0`, `dsmax=0.5`, file-size args `file2e1max=18 file2e2max=36 file2lmax=18`, `file3e1max=18 file3e2max=36 file3e3max=24`, `omega_norm_max=0.25`,
  commented alternatives for `core_generator`, `valence_generator`, `method` (`magnus`, `brueckner`, `flow`, `HF`, `MP3`), `ode_tolerance`; loops over elements/nuclei). Useful as a catalog of how users call the program; the interesting default is `smax=0` in the script (meaning "HF only"?: with `smax=0` the loop `for s<smax` does not execute, i.e. no flow, only normal-ordering/outputs).
* `work/scripts/python/GetNormalOrderedH.py`: Python 2 pyIMSRG example: read NN+3N, optional HF, write normal-ordered two-body interaction in Oslo format for MBPT code (`basis='oscillator'|'HF'|'IMSRG'`).

---------------------------------------------------------------------------------------------------------

## 8. Pitfalls and items to replicate or deliberately fix

### 8.1 Defaults and parameter interactions
* **P1 `nsteps` / generators**: `nsteps=-1` -> 1 for single-reference (no valence orbits), 2 otherwise. With `nsteps=1` and a valence space, `valence_generator` is never used (the docs in `Parameters.hh` say it is). `brueckner1step` forces `core_generator=valence_generator`.
  The second stage condition is `nsteps>1 and valence_space != reference` (string comparison).
* **P2 `smax`**: default 200 (very large). Magnus doubles `smax` for stage 2 (`smax *= 2`), but flow methods do not, so a flow that stopped at `smax` in stage 1 does nothing in stage 2. `goUniversal.py` sets `smax=0`.
* **P3 `ds_0` is effectively dead in magnus**: `ds` is recomputed from `domega`, `omega_norm_max`, `dsmax` at every step. Only `magnus_backoff` and the ODE methods use `ds_0`; `imaginary`/`wegner` generators force `ds_0=1e-4`, `dsmax=1e-2` iff `ds_0>1e-2`.
* **P4 `domega`/`omega_norm_max`**: if `|omega_norm_max - domega|<1e-6` then `norm_domega += 1e-4` (to avoid platform-dependent splitting). Also `ds` uses `omega_norm_max + 1e-5` so a step never lands exactly on the split threshold.
  Magnus default `omega_norm_max=0.25` but the constructor default is 2.0; `NSmagnus` -> 50000; `brueckner*` -> 500. `domega` is not clamped to `omega_norm_max` (the old `SingleRef.cc` did: `min(domega, omega_norm_max+1e-6)`).
* **P5 `denominator_delta`**: silently ignored unless `denominator_delta_orbit != none` (`all` or an orbit name), because the delta is gated by `denominator_delta_index` (-1 by default). For 2-body denominators only `all` works.
* **P6 `denominator_cutoff`**: ctor default 1e-6; for `shell-model-atan` in `flow_RK4` the cutoff is initially set to 1.0 then to 1e-6 when `||eta||<1` (a "soft start"), but the same code in `magnus` is a no-op. Floor replaces `Delta` by `+cutoff` irrespective of sign. 3-body denominators are never floored.
* **P7 `eta_criterion`** applies to `||eta||` of the full operator including 3-body (and in PV: combined). `SetEtaCriterion` takes a `float` (precision loss: `1e-6` stored as float then promoted); `SetODETolerance(float)` likewise.
* **P8 `emax_*_imsrg`**: any one of the four set triggers re-initialisation of the IMSRG model space with defaults for the rest (`e2max=2*emax_imsrg`, `e3max=min(e3max,3*emax_imsrg)`); `file2e2max=-1` means `2*emax`; `file2lmax=-1` means no cut; `lmax=99999`; `e3max` default 12 but `ModelSpace` ctor default is `min(14,3*emax)`; `lmax3` only applied if `>0`.
* **P9 `A=-1`** -> use `A` of the reference for kinetic energy; `modelspace.SetTargetMass(A)` only if `A>0`. Setting `reference` != `A` is the "A_for_kinetic_energy" in the output header.
* **P10 `perturbative_triples`** only executes for `method=="magnus"`, and right after the FIRST stage solve (before the valence step), at `E3max=3*Emax` of `modelspace`; the solver flag `SetPerturbativeTriples` is unused inside the solver. `modelspace.CountThreeBodyStatesInsideCut()` banner is printed with `dE3max`, `OccNat3Cut`.
* **P11 `imsrg3_at_end`** needs `method` containing `magnus`; builds `H3` from `HNO` (the INPUT NO H), so IMSRG(3) at the end re-transforms the original Hamiltonian with all stored Omega: the 3N interaction is present only if `HNO` already had rank 3 (requires `IMSRG3=true` for `hno_particle_rank=3`). `new_E3max` computed from Fermi energies of the IMSRG model space.
* **P12 Operators + method**: `PT1/RPA/TDA` and `OperatorsFromFile` handling differs between magnus (OperatorsFromFile only) and flow (everything); operators are in `FlowingOps` only for `flow` and `flow_RK4`; with other non-magnus methods they are never transformed. `magnus` requires `scratch` not `/dev/null` iff operators requested.
* **P13 `hwBetaCM`** default -1 -> replaced by the basis `hw`; the constant zero-body shift `-1.5 hwBetaCM BetaCM` is subtracted from `HNO` and the 1B+2B Lawson term is added in the HO basis before HF.
* **P14 default file names** use `setprecision(0)` without fixed -> `hw=20` -> "2e+01"; they require an `output/` directory.
* **P15 `reference` vs `valence_space` strings** carry semantic meaning: `reference=default` means "reference = core of the valence space (or the nucleus)". Custom spaces are given as a comma-separated list "core orbits ..., valence orbits ..." (see `ParseCommaSeparatedValenceSpace`).
* **P16 `basis=NAT` or `OccNat3Cut>0`** both trigger `GetNaturalOrbitals()` even for `basis=HF`/`oscillator`.

### 8.2 Code-level bugs/oddities (decide per item: replicate to match numbers, or fix)
1. `Standard_BCH_Transform` for tensor operators prints diagnostics each iteration and calls `OpOut.modelspace->GetOrbitIndex(0,2,5,+1)` (arbitrary orbit) just to define an unused `id5`.
2. BCH_Product omits the `-1/24 [Y,[X,[X,Y]]]` term and takes only a leading `1/12[[Y,X],X]` correction; it is accurate only for small X. Replicating would give identical numbers, fixing would change numerics at the 1e-4 threshold level.
3. `BCH_ProductPV` while-condition `sqrt(nxy*nxy+nxyPV+nxyPV)` (missing squares); `Solve_*_PV` exit with success code; `IMSRGSolverPV::method` shadows base.
4. `Generator::Get1bDenominator` / `Get2bDenominator` floor sign loss (commented alternative preserves sign). `Get3bDenominator` has no floor.
5. `Generator::H` uninitialised pointer before first `AddToEta`; used by `SetDenominatorDeltaOrbit`.
6. `ConstructGenerator_HartreeFock` always uses `H/Delta` (ignores selected function). `wegner_func` described as untested.
7. `Solve_magnus_modified_euler` uses `0.5*ds*eta_mid` (see table) and lacks every safeguard of `magnus_euler`.
8. `Solve_ode_adaptive` copies `*this` (monitor and eta statistics refer to the original), prints a flow line for every RHS evaluation (including rejected stages), `ode_mode="Omega"` mutates the Omega stack inside RK stages, `NewOmega` inside the RHS.
9. `IMSRGSolver::FlushOmegaToScratch` computes filenames as `i + n_omega_written` while incrementing `n_omega_written` in the loop (index drift when more than one Omega is in memory, plausible only with no scratch and `Omega.size()>1`) and logs `i + n_omega_written` after the increment.
   `write_omega` then copies `OMEGA_<pid><i>` for `i < GetNOmegaWritten()`: this assumes the flush created contiguous files. Without scratch (`""`), `FlushOmegaToScratch` returns early, so `write_omega` copies nothing and silently writes only occupations and C.mat.
10. `CleanupScratch` is called from the destructor, including destruction of the copy made by `auto system = *this` (shared scratch dir/pid-based file names): claim (unverified) risk of deleting live Omega files; plus `EstimateBCHError` returns 0 ("not ready for prime time") and prints a cumulative counter.
11. `GatherOmega` when `Omega.size()<2` pushes a copy then `Erase()`: July 2024 bug fix (comment). Splitting thresholds adjusted (1e-12 vs 1e-6) in the same revision; a bug-for-bug numerical replication of older results must account for this.
12. `IMSRGSolver::SetHin` (used by brueckner2 and old drivers) triggers `NewOmega()` if the current Omega has `||.||>1e-6`, which silently changes `H_saved` semantics.
13. Mixing norm conventions: one-body `(2j+1)`, two-body `(2J+1)^2` weighting, and `||Eta|| < eta_criterion` is a full-operator norm; with 3-body included in IMSRG(3).
14. `Parameters` parsing: `int` params given a float string truncate silently; booleans are strings compared to the literal `"true"` (anything else, e.g. `True`, `1`, `yes` -> false).
15. `HNO -= BetaCM*1.5*hwBetaCM` is applied even when `method` has no Lawson term (0) and uses the possibly-updated `hwBetaCM`.
16. In `imsrg++.cc` the 3N reading branch: `if (full) ...; if (no2b) ... else if (mono) ...` - OK but fragile; `rw.File3N` set only for `no2b`/`mono` (the header comment in the `.snt` file prints `input 3N:` from `File3N`, empty for `full`; for `full` the filename is probably recorded inside `Read_Darmstadt_3body`).
17. `Hbare.ThreeBodyNorm()` is printed at `imsrg3_at_end` after `Hbare` was overwritten by the normal-ordered H (aliasing).
18. The IMSRG(3) paths require pn-mode 3-body storage; `Commutator::use_imsrg3` is global so a mixed IMSRG(2)/IMSRG(3) calculation in one process works only by toggling globals (as `BCH_Product` and `imsrg3_at_end` do).
19. `ModelSpace::scalar3b_transform_first_pass` forces single-threading at the first IMSRG(3) call; parallel efficiency of the first step is not representative.

### 8.3 Numerical-equivalence checklist for the rewrite
To reproduce legacy numbers bit-near (1e-6..1e-8 in energy):
1. Same Omega-step rule (Sec. 2.1, including `1e-9`, `1e-5` fudge terms and the domega bump), same `Norm()` convention, same splitting semantics (strict `>`), same `H_saved` update rule and the `(Omega.size()+n_omega_written)<2` start condition.
2. Same BCH_Transform termination (`epsilon` recurrence, `bch_transform_threshold=1e-9`, `max_iter=40`) and BCH_Product truncation (`1e-4`, Bernoulli list to k=8, `1/12[[Y,X],X]` only if `nxy*nx>1e-4`).
3. Same denominator definitions (EN with J-averaged monopoles, `occ` weighted), same cutoff 1e-6 positive floor, `atan` form `0.5 atan(2 H/Delta)`.
4. Generator block definitions exactly as in 4.3/4.4 including: valence-valence off-diagonal one-body elimination in `shell-model*`, no `qc` bras in shell-model, `cc` ket only for single-ref (not `vc`).
5. The order of operations in the two-step: core generator solve -> (perturbative triples) -> `SetGenerator(valence)` which opens a new Omega (stack size 2) -> `smax*=2` -> second solve -> optional `imsrg3_at_end` -> optional re-NO -> output -> transform ops by replaying both Omegas.
6. Operators transformed from the HF-normal-ordered operator, truncated to `modelspace_imsrg`, then replayed, then re-NO if the reference != core.
7. Reference outputs to snapshot: flow file column values (E0, ||H||, ||Omega_k||, ||Eta_k||, E(MP2)), `Hbare 0b`, `EMP2`, `EMP3`, `Core Energy`, and `.snt` contents (zero-body comment, one-body, two-body lists). The `Walltime`, `Memory`, `Ncomm` columns are not comparable.

### 8.4 Suggested feature checklist (what a rewrite must support to be at parity)
- [ ] Parameter system (typed, unknown-key diagnostics) with all keys of Sec. 1.1, key=value CLI + Python API
- [ ] Model-space definition: nucleus/valence-space table, custom lists, FCI, `0hw`/`jj`, `emax_unocc`, `lmax`, `emax_imsrg` family, ensemble occupations (`occ_file`, fractional)
- [ ] Interaction readers: me2j (e1max/e2max/lmax cuts), Navratil, Oslo, OakRidge (+bin), Takayuki, NuShellX, Minnesota; 3N: me3j full, no2b (single/half), mono; Miyagi/Navratil operator inputs
- [ ] Trel/CM terms: `Trel_Op`, `BetaCM`/`hwBetaCM`, `hw_trap`, mass correction, relativistic p^4, atomic units variant
- [ ] HF, NAT (3 orderings, `use_NAT_occupations`), oscillator basis; NO2B with 3N; basis-transform of operators; targeted/ensemble NO and re-NO wrt core
- [ ] MBPT2/3 diagnostics
- [ ] Solvers: magnus (+backoff, modified Euler, hunter-gatherer, Omega stack with scratch), flow_RK4, adaptive dopri5 flow, flow_euler; optional magnus_adaptive/restore_4th_order
- [ ] Generators: white, atan, imaginary-time, wegner, qtransfer-atan_n, shell-model (+atan, +imaginary-time, +wegner, +npnh), hartree-fock, 1PA; denominator options (EN/MP/MP_isospin, delta, cutoff); IMSRG(3) generator blocks with dE3max/OccNat3Cut; `only_1b/2b_eta`
- [ ] BCH transform/product, Brueckner variant, goose-tank, factorised corrections, `bch_skip_ieq1` / perturbative triples, `only_2b_omega`, IMSRG(3) at end, Omega transform replay and inverse transform
- [ ] Operator pipeline: scalar/tensor/dagger operators, PT1/RPA/TDA, from-file operators, HO/HF op writing, Rp2 charge radius post-processing
- [ ] Output: flow file, Tokyo `.snt` / NuShellX `.int/.sp` / Antoine, tensor `.op`/`_2b.snt`, `.dag`, occupations, C.mat, Omega copy, SPWF
- [ ] Optional extension: PV (parity-violating) pair solver
