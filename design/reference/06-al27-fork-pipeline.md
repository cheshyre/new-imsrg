# 27Al VS-IMSRG(2) NREFT responses: what imsrg++ (fork, bleeding_edge 79e9330) actually does

Analysis date 2026-11-26. Read-only. Sources:

* Fork at 79e9330: bare clone `$IMSRG_REFERENCE_ROOT/imsrg-fork-79e9330/fork.git`; `src/` extracted with `git archive` to
  `$IMSRG_REFERENCE_ROOT/imsrg-fork-79e9330/src79/src/` (all citations `X.cc:N` below refer to `src79/src/X.cc` unless stated otherwise).
  `code_reference/imsrg++.cc` and `code_reference/DarkMatterNREFT.cc` in the provenance package are
  byte-identical to these files (checked with `cmp`). The `OperatorFromString` excerpt in the
  package has different line numbers (FM0p at excerpt:140 = `imsrg_util.cc:199`).
* imsrg-devel (the project's reference): `$IMSRG_REFERENCE_ROOT/imsrg-devel/src` ("devel" below).
* The archived logs/files: `$IMSRG_WORK/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw/`
  (example job J00011151, N2LOsat, FPhipp2p).
* Helper scripts written for this analysis (pure Python, venv python, run with `-I`), in
  `$IMSRG_WORK/inputs/al27/analysis/scripts/`:
  `scripts/dmops.py` (port of the DM operators, extended with Phi' and tilde-Phi'),
  `scripts/herm.py` (Hermiticity classification), `scripts/q0.py` (q -> 0 limits),
  `scripts/nobug.py` (size of the tensor normal-ordering bug in the archived files).

Template arguments (templates/SLURM_template_*.sh): `hw=16 A=27 reference=Al27 valence_space=sd-shell
BetaCM=0 hwBetaCM=16 basis=NAT use_HF_reference_in_NAT=true emax=14 e3max=24 denominator_delta=0
IMSRG3=false emax_imsrg=10 e3max_imsrg=10 emax_3body_imsrg=6 method=NSmagnus smax=100 dsmax=0.5
ds_0=0.01 omega_norm_max=0.25 threebody_threshold=0 eta_criterion=0.001 write_omega=true
only_2b_omega=false only_2b_omega_at_end=false reduced_232_impl=false valence_file_format=tokyo
3bme_type=no2b no2b_precision=single`, operators `Operators=<op>_<q>,...,TCM`, scratch dir set.
Defaults that matter (Parameters.hh / parameter_defaults): `core_generator=atan`,
`valence_generator=shell-model-atan`, `denominator_partitioning=Epstein_Nesbet`, `domega=0.2`,
`NAT_order=occupation`, `use_NAT_occupations=false`, `freeze_occupations=false`,
`hunter_gatherer=false`, `nsteps=-1` (-> 2 because the valence space is non-empty, imsrg++.cc:364-365).

---------------------------------------------------------------------------------------------------

## 1. The pipeline, step by step

### 1.0 Method shorthand
* `NSmagnus` -> `omega_norm_max = 50000`, `method = "magnus"` (imsrg++.cc:274-278). The scripted
  `omega_norm_max=0.25` is therefore overridden; no Omega splitting within a stage.
* With `magnus` and operators requested, the scratch dir must be writable (imsrg++.cc:297-322).

### 1.1 Model space and reference
* `ModelSpace(eMax=14, "Al27", "sd-shell")` (imsrg++.cc:339) -> `Init` (ModelSpace.cc:225-233):
  `GetOrbitsAZ(27,13)` gives the hole list with fractional occupation (equal filling):
  proton 0s,0p full, proton 0d5/2 occ 5/6; neutron 0s,0p,0d5/2 full. Core list = orbits of 16O
  (from the valence-space definition), valence = sd orbits (ModelSpace.cc:238-302, 378-480:
  `cvq` = 0 core, 1 valence, 2 q-space; `occ` from the hole list).
  So "core" (16O) != "holes" (27Al ensemble). Generators loop over `core`; normal ordering uses `holes`.
* `SetTargetMass(27)` (imsrg++.cc:369-370), E3max 24, Lmax unrestricted, hw 16.

### 1.2 Hamiltonian input
* `Hbare = Operator(ms,0,0,0,3)` Hermitian (imsrg++.cc:436-438).
* 2N: `ReadBareTBME_Darmstadt(file, Hbare, file2e1max, file2e2max, file2lmax)` (imsrg++.cc:449-450).
* 3N: `ThreeBody.SetMode("no2b")` (single precision), `ReadFile({3bme},{16,32,24,16})` (imsrg++.cc:483-492).
* `Hbare += Trel_Op(ms)` (intrinsic kinetic energy with A = 27; imsrg++.cc:522-530). No Lawson term
  (`|BetaCM|<1e-3`, imsrg++.cc:554), no mass/relativistic corrections.

### 1.3 HF with ensemble reference
* `HFMBPT hf(Hbare)`; `freeze_occupations=false` -> `hf.UnFreezeOccupations()` (imsrg++.cc:566-568):
  during the HF iterations the occupations are re-filled according to the HF single-particle energies
  (`HartreeFock::FillLowestOrbits`, HartreeFock.cc:81, 685). For 27Al the result equals the naive
  filling (archived `_occ.dat`: proton 0d5/2 = 0.8333).
* `hf.Solve()` (imsrg++.cc:574). Converged after 33 (N2LOsat) / 29 (DN2LOGO) iterations.
  HartreeFock.cc differs from devel only in the 3N-E3max bookkeeping (`Hbare.E3max` vs
  `modelspace->GetE3max()`, same value here) and the GetTransformed3B signature (unused, NO2B).

### 1.4 NAT basis with use_HF_reference_in_NAT  (fork-only parts marked FORK)
imsrg++.cc:589-604:
1. `hf.UseNATOccupations(false)`, `hf.OrderNATBy("occupation")`.
2. `hf.GetNaturalOrbitals()` (HFMBPT.cc:30-200):
   * `GetDensityMatrix()` (HFMBPT.cc:542-561): `Hhf = GetNormalOrderedH()` (HF basis, NO2B),
     `rho` = diag(occ of holes) (fractional 5/6 included) + MBPT2 corrections `DensityMatrixPP/HH/PH`
     (same as devel HFMBPT.cc:421-440).
   * `DiagonalizeRho()` (HFMBPT.cc:202-237): per (l,j,tz) channel `eig_sym`, eigenvalues reversed
     -> NAT orbits ordered by descending (signed) occupation; `C_HF2NAT` columns accordingly.
   * Check `Tr(rho)(2j+1) == A` to 1e-8, else `exit(0)`.
   * **FORK: re-sorting by |occupation|** (HFMBPT.cc:131-182, not in devel): within each channel the
     NAT columns are re-sorted by **absolute value** of the occupation (descending). This is triggered in
     the archived runs (11 channels re-sorted in the J00011151 log, lines 66-87). Example, proton 0d5/2
     channel (orbit 6): the NAT with occupation -0.013581 (old slot 234, the highest-n slot) is moved to
     the second slot (orbit 24, i.e. labelled n=1 "1d5/2"), ahead of +0.010413. So which natural orbitals
     survive the emax 14 -> 10 truncation, and their labels, depend on this. Must be replicated.
   * End of GetNaturalOrbitals: `rho` rebuilt in HO basis with the NAT hole orbits and the reference
     occupations, `UpdateF()`, `ReorderHFMBPTCoefficients()` (sign convention: column i flipped if
     `C_HF2NAT(i,i) < 0`; HFMBPT.cc:1170-1175; no reordering for NAT_order=occupation),
     `C_HO2NAT = C * C_HF2NAT`.
3. **FORK: `hf.UseHFForHoleStates()`** (HFMBPT.cc:892-1136; called at imsrg++.cc:601-603; not in devel):
   * For every orbit i with `occ > 1e-8` (all 27Al holes, incl. the fractional proton 0d5/2): column i of
     `C_HF2NAT` := unit vector e_i, **and row i := 0 for all other columns j** (removes the HF-hole
     component from every NAT virtual).
   * Gram-Schmidt within each (l,j,tz) channel in increasing orbit index (j < i), projection coefficient
     `<i|j>/<j|j>` (vectors not yet normalized; modified GS since overlaps are recomputed per j),
     then normalize every column.
   * Trace check of the (old) `rho` again; `C_HO2NAT = C*C_HF2NAT`; `occ_nat = |Occ|`;
     `rho` rebuilt from the new hole columns (= the HF density), `UpdateF()` (= HF Fock matrix),
     `ReorderHFMBPTCoefficients()` (sign fix again), `C_HO2NAT = C*C_HF2NAT`.
   * Consequence: E_Nat == E_HF (log: e1Nat 443.7565501 vs e1hf 443.7565509 etc., E_Nat = EHF =
     -56.4272908). Occupied orbitals are exactly HF orbitals; virtuals are "NAT, orthogonalized
     against the HF holes, then GS in label order".
4. `HNO = hf.GetNormalOrderedHNAT(2)` (HFMBPT.cc:378-466): `ZeroBody = EHF` (via `CalcEHF` with the
   current rho), `OneBody = C_HO2NAT^T F C_HO2NAT`, `TwoBody = D^T (V2 + V3NO) D` with
   `V3NO(i,j) = sum_{a,b} rho_ab V3_no2b(bra.p,bra.q,a; ket.p,ket.q,b; J) / (2J+1)` (E3max = 24 cut
   `2n_a+l_a+e2bra <= E3max`, |rho_ab| < 1e-8 skipped, 1/sqrt2 for p==q). Residual 3N discarded
   (`hno_particle_rank = 2` because no2b, imsrg++.cc:578-581).
* `HNO -= BetaCM*1.5*hwBetaCM` = 0 (imsrg++.cc:667). MP2 printout (EMP2 at emax 14).

### 1.5 emax 14 -> emax_imsrg 10 truncation
imsrg++.cc:887-929:
* `modelspace_imsrg = modelspace` copy, then `SetEmax(10)`, `SetE2max(20)` (= 2*emax_imsrg, default),
  `SetE3max(10)`, `SetEmax3Body(6)`, `Init(10,"Al27","sd-shell")`, then
  `SetReference(hole_map)` with the occupations copied orbit-by-orbit (n,l,j,tz) from the emax-14
  model space (keeps fractional 5/6).
* `HNO = HNO.Truncate(modelspace_imsrg)` (Operator.cc:816-876): zero body copied, 1b = submatrix of
  the orbits present (by (n,l,j,tz) labels, i.e. the NAT labels after the |occ| re-sorting),
  2b = submatrix of kets with both orbits in emax 10 and e2 <= 20; Hermiticity flags copied.
  No re-normal-ordering: the normal-ordered pieces from emax 14 are simply cut.
* MP2/MP3 printout at emax 10 (imsrg++.cc:946-957).
* **Operators**: with method magnus they are built only after the flow (section 1.8), in the emax-14
  `modelspace`, transformed HO -> NAT, normal ordered w.r.t. 27Al in emax 14, and then truncated with the
  same `Truncate(modelspace_imsrg)` (imsrg++.cc:1427-1475).

### 1.6 Solver setup
imsrg++.cc:967-1013: `IMSRGSolver(HNO)`, `SetReadWrite(rw)` (only copies the scratch dir,
IMSRGSolver.hh:92), method `magnus`, `Epstein_Nesbet`, `eta_criterion 1e-3`, `max_omega_written 500`,
hunter-gatherer off, `smax 100`, `ds 0.01` (also stored as ds_0), `dsmax 0.5`, `denominator_delta 0`,
`domega 0.2`, `omega_norm_max 50000`. Commutator switches: IMSRG3 off, Brueckner off, goose tank off.

### 1.7 The two decoupling stages
Solver: `Solve_magnus_euler` (IMSRGSolver.cc:230-302), identical in logic to devel (devel adds
`+1e-5` to omega_norm_max in the ds formula and a NaN guard; irrelevant here).
Each step:
1. `norm_eta = Eta.Norm()`; stop if `< eta_criterion` (1e-3). Checked before the step.
2. If `||Omega.back()|| > omega_norm_max` (never, 50000) -> NewOmega.
3. `ds = min(domega/|eta|, domega/(|eta| (|Omega|+1e-9)), omega_norm_max/|eta|, dsmax)`,
   `ds = min(ds, smax - s)`; `s += ds`; `Eta *= ds`.
4. `Omega = BCH_Product(Eta, Omega)` (Commutator.cc:723-790).
5. `H(s) = BCH_Transform(H_0, Omega)` in stage 1; in stage 2 `BCH_Transform(H_saved, Omega_2)`
   where `H_saved` = H at the end of stage 1 (IMSRGSolver.cc:279-286, 42-84).
6. For `shell-model-atan` with `norm_eta < 1`: `SetDenominatorCutoff(1e-6)` (no-op: default already 1e-6).
7. `generator.Update(H(s), Eta)` (Eta erased and rebuilt from the current H(s)).

Stage 1 (imsrg++.cc:1045-1057): generator `atan` = `ConstructGenerator_SingleRef(atan_func)`
(Generator.cc:194-237), `atan_func(H,D) = 0.5*atan(2H/D)` (Generator.cc:19):
* 1b: eta(i,a) for a in **core (16O)**, i in valence U qspace; eta(a,i) = -eta(i,a).
* 2b: kets cc (core-core) vs bras in qq U vv U qv; antisymmetric.
* So stage 1 decouples the 16O core, with 27Al-ensemble normal ordering and 27Al occupations in the
  denominators.

Stage 2 (imsrg++.cc:1086-1106): `smax *= 2` (100 -> 200; only for method magnus), generator
`shell-model-atan` = `ConstructGenerator_ShellModel(atan_func)` (Generator.cc:382-445):
* 1b: (core U valence) x (valence U q), i != a.
* 2b: (cc U vc) kets vs (vv U qv U qq) bras, and vv kets vs (qv U qq) bras.
* `SetGenerator` sees `||Omega_1|| > 1e-6` -> `NewOmega()` (IMSRGSolver.cc:155-167): Omega_1 is written
  to `scratch/OMEGA_<pid6><000>` (log line 396), the in-memory Omega is erased and reused for Omega_2,
  `H_saved = H(s)`; `ds` reset to ds_0 (irrelevant with adaptive ds).
* **s is not reset**: stage 2 continues from s(stage 1) up to smax = 200 (archived: stage 1 ends at
  s = 13.32, stage 2 at s = 57.66 for N2LOsat; 37 + 272 steps).

Energy denominators (Generator.cc:92-145, Epstein-Nesbet; devel equivalent with H_denom = H):
* 1b: `D = f_ii - f_jj + (n_i - n_j) Vbar_ijij`, monopole
  `Vbar_ijij = sum_J (2J+1) V^J_ijij / ((2j_i+1)(2j_j+1))` with **unnormalized** `GetTBME`
  (TwoBodyME.cc:577-601).
* 2b: `D = f_ii+f_jj-f_kk-f_ll + (1-n_i-n_j)Vbar_ijij - (1-n_k-n_l)Vbar_klkl + (n_i-n_k)Vbar_ikik
  + (n_i-n_l)Vbar_ilil + (n_j-n_k)Vbar_jkjk + (n_j-n_l)Vbar_jljl` (monopoles, J-independent).
* `|D| < 1e-6` -> `D = +1e-6` (sign lost). `denominator_delta = 0`.
* Occupations n are the 27Al ensemble occupations (5/6 for proton 0d5/2).

Norms (Operator.cc:1600-1641, TwoBodyME.cc:648-669): `||O||^2 = ||O1||^2 + ||O2||^2`,
1b: `sum_{pq} O_pq^2 (2j_p+1) * #m_q` (for rank 0: weight 2j+1; rank_J>0: (j2_p+1)*number of allowed
m_q), 2b: `sum_{channel pairs} [(2J_ket+1)(#M_bra) ||M||_F]^2`, x2 for off-diagonal channel pairs
(rank 0: (2J+1)^2 weight). These norms drive ds, eta_criterion, and the BCH stopping rule.

What is stored: two Magnus operators. Omega_1 (atan stage) in the scratch file `OMEGA_..000`,
Omega_2 (valence stage) in memory (`Omega.size() = 1`, `n_omega_written = 1`).

### 1.8 Transforming the operators (after the flow)
imsrg++.cc:1365-1560, per operator name, in the order given on the command line
(e.g. `FPhipp2p_0, ..., FPhipp2p_60, TCM`):
1. `op = OperatorFromString(modelspace, name)` in the **emax-14** model space (A = 27).
2. `op = hf.TransformHOToNATBasis(op).DoNormalOrdering()` (HFMBPT.cc:306-375; Operator.cc:400-408):
   1b `C_HO2NAT^T O C_HO2NAT` (works for reduced tensor matrix elements because C is block diagonal
   in (l,j,tz)); 2b transformed with D matrices (the bare operators have no 2b). Normal ordering
   w.r.t. the 27Al holes (fractional occ). Zero body printed as `<name>_HF`.
3. `op = op.Truncate(modelspace_imsrg)` (emax 10, e2max 20).
4. `op = imsrgsolver.Transform(op)` = `Transform_Partial(op,0)` (IMSRGSolver.cc:892-937):
   first `BCH_Transform(op, Omega_1)` with Omega_1 read from the scratch file (log "Transforming using
   .../OMEGA_004096000"), then `BCH_Transform(op, Omega_2)` with the in-memory Omega.
   I.e. `O' = e^{Omega_2} ( e^{Omega_1} O e^{-Omega_1} ) e^{-Omega_2}`, each factor a separate BCH series
   (Omegas are not combined with BCH_Product).
   * Scalars (rank_J+rank_T+parity = 0: FM0, FPhipp0, TCM) use `CommutatorScalarScalar`; tensors
     (FM2/4, FPhipp2/4, FSigma', FSigma'', FDelta) use `CommutatorScalarTensor` = comm111st, comm121st,
     comm122st, comm222_pp_hh_221st, comm222_phst, then Symmetrize/AntiSymmetrize according to the
     resulting flag (Commutator.cc:121-160, 444-505). Tensor results have no zero body.
   * The BCH transform prints `Tensor BCH, i=<k> Norm = <1b> <2b> <total>` for every nested commutator of a
     tensor operator (Commutator.cc:633-640).
5. Re-normal ordering to the core (`renormal_order` is true because the 27Al holes != 16O core;
   imsrg++.cc:1187-1209, 1500-1519): `op.UndoNormalOrdering()` (w.r.t. the 27Al holes of
   modelspace_imsrg), `op.SetModelSpace(ms2)` (ms2 = modelspace_imsrg with reference = core,
   `SetReference(core)` -> Init -> Aref=16, Zref=8, **TargetMass = 16**), `op.DoNormalOrdering()`
   (w.r.t. 16O). Scalars additionally get `.vs1b/.vs2b/_coreNO_e_2.me1j/.me2jp` dumps
   (imsrg++.cc:1513-1518; not part of the archive). Zero body printed as `<name>_IMSRG`.
6. Write (imsrg++.cc:1529-1558): scalar (and even legs) -> `WriteTokyo(op, intfile+name+".snt", "op")`;
   otherwise `WriteTensorTokyo(intfile+name+"_2b.snt", op)`.

`write_omega=true` does **not** control the operator transformation: Omegas are always kept (scratch +
memory) for magnus. `write_omega` only (after all operators are done, imsrg++.cc:1630-1696) flushes the
in-memory Omega_2 to scratch (`OMEGA_..001`, log line 1238), writes `<intfile>_occ.dat` (27Al occupations
of the emax-14 space), `<intfile>_occ_nat.dat` (|NAT eigenvalues|) and tries `hf.C_HO2NAT.save(intfile+"C.mat")`
(not in the archive). The copy of the Omega files to the output dir is commented out (imsrg++.cc:1639),
and the SLURM script deletes the scratch dir. So the Omegas are not archived.

### 1.9 Hamiltonian re-normal-ordering and writer
imsrg++.cc:1210-1287: `HNO = H(s)`, printout of 27Al-NO s.p.e., `UndoNormalOrdering` (27Al),
`SetModelSpace(ms2)`, `DoNormalOrdering` (16O), extra dumps `.vs1b .vs2b _coreNO_e_2.me1j/.me2jp`,
`FlowingOps[0] = HNO`, then `WriteTokyo(H, intfile+".snt", "")` (no isospin averaging).
No additional flow after re-normal-ordering (commented out).

### 1.10 Writers (ReadWrite.cc:5505-5640 WriteTokyo, 5733-5830 WriteTensorTokyo; package excerpt)
WriteTokyo (scalars and H):
* header: build version, input files, `e1max e2max e3max hw`, `Aref Zref` (= 16, 8 of ms2),
  `A_for_kinetic_energy: GetTargetMass()` (= **16**, a label artifact of `SetReference(core)`;
  H and TCM were built with A = 27), `Zero body term: <ZeroBody>` with default stream precision
  (**6 significant digits**).
* valence orbits protons first then neutrons (kshell index 1..6: pi d5,d3,s1, nu d5,d3,s1).
* 1b: pairs a >= b (imsrg orbit index), skipped if `|me| < 1e-7`; `scientific, precision 6`
  (7 significant digits), width 18.
* 2b: every vv bra/ket pair with iket >= ibra in every channel, **no threshold** (zeros written),
  `GetTBME_norm` (normalized, antisymmetrized), format as 1b.
* mode "op" (operators only): for pn kets (`oa.tz2 != ob.tz2`) the written value is
  `( <ab|O|cd> + <a'b'|O|c'd'> ) / 2` where `x' = x - tz2_x` is the isospin mirror orbit
  (proton index even, neutron = proton+1). This symmetrizes the pn two-body part of the proton-only and
  neutron-only scalar operators (FM0p/n, FPhipp0p/n). H is written with mode "" (no averaging).

WriteTensorTokyo (all tensor operators):
* header: `Aref Zref` = 16 8, `A_for_kinetic_energy: 16`, `Zero body term: 0`; "! reduced matrix elements".
* 1b: **all** (a,b) pairs of valence orbits (both orders), skipped if `|me| < 1e-7`; printed as
  `setw(12) << " " << scientific << setprecision(6) << me` (fork fix 11f89d7a; 7 significant digits).
* 2b: for every stored channel pair (only `ch_bra <= ch_ket`, i.e. J_bra <= J_ket for the parity-even,
  Tz-conserving operators here) and **all** vv bra x vv ket (no ibra <= iket restriction), skipped if
  `|me| < 1e-7`; columns `a b c d J_bra J_ket me`; value = stored normalized reduced m.e.
  `<ab J_bra || O || cd J_ket>`. The 2b numbers use `setprecision(6)` with the scientific flag that is
  sticky from the 1b loop; if no 1b element is written, 2b values would come out in default (6
  significant digit, non-scientific) format.
* The reader must rebuild J_bra > J_ket pairs by Hermiticity: `<J_ket||O||J_bra> = (-1)^{J_bra-J_ket} <J_bra||O||J_ket>`
  (what the fork's `GetTBME_norm` does, TwoBodyME.cc:147-150, **always assuming a Hermitian operator**).

Devel differs: WriteTokyo/WriteTensorTokyo use `precision 16` (fixed) and relative thresholds
(`OneBodyNorm()*1e-7`). A reproduction compared against the archived `.snt` must use the 79e9330 format
(6-digit scientific, absolute 1e-7) or compare numerically with tolerance >= 5e-7 relative.

### 1.11 Fork vs devel for these steps (summary)
| Step | Fork 79e9330 | devel |
|---|---|---|
| `use_HF_reference_in_NAT` / `UseHFForHoleStates` | present | absent |
| NAT re-sorting by |occupation| (HFMBPT.cc:131-182) | present, triggered | absent (signed descending order only) |
| `method=magnus_backoff` (Solve_magnus_backoff) | present (unused here) | present |
| DarkMatterNREFT.cc and the `F*` shorthands | present | absent |
| Tensor normal ordering, a==b, J_bra != J_ket | **bug: one ordering missing** | fixed (`ME *= 2`, Operator.cc:555-558, "bug found by Antoine Belley, May 2025") |
| 9j skip threshold in tensor commutators | `|9j| < 1e-8` (Commutator.cc:12132 etc.) | `1e-10` (Commutator/TensorCommutators.cc:624 etc.) |
| Early return for zero-norm operands in commutators/BCH (ed6a99c3) | present | equivalent |
| Writers | 6-digit scientific, absolute 1e-7 | 16-digit, relative threshold |
| Generator / Solve_magnus_euler / BCH thresholds and stopping rule | as described | same logic (BCH.cc:19-20, 104-213, 309-341) |

---------------------------------------------------------------------------------------------------

## 2. The operators

### 2.1 Parsing (imsrg_util.cc, `OperatorFromString`, lines 55ff)
The name is split on "_" (imsrg_util.cc:41-60): `<shorthand>_<q>`; `q` read with `istringstream >> double`
in **MeV**. Each shorthand is an explicit `else if` branch; L and p/n are part of the shorthand.

| Shorthand | Lines | Code | Prefactor |
|---|---|---|---|
| FM0p, FM2p, FM4p, FM0n, FM2n, FM4n | 199-245 | `sqrt(4*M_PI) * DM_NREFT::M(ms,"p"/"n",L,q)` | sqrt(4 pi) |
| FPhipp0p/2p/4p, FPhipp0n/2n/4n | 247-293 | `sqrt(4*M_PI) * DM_NREFT::Phipp(ms,"p"/"n",L,q)` | sqrt(4 pi) |
| FSigma0p/0n | 295-309 | `FSigma0p_Op` = `1.0 * DM_NREFT::Sigma(ms,"p",0,q)` (imsrg_util.cc:~1580) | 1 (unused; Sigma_0 = 0) |
| FSigmap1p/3p/5p, ...n | 311-357 | `DM_NREFT::Sigmap(ms,"p"/"n",L,q)` | 1 |
| FSigmapp1p/3p/5p, ...n | 359-405 | `DM_NREFT::Sigmapp(...)` | 1 |
| FDelta1p/3p/5p, ...n | 407-453 | `DM_NREFT::Delta(...)` | 1 |
| TCM | 111 | `TCM_Op(ms)` (imsrg_util.cc:1032ff): 1b `p^2/(2mA)` (HO formula with hw/A), 2b `p_i.p_j/(mA)` via Moshinsky; A = `GetTargetMass()` = 27 | |
| DMNREFT_<name><iso>_<q>_<J> | 541-567 | generic: `<name>` in {M, Sigma, Sigmap, Sigmapp, Delta, Deltap, Phip, Phipp, Phitp, Omega, Omegat}, `<iso>` in {+,-,p,n} (last character), raw DM_NREFT operator, **no prefactor** | 1 |

History: the sqrt(4 pi) was deliberately removed from FSigma/FSigmap/FSigmapp/FDelta in commit 8afeca0d
("Fix normalization of responses", 2024-01-23; the old lines are kept as comments `// const double norm =
sqrt(4 * M_PI);`). Higher multipoles were added in 8a25ca86 (2024-02-05). A second, unreachable
`opnamesplit[0] == "FM0p"` branch (FourierBesselCoeff) exists further down (dead code).

**There is no `FPhitp` shorthand** at 79e9330, nor on any branch of the fork (searched all 24 branches,
including bleeding_edge head 6fcaad0, master 771907c, phip_fix 81cf5ea, mpi_responses). The only way to
get tilde-Phi' is the generic `DMNREFT_Phitp<p|n|+|->_<q>_<J>` (raw, no sqrt(4 pi)), and at 79e9330 that
operator carries the wrong Hermiticity flag (2.4).

### 2.2 DM_NREFT building blocks (DarkMatterNREFT.cc; port of D. Gazda's Fortran)
Common: `b^2 = 1/(hw * M_NUCLEON)` [MeV^-2] (hbar c = 1 units; equals (hbar c)^2/(m_N hw) in fm^2),
`y = q^2 b^2 / 4`, `M_NUCLEON = 938.9187` (PhysicalConstants: (938.2720813+939.5654133)/2),
isofactor `{p, n}` = {1,1} for "+", {1,-1} for "-", {1,0} for "p", {0,1} for "n"
(applied with the bra orbit's species, `isofactor[(oa.tz2+1)/2]`; the operators are tz-diagonal, so
bra and ket species agree). Tz rank 0, particle rank 2 (2b part zero). Stored as reduced
matrix elements `<a||O_J||b>` (Wigner-Eckart `<jm|T_JM|j'm'> = (-1)^{j-m} (j J j'; -m M m') <j||T||j'>`);
**for J = 0 and parity 0 the value is divided by sqrt(2j_a+1)** (plain matrix element, because rank-0
operators are stored non-reduced). Radial integrals of HO functions in terms of 1F1:

* `jho(na,la,nb,lb,L,y)` (60-85) = <na la| j_L(qr) |nb lb>;
  `jdmho`, `jdpho` (127-231): <na la| j_L(qr) (d/dr -/+ ...)/q |nb lb> combinations (Gazda's jdmHO/jdpHO).
* `M(J)` (259-305): parity J%2 (normal),
  `<a||M_J||b> = (1/sqrt(4pi)) (-1)^{(2J+jb2+1)/2} sqrt((2la+1)(2lb+1)(ja2+1)(jb2+1)(2J+1))
  {la ja 1/2; jb lb J} (la J lb; 0 0 0) jho(na,la,nb,lb,J,y)`.
  (A comment `//bhu` shows Gazda's `2*J` argument was corrected to `J`.)
* `Ms(J,L)` (326-368): parity L%2,
  `<a||M_JL.sigma||b> = sqrt(3/(2pi)) (-1)^la sqrt((2la+1)(2lb+1)(ja2+1)(jb2+1)(2L+1)(2J+1))
  {la lb L; 1/2 1/2 1; ja jb J} (la L lb;000) jho(na,la,nb,lb,L,y)`.
* `Mg(J,L)` (399-450): parity (L+1)%2 (abnormal), zero if la==0 or lb==0,
  `(1/sqrt(4pi)) (-1)^{(2L+jb2+1)/2} sqrt((2la+1)(ja2+1)(jb2+1)(2L+1)(2J+1)) {la ja 1/2; jb lb J}
  [ -sqrt(lb+1) sqrt(2lb+3) {L 1 J; lb la lb+1} (la L lb+1;000) jdmho(..L..)
    + sqrt(lb) sqrt(2lb-1) {L 1 J; lb la lb-1} (la L lb-1;000) jdpho(..L..) ]` = <a|| M_JL . (1/q)grad ||b>.
* `Sigma(J) = Ms(J,J)`, flagged **anti-Hermitian** (460-465).
* `Sigmap(J) = -sqrt(J/(2J+1)) Ms(J,J+1) + sqrt((J+1)/(2J+1)) Ms(J,J-1)` (478-483), default (Hermitian) flag.
* `Sigmapp(J) = sqrt((J+1)/(2J+1)) Ms(J,J+1) + sqrt(J/(2J+1)) Ms(J,J-1)` (499-504), Hermitian flag.
* `Delta(J) = Mg(J,J)` (516-519), Hermitian flag at 79e9330 (master 5bea403f sets anti-Hermitian; see 2.4).
* `Deltap(J) = -sqrt(J/(2J+1)) Mg(J,J+1) + sqrt((J+1)/(2J+1)) Mg(J,J-1)`, anti-Hermitian flag (532-538).
* `PhiF(la,ja,jb) = (-1)^{la+1} 6 sqrt((2la+1)(ja2+1)(jb2+1))/sqrt(4pi)` (550-553);
  `PhiS1..PhiS4` (582-737): sums over lambda of 6j x 6j x 9j times jdmho/jdpho at multipole J+1 (S1,S2)
  or J-1 (S3,S4).
* `Phip(J)` (763-808), parity J%2 (normal):
  `PhiF * [ -sqrt(2J+3) sqrt(J) (S1+S2) + (J>0) sqrt(2J-1) sqrt(J+1) (S3+S4) ]`, **flagged anti-Hermitian**
  (`Phip_op.SetAntiHermitian()` at line 776, before filling).
* `Phitp(J) = Phip(J) + 0.5 * Ms(J,J)` (816-820) = Gazda's `ME1b%Phitp = Phip + Ms(J,J)/2` = AFH
  tilde-Phi'_J = Phi'_J + Sigma_J/2. Built with `Operator::operator+` (Operator.cc:192-198): the result is a
  copy of the left operand (both have particle rank 2), so it **inherits Phip's anti-Hermitian flag**.
* `Phipp(J)` (848-891), parity J%2:
  `PhiF * [ sqrt(2J+3) sqrt(J+1) (S1+S2) + (J>0) sqrt(2J-1) sqrt(J) (S3+S4) ]`, Hermitian flag.
* `Omega(J)` (923-980, abnormal parity, prints a warning that it has no symmetry), `Omegat = Omega + Sigmapp/2`
  (anti-Hermitian flag). Not used.

### 2.3 q -> 0 limits (from scripts/q0.py and audit/bare_ops_check.log; these are exactly AFH's operators)
* sqrt(4pi) M_0 -> 1 (number operator), Sigma'_1 -> sigma/sqrt(6pi), Sigma''_1 -> sigma/(2 sqrt(3pi)),
  Delta_1 -> -l/(2 sqrt(6pi)) (raw, no prefactor; these equal AFH's long-wavelength forms).
* sqrt(4pi) Phi''_0 -> (1/sqrt3) [l x sigma]^0 = -(1/3) sigma.l.
* sqrt(4pi) Phi''_2 -> -(1/sqrt30) [l x sigma]^2.
* sqrt(4pi) tildePhi'_2 -> -(1/(2 sqrt5)) [l x sigma]^2 (ratio tildePhi'_2 / Phi''_2 = sqrt(3/2) at q = 0).
  (Sigma_2 = Ms(2,2) vanishes on the diagonal, so Phi'_2 and tildePhi'_2 agree on the diagonal.)

### 2.4 tilde-Phi' (Phitp): answer
* Definition: `Phitp(J) = Phip(J) + 0.5*Ms(J,J)` = tilde-Phi'_J = [(1/q)grad x M_JJ].[sigma x (1/q)grad] + Sigma_J/2
  (AFH, Fitzpatrick et al. JCAP 2013 / PRC 89 065501). Spherical tensor of rank J, **normal parity (-1)^J**,
  isospin as chosen (p/n/+/-), Tz rank 0, one-body (particle rank 2 container).
* Hermiticity (numerical, scripts/herm.py, 12 orbits up to 0g9/2, q = 150 MeV, convention h = <b||O||a>/((-1)^{ja-jb}<a||O||b>)):
  Phitp is **Hermitian** (h = +1 for all J = 1..4); Phi' alone is **neither** (h between -9.8 and 1.4);
  Sigma = Ms(J,J) is **anti-Hermitian**. This is consistent with the operator identity
  A.grad = (A.grad + grad.A)/2 - (div A)/2 with A = (grad x M_JJ) x sigma / q^2, div A = Sigma_J:
  Phi' = (Weyl-ordered part) - Sigma/2, so adding Sigma/2 leaves exactly the symmetric part.
* At 79e9330 the flag is wrong: Phitp inherits `antihermitian` from Phip. Every flag-dependent step would
  then corrupt it: `DoNormalOrdering2` ends with `AntiSymmetrize()` (Operator.cc:566-567) which overwrites
  the lower triangle with the wrong sign, commutators set the result flag from the operands
  (Commutator.cc:464-466) and symmetrize accordingly, `TwoBodyME::GetTBME_norm` assumes Hermitian.
  **Any FPhitp evolution must set the Hermitian flag**, as phip_fix 81cf5ea1 (2026-08-06) does
  (`Phitp_op.SetHermitian()`; Phip loses its anti-Hermitian flag and prints a warning that it is neither).
* Elastic selection: Hermitian + real reduced m.e. + parity even -> survives for **J even**: L = 2, 4 for
  27Al (j = 5/2 allows J <= 5; Phitp_0 = 0 identically because Phip_0 has a factor sqrt(J) = 0 in the
  S1+S2 term and no S3/S4 term, and Ms(0,0) = 0 by the triangle rule).
* Normalization for a shorthand `FPhitp<L><p|n>_<q>`: the archived columns use sqrt(4 pi) for the
  normal-parity, even-J "charge-like" family (M, Phi''; these interfere in W_{M Phi''}) and 1 for the
  abnormal-parity family (Sigma', Sigma'', Delta; Sigma' and Delta interfere in W_{Delta Sigma'}).
  tilde-Phi' is normal parity, even J, and is the transverse partner of Phi'' in the same spin-velocity
  current (as Sigma' is the transverse partner of Sigma''), so the consistent choice is
  **`FPhitp<L>p/n = sqrt(4 pi) * DM_NREFT::Phitp(ms, "p"/"n", L, q)`, Hermitian, L in {2,4}**. Then
  FPhitp2(q->0) = -(1/(2 sqrt5)) [l x sigma]^2 per nucleon, parallel to FPhipp0 -> -(1/3) sigma.l and
  FPhipp2 -> -(1/sqrt30)[l x sigma]^2. W_{tilde Phi'} has no interference term in AFH, so the factor only
  scales that response; with AFH's W = (4pi/(2j+1)) sum_J |<j||O_J||j>|^2 the sqrt(4pi)-scaled columns
  (FM, FPhipp, and FPhitp by this choice) already carry the 4pi, the others do not.
  This is a recommendation; the owner should confirm (QUESTIONS entry suggested).
* Later fixes touching Phip/Phitp/Delta (code_reference diffs): master 5bea403f (2024-10-23) adds debug
  printing and **flags Delta anti-Hermitian** (numerically Delta is Hermitian in this convention, see 3;
  so that change looks wrong for these matrix elements, unless the storage convention changed elsewhere);
  phip_fix 81cf5ea1 removes Phip's anti-Hermitian flag and sets Phitp Hermitian (correct).
  Other master commits after 79e9330 that touch tensor operators: 4c840df (2025-05, A. Belley) "Fix NO
  tensor operator bug" (the a==b normal-ordering term, see 4.1), 2f4234d/a05c9fd/f2821ff (parity-violating
  generators and operators; not relevant for parity-even operators), 5824409 (Rp spin-orbit correction;
  not used here).

---------------------------------------------------------------------------------------------------

## 3. Hermiticity, parity, rank, time reversal

Convention (Operator.cc:310-320, 1814-1863): tensor Hermitian means `O(j,i) = (-1)^{(j_i-j_j)} O(i,j)` for
1b reduced m.e. and `<J_ket||O||J_bra> = (-1)^{J_bra-J_ket} <J_bra||O||J_ket>` for 2b; anti-Hermitian
with an extra minus. Numerical classification (scripts/herm.py), flags at 79e9330 in brackets:

| Operator | Rank | Parity | Hermiticity (numeric) [flag 79e9330] | Elastic (27Al, j=5/2) |
|---|---|---|---|---|
| M_J (FM) | J | (-1)^J | Hermitian [H] | J = 0, 2, 4 |
| Phi''_J (FPhipp) | J | (-1)^J | Hermitian [H] | J = 0, 2, 4 |
| tilde-Phi'_J (Phitp) | J | (-1)^J | Hermitian [**anti**, wrong] | J = 2, 4 |
| Phi'_J (Phip) | J | (-1)^J | neither [anti] | - |
| Sigma_J = Ms(J,J) | J | (-1)^J | anti-Hermitian [anti] | vanishes (anti-Hermitian, real) |
| Sigma'_J (FSigmap) | J | (-1)^{J+1} | Hermitian [H] | J = 1, 3, 5 |
| Sigma''_J (FSigmapp) | J | (-1)^{J+1} | Hermitian [H] | J = 1, 3, 5 |
| Delta_J (FDelta) | J | (-1)^{J+1} | Hermitian [H] (master: anti) | J = 1, 3, 5 |
| TCM | 0 | + | Hermitian [H] | expectation value |

Time reversal: with real HO radial functions and this phase convention all used operators have real reduced
matrix elements. A tensor-Hermitian operator with real m.e. can have a nonzero diagonal <j||O_J||j>; a
tensor-anti-Hermitian one cannot (its diagonal reduced m.e. must equal minus itself). Together with parity
this reproduces AFH's elastic selection: M, Phi'' (T-even charge/longitudinal), Sigma', Sigma'', Delta
(T-odd magnetic/longitudinal spin, odd J), tilde-Phi' (even J >= 2); Sigma_J, Phi'_J (non-tilde),
Delta', Omega do not contribute elastically. Delta's Hermitian character follows from div(M_JJ) = 0
(no ordering ambiguity in M_JJ . grad). Parity codes in the fork: 0 = even, 1 = odd; all operators used
here have parity 0 (even J for normal parity, odd J for abnormal parity).

---------------------------------------------------------------------------------------------------

## 4. What a reproduction must replicate to match the archived .snt at the 1e-4 level

### 4.1 Fork-specific behaviors (must implement, or knowingly deviate)
1. **use_HF_reference_in_NAT** exactly as `UseHFForHoleStates` (1.4.3): reset hole columns to HF,
   zero the hole rows of all other columns, modified Gram-Schmidt in orbit-index order within each (l,j,tz)
   channel, normalize, rebuild rho/F from the HF holes, sign convention C_HF2NAT(i,i) > 0.
2. **NAT re-sorting by |occupation|** (1.4.2) before step 1. Negative MBPT2 eigenvalues (up to -0.0136
   here) move low-n labels and therefore change the emax-10 truncation. Without it the IMSRG model space
   differs.
3. **Tensor normal-ordering bug** (Operator.cc:520-555 at 79e9330, fixed in devel Operator.cc:555-558):
   in `DoNormalOrdering2`, for tensor operators the contraction `<a h J_bra||O||b h J_ket>` is summed only
   over stored channel pairs (ch_bra <= ch_ket); for a != b the reversed pair is recovered through the
   `a > b` branch, but for **a == b with J_bra != J_ket the reversed-pair term is missing**. It enters the
   final Undo(27Al)/Do(16O) step of every tensor operator; the core contributions cancel, leaving
   `archived - correct = sum_{h in {pi 0d5/2 (n=5/6), nu 0d5/2 (n=1)}} n_h S'_h(a)` on the **diagonal 1b reduced
   m.e.** (S' = the stored J_bra < J_ket part of the contraction). Estimated from the archived 2b parts
   (scripts/nobug.py; same conventions as the fork; magnitude reliable, signs depend on my reconstruction
   of the phase conventions):
   * N2LOsat FPhipp2p q=60: delta = +1.1e-2 (pi d5, value -0.922), -4.6e-2 (nu d5, value 0.083),
     -6.4e-2 (pi d3, value 0.197), -4.7e-2 (nu d3, value -0.058).
   * FSigmap1p q=150: up to 1.1e-2 (pi s1/2, value 0.079); FDelta1p q=150: up to 1.3e-2;
     FSigmapp1n q=150: up to 4.6e-3.
   These are far above 1e-4. A devel-based reproduction (with the fix) will **not** match the archived tensor
   `.snt` diagonal 1b elements unless the bug is emulated (e.g. a switch that drops the reversed-pair term in
   the re-normal-ordering), and the archived tensor responses themselves carry this error.
   Scalars (FM0, FPhipp0, TCM, H) are unaffected (only ch_bra == ch_ket stored).
4. Operator definitions and prefactors of section 2 (sqrt(4pi) only for FM, FPhipp; J=0 stored as plain
   m.e.; isofactor by species; Hermitian flags as at 79e9330, all five families Hermitian).
5. Writers in the 79e9330 format (1.10): 7 significant digits, absolute 1e-7 thresholds (1b always,
   2b only in the tensor writer), all orderings in the tensor writer, `ch_bra <= ch_ket` only, isospin
   averaging of pn 2b in `WriteTokyo(...,"op")` for scalar operators (not for H, not for tensors),
   6-digit zero body, header `A_for_kinetic_energy: 16`.
6. 9j skip threshold 1e-8 in the tensor commutators (devel 1e-10): very likely below 1e-7, but a source of
   differences in principle.

### 4.2 Solver details shared with imsrg-devel that must also be matched
* Two-step decoupling: atan (core = 16O vs everything, kets cc only) then shell-model-atan, with
  27Al-ensemble normal ordering and Epstein-Nesbet monopole denominators using fractional occupations,
  cutoff |D| < 1e-6 -> +1e-6, denominator_delta 0.
* Magnus Euler with adaptive ds = min(0.2/|eta|, 0.2/(|eta||Omega|), 50000/|eta|, 0.5), s continues
  across stages, smax 100 then 200, eta_criterion 1e-3 on the full (1b+2b, weighted) norm, checked before
  each step. With eta_criterion 1e-3 the end point is a discontinuous function of the step history; to hit
  the archived step counts (37/272 and 30/253) the norms must agree to many digits.
* One Omega per stage (NSmagnus, omega_norm_max 50000): Omega_1 stored at the generator switch, H(s) in
  stage 2 = BCH_Transform(H_saved, Omega_2).
* BCH_Product (Commutator.cc:723-790): Z = X + Y + 1/2[X,Y]; + 1/12 [X,[X,Y]] if `||[Y,X]|| ||X|| > 1e-4`;
  then Bernoulli series in ad_Y ([Y,[Y,X]] 1/12, -1/720 ...) while `||nested|| > 1e-4`, break if
  `2||Y|| ||nested|| < 1e-4` or k >= 9.
* BCH_Transform (Commutator.cc:561-650): skipped if `||Omega|| == 0`; terms added while needed with
  `epsilon_0 = ||O|| exp(-2||Omega||) 1e-9 / (2||Omega||)`, after the i-th term `epsilon *= (i+1)`, stop when
  `||ad^i_Omega O|| < epsilon` (i.e. ||ad^i O||/i! < epsilon_0 (i+1)), hard limit 40, informational message
  "not converged after 12 nested commutators" at i = 12 (seen ~220 times per log; harmless). Skipped
  entirely if `||O|| <= 1e-9`. 1b-only operators are promoted to particle rank 2 first.
* Operators are transformed with Omega_1 then Omega_2 separately (not with a combined Omega).
* Re-normal-ordering from the 27Al ensemble to the 16O core for H and all operators after the transformation;
  no further flow.
* Hermiticity-flag logic of the commutators (Z anti-Hermitian for (H,H) or (A,A), Hermitian otherwise,
  then Symmetrize/AntiSymmetrize).
* Truncation 14 -> 10 by cutting already normal-ordered matrices (no re-normal-ordering), e2max 20.
* NO2B 3N with single precision storage, E3max 24, |rho| < 1e-8 skip.

### 4.3 Precision expectations
* `.snt` values have 7 significant digits; 1b elements below 1e-7 are absent (FDelta5 at q = 15 is
  exactly zero for that reason); H's zero body is printed with 6 significant digits (-118.293).
* The archived shell-model numbers are converged only to ~1e-6 MeV in energy (README section 6), so
  matching the CSV to better than ~1e-3 relative for small tensor matrix elements is not meaningful;
  matching the `.snt` to 1e-4 absolute is meaningful only with items 4.1.1-4.1.5 implemented.
* Scalar zero bodies are printed to 24 digits in STDOUT (`<op>_HF`, `<op>_IMSRG`), a good first
  checkpoint (e.g. TCM_IMSRG = 6.956645787277093 for N2LOsat).

---------------------------------------------------------------------------------------------------

## 5. Open points
* The `+` in the build string (uncommitted changes) remains unknown; nothing in the logs contradicts the
  79e9330 source (all messages found, e.g. "Setting occupied states back to HF", the `_old ->` re-sorting
  printout, "Tensor BCH").
* Confirm with the owner: FPhitp normalization (sqrt(4 pi) recommended) and whether to emulate the tensor
  normal-ordering bug for the comparison (and to recompute the published tensor responses without it).
* The sign/phase details of the bug estimate (scripts/nobug.py) should be cross-checked once the new code
  can produce both variants.
