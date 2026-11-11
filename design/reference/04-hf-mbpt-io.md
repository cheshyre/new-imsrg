# Reference analysis 04: Hartree-Fock, HFMBPT / natural orbitals, RPA, and file I/O (imsrg++)

Source tree analysed (read-only): `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel/src`.
All `file:line` refs are relative to that `src/` unless otherwise stated.
Line counts: HartreeFock.cc 1722, HFMBPT.cc 4440 (about 3000 lines are 39 hand-written MBPT(4) diagrams), RPA.cc 612, ReadWrite.cc 6393, Jacobi3BME.cc 1145, imsrg_util/Pwd.cc 436.

Conventions used in this document:
- "HO basis" = the spherical oscillator basis the interaction files live in. "HF basis" = columns of `C`.
- Orbit labels: modelspace orbit index `i`, `Orbit{n,l,j2,tz2}`; tz2 = -1 is a proton, +1 a neutron.
- All J-coupled matrix elements are stored NORMALIZED (the `sqrt(1+d_ab)` factors are removed), see section 4.1.

---------------------------------------------------------------------------------------------------

## 0. Where these pieces are used (driver flow, `imsrg++.cc`)

The driver is the only real "workflow" consumer of this code; the rewrite should reproduce this sequence as a library pipeline.

1. Build `ModelSpace` (emax, e2max, e3max, reference, valence, optional `occ_file`), then `Hbare = Operator(ms,0,0,0, rank 2 or 3)` (imsrg++.cc:431).
2. Read 2N (`fmt2`: me2j/navratil/oslo/oakridge/takayuki/nushellx/schematic, imsrg++.cc:440-466) and 3N (imsrg++.cc:470-492).
3. Add kinetic energy `Trel_Op` (imsrg++.cc:~503). `Trel = Tlab - TCM` (imsrg_util.cc:743-748), where `TCM` has both a 1-body and a 2-body (`p_i.p_j`) part and uses `A = modelspace target mass`. Optional extras: `nucleon_mass_correction` (Trel_Masscorrection_Op), `relativistic_correction` (p^4 term, imsrg_util.cc:751ff), Lawson `BetaCM*HCM_hw` term (imsrg++.cc:~525).
   **There is no CM / relativistic handling inside HartreeFock/HFMBPT themselves**: HF just sees whatever `Hbare.OneBody` and `Hbare.TwoBody` contain. `KE` in HartreeFock is a copy of `Hbare.OneBody` taken in the constructor (HartreeFock.cc:21).
4. `HFMBPT hf(Hbare)` (inherits HartreeFock); `hf.Solve()` if `basis != "oscillator"` (imsrg++.cc:560-568).
5. If `basis=="NAT"` (or `OccNat3Cut>0`): `UseNATOccupations`, `OrderNATBy`, `GetNaturalOrbitals()` (imsrg++.cc:570-578).
6. `hf.SetModelspaceForOutput3N(modelspace_imsrg)` (smaller emax/e3max for the IMSRG part), then `HNO = hf.GetNormalOrderedH(rank)` (HF) or `hf.GetNormalOrderedHNAT(rank)` (NAT) or `Hbare.DoNormalOrdering()` (oscillator) (imsrg++.cc:645-690).
7. Every other operator: `TransformToHFBasis` (or `TransformHOToNATBasis`) and then `DoNormalOrdering()` (imsrg++.cc:820-830). Note: transformation and normal ordering are separate steps for general operators, but fused for the Hamiltonian.
8. `method=="MP3"`/etc. call `HNO.GetMP2_Energy()`, `GetMP3_Energy()` (Operator members, imsrg++.cc:739-744). The `HFMBPT::GetMP*` methods are duplicates living on the HFMBPT class.
9. After the IMSRG, outputs: `.snt` (Tokyo), `.int/.sp` (NuShellX), Antoine, `C.mat` (armadillo binary, `hf.C.save`, imsrg++.cc:1687-1691), `*_occ.dat`, `*_occ_nat.dat`, spwf files.
10. `brueckner_restart`: `newC = hf.C * expm(-Omega_1b)`, `hf.GetNormalOrderedH(newC)` re-normal-orders with a rotated basis, then re-runs the IMSRG (imsrg++.cc:1112-1119).

---------------------------------------------------------------------------------------------------

## 1. HartreeFock (HartreeFock.hh/.cc)

### 1.1 Data model

Members (HartreeFock.hh:35-75): `C` (norb x norb, `C(i_HO, alpha_HF)`, block diagonal in (l, j, tz)), `rho`, `KE`, `Vij`, `V3ij`, `F`, `Vmon[3][2]` and `Vmon_exch[3][2]` (monopole 2-body per (Tz, parity)), `Vmon3_keys`/`Vmon3` (sparse monopole 3-body), `holeorbs` (uvec of orbit indices that are occupied), `hole_occ` (rowvec of fractional occupations), `energies`, `prev_energies`, `tolerance=1e-8`, `EHF,e1hf,e2hf,e3hf`, `freeze_occupations` (ctor default **true**), `discard_NO2B_from_3N` (diagnostic), `use_jacobi_3body`/`jacobi3bme` (experimental, see 1.6), `ms_for_output_3N`.

Everything is "per-J-reduced" in the sense that rho is the *scalar* density matrix in the m-scheme-averaged sense: `rho(a,b)` is nonzero only within the same (l,j,tz) block, and the weight `(2j+1)` is applied when summing (traces).

### 1.2 Algorithm (Solve, HartreeFock.cc:55-166)

Fixed hard-coded parameters (HartreeFock.cc:59-62): `maxiter=1000`, `density_mixing_factor=0.2`, `field_mixing_factor=0.0`, `fill_modulus=3`. None are user-settable except `tolerance` (public member) and freeze/unfreeze.

Sequence:
1. `BuildMonopoleV()`; if `Hbare.GetParticleRank()>2` also `BuildMonopoleV3()`.
2. `UpdateDensityMatrix()` with `C = I` (so the initial `rho` is the naive HO filling from `ModelSpace` occupations), then `UpdateF()`.
3. Loop `iterations = 0..999`:
   a. `Diagonalize()`: for each one-body channel (l,j,tz) (`Hbare.OneBodyChannels`) take the sub-block `F_ch`, `arma::eig_sym`, write eigenvalues into `energies(orbvec)` and eigenvectors into `C.submat(orbvec,orbvec)` (HartreeFock.cc:211-245). `prev_energies = energies` is saved first. Up to 5 retries on diag failure (it does not change the matrix between retries; this is a no-op retry loop).
   b. `ReorderCoefficients()` (HartreeFock.cc:866-905): bubble-sort columns of C within each channel by energy (eig_sym already sorts ascending, so this is effectively a no-op), then **flip the sign of every column whose diagonal element `C(i,i)<0`** so the diagonal is positive. This fixes the arbitrary eigenvector phases. A reimplementation must apply the same sign convention if it wants bit-comparable C matrices (and `C.mat` files). Caveat: the sign loop is nested inside the channel loop but runs over all columns (idempotent, wasteful).
   c. If not frozen and `iterations>10` and `iterations % fill_modulus == 0`: `FillLowestOrbits()` (re-determine occupations from current spe), then `fill_modulus = (fill_modulus % 5) + 1` (cycle 3,4,5,1,2,3,...). See 1.4.
   d. Rescue schedule: at `iterations==500` set `density_mixing_factor=0.7`, `field_mixing_factor=0.5`; for `iterations>600 and iterations%10==3` set both mixing factors to a **random** value in (0.5,1] using `std::rand()` (non-deterministic across platforms/seeds); at `iterations==800` force `freeze_occupations=true`. (HartreeFock.cc:86-103.)
   e. Density update. Default branch (HartreeFock.cc:122-127):
      `last_rho = rho; UpdateDensityMatrix(); rho = (1-a)*rho_new + a*last_rho` with a=0.2. So the **new** density gets weight 0.8 and the old gets 0.2 (the variable name "density_mixing_factor" is the weight of the OLD density).
      DIIS branch if `iterations>100 and (no error mats yet or ||last DIIS error||_F > 0.01)` (HartreeFock.cc:105). The block `if (iterations==100)` inside that branch can never execute (dead code: the outer condition requires `iterations>100`), so the intended "reset to C=I at the DIIS switch" never happens. Once the DIIS error norm drops below 0.01 the code switches back to linear mixing and never re-enters DIIS (the stored `DIIS_error_mats.back()` is no longer updated).
   f. `UpdateF()`; then `F = (1-b)*F + b*last_F` with b=0 by default (so no field mixing).
   g. `CheckConvergence()`: calls `CalcEHF()`, then `ediff = ||energies - prev_energies||_F / norbits`, converged if `ediff < 1e-8` (HartreeFock.cc:834-845). Note the division by the total number of orbits (norm is an RMS-like measure, not a plain Frobenius norm as the comment above the function claims). Energies of **all** orbits (including high-lying ones) enter.
4. After loop: `CalcEHF()`, print convergence (prints a warning with the last 7 `ediff` and `EHF` values from the deques if `iterations==maxiter`; it does NOT abort), `UpdateReference()`, `PrintEHF()`.

DIIS (`UpdateDensityMatrix_DIIS`, HartreeFock.cc:623-678): error matrix `F*rho - rho*F`; keeps up to 6 (not 5: `N_MATS_STORED=5` but pop happens only when size>5) past densities/errors; builds the Pulay B matrix using `||e_i e_j^T||_F`; solves `B c = (0,..,0,1)`, then **takes `abs(c)` and renormalises to sum 1** (non-standard; standard DIIS allows negative coefficients); the new rho is the combination of the oldest `nsave` stored densities (the most recently pushed one is skipped due to an off-by-one between `nsave` and the post-push deque size). It is a rarely-used rescue, so the rewrite can implement standard Pulay DIIS and document the difference.

Density matrix (HartreeFock.cc:613-618): `rho = sum_h occ_h C(:,h) C(:,h)^T`, i.e. `(C.cols(holeorbs) .* hole_occ) * C.cols(holeorbs)^T`. `holeorbs` is a list of *HF orbit labels* (the column index in C, which coincides with the label of the orbit the HF state is adiabatically connected to by the within-channel energy ordering).

### 1.3 Hamiltonian ingredients: monopole 2-body and 3-body

2-body (`BuildMonopoleV`, HartreeFock.cc:250-275). For each (Tz, parity) block of "monopole kets" (`modelspace->MonopoleKets[Tz+1][parity]`, a map ket index -> local index, apparently all (p<=q) kets with given Tz and parity, J-independent), build
`Vmon(ab,cd) = (2ja+1)(2jb+1) * TBMEmonopole(a,b,c,d)` and `Vmon_exch(ab,cd) = (2ja+1)(2jb+1)*TBMEmonopole(a,b,d,c)`, where `TBMEmonopole` is the un-normalised J-average `sqrt((1+d_ab)(1+d_cd)) sum_J (2J+1) <ab|V|cd>_J / ((2ja+1)(2jb+1))` (see commented doc at TwoBodyME.cc:653ff). Only the upper triangle (`itket.second >= itbra.second`) is computed and then symmetrised with `arma::symmatu`.

Fock matrix (`UpdateF`, HartreeFock.cc:750-832):
`F_ij = KE_ij + (1/(2j_i+1)) sum_{ab} rho_ab Vbar_{iajb} + (1/(2(2j_i+1))) sum_{abcd} rho_ab rho_cd Vbar3_{iacjbd}` (note the code stores `Vij` and `V3ij` and sets `F = KE + Vij + 0.5*V3ij`).
- The `(a>i) xor (b>j)` test selects `Vmon_exch` versus `Vmon` to handle the ordering of the ket label (min,max) (HartreeFock.cc:~790). A reimplementation that stores both orderings can drop this.
- 3-body: `v3ij(i,j) += rho(a,b)*rho(c,d)*Vmon3[ind]`, parallelised with one private `arma::mat` per OpenMP thread then reduced (HartreeFock.cc:~810).
- The 1-body loops over `Hbare.OneBodyChannels`; comment says "this loop isn't thread safe for some reason" yet there is an `omp parallel for` on channels (HartreeFock.cc:759-762). Treat as suspicious.
- Final `symmatu` on both `Vij`, `V3ij`.

HF energy (`CalcEHF`, HartreeFock.cc:169-190): `e1 = sum_i (2j_i+1) sum_j rho_ij KE_ij`, `e2 = (1/2) sum (2j+1) rho_ij Vij_ij`, `e3 = (1/6) sum (2j+1) rho_ij V3ij_ij`; `EHF = e1+e2+e3`. Printed via `PrintEHF` as `e1hf, e2hf, e3hf, EHF`.

3-body monopole (`BuildMonopoleV3`, HartreeFock.cc:287-377):
- `SetUpMonopoleV3Keys` enumerates all `(i<=j; a; b in channel of a; c; d in channel of c)` with `e_a+e_c+e_i <= E3max`, `e_b+e_d+e_j <= E3max`, and total parity even; key = `Vmon3Hash(a,c,i,b,d,j)` (a packed uint64, HartreeFock.cc:554ff).
- Each value is `Hbare.ThreeBody.GetME_pn_mono(a,c,i,b,d,j) / (2 j_i + 1)` (HartreeFock.cc:362-366). `GetME_pn_mono` (ThreeBodyME/ThreeBodyStorage.cc:412) sums `GetME_pn_no2b(a,b,c,d,e,f,J2b)` over `J2b in [|ja-jb|,ja+jb]`; `GetME_pn_no2b` sums `GetME_pn(J2b,J2b,twoJ,...)*(2J+1)` over total twoJ. Note a precedence bug in the *reduced-operator* branch there: `sqrt(2*J2b+1 / (twoJ+1.))` should be `sqrt((2*J2b+1)/(twoJ+1.))` (ThreeBodyStorage.cc:~398). Probably dead for the Hamiltonian but must not be copied.
- Dedicated storage modes exist: `"mono"` storage (file already J-summed) and `"no2b"` storage (J12-coupled NO2B file); with them `GetME_pn_mono` is overridden (ThreeBodyStorage_mono.cc:465, ThreeBodyStorage_no2b.cc:362).
- Memory/time: one `(uint64 key, double)` pair per `(i,j,a,b,c,d)` tuple, so scaling ~ N_orb(channel)^2 * N_orb^2 (N_lj,n); `std::cout` prints the number of doubles stored. 3-body parity and e3max cuts keep it manageable. `FreeVmon()` swaps vectors away after HF.
- `BuildMonopoleV3Jacobi` (HartreeFock.cc:379-552, plus `Jacobi3BME::GetV3mon_all`): header comment says "So far this is completely untested and certainly wrong". Do not port; treat the Jacobi path as experimental (see 1.6).

### 1.4 Occupation handling

- **Initial occupations** come from `ModelSpace`: `Orbit::occ` (0..1; a fraction of the orbit filled), set via the `reference` string, or from `occ_file` (`ModelSpace::Init_occ_from_file`, ModelSpace.cc:319-355). `occ_file` format: whitespace separated lines `n l j2 tz2 occ` with `tz2=-1` proton; occupations below 1e-8 are dropped; duplicate lines with different values print a warning; an unreadable file only prints a warning (does not abort). Such fractional occupations define an "ensemble" reference (fractional occupation for the whole orbit, i.e. each m-state occupied with probability `occ`).
- `holeorbs`/`hole_occ` initialised in the HF ctor from `modelspace->holes` (HartreeFock.cc:36-39).
- **freeze_occupations**: class default `true`; the driver parameter `freeze_occupations` default is `"false"` and the driver calls `hf.UnFreezeOccupations()` if the parameter is false (imsrg++.cc:562). Net effect: **by default occupations are NOT frozen**. When not frozen: re-filling every few iterations after iteration 10 (1.2c) and `GetNormalOrderedH()` calls `UpdateReference()` (HartreeFock.cc:1082-1085).
- **FillLowestOrbits** (HartreeFock.cc:685-748): computes `F_HF = C^T F C`, stable-sorts the orbit indices by `diag(F_HF)`, then fills protons (tz2<0) up to `Zref` and neutrons up to `Nref = Aref - Zref`. The last partially filled orbit gets a fractional occupation `min(1,(remaining)/(2j+1))` (so open-shell nuclei get **equal filling** over the degenerate j-shell: this is "ensemble"/"equal-filling" normal ordering). Orbit labels are used both as "HF state label" and as a row of F_HF (works since C is block diagonal and energy ordered within the channel). Occupation changes are only reported to stdout.
- **UpdateReference** (HartreeFock.cc:1034-1054): if any hole occupation differs by more than 1e-3 from `ModelSpace`'s, calls `modelspace->SetReference(map)` which **re-runs `ModelSpace::Init`** (ModelSpace.cc:810-831: clears all vectors, rebuilds orbits, kets, channels). This mutates the shared model space under the feet of every `Operator` holding a pointer to it; orbit indices stay valid only because emax/orbit sets are unchanged. A rewrite should separate "occupation/reference" from "basis structure" so a reference change does not rebuild the channel tables.
- Natural-orbital fractional occupations: see section 2.

### 1.5 Outputs

- `EHF, e1hf, e2hf, e3hf` (public members).
- `energies` (eigenvalues of the last Fock diagonalisation), and `PrintSPE()`/`PrintSPEandWF()` print `diag(C^T F C)` and the expansion coefficients (HartreeFock.cc:1211-1246). Note: printed spe are diagonal of F in HF basis, which equal the eigenvalues after convergence.
- `C` (`C(i_HO,alpha_HF)`), saved by the driver with `arma::mat::save` (armadillo native binary, file `<intfile>C.mat`).
- `GetRadialWF`, `GetRadialWF_r`, `GetHFPotential`, `GetAverageHFPotential` (HartreeFock.cc:1248-1325) use GSL generalised Laguerre (`gsl_sf_laguerre`) for radial HO wave functions; used by `imsrg_util::WriteSPWaveFunctions` (R mesh: 200 points, Rmax=20 fm).
- `GetOmega()` (HartreeFock.cc:1191): one-body generator `Omega = -log(C)` evaluated by `eig_gen` per channel. Debug use only.

### 1.6 Transformation to the HF basis and normal ordering

`TransformToHFBasis(Operator&)` (HartreeFock.cc:914-1031), applied to any operator (rank J,T,parity arbitrary):
- 1-body: `O_HF = C^T O C` for even-leg operators; `C^T O` for odd-leg (a^dagger-type "DaggerOperator", legs odd) (HartreeFock.cc:921-928).
- 2-body: for each stored channel pair `(ch_bra, ch_ket)` in `OpHO.TwoBody.MatEl` (covers tensor operators with different bra/ket channels): `OUT = Dbra * IN * Dket`, with, per channel,
  `D(i,j) = C(p_i, p_j') C(q_i, q_j') + (p_i != q_i) * Phase(J) * C(q_i,p_j') C(p_i,q_j')`, then `*= sqrt2` if `p_i==q_i` and `/= sqrt2` if `p_j'==q_j'` (normalised-TBME convention; HartreeFock.cc:957-969). `Ket::Phase(J)` is `phase_prefactor * (-1)^J` with the fermionic sign included (ModelSpace.hh:80). For bra==ket channel `Dbra = Dket^T`.
- Odd-leg operators take a "ThreeLeg" branch (`Dbra * IN * C`).
- 3-body: if `OpHF.GetParticleRank() >= 3`: `OpHF.ThreeBody = GetTransformed3B(OpHO, C)`.
- **Operator is NOT normal ordered here**; the driver calls `DoNormalOrdering()` afterwards.

`GetNormalOrderedH(int particle_rank=2)` (HartreeFock.cc:1079-1172) is the Hamiltonian-specific fused routine:
- `HNO.ZeroBody = EHF`; `HNO.OneBody = C^T F C` (**the Fock matrix, which already includes the NO 2-body and 3-body mean field**, so the 1-body part is `f` not `t`).
- For each two-body channel J, build `D` as above (square, same-channel). If the Hamiltonian has a 3-body part, build the **NO2B contribution of the 3N force** `V3NO(i,j) = (1/(2J+1)) sum_{ab} rho_ab * ThreeBody.GetME_pn_no2b(bra.p,bra.q,a, ket.p,ket.q,b, J)` with `a`,`b` restricted to the same one-body channel and the `e_a+e_bra2 <= E3max`, `e_b+e_ket2 <= E3max` cuts, skipping `|rho_ab|<1e-8`, then `/sqrt2` for identical-orbit bra/ket (HartreeFock.cc:1118-1141). Only the upper triangle is computed and mirrored (`V3NO(j,i)=V3NO(i,j)`).
- `OUT = D^T (V2 + V3NO) D` (HartreeFock.cc:1152). So **the 2-body of the HF-basis Hamiltonian = transformed bare 2N + transformed NO2B part of 3N, evaluated with the HO-basis rho**. Since rho is expressed in the HO basis and V3NO is built in the HO basis, no 3-body transformation is needed for this term.
- If `particle_rank>2`: `HNO.ThreeBody = GetTransformed3B(Hbare, C)` computes the *full* 3-body ME in the HF basis (the residual 3N is then the raw transformed 3N; `DoNormalOrdering` is NOT applied to the 3-body part, i.e. the 3-body part carried into the IMSRG(3) is the un-subtracted transformed 3N; the NO2B parts above are added to the lower-body parts. This is the standard "normal ordered 3N: E0, f, Gamma, W" decomposition where W is the (transformed) 3N interaction in normal-ordered form; the commutator code treats the stored 3-body as the normal-ordered residual).
- `discard_NO2B_from_3N` zeroes `V3NO` (diagnostic only; prints a warning).
- `GetNormalOrderedH(arma::mat& Cin, int rank)` (HartreeFock.cc:1058-1068) sets `C=Cin`, recomputes rho/F/EHF without iterating, and calls the above: used for Brueckner restarts. It does not re-run `ReorderCoefficients`.

`GetTransformed3B(OpIn, C_3b)` (HartreeFock.cc:1326-1625), the 3-body transformation:
- Output container is `ThreeBodyME(ms_for_output_3N,...)` in **pn storage** (`SwitchToPN_and_discard`); channels are the allowed `(ch_bra,ch_ket)` from `Get_ch_start()`, parallel over channels with `omp parallel for schedule(dynamic,1)` (the loop is the only parallelism).
- For each channel it builds `Dbra` (nbras_kept x nbras_HO) and `Dket` (nkets_HO x nkets_kept), where for each 3-body ket the HF->HO overlap sums over the **unique permutations** of the three HO labels: `phase * recoupling_coefficient(perm, j's, JijHF, JijHO, twoJ) * C(i',i)C(j',j)C(k',k)`, skipping |overlap|<1e-8 (HartreeFock.cc:1465-1490). These helpers (`UniquePermutations`, `Permute`, `PermutationPhase`, `RecouplingCoefficient`) live in `ThreeBodyME`.
- `Vho` is the dense HO matrix (`GetME_pn` entry by entry, HartreeFock.cc:1552-1573), `Vhf = Dbra*Vho*Dket`, entries with |V|<1e-9 are not stored.
- Cut for kept HF states: `|e_i - e_F(tz)| + |e_j - e_F| + |e_k - e_F| <= dE3max` where `e_F = ms_for_output_3N->GetEFermi()`. **Bug**: the energies are computed as `ej = 2*oq->n + op->l` and `ek = 2*oR->n + op->l` (uses `op->l`, the l of the first orbit, instead of `oq->l`/`oR->l`) at HartreeFock.cc:1368-1369 and 1390-1391. Harmless with default `dE3max=99` but wrong when the cut is active. Fix, but record that results with active dE3max in published runs were affected.
- Bra uses `C_3b`, ket uses the member `C` (inconsistency; identical in all current call sites because `GetNormalOrderedHNAT` passes `C_HO2NAT` for bra while the ket branch still uses `C` (HartreeFock.cc:1465ff, 1428ff). When `ch_bra==ch_ket` `Dket = Dbra^T` is used so the inconsistency only matters for off-diagonal-channel (tensor) operators.)
- `GetValence3B` and `GetTransformed3bme` (HartreeFock.cc:1635-1722) are slow reference implementations (six nested loops over alpha,beta,... with 1e-8 cuts), "good for debugging".

### 1.7 Jacobi 3N in HF (experimental)

`SetJacobi3BME(jac)` -> `use_jacobi_3body=true`. `Jacobi3BME` (Jacobi3BME.cc/.hh) holds Navratil-style Jacobi-coefficient (cfp) data, with `ReadTcoeffNavratil` (Jacobi3BME.cc:915), `ComputeNAS_MatrixElements` (243), `GetLabMatEl` (310/378, two overlapping definitions of the same function), `GetV3mon_all(HartreeFock&)` (497-900, a huge monopole routine) and `Tcoeff_wrapper` -> `AngMom::Tcoeff`. The driver (`imsrg++.cc`) never uses it (no "Jacobi" string in the driver); only `pyIMSRG.cc` and tests do. Binary readers for it: `ReadWrite::ReadJacobi3NFiles` (ReadWrite.cc:5460-5628; Fortran unformatted records with 4-byte record markers; files `poi`, `eig` (double cfp), `v3int` (hw, A, Z, N, 2J, 2T, parity, Nmax header per block)). Recommendation: keep out of the first rewrite milestone; if ever wanted, re-derive it (the source comment says its HF hook is untested/wrong).

### 1.8 Dependencies of HartreeFock

Includes `ModelSpace`, `Operator`, `IMSRGProfiler` (static timer map), `Jacobi3BME`, `ThreeBodyME`, armadillo, OpenMP (`omp_get_wtime`), GSL (laguerre). It mutates `ModelSpace` (via `SetReference`), reads internal `ModelSpace` tables (`MonopoleKets`, `all_orbits`, `holes`, `GetEFermi`, `GetE3max`), `Hbare.OneBodyChannels`, `Hbare.TwoBody.GetTBMEmonopole`, `Hbare.ThreeBody.GetME_pn_mono/no2b/pn`. `HFMBPT` and imsrg_util's `WriteSPWaveFunctions` depend on it. Avoid: HF mutating the model space; `Operator& Hbare` held by reference (dangling if the caller replaces it, as the driver does with `Operator& HNO = Hbare` which overwrites Hbare, i.e. **HF object holds a reference to an operator that is then overwritten by the normal-ordered one**; this works only because `KE`/C are copies).

---------------------------------------------------------------------------------------------------

## 2. HFMBPT: natural orbitals and perturbation theory (HFMBPT.hh/.cc)

`class HFMBPT : public HartreeFock` (HFMBPT.hh:25). Extra members: `C_HO2NAT`, `C_HF2NAT`, `Occ` (eigenvalues of the correlated rho), `use_NAT_occupations` (default **uninitialised in header; set by driver**), `NAT_order` (string; "occupation" default from driver), `vv_*` (Van Vleck). HFMBPT.hh:24-39. The hh declares `OCC_CUT` comment but uses `ModelSpace::OCC_CUT = 1e-6` (ModelSpace.cc:14).

### 2.1 Natural orbitals (GetNaturalOrbitals, HFMBPT.cc:36-148)

1. `GetDensityMatrix()` (HFMBPT.cc:421-443): calls `HartreeFock::GetNormalOrderedH()` (rank 2; this also re-normal-orders and possibly updates the reference), sets `rho` = diagonal with `occ` of each hole orbit **in the HF basis** (zeroing rho first, bug fix of Dec 2020), then adds 2nd-order corrections `DensityMatrixPP`, `DensityMatrixHH`, `DensityMatrixPH`. The Hamiltonian used is the NO2B-truncated one in the HF basis, using `H.OneBody(a,a)` diagonal energies (Moller-Plesset partition, so Fock off-diagonals are ignored in denominators).
2. `DiagonalizeRho()` (HFMBPT.cc:151-188): per one-body channel, `eig_sym(rho_ch)`; eigenvalues are reversed (descending occupation) into `Occ(orbvec)`, eigenvectors reversed (columns) into `C_HF2NAT.submat`. Failure -> `exit(0)` (hard exit with success code!).
3. Check `Tr(rho) = sum (2j+1) rho_ii == A` to 1e-8, else prints a warning and `exit(0)`.
4. `C_HO2NAT = C * C_HF2NAT`; `oi.occ_nat = |Occ(i)|` stored on the `Orbit` objects.
5. If `use_NAT_occupations` (HFMBPT.cc:67-130): switch the reference to the fractional occupations of rho. Orbits with `Occ > 0.02` (hard-coded `keep_occ_threshold`) become "holes" with their eigenvalue occupation (everything else set to 0, to keep the hole space small); then **back-fill** to restore integer particle numbers: repeatedly (while `Z - Zfromtr > OCC_CUT`) add `min(0.005, deficit/(2j+1), 1-occ)` to each proton hole orbit in list order (and likewise neutrons) (aloquot=0.005). Then `UpdateReference()` (rebuilds ModelSpace) and re-sets `occ_nat` (since the ModelSpace rebuild wipes it).
6. `rho = C_HO2NAT.cols(holeorbs) * hole_occ * ^T` (a rho in the HO basis with the prescribed occupations in the NAT basis), `UpdateF()` (F in HO basis with this rho), `ReorderHFMBPTCoefficients()`, then `C_HO2NAT = C * C_HF2NAT` again ("bug fix suggested by Emily Love", HFMBPT.cc:146). No self-consistency iteration (comment at HFMBPT.cc:136-138).

### 2.2 Which diagrams enter the correlated density matrix

All three terms use the second-order wave function (|1><1|) with a two-level mixing regulator to protect against small denominators. Notation: i,j,k holes, a,b,c particles; `e` = diagonal of `H.OneBody`; occupation factors `(1-occ)`, `occ` insert fractional occupations.
- **PP** (HFMBPT.cc:487-557), `rho_ab = 1/2 (1/(2ja+1)) sum_{c,i,j,J} (2J+1) <ac|V|ij>_J<ij|V|bc>_J / ((e_a+e_c-e_i-e_j)(e_b+e_c-e_i-e_j))`. Regulated by replacing `X/(Da Db)` with `(1/2)[sqrt(X + E^2) - E]/sqrt(X + E^2)` where `E = (1/2) sqrt|Da Db|`, i.e. `r += 0.5*(sqrt(tbme+eps^2)-eps)/sqrt(tbme+eps^2)` where `tbme` here is the (occupation weighted) numerator *V V without the denominator*. (Look carefully: the code computes `tbme*...`, `epsilon = 0.5*sqrt|e_acij e_bcij|`, and uses `0.5*(sqrt(tbme+eps^2) - eps)/sqrt(tbme+eps^2)`. The comment above the function states the same replacement. It is **not** equal to `tbme/(Da Db)` in the perturbative limit unless one expands: `(sqrt(t+e^2)-e)/(2 sqrt(t+e^2)) ~ t/(4 e^2) = t/(Da Db)` with `e^2 = Da Db/4`. OK, consistent.) Only `b<=a`, filled symmetric. Pauli blocking factors `(1-n_a)(1-n_b)(1-n_c)^2 n_i^2 n_j^2` are applied for fractional occupation reference.
- **HH** (HFMBPT.cc:562-655), `rho_ij = -1/2 (1/(2ji+1)) sum_{a,b,k,J} (2J+1) <ab|V|ik><jk|V|ab>/((e_a+e_b-e_i-e_k)(e_a+e_b-e_j-e_k))` with the same regulator.
- **PH** (HFMBPT.cc:657-777), the "mixed" block `rho_ai` (a particle, i hole, same channel): two terms, `+ sum <aj|V|bc><bc|V|ij>/((e_a-e_i)(e_b+e_c-e_i-e_j))` and `- sum <kj|V|ib><ab|V|kj>/((e_a-e_i)(e_a+e_b-e_k-e_j))`, each times `1/2/(2ja+1)`, filled symmetrically into `rho(a,i)` and `rho(i,a)`. Quirk: the skip condition is `if (e_ai*e_bcij < 1e-8) continue;` (no `abs`), so negative products are skipped; it works since both factors are normally positive, but not generally.
- No 3-body force contributions to rho, no third order. The 3N enters only through the NO2B two-body (from HF's `GetNormalOrderedH`).
- Complexity: unoptimised 5-6 nested loops over orbit labels with `GetTBME_J` inside the J loop (HFMBPT.cc:487-777); no OpenMP. Cost grows like N_orb^5 J-summed; this dominates for emax>=10 and is a candidate for channel-matrix reformulation (`rho_pp ~ sum_J V_{ac,ij}^J (Delta^-1) V_{ij,bc}^J` as matrix products per J channel).

### 2.3 NAT_order options (ReorderHFMBPTCoefficients, HFMBPT.cc:812-877)

First, flips signs of columns of `C_HF2NAT` with negative diagonal (`C_HF2NAT(i,i)<0`). Then:
- `"occupation"` (default): nothing more (descending occupation from `DiagonalizeRho`).
- `"energy"`: per channel, sort NAT orbitals by the diagonal of `C_NAT^T C_HF^T F_HO C_HF C_NAT` ascending (one pass only; comment says occupied/unoccupied swap not handled), reorder `Occ` along. The legacy driver flag `order_NAT_by_energy` overrides `NAT_order` to "energy" (imsrg++.cc:573-574).
- `"mp2"`: builds `H_temp = GetNormalOrderedHNAT(2)`, evaluates `GetMP2_Impacts(H_temp)` (per-orbit MP2 energy contributions, HFMBPT.cc:923-1000; each term credited to all four orbit labels) and sorts ascending by impact (most negative first).
`use_NAT_occupations`: described in 2.1 step 5 (driver param, default false).

`OccNat3Cut` (driver) triggers NAT construction even when `basis != NAT` so that `occ_nat` can be used to cut 3-body states in IMSRG(3) (see `ModelSpace::CountThreeBodyStatesInsideCut`).

### 2.4 Natural-orbital normal ordering and transformations

- `GetNormalOrderedHNAT(rank)` (HFMBPT.cc:327-413): copy of `GetNormalOrderedH` but with `C_HO2NAT` in place of `C`, and `rho`/`F` as already left by `GetNaturalOrbitals` (rho built from the NAT-basis fractional occupations). Calls `CalcEHF()` and prints `e1Nat,e2Nat,e3Nat,E_Nat`. Same NO2B-of-3N code, same `GetTransformed3B(Hbare, C_HO2NAT)`.
- `TransformHOToNATBasis` / `TransformHFToNATBasis` (HFMBPT.cc:191-320): the 1- and 2-body analogues of `TransformToHFBasis` with the same D-matrix formulas. They have **no 3-body and no odd-leg branches**. `TransformHFToNATBasis` is unused in the driver.
- `PrintSPEandWF` override prints NAT wave functions in terms of HO components using `C_HO2NAT`.
- `GetTransformed3bme` (HFMBPT.cc:879-921): a slow duplicate of HartreeFock's.

### 2.5 Everything else in HFMBPT.cc (feature catalog)

| Feature | Lines | Notes |
|---|---|---|
| `GetMP2_Energy(H)` | 1005-1062 | OpenMP reduction over particles. `E2 = sum_{ia} (2ja+1) occ_a f_ia f_ai/(e_a-e_i) + sum_{i<=j, a<=b} sum_J (2J+1) occ_a occ_b |<ab|V|ij>_J|^2 / (e_a+e_b-e_i-e_j)` using *normalised* `GetTBME_J_norm`; "restricted sum, no factor 1/4"; for identical orbits `J` stepped by 2 starting at the even J (dJ=2, Jmin += Jmin%2). (Naming is confusing: variables named `i,j` are particles and `a,b` holes, sign of denominator `ea+eb-ei-ej` is hole minus particle, negative.) Includes the singles (`f_ia`) term. |
| `GetMP3_pp/hh/ph`, `GetMP3_Energy` | 1066-1285 | Third-order energy split into pp, hh, ph ladder/ring pieces; sums returned as one double (the Operator version returns `std::array<double,3>`). |
| `GetMP4_term(H,id)` + `GetMP4_F1..F39` | 1541-4440 | All 39 fourth-order diagrams (labelled F1..F39 after Shavitt-Bartlett / "ADG" numbering; F1 corresponds to diagram 4 in Shavitt-Bartlett) in J-coupled form for closed-shell reference; id dispatch through a `std::array<std::function>` (HFMBPT.cc:1541). Many are factorised (e.g. F1 builds two one-body-like intermediates chi_A, chi_B). `GetDenom(H,holes,particles)` helper. Some functions are stubs that return others' results (F3, F8, F14, F15, F25, F26, F29, F30 are 1-12 lines; the larger F9 appears twice in the file at 2226 and 2336, presumably one in an `#if 0`/comment block). **Reference only**: single-reference closed-shell energies; not needed for the IMSRG workflow. |
| Van Vleck PT: `SolveVanVleck`, `VV_TransformH`, `VV_Transform`, `VV_Transformed0b`, `GetVVOmegas`, `SetVVApprox/Generator_type/partitioning` | 1291-1525 | Order-by-order generator `Omega_n` from the Van Vleck (canonical Hermitian) PT recursion [Kutzelnigg, JCP 73, 5711 (1980)]; uses `Generator vv_gen` (`GetHod`, `UpdateGeneral`) and commutators `Commutator::Commutator`, optional `imsrg3n7`, `imsrg3f2` (factorised 223 commutators). Implemented through 3rd order for `Omega` (4th-order block is commented out; prints "Orders beyond 4 not yet implemented"). Tied closely to the commutator code. |
| `GetMP2_Impacts` | 923-1000 | See 2.3. |

`ReferenceImplementations/HFMBPT4.cc` (3048 lines, `namespace ReferenceImplementations`): the earlier **straightforward J-coupled implementation of the same 39 MBPT(4) diagrams** (`GetMP4_term`, `GetMP4_F1..F39`, `GetDenom`), declared in `ReferenceImplementations.hh:114ff`. Header says "straightforward ... without optimizations. This should be benchmarked against the m-scheme implementation and then left untouched". It is in `ReferenceImplementations/CMakeLists.txt` and is the validation baseline for the factorised versions inside `HFMBPT.cc`. Not compiled via the top-level CMake list at line 456 (commented out there).

Design note: for the rewrite, MBPT(2)/(3) energies and the second-order density matrix are the useful pieces; MBPT(4) and VV are optional "later" features and good candidates for tests (reference and factorised versions agree).

---------------------------------------------------------------------------------------------------

## 3. RPA (RPA.hh/.cc)

Scope: **ph-RPA/TDA built from a normal-ordered Hamiltonian in one `J,parity,Tz` cross-coupled (CC) channel**, plus operator-correction helpers. It is a small class (`RPA.hh:25-70`) with public matrices `A, B, X, Y, Energies` and a single-channel state (`channel`).

- `ConstructAMatrix_byIndex(ich_CC, Isovector)` (RPA.cc:41-133): in the CC channel (`TwoBodyChannel_CC`, ph kets `GetKetIndex_ph()`), `A_{ai,bj} = delta (e_a - e_i) + sum_{J1} -(2J1+1) {ja ji Jph; jb jj J1} <a j|V|b i>_{J1}` (Pandya transform of the J-coupled TBME, non-normalised `GetTBME_J`). Quirks: orbit order within the ph ket is swapped so the more occupied orbit is the "hole" (`ket.op->occ > ket.oq->occ`), tracking phases `phase_ai/phase_ia`; `Isovector` multiplies by `tz2`. It prints debug `a i spe` lines on every matrix build (RPA.cc:~58).
- `ConstructBMatrix_byIndex` (RPA.cc:135-211): `B_{ai,bj}` ph-ph backward amplitudes from `<ab|V|ij>` via a 6j sum (Suhonen 11.70). `phase_ib` etc. as in A.
- `SolveCP()` (RPA.cc:215-224): "core polarization": diagonal of A only (first-order approximation; eigenvalues are `e_a-e_i`, X=1, Y=0). `SolveTDA()`: `eig_sym(A)`, `Y=0`. `SolveRPA()`: builds the 2n x 2n matrix `[[A,B],[-B,-A]]`, `eig_gen`, keeps eigenvectors with `|X|>|Y|` (positive-norm), sorts by real energy, normalises by `sqrt(|X|^2-|Y|^2)`; warns if the imaginary part of eigenvalues exceeds 1e-3 (instability), keeps only real parts. (The sign choice of eigenvectors is arbitrary; the commented-out code tried to fix it.)
- `GetEgs()` (RPA.cc:331-352, "not benchmarked"): RPA correlation energy `-sum_mu E_mu |Y_mu|^2` summed over all CC channels (Ring-Schuck 8.93).
- `TransitionToGroundState(OpIn, mu)` (RPA.cc:355-387): `<mu||O||0> = sum_{mi} O_mi ((-1)^lambda X_mi + Y_mi)` (Suhonen 11.223) for the pre-built channel.
- `PVCouplingEffectiveCharge(OpIn,k,l)` (RPA.cc:391-556) and `StraightforwardCPEffectiveCharge` (573-611): effective-charge (core-polarization) 1-body corrections for a rank-lambda operator, from the Pandya-transformed interaction and X,Y; the latter is the explicit first-order CP formula with `Vbar_{ijba}` and `Vbar_{ijab}` terms divided by `Delta = e_i + e_a - e_b - e_j`. Used for tests and the PV (parity violation) work.
- Not used by the driver directly.

**"OperatorsRPA / OperatorsTDA"** (Parameters.hh:153-154) are driver parameters that call `imsrg_util::RPA_resummed_1b(OpIn, H, "RPA"|"TDA"|"CP")` (imsrg++.cc:769-778; imsrg_util.cc:3568ff), **not** the `RPA` class. `RPA_resummed_1b` (the second, live definition after a commented-out "wrong" version, imsrg_util.cc:3546-3700): builds the list of ph (and hp) kets with the same J/parity/Tz quantum numbers as the operator; builds `Mphph` (and, for RPA, `Mphhp`) from the Pandya-transformed interaction `GetPH_transformed_Gamma`, divides by the operator-dependent denominators `Delta_ph +- (e_v1 - e_v2)`, resums the geometric series by `O_RPA = (1 - M)^-1 O_ph` (dense `arma::inv`) **per valence pair (v1,v2)**, packs `O_RPA` back into an operator and evaluates it in first-order perturbation theory (`FirstOrderCorr_1b`). Mode `"CP"` zeroes `Mphph` (first-order core polarization only). The comment above the function states the formula is still not worked out cleanly ("TODO this is still not right in the comment"). The effective operator correction is returned as a one-body operator (and appended to `ops` with names `<op>TDA`/`<op>RPA`). Cost: one dense inverse per valence pair per call (small ph-space for typical emax).

Dependencies: `ModelSpace` (CC channels, 6j via `GetSixJ`), `Operator`, `AngMom`. No circularity issues. Port as a standalone module on top of Pandya-transformed blocks; it has no hidden global state. Python bindings expose RPA class (pyIMSRG.cc:767-791).

---------------------------------------------------------------------------------------------------

## 4. ReadWrite: catalog of file formats

`ReadWrite` is a stateful grab-bag (`ReadWrite.hh:36-139`): it remembers `File2N`, `File3N` (for headers of outputs), `Aref/Zref`, `LECs` (5 doubles), `scratch_dir`, `format3N` ("me3j" or "navratil" for binary 3N), `doCoM_corr`, `goodstate`. Errors are reported by `std::cerr`/`std::cout` and `goodstate=false`, or **`std::exit`** (several readers kill the process). Compression: Boost.Iostreams `filtering_istream` + `gzip_decompressor` (the header-only mode of Boost.Iostreams is used: `#define BOOST_IOSTREAMS_NO_LIB 1` plus zlib) and `gzip_compressor(best_speed)` for writing (ReadWrite.hh:22; ReadWrite.cc:17-21; Write_me2j_gz at ReadWrite.cc:3243). HDF5 is compiled out by default (`NO_HDF5="true"` in src/CMakeLists.txt:229). `half.hpp` (half-float) is used by the NO2B/mono storage. Build deps: BLAS (OpenBLAS), GSL, ZLIB, Boost, OpenMP, Armadillo (vendored in `extern/`).

### 4.1 Normalisation conventions (read this first)

- In-memory 2-body storage (`TwoBodyME::MatEl[(ch_bra,ch_ket)]`, an `arma::mat` indexed by the *local ket index* in the channel; kets are `(p<=q)` in modelspace orbit order) is **normalised**: `|ab;J> / sqrt(1+d_ab)`. `GetTBME_norm` returns the stored value; `GetTBME` multiplies by `sqrt2` for each identical-orbit pair (TwoBodyME.cc:170-176). `SetTBME(...)` stores what it is given (applying a `Ket::Phase` if `a>b` or `c>d`), so callers must pass normalised values (TwoBodyME.cc:206-231).
- The modelspace orbit order is **not** the file order: modelspace loops `N=0..Emax`, `l = N, N-2, ..., ` (**descending l**), `j2 = 2l+1 then 2l-1` (**descending j**), `tz = -1, +1` (ModelSpace.cc:381-398), giving proton index even and neutron = proton + 1 (assumed by `Set_pn_TBME_from_iso`, which does `a -= a%2`). This breaks for `single_species` model spaces. File-order lists are rebuilt with `l` ascending and `j` ascending inside each shell (see 4.2), and mapped via `GetOrbitIndex(n,l,twoj,-1)` (`orbits_remap`). The reader never relies on modelspace ordering for the loops.
- `Ket::Phase(J) = phase_prefactor * (-1)^J` includes the fermionic sign: `|pqJ> = Phase |qpJ>`.

### 4.2 The Darmstadt .me2j / .me2j.gz reader (full spec)

Entry: `ReadWrite::ReadBareTBME_Darmstadt(filename, Hbare, emax_file, E2max_file, lmax_file)` (ReadWrite.cc:664-727), called from the driver with `(file2e1max, file2e2max, file2lmax)` (imsrg++.cc:444).

Defaults (Parameters.hh:135-137): `file2e1max=12`, `file2e2max=-1`, `file2lmax=-1`. In the template reader (ReadWrite.cc:834ff):
- `emax = file2e1max` (if negative, use the modelspace `Emax`).
- `lmax = (lmax_in > 0) ? lmax_in : emax` (so lmax in the file = file emax by default).
- `E2max = (E2max_in > 0) ? E2max_in : 2*emax`.
**The reader does not read file parameters from the header; the three numbers must exactly match how the file was written, otherwise the stream is silently mis-parsed.**

Dispatch by extension (ReadWrite.cc:676-720): missing file -> `exit`. `.gz` -> `boost::iostreams::filtering_istream` with `gzip_decompressor` (text); `.bin` -> single-precision binary: seek to size, skip `HEADERSIZE=255` bytes, read all `float`s into a `std::vector<float>`, wrap in `VectorStream` which mimics `operator>>`; anything else -> plain `std::ifstream` text. The whole `.bin` file is loaded in memory; the `.gz` and text readers stream.

Header: one line is discarded with `getline(line, LINESIZE=496)` (no check that it is a header). For `.bin`, `getline` is a no-op (the 255-byte header was skipped separately).

Orbit list in file order (ReadWrite.cc:857-871): for `e = 0..min(emax_file, ms.Emax)`, for `l = e%2, e%2+2, ..., min(e,lmax)` (ascending), `n=(e-l)/2`, for `twoj = |2l-1|, ..., 2l+1` (ascending, step 2): push `GetOrbitIndex(n,l,twoj,-1)` (proton orbit; `NOT_AN_ORBIT = 99999` if absent from the model space, e.g. beyond `EmaxUnocc`), plus parallel arrays of `e`, `l`, `2j`. `nljmax = size-1`. Because the list is cut at `min(file emax, model emax)` the reader only consumes a **prefix** of the file when the file has a larger emax (valid because the loops below only reference index <= nlj1; see "larger emax" below).

Loop order (ReadWrite.cc:880-947), literally:

```
for nlj1 = 0..nljmax:           a = remap[nlj1]; if e1 > ms.Emax break
  for nlj2 = 0..nlj1:           b; if e1+e2 > E2max break
    for nlj3 = 0..nlj1:         c
      for nlj4 = 0..(nlj3==nlj1 ? nlj2 : nlj3):   d; if e3+e4 > E2max break
        skip if (l1+l2+l3+l4) odd
        Jmin = max(|j1-j2|,|j3-j4|)/2 ; Jmax = min(j1+j2, j3+j4)/2 ; skip if Jmin>Jmax
        for J = Jmin..Jmax:
           read 4 floats:  tbme_00  tbme_nn  tbme_10  tbme_pp
```

So pairs are ordered `(a>=b)`, `(c>=d)` (in file nlj index), bra index >= ket index (`nlj3<=nlj1`, and if equal then `nlj4<=nlj2`), i.e. the lower triangle of a bra-ket matrix. J runs over the **full** allowed range `[Jmin, Jmax]` with no extra parity restriction on J, even for identical orbits (so for `a==b` entries for "forbidden" J are present in the file as zeros; the 4 columns are always written). For each J, exactly 4 numbers per line are consumed (even if later skipped); `nreads` counts them (printed `nreads*4`, which is a bug: it should be `nreads*4` only if nreads counted J lines, which it does; fine).

Columns (ReadWrite.cc:921-934), in the **file order** `T=0,Tz=0 (pn)`, `T=1,Tz=+1 (nn)`, `T=1,Tz=0 (pn)`, `T=1,Tz=-1 (pp)`:
- col 1 `tbme_00`: isoscalar (T=0) pn matrix element `<ab; J T=0 | V | cd; J T=0>`.
- col 2 `tbme_nn`: T=1, Tz=+1 (neutron-neutron), imsrg++ calls proton tz2=-1 and neutron tz2=+1, so nn -> `a+1,b+1,c+1,d+1`.
- col 3 `tbme_10`: T=1, Tz=0 isospin-symmetric pn matrix element.
- col 4 `tbme_pp`: T=1, Tz=-1 (pp, includes Coulomb if the file has it).
There is no Tcm / p.p / kinetic column in this format (that is the "navratil" format); the file contains only V (plus Coulomb in pp if the generator included it).
The element values are **un-normalised isospin-formalism matrix elements**: `<ab|V|cd>_{JT}` for non-normalised (not divided by sqrt(1+d)) two-body states. Values are parsed as `float` (single precision) from text (or raw floats for `.bin`).

Conversion into pn form (ReadWrite.cc:905-941):
```
norm = 1; if (a==b) norm /= sqrt2; if (c==d) norm /= sqrt2;     // un-normalised -> normalised
if (norm>0.9 || J%2==0):                       // T=1 channels allowed
    SetTBME(J,parity,-1, a  ,b  ,c  ,d  , tbme_pp*norm)        // pp: Tz=-1, orbit labels proton (even index)
    SetTBME(J,parity,+1, a+1,b+1,c+1,d+1, tbme_nn*norm)        // nn
    Set_pn_TBME_from_iso(J, T=1, Tz=0, a,b,c,d, tbme_10*norm)
if (norm>0.9 || J%2!=0):                       // T=0
    Set_pn_TBME_from_iso(J, T=0, Tz=0, a,b,c,d, tbme_00*norm)
```
The `norm>0.9 || parity of J` test encodes Pauli exclusion: if either pair is identical-orbit, T=1 needs J even and T=0 needs J odd. (If exactly one pair is identical, `norm=1/sqrt2<0.9` and the J test applies to both; if neither, `norm=1` and both T always allowed.) Corner case: **if only `a==b` but not `c==d` the file has T=1 only for even J and T=0 for odd J; if both `a==b` and `c==d` the same statement applies**; in all other cases both T are set.

`Set_pn_TBME_from_iso(J,T,Tz,a,b,c,d,tbme)` (TwoBodyME.cc:615-635): converts everything to proton labels (`x -= x%2`), then with `tbme *= 0.5` and, **re-introducing** `sqrt2` for each identical orbit pair (`if (a==b) tbme*=sqrt2; if (c==d) tbme*=sqrt2`), does up to four `AddToTBME(J,parity,Tz=0,...)` calls, each `+=`:
1. `(a p, b n | c p, d n)` += tbme
2. if `c!=d`: `(a p, b n | c n, d p)` += tbme * (2T-1)
3. if `a!=b and c!=d`: `(a n, b p | c n, d p)` += tbme
4. if `a!=b and (a!=c or b!=d)`: `(a n, b p | c p, d n)` += tbme * (2T-1)
with isospin phase `+1` for T=1 and `-1` for T=0. Calling it for T=1 and then T=0 therefore yields e.g. `V_pnpn = (V_T1 + V_T0)/2` and `V_pnnp = (V_T1 - V_T0)/2`. It is the inverse of `Get_iso_TBME_from_pn` (TwoBodyME.cc:637-651; used by the writer). `AddToTBME` applies the `Ket::Phase` for reversed ordering and, for diagonal-channel Hermitian operators, mirrors the (bra,ket) entry to (ket,bra) except on the diagonal (TwoBodyME.cc:234-260), which is why the file only needs the lower triangle and why call 4 is skipped when bra==ket (it would double count the mirrored term). NOTE `norm` is applied to the isospin ME *before* `Set_pn_TBME_from_iso` and then `sqrt2` is multiplied back for identical orbits there; the net effect for the pn element with `a==b` is a factor 1/sqrt2*sqrt2 = 1 relative to the un-normalised file value, times 0.5.

Skips: `if (a>=norb || b>=norb || c>=norb || d>=norb) continue;` (ReadWrite.cc:921): orbit indices of `NOT_AN_ORBIT`, or above the model space, are skipped **after** the four floats were consumed, so the stream stays in sync.

Larger emax in the file than in the calculation: handled by (i) restricting `orbits_remap` to `e <= min(emax_file, ms.Emax)`, so the loops cover only the first `n_lj(emax_ms)` orbits as `nlj1` and (because `nlj2,nlj3,nlj4 <= nlj1`) as all indices; the data for these quadruples is a *prefix* of the file stream, so the rest (higher shells) is never read (a time saver; with gz this also avoids decompressing the remainder). The `E2max` breaks refer to the *file's* E2max, i.e. `file2e2max`; for orbit set below `emax_ms` and default `2*emax_file` they never trigger. (ii) the `a>=norb` skip for any orbit not in the model space (e.g. `EmaxUnocc`, `Lmax`). The model space `E2max` is **not** consulted by the reader; if `ms.E2max < 2*ms.Emax`, TBMEs outside the ket set would call `GetLocalIndex` with kets not in the channel (undefined behaviour); default `E2max = 2*emax` makes this a non-issue.
If the file has a *smaller* emax than the model space the high-shell elements simply stay zero and no warning is printed.

Applied to the planned test: `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` with `2bme=<file> file2e1max=14 hw=16 emax<=6`:
- Pass `file2e1max=14`; `file2e2max` defaults to `2*14=28`, `file2lmax` to `14`. **Check the file's header/E2max**: if it was generated with `E2max != 28` (NuHamil `E2max` parameter, e.g. 28 vs 24), you must pass `file2e2max` or the parse silently goes out of sync (only matters if the prefix we read hits a break; with `emax<=6` and `E2max_file >= 12` no `E2max` break fires, so a prefix-read still works even if `file2e2max` is wrong, as long as `file2e2max >= 2*emax_ms`. But `file2lmax` matters only through the orbit list, also irrelevant for `emax<=6` unless the file was lmax-cut below 6).
- `hw=16` does not affect reading (the file is just numbers); it affects `Trel` and the model space frequency (the `Trel_Op` 1-body matrix elements use `hbar_omega`). Fermionic phases and tz convention as above.
- The first line of the .gz file is dropped unconditionally.
- Output check values worth comparing: number of floats consumed for the prefix, i.e. sum over `(nlj1,nlj2,nlj3,nlj4)` of `(Jmax-Jmin+1)*4`.

### 4.3 Other 2-body input formats

| Format (driver `fmt2`) | Function (ReadWrite.cc) | Details |
|---|---|---|
| **me2j** Darmstadt (`.me2j`, `.me2j.gz`, `.bin`) | ReadBareTBME_Darmstadt (664) | section 4.2. Binary variant = raw float32 after a 255-byte header (endianness of the producing machine, no marker). |
| **navratil** (Petr Navratil NCSM, text or .gz) | ReadBareTBME_Navratil (441), `_from_stream` (472) | One header line; then lines `nlj1 nlj2 nlj3 nlj4 J T trel h_ho_rel vcoul vpn vpp vnn` (1-based Fortran orbit labels `nlj = N(N+1)/2 + max(l-1,0) + (j2-|2l-1|)/2`, N = 2n+l; only proton-labelled orbits are mapped). Values **normalised** JT-coupled. If `doCoM_corr` the CoM term `trel*2*hw/A` is added to vpn/vpp/vnn (a user-settable flag, off by default). For T==1 sets pp (Tz=-1) and nn (Tz=+1) directly with `SetTBME` (no norm factor), and `Set_pn_TBME_from_iso(J,T,0,...vpn)` if `|vpn|>1e-6`. `hw` is taken from the modelspace, not the file. |
| **oslo** | ReadTBME_Oslo (60) | Text; skip lines until one contains `<ab|V|cd>`; rows `Tz Par J2 a b c d tbme f0 f1 f2` with 1-based orbit labels; **normalised** values; optional CoM correction `tbme -= f2*hw/A` if `doCoM_corr`; elements with any label `> norbits` skipped (off-by-one: should be `>=`; `norbits` itself would be a valid label after decrement). Writers: WriteTwoBody_Oslo (230), WriteOneBody_Oslo (338), WriteOneBody_Simple (295). |
| **oakridge** (`oakridge`, `oakridge_bin`) | ReadTBME_OakRidge (111) | Two files joined by a comma in `2bme`: single-particle file (`hw` first, then `index n l j2 tz2 spe dummy`) and TBME file with records of 7 ints + 3 doubles (`Tz P J2 a b c d g1 g2 g3`); binary variant reads 7 int32 then 3 doubles (note `vdouble.size()*sizeof(int64_t)`, fine since 8 bytes). Uses `g1` only; `Tz` is the Tz channel (not 2Tz) directly; cut by model space `E2max` here. |
| **takayuki** | ReadOneBody_Takayuki (4608), ReadTwoBody_Takayuki (4639); writers 4678/4717 | Text rows `tza a tzb b me` etc.; 1-based nlj+1 labels; Miyagi-style. Not documented in the repo. |
| **nushellx** (`.int`) | ReadNuShellX_int (3807), ReadNuShellX_int_iso (3872), ReadNuShellX_sp (declared) | Reads a valence-space `.int`; if used, no kinetic energy is added (`fmt2 != "nushellx"` gate in the driver). |
| **jason**, **engel**, **Nathan**, **H2**, **Darmstadt 2-body rel.**, **Javier rel-CM** | ReadBareTBME_Jason (376), ReadTwoBodyEngel (5141, text/gz, `a b c d J tbme`), ReadOperator_Nathan / ReadTensorOperator_Nathan (2128/2178), ReadH2_2body (2405, binary), ReadDarmstadt_2bodyRel (2243, relative-coordinate TBMEs transformed to lab frame by Moshinsky brackets), ReadRelCMOpFromJavier (5178, two versions) | Special-purpose formats for operators (rel. CM kinetic, currents, test data). Low priority; Jason's reader skips 6 lines and reads `n l j` x4, `Tz2 J2 tbme` rows. |
| **schematic** | in driver | `Minnesota` potential generated analytically (`imsrg_util::MinnesotaPotential`). |
| **Miyagi (general operator 2b)** | ReadOperator2b_Miyagi (6241) | Always gz text: skip 1 line, `J P Z emax e2max` header line, zero-body, then 1-body block (loop `nlj1,nlj2 (full)`, 4 floats `pp nn np pn` per (allowed) pair) and 2-body blocks of 10 floats `pppp pppn ppnp ppnn pnpn pnnp pnnn npnp npnn nnnn` per `(nlj1>=nlj2, nlj3>=nlj4, Jij, Jkl)`; uses `SetTBME_J` (J-coupled, `Jij`,`Jkl` separate for tensor operators); threshold 1e-10. Used with `input_op_fmt=miyagi`. |
| **Navratil 2-body currents / MECs** | Read2bCurrent_Navratil (2852 first version, 2994 current), driver `input_op_fmt=navratil` | gz text. Skips 5 words, reads `J_op T_op`, skips 8 words, then rows `a b c d j12 j34 t12 t34 tza tzb me` ("double-reduced" ME with 1-based nlj labels, reordering phase `(-1)^{ja+jb-J-T}` when `a>b`). Stored in a hash map (`Petr2BC_hash`, 8 bytes each of a,b,c,d,jab,jcd,tab,tcd packed into a uint64), then the operator's TwoBody is filled by iso->pn CG sums and Wigner-Eckart factors. An older duplicate version precedes it (2852). |

Note: `ReadBareTBME_Darmstadt`, `Navratil`, `Engel` accept gz via `boost::iostreams`; no checksum, no version.

### 4.4 3-body input formats

Driver (imsrg++.cc:470-492): `3bme_type` selects `full` -> `Read_Darmstadt_3body`; `no2b` -> `ThreeBody.SetMode("no2b"|"no2bhalf")` + `ThreeBody.ReadFile`; `mono` -> `SetMode("mono")` + `ReadFile`. `fmt3` (`me3j`|`navratil`) sets `rw.format3N`. Defaults `file3e1max=12`, `file3e2max=-1` (=> `2*file3e1max` in the counter), `file3e3max=12`. `store_3bme_pn` converts isospin storage to pn (`Hbare.ThreeBody.TransformToPN()`).

**Full me3j (`Read_Darmstadt_3body`, ReadWrite.cc:730-828; `_from_stream` 1398-1490; `Count_..._to_read` 1210-1396; `Store_..._` 1492-1738).** Extension dispatch: `.me3j` text, `.gz` gzip text, `.bin` raw float32 after 255-byte header, `.h5` HDF5 (compiled out by default), anything else is treated as `me3j` text with a message. Two-pass design: (1) `Count_Darmstadt_3body_to_read` walks the *file's* index structure (restricted to `min(E1max, ms emax)` for the outermost orbit; also `E2max`, `E3max` file cuts) and records the cumulative float offset `nread_list[nlj1]` for each `nlj1`; (2) all required floats are read **sequentially into one `std::vector<float>`** (ASCII `operator>>` single-threaded, a performance bottleneck: ~4 bytes/number, e.g. emax 14/e3max 16 file is 10-100 GB of floats), (3) `Store_Darmstadt_3body` parses the loops again in parallel (`omp parallel for schedule(dynamic,1)` over `nlj1`) using the offsets to place each block, applying the model space cuts `e<=e1max`, `ea+eb+ec<=e3max`, `l<=lmax3`, `l<=lmax`, and the antisymmetry check (`autozero`: if the ME must vanish by Pauli (`a==b and (tab+Jab)%2==0`, etc.) and is non-zero beyond 1e-6 a loud "AAHH" is printed and `goodstate=false`), skipping `|V|<=1e-6` (stored via `SetME_iso`). Header: one line skipped (`getline(496)`); for `Hbare.GetTRank()>0` (beta decay operators) an extra header line with `opJ opP opT e1max e2max e3max lmax` is parsed and overrides the file limits (exits if J/T/parity mismatch).

Loop order of the 3-body file (ReadWrite.cc:1412-1490 count and 1500-1730 store):
```
for nlj1 (a):  (e_a<=min(E1max,ms.emax))
 for nlj2<=nlj1 (b):  e_a+e_b<=E2max
  for nlj3<=nlj2 (c): e_a+e_b+e_c<=E3max
   for nnlj1<=nlj1 (d):
    for nnlj2<=(nnlj1==nlj1 ? nlj2 : nnlj1) (e):
     for nnlj3<=((nnlj1==nlj1&&nnlj2==nlj2)? nlj3 : nnlj2) (f): e_d+e_e+e_f<=E3max; total parity even
      for Jab, JJab (all allowed), for twoJ in [max(|2Jab-jc|,|2JJab-jf|), min(2Jab+jc,2JJab+jf)] step 2,
       for (tab,ttab) in {0,1}^2, twoT in 1..2tab+1, twoTT in 1..2ttab+1 (|twoT-twoTT|<=2*TRank): one float
```
i.e. an isospin-coupled block of `iso_dim = 5` floats per `(Jab,JJab,twoJ)` (for TRank=0: (T12,T12',T): (0,0,1/2),(0,1,1/2),(1,0,1/2),(1,1,1/2),(1,1,3/2)). Ordering of T within a block: `tab` outer, `ttab`, then `twoT`, `twoTT` (in the `Store` routine: `Tcounter++` in that nested order). **me3j matrix elements are in the "antisymmetrised, isospin-coupled, non-reduced in J" convention (matrix elements of V, not reduced), with normalisation of the `|abc>` states as in the Roth/NuHamil code (no 1/sqrt(2) for identical orbits; zero for forbidden combinations).** The 3-body hamiltonian is stored in iso formalism (`SetME_iso`) and converted on demand to pn (CG sums with `AngMom::CG`).
- `format3N="navratil"` (binary Fortran sequential records): each float is wrapped in 4-byte record markers (`uint32 delimiter; float v; uint32 delimiter`), machine/endianness dependent, comment in source says so (ReadWrite.cc:1452-1461).
- `.bin`: raw floats; in the 3-body `.bin` code path the file size is `size_t` (the 2-body path uses `int n_elem`, which would overflow above 2 GB; irrelevant for me2j).
- Precision: floats (32-bit) read, stored as `float` (`ThreeBME` vector, `ThreeBME_type`).
- Memory: `Read` holds the entire prefix as `float`; storage as `float` in `ThreeBodyME` (isospin or pn) afterwards. Time dominated by text parse for `.me3j`/`.gz`.

**NO2B (`ThreeBodyStorage_no2b::ReadFile`, ThreeBodyME/ThreeBodyStorage_no2b.cc:552-809).** Input is the "NO2B" 3-body file (3N already summed over the total J with weight (2J+1)). **The format is `<a b c; J T12 T3| V |d e f; J T45 T3>` with `c` and `f` restricted to the same (l,j)** (loop condition `j6==j3 and l6==l3`), i.e. the third particle is the "spectator" of the NO2B contraction, and `J` is the 2-body `Jab=Jde`; there is no total-J index (inferred from the reader loop and from `GetME_pn_no2b`, not from format documentation). File detection: name contains `stream.bin` -> raw binary (no header, `sizeof(StoreType)` per value), `.gz` -> gzip text with header line skipped, else text with header line skipped. StoreType is `float` (`no2b_precision=single`, default) or `half_float::half` (`no2bhalf`, 16-bit, `half.hpp`), written in the file as the same type for binary; text is parsed into the half type through `operator>>`. Loop (all with `e<=Emax_file` cuts; `iF*` index into the file's orbit list generated like me3j, `Lmax_file=file emax` unless a 4th IntInput is given):
```
for iF1; for iF2<=iF1 (e1+e2<=E2max); for iF3 all (e2+e3, e1+e3 <=E2max; e1+e2+e3<=E3max);
 for iF4<=iF1; for iF5<=iF4 (e4+e5<=E2max); for iF6 with (j6,l6)==(j3,l3)  (+ e cuts, parity (l1+l2+l3)%2==(l4+l5+l6)%2)
  for J in [max(|j1-j2|,|j4-j5|)/2, min(j1+j2,j4+j5)/2]
   for T12 in {0,1}; for T45 in {0,1}; for T3 (twoT) = 1..min(2T12+1,2T45+1) step 2:  one value
```
The whole file (clamped to `n_elem_to_read = CountME(...)`) is processed in 8 GB chunks into a buffer by thread 0 (barrier-synchronised) while all OpenMP threads cycle through the orbit-quadruple (`orbit_counter % nthreads == tid`) and call `SetME_iso_no2b` (cut `|v|>1e-8`). `GetME_pn_no2b` converts to pn by 4 Clebsch-Gordan factors (ThreeBodyStorage_no2b.cc:362-398). Sanity check prints "something wrong, this three-body matrix element has to be zero" when the Pauli-forbidden entries are non-zero.
**mono**: same file layout but with J summed out (`ThreeBodyStorage_mono::ReadFile`, ThreeBodyStorage_mono.cc:722-1011): loops `iF1,iF2,iF3 all`, `iF4<=iF1` with `(j4,l4)==(j1,l1)`, `iF5` with `(j5,l5)==(j2,l2)`, `iF6` with `(j6,l6)==(j3,l3)` and then `T12,T45,T3` (no J loop). Stored via `SetME_iso_mono`. Used only for HF (monopole).

**HDF5 (3N)**: `Read3bodyHDF5` (1812) and `Read3bodyHDF5_new` (1999) read Heiko Hergert's `alphas` (int table: columns `?, n,l,2j` x3, `J12`, `2J`) and `vtnf` (float table, 5 columns = coefficients of `c1, c3, c4, cD, cE`; the combination `sum LEC_i * vtnf_i * HBARC` is formed on the fly; `LECs` set by `SetLECs_preset`: "EM1.8_2.0","EM2.0_2.0","EM2.2_2.0","EM2.8_2.0","PWA2.0_2.0","N2LOSAT"; default `{-0.81,-3.20,5.40,1.271,-0.131}`). Disabled (`NO_HDF5`) in the build. The default driver parameter `LECs="EM2.0_2.0"`. A rewrite can drop HDF5 initially.

### 4.5 Output formats

| Format | Writer (ReadWrite.cc) | Description |
|---|---|---|
| **.snt (Tokyo / KSHELL)** | WriteTokyo (5841-5990), WriteTokyoFull (5994, Hamiltonian only), WriteTensorTokyo (6087), reader ReadTokyo (5636), ReadTensorTokyo (5741) | Text. `!` comment lines (provenance: version, `File2N`, `File3N` basenames, `e1max,e2max,e3max,hw,Aref,Zref,A_for_kinetic_energy`, zero-body term), `! model space`, line `nprot_valence nneut_valence Zcore Ncore`, then one line per valence orbit `index n l 2j 2tz` (1-based; protons first then neutrons; order of the sets from `valence ∩ proton_orbits`), then `! interaction`: `n1b 0 hw` + rows `a b <a|h|b>` (only `a>=b`, thresholded `|v|<1e-7*||O||_1b`), `n2b 0 hw` + rows `a b c d J <ab;J|V|cd;J>` (valence-valence kets, only `iket>=ibra` within channel; thresholded `1e-7*||O||_2b`; format widths 4 ints, `%16.16f`). Values are the **normalised** J-coupled elements (`GetTBME_norm`) in pn form. For `mode=="op"` and pn pairs the value is averaged with the `(a-tz2, b-tz2, ...)`-flipped element ("isospin averaging for an operator file"), a hack for isoscalar operators. Matrices counted first (loop 1) then written (loop 2) with the same thresholds. `WriteTensorTokyo` writes reduced matrix elements for rank>0 (all channel pairs). |
| **NuShellX `.int` / `.sp`** | WriteNuShellX_int/op (3599/3595, impl `WriteNuShellX_intfile` 3610), WriteNuShellX_sps (3753); readers ReadNuShellX_int (3807), `_iso` (3872) | `.sp`: `! comment`, `pn`, `Acore Zcore`, number of valence orbits, `2 nprot nneut`, then rows `index n+1 l 2j` (**note `n+1`**, NuShellX counts from 1). `.int`: comment header, `! Index n l j tz` list, line `-999 e_1 ... e_N Acore Acore+2 0.0` (single-particle energies **as `op.OneBody(i,i)` diagonal only, default precision**; Acore computed from core orbits; the "mass dependence" parameters are fixed `Acore, Acore+2, 0.0`), then TBME rows `a b c d J T V` with `V` in `%12.6f`, `T=|Tz|` (T=1 for pp/nn, 0 for pn) and for pn **`T = (J+1)%2` after multiplying by `sqrt2` for each non-identical-orbit bra/ket** (the code's own comment: "pn TBMEs are unnormalized"); then pair labels are ordered (`a<=b`, `c<=d`, `(a,b)<=(c,d)`) with the appropriate `Ket::Phase(J)`. Cut `|tbme|<1e-6`. This is a *pn-formalism* NuShellX file with the T field used as a pn marker (the `T` for pn is set to `(J+1)%2`, which for non-identical-orbit pn kets is not the physical isospin); preserve if NuShellX compatibility is needed, otherwise prefer Tokyo. |
| **Antoine** | WriteAntoine_int (4011), WriteAntoine_input (4099) | Text interaction file and an input deck; driver doc says "antoine fmt is buggy" (Parameters.hh:80) though the code comment says "now appears to be working properly". |
| **operator (text, machine)** | WriteOperator (4270), ReadOperator (4371) | Header `Hermitian|Anti-Hermitian|Non-Hermitian`; `Jrank Trank parity`; `$ZeroBody:`, `$OneBody:` (rows `i j value`, `i<=j` unless non-Hermitian, chop 1e-9), `$TwoBody:` (rows `ch_bra ch_ket ibra iket value` of the **raw channel-matrix indices** (so the file is tied to the exact channel ordering of the model space)), optional `$ThreeBody:` (rows `ch_bra ch_ket ibra iket value`, upper triangle in same channel). Precision `setprecision(10)` / `9`. |
| **operator (human readable)** | WriteOperatorHuman (4178), ReadOperatorHuman (4465) | Same sections plus an orbit table (`i n l 2j 2tz`), and 2-body rows `J p Tz J' p' Tz' i j k l value` with 12-digit precision and `1e-7` chop; self-describing, model space independent. |
| **binary operator scratch** | `Operator::WriteBinary/ReadBinary` (Operator.cc:337-376), used for `OMEGA_<pid>_<nnn>` in `scratch` dir (IMSRGSolver.cc:56-65) | Raw dump of `rank_J, rank_T, parity, particle_rank, legs, hermitian, antihermitian, nChannels, ZeroBody, OneBody (norb^2 doubles, column-major), TwoBody (arma matrices), ThreeBody (storage-specific)` in native endianness and native `int`/`bool` sizes; not portable, not versioned; contents depend on the exact channel ordering. `scratch=""` disables; `/dev/null` special-cased. |
| **me2j / me3j writers** | Write_me2j (3123), Write_me2j_gz (3243), Write_me3j (3404), WriteTBME_Navratil (552) | `Write_me2j` writes the header line `    generated by imsrg++ code version ... on <ctime>` and the same loops as the reader (section 4.2) writing the four columns `tbme_00*norm, tbme_nn, tbme_10*norm, tbme_pp` where `tbme_pp,tbme_nn` come from `GetTBME` (un-normalised), `tbme_10,tbme_00` from `Get_iso_TBME_from_pn` (normalised) multiplied by `norm = sqrt2` per identical-orbit pair (so everything is un-normalised); fixed `%12.7f` (**7 decimals, 10 values per line**, text, single precision resolution). `Write_me2j_gz` is the same via `gzip_compressor(best_speed)`. Round-trip test vector: read my own writer output. |
| **Valence 3-body** | WriteValence3body (5010) | Text; valence-only 3-body, pn. |
| **Tensor 1b/2b & dagger operators** | WriteTensorOneBody (4765), WriteTensorTwoBody (4823), WriteDaggerOperator (4903) | Miyagi/He formats for transition operators (1b: `a_ind b_ind reduced-me`; 2b: `a b c d J ...` with orbit tz labels); `_HO_`/`_HF_` suffix variants written for `write_HF_ops`. |
| **spwf, occupations, C matrix** | imsrg_util `WriteSPWaveFunctions`; driver | `*_spwf_<orbit>.dat` (r, psi), `*_occ.dat` / `*_occ_nat.dat` (`n l 2j 2tz occ`, scientific), `*C.mat` (armadillo `.save` default binary `ARMA_MAT_BIN_FN008`). |
| **Jacobi 3N** | ReadJacobi3NFiles (5460) | See 1.7; Fortran unformatted. |
| **Flow/omega files** | IMSRGSolver (not ReadWrite) | Only the binary `OMEGA_*` scratch above and the `.dat` flow logs written by the driver; `ReadWrite::CopyFile` copies them (6368, naive `rdbuf` copy; `WriteOmega` is commented out). |

### 4.6 Parallel reading, memory, endianness, precision

- 2-body me2j: single-threaded stream parse; parse cost is small (emax 14: ~10^8 numbers for the full file, but a prefix of ~10^6 for `emax<=6`). 3-body: `Count` is serial; text read serial into a `std::vector<float>`; `Store` is OpenMP-parallel over `nlj1` with `schedule(dynamic,1)`. The NO2B/mono readers do the file read on thread 0 and distribute decoding across threads with barriers (8 GB buffer cap).
- Endianness: all binary formats (`.bin` me2j/me3j, Fortran-record 3N, Jacobi files, no2b `stream.bin`, `Operator::WriteBinary`, `arma::save`) are **native endianness, with no magic numbers** (little-endian x86 assumed). Fortran record delimiters assumed 4-byte.
- Precision: files are single-precision (float32) or half; internal storage `double` for 1/2-body, `float` (or half for NO2B/mono) for 3-body (`ThreeBME_type`); thresholds `1e-6` (3N store), `1e-8` (no2b), `1e-5`...
- Compression: gzip via Boost.Iostreams streams (single-threaded zlib).

---------------------------------------------------------------------------------------------------

## 5. imsrg_util/Pwd.* (partial-wave decomposition)

`class PWD` (imsrg_util/Pwd.hh:20-100): computes partial-wave matrix elements `W(p,p';S,L,L',J)` of a momentum-space operator given as `std::function(p,p',cosθ)` (central, spin-spin, spin-orbit, sigma_L, tensor, sigma_k forms) by Gauss-Legendre quadrature (`gsl_integration_glfixed_table` for momentum and angle meshes), with pre-computed angular integrals `A(p,p',J,l,type)` stored in `std::unordered_map<uint64_t,double> AList` (`AHash` packs the indices) and local/non-local regulators (`regulator_local/nonlocal`). Used by the 0vbb neutrinoless double-beta operator code (`M0nu.cc`) to build lab-frame operators from relative-space kernels. Dependencies: GSL, `std::function`. No HF/IO coupling. It is **out of scope for the first rewrite** except as a library module (needs only quadrature and `Operator` construction).

---------------------------------------------------------------------------------------------------

## 6. Global/static state, dependency graph, awkward couplings

- **Static/global**: `IMSRGProfiler::timer/counter/start_time` (static maps, written without locks from multithreaded routines such as `GetTransformed3B`, which is a data race on `std::map` inside `omp parallel` only at the end (timing is added after the parallel region in the version read, so OK; `Store_Darmstadt_3body` adds timers after the region too). `ModelSpace::OCC_CUT` (static, 1e-6), `ModelSpace::NOT_AN_ORBIT` (static 99999), `Commutator::Set*` flags (global toggles for IMSRG3 used in Van Vleck), `std::rand()` in HF (global RNG). `PhysConst` constants (HBARC, M_NUCLEON, SQRT2) as globals. Replace by explicit context objects.
- `HartreeFock` holds `Operator&` + raw `ModelSpace*` and mutates the model space (`SetReference`). `HFMBPT` extends HartreeFock (inheritance for NAT) and additionally contains MBPT energies and Van Vleck PT, which depend on `Commutator`, `Generator`, `Operator`: this makes HF -> Operator -> Commutator -> Generator -> HFMBPT a **circular-feeling** dependency web (HFMBPT.hh includes HartreeFock.hh and Generator.hh; Generator includes Operator; HartreeFock includes Jacobi3BME which forward-declares HartreeFock and defines `GetV3mon_all(HartreeFock&)`: a real circular include between Jacobi3BME.hh and HartreeFock.hh, broken only by forward declaration). Rewrite: split into `hf` (SCF), `basis/natural_orbitals`, `mbpt` (energies, density) and `vanvleck` packages; the SCF only needs `Hamiltonian view` + `ModelSpace view`.
- `ReadWrite` depends on `ModelSpace`, `Operator`, `ThreeBodyME`, `Jacobi3BME`, `AngMom`, Boost.Iostreams, HDF5 (optional). The class mixes file format code with state (`Aref`, `LECs`, `format3N`, scratch dir). Readers write into `Operator::TwoBody` through `SetTBME`/`Set_pn_TBME_from_iso`, whose conventions (normalised storage, proton-even labels, ket ordering p<=q, `Ket::Phase`) are the real "format adapter" interface; a rewrite should make the *reader emit a canonical J-coupled, normalised, per-(J,parity,Tz) block representation* and let a separate step place it into the operator.
- `ReadWrite::ReadJacobi3NFiles` and `Jacobi3BME` form a self-contained subsystem used only in tests.
- Mixed responsibilities in the driver: `imsrg++.cc` is a 1700-line procedural main that knows the HF -> NAT -> NO -> IMSRG sequencing; port it as a documented pipeline, not as a class.

---------------------------------------------------------------------------------------------------

## 7. Pitfalls for a rewrite (replicate vs. fix)

Must replicate for numerical equivalence:
1. **me2j reading**: the 4-column order (`T0, nn, T1pn, pp`), the loop order (a>=b, c<=a etc. in *file* orbit ordering l/j ascending), the full `[Jmin,Jmax]` J range, the identical-orbit `1/sqrt2` normalisation, the T/J-parity rule for identical orbits, the `Set_pn_TBME_from_iso` 0.5 factor and the (2T-1) exchange phase, skip of out-of-model-space orbits after reading the numbers, first-line skip, single-precision parse of text. Also the `Hermitian` mirror rule that only the lower triangle is in the file.
2. Quantum-number conventions: proton `tz2=-1`; `Ket::Phase`; normalised internal TBMEs; `GetTBME` = un-normalised.
3. HF: density-matrix definition (`C.cols(holes) .* occ`), `F = t + Vbar/(2j+1) + V3bar/(2(2j+1))`, `E_HF = e1 + e2/2 + e3/6` with `(2j+1)` weights, per-channel `eig_sym` with positive-diagonal sign convention, tolerance `||dE||_F/norbits < 1e-8`, density mixing 0.8 new/0.2 old, refill schedule if reproducing iteration counts, occupation rules (equal-filling of the open shell by `FillLowestOrbits`).
4. NO2B of 3N: `V3NO = (1/(2J+1)) sum_{ab} rho_ab sum_{twoJ} (2twoJ+1)...` as implemented with `GetME_pn_no2b`, then `/sqrt2` per identical pair, then `D^T (V2+V3NO) D`.
5. NAT: `rho2 = rho0 + PP + HH + PH` with the regularised denominators, trace check, `C_HO2NAT = C*C_HF2NAT`, descending-occupation ordering, `|Occ|` stored as `occ_nat`; `keep_occ_threshold=0.02` and back-fill `0.005` for `use_NAT_occupations`.
6. Tokyo/NuShellX writers (including thresholds, ordering by proton orbits then neutron orbits, `n+1` in `.sp`).

Deliberately fix (and test the effect):
1. `dE3max` energy bug (`op->l` instead of `oq->l`,`oR->l`; HartreeFock.cc:1368-1369, 1390-1391).
2. `GetME_pn_no2b` precedence error in the reduced branch (ThreeBodyStorage.cc:~398).
3. DIIS dead code / off-by-one (HartreeFock.cc:105-111, 623-678) and the abs() of the DIIS coefficients; the convergence-criterion docs.
4. `std::rand()` in the SCF (non-reproducible); hard-coded `maxiter`, mixing parameters, `1e-8` tolerance, 0.02 NAT threshold, 0.005 back-fill step: expose as options.
5. `exit(0)` on failure paths (DiagonalizeRho, Tr(rho) check), `std::exit` in readers, `goodstate` flags: use proper error returns/exceptions.
6. `ModelSpace::SetReference` rebuilding the whole model space on an occupation change; unfreezing by default with refilling schedule 3,4,5,1,2...
7. `HartreeFock` ctor `freeze_occupations=true` vs driver `freeze_occupations=false` (inconsistent defaults).
8. `ReadTBME_Oslo` off-by-one bound (`a>norbits`), `Count` printing, `nreads*4` printing.
9. Reader parameter silent mismatch: record `emax,e2max,lmax` in a sidecar or validate against the number of values in the file (compare counted values to actual file length, or file size for `.bin`). The reader currently cannot detect a wrong `file2e2max` for text files.
10. Tokyo writer: `Aref/Zref` printed inconsistently between `WriteTokyo` (reference) and `WriteTensorTokyo` (core); `WriteNuShellX` `-999` line uses default stream precision and `Acore+2` constants.
11. `WriteNuShellX_int` T-field hack for pn.
12. Several readers assume `a -= a%2` proton labelling and break for `single_species` model spaces.
13. `PrintSPE` and RPA matrix builders print debug output unconditionally.
14. Native-endian, unversioned binaries: add magic + version + endianness check for any new binary scratch/Omega/C-matrix format (while still being able to read the old `.bin` me2j/me3j and Fortran `stream` formats).

Open questions I could not resolve from the source:
- The exact meaning of the `.bin` 255-byte header in me2j/me3j `.bin` (written by NuHamil/ other tools; not by imsrg++).
- Whether `DiagonalizeRho` ordering of equal occupations (degenerate) gives stable results across LAPACK versions (it does not guard against degenerate-eigenvalue rotations; for the rewrite, expect sign/rotation ambiguity in `C_HF2NAT` for degenerate occupations).
- Exact semantic of the "NO2B" 3N file's `J` index (code names it `J` and sums total J inside `GetME_pn_no2b`); verified only through the reader loop structure.
