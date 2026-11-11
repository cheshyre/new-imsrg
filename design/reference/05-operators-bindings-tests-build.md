# 05 - Operator library, Python bindings, tests, build system (imsrg++ reference analysis)

Reference tree: `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only). All paths below are relative to it.
`src/imsrg_util/Pwd.*` is covered by another analyst; it is mentioned only where the operator library depends on it.

Conventions used in this document:
- "Operator(ms, J, dTz, parity, rank)" is the constructor `Operator(ModelSpace&, int Jrank, int Trank, int parity, int particle_rank)`.
  The second integer is named `Trank` but is used as the change in Tz (|dTz|), e.g. Fermi/GT use 1, double-beta operators use 2.
  Parity = 0 means even (parity-conserving), 1 means odd.
- Rank = particle rank: 1 means a one-body operator stored as OneBody only, 2 means one plus two-body, 3 adds three-body.
- "Reduced" means J-reduced matrix elements are stored (all J-rank > 0 operators; scalars are stored non-reduced unless parity or Tz changing).

---------------------------------------------------------------------------------------------------------------------

## 1. Operator catalog

### 1.1 Entry point and name parsing

- `imsrg_util::OperatorFromString(ModelSpace&, std::string)` at `src/imsrg_util/imsrg_util.cc:56`. A single 480-line if/else-if chain
  (lines 56-536). There is no registry, no enum, no error handling beyond printing "Unknown operator" and returning a default-constructed `Operator`
  (imsrg_util.cc:527-531). Timing is stored in `theop.profiler.timer[opname]`.
- Parsing convention: split the name on `_` (`split_string`, imsrg_util.cc:40) and dispatch either on the full string (no parameter) or on token 0 (parameters
  follow, numeric via `std::istringstream`). No validation of token count (`opnamesplit[1]` out-of-bounds is UB for malformed names).
- The main driver (`src/imsrg++.cc`) takes the parameter `Operators` (a comma-separated vector, `imsrg++.cc:166`), plus `OperatorsPT1`, `OperatorsRPA`,
  `OperatorsTDA` (lines 167-170) which feed the same string factory and then wrap the result with `FirstOrderCorr_1b` / `RPA_resummed_1b(..., "TDA"|"RPA")`
  (imsrg++.cc:757-778). The driver itself does these name expansions before the factory is called:
  - `rhop_all` / `rhon_all` expand to `rhop_r` / `rhon_r` for r = 0.0, 0.2, ..., 10.0 (imsrg++.cc:382-402).
  - `DaggerHF_valence` expands to `DaggerHF_<orbit>` for each valence orbit; `DaggerAlln_valence` likewise (imsrg++.cc:404-424).
  - Special post-processing for the exact name `Rp2`: prints HF point-proton radius and charge radius with `PROTON_RCH2 + NEUTRON_RCH2*(A-Z)/Z + DARWIN_FOLDY`
    (imsrg++.cc:839-846 and 1360-1370).
  - `HCM_<hw>` is also constructed from the driver for the Lawson center-of-mass term (`BetaCM`, imsrg++.cc:551-553).
- `OperatorsFromFile` is a separate path. Tag format `OpName^j_t_p_r^/path/2bfile^/path/3bfile` (imsrg++.cc:200-232); the operator is read from file via
  ReadWrite instead of the factory. Not part of the string catalog, but the rewrite should keep an equivalent "operator from file" input.
- Python exposes the same factory as `pyIMSRG.OperatorFromString` (pyIMSRG.cc:920).

### 1.2 Catalog of names accepted by `OperatorFromString` (exact strings)

Notation: `S` scalar, `T(k)` rank-k tensor in J; "ch-chg" means changes Tz; "dTz=1" etc. Source line numbers refer to imsrg_util.cc unless noted.
Constants are in `src/PhysicalConstants.hh` (HBARC, M_NUCLEON, ALPHA_FS, PROTON_SPIN_G, ...).

#### Radii, densities, moments (one-body unless noted)
| Name | Construction | What it is | Rank/symmetry |
|---|---|---|---|
| `Rp2` | `Rp2_corrected_Op(ms, A_target, Z_target)` (1260) | Point-proton mean-square radius, intrinsic: R2CM + (A-2)/(AZ) sum r_p^2 - 4/(AZ) sum_{i<j} r_i.r_j (proton only) + (1/Z)*spin-orbit correction (`Rso`). Z==0 returns zero op | S, 2-body |
| `Rn2` | `Rn2_corrected_Op` (1268) | Same for neutrons (no spin-orbit term) | S, 2-body |
| `Rm2` | `Rm2_corrected_Op` (1275) | Point-matter radius: (1/A) r^2 - R2CM | S, 2-body |
| `Rm2lab` | `RSquaredOp` (1418) | Lab-frame sum r_i^2 (one-body) | S, 1-body |
| `R2CM` | `R2CM_Op` (1222) | Center-of-mass R^2 (one+two-body, uses r1.r2 integrals) | S, 2-body |
| `R2_p1`, `R2_n1` | `R2_1body_Op(ms,"proton"/"neutron")` (1457) | One-body part of proton/neutron r^2 | S, 1-body |
| `R2_p2`, `R2_n2` | `R2_2body_Op` (1499) | Two-body part (r_i.r_j) of proton/neutron radius, uses `Calculate_r1r2` | S, 2-body |
| `Rso` | `RpSpinOrbitCorrection` (1632) | Spin-orbit correction to the proton radius (Friar-Negele style, uses `PhysConst` anomalous moments) | S, 1-body |
| `Rp2Z_<Z>` | `Rp2_corrected_Op(ms, A_target, Z)` | As Rp2 with Z overridden | S |
| `Rp2AZ_<A>_<Z>` | idem with A and Z both overridden | | S |
| `Rn2Z_<Z>` | `Rn2_corrected_Op` with Z overridden | | S |
| `rhop_<r>`, `rhon_<r>` | `DensityAtR(ms, r, "proton"/"neutron")` (1541) | Point nucleon density at radius r [fm], <i|rho(r)|j> = psi_i(r) psi_j(r)/(4 pi), HO wavefunctions | S, 1-body |
| `rhocentralp_<rmax>`, `rhocentraln_<rmax>` | 10-point weighted average of DensityAtR on a sphere of radius rmax | Central density | S |
| `FFp_<q>`, `FFn_<q>` | `FormfactorAtQ` (1567) | Point proton/neutron form factor F(q)=4pi int r^2 j0(qr) rho(r), q in fm^-1; uses 200-point Gauss-Laguerre + `gsl_sf_bessel_j0` | S |
| `protonFBC_<nu>`, `neutronFBC_<nu>` | `FourierBesselCoeff(ms, nu, 8.0 fm, proton/neutron orbits)` (1695) | Fourier-Bessel expansion coefficient of order nu of the density (R=8 fm hard-coded) | S |
| `E0` | `E0Op` (1661) | Isoscalar monopole, sum r^2 (e-weighted?) | S (J=0) |
| `E1`..`E6` | `ElectricMultipoleOp(ms, L)` (2026) | Electric multipole e fm^L, proton only, J=L, parity = L%2. Reduced | T(L) |
| `E2int` | `IntrinsicElectricMultipoleOp(ms,2)` (2130) | Intrinsic (CM-corrected) E2; flagged "Untested" | T(2) |
| `nE2` | `NeutronElectricMultipoleOp(ms,2)` (2051) | Neutron E2 | T(2) |
| `M1`..`M5` | `MagneticMultipoleOp(ms, L)` (2075) -> `MagneticMultipoleOp_pn(ms,L,"both")` (2083) | Magnetic multipole in mu_N fm^(L-1), uses g_l and PROTON_SPIN_G/NEUTRON_SPIN_G, Suhonen eq 6.24. Reduced | T(L), parity (L+1)%2 |
| `M1p`, `M1n`, `M1S`, `M1L` | `MagneticMultipoleOp_pn` with "proton","neutron","spin","orbit" | M1 restricted to protons, neutrons, spin part, orbital part | T(1) |
| `Anapole` | `AnapoleMoment` (1880) | Anapole moment operator `Operator(ms,1,0,1,2)` | T(1), parity-odd |
| `Schiff_<R>` | `SchiffOp(ms,3,1,R)` (1994) | Schiff moment (r^3 Y1 with CM-correction using Rms R) | T(1), parity-odd |
| `ISM`,`IVM`,`ISQ`,`IVQ`,`ISO`,`IVO` | `MultipoleResponseOp(ms, rL, YL, isospin)` (1911) | Isoscalar/isovector monopole (r^2 Y0), quadrupole (r^2 Y2), octupole (r^3 Y3) response operators (PRC97 054306) | T(YL) |
| `IVD` | `IVDipoleOp(ms,1,1)` (1939) | Isovector dipole | T(1) |
| `ISD_<Rms>` | `ISDipoleOp(ms,3,1,Rms)` (1965) | Isoscalar dipole r^3 Y1 with CM correction 5/3 Rms^2 r | T(1) |

#### Kinetic/CM/trap and generic one/two-body scalars
| Name | Construction | What | Rank |
|---|---|---|---|
| `Trel` | `Trel_Op` (743) | Intrinsic kinetic energy (1-N^-1 form, T - T_cm) with p1.p2 two-body part (`Calculate_p1p2`, lines 821-1092) | S, 2-body |
| `TCM` | `TCM_Op` (821) | Center-of-mass kinetic energy | S, 2-body |
| `Tlab` | `KineticEnergy_Op` (717) | Lab kinetic energy (one-body) | S, 1-body |
| `TrelMassCorrection` | `Trel_Masscorrection_Op` (1049) | Mass-difference (p/n) correction to Trel | S |
| `HCM` / `HCM_<hw>` | TCM + 1/2 A m w^2 R2CM/(hbarc)^2 (imsrg_util.cc:160-176) | Lawson CM Hamiltonian, optional frequency in name (default ms.hbar-omega) | S |
| `VCM_<hw>` | The oscillator term only of HCM | | S |
| `Iso2` | `Isospin2_Op` (1754) | Total isospin squared T^2 (one and two-body, diag 3/4 one-body piece) | S |
| `Tz2` | `TzSquared_Op` (1861) | Tz^2 | S |
| `OccRef` | `NumberOpRef` (564) | Number operator on reference (hole) orbits | S, 1-body |
| `OneOcc_<orb>` | `NumberOp(ms, n,l,j2,tz2)` (541), orbit given as string e.g. `1p3` -> via `ModelSpace::String2Index` | Occupation number of orbit | S |
| `AllOcc_<l j tz>` | `NumberOpAlln` (553) | Occupation summed over radial quantum number n | S |
| `OBD_<orb1>_<orb2>` | `OneBodyDensity` (577) | One-body density matrix element: 0.5 symmetric pair at (i,j),(j,i) | S, 1-body |
| `RadialOverlap` | `RadialOverlap` (2624) | `Operator(ms,0,1,0,1)`: radial overlap <n l|n' l> between p and n (isospin-changing); "Untested" | S, dTz=1 |
| `Sigma`, `Sigma_p`, `Sigma_n` | `Sigma_Op`, `Sigma_Op_pn` (2506-2512) | Spin operator sigma (all / proton / neutron), `Operator(ms,1,0,0,2)` | T(1) |
| `LdotS` | `LdotS_Op` (2641) | L.S one-body operator | S |
| `L2rel` | `L2rel_Op` (4302) | Relative orbital angular momentum squared (two-body); "Untested" | S |
| `LCM` (not in factory, only in header; `LCM_Op`, 2656) | CM angular momentum | | T(1) |
| `QdotQm`,`QdotQp`,`QdotQn`,`QdotQE3` | `QdotQ_Op(ms, lambda, ep, en)` (2687) | Q.Q two-body, isoscalar/proton/neutron/octupole; "Untested" | S |
| `VQQ` | `VQQ_Op` (2771) | Quadrupole-quadrupole interaction | S |
| `VCoul` | `VCoulomb_Op` (2866) | Coulomb interaction e^2/r between protons in HO basis (Talmi-Moshinsky, uses `TalmiI`, `lmax` parameter) | S |
| `VCentralCoul` | `VCentralCoulomb_Op` (2901) | Coulomb with central-only (lmax) variant | S |
| `AxialCharge` | `AxialCharge_Op` (3052) | `Operator(ms,0,1,1,2)`: axial charge (parity-odd, dTz=1) | S, dTz=1, odd |
| `VMinnesota` | `MinnesotaPotential` (4436) | Minnesota NN potential, J-coupled via `MinnesotaMatEl` (4681) | S |
| `VGaus_<sigma>`, `VGausCSB_<sigma>` | `GaussianPotential`, `GaussianCSBPotential` (4475, 4514) | Gaussian two-body potential (CSB variant) | S |
| `VSDI_<V0>_<R>` | `SurfaceDeltaInteraction` (5004) | Header says "not yet implemented" but a body exists (5004) | S |
| `VBareDelta` | `BareDelta` (4557) | Delta (contact) interaction | S |
| `VWS_<V0>_<R0>_<a0>` | `WoodsSaxon1b_Op` (3308) | Woods-Saxon one-body potential via radial grid + HO wavefunctions (ModelSpace 1b) | S, 1-body |
| `HOtrap_<hw>` | `HOtrap_Op` (3346) | HO trap potential, one-body, numeric radial integration | S |
| `VPT_<c1>_<c2>_<c3>...` | `TViolatingPotential_Op(ms, LECs)` (3081) | Parity- and time-reversal-violating NN potential, `Operator(ms,0,0,1,2)`; needs >= 3 LECs | S, parity-odd |
| `Fermi` | `AllowedFermi_Op` (2453) | Allowed Fermi beta decay operator tau_+ (`Operator(ms,0,1,0,2)`) | S, dTz=1 |
| `GamowTeller` | `AllowedGamowTeller_Op` (2483) | Allowed GT sigma tau_+ (`Operator(ms,1,1,0,2)`) | T(1), dTz=1 |
| `hfsNMS`, `hfsSMS` | `atomic_hfs::NormalMassShift/SpecificMassShift(ms, A=1)` (Atomic.cc:160ff) | Atomic normal/specific mass shift (electron-based units, eV) | S |
| (header only, not in factory) | `atomic_fs::{Darwin,RelativisticT,SpinOrbit}`, `atomic_hfs::{hQ,hD,CombinedMassShift}` (Atomic.cc:42-190) | Atomic-physics (electron HO basis) fine/hyperfine-structure operators | S or T(1) |
| `Dagger_<orb>`, `DaggerHF_<orb>`, `DaggerAlln_<orb>` | `Dagger_Op`, `DaggerAlln_Op` (2836, 2848) | Particle-adding "dagger" operator a^dag_Q: `SetNumberLegs(3)`, `SetQSpaceOrbit(Q)`, `OneBody(Q,0)=1`, non-Hermitian. Used for the "dagger" (valence-Q-space) formalism and its commutators | Rank-1/2 spherical tensor with 3 legs |

#### Neutrinoless double beta decay (src/imsrg_util/M0nu.cc, 1732 lines)
All are `Operator(ms, J=0, dTz=2, parity=0, rank=2)`, built in the lab-frame HO basis from momentum-space potentials via partial-wave decomposition (`PWD` class),
Talmi-Moshinsky transformation of relative matrix elements (`M0nu::TalmiMoshinkyTransform`, uses `ModelSpace::PreCalculateMoshinsky`) and Gauss-Legendre
(`gsl_integration_glfixed_table`) momentum integrals (M0nu.cc:883, 926). Reduced, Hermitian-flagged (M0nu.cc:400-403).
| Name | Function | Meaning |
|---|---|---|
| `M0nu_GT_<Ecl>_<src>[_<ff>]` | `M0nu::GamowTeller(ms, Eclosure, src, formfactor)` (507) | Light-neutrino-exchange Gamow-Teller part with closure energy Ecl (MeV); optional ff in {AA,AP,PP,MM} selects separated form-factor pieces `hGT_AA`... default `GTFormFactor`. R = R0 A^(1/3), R0=1.2 fm |
| `M0nu_F_<Ecl>_<src>` | `M0nu::Fermi` (535) | Fermi part (`hF_VV` / `FermiFormFactor`) |
| `M0nu_T_<Ecl>_<src>[_<ff>]` | `M0nu::Tensor` (563) | Tensor part (ff AA/AP/PP/MM) |
| `M0nu_C_<cutoff>_<power>` | `M0nu::Contact` (600) | Contact (short-range counterterm) with regulator cutoff and power |
| `M0nuHeavy_{GT,F,T}_<src>_<ff>` | `GamowTellerHeavy`, `FermiHeavy`, `TensorHeavy` (627-671) | Heavy-neutrino exchange variants (GT: AA/AP/PP; F: VV; T: AP/PP) |
| `M0nuIntegrand_{GT,F,T}_<Ecl>_<src>_<ip>_<ipp>` and `M0nuIntegrand_C_<cut>_<pow>_<ip>_<ipp>` | `*_integrand` (1534-1705) | The integrand at one (p, p') momentum-mesh point, for external integration (Python-side). Uses `PreCalculateMomentumSpaceIntegrands` |
| `M0nuR_<Ecl>_<r12>` | `GamowTeller_R` (945) | GT matrix elements at fixed relative distance r12 (radial dependence) |
| `DGT` | `DGT_Op` (700) | Double Gamow-Teller (2nuBB) operator |
| `DGTR_<r12>`, `DGTRLocal_<r12>` | `DGT_R` (1081), `DGT_R_SurfaceLocalization` (1311) | Radial dependence of DGT (with surface localization variant, uses `integrateRcom`/`GetDGTRcomIntegral`) |
Notes: the `src` argument (short-range-correlation "src" string like `none`, `AV18`, ...) is accepted and propagated by every `M0nu::*` function but is
**never read** in M0nu.cc (grep shows only the signatures), so SRC is effectively not implemented in this layer. The "closure" potentials are hard-coded
constants (`MAGMOM=4.706`, `CUTV=850`, `CUTA=1086` MeV dipole form-factor cutoffs, `M0nu.hh`). PWD mesh sizes are hard-coded: 100 angular, 500 momentum points,
max momentum 25 fm^-1 (M0nu.cc:515-519). Legacy `M0nu_contact_Op(ms,R0)` (imsrg_util.cc:4112) is the older contact term (not in the factory).

#### Dark-matter non-relativistic EFT (src/imsrg_util/DarkMatterNREFT.cc, 1014 lines)
Name format `DMNREFT_<Op><IsoSV>_<q>_<J>` e.g. `DMNREFT_M+_0.5_2` (imsrg_util.cc:290-325). `<Op>` in {M, Sigma, Sigmap, Sigmapp, Delta, Deltap, Phip, Phipp, Phitp,
Omega, Omegat}; the single trailing character is the isospin selector IsoSV: `+` isoscalar, `-` isovector (neutron gets minus sign), `p` proton only, `n`
neutron only (DarkMatterNREFT.cc:259-271). q is the momentum transfer in MeV (y = q^2 b^2/4), J the multipole. Each is a reduced one-body tensor of rank J,
Tz=0, parity = J%2 for normal-parity operators (M, Sigma', Sigma'', Delta, ...) and opposite for others. Helpers `Ms(J,L,q)` and `Mg(J,L,q)` are Bessel-function
operators (`jho`, `jdmho`, `jdpho`, `PhiS1..PhiS4`, `PhiF`) built from `gsl_sf_hyperg_1F1`, `gsl_sf_coupling_3j/6j`, `gsl_sf_fact/doublefact`
(55 GSL calls). Unknown operator part silently leaves `theop` empty (no else branch, imsrg_util.cc:314-317). Comment: `Omega` "neither Symmetrize nor AntiSymmetrize"
(DarkMatterNREFT.hh).

#### Not callable through the string factory (exposed only in C++/Python)
`E0`..`E6` are in the factory; these are not: `LCM_Op`, `Isospin2` is; `SerberTypePotential` (Python only), `EKKShift` (5094, used by driver
directly), `Single_Ref_1B_Density_Matrix` / `Get_Charge_Density` (header: "This doesn't work"), `SplitUp`, `Embed1BodyIn2Body`, `GetEmbeddedTBME`, `Reduce`/`UnReduce`,
`FirstOrderCorr_1b`, `RPA_resummed_1b` (3568-4112, TDA/RPA resummation of 1-body operators, ~550 lines), `GetPH_transformed_Gamma`,
`MBPT2_SpectroscopicFactor` (3401), `GetOccupationsHF`, `GetDensity`, `WriteSPWaveFunctions`, `GetSecondOrderOccupations` (declared; not checked).

### 1.3 Helper numerics in imsrg_util (what the rewrite must provide)
- `HO_density(n,l,hw,r)` (590), `HO_Radial_psi(n,l,hw,r)` (599): HO radial wave function using `gsl_sf_laguerre_n`, `gsl_sf_fact`, `gsl_sf_doublefact`. `HartreeFock::GetRadialWF` (HartreeFock.cc:1275) duplicates the formula inline. `M0nu::HO_Radial_psi` also declared (M0nu.hh) and `HO_Radial_psi_mom` (momentum-space).
- `RadialIntegral(na,la,nb,lb,L)` (2247): <n_a l_a| r^L |n_b l_b> in HO units (closed form sum over sigma with `tgamma` and `gsl_sf_fact`); for odd la+lb+L it falls through to `RadialIntegral_RpowK` (2271) which uses Talmi B coefficients (long double, lgamma-based) and `TalmiI(p,k)` (2340). `RadialIntegral_Gauss(..., sigma)` (2318) uses 200-pt Gauss-Laguerre.
- `TalmiB`, `TalmiB_SingleTerm`, `TalmiB_SingleTermPair` (2349-2440): Talmi-integral coefficients with log-gamma stabilised long double; duplicate declaration `AngMom::TalmiB` (exported to Python separately) and `M0nu::TalmiB` (declared in M0nu.hh).
- `FrequencyConversionCoeff(n1,l1,hw1,n2,l2,hw2)` (5223): overlap between HO functions of different frequencies via `gsl_integration_qagiu` (the only adaptive GSL quadrature in the tree).
- `GaussLaguerre.hh` (1702 lines): hard-coded Gauss-Laguerre nodes/weights, `gauss_laguerre_points[n][i] = {x_i, w_i}` for n = 0..50 and 100 (comment says "0-50, 100, and 200") and a 200-point table `gauss_laguerre_points_200` used by `FormfactorAtQ`, `RadialIntegral_Gauss`, `FourierBesselCoeff`. Note the weights include exp(x) (`int f = sum w_i exp(x_i) f(x_i)`), i.e. generalised "scaled" weights. In a rewrite generate these at startup (Golub-Welsch) or use a library; the 200-point table is the one that matters.
- Moshinsky brackets: `ModelSpace::GetMoshinsky`, `PreCalculateMoshinsky`, `AngMom::Moshinsky` (exposed in Python). M0nu, Coulomb, Minnesota, Gaussian, SDI, r1.r2, p1.p2 all use Talmi-Moshinsky relative-to-lab transformations.
- `Calculate_p1p2`, `Calculate_r1r2`, `Calculate_p1p2_all(Operator&)` (1092, "IN PROGRESS... doesn't work yet, and for now it's slower": imsrg_util.cc:1089).
- `Helicity.cc/.hh` (791+64 lines): helicity-basis partial-wave decomposition of tensor 2-body operators and Wigner-d/Jacobi polynomials. **Not compiled**: `src/imsrg_util/CMakeLists.txt` has `# Helicity.hh Helicity.cc` commented out; M0nu.cc line 4 has `// #include "Helicity.hh"`. The Makefile OBJ list also omits it. Treat as dead/experimental code.
- `GetSecondOrderOccupations`, `Reduce`/`UnReduce`, `Embed1BodyIn2Body` are generic utilities that probably belong in the core library rather than the operator library.

### 1.4 Numbers
- `imsrg_util.cc`: 5323 lines, ~118 `std::cout` statements inside operator constructors (debug printing in library code: `NumberOp`, `OneBodyDensity`, `Dagger_Op`, `AxialCharge_Op` etc.).
- 37 GSL call sites in imsrg_util.cc, 55 in DarkMatterNREFT.cc, 22 in M0nu.cc, 16 in Pwd.cc, 18 in Helicity.cc (not built), 4 in Atomic.cc.
- OpenMP is used in the constructors (`#pragma omp parallel for schedule(dynamic,1)` over two-body channels) - 32 files, 400+ pragmas across src.

---------------------------------------------------------------------------------------------------------------------

## 2. Python bindings (src/pyIMSRG.cc, 961 lines)

### 2.1 Technology and data exchange
- pybind11 vendored in `src/extern/pybind11`, version string in `include/pybind11/detail/common.h`: **2.13.0.dev1** (a development snapshot). Headers used: `pybind11.h`, `operators.h`, `stl.h`, `numpy.h` (included but **never used**: zero occurrences of `py::array`, `buffer_info` or `def_buffer` in pyIMSRG.cc).
- Module name `pyIMSRG` (`PYBIND11_MODULE(pyIMSRG, m)`, pyIMSRG.cc:19); built with `pybind11_add_module(pyIMSRG pyIMSRG.cc)` linking the big shared lib `libIMSRG.so`, `SUFFIX ".so"` forced (src/CMakeLists.txt).
- **Armadillo matrices are NOT exchanged as numpy arrays.** `arma::mat` is registered as an opaque class `ArmaMat` (pyIMSRG.cc:260-284) with methods `zeros`, `Print`, `__str__`, `save`, `load`, `t`, `Set(i,j,x)`, `__call__(i,j)` (returns a pointer to the element), `Getn_rows`, `Getn_cols`, `Schur_Prod`, `Norm`, `trace`, `sum`, arithmetic operators. `Operator.OneBody` is exposed as `def_readwrite` of type `arma::mat` (i.e. as an `ArmaMat` object); `Operator.TwoBody` / `.ThreeBody` as `TwoBodyME`/`ThreeBodyME` classes. A numpy matrix cannot be passed in; users must loop `Set(i,j,x)` or use `Operator.SetOneBodyME`. Armadillo vectors/index lists are converted with hand-written helpers `ArmaToSTL_uint/double` (pyIMSRG.cc:13-14) to `std::vector` (Python list) for `TwoBodyChannel.GetKetIndex_*`.
- STL containers are auto-converted by `pybind11/stl.h` (std::vector, std::set<index_t>, std::map, std::deque of Operator, std::pair, tuples). `ModelSpace.holes/particles/core/valence/qspace/all_orbits` are `def_readwrite` of `std::set/vector<index_t>` so they are **copied** on each access.
- Return-value policies: mostly default; many methods wrap lambdas returning copies. `Operator` returned by value everywhere (copy of potentially hundreds of MB; a performance hazard for large 3-body storage).
- Hash/unhash helpers are exposed as functions returning tuples (`NineJUnHash`, `HartreeFock.Vmon3UnHash`).
- No GIL release anywhere (`py::call_guard<py::gil_scoped_release>` is absent), so long computations block the interpreter; no buffer protocol; no `__repr__` except ArmaMat `__str__`; no pickling.
- A custom pre-link step removes the old `.so` before relinking "to prevent SIGBUS on running jobs" (src/CMakeLists.txt, `add_custom_command(TARGET pyIMSRG PRE_LINK ...)`).

### 2.2 Exposed classes and methods (the checklist for the new bindings)
Counts are number of `.def` entries.

- `Orbit`: ctor; attrs `n, l, j2, tz2, occ, cvq, index`.
- `TwoBodyChannel` (rw `J, parity, Tz`): `GetNumberKets, GetLocalIndex(p,q), GetKetIndex, GetKet(i)`, `GetKetIndex_{pp,hh,ph,cc,vc,qc,vv,qv,qq}` (lists). `TwoBodyChannel_CC` (attr only, note: `def_readwrite("J", &TwoBodyChannel::J)` is bound on the CC class, copy/paste bug).
- `ThreeBodyChannel` (rw `twoJ, parity, twoTz`): `GetNumberKets, GetLocalIndex(p,q,r,Jpq), GetKet(i)`. `Ket(Orbit,Orbit)` rw `p,q`; `Ket3(Orbit x3)` rw `p,q,r,Jpq`.
- `ModelSpace` (ctors: default, copy, `(emax, reference)`, `(emax, reference, valence)`, `(emax, hole_list, valence_list)`, `(emax, hole_list, core_list, valence_list)`):
  setters `SetHbarOmega, SetTargetMass, SetTargetZ, AddOrbit(n,l,j2,tz2,occ,cvq), SetupKets, Setup3bKets, SetOcc, SetOccNAT, SetEmax, SetE2max, SetE3max, SetdE3max, SetLmax, SetEmaxUnocc, SetEmax3Body, SetReference (set / string / occupations map), SetScalarFirstPass, SetScalar3bFirstPass, ResetFirstPass, Init_occ_from_file, InitSingleSpecies`;
  getters `GetHbarOmega, GetTargetMass, GetTargetZ, GetAref, GetZref, GetNumberOrbits, GetNumberKets, GetNumberTwoBodyChannels(+_CC), GetNumberThreeBodyChannels, GetEmax, GetE2max, GetE3max, GetOrbit(i), GetKet(i), GetTwoBodyChannel(ch), GetTwoBodyChannelIndex(J,parity,Tz), GetThreeBodyChannel, GetThreeBodyChannelIndex, GetOrbitIndex(n,l,j2,tz2), GetOrbitIndex_fromString, GetOneBodyChannels(l,j2,tz2), Index2String`;
  precalc/cache `PreCalculateSixJ, PreCalculateNineJ, PreCalculateMoshinsky, GetMoshinsky, GetSixJ, GetNineJ, NineJHash, NineJUnHash`; misc `FindEFermi, ClearVectors, Print`; attrs `holes, particles, core, valence, qspace, all_orbits`.
- `Operator` (ctors: default, `(ms)`, copy, `(ms, j_rank, t_rank, parity, particle_rank)`): arithmetic `+ - += -= * / (double) unary-`, `+/- double`; attrs `ZeroBody, OneBody (ArmaMat), TwoBody, ThreeBody`;
  access `GetOneBody, SetOneBody, SetOneBodyME, GetTwoBody(ch_bra,ch_ket,ibra,iket), SetTwoBody, GetTwoBodyDimension, GetOneBodyChannel, ScaleOneBody, ScaleTwoBody, EraseZeroBody/OneBody/TwoBody/ThreeBody`;
  normal ordering `DoNormalOrdering, DoNormalOrderingCore, DoNormalOrderingFilledValence, UndoNormalOrdering, UndoNormalOrderingCore, ReNormalOrderCore(copy_3body=False)`;
  misc `SetModelSpace, Truncate, DoIsospinAveraging, Norm, OneBodyNorm, TwoBodyNorm, ThreeBodyNorm, Set/IsHermitian, SetAntiHermitian, IsAntiHermitian, SetNonHermitian, IsReduced, MakeReduced, MakeNotReduced, MakeNormalized, MakeUnNormalized, GetParticleRank, SetParticleRank, GetJRank, GetTRank, GetParity, GetNumberLegs, Size, Print{OneBody,TwoBody(_ch,_chch),ThreeBody}, PrintTimes, GetMP2_Energy, GetMP2_3BEnergy, GetMP3_Energy, GetPPHH_Ladders, ReadBinary(filename), WriteBinary(filename)`.
- `ArmaMat`: see 2.1.
- `TwoBodyME`: `GetTBME_J(Jbra,Jket,a,b,c,d), GetTBME_J_norm, GetTBMEmonopole(+_norm), GetTBME_norm(ch_bra,ch_ket,a,b,c,d), GetChannelMatrix(J,parity,Tz) (ArmaMat), GetTBMEnorm_chij, SetTBME_chij, PrintAll, PrintMatrix, IsReduced, Erase, Norm`, arithmetic with TwoBodyME.
- `ThreeBodyME`: `GetME_iso(13 ints), GetME_pn(Jab,Jde,twoJ,a..f), GetME_pn_tensor(...), SetME_pn_ch, GetME_pn_no2b, RecouplingCoefficient, TransformToPN, SwitchToPN_and_discard, Erase, SetMode, IsAllocated, IsHermitian,...` (pyIMSRG.cc:330-364).
- `ReadWrite` (51 methods): `ReadTBME_Oslo, ReadTBME_OakRidge, ReadBareTBME_Jason/Navratil/Darmstadt, Read_Darmstadt_3body, ReadOperator2b_Miyagi, Read3bodyHDF5, Write_me2j(_gz), Write_me3j, WriteTBME_Navratil, WriteNuShellX_{sps,int,op}, ReadNuShellX_int(_iso), WriteAntoine_{int,input}, WriteOperator(+Human), ReadOperator(+Human), CompareOperators, WriteOneBody_Simple, Read/Write OneBody/TwoBody_Takayuki, WriteTensorOneBody/TwoBody, Write/ReadTokyo, Write/ReadTensorTokyo, WriteOneBody_Oslo, WriteTwoBody_Oslo, SetCoMCorr, ReadTwoBodyEngel, ReadOperator_Nathan, ReadTensorOperator_Nathan, ReadRelCMOpFromJavier, Set3NFormat, WriteDaggerOperator, ReadJacobi3NFiles, WriteValence3body, SetScratchDir, GetScratchDir, CopyFile, ReadDarmstadt_2bodyRel, ReadH2_2body, Read2bCurrent_Navratil`.
- `HartreeFock`: `Solve, TransformToHFBasis, GetHbare, GetNormalOrderedH(_Cin), GetOmega, PrintSPE, PrintSPEandWF, GetRadialWF_r, GetHFPotential, GetAverageHFPotential, GetValence3B, FreeVmon, UpdateDensityMatrix, UpdateF, BuildMonopoleV, CalcEHF, PrintEHF, FillLowestOrbits, DiscardNO2Bfrom3N, FreezeOccupations, UnFreezeOccupations`; static `Vmon3Hash/UnHash`; attrs `EHF, F, rho, C, Vmon3_keys, Vmon3`.
- `HFMBPT(HartreeFock)`: `UseNATOccupations, GetNaturalOrbitals, TransformHOToNATBasis, TransformHFToNATBasis, GetNormalOrderedHNAT, GetMP2_Energy, GetMP3_Energy, GetMP3_{pp,hh,ph}, GetMP4_term, GetMP4_Diagram, SetVVApprox, SetVVGenerator_type/_partitioning, SolveVanVleck, VV_TransformH, VV_Transform, C_HO2NAT, C_HF2NAT`.
- `IMSRGSolver` (ctor from Operator): `Solve, Transform, InverseTransform, SetFlowFile, SetMethod, SetEtaCriterion, SetDs, SetdOmega, SetOmegaNormMax, SetSmax, SetDsmax, SetHin, SetODETolerance, Reset, SetGenerator, SetOnly2bEta, SetOnly1bEta, SetDenominator{Cutoff,Delta,DeltaOrbit,Partitioning}, GetSystemDimension, GetOmega (overloaded: index / whole deque), SetOmega, GetEta, GetH_s, SetH_s, GetS, SetMagnusAdaptive, SetReadWrite, SetHunterGatherer, CalculatePerturbativeTriples (2 overloads), AddOperator, GetOperator, EstimateBCHError, UpdateEta, GetNOmegaWritten, GetOmegaSize, GetScratchDir, FlushOmegaToScratch`; attrs `generator, Eta, n_omega_written`.
- `IMSRGSolverPV(IMSRGSolver)`: `Etapv, Solve_RK4, Solve_magnus_euler, AddOperatorPV, GetOperatorPV, GetVPT_s, SetGeneratorPV, SetOnly1bEta, Transform`.
- `Generator`: `SetType, SetDenominatorPartitioning, SetUseIsospinAveraging, Update, UpdateGeneral, GetHod_SingleRef, GetHod`. `GeneratorPV(Generator)`: `SetType, Update`.
- `IMSRGProfiler`: `PrintTimes, PrintCounters, PrintAll, PrintMemory, Clear, GetTimer, GetCounter`.
- `Jacobi3BME`: `GetDimensionAS/NAS, GetMatElAS/NAS, SetEmax/E2max/E3max, ComputeNAS_MatrixElements, GetLabMatEl, TestReadTcoeffNavratil, GetV3mon_all`.
- Submodule `Commutator` (pyIMSRG.cc:595-): `Commutator, CommutatorScalarScalar, CommutatorScalarTensor, SetUseIMSRG3, SetUseIMSRG3N7, SetUseIMSRG3N7_Tensor, SetUseIMSRG3_Tensor, TurnOnTerm(name), TurnOffTerm(name), SetThreebodyThreshold, SetVerbose, SetSingleThread, PrintSettings, SetPerturbativeTriples, DoPandyaTransformation_SingleChannel`, and every individual term `comm110ss ... comm333_pph_hhpss` (ss), `comm111st ... comm222_phst` (st), `SetIMSRG3Noqqq, SetIMSRG3Onlyvvv, SetIMSRG3valence2b, SetPertTripNovvv, Discard{0,1,2}bFrom3b`, `comm331st ... comm333_pph_hhpst`. Sub-submodule `Commutator.FactorizedDoubleCommutator`: `comm223_231, comm223_232, comm223_231_chi{2b,1b}, comm223_232_chi{2b,1b}, SetUse_GooseTank_{1b,2b}, SetUse_1b/2b_Intermediates, SetUse_GooseTank_only_{1b,2b}, SetUse_TypeII_{1b,2b}, SetUse_TypeIII_{1b,2b}, SetUse_GT_TypeI_2b, SetUse_GT_TypeIV_2b`.
- Submodule `BCH`: `BCH_Transform, BCH_Product, SetUseGooseTank, SetUseFactorizedCorrection, SetUseFactorizedCorrectionBCH_product, SetUseFactorized_Correct_ZBTerm, SetOnly2bOmega, SetComm223_231, SetComm223_232, Set_BCH_Transform_Threshold, Set_BCH_Product_Threshold, SetBCHSkipiEq1`.
- Submodule `ReferenceImplementations`: all reference `commXYZ{ss,st}` terms (~60 functions), the factorised diagrams `diagram_CIa ... diagram_DIVb(_intermediate)`, `comm223_231_BruteForce`, `comm223_232_BruteForce`, `TriplesGuess, TestMBPT6, TestMBPT6_2`.
- `RPA`: `ConstructAMatrix, ConstructBMatrix, SolveCP, SolveTDA, SolveRPA, TransitionToGroundState, PVCouplingEffectiveCharge, GetEnergies, GetX, GetY, PrintA, PrintB, GetEgs, StraightforwardCPEffectiveCharge`.
- `UnitTest` (ctor from ModelSpace; ~130 methods): see section 3.
- Free functions on the module `m`: `BuildVersion`; operator library: `TCM_Op, Trel_Op, R2CM_Op, HCM_Op, NumberOp, RSquaredOp, RpSpinOrbitCorrection, E0Op, AllowedFermi_Op, AllowedGamowTeller_Op, ElectricMultipoleOp, MagneticMultipoleOp, SchiffOp, Sigma_Op, Isospin2_Op, LdotS_Op, OperatorFromString, SerberTypePotential, RPA_resummed_1b, FirstOrderCorr_1b, MBPT2_SpectroscopicFactor, Single_Ref_1B_Density_Matrix, Get_Charge_Density, Embed1BodyIn2Body, Calculate_p1p2_all`; numerics: `HO_density, HO_Radial_psi, RadialIntegral, RadialIntegral_RpowK, RadialIntegral_Gauss, FrequencyConversionCoeff, GetOccupationsHF, GetDensity, TalmiI, TalmiB, Tcoeff`; angular momentum: `CG, ThreeJ, SixJ, NineJ, NormNineJ, Moshinsky, FillFactorialLists, factorial, double_fact, AngMomJmin, AngMomJmax`; global switches: `SetUseGooseTank, SetUseIMSRG3, SetUseIMSRG3N7` (duplicates of the Commutator/BCH submodule ones).

Observation: the bindings expose implementation-level internals (every commutator term, factorial caches, Hash/UnHash, Vmon3 internal containers). A new API should separate a small stable "physics" API (ModelSpace, Operator, HF, IMSRGSolver, ReadWrite, operator factory, numpy views of 1b/2b blocks) from a debug/test submodule.

### 2.3 How Python scripts use the bindings
- `test/*.py`, `tests/UnitTest.py` (see section 3): only `ModelSpace`, `UnitTest.{RandomOp,TestCommutators,TestCommutators_Tensor,TestNormalOrdering,TestFactorizedDoubleCommutators,TestPerturbativeTriples,SanityCheck}`, `Operator.{MakeNotReduced,IsReduced,UndoNormalOrdering,DoNormalOrderingCore,ReNormalOrderCore,Norm}`, `Operator` arithmetic (`1.0*Rando`, `Rv1 - Rv2`), `Operator.ThreeBody` attribute assignment, `pyIMSRG.Commutator.{TurnOnTerm,TurnOffTerm,SetUseIMSRG3,SetUseIMSRG3N7}`, `ModelSpace.PreCalculateSixJ`, `IMSRGProfiler().PrintTimes()`. Exit code = `not passed`.
- `work/scripts/python/GetNormalOrderedH.py` (the only Python file using the library besides tests): `from pyIMSRG import *`, loads the library, constructs a ModelSpace, reads a Hamiltonian with `ReadWrite`, normal-orders. `work/scripts/goUniversal.py` is a job-submission wrapper (PBS/SLURM detection, builds the `imsrg++` command-line from a dict of parameters, hard-coded absolute path `/Users/antoinebelley/...` and a hard-coded e-mail address) and does not import pyIMSRG.
- `work/compiled/` holds a Makefile, `Parameters.hh` copy and example C++ drivers (`SingleRef.cc`, `Universal.cc`, `Universal_tensor_debug.cc`, `ValenceSpace.cc`, `ValenceSpace_targeted.cc`); the C++ "driver as library user" pattern is `#include "IMSRG.hh"` (aggregate header, 41 lines) and link `-lIMSRG`.

---------------------------------------------------------------------------------------------------------------------

## 3. Unit tests

### 3.1 Test infrastructure
- Tests are **methods of a `UnitTest` class compiled into the production library** (`src/UnitTest.hh` 183 lines / `UnitTest.cc` 8662 lines, 47% of the build's `src/*.cc` is test code), driven only from Python (ctest -> python3 script) or the command line `imsrg++` driver is not involved. No gtest/catch2; no C++ test executable.
- Pass/fail: each test returns `bool`; a Python script exits with `not passed`.
- Tolerances: **absolute, summed errors < 1e-6** nearly everywhere (99 uses of `< 1e-6`, 41 of `> 1e-6`); a few stricter 1e-7/1e-8/1e-9 and one 1e-4. No relative tolerances; the J-coupled test uses sum of squared/absolute errors across all matrix elements.
- Random operators: `UnitTest::RandomOp(ms, jrank, tz, parity, particle_rank, hermitian)` (UnitTest.cc:28-208).
  - Seed: static member `UnitTest::random_seed = 1` (UnitTest.cc:20), incremented on every call (`random_seed++`); Armadillo RNG set with `arma_rng::set_seed(random_seed)` and 3-body uses `std::default_random_engine(random_seed)` + `std::normal_distribution(0,1)`. Therefore results are reproducible but **order dependent** (the n-th RandomOp call in a process always gets seed n+1). `SetRandomSeed` is exposed to Python.
  - Symmetry handling: one-body: fill `randn`, symmetrize `A + h A^T` with h = +/-1, then mask with `symmetry_allowed(i,j)` (nonzero only inside allowed one-body channels `OneBodyChannels[{l,j2,tz2}]`, with a phase `(-1)^((j2i-j2j)/2)` for reduced storage with j > i). Two-body: `randn` each channel matrix, hermitise only for bra-channel == ket-channel; stored two-body matrices are normalised, antisymmetrised J-coupled (so no extra antisymmetry needed).
  - Three-body (the hard part): fill in `pn` mode (`SwitchToPN_and_discard`); loop bra/ket 3-body kets; for kets with repeated orbits (|a b b>, |a a a>) the random value is built from a random buffer of 80 values indexed `(Jab + Jde) % 80` and recoupled with the permutation phases and `RecouplingCoefficient` so antisymmetry constraints are honored; hermiticity/antihermiticity is imposed by sign flips on the diagonal block (`ibra == iket and Jde > Jab` -> `*= herm`) and skipping `<abb|abb>` for anti-Hermitian.
  - **Bug**: in the "easy" branch (no repeated orbits) the line reads `double random_me = distribution(generator);` which *re-declares* a local shadowing the outer `random_me` (UnitTest.cc:~190) so the outer value stays 0 and `SetME_pn_ch` stores 0. As written, three-body test operators are non-zero only for kets with a repeated orbit. (The lines `size_t bufsize = 80; randbuf` use of `(Jab+Jde) % bufsize` is also a crude, non-independent hash.) Worth re-checking against the compiler warning `-Wshadow`; the rewrite should generate fully independent random ME for the independent entries.
  - `RandomDaggerOp(ms, Q)` (UnitTest.cc:211) builds random 3-leg dagger operators (functions `TestDaggerCommutators(Q)` and `...Alln(Q)`).
- M-scheme evaluation: `GetMschemeMatrixElement_{1b,2b,3b,1leg,3leg}(Op, a,ma, b,mb, ...)` (UnitTest.cc:1134-1456). They compute m-scheme matrix elements of a J-coupled (reduced or not) operator by explicit Clebsch-Gordan coupling (`AngMom::CG`), recoupling with `Op.TwoBody.GetTBME_J` / `ThreeBody.GetME_pn`; only implemented for scalar operators (calls `exit(0)` with "WARNING!!! not yet implemented for tensor operator" for J-rank > 0, UnitTest.cc:1456-1457).

### 3.2 What is tested and how (two independent validation layers)
1. **Optimised commutators vs. reference J-coupled implementations** (`Test_against_ref_impl(X, Y, ComOpt, ComRef, tag)`, UnitTest.cc:1463-1555): builds `Z`, `Zref` with the correct J/Tz/parity/rank and hermiticity, calls the optimised `Commutator::commXYZ` and the straightforward loop implementation in `ReferenceImplementations::commXYZ` (src/ReferenceImplementations, separate CMake object library), and requires `|norm(Z - Zref)| < 1e-6` where norm = `Operator::Norm() + ZeroBody`. Wrappers `Test_comm110ss ... Test_comm333_pph_hhpss` (scalar-scalar, 24 terms: 110, 220, 111, 121, 221, 122, 222_pp_hh, 222_ph, 222_pp_hh_221, 330, 331, 231, 132, 232, 223, 133, 332_ppph_hhhp, 332_pphh, 233_pp_hh, 233_ph, 333_ppp_hhh, 333_pph_hhp) and `Test_comm*st` (scalar-tensor, 18 terms: 111st, 121st, 122st, 221st, 222_pp_hhst, 222_phst, 331st, 231st, 232st, 132st, 223st, 133st, 332_pphhst, 332_ppph_hhhpst, 233_pp_hhst, 233_phst, 333_ppp_hhhst, 333_pph_hhpst). Dagger tensors `Test_comm{211,231,431,233,413,433_pp_hh,433_ph}sd` (UnitTest.cc:5294-5900) compare `DaggerCommutators` with an m-scheme calculation.
2. **J-coupled / reference implementation vs. brute-force m-scheme formula** (`Mscheme_Test_comm*`, UnitTest.cc:1774-8540; ~6000 lines of nested m-loops): each computes the commutator element by element in m-scheme with the textbook uncoupled formula (documented in a comment above each function, e.g. `Z0 = sum_ab n_a (1-n_b)(X_ab Y_ba - Y_ab X_ba)`), then compares with the J-scheme result extracted via `GetMschemeMatrixElement_*` (sum of squared errors < 1e-6). These are not wired into the C++ aggregate tests (the calls in `TestCommutators_Tensor` are commented out at UnitTest.cc:~706-711); they are only reachable manually from Python. Several are marked "// test not yet implemented" at the function header (comm332_*, comm233_pp_hh, comm333_*; UnitTest.cc:3663, 3815, 4423, 4853, 5010, 7679) although bodies exist; those 3-body-3-body terms are too slow in m-scheme.
3. Aggregators:
   - `TestCommutators(X,Y)` (617): runs layer-1 tests for every term whose entry in `Commutator::comm_term_on` is true (global `std::map<string,bool>` toggled by `TurnOnTerm/TurnOffTerm`). Calls `PreCalculateSixJ`.
   - `TestCommutators_Tensor(X,Y)` (693): scalar-tensor terms (rank-dependent).
   - `TestCommutators_IsospinChanging()` (774) / `TestCommutators_ParityChanging()` (848): hard-code X scalar anti-Hermitian, Y with dTz=1 / odd parity; both use particle rank 3. `TestCommutators3(X, Y, skiplist)` (925) marked "THIS IS DEPRECATED. I SHOULD PROBABLY JUST REMOVE IT" (UnitTest.cc:924).
   - `TestNormalOrdering(Op)` (390): independent check of `Operator::DoNormalOrdering`: evaluates the 0b piece, 1b piece and 2b piece of the normal ordered operator via explicit m-scheme sums over reference (hole) states with occupations, compares to J-scheme result, tolerance 1e-6. Handles 1/2/3-body, tensor, parity- and Tz-changing operators (only the ones the Python script feeds it).
   - `TestFactorizedDoubleCommutators(eta, H)` (6021): compares the double commutator [eta,[eta,H]]_2b obtained via the intermediate 3-body route (`comm223ss` then `comm231ss`, `comm232ss`) with `FactorizedDoubleCommutator::comm223_231/232` (no 3-body intermediate; memory-efficient). Tolerance `Norm() < 1e-6`.
   - `TestPerturbativeTriples()` (6122): builds random H (norm scaled to 1e3) and Omega (norm 1, plus 10x denominators-diagonal `Hod`), calls `IMSRGSolver::CalculatePerturbativeTriples()` and checks it against an explicit evaluation of eq. (26)-(27) of PRC 110, 044316 (2024) using `comm223ss` and `comm330ss`; tolerance 1e-6. Requires `Generator.Update` white/Moller_Plesset and `BCH::SetBCHSkipiEq1`.
   - `SanityCheck()` (6182): CG(1/2,1/2;1/2,-1/2|0,0) == sqrt(1/2), `GetAZfromString("Pb208")` == (208,82), `mix_A4.5_Z2.5` parsing, `Trel_Op` norm > 1e-8, `|| [Trel,Trel] ||` < 1e-8 (via full `Commutator::Commutator`).
   - Other: `Test3BodyAntisymmetry/Hermiticity`, `Test3BGenerator(H,Hdenom)` (8542), `TestMBPT4(H)` (8630), `TestRPAEffectiveCharge`.
   Interesting note: `Test_against_ref_impl` has its **own** compile-time dependence on `Commutator::use_imsrg3` for the output particle rank.
- There are **no tests of the operator library** (`OperatorFromString`, E2/M1/Rp2/M0nu/DM/Trel against analytical results), no tests of IMSRGSolver end-to-end flow (energy vs. known reference), HF, I/O round trips, ModelSpace channels, angular momentum recoupling coefficients other than one CG and the commutator tests themselves, or of the Python API surface beyond the above scripts. Only physics-level regression would be end-to-end runs the authors do by hand (docs `benchmarking.dox` lists a NCSM comparison with a table but no automated test).

### 3.3 Python test scripts
| Script | Model space | What it does | Needs input files? |
|---|---|---|---|
| `tests/UnitTest.py` | emax=2, He4 | `ut.SanityCheck()`, asserts True | No |
| `test/UnitTestCommutators2.py` | emax=2, `He6`/`He6` (reference/valence strings; no 3b) | RandomOp X antiHermitian, Y Hermitian, J=0,dTz=0,parity even, rank 2; `MakeNotReduced`; `TestCommutators(X,Y)` | No |
| `test/UnitTestCommutators3.py` | emax=2 then emax=1, He6 | rank-3 random X,Y; turns off all 1-2-body terms and turns on the 3-body ones (list 1: 330, 131 (sic, nonexistent), 231, 331, 132, 232, 223, 133; list 2: 332_*, 233_*, 333_*); `SetUseIMSRG3(True)`, `SetUseIMSRG3N7(True)`; `TestCommutators(X,Y)` | No |
| `test/UnitTestFactorized.py` | emax=2, He6 | Two cases (eta Hermitian-sign combos) `TestFactorizedDoubleCommutators(A,B)` | No |
| `test/UnitTestNormalOrdering.py` | emax=2, He6, then C12 `p-shell` | `TestNormalOrdering` for scalar rank-3, J=1 rank-2, parity-changing, charge-changing; then compares `UndoNormalOrdering + DoNormalOrderingCore` vs `ReNormalOrderCore` on C12/p-shell (prints norm of difference, not asserted, and **not included in `passed`**) | No |
| `test/UnitTestTensor.py` | emax=1, He6 | Scalar X (J=0) vs rank-2 tensor Y (J=2), both rank-3: `TestCommutators_Tensor(X,Y)`; prints profiler times | No |
| `test/UnitTestTriples.py` | emax=3, C14 | `TestPerturbativeTriples()` | No |
All tests are self-contained (random operators + built-in model spaces `He4/He6/C12/C14` generated from reference strings). **None needs an interaction file.** They need the compiled `pyIMSRG.so` on `PYTHONPATH`.
Observations: variable named `py` (parity) shadows nothing since module is `pyIMSRG`, but is confusing; the NormalOrdering script's last check is informational only.

### 3.4 How tests are registered
- Root `CMakeLists.txt` (lines 53-85, only when python bindings ON): `enable_testing()` and tests `UnitTest`, `UnitTestCommutators2`, `UnitTestCommutators3`, `UnitTestFactorized`, `UnitTestTriples`, `UnitTestTensor` using `${PYTHON_EXECUTABLE}` (never defined by that CMake - pybind11's `PYTHON_EXECUTABLE` var may or may not be set) with `WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}`; `UnitTestNormalOrdering` is **missing** here.
- `src/CMakeLists.txt` (tail) separately `include(CTest)` and registers `Commutators2, Commutators3, Factorized, Triples, Tensor, NormalOrdering` with `python3 ../test/...py` (relative path, so only works from build dir = `../build`), `--verbose` ctest args. The two top-level CMakeLists thus define overlapping/different test sets.
- `src/MakefileRagnar` targets `test`, `test-factorized`, `test-triples`, `test-tensor` run inline `python -c` one-liners calling the same UnitTest methods (e.g. `ut.TestCommutators()` with no args - the signature was later changed to take `(X, Y)`, so the Makefile `test` target is stale and would fail).

---------------------------------------------------------------------------------------------------------------------

## 4. Build system

### 4.1 Layout: there are two CMake roots (confusing)
1. `CMakeLists.txt` (85 lines, repo root): `cmake_minimum_required(3.22)`; `set(CMAKE_BUILD_TYPE Release)` unconditionally; **C++14** (`CMAKE_CXX_STANDARD 14`); options `IMSRG_BUILD_PYTHON_BINDINGS` (ON) and `USE_ARMADILLO_EXTERN` (ON); `project(imsrg++ VERSION 0.1.0)`; `find_package(OpenMP REQUIRED)`, `BLA_VENDOR OpenBLAS` + `find_package(BLAS REQUIRED)`, `GSL REQUIRED`, `ZLIB REQUIRED`, `Boost REQUIRED` (header-only use: no COMPONENTS), optional `Armadillo`. Then `add_subdirectory(src)` and `add_subdirectory(extern)`, which does not exist at the root (extern is `src/extern`) -> the root CMakeLists would fail, and src/CMakeLists.txt itself calls `pybind11_add_module(pyIMSRG ...)` so the root's second `pybind11_add_module` would double-define it. The root file is stale/inconsistent with the actual one. `build/INSTALL.txt` says: `cd build; cmake ../src; make -j; make test`, i.e. the real root is `src/CMakeLists.txt` (so `IMSRGPLUSPLUS_MAIN_PROJECT` logic).
2. `src/CMakeLists.txt` (600 lines, ~half comments): the effective build. `cmake_minimum_required(3.18)`, `project(imsrg++ VERSION 0.1.0)`, `include(local_settings.cmake OPTIONAL)` for machine-specific overrides, `find_package(OpenMP, BLAS, GSL, Boost, ZLIB REQUIRED)`, `add_compile_options(-frounding-math)` (needed for the hand-written `Moshinsky`/factorial code? unknown, flag is global), C++14, PIC if python on. It overrides `CMAKE_CURRENT_BINARY_DIR` to `"../build"` and hard-codes `INSTDIR = $ENV{HOME}`.

### 4.2 Targets (all as small OBJECT libraries aggregated into one shared lib)
- Object libs in `src/CMakeLists.txt`: `AngMom` (AngMom.cc, AngMomCache.cc), `IMSRGProfiler`, `ModelSpace` (ModelSpace.cc, TwoBodyChannel.cc, ThreeBodyChannel.cc), `PhysicalConstants` (INTERFACE), `TwoBodyME`, `ThreeLegME`, `Parameters` (INTERFACE, empty), `Operator`, `Version` (generated `version.cc`), `ReadWrite` (defines `NO_HDF5` publicly), `BCH`, `IMSRGSolver` (+IMSRGSolverPV), `Jacobi3BME`, `HartreeFock`, `Generator` (+GeneratorPV), `HFMBPT`, `RPA`, `UnitTest`. Subdirectory object libs: `Commutator` (Commutator.cc, DaggerCommutators, TensorCommutators, IMSRG3Commutators, FactorizedDoubleCommutator; note `Commutator232.cc/hh` are present on disk but **not** listed), `ReferenceImplementations` (ReferenceImplementations.cc, HFMBPT4, IMSRG2/3(Tensor)Commutators, FactorizedNestedCommutators), `ThreeBodyME` (ThreeBodyME + Storage, _pn, _no2b, _mono, _iso; links `IMSRGBoostZip` + `HalfPrecision`), `imsrg_util` (see 1; includes Pwd, M0nu, DarkMatterNREFT, Atomic; Helicity commented out).
- Final: `IMSRG` SHARED library (empty source list, links all the object libs), `imsrg++` executable (`imsrg++.cc`, 1715 lines), `pyIMSRG` pybind11 module (links `IMSRG`). Link graph oddities: `ReferenceImplementations` is part of the production shared lib; `UnitTest` (8700 lines) is part of the production shared lib; `Generator` links `imsrg_util` while `imsrg_util` links `HartreeFock` etc. (cyclic-looking dependencies resolved only because they are OBJECT libs aggregated into one SO). `imsrg_util` depends on `Pwd` (momentum-space PWD) for M0nu.
- Install: `install(TARGETS imsrg++ -> ${HOME}/bin, IMSRG -> ${HOME}/lib)` plus all `*.hh` (flat copy of headers from `src/` only, not subdirectories) to `${HOME}/include`. No exported CMake package, no `install(EXPORT)`, no versioning, `pyIMSRG` is not installed.
- No compile-feature or warning flags in CMake (no `-Wall`), optimisation defaults to CMake `Release` (`-O3 -DNDEBUG`); `-march=native` is absent. Makefile uses `-O3 -std=c++17 -fopenmp -fPIC`.

### 4.3 Dependencies
| Dep | How obtained | Used for |
|---|---|---|
| Armadillo | **vendored header-only** `src/extern/armadillo` (version 15.6.0 "Medium Roast Cortado" per `armadillo_bits/arma_version.hpp:24-27`; CMake INTERFACE target links `BLAS::BLAS`; `BLA_VENDOR OpenBLAS`; LAPACK linked via OpenBLAS/`-llapack` in Makefile). `ARMA_NO_DEBUG` is commented out in `ModelSpace.hh:23`. | Dense matrices/vectors, eigen solves in HF (`eig_sym`), SVD, matrix products in Pandya-transformed commutators. 17 headers include `<armadillo>` |
| pybind11 | vendored `src/extern/pybind11` (2.13.0.dev1), `add_subdirectory` when Python bindings ON | Python module |
| half | vendored `src/extern/half` = Christian Rau's `half` library **2.2.1** (2025-03-02) | `half_float::half` (`ME_half_type`) as storage type for the 3-body "no2b" and "mono" matrix elements (ThreeBodyStorage_no2b.hh:19-20, _mono.hh:22). 16-bit floats for 3N files (parameter `no2b_precision` single/half, Parameters.hh:63) |
| Boost | **system** install `find_package(Boost REQUIRED)` (header-only). `src/extern/boost_src` holds only 2 compiled sources `gzip.cpp` and `zlib.cpp` (Boost.Iostreams, built as `IMSRGBoostZip`, linked to `Boost::boost` + `ZLIB::ZLIB`) so no compiled boost libs need to be installed. | (a) `boost/numeric/odeint.hpp` in IMSRGSolver.cc:13 and IMSRGSolverPV.cc:13: `runge_kutta_dopri5` with `make_controlled` and `integrate_adaptive` over `std::deque<Operator>` using a custom vector-space algebra (IMSRGSolver.cc:759-835, 939...). (b) `boost/iostreams/{copy,filter/gzip,filtering_stream,filtering_streambuf}.hpp`: gzip read of `.me2j.gz`/`.me3j.gz` and gzip write (`gzip_compressor`, `best_speed`) in ReadWrite.cc (4 includes, 9 `gzip_decompressor`, 2 `gzip_compressor` uses) and in `ThreeBodyStorage_mono.hh`, `ThreeBodyStorage_no2b.hh` (4 includes each, gz-compressed binary 3N). (c) `boost::math::tgamma_ratio` appears only in a comment (imsrg_util.cc:2343). |
| GSL | system `find_package(GSL REQUIRED)`, `GSL::gsl` | See 4.4 |
| ZLIB | system | via boost iostreams gzip |
| OpenMP | `OpenMP::OpenMP_CXX`; macOS Makefile uses `-Xpreprocessor -fopenmp -lomp` | 400+ pragmas; `omp_get_wtime` for profiling everywhere; `omp critical` x19; pragmas: `parallel for schedule(dynamic)` (238 w/ schedule), `collapse` (18), `reduction` (4), `barrier` (4) |
| BLAS/LAPACK | `BLA_VENDOR OpenBLAS` (CMake), `-lopenblas -llapack` (Makefile), or MKL on some clusters (`oak`, `mox`, `bebop`, `crc` host branches in Makefile) | Behind Armadillo only (no direct cblas/lapack calls in the source) |
| HDF5 | optional, off by default (`NO_HDF5` defined for ReadWrite; `-DNO_HDF5` unless `HDF5=on` in Makefile). 22 mentions in Parameters.hh/ReadWrite.*/pyIMSRG (`Read3bodyHDF5`). | Reading Darmstadt-format HDF5 3N files. The Dockerfile installs `libhdf5-dev` anyway |
| Python | `python3`, `python3-numpy` (Docker only; numpy not used by the bindings) | tests, bindings |

### 4.4 GSL usage across the whole `src/` tree (excluding `src/extern`), grep counts
Headers included by file: AngMom.cc (`gsl_sf_coupling.h`), Operator.cc (`gsl_math.h`, effectively only constants like `M_SQRT2`/`M_PI`), HartreeFock.cc (`gsl_sf_laguerre.h`), imsrg_util.hh/M0nu.hh (`gsl_math.h`, `gsl_integration.h`), DarkMatterNREFT.cc (`gsl_sf_gamma.h`, `gsl_sf_hyperg.h`, `gsl_sf_coupling.h`), Pwd.cc/Helicity.cc (`gsl_sf_legendre.h`, `gsl_integration.h`), imsrg_util.cc/Atomic.cc/M0nu.cc.
Files with GSL calls (hit counts for the pattern `gsl_sf_fact|gamma|coupling|doublefact|laguerre|hyperg|bessel|legendre|gsl_integration`): imsrg_util.cc 37, DarkMatterNREFT.cc 55, M0nu.cc 22, Helicity.cc 18, Pwd.cc 16, AngMom.cc 9, M0nu.hh 3, Atomic.cc 4, HartreeFock.cc 3, Pwd.hh 9, Helicity.hh 3.
Function counts (token occurrences, includes comments):
| GSL symbol | count | what it is used for |
|---|---|---|
| `gsl_sf_fact` | 68 | factorial (n!) as double (Talmi/radial integrals, HO norm, DM operators) |
| `gsl_sf_gamma` | 31 | Gamma(x) (half-integer arguments in HO normalisation; partially replaced by `std::tgamma` in some places) |
| `gsl_integration_glfixed_table` (+`_alloc` 6, `_free` 8, `_point` 13) | 31 (+27) | **Gauss-Legendre fixed-order quadrature** (momentum/angle meshes in PWD, M0nu, Helicity) - replace by an in-house Golub-Welsch or `std`-based routine |
| `gsl_sf_coupling_6j` | 19 | Wigner 6j (AngMom::SixJ, 9j recoupling in DM operators, Atomic) |
| `gsl_sf_coupling_3j` | 11 | Wigner 3j (AngMom::ThreeJ, used in CG and in all multipole operators) |
| `gsl_sf_doublefact` | 10 | (2n+1)!! in HO normalisation |
| `gsl_sf_laguerre_n` | 9 (+ `gsl_sf_laguerre` header 4) | generalised Laguerre L_n^{(a)}(x) for HO radial functions (imsrg_util.cc, HartreeFock.cc:1275, DM operators) |
| `gsl_sf_coupling_9j` | 6 | Wigner 9j (`AngMom::NineJ`; Atomic hD operator; reduced-me recoupling) |
| `gsl_sf_hyperg_1F1` | 6 | Kummer 1F1 for HO Bessel/plane-wave integrals in DM operators (`jho`, `PhiS`) |
| `gsl_sf_bessel_j0` | 3 | spherical Bessel j0 in `FormfactorAtQ`, `FourierBesselCoeff` |
| `gsl_sf_legendre_Pl` | 2 (+ header) | Legendre P_l in PWD angular integrals |
| `gsl_integration_qagiu`, `gsl_integration_workspace(_alloc/_free)` | 1 / 1 / 1 | adaptive semi-infinite integral in `FrequencyConversionCoeff` |
| `gsl_function` | 2 | wrapper for the above |
| `gsl_math.h` | 6 includes | constants (`M_PI`, `M_SQRT2` ...); `GSL_...` macros not used otherwise |
Key point: the hot-path GSL functions are the Wigner symbols, called from `AngMom::{ThreeJ,SixJ,NineJ}` (AngMom.cc:110-151) and cached (`AngMomCache`, `ModelSpace::PreCalculateSixJ/NineJ`). `AngMom.cc` also has its own `factorial` lists (`FillFactorialLists`) and commented-out `gsl_sf_fact` code (line 71). Replacement checklist for the rewrite: Wigner 3j/6j/9j (exact integer-arg versions, exact for large j, thread-safe caches), factorials/gamma/double-factorials (log-gamma table), generalised Laguerre, 1F1 (small, can use finite sum for HO-type integrals), spherical Bessel j0 (`std::sph_bessel(0,x)`, or sin x / x), Legendre (`std::legendre`), Gauss-Legendre/Gauss-Laguerre nodes, adaptive quad (qagiu; a simple replacement is Gauss-Laguerre/Hermite since integrand is Gaussian-damped).

### 4.5 Boost usage summary (whole src tree, excluding extern)
Includes: `boost/iostreams/copy.hpp` x3, `boost/iostreams/filter/gzip.hpp` x3, `boost/iostreams/filtering_stream.hpp` x3, `boost/iostreams/filtering_streambuf.hpp` x3, `boost/numeric/odeint.hpp` x2. Files: ReadWrite.cc, ThreeBodyStorage_mono.hh/.cc, ThreeBodyStorage_no2b.hh/.cc, ThreeBodyStorage.hh (via these), IMSRGSolver.cc, IMSRGSolverPV.cc. Symbols: `boost::iostreams::filtering_istream` (9), `gzip_decompressor` (9), `filtering_ostream` (1), `gzip_compressor` (2), `gzip::best_speed` (1), `boost::numeric::odeint::{runge_kutta_dopri5, vector_space_algebra, integrate_adaptive, make_controlled}` (4 namespace uses). The Makefile flags `-DNO_ODE` (comment only: "compiles without boost/ode") and `-DOLD_BOOST=1` (cronos host) are not referenced anywhere in code except the Makefile. Replacement: a small adaptive Dormand-Prince 5(4) integrator operating on a "flow state" vector-space type (only `+`, `*`, abs-norm needed), and zlib directly (or `std::filesystem`+zlib gz streams).

### 4.6 Version string
`src/git_version.cmake` is run as a custom command (`ADD_CUSTOM_COMMAND(OUTPUT version.cc _version.cc COMMAND cmake -P git_version.cmake)`, src/CMakeLists.txt) producing `../build/version.cc` with `std::string version::BuildVersion()` returning `"<branch>_<7-char hash>[+]"` where `+` = dirty (uses `git log`, `git diff --quiet`, `git describe --exact-match --tags`, `git rev-parse --abbrev-ref HEAD`; "N/A" if no git). Matches the idea of the Makefile's `BUILDVERSION` (`git branch -v | awk`) passed as `-DBUILDVERSION` (which nothing reads: `version.hh` only declares `std::string BuildVersion();`, version.hh:1-6). Written only if content changed (avoids rebuilds). Exposed to Python as `pyIMSRG.BuildVersion`; printed by the driver and `SanityCheck`. `project(imsrg++ VERSION 0.1.0)` is never used for anything.

### 4.7 Hand-written Makefile (`src/MakefileRagnar`, 313 lines)
Original author's build; NOT used by CMake and **stale**: `EXTERN = ../extern` but the tree has `src/extern`; OBJ list lacks `Atomic.o`, `ReferenceImplementations/*`, `Commutator232`, source lives in subdirectories now (so the file will not build). Notable ideas that CMake lost: OS and host detection (`jureca`, `cougar`, `cronos`, `oak`, `cedar`, `mox`, `bebop`, `crc`) with MKL link lines (`-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp`), macOS flags (`-Xpreprocessor -fopenmp`, `-lomp`, `-DNO_x86`, `-undefined dynamic_lookup`), `DEBUG=on`, `HDF5=on`, `PYTHON=off`. Defines `-DNO_x86` on macOS to avoid `x86intrin.h` (used for `_mm_prefetch`/popcount-style intrinsics somewhere in ModelSpace/commutators). Targets: `libIMSRG.so`, `imsrg++`, `pyIMSRG.so`, `install` (symlinks into `$HOME/{lib,bin,include}`), `test*`, `clean`.

### 4.8 Containers, packaging, CI
- `Dockerfile` (43 lines): multi-stage; build stage `ubuntu:24.04` installing `build-essential, libomp-dev, libopenblas-dev, liblapack-dev, libgsl-dev, libhdf5-dev, zlib1g-dev, libboost-all-dev, python3, python3-numpy, cmake`, then `RUN cmake .` and `RUN make` (note: no `COPY` of the source and it configures with the root CMakeLists that is itself broken); the release stage installs runtime libs (openblas, gsl, python3-numpy/sympy/pandas) and `COPY pyIMSRG.cpython-312-x86_64-linux-gnu.so /usr/local/lib/` from the **build context**, not `--from=build`. `.dockerignore` excludes `build, CMakeFiles, doc, extern, src, work, .gitmodules, CITATION.cff, cmake*`.
- `imsrg.def` (Singularity/Apptainer, author Antoine Belley): two stage `ubuntu:latest` compile (apt dependencies as above) with `cmake CMakeLists.txt; make`, final stage copies `/imsrg/pyIMSRG.cpython-*-x86_64-linux-gnu.so /lib/` and sets `PYTHONPATH`. Same fragility.
- `.gitmodules` is **empty** (0 lines; all third-party code is vendored by copy). `DartConfiguration.tcl` is a CMake/CTest-generated file committed by accident (contains an absolute path `/Users/antoinebelley/Documents/TRIUMF/imsrg`, host name, no dashboard site) - no CDash.
- **No CI**: no `.github/`, no `.travis.yml`, no `*.yml`/`*.yaml` outside vendored pybind11 (`find` returned none in the repo's own files). Tests are run manually via `make test` / `ctest`.
- `README.md`: GPL licence note (Armadillo MPL, Boost licence), credits (Stroberg; Miyagi for HFMBPT/ThreeBodyMENO2B/Pwd; Charlie Payne and Antoine Belley for double beta; Johannes Simonis; Bingcheng He for IMSRG(3f2) and tensor IMSRG(3)). `CITATION.cff` present.

---------------------------------------------------------------------------------------------------------------------

## 5. Documentation and CI state

- Documentation: Doxygen. `doc/srg.dox` (2304 lines) is actually the **Doxyfile** (header "Doxyfile 1.8.6", `PROJECT_NAME = "IM-SRG++"`) despite its extension; `doc/installation.dox` (28 lines) is a stale 3-step "git clone; cd IMSRG/src; make; make install" page (old repo URL `ragnarstroberg/imsrg`); `doc/benchmarking.dox` (41 lines) has NCSM comparison figures referenced as `Benchmark_NCSM.png`/`O22NO.png` that are not in the tree, empty "6Li" section and a placeholder "More text." / "Performance". There is no user manual, no description of parameters (the only parameter documentation is the comment text in `src/Parameters.hh`), no description of operator names (only inline comments in `OperatorFromString`), and no doxygen invocation from CMake. Doxygen-style `///` comments exist in source for many functions (e.g. `Rp2_corrected_Op` with LaTeX formulae, M0nu operators with links to external Google-Drive PDFs `drive.google.com/...` for derivations - unreachable in future).
- CI: none (see 4.8). Tests need a build of the Python module plus an environment; test results are not automated.
- Code comments carry domain knowledge (formulae and references: Suhonen 6.23/6.24, PRC97 054306, PRC 110 044316) that should be preserved in the rewrite's documentation.

---------------------------------------------------------------------------------------------------------------------

## 6. Pitfalls and observations for the rewrite

### Operator library
1. **String-keyed factory with parameters embedded in the name** (underscore-delimited, unvalidated, case-sensitive, with ambiguities: e.g. the "Rp2" exact match precedes "Rp2Z"/"Rp2AZ" token matching; operator names such as `E2` vs `E2int`; `OneOcc` documented as `OneOccp_1p3` in a comment but implemented as `OneOcc_<orb>`; `AllOcc` comment says `AllOccpp3`). The rewrite should use a typed registry with parsed specs (name + typed params) while still accepting the legacy strings for compatibility with existing input decks (they appear in user scripts and the driver's `Operators` list).
2. Operator "rank" semantics are encoded in a constructor with four ints; the second is |dTz| (not "isospin rank"). Reduced/non-reduced convention is implicit: tensor operators and parity/Tz changing operators are produced **reduced** (`ML.MakeReduced()` etc.) while scalars are not; `Reduce`/`UnReduce` free helpers exist; tests must call `MakeNotReduced` before comparing. Make the convention explicit in the new API.
3. Many constructors print to `std::cout` (118 statements in imsrg_util.cc) - need a logger.
4. **Duplicate and copy-paste code**: (a) `Isospin2_Op` is defined twice (one commented-out in `/* ... */`, imsrg_util.cc:1754 and 1790); (b) HO radial wavefunction formula repeated in `HartreeFock::GetRadialWF`; (c) `TalmiB` exists in `AngMom`, `imsrg_util`, and declared in `M0nu`; `HO_Radial_psi` exists in `imsrg_util` and `M0nu.hh`; `phase(int)` in `M0nu` and `Helicity` and `AngMom`; `asNorm` (M0nu.hh) vs `cpNorm` (imsrg_util.hh), `Fermi`/`GT`/`Tensor` M0nu functions nearly identical with different form factors, and the `rhocentralp`/`rhocentraln` branches, `rhop`/`rhon`, `FFp`/`FFn` pairs are copy-pasted; the three Minnesota/Gaussian/GaussianCSB potentials repeat the same 40 lines; `M0nu::GamowTeller_R`, `DGT_R`, `DGT_R_SurfaceLocalization` share bodies.
5. **Dead / declared-but-undefined**: `M0nu_TBME_Op`, `CPrbmeGen` (imsrg_util.hh, "put in by CP, still in development"; no definition found), `Helicity.cc` not compiled, `M0nu::TalmiB` declared (no definition in M0nu.cc?), `SerberTypePotential` ("not yet implemented" comment but body exists and is exposed to Python), `SurfaceDeltaInteraction` (same), `Single_Ref_1B_Density_Matrix`/`Get_Charge_Density` ("This doesn't work"), `M0nu_contact_Op` (replaced by `M0nu::Contact`), `GetOccupations(hf, imsrgsolver)` deleted "BECAUSE IT CAUSES A CIRCULAR DEPENDENCY WITH IMSRGSolver" (imsrg_util.cc:625-650), lots of commented-out blocks (e.g. `CommutatorTest`). `M0nu` argument `src` is unused everywhere.
6. Hard-coded numerical parameters: FBC R=8 fm; 10-point central density; Gauss-Laguerre 200 points; M0nu meshes 100/500/25 fm^-1; `qagiu` tolerance 1e-7 with limit 1000; `n_radial_points=200, Rmax=20` in `WriteSPWaveFunctions`; WS/HOtrap `ngrid`; `npoints` options limited to tabulated sets. No accuracy control. Expose these as parameters with convergence tests.
7. Physical constants: nucleon mass `M_NUCLEON` is a single average value 938.9185 (commented) in HBARC units; `PROTON_SPIN_G=5.586`, `NEUTRON_SPIN_G=-3.826`; Magnetic operator uses `3.1415926` literal instead of `M_PI`; Anapole/axial charge have debug printing; some operator conventions (b = sqrt(hbarc^2/(m*hw))) are inline repeated. `AxialCharge_Op` defines `oscillator_b = HBARC*HBARC/sqrt(M_NUCLEON*hw)` - dimensionally suspicious (compare with the b formula in other places; looks like a precedence bug, though unused in the result if `prefactor` doesn't use it). Worth a careful validation before porting ("Untested" operators: E2int, RadialOverlap, L2rel, QdotQ*, VCoul, AxialCharge).
8. DM-NREFT: unknown operator name leaves an empty (default-constructed) Operator silently; `Omega` warns it is neither symmetrised nor antisymmetrised.
9. Operators depend on global-ish `ModelSpace` state (`GetTargetMass/Z`, `GetHbarOmega`, `GetE2max`, `PreCalculateMoshinsky` mutates the modelspace cache; `OneBodyChannels`, `proton_orbits`).
10. The CM-corrected radius operators implicitly assume an HO basis with the model space hbar-omega (valid for the HO basis; after HF transformation the operator is transformed by `HartreeFock::TransformToHFBasis` (imsrg++.cc:817-830)). The new design needs an explicit "basis-bound" notion for operators whose matrix elements are only defined in the HO basis (`rho`, `FF`, `FBC`, `VWS`, `Coulomb`, `M0nu`, `DMNREFT`).
11. Performance: Coulomb, `QdotQ`, Minnesota/Gaussian, M0nu, TCM/Trel two-body operators are O(channels x kets^2) with `omp parallel for schedule(dynamic,1)` and inner `omp critical` in places (19 `critical` sections across the tree); good candidates for a Kokkos-style `parallel_for` over channels. M0nu precomputes momentum-space integrals in an `unordered_map<uint64_t,double>` keyed by hashed (n,l,n',l',S,J) (`IntHash`, `PreCalculateMomentumSpaceIntegrals`) - replace with dense arrays.

### Bindings
12. `arma::mat` as an opaque `ArmaMat` is the biggest usability gap: no numpy interoperability (`numpy.h` included but unused). New bindings should expose one-body matrices and two-body channel blocks as numpy views (zero-copy, column-major to match Fortran-order) with proper lifetime (`py::keep_alive`).
13. Large objects returned by value and containers (`holes`, etc.) copied on each access; no GIL release for long calls; no exception translation policy; `Operator.TwoBody` exposes mutable internal `TwoBodyME`.
14. API surface is tied to internal commutator term names (~150 functions); only 6 tests exercise 6 methods.
15. `TwoBodyChannel_CC` binds `TwoBodyChannel::J/parity/Tz` pointers-to-members of a different class (copy/paste bug).
16. Python scripts embed machine-specific absolute paths and e-mail address.

### Tests
17. Tests live inside the production library and use global state (`Commutator::comm_term_on` map, static `random_seed`, `use_imsrg3` flags set by Python): tests are order-dependent and cannot be run in parallel in one process. A new design should isolate these.
18. The 3-body random-operator generator bug described in 3.1 (shadowed `random_me`); note it means the 3-body commutator tests might not have exercised fully general 3-body operators.
19. `TestCommutators3` marked deprecated; the Makefile `test*` targets are stale; `UnitTestCommutators3.py` refers to a nonexistent term `comm131ss`; `TestNormalOrdering` Python script does not include its last check in `passed`; the C++ aggregator test bodies are heavily duplicated (4 near-identical 70-line functions).
20. ~6000 lines of near-duplicate nested-loop m-scheme code in `UnitTest.cc` (each of ~50 terms) - high cost; the rewrite should provide a *generic m-scheme operator algebra* (an m-scheme "reference" Operator class with second-quantised commutator evaluation by normal-ordering rules, i.e. a Wick/brute-force engine) and test J-coupled commutators against that, instead of one hand-written formula per term.
21. No physics-level regression (HO-limit energies, known HF energies, E2/radii vs analytic), no tests for the operator catalog, I/O, HF, ODE solver, nor MPI. `SanityCheck` is the only fast smoke test.
22. Tolerance 1e-6 absolute with norms of random O(1) numbers; with larger emax this can be loose/tight; a relative criterion with scaling by operator norm would be preferable.
23. Tests require `He4/He6/C12/C14` references and `emax <= 3` for speed; the reference implementations are slow ("THIS IS SO SLOW" comment at Test_comm233_phss).

### Build system
24. Two competing CMake roots (root is broken), CMake 3.18 vs 3.22, C++14 in CMake vs C++17 in the Makefile; the code appears to compile as C++14 (check: structured bindings/inline variables? `std::optional`? not verified).
25. `CMAKE_CURRENT_BINARY_DIR` is overridden to `../build` (out-of-source assumption), `INSTDIR = $ENV{HOME}`, install of headers is flat, tests use relative paths `../test/*.py`, and `python3` rather than the found interpreter; no `find_package(Python)` / `pybind11_add_module` default install location.
26. `-frounding-math` globally: prevents some floating-point optimisations (and is incompatible with aggressive vectorisation of reductions); verify necessity (likely for exact conversions with `lround`/`ceil` in index arithmetic).
27. The shared library includes `UnitTest` and `ReferenceImplementations` (~10k+ lines), increasing binary size and compile time.
28. Vendored third-party: Armadillo 15.6.0 (6.7 MB), pybind11 2.13.0.dev1 (3.1 MB), half 2.2.1 (252 KB), Boost.Iostreams sources `gzip.cpp`, `zlib.cpp` (20 KB). Boost headers are NOT vendored (system boost required, headers only). For the rewrite: pin deps via FetchContent/CPM or vcpkg/conan and drop vendoring; no git submodules are used.
29. Platform scripts: cluster-specific logic (MKL vs OpenBLAS) lives only in the stale Makefile; CMake relies on `BLA_VENDOR OpenBLAS`. Docker/Singularity files are incomplete (no copy of source, wrong artifact origin) and untested.
30. TODO/FIXME-style comments: 50 occurrences of `TODO|FIXME|XXX|HACK` in `src` (excluding extern): concentrated in `Commutator/IMSRG3Commutators.cc` (11), `Jacobi3BME.cc` (4), `ModelSpace.cc` (4), `Commutator/TensorCommutators.cc` (4), `UnitTest.cc` (3), `Commutator/Commutator.cc` (3); in `imsrg_util` only one (the TDA/RPA comment: "TODO this is still not right in the comment. Work this out cleanly", imsrg_util.cc:3558). Plus "Untested" annotations on operators (15 in the operator library) and "not yet implemented" notes on 8+ tests.
31. `Parameters.hh` is the single runtime parameter table (string-keyed defaults, e.g. `no2b_precision`), used by `imsrg++.cc`; command line `key=value`. Part of the operator-related config: `Operators`, `OperatorsFromFile`, `OperatorsPT1/RPA/TDA`, `BetaCM`, `hwBetaCM`.

---------------------------------------------------------------------------------------------------------------------

## 7. Suggested design takeaways (for the rewrite team)

1. Make the operator library a set of pure, side-effect-free "builders" with a typed spec: `OperatorSpec{kind, params}`; keep a legacy string parser (`name_p1_p2`) as a front end. Group by family: radii/densities, multipoles (E/M/ISM/...), kinetic/CM, weak (Fermi/GT/axial), potentials (Coulomb/Minnesota/Gaussian/VQQ/WS/trap), double-beta (M0nu GT/F/T/contact, heavy, integrand, radial, DGT), DM-NREFT, dagger operators, atomic.
2. Radial/Talmi/Moshinsky/HO-wavefunction helpers and quadrature nodes should be a small, separately tested numerics module (GL, GLaguerre, log-factorials, 3j/6j/9j, 1F1, j0), replacing all 15 GSL entry points listed in 4.4 and the 2 Boost features in 4.5.
3. Bindings must offer numpy interchange for 0/1/2-body blocks and operator factory; keep the checklist in section 2.2 as the compatibility baseline (ModelSpace, Operator, TwoBodyME/ThreeBodyME accessors, ReadWrite, HartreeFock, HFMBPT, IMSRGSolver, Generator, Commutator toggles, RPA, BCH, UnitTest-like validation helpers, and the ~25 free functions).
4. Tests: reproduce the three-tier scheme (1) optimised vs reference implementations, (2) reference vs m-scheme brute force through a generic m-scheme engine, (3) normal-ordering m-scheme test; add operator-library, HF, flow, and I/O tests; make randomness explicit (seed argument), remove global flags, tolerances relative. Use ctest + a real unit test framework; provide CI.
5. Build: single CMake root, presets, FetchContent/CPM for Kokkos/Catch2/pybind11 (or nanobind), MPI optional, no `$HOME` install default, generated version header from git at configure time.
