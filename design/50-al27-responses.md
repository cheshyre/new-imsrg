# 50 — The ²⁷Al dark-matter response task (weekend task, from 2026-11-26)

Status: decided in the owner interview of 2026-11-26 (D-rows D61ff in `00`); implementation started
the same night (leads A, B, C below); the unsupervised session runs until **Monday 2026-11-30,
10:00 Knoxville time**. Sources: the provenance package `$IMSRG_WORK/inputs/al27/`
(`README_al27_provenance.md`, unpacked `al27_provenance/`) and the analysis of the fork
`reference/06-al27-fork-pipeline.md` (fork sources: `$IMSRG_REFERENCE_ROOT/imsrg-fork-79e9330/`).

## 1. The task (owner, 2026-11-26, verbatim)

> Your task is to reproduce these calculations using the updated code. Some of these operators come
> from shorthands (with additional normalizations) defined in my fork of the imsrg++ code
> (cheshyre/imsrg, bleeding_edge).
>
> You have the following reproduction goals:
> - Reproduce the output .snt matrix elements to at least 1e-4 using the given code. If it requires
>   undoing some modifications to the solver then implement these modifications.
> - Extend the calculations with computations of Tilde Phi' (the function is Phitp, from what I
>   remember) for J=1, 3, 5 (same as Sigma). These may require an overall normalization, look at the
>   other results for inspiration.
> - The fastest benchmarks are the J=0 and J=1 responses.
> - In the case that significant solver changes were required to reproduce the given results, rerun
>   the all the responses with the updated solver, still using the approach where the Magnus operator
>   is not split. Document the differences.
> - Finally rerun the full set of calculations with a split Magnus operator using a reasonable
>   default value for the splitting (typically 0.25 for the omega_norm_threshold is sufficient).
>
> You have the following technical goals:
> - We are running on Frontier so the calculations should use GPUs for expensive operations.
> - The IMSRG(2) solution will likely run most efficiently on 1 node, but the omega files should be
>   saved to disk.
> - The evaluation of the responses can then be done in parallel using the checkpointed omega files,
>   and this evaluation should be done at scale using many ranks (ideally more than 15,800). Do not
>   allocate extra ranks that you cannot use efficiently just to reach this number, but try to reach
>   it.
> - Implement a full checkpointing system for the IMSRG(2) solution by saving the omega files and the
>   flowing Hamiltonian + some information about the current state of the integration. Checkpoint
>   every 30 minutes.

Later correction (owner, 2026-11-26): "J=2,4 are the ones I need" for Tilde Phi' (Phi-tilde' has
parity (−1)^J, so only even J survive in the elastic J=5/2 response). Normalization: §4.

## 2. Calculations

Configuration: provenance README §3 (hw 16, A 27, ensemble reference Al27, sd shell, NAT with
`use_HF_reference_in_NAT`, HF/NAT emax 14 / e3max 24, IMSRG(2) at emax 10 / e2max 20, two-stage
decoupling atan then shell-model-atan, Epstein-Nesbet, smax 100 (200 in stage 2), dsmax 0.5,
ds_0 0.01, eta_criterion 1e-3). Inputs in `/lustre/orion/nph123/proj-shared/nuccor_interactions/me_matthias/MEs`:

| Hamiltonian | 2N file (file2e1max/e2max/lmax) | 3N NO2B file (file3e1max/e2max/e3max), single precision |
|---|---|---|
| N2LOsat | `NNLOsat/TwBME-HO_NN-only_N2LO_sat_bare_hw16.0_emax16_e2max32.me2j.gz` (16/32/16) | `NNLOsat/NO2B_ThBME_N2LOsat_3NFJmax15_IS_hw16.0_ms16_32_24.stream.bin` (16/32/24) |
| ΔN2LOGO(394) | `Delta_N2LOGO_394/TwBME-HO_NN-only_DN2LOGO394_bare_hw16.0_emax18_e2max36.me2j.gz` (18/36/18) | `Delta_N2LOGO_394/NO2B_ThBME_DN2LOGO394_JJmax15_IS_hw16.0_ms16_32_24.stream.bin` (16/32/24; owner: same content as the archive's `..._DNNLOgo_...` file) |

Variants (each: two flows, one per Hamiltonian, then the response farm):

1. **Reproduction**: unsplit Magnus (NSmagnus, omega_norm_max 50000) with the fork's behaviour
   (profile `compat = "imsrg_fork_79e9330"`, §5). Acceptance: `.snt` matrix elements equal the
   archived ones to 1e-4 absolute (they carry 7 significant digits), HF and flow numbers of README
   §3.3 (HF iterations, e1/e2/e3hf, E_HF, EMP2/EMP3, stage ends: steps, s, E0, norms).
2. **Updated solver, unsplit**: only if variant 1 needed significant solver changes (the compat
   profile is that): the same with the fixed default behaviour; differences documented per column.
3. **Split Magnus**, omega_norm_max 0.25, fixed default behaviour.

Operators per Hamiltonian: FM0/2/4, FPhipp0/2/4, FSigmap1/3/5, FSigmapp1/3/5, FDelta1/3/5, FPhitp2/4,
each p and n, at q = 0, 15, ..., 1050 MeV (71 values), plus TCM: 34 × 71 = 2414 operator evaluations
per Hamiltonian and variant, 4828 per variant, 14,484 for three variants. Fastest checks: the J=0
and J=1 operators. The shell-model step is out of scope (`.snt` only).

## 3. Capabilities to build (C1-C14) and where

| # | Capability | Status 2026-11-26 | Lead |
|---|---|---|---|
| C1 | Fork reference build (79e9330 + C++14 diff, + phip_fix for Phitp), harness for emax 2-4 goldens through `tools/reference` | missing | B (tonight) |
| C2 | NREFT operator library: DM_NREFT port (HO radial integrals in y=(qb/2)², M, Ms, Mg, Phip, Phipp, Phitp, Sigmap, Sigmapp, Delta), shorthands `F<family><L><p|n>_<q>` with the fork's prefactors | missing | B (tonight) |
| C3 | Tensor operators (M15): storage (imsrg++ gauge, D-row of M15-Q1), basis transform HO→HF/NAT, [scalar, tensor] IMSRG(2) commutators (111/121/122/221/222 st), BCH transform with scalar Omegas, normal ordering w.r.t. another reference (core), tensor tokyo writer | goldens only | A (tonight) |
| C4 | `use_HF_reference_in_NAT` (holes reset to HF orbitals + Gram-Schmidt, fork-only) | missing | C (tonight) |
| C5 | Truncation of basis and operators emax 14 → emax_imsrg 10 (`[imsrg.space]`) | missing (Planned key) | C (tonight) |
| C6 | Omega dump/load: native float64 binary AND ME2JP float32 (owner: both, compare in the farm) | missing | C (tonight) |
| C7 | Flow checkpoint/restart every 30 min wall time: Omega stack (finished Omegas written once), H(s), integrator state (s, ds, step, stage, generator, config hash) | missing | C (tonight) |
| C8 | Fork compat profile (§5) | missing | A (tensor NO, 9j) / C (NAT sort) / B (writer) |
| C9 | Readers and HF at production size (me2j emax 16/18, NO2B e3max 24, ~48 GB files, memory ~30-45 GB) | untested above emax 6 | weekend |
| C10 | `transform` task: farm of (Hamiltonian, variant, operator, q) over MPI ranks, one GCD each, static cost-weighted list, failure manifest; Omegas read once per node and shared through MPI shared memory | missing | weekend |
| C11 | GPU (M14): device path for the ladder/ph gemms — try both rocBLAS/hipBLAS behind `linalg` and Kokkos Kernels; prefer vendor only at >= 5x (owner) | missing | weekend |
| C12 | Flow layout on one node: measure 1 rank / 1 GCD and 8 ranks × GCD (M12b), use the faster | — | weekend |
| C13 | Production scripts (Slurm) for flows, checkpoints, farm sweeps; comparison scripts vs the archive | missing | weekend |
| C14 | Isospin-averaging test: one nontrivial example with and without averaging, documented | — | weekend |

## 4. Tilde Phi' normalization

Analysis of Fitzpatrick et al. 2013, JCAP02(2013)004 (owner's PDF; notes with equations and page
numbers: `$IMSRG_WORK/inputs/al27/analysis/phitp-normalization-notes.md`):

* The paper gives none of its operators (eq. 3.15) a √(4π) of its own; every response carries the
  same 4π/(2J_i+1) (eqs. 3.13/3.20, Table 1). The paper alone does not explain the fork's split
  (√(4π) on FM and FPhipp, 1 on FSigmap/FSigmapp/FDelta); it resembles the Hoferichter et al. 2016
  convention, which normalizes F^M and F^Φ'' so that F^M(0) = A.
* The fork's `Phitp = Phip + 0.5*Ms(J,J)` equals the paper's Φ̃' = Φ' + ½ Σ_J (sign, factor, index).
  Φ̃' is Hermitian (eq. D.1, same symmetry as M), normal parity, even multipoles only: J = 2, 4
  survive for the J = 5/2⁺ ground state.
* Φ̃' falls in the symmetry class of M and Φ'' (even, normal parity), and the prefactor-1 group is
  exactly the odd-multipole Σ', Σ'', Δ. It is the transverse partner of Φ''; the paper's
  long-wavelength ratio Φ̃'_2/Φ''_2 = √(3/2) holds between the columns only with a common prefactor.

**Choice (pending the owner's confirmation, 2026-11-26 20:00 EST):** `FPhitp<L><p|n>_<q> =
√(4π)·DM_NREFT::Phitp(L, q)`, L = 2, 4, Hermitian. q → 0: FPhitp2 → −(1/(2√5))[l⊗σ]_2 per
nucleon, FPhipp2 → −(1/√30)[l⊗σ]_2 (0d5/2 diagonal reduced element of FPhitp2: −1.4967).

## 5. Fork compatibility profile (variant 1)

Assumed (owner did not answer; golden rule 4, most reversible): one config key
`compat = "imsrg_fork_79e9330"` switches on every fork behaviour needed to match the archive, each a
named internal option with its own test against the fork reference build; the default is the fixed
behaviour. Behaviours (`reference/06` §4.1): tensor normal-ordering a==b / J_bra≠J_ket term omitted;
NAT orbitals re-sorted by |occupation| (variant 1 only); 9j skip threshold 1e-8 in the tensor
commutators; isospin averaging of pn 2B elements in the scalar operator writer (owner: always, all
variants, plus one documented example without it); 79e9330 writer details (7 significant digits,
1e-7 drop thresholds, tensor writer orderings, `A_for_kinetic_energy` header). FDelta and FPhitp are
flagged Hermitian (numerically confirmed; master's anti-Hermitian Delta flag treated as a bug).

## 6. Compute

Budget: 50,000 node-hours until the end of the weekend (Knoxville), standard partitions. Flows: one
node each (512 GB host for the 3N files and HF at emax 14; GPUs for the IMSRG(2) commutators).
Farm: one MPI rank per GCD (8 per node); one variant ≈ 4828 ranks ≈ 604 nodes, all three ≈ 14,500
ranks ≈ 1,810 nodes. Frontier batch limits: 184+ nodes → 12 h jobs.

### 6.1 Transform farm (C10, lead F, 2026-11-26)

`run.task = "transform"` (design/16 §2.11b; `src/imsrg/driver/transform/`), Slurm template
`tools/al27/farm_sweep.sh FLOW_DIR` (one job per flow directory = Hamiltonian x variant x Omega
format), work list `tools/al27/make_work_list.py` (34 x 71 names + TCM = 2415 lines).

* **Layout:** one MPI rank per GCD, 8 per node, `-c 7`, `--gpu-bind=closest` (the device path of
  lead E slots into `transform::TransformItem`, the per-item unit; today it runs on the host).
  One work item per rank: 2415 ranks = 302 nodes per (Hamiltonian, variant, format); variant 1 with
  both Omega formats and two Hamiltonians = 9660 ranks = 1208 nodes; all three variants (native only)
  = 14,490 ranks = 1812 nodes, all four sets with the ME2JP comparison 19,320 ranks = 2415 nodes.
* **Cost model (static, written down in `farm.h`):** cost(item) = num_omegas x W(L), W(L) = sum over
  the stored two-body blocks (ch_bra <= ch_ket allowed by rank L) of n_bra n_ket (n_bra + n_ket), the
  block-gemm multiply-adds of one [Omega, O]; the BCH terms per Omega are taken equal for all items.
  At emax 10: W(L)/W(0) = 1.97, 2.86, 3.61, 4.20, 4.57 for L = 1..5. Assignment: longest
  processing time first (deterministic, no communication); with one item per rank it is the identity
  up to order, with fewer ranks than items it balances to <= mean + one item.
* **Memory per rank (estimates from the channel counts, emax 14 -> 10, e2max 2 emax):** a scalar
  two-body part is 3.9 GB at emax 14 (TCM only; the NREFT operators are built one-body only, else a
  rank-5 tensor would take 20.5 GB at emax 14) and 0.29 GB at emax 10; a rank-L tensor at emax 10
  0.58 / 0.84 / 1.06 / 1.25 / 1.37 GB (L = 1..5). BCH keeps ~4 operators plus the tensor
  particle-hole scratch (AL-A-Q2: 10-20 GB at rank 5 until the Zbar symmetries are used): <= ~25 GB
  per rank, ~200 GB per node of 512 GB; TCM peaks at ~12 GB during the emax-14 basis change.
* **Omegas once per node:** node rank 0 reads them into an MPI shared window
  (`comm::Communicator::AllocateShared`), 0.29 GB per Omega at emax 10: the unsplit variants (2
  Omegas) 0.6 GB per node, the split variant (omega_norm_max 0.25; O(100) Omegas expected, to be
  measured) ~30 GB per node instead of 8x that. A rank materializes one Omega at a time.
* **Failure manifest:** `failed.txt` (name, reason) is written with every item first and rewritten
  at the end, so a killed job leaves a complete list; resubmit with it as the work list, or with the
  same list and `transform.skip_existing = true` (outputs are renamed into place, an existing file is
  complete).
* **Not done yet:** a timing of one item at emax 10 (needed for the walltime; the template uses 2 h),
  the device path, measuring the split-variant Omega count.

