# 02 — Commutator kernels (reference analysis of imsrg++)

Scope: `src/Commutator/*` and `src/ReferenceImplementations/*` of imsrg-devel (read-only reference).
Purpose: feature checklist, numerical-equivalence guide, and performance-porting guide for the
rewrite (C++20 / CMake / Kokkos / MPI / Python). File:line citations refer to
`imsrg-devel/src/`. Headers of `ModelSpace`, `Operator`, `TwoBodyME`, `ThreeBodyME` were read
only as far as the kernels depend on them.

Notation used below:

- `n` — number of single-particle m-scheme states (for the *naive* scaling quoted in code comments).
- `n_o` — number of j-shell orbits (`ModelSpace::Orbits`), `N2(ch)` — kets in a J-coupled 2B channel,
  `Ncc(ch)` — kets in a cross-coupled (Pandya) channel, `N3(ch)` — kets in a 3B channel.
- "1b channel" = orbits sharing (l or parity, j, tz). "ch" = 2B channel (J, π, Tz).
- `X,Y` inputs, `Z = [X,Y]` output. `hX = ±1` for (anti)Hermitian X.

---

## 0. File inventory and build status

| File | Lines | Status |
|---|---|---|
| `Commutator/Commutator.{hh,cc}` | 93 / 2173 | compiled; dispatch + all IMSRG(2) scalar kernels + scalar Pandya |
| `Commutator/IMSRG3Commutators.{hh,cc}` | 94 / 8144 | compiled; scalar IMSRG(3) kernels. ~890 lines fully commented out (2240–3139: `comm232ss_mh_optimized`, `_expand_*` wrappers); ~400 lines dead after early `return` in `comm232ss_srs_optimized` (906–1301) |
| `Commutator/TensorCommutators.{hh,cc}` | 64 / 3945 | compiled; scalar–tensor IMSRG(2) and IMSRG(3) |
| `Commutator/DaggerCommutators.{hh,cc}` | 51 / 749 | compiled; scalar–dagger (odd particle number) |
| `Commutator/FactorizedDoubleCommutator.{hh,cc}` | 79 / 3553 | compiled; IMSRG(3f2)-style `[Eta,[Eta,Gamma]_3b]_{1b,2b}` without 3B storage |
| `Commutator/Commutator232.{hh,cc}` | 379 / 1409 | **NOT compiled** (absent from `Commutator/CMakeLists.txt:1-8`; include commented at `Commutator.cc:3`; callers commented at `IMSRG3Commutators.cc:2254-2271`). Alternative flat-array comm232 (see §8.4) |
| `ReferenceImplementations/*` | ~17k | compiled into the library; slow, transparent J-coupled loops used by `UnitTest.cc` (§5.9) plus MP4 diagrams (`HFMBPT4.cc`) and a triples estimate |

Declared but never defined (link error if used): `prod111ss`, `prod112ss` (`Commutator.hh:87-88`),
`comm431sd`, `comm433sd_pphh` (`DaggerCommutators.hh:35,37`), `comm232ss_expand_full/_reduced`,
`comm232ss_mh_optimized`, `Hash_comm232_key` (`IMSRG3Commutators.hh:62-68`).

---

## 1. Data-model facts the kernels silently depend on

These are not commutator code, but every kernel bakes them in; a rewrite that changes them must
re-derive the kernel index arithmetic.

1. **Orbits**: `Orbit{n,l,j2,tz2,occ,occ_nat,cvq,index}` (`ModelSpace.hh:41-58`). `occ` may be
   fractional (ensemble references); `occ_nat` is the natural-orbital occupation used only for
   the `OccNat3Cut`; `cvq` = core/valence/qspace (0/1/2) used by `imsrg3_no_qqq`, `imsrg3_only_vvv`,
   `imsrg3_valence_2b`.
2. **2B kets** `|pq>` with `p<=q`, global index `Index2(p,q)=p(2N-1-p)/2+q` (`ModelSpace.cc:1166-1175`),
   i.e. lexicographic. Channel ket lists (`KetList`) are built in increasing global index
   (`TwoBodyChannel.cc`), so **within a channel kets are lexicographically ordered**. The scalar
   inverse Pandya transform relies on this (§9, P-12). Pauli: `p==q` only for even J
   (`TwoBodyChannel.cc:145`). Channel index `6J + 2(Tz+1) + parity` (`ModelSpace.cc:1104-1111`).
3. **Cross-coupled (CC) channels** `TwoBodyChannel_CC` use `Tz_cc = |tz_p - tz_q|/2 ∈ {0,1}`
   (`TwoBodyChannel.cc:181-189`) but the same index formula and the same count as normal channels,
   so one third of the CC slots are empty. `SortedTwoBodyChannels(_CC)` are sorted by
   decreasing size with empty channels dropped, for load balance (`ModelSpace.cc:1238-1257`).
4. Per channel: `KetIndex_pp/hh/ph/cc/vc/qc/vv/qv/qq` (`arma::uvec`, *not contiguous*) and
   occupation vectors `Ket_occ_hh = n_a n_b`, `Ket_unocc_hh/ph = (1-n_a)(1-n_b)` (`ModelSpace.hh:132-144`).
   "hh" = both orbits (partially) occupied, "ph" = one of each.
5. **TwoBodyME** (`TwoBodyME.hh:30-56`): `std::map<{ch_bra,ch_ket}, arma::mat>` with only
   `ch_bra <= ch_ket` stored; matrices hold **normalized antisymmetrized** J-coupled MEs.
   `GetTBME*` (no `_norm`) return the **unnormalized** `\tilde V = sqrt((1+δab)(1+δcd)) V`
   (`TwoBodyME.cc:170-177`). Matrix products over the stored `a<=b` basis therefore equal
   `1/2 Σ_ab` of unnormalized MEs (header comment `TwoBodyME.hh:42-50`).
   `AddToTBME(ch,ch,ibra,iket,v)` also writes the mirror `(iket,ibra)` with `±` for (anti)Hermitian
   (`TwoBodyME.cc:307-324`); `AddToTBMENonHerm` writes one element. `GetTBME_norm` with
   `ch_bra>ch_ket` on a *non-Hermitian* operator returns the antihermitian conjugate (`TwoBodyME.cc:200-204`).
6. **ThreeBodyME** in `"pn"` mode (`ThreeBodyME/ThreeBodyStorage_pn.*`): one flat
   `std::vector<double>`; blocks per 3B channel pair `(ch_bra<=ch_ket)` located through an
   `unordered_map ch_start`; diagonal blocks stored as packed upper triangle, off-diagonal dense;
   `herm_flip` sign on access (`ThreeBodyStorage_pn.cc:237-338`). 3B kets `|pqr; Jpq>` with
   `p<=q<=r` (`ModelSpace.cc:1282-1307`), restricted by `emax_3body` and `E3max` at storage level.
   `GetME_pn(Jab,Jde,twoJ,a,b,c,d,e,f)` sorts orbits and recouples on the fly via
   `GetKetIndex_withRecoupling` (returns up to a few canonical kets + 6j-based coefficients);
   `GetME_pn_TwoOps` fetches X and Y with one recoupling using `thread_local` scratch vectors
   (`ThreeBodyStorage.cc:314-339`). Element type is `double` (`ThreeBME_type`, float commented out).
7. **Reduced MEs**: tensors (`rank_J>0`) and by default parity- or Tz-changing scalars are stored
   reduced (`Operator.cc` ctor: `is_reduced(Jrank>0 or Trank>0 or p>0)`). Convention: 1B rows
   `×sqrt(2j_bra+1)`, 2B blocks `×sqrt(2J_bra+1)` (`Operator.cc:1051-1077`, `TwoBodyME.cc:145-166`).
   Scalar kernels require **non-reduced** inputs; tensor kernels require X non-reduced, Y and Z reduced.
8. **1b channel lists** are `std::set<index_t>` (`Operator::GetOneBodyChannel`), iterated in hot loops.
9. **Dagger operators**: `legs` odd; a fictitious "Q-space orbit" carries the extra leg
   (`Operator::Q_space_orbit`); the 1-leg part lives in `OneBody` column 0; 3-leg part in
   `ThreeLegME` (map ch → matrix) (`ThreeLegME.hh`).

---

## 2. Catalog of commutator routines (Q1)

Rank notation `[a,b]→c`: particle rank of X, Y, Z (dagger routines use *leg* counts, see §6.3).
Scaling is the naive m-scheme cost quoted in code comments / derived from the formula; "impl" says
how the J-coupled code actually evaluates it.

### 2.1 IMSRG(2), scalar–scalar (`Commutator.cc`)

| Routine | Line | Ranks | What it computes / how | Scaling |
|---|---|---|---|---|
| `comm110ss` | 650 | [1,1]→0 | `Σ_a n_a(2j_a+1)(XY−YX)_aa` over holes, row·col products; returns if Z antihermitian or non-scalar | n² |
| `comm220ss` | 687 | [2,2]→0 | per stored block: `2(2J+1) h · Σ diag(X_hh,: (n̄n̄∘Y_hh,:)ᵀ)∘nn`, handles off-diagonal blocks for odd-parity X,Y | n⁴ |
| `comm111ss` | 792 | [1,1]→1 | dense `XY−YX` on full `n_o×n_o` (ignores block diagonality) | n³ |
| `comm121ss` | 818 | [1,2]+[2,1]→1 | monopole contraction `Σ_ab n_a n̄_b (2j_b+1)X_ab Ȳ_biaj …`, upper triangle + mirror | n³–n⁴ |
| `comm221ss` | 908 | [2,2]→1 | builds Mpp/Mhh (full ladder GEMMs) then `Σ_cJ (2J+1)[n_c Mpp + n̄_c Mhh]_{cicj}/(2j_i+1)`; only used when `comm222_pp_hhss` is off | n⁵ (+n⁶ build) |
| `comm122ss` | 993 | [1,2]+[2,1]→2 | per channel, per ket column: gather 1B column × selected 2B columns (GEMV), `Z2 −= W + hZ Wᵀ`; parity/Tz-changing → `comm122ss_slower` (1090, element-wise) | n⁵ |
| `comm222_pp_hhss` | 1216 | [2,2]→2 | `Z2 += Mpp − Mhh` | n⁶ |
| `ConstructScalarMpp_Mhh` | 1236 | helper | `Mpp = X P_{n̄n̄} Y − Y P_{n̄n̄} X`, `Mhh = X P_{nn} Y − Y P_{nn} X` per channel GEMM | — |
| `comm222_pp_hh_221ss` | 1370 | [2,2]→1,2 | fused 222pp/hh + 221 (default path) | n⁶ |
| `comm222_phss` | 1846 | [2,2]→2 | Pandya transform → GEMM per CC channel → inverse Pandya; non-channel-diagonal → `comm222_phst` with reduced copies | n⁶ |
| `comm222_phss_slower` | 2003 | [2,2]→2 | element-wise with `AngMom::SixJ` computed on the fly — **unused** | n⁶+ |
| `DoPandyaTransformation*`, `AddInversePandyaTransformation*` | 1475–1781 | helpers | §4.6 | — |
| `prod110ss` | 2158 | — | debugging product | — |

### 2.2 IMSRG(3), scalar–scalar (`IMSRG3Commutators.cc`)

| Routine | Line | Ranks | Implementation | Scaling |
|---|---|---|---|---|
| `comm330ss` | 72 | [3,3]→0 | flat loop over (3B ch, ibra), `iket<ibra`, OMP reduction | n⁶ |
| `comm331ss` | 199 | [3,3]→1 | element loops (i, ch_ab, j, ab, twoJ, cde) with `GetME_pn_TwoOps` recoupling; per-thread Z1 copies | n⁷ |
| `comm231ss` | 426 | [2,3]+[3,2]→1 | element loops over (i≥j) pairs × 2B blocks × twoJ, `GetME_pn` | n⁶ |
| `comm132ss` | 615 | [1,3]+[3,1]→2 | per Z2 element: Σ_a,b,twoJ `GetME_pn` | n⁶ |
| `comm232ss` | 814 | [2,3]+[3,2]→2 | dispatch → `comm232ss_srs_optimized` (867) → `_old` (1305, GEMM per 1b channel) or `comm232ss_slow` (2065, element-wise) | n⁷ |
| `comm133ss` | 4517 | [1,3]+[3,1]→3 | per 3B block: dense embedding of 1B op into 3B space, dense GEMM | n⁷ ("really more like n⁶") |
| `comm223ss` | 5012 | [2,2]→3 | direct element-wise sum (3×3 permutations × J1',J2' × a) per 3B element; doubles as perturbative-triples evaluator | n⁷ |
| `comm233_pp_hhss` | 5403 | [2,3]+[3,2]→3 | per 3B channel: dense embedding of 2B op (incl. P(ij/k) recoupling), dense GEMM | n⁸ |
| `comm233_phss` | 5994 | [2,3]+[3,2]→3 | 2B Pandya ⊗ 3B recoupled to `(lm)(ij)'` basis, GEMM per CC channel, 6j unpack | n⁸ |
| `comm332_ppph_hhhpss` | 3299 | [3,3]→2 | per 2B channel GEMM over `(abc)d` | n⁸ |
| `comm332_pphhss` | 3927 | [3,3]→2 | 3B → ph-coupled `(ij')(cd a'b')` matrices, 4 GEMMs per CC channel, inverse transform | n⁸ |
| `comm333_ppp_hhhss` | 7024 | [3,3]→3 | per 3B channel GEMM over `abc` | n⁹ |
| `comm333_pph_hhpss` | 7306 | [3,3]→3 | pph-recoupled (`(ij)n'` basis) GEMM per pph channel, 3×3-permutation 6j unpack | n⁹ |
| `*_tz` variants | 3140, 3589, 5733, 6569, 7168, 7781 | — | J=0 *channel-changing* Y with fully scalar X; header says "implemented by AI" and `comm233_phss_tz` "FAILED unit test" (`IMSRG3Commutators.hh:84-89`) — yet it is wired into dispatch (§9, P-20) | — |

### 2.3 Scalar–tensor (`TensorCommutators.cc`), X scalar (non-reduced), Y rank λ (reduced)

IMSRG(2): `comm111st` 25, `comm121st` 46, `comm122st` 158, `comm222_pp_hh_221st` 336,
`comm221st` 506 / `comm222_pp_hhst` 512 (wrappers: run the fused routine and restore the
discarded half of Z — test helpers), `comm222_phst` 1099 (+ `DoTensorPandyaTransformation` 536,
`_SingleChannel` 673, `AddInverseTensorPandyaTransformation` 790, `InitializePandya` 1004).
IMSRG(3) ("3n7" first): `comm331st` 1425, `comm223st` 1579, `comm231st` 1800, `comm232st` 1954,
`comm133st` 2327, `comm132st` 2531; above n⁷: `comm332_ppph_hhhpst` 2645, `comm332_pphhst` 2781,
`comm233_pp_hhst` 3064, `comm233_phst` 3314, `comm333_ppp_hhhst` 3569, `comm333_pph_hhpst` 3660.

### 2.4 Scalar–dagger (`DaggerCommutators.cc`), Y odd-leg

`comm211sd` 33 (`Z1 += X1·Y1`), `comm231sd` 52 ([1B, 3-leg]→1-leg), `comm413_233sd` 109
([2B,1-leg]→3-leg and [1B,3-leg]→3-leg, brute force), `ConstructDaggerMpp_Mhh` 198,
`comm433_pp_hh_431sd` 236 ([2B,3-leg]→3-leg pp/hh and →1-leg), `comm433sd_ph` 463 (Pandya),
`comm433sd_ph_dumbway` 310 (loop reference), `DoPandyaTransformation_SingleChannel_Dagger` 588,
`AddInversePandyaTransformation_Dagger` 662.

### 2.5 Factorized double commutators (`FactorizedDoubleCommutator.cc`, namespace `Commutator::FactorizedDoubleCommutator`)

`comm223_231(Eta,Gamma,Z)` 92 → `comm223_231_chi1b` 133 (1B intermediates, "fast") +
`comm223_231_chi2b` 444 (2B intermediates, "slow"); `comm223_232` 1046 → `_chi1b` 1090 +
`_chi2b` 1398. Computes `[Eta,[Eta,Gamma]_3b]_1b` and `_2b` without forming the 3B intermediate.
Called from `BCH.cc:188-190` (factorized correction in the BCH series) and `HFMBPT.cc:1364-1512`,
**not** through `Commutator::Commutator`.

### 2.6 Reference implementations (`ReferenceImplementations.hh:29-153`)

Same names as above for IMSRG(2)/(3) scalar and tensor terms; `diagram_C*/D*` and
`comm223_231/232(_BruteForce)` for nested commutators; `TriplesGuess`; `GetMP4_F1..F39`.

---

## 3. Dispatch logic and configuration state (Q2)

### 3.1 `Commutator::Commutator(X,Y)` (`Commutator.cc:221-315`)

```
PreCalculateSixJ()                       // no-op after first call
if X,Y number conserving:
   both J=0 : de-reduce copies if needed → CommutatorScalarScalar → if Z parity/Tz≠0: Z.MakeReduced()
   X scalar, Y tensor : CommutatorScalarTensor(Xnred, Y)
   X tensor, Y scalar : −CommutatorScalarTensor(Ynred, X)
   both tensor : print + exit(EXIT_FAILURE)
else: one dagger → (±)CommutatorScalarDagger; two daggers → exit
```

### 3.2 `CommutatorScalarScalar` (`Commutator.cc:322-487`)

- Z: `J=X.J+Y.J`, `T=X.T+Y.T`, parity sum mod 2, **particle rank = 3 iff `use_imsrg3`**; Z
  antihermitian if X,Y same hermiticity, Hermitian if opposite, else non-Hermitian (337-342).
  With `use_imsrg3`, `Z.ThreeBody.SetMode("pn")` **allocates and zero-fills the full 3B storage of
  Z on every call** (344-345, `ThreeBodyME.cc:365-412`), even if the 3B terms are then skipped.
- IMSRG(2) terms in this order, each gated by `comm_term_on["name"]` (359-391):
  `110, 220, 111, 121, 122`, then `comm222_pp_hh_221ss` if both `222_pp_hhss` and `221ss` are on
  (else whichever is on), then `222_phss`.
- IMSRG(3) block iff `use_imsrg3 && X.Norm()>threebody_threshold && Y.Norm()>threebody_threshold`
  (396; `Norm()` includes a full pass over 3B storage, `Operator.cc:1644-1663`). Order:
  `133, 330, 223, 232, 331, 231, 132, 233_pp_hh, 233_ph, 332_ppph_hhhp, 332_pphh, 333_ppp_hhh,
  333_pph_hhp` (403-454). If `scalar3b_transform_first_pass`, `SetSingleThread(true)` (398-399)
  and the flag is cleared afterwards (459).
- Optional `discard_{0,1,2}b_from_3b` post-processing (461-477): normal-orders a 3B-only copy and
  subtracts its induced 0/1/2B parts — but also calls `Z.EraseThreeBody()` (474), i.e. **zeroes
  Z's entire 3B part** (suspicious, §9 P-27). Only reachable from Python (`pyIMSRG.cc:649-651`).
- No final `Symmetrize()`: every scalar kernel must fill both triangles itself.

### 3.3 `CommutatorScalarTensor` (`Commutator.cc:491-611`)

Z gets Y's J, T, particle rank; first call per `(λ, πX, πY)` key sets `single_thread`
(`tensor_transform_first_pass[J*4+πX+2πY]`, 517-518). Terms: `111st, 121st, 122st`,
`comm222_pp_hh_221st` **only if both `222_pp_hhst` and `221st` are on** (the else-branch is
commented out, 529-539: turning off one silently disables both), `222_phst`; IMSRG(3) tensor block
with the same threshold (558-599). Ends with `Z.Symmetrize()`/`AntiSymmetrize()` (605-608).

### 3.4 `CommutatorScalarDagger` (`Commutator.cc:615-632`)

`Z = Y` copy with 0B/1B/3-leg erased, then `comm211sd, comm231sd, comm413_233sd,
comm433_pp_hh_431sd, comm433sd_ph`. No term switches, no threshold.

### 3.5 Internal (per-kernel) dispatch

- `comm122ss` → `comm122ss_slower` unless X,Y,Z all parity-even and T=0 (`Commutator.cc:995-999`).
- `comm222_phss` with parity/Tz-changing X or Y → **copies** operators, makes Y and Z reduced,
  calls `comm222_phst`, de-reduces Z (`Commutator.cc:1855-1903`); two isospin-changing → exit.
- `ConstructScalarMpp_Mhh` picks intermediate channels for parity/Tz-changing X; both X and Y
  isospin-changing → exit (`Commutator.cc:1281-1302`).
- `comm232ss` → `comm232ss_srs_optimized` → immediately returns via `comm232ss_srs_optimized_old`
  (channel-diagonal X,Y,Z) or `comm232ss_slow` (otherwise) (`IMSRG3Commutators.cc:895-904`).
- The n⁸/n⁹ IMSRG(3) kernels (`332_ppph`, `332_pphh`, `233_pp_hh`, `233_ph`, `333_ppp`, `333_pph`)
  with parity/Tz-changing operators: if X fully scalar and both J=0 → `*_tz`; otherwise make a
  reduced copy of Y, reduce Z, call the `*st` routine, de-reduce (e.g. `IMSRG3Commutators.cc:3323-3338`, `6011-6024`).
- `comm133ss` with both X and Y non-channel-diagonal prints "not yet implemented" and **returns
  silently** (4565-4569).

### 3.6 Global configuration state (all namespace-scope mutable globals)

| Knob | Default | Defined / setter | Read by | Notes |
|---|---|---|---|---|
| `comm_term_on` (map string→bool) | 2B on, 3B off | `Commutator.cc:36-82`; `TurnOn/OffTerm` | dispatchers | string keys; typos create new keys silently (test uses `'comm131ss'`, `test/UnitTestCommutators3.py`) |
| `use_imsrg3` | false | `SetUseIMSRG3` (99) sets the 13 scalar 3B terms | dispatch, Z rank, BCH | also set by `SetUseIMSRG3N7_Tensor`/`_Tensor` |
| `use_imsrg3_n7` | false | `SetUseIMSRG3N7` (111) | **never read by kernels** | `true` turns off the 6 n⁸/n⁹ terms and on the 7 n⁷ terms; `false` does nothing (not reversible) |
| `use_imsrg3_mp4` | false | **never assigned**; `SetUseIMSRG3_MP4(true)` (164) keeps only `330,232,133,223` | printed only | order-dependent with N7 (`imsrg++.cc:1048-1051`) |
| `single_thread` | false | `SetSingleThread` | **only TensorCommutators IMSRG(2) pragmas** (`if (not single_thread)`) | scalar kernels ignore it |
| `verbose` | false | `SetVerbose` | extra timers | |
| `threebody_threshold` | 0 | `SetThreebodyThreshold` (`IMSRG3Commutators.cc:35`) | dispatch | compared to full operator norms |
| `imsrg3_no_qqq` | false | `SetIMSRG3Noqqq` | `comm232` (1682,1825), `comm223ss` (5125,5164) | drop 3B states with cvq sum >5 (all q-space) |
| `imsrg3_only_vvv` | false | setter | `comm223ss` only | |
| `imsrg3_valence_2b` | false | setter | `comm232ss` variants only | restrict Z2 output and klj states to valence |
| `discard_{0,1,2}b_from_3b` | false | `Discard*From3b` | dispatch | see 3.2 |
| `perturbative_triples`, `pert_trip_novvv` | false | assigned directly (`IMSRGSolver.cc:1206`, `HFMBPT.cc:1452`) | `comm223ss` | changes semantics of `comm223ss` (§5.4) |
| `imsrg3_dE6max` | 1e20 | `IMSRG3Commutators.cc:25` | **unused** | dead knob |
| `FactorizedDoubleCommutator::use_*` (12 flags) | mostly true | `FactorizedDoubleCommutator.cc:12-86` | only `use_1b_intermediates`, `use_2b_intermediates`, `use_goose_tank_only_1b` are read | 9 flags are dead |
| ModelSpace: `dE3max` (999), `occnat3cut` (−1), `E3max` (`min(14,3·emax)`), `emax_3body` (= emax), `e_fermi[tz]` | | `ModelSpace.cc:73,164` | IMSRG(3) kernels | `dE3max`: Σ|e−e_F(tz)| over the 3 orbits; `OccNat3Cut`: Π occ_nat(1−occ_nat) |
| ModelSpace: `scalar_transform_first_pass`, `scalar3b_transform_first_pass`, `tensor_transform_first_pass[]` | true | `ModelSpace.hh:351-353` | gating of some parallel loops | "warm-up" pass to fill 6j/9j hash maps |
| `ModelSpace::OCC_CUT` | 1e-6 | static | `comm121ss`, dagger | |
| `IMSRGProfiler::timer/counter` | | **static maps** (`IMSRGProfiler.hh:41-44`) | every kernel | written unguarded outside parallel regions, under `critical` in `comm133ss` |
| `ModelSpace::SixJList/NineJList` | | static `unordered_map` | `GetSixJ/GetNineJ` | shared across all model spaces |

BCH/solver-level knobs that select or wrap commutator kernels (outside this module, but they
change what is computed): `BCH::use_goose_tank_correction` (`GooseTankUpdate` uses `comm221ss` /
`comm222_pp_hh_221st`, `BCH.cc:225-252`), `BCH::use_factorized_correction` + `TurnOn_comm223_231/232`
(calls `FactorizedDoubleCommutator`, `BCH.cc:183-191`), `use_brueckner_bch` (`BCH.cc:93`),
`only_2b_omega` (BCH), `only_2b_eta` (`Generator.cc:328,509`; leaves `X.ThreeBody` unallocated, which
the 3B kernels detect via `IsAllocated()`/norms). Perturbative triples
(`IMSRGSolver.cc:1196-1213`) temporarily sets `E3max = 3·emax`, preloads `Wbar.OneBody` with the
Hamiltonian one-body part, flips `perturbative_triples`, and calls `comm223ss` directly.

---

## 4. IMSRG(2) algorithms (Q3)

### 4.1 Conventions shared by all scalar IMSRG(2) kernels

- Restricted sums: GEMMs over stored `a<=b` kets absorb the `1/2` of `1/2 Σ_ab`; explicit loops
  over unordered labels use unnormalized `GetTBME*` and keep the `1/2`/`1/4`.
- Occupations enter as `n` and `n̄ = 1−n` everywhere (fractional occupations supported); pp/hh/ph
  index classes are only used to skip zero blocks.
- Output symmetry: compute upper triangle (`j>=i`, `iket>=ibra`) and mirror with `hZ`, or compute
  `A` and form `A ± Aᵀ` (see 4.4/4.6). The mirror trick **assumes X and Y each have definite
  hermiticity** in `comm122ss`, `comm222_phss`, inverse Pandya (P-9).
- All dense products are Armadillo expressions (column-major) dispatched to the linked BLAS `dgemm`;
  zero-size operands are guarded explicitly "because for size=0, we get DGEMM errors" (1329).
  Expressions like `X*Y - Y*X` materialize temporaries; no in-place `beta=1` accumulation is used.

### 4.2 `comm110ss`, `comm220ss`, `comm111ss`

- `comm110ss`: loop over holes, `(2j_a+1) occ_a (X_a,: Y_:,a − Y_a,: X_:,a)` (664). Code comment
  notes an unexplained orders-of-magnitude slowdown of a previous version under IMSRG(3).
- `comm220ss` (687-779): for each stored block: rows restricted to hh kets of the bra channel,
  columns weighted by `n̄n̄` (built from `Ket_unocc_hh/ph`), `z0 += 2(2J+1) hY Σ diag(X Yᵀ)∘nn`;
  for `ch_bra != ch_ket` the transposed contribution is added with `hX` (751-773). Returns early for
  antihermitian Z. OMP: `parallel for schedule(dynamic) reduction(+:z0)` (716).
- `comm111ss`: `Z1 += X1 Y1 − Y1 X1` on full `n_o×n_o` matrices (795).

### 4.3 `comm121ss` (818-876)

`Z_ij += Σ_{a∈holes} Σ_{b∈1b-ch(a)} n_a n̄_b [ (2j_b+1) X_ab Ȳ_biaj − (2j_a+1) X_ba Ȳ_aibj ]` and the
X↔Y term, with `Ȳ = GetTBMEmonopole` = `sqrt((1+δ)(1+δ)) Σ_J(2J+1)V^J/((2j_1+1)(2j_2+1))`
(`TwoBodyME.cc:693-760`). Upper triangle in the 1b channel of i, mirrored with `hZ`. Each monopole
call loops over J and does channel lookups — pure gather.

### 4.4 Ladder terms: `ConstructScalarMpp_Mhh`, `comm222_pp_hh_221ss`, `comm221ss` (1236-1457)

Per stored channel block (list built from `Z.TwoBody.MatEl`, skipping blocks where X or Y is
`is_zero()` — "optimization suggested by Antoine Belley", 1255-1262):

```cpp
if (bras_pp.size() > 0) Mpp += X_ijab.cols(bras_pp) * Y_abkl.rows(bras_pp);
if (bras_hh.size() > 0) { Mhh += X_ijab.cols(bras_hh) * diagmat(nanb)      * Y_abkl.rows(bras_hh);
                          Mpp += X_ijab.cols(bras_hh) * diagmat(nbarnbar_hh) * Y_abkl.rows(bras_hh); }
if (bras_ph.size() > 0) Mpp += X_ijab.cols(bras_ph) * diagmat(nbarnbar_ph) * Y_abkl.rows(bras_ph);
if (Z Hermitian && ch_bra==ch_ket) { Mpp += Mpp.t(); Mhh += Mhh.t(); }   // XY−YX = XY+(XY)^T
else if (Z antiHermitian && diag) { Mpp -= Mpp.t(); Mhh -= Mhh.t(); }
else  { /* explicit −Y P X products with the YX intermediate channel */ }
```
(`Commutator.cc:1330-1360`). So `Mpp ≡ X P_{n̄n̄} Y − Y P_{n̄n̄} X`, `Mhh ≡ X P_{nn} Y − …`, and
`Z2 += Mpp − Mhh`. The 1B part (221) contracts the same intermediates:
`Z_ij = Σ_c Σ_J (2J+1) [n_c Mpp^J_{cicj} + n̄_c Mhh^J_{cicj}] / (2j_i+1)` with `GetTBME_J` (unnormalized)
lookups (1421-1447). For channel-changing X the intermediate channel is recomputed from the
operator's parity/Tz and transposed blocks are fetched with `hX/hY` (1281-1311). Note the
`.cols(uvec)`/`.rows(uvec)` selections allocate gathered temporaries before every GEMM.

### 4.5 `comm122ss` (993-1086)

For each channel (parallel, `SortedTwoBodyChannels`) and each ket `(ij)`: build index lists
`a ∈ 1b-ch(i)` with partner ket `(a j)` and phase/normalization factor
(`flipphaseij = −(−1)^{(j_i+j_j)/2−J}` when `a>j`, `√2` when `a==j`), same for `(i a)`; then
`W.col(ij) = [Y2(:,aj ∪ ia) | X2(:,aj ∪ ia)] · [X1 col ∘ f ; −Y1 col ∘ f]` (one gathered GEMV per column),
`×2` for `i==j`; finally `Z2 −= W + hZ Wᵀ` (1083). Allocations of `std::vector`/`arma::uvec` happen
per ket inside the hot loop.

### 4.6 Particle–hole term `comm222_phss` and the Pandya machinery (1459-2000)

Pandya transform (doc, 1459-1473):

```
Xbar^J_{i j̄ k l̄} = − Σ_J' (2J'+1) { j_i j_j J ; j_k j_l J' } X^{J'}_{i l k j}
```

Algorithm:

1. Allocate `Z_bar[ch_cc] = zeros(Ncc, 2·Ncc)` for every CC channel (1913-1918).
2. Parallel over CC channels (1920-1983; guarded by `#ifndef OPENBLAS_NOUSEOMP`):
   `DoPandyaTransformation_SingleChannel_XandY` (1557-1650) builds, for "ph" bras = hh ∪ ph kets of
   the CC channel, **both orderings** `(ab)` and `(ba)` stacked (offset `nph`):
   `Xt(cd, ab) = Xbar_{c d̄, a b̄} · (n_a − n_b) · norm`, `Y(ab, cd) = Ybar_{a b̄, c d̄} · norm`,
   where `norm = √2^{[a==d]+[c==b]}` converts the normalized ME fetched by
   `GetTBME_J_norm_twoOps` (one index lookup, two operators) to unnormalized; for `a==d` or
   `c==b` only even J' contribute (`dJ_std=2`, 1625-1629); 6j from the dense cache
   `GetCachedSixJ`; kets failing a Tz consistency test are skipped (1606-1607).
3. Full-ket Y via phase flip (1944-1966): `PhaseMatZ(i,j) = (−1)^{(j_p+j_q)/2}` products,
   `Y_flip = [Y_tail ∘ PhaseMatY ; Y_head ∘ PhaseMatY]` (`PhaseMatY` includes `hY`),
   `Zbar = Xt · [Y | Y_flip]` — one GEMM of size `Ncc × 2nph × 2Ncc`.
4. Hermiticity (1969-1982): head columns `± transpose` (Hermitian Z: `+`, else `−`), tail columns
   `+= tailᵀ ∘ PhaseMatZ`. This replaces the explicit `YX` product (saves two GEMMs).
5. `AddInversePandyaTransformation` (1666-1781): flattened parallel loop over `(ch, ibra)`
   (`dynamic,1`), `iket >= ibra` (`> ibra` for antihermitian):
   `commij = −Σ_J' (2J'+1) {j_i j_j J; j_k j_l J'} Zbar^{J'}[il, kj]` with column offset `+Ncc` when
   `k>j`; `commji` analogously with `(ik, lj)` unless `k==l` (`commji=commij`) or `i==j`
   (`commji = (−1)^{j_i+j_j+j_k+j_l} commij`); then
   `z = −(commij − (−1)^{j_k+j_l−J} commji)/norm`, `norm = 1+δ` if `δ_ij==δ_kl` else `√2`, added to
   `(ibra,iket)` and mirrored with `hZ` (1773-1778). Row index never exceeds `Ncc` only because
   `i<=l` and `i<=k` follow from `ibra<=iket` and lexicographic ket order (comment 1765).
   Overall sign: "the Pandya transform has a minus sign in the definition, and the ph commutator
   has an overall minus sign" (1990-1994).

`DoPandyaTransformation_SingleChannel` (1475-1555, used by tensor and dagger code) builds one
operator in "normal" (`2nph × Ncc`) or "transpose" (`Ncc × 2nph`, times `herm·(n_a−n_b)`)
orientation using the *hash-map* `GetSixJ`.

### 4.7 6j/9j usage and caching

- `GetCachedSixJ(jj1,jj2,J3,jj4,jj5,J6)` (`ModelSpace.hh:508-511`): dense table
  `SixJCache_112112` of size `(emax+1)⁴ (2emax+2)²` doubles (e.g. ≈364 MB at emax=14), built eagerly
  per ModelSpace (and copied with it), no symmetry reduction, **no bounds checks**
  (`AngMomCache.{hh,cc}`). Thread-safe (read-only). Used by the scalar ph kernel, inverse Pandya,
  `comm223ss` (when `twoJ <= 2emax+1`).
- `GetSixJ(double…)` (`ModelSpace.cc:1699-1745`): static `unordered_map` keyed by a hash;
  `PreCalculateSixJ` (1763-1885) inserts all symbols of the forms `{ja jb J1; jc jd J2}` (jb, jd up
  to 3-body j) and `{J1 J2 J3; ja jb jc}` (ja, jb up to 3-body j; restricted for emax>10) and fills
  them in parallel. A miss outside a parallel region computes and inserts under `critical`; a miss
  **inside** a parallel region prints "DANGER" and calls `exit()`.
- `GetNineJ`: hash map, misses computed from three `GetSixJ`, inserted only when single-threaded
  (2062-2110). `comm222_phst` calls `PreCalculateNineJ` first.
- "First pass" logic exists to make misses happen while single-threaded: e.g. `comm332_pphhss`
  explicitly touches `{J J J; j j J3}` symbols on the first pass (4002-4031); some loops carry
  `if (not ...first_pass)` clauses (1656, 4383, 5463). A rewrite should precompute dense,
  read-only tables for every symbol family used and delete first-pass semantics entirely.

### 4.8 Normalization cheat-sheet

| Situation | Factor |
|---|---|
| stored 2B ME → unnormalized | `×√2` per identical pair (`GetTBME`) |
| writing an unnormalized result into stored ME | `/√((1+δij)(1+δkl))` (e.g. `comm132ss` 799, `comm232` 2050, `comm122ss_slower` 1177-1180) |
| GEMM over `a<=b` with an `a==b` internal pair, 3B kernels | `occ *= 0.5` for `a==b` (`comm232`, 1554-1555; `comm231ss` 560-563) |
| 3B stored kets `p<=q<=r` vs unrestricted sums | permutation multiplicity 6 / 3 / 1 for distinct / two equal / all equal (`comm330ss` 132-136, 168-172; `comm331ss` 353-357); `symm/6` in `comm332_ppph` (3416) and `comm333_ppp` (7101-7105) |
| 3B recoupling with an identical pair in a 2B factor | `rec *= √2` when `I1==I2` (`comm223ss` 5200-5201, 5256-5257, 5310-5313) |
| reduced ↔ non-reduced | `√(2j+1)` (1B), `√(2J_bra+1)` (2B) |

---

## 5. IMSRG(3) and IMSRG(3f2) (Q4)

### 5.1 3B data structures and truncations

- Output Z3 and inputs X3/Y3 use the pn storage of §1.6. Kernels iterate `Z.ThreeBody.Get_ch_start()`
  (stored channel pairs) and `ThreeBodyChannel::GetKet(i)` (`Ket3{p,q,r,op,oq,oR,Jpq}`).
- Truncations, applied redundantly and inconsistently inside kernels:
  - storage: `emax_3body` per orbit, `E3max` on `e_a+e_b+e_c` (`ThreeBodyStorage.cc:193-234`);
  - commutator: `dE3max` on `Σ|e − e_F(tz)|` and `OccNat3Cut` on `Π occ_nat(1−occ_nat)` — applied to
    external and (in most kernels) internal 3B states; plus `cvq` filters (§3.6).
  - Semantics differ at the boundary: most kernels skip when `product < cut`, `comm231ss` keeps only
    when `product > cut` (521-522, 551-552) — identical only for the default `cut = −1`.
- `GetME_pn` performs orbit sorting, channel lookup (`GetThreeBodyChannelIndex`), up to 3 recoupling
  terms per side, and `ch_start` hash lookups for every element; `MatEl.at()` is bounds-checked.

### 5.2 Element-wise kernels (`330, 331, 231, 132`)

- `comm330ss` (72-186): `z0 += 1/36 · occ · symm_abc · symm_def · (2J+1) (x_abcdef y_defabc − y x)` with
  `occ = nnn n̄n̄n̄ − n̄n̄n̄ nnn` over `iket < ibra` (the antisymmetric occupation factor accounts for
  the other triangle; no factor 2).
- `comm331ss` (199-411): `parallel for collapse(2)` over `(i, ch_ab)`; for each `j<=i` in the 1b channel,
  `ab` ket, `twoJ`, `(π,Tz)` of `cde`, and every `cde` ket: `zij += 1/12 · sym_ab · sym_cde · occ ·
  (twoJ+1)/(2j_i+1) (x_abicde y_cdeabj − y x)` with `occ = n_a n_b n̄_c n̄_d n̄_e + n̄_a n̄_b n_c n_d n_e`
  (corrected Oct 2022, 326). Writes go to per-thread `n_o×n_o` matrices reduced
  serially at the end (227-233, 399-408). `dE3max`/`OccNat3Cut` use `(a,b,j)` but never `(a,b,i)`
  (289-292) — asymmetric truncation (P-23).
- `comm231ss` (426-596): parallel over `(i>=j)` pairs; for both orderings (`xy_iter`), loops over all
  2B blocks of the 2B operator, `iket<=ibra`, twoJ: `Z_ij += prefactor (twoJ+1)(X_abcd y_cdiabj − y_abicdj X_cdab)/(2j_i+1)`
  with `prefactor = n_a n_b n̄_c n̄_d − n̄_a n̄_b n_c n_d`, halved for `a==b` and for `c==d`.
  Calls `OP3.ThreeBodyNorm()` (full pass over 3B storage) **inside the parallel loop for every pair**
  (488). Copy-paste bugs in the truncation: `d_ed` uses `ec` (544), `occnat_c/d` read from the bra
  (545-546) (P-22).
- `comm132ss` (615-811): parallel over stored blocks with an inner **nested** `parallel for collapse(2)`
  (660, 672; nested regions run single-threaded by default). Per element: `Σ_a Σ_{b∈1b-ch(a)} Σ_twoJ (n_a−n_b)
  (twoJ+1)/(2J+1) (X_ab y_ijbkla − Y_ab x_ijbkla)` (J = 2B channel J), then normalized by
  `1/√((1+δij)(1+δkl))`. A `std::set` is built per (element, a, b_loop) (742-748).

### 5.3 `comm232ss` (the slowest n⁷ term per comments, 814-2059)

Active path `comm232ss_srs_optimized_old` (1305-2059), channel-diagonal case:

1. Enumerate 1b channels `{j2, π, tz}` (`local_one_body_channels`; `external_*` restricted to valence if
   `imsrg3_valence_2b`) and, per channel of `i`, the list of `|klj'>` states `{k,l,j,Jkl}` with
   `(kl)` from 2B channels compatible with total `j_i` (1373-1454).
2. Parallel over 1b-channel keys (1492): select internal `|abc'>` from the same list with occupation
   `n_a n_b n̄_c + n̄_a n̄_b n_c` (×½ if `a==b`); allocate dense
   `X2MAT(dim_i × dim_abc)`, `Y3MAT(dim_abc × dim_klj)` (and Y2/X3).
3. Collect unique `(k,l,c,Jkl,twoJ')` / `(a,b,j,Jab,twoJ')` recouplings (hash set), compute
   `GetKetIndex_withRecoupling` once each and store as a CSR-like cache
   (`recouple_info/kets/coeff`, 1700-1739).
4. Fill (with `#pragma omp taskloop grainsize(1)` over rows so idle threads help big channels, 1753):
   `X2MAT(i,abc) = −√(2Jab+1) occ X^{Jab}_{ciab}` and
   `Y3MAT(abc,klj) = Σ_{J'} (2J'+1)/√(2Jkl+1) {j_j j_i Jkl; j_c J' Jab} Σ_{I,K} r_I r_K Y3(abj_I, klc_K)`.
5. `ZMAT = X2MAT·Y3MAT − Y2MAT·X3MAT` (1914).
6. Unpack (parallel over `(ch,ibra)`, 1939-2056):
   `z_ijkl = [ (−1)^{(j_i+j_j)/2−J} Z_{i,klj} − Z_{j,kli} + hX hY (Z_{l,ijk} − (−1)^{(j_k+j_l)/2−J} Z_{k,ijl}) ] / √((1+δij)(1+δkl))`,
   where positions are found with **`std::find` linear searches** in the 1b-orbit and klj lists for
   every output element (1964-1965, 1996-1997, 2029-2032).
- Packed hash key `k + l<<8 + j<<16 + Jkl<<24 + twoJ<<32` (1337-1338): assumes orbit indices < 256
  (≈ emax ≤ 14 for two species) (P-25).
- `comm232ss_slow` (2065-2253): parallel over blocks, per element Σ_c Σ_ab Σ_twoJ with `GetME_pn`, 6j
  via hash map; handles parity/Tz-changing operators.

### 5.4 `comm223ss` (5012-5376) — produces all induced 3B; n⁷; "taking all the time" (comment at 852)

Flattened parallel loop over `(ch_bra, ch_ket, ibra)` (5037-5076, `dynamic,1`), `iket <= ibra` in the
same channel. Per element: 3 permutations of `ijk` (`ABC, CBA, ACB`, with recoupling coefficient,
permutation phase, `√2` for identical pair) × `J1'` × 3 permutations of `lmn` × `(π_a, tz_a)` × `J2'` ×
1b channels of `a` (filtered by j range) × orbits `a`:
`z += rec_ijk rec_lmn 6j √((2J1'+1)(2J2'+1)) phases (x_126a y_3a45 − y_126a x_3a45)`, with the two 2B
matrices of each product looked up once per `(J1',J2',π,tz)` and indexed directly in the innermost
loop (5238-5265, 5320-5325). 6j from the dense cache when `twoJ <= 2emax+1`, else from the hash map
with a symmetry permutation (5292-5301). Writes `Z3.AddToME_pn_ch(ch_bra,ch_ket,ibra,iket)` —
unique per thread (packed storage maps `(i,j)` and `(j,i)` to one slot).
**Perturbative-triples mode** (`perturbative_triples=true`): iterate all 3B kets of the model space
(Z3 need not be allocated), skip pairs without the right occupations, and instead of storing,
accumulate `E += 1/36 sym_ijk sym_lmn (2J+1) z² (occ_ijk unocc_lmn − occ_lmn unocc_ijk)/(E_ijk − E_lmn)`
with `E_ijk` from **Z.OneBody** diagonal; finally `Z.ZeroBody = E` (overwrite) (5050-5070, 5343-5373).

### 5.5 GEMM-per-3B-channel kernels (`133`, `233_pp_hh`, `333_ppp_hhh`)

- `comm133ss` (4517-4989): parallel over stored 3B blocks; filter kept kets (dE3max/occnat, flat
  lookups); dense `X1MAT_bra (nbra × nxy)` etc.: the 1B operator embedded in 3B space — for each bra
  `ijk` and each of the 3 positions, replace the orbit by `a` in its 1b channel, recouple
  `GetKetIndex_withRecoupling`, scatter `X1(i,a)·recouple` (4782-4867). Fill 3B blocks with
  `GetME_pn_ch` (inner `parallel for collapse(2)`, nested → serial). `Z3MAT = X1·Y3 − Y1·X3`, then
  `Z3MAT −= hX hY Z3MATᵀ` (channel-diagonal) or two more GEMMs (4951-4962). Up to 9 dense `N3×N3`
  matrices per block per thread; the embedding matrices are very sparse (~3·n_a·k nonzeros per row)
  yet multiplied densely (N3³). Known disagreement for isospin-changing operators (comment 4514).
- `comm233_pp_hhss` (5403-5731): parallel over 3B channels (guarded by first-pass), dense
  `X2MAT/Y2MAT/X3MAT/Y3MAT/Z3MAT (N3×N3)`; X2MAT embeds the 2B operator with
  `P(ij/k) = 1 − P_ik − P_jk` recoupling (6j `{j_i j_j Jij; j_k J Jab}`), factor `½·symm·(1−n_a−n_b)`;
  `Z3MAT = X2·Y3 − Y2·X3; Z3MAT −= hX hY Z3MATᵀ` (5700-5701); unpack over a `std::map kept_lookup`.
- `comm333_ppp_hhhss` (7024-7166): per 3B channel, `XMAT(N3 × Nabc) = f_abc X3`, `YMAT = Y3`,
  `ZMAT = hY XMAT·YMATᵀ; ZMAT −= hX hY ZMATᵀ` (7144-7145) — the cleanest dense-GEMM IMSRG(3) term.

### 5.6 Recoupled ("Pandya-like") 3B kernels (`233_ph`, `332_pphh`, `333_pph`, `332_ppph`)

- `comm233_phss` (5994-6566): serial setup builds, per CC channel, ph 2B kets with `n_a−n_b ≠ 0` and an
  `unordered_map` from packed `(l,m,Jlm,i,j,Jij)` (12/12/12/12/8/8 bits, 6049-6050) to columns; stores
  `Z3_ph[ch_cc] (2Ncc × N_lmij)` for **all** CC channels simultaneously ("This will be the big one").
  Parallel over CC channels: `X2_ph (2Ncc × 2Nph)`, `X3_ph (2Nph × N_lmij)` and `Z3_ph = X2_ph·Y3_ph −
  Y2_ph·X3_ph` (6338). Parallel over 3B channels to unpack with precomputed ket recouplings, 3×3
  permutations, `Σ_{J1',J2',Jph} (2Jph+1) 6j(J1' J2' Jph; j6 j3 J)` and hash lookups (6349-6545).
- `comm332_pphhss` (3927-4499): four `Zbar` variants per CC channel (`ij/ji` × `kl/lk`), each
  `X3_ij (Ncc × N_abcd) · Y3_ij (N_abcd × Ncc)` (4370-4373), with 3B MEs recoupled via cached
  `GetKetIndex_withRecoupling` lists; inverse transform per 2B channel (4383).
- `comm333_pph_hhpss` (7306-7779): pph channels keyed `(twoJ_ph, π, twoTz)`, kets `(i,j,n,Jij)` (packed
  12-bit key, 7364), `zbar = Xbar_ijnabc · Ybar_abcijn; zbar −= hX hY zbarᵀ` (7574-7575) for all pph channels
  stored at once; unpack with `{j3 J J1'; j6 Jph J2'}` and 3×3 permutations (7585-7760). "This one works, but
  it's incredibly slow" (`Commutator.cc:451`). TODO: dE3max not applied to internal states (7292).
- `comm332_ppph_hhhpss` (3299-3478): per 2B channel (parallel), list all `(abc ∈ 3B kets, d)` compatible
  with the channel, `XMAT/YMAT (N2 × N_abcd)` via `GetME_pn` (recoupling per element), `ZMAT = hY X·Yᵀ −
  hX Y·Xᵀ` (3466). Occupation sign fixed Sep 3 2025 (3403).

### 5.7 Memory hot spots (IMSRG(3))

Z3 allocation per commutator call (§3.2); dense per-channel `N3×N3` scratch per thread in `133` (≤9) and
`233_pp_hh` (5); all `Z3_ph` (`233_ph`) and all `Zbar` (`333_pph`) alive simultaneously; per 1b channel
`dim_abc × dim_klj` in `232`; operator copies for channel-changing dispatch (copying X/Y including 3B,
§3.5); `MakeReduced/NotReduced` on Z rescales its full 3B storage twice.

### 5.8 Factorized double commutators (IMSRG(3f2)-style)

Goal: `[Eta,[Eta,Gamma]_3b]_{1b}` (`comm223_231`) and `_{2b}` (`comm223_232`) with the 3B intermediate
never stored. Structure: de-reduce inputs if needed (92-128, 1046-1088); `_chi1b` builds 1B
intermediates (e.g. `Chi_221_a = Σ (2J0+1)(n n n̄ n̄ − …) Eta·Eta` through a per-channel GEMM
`2(2J0+1) Eta_nnnn · Eta` symmetrized, 160-199) then contracts with Gamma; `_chi2b` builds 2B and
Pandya-transformed intermediates (`bar_Eta`, `bar_Gamma`, `barCHI_III/V/VI…` in `std::deque<arma::mat>`
per CC channel, 1662-1677, 2200-2201, 2508-2529, 2836-3363) and contracts via GEMMs and Pandya/inverse
transforms; diagrams named I, II, IIIa–f, IV; all channel loops are `parallel for`.
Assumptions/pitfalls: `hZ = hGamma` and the sign `Eta_nnnn(iket,ibra) *= −occfactor` assume an
**antihermitian Eta** (146, 196-197: "SRS this is a guess"); diagonal elements are multiplied by
`−occfactor²`; **no E3max/dE3max/emax_3body truncation** of the implied intermediate (no occurrences
in the file), so it equals explicit IMSRG(3) only without 3B truncations; recent bug fixes
("Bug found by Bingcheng Sep 2026", 1014, 1023, 2182); charge-changing Gamma path in `_chi2b` is
"MAGICAL MYSTERY CODE WRITTEN BY AI" (1400-1625).
Validation (`UnitTest.cc:6021-6087`): `comm223ss → comm231ss + comm232ss` vs factorized, `|Δ| < 1e-6`.

### 5.9 Reference implementations and how they are used

`ReferenceImplementations` are straightforward J-coupled sums over unrestricted orbit labels with
`GetTBME_J`/`GetME_pn` (e.g. `comm220ss` = `¼ Σ_abcd Σ_J (2J+1) n_a n_b n̄_c n̄_d (x y − y x)`,
`IMSRG2Commutators.cc:44-86`), "benchmarked against the m-scheme implementation and then left
untouched" (`IMSRG2Commutators.cc:6-8`). Several are still OpenMP-parallel (15 pragmas in the IMSRG(3)
file). `UnitTest::Test_against_ref_impl` (`UnitTest.cc:1463-1550`) runs optimized and reference
kernels on the same random operators and requires `Norm(Z−Zref) + ΔZ0 < 1e-6`. The Python tests
use tiny spaces (`ModelSpace(2,'He6','He6')` for the n⁷ terms, emax=1 for n⁸/n⁹, X antihermitian,
Y Hermitian, ranks 3; `test/UnitTestCommutators3.py`) — **no 3B truncation paths (dE3max, OccNat3Cut,
E3max<3emax, emax_3body<emax) are exercised**, which is where the copy-paste bugs live.
`FactorizedNestedCommutators.cc` holds brute-force nested diagrams (`diagram_CIa … DIVb`,
`comm223_231_BruteForce`); `HFMBPT4.cc` holds 39 MP4 diagram evaluators (not commutators).

---

## 6. Tensor, isospin-changing, and dagger commutators (Q5)

### 6.1 Scalar–tensor (rank λ>0, or J=0 channel-changing routed here)

- Inputs: X scalar **non-reduced**, Y and Z **reduced**; Z gets Y's J/T/particle rank.
- 2B blocks connect `ch_bra ≠ ch_ket` with `|J_bra − J_ket| ≤ λ ≤ J_bra+J_ket`; matrices fetched as
  `ch_bra<=ch_ket ? M : Mᵀ·h` (`TensorCommutators.cc:383-390`).
- `comm222_pp_hh_221st` (336-504): **serial channel loop** (BLAS threads only), stacked GEMMs
  `[X_dir | −Y_exc] · [Y_dir; X_exc]` with occupations, then 1B part with
  `√((2J1+1)(2J2+1)) {J1 J2 λ; j_j j_i j_c} (−1)^{j_j+j_c+J1+λ}` (comment: phase differs from
  Parzuchowski et al., "published phase is wrong", 492); mirror `Z1(j,i) += hZ (−1)^{(j_i−j_j)} Z1(i,j)`.
  Exits for isospin-changing scalar X (359-363).
- Tensor Pandya (536-788): `Xbar = −Σ_{J1,J2} √((2J1+1)(2J2+1)(2J_bra+1)(2J_ket+1)) (−1)^{j_b+j_d+J_ket+J2}
  9j{j_a j_d J1; j_b j_c J2; J_bra J_ket λ} X^{J1 J2}_{adcb}`; for λ=0 the 9j is replaced by a 6j with
  `/√((2J2+1)(2J_bra+1))` (615-622). `comm222_phst` (1099-1413): X Pandya'd once ("transpose",
  stored per CC channel), Y Pandya'd per `(ch_bra_cc, ch_ket_cc)` pair, `Zbar = [Xbar_J1 | −φ Ybar_J2J1ᵀ] ·
  [[Ybar_J1J2 | Ybar'] ; hX [Xbar_J2 | Xbar']ᵀ]` with phase matrices (1258-1281); λ>0 recomputes Y Pandya
  inside the loop, λ=0 precomputes all (memory-hungry) (1285-1405); explicit handling of both-parity-odd
  X,Y (A. Belley comments 1127-1142).
- Tensor IMSRG(3) mirrors the scalar kernels with extra 6j/9j recoupling (`comm223st` uses
  `{j3 ja J3; j6 j1 J1'}{J3 J2' λ; j2 j1 j6}`-type products, 1640-1780). Three kernels note
  "this needs to be modified for beta decay" (1780, 3556, 3646). These parallel loops are **not**
  gated by `single_thread`, and 6j coverage for 3-body j's is restricted for emax>10
  (`ModelSpace.cc:1781-1785`) → potential `GetSixJ` miss → `exit()` in parallel regions.

### 6.2 Isospin/parity-changing scalars (J=0, `rank_T≠0` or `parity=1`)

Stored reduced by default; `CommutatorScalarScalar` returns Z reduced if Z changes parity/Tz
(`Commutator.cc:250-253`). IMSRG(2): `comm122ss_slower`, Mpp/Mhh intermediate-channel logic,
`comm222_phss → comm222_phst`. IMSRG(3): `*_tz` (X scalar) or `*st` fallbacks; `comm232ss_slow`;
`comm133ss` handles X *or* Y channel-changing (bugfix "Sept 2026", 4562) but not both; two
isospin-changing operators are unsupported everywhere (exits).

### 6.3 Dagger (odd particle number) commutators

Legs notation: `(1)` = single creation operator, `(2)` = 1B, `(3)` = `a†a†a`, `(4)` = 2B
(`DaggerCommutators.cc:15-18`). The extra creator is modeled as a fixed Q-space orbit; Y's 1-leg part
is `OneBody(:,0)`; 3-leg MEs `ThreeLegME` per 2B channel of `(ij)` with the third index `k` coupled to
Q — channel checks use `tz_k − tz_Q` (`DaggerCommutators.cc:141`, also 344, 697). Kernels are mostly
serial (pragmas commented out in `comm231sd`, `comm413_233sd`); `comm433sd_ph` is the Pandya analogue with
`Zbar(norb × 2Ncc) = Ybar_dagger · [Xbar | Xbar_flip]` (463-582); `hx` "assuming X is either hermitian or
antihermitian" (466). Performance is "so far from being the bottleneck" (comment at 32).

---

## 7. Parallelization (Q6)

### 7.1 Inventory

| Kernel | OpenMP construct | Race handling |
|---|---|---|
| `comm220ss` | `parallel for schedule(dynamic) reduction(+:z0)` over blocks (716) | reduction |
| `comm121ss`, `comm221ss`, 221-part of fused | `parallel for dynamic,1` over orbits i (824, 928, 1405) | each i writes row i + mirror column i (disjoint since `j>=i`) |
| `comm122ss`, `comm122ss_slower` | `dynamic,1` over channels (1006, 1107) | per-channel output block |
| `ConstructScalarMpp_Mhh` | `dynamic,1` over blocks, removed if `OPENBLAS_NOUSEOMP` (1268) | per-block outputs |
| `comm222_phss` Zbar build | `dynamic,1` over CC channels, `OPENBLAS_NOUSEOMP` guard (1920) | per-channel Zbar |
| `AddInversePandyaTransformation` | `dynamic,1` over flattened `(ch,ibra)` (1686) | row `ibra` + mirror (disjoint because `iket>=ibra`) |
| `DoPandyaTransformation` (deque) | `dynamic,1 if (not scalar_transform_first_pass)` (1656) | hash-map 6j safety |
| `comm330ss` | flattened `(ch,ibra)` + `reduction(+:z0)` (107) | reduction |
| `comm331ss` | `collapse(2)` over `(i,ch_ab)` (234) | **per-thread `n_o×n_o` copies** + serial reduction |
| `comm231ss` | `dynamic,1` over `(i,j)` pairs (466) | per-pair output |
| `comm132ss` | outer over blocks + nested `collapse(2)` (660/672) | `AddToTBMENonHerm` both halves explicitly |
| `comm232ss_srs_optimized_old` | outer over 1b channels (1492) + `omp taskloop grainsize(1)` rows (1753); unpack over `(ch,ibra)` (1939) | channel-private matrices; `ZMAT_list`/`klj_list` pre-created serially so `operator[]` in the loop only finds |
| `comm133ss` | outer over 3B blocks (4584) + nested `guided collapse(2)` / `dynamic` loops (nested → 1 thread) | per-block Z3 writes; named `critical(comm133ss_profiler)` around profiler updates |
| `comm223ss` | flattened `(ch_bra,ch_ket,ibra)` `dynamic,1` (+ `reduction(+:Emp2)`) (5076) | unique `(ibra,iket)` slots |
| `comm233_pp_hhss` | over 3B channels `if (not first_pass)` (5463) | per channel |
| `comm233_phss` | setup serial; over CC channels (6162); over 3B channels (6349) | per channel |
| `comm332_*`, `comm333_*` | over 2B/CC/pph/3B channels, `dynamic,1` | per channel |
| Tensor IMSRG(2) | `if (not single_thread)` on every pragma | first-pass serialization |
| Factorized | plain `parallel for` / `dynamic,1` over channels/orbits | per-channel intermediates |

No `omp atomic` in the commutator files; `critical` only for the profiler (and inside 6j/9j caches).
Determinism: dynamic schedules + reductions + per-thread copies make results bitwise non-reproducible
run-to-run (~1e-15 relative).

### 7.2 Threading hazards to remove by design

- Hash-map 6j/9j caches with lazy insertion and first-pass serialization (§4.7).
- Nested `parallel for` inside outer parallel loops (`comm132ss`, `comm133ss`) — silently serial.
- BLAS threading vs outer OpenMP: comments in `comm133ss` (4575-4579) report OpenBLAS on a laptop prefers
  BLAS threads while MKL on 24 cores prefers channel threads by 10×; `OPENBLAS_NOUSEOMP` is a compile-time
  switch (`MakefileRagnar:9,113`).
- `std::map<int,double> e_fermi` copied per kernel and accessed with `operator[]` in parallel loops (safe
  only because keys ±1 exist); prefer a 2-element array (as `comm232_old`/`comm133ss` already do).
- Static profiler maps, global `comm_term_on`: commutators cannot run concurrently (they currently never do).

### 7.3 GPU suitability classification

Flat/regular (good GPU fit as-is or after flattening): per-channel GEMMs everywhere (Mpp/Mhh, ph Zbar,
232 ZMAT, 133/233pphh/333ppp per-3B-channel GEMMs, 332/333 recoupled GEMMs) → grouped/variable-size
batched GEMM; `comm330ss`, `comm220ss`, `comm110ss` reductions; `comm223ss` (output-stationary, one
work item per 3B element, no write conflicts — but deep loop nest with data-dependent bounds).

Irregular (ragged channels, gathers, branching): forward/inverse Pandya transforms (6j-weighted gathers
across CC channels), 3B fills via `GetME_pn` (orbit sorting + recoupling + hash lookups per element),
`comm331ss`/`comm231ss`/`comm132ss` element loops, `comm232` unpack (`std::find`), all hash-map
lookups (`ph_kets3_lookup`, `ket_lookup_pph`), embedding-matrix construction in `133`/`233pphh`.

---

## 8. Hotspots, data movement, restructuring opportunities (Q7)

### 8.1 Per-kernel summary

| Kernel | Dominant cost | Access pattern | Opportunity |
|---|---|---|---|
| `comm222_pp_hh_221ss` | Σ_ch N2³ GEMMs (2–4 per block) | dense GEMM after gathering pp/hh/ph column subsets | store kets sorted by occupation class so subsets are contiguous views; one grouped batched GEMM over channels; skip-zero-block test retained |
| 221 one-body part | n_o² × n_o × J lookups | `GetTBME_J` gathers from Mpp/Mhh | precompute index lists / express as SpMV on flattened 2B storage |
| `comm222_phss` | Pandya fwd (Σ_cc Ncc·2nph·J' with 6j) + GEMM (4 Ncc² nph) + inverse (Σ_ch N2²·J'·2) | gather/scatter across CC channels | treat Pandya and inverse Pandya as precomputed **sparse linear maps** (CSR of `(target, source, 6j·(2J'+1)·phase)`), applied as SpMV/SpMM on device; batched GEMM in between; keep the hermiticity trick |
| `comm122ss` | Σ_ch N2² · 2n_a | per-column gather + GEMV | express as `Z2 += Y2·E(X1)` with sparse embedding `E` (or batched GEMM on padded 1b blocks) |
| `comm121ss` | monopole gathers | irregular | precompute monopole matrices once per operator (cheap) |
| `comm223ss` | ~N3²·9·J'²·n_a per channel | element-wise, reads 2B blocks by computed indices | GPU kernel per output element with flattened 2B storage and dense 6j table; or recast as GEMM (the code comment describes the "flip and recouple" mat-mult form, 5002-5008, but the active version is the direct one) |
| `comm232ss` | fill of `Y3MAT` (recoupled gathers) + GEMM per 1b channel | gather-heavy fill, dense GEMM | the recoupling cache is already a CSR matrix: build `Y3MAT = R_abj · Y3 · R_klcᵀ` contracted with 6j as SpMM on device; replace `std::find` unpack with direct index maps |
| `comm133ss` | dense N3³ GEMM with ~3·n_a-sparse embedding | dense GEMM | SpMM (sparse 1B embedding × dense 3B block) cuts flops and the 9 N3² scratch matrices |
| `comm233_pp_hhss` | dense N3³ GEMM with sparse-ish 2B embedding | dense GEMM | SpMM or block-structured GEMM over `(ab)` sub-blocks; avoid N3² scratch per thread |
| `comm333_ppp_hhhss` | N3·Nabc·N3 GEMM | dense, regular | ideal batched GEMM |
| `comm233_phss`, `comm332_pphhss`, `comm333_pph_hhpss` | recoupled fills + GEMM + 6j unpack | hash-map gathers | precompute recoupling sparse maps per channel (reused across BCH/flow steps since the basis is fixed); stream channels to bound memory |
| `comm331ss` | element loops with on-the-fly recoupling | gather | recast as per-channel GEMM of 3B blocks in a `(ab)i` recoupled basis |
| `comm231ss` | element loops + **full 3B norm per pair** | gather | hoist the norm (trivial fix) |

### 8.2 Cross-cutting data-movement issues

- Every 3B access goes through `unordered_map ch_start` + recoupling; hoist block pointers per channel pair.
- Armadillo `cols(uvec)/rows(uvec)` and `join_*` create temporaries before each GEMM (`comm122ss`, Mpp/Mhh,
  `comm222_phst` 1272-1277).
- `std::set` iteration of 1b channels in innermost loops; `std::map` for kept lists; per-iteration
  `std::vector` allocations (`comm122ss`, `comm132ss`, `comm223ss` permutation vectors).
- Full-3B passes hidden in `Norm()` calls (dispatch threshold, `comm231ss`, `comm330ss` X3/Y3 norms).

### 8.3 Batched-GEMM inventory (sizes per channel; all ragged)

`M×K×N`: Mpp/Mhh `N2×|subset|×N2`; ph `Ncc×2nph×2Ncc`; 232 `dim_i×dim_abc×dim_klj`; 133 `N3×N3×N3`;
233pphh `N3×N3×N3`; 333ppp `N3×Nabc×N3`; 233ph `2Ncc×2nph×N_lmij`; 332pphh 4×`Ncc×N_abcd×Ncc`;
332ppph `N2×N_abcd×N2`; 333pph `N_ijn×N_abc×N_ijn`. Channel sizes span orders of magnitude, so the
rewrite needs grouped batched GEMM (or size-binned streams) plus host-side largest-first scheduling.

### 8.4 Prior art for restructuring: `Commutator232.cc` (not compiled)

Builds per-(3B channel, 2B channel) `OneBodyBasis/TwoBodyBasis/ThreeBodyBasis` objects with dense
local index maps (`wrap_factor` arithmetic), dense 6j matrices (`GenerateSixJMatrixIJ/JI`), occupation and
phase vectors, and evaluates four diagrams with a contiguous inner kernel
`Comm232Core = Σ sixj·occ·mat2b·mat3b` over flat `std::vector<double>` slices (`Commutator232.cc:1102-1311`;
`Commutator232.hh:63-374`). This "precompute bases + flat arrays + simple dot-product kernel" layout is
the most GPU-ready design in the tree and a good template for the new 3B kernels.

---

## 9. Pitfalls to replicate or deliberately fix (Q8)

Conventions to replicate exactly:

- **P-1** Normalized storage / unnormalized math (§1.5, §4.8); `√2` handling for identical orbits,
  even-J-only for identical pairs (1625-1629).
- **P-2** Pandya sign: transform carries `−`, ph commutator carries `−`, inverse adds
  `−(commij − (−1)^{j_k+j_l−J} commji)/norm` (§4.6). Ket-flip phase `(−1)^{(j_p+j_q)/2−J}` (`Ket::Phase`,
  `ModelSpace.hh:80`); CC "flip" phase matrices use `(−1)^{(j_p+j_q)/2}` products.
- **P-3** `ModelSpace::phase(int)` is called with double arguments in places (`phase(ji+jj+jk+jl)`,
  `phase(jk+jl−J)`, 1746, 1774): implicit truncation is exact only because those sums are integers.
- **P-4** Reduced-ME convention and which operand is reduced in tensor kernels (§1.7, §6.1).
- **P-5** Z hermiticity rule (same → antihermitian, opposite → Hermitian, else non-Hermitian).
- **P-6** Fractional occupations everywhere (`n`, `n̄`; pp/hh/ph classes only for skipping); `OCC_CUT=1e-6`
  and various hard-coded skip thresholds (1e-6, 1e-8) affect results at ~1e-10 level.
- **P-7** IMSRG(3) truncation semantics (E3max, emax_3body, dE3max relative to `e_fermi[tz]`, OccNat3Cut,
  cvq filters) and their per-kernel application — results with active cuts depend on these details.
- **P-8** Tensor 1B phase in `comm222_pp_hh_221st` deliberately differs from the published formula (492).

Silent assumptions (make explicit or generalize):

- **P-9** `comm122ss`, `comm222_phss`/inverse Pandya, `comm433sd_ph`, and the hermiticity tricks assume each of
  X, Y is Hermitian or antihermitian; non-Hermitian inputs give wrong results (non-Hermitian Y treated as
  antihermitian, `ketmin = ibra+1` for non-Hermitian Z at 1706). Only `comm121ss/221ss`, Mpp/Mhh and the
  `_slower` variants handle general Z.
- **P-10** Factorized double commutators assume antihermitian Eta (`hZ=hGamma`, `−occfactor` guess, 146, 196).
- **P-11** `comm220ss`/`comm110ss` return early for antihermitian Z (zero-body must vanish) — fine, but do not
  "fix" into computing it.
- **P-12** Inverse Pandya indexes `Zbar` rows with `GetLocalIndex(min(i,l),max(i,l)) + (i>l ? Ncc : 0)` but
  Zbar has only `Ncc` rows: correct only because `iket>=ibra` and lexicographic ket order imply `i<=l`,
  `i<=k` (1734, 1765). Re-sorting kets (e.g. by occupation class) breaks this silently.
- **P-13** Tensor/scalar dispatch: `comm221st` and `comm222_pp_hhst` cannot be toggled independently in
  `CommutatorScalarTensor` (529-539).
- **P-14** Order-dependent, non-reversible setters (`SetUseIMSRG3N7(false)` no-op; `use_imsrg3_mp4` never set;
  `SetUseIMSRG3N7_Tensor` sets `use_imsrg3`) (§3.6).
- **P-15** `perturbative_triples` reuses `comm223ss` with `Z.OneBody` as energy denominators and overwrites
  `Z.ZeroBody` (5357-5372); caller must preload the Fock diagonal.
- **P-16** `single_thread` affects only tensor IMSRG(2) loops; scalar first-pass relies on
  `scalar*_first_pass` checks scattered in kernels.
- **P-17** `GetCachedSixJ` has no bounds checks; `comm223ss` switches to the hash map for `twoJ>2emax+1`.
- **P-18** Channel-changing paths copy whole operators (incl. 3B) and toggle reduced/non-reduced on Z
  (O(N_3B) work each), and `CommutatorScalarScalar` allocates Z's full 3B storage on every call.

Known bugs, approximations, disabled/experimental code (decide deliberately):

- **P-19** `comm232ss_srs_optimized` body after the early returns (906-1301) and the commented
  `comm232ss_mh_optimized` are dead; the live path is `_old` + `_slow`.
- **P-20** `*_tz` kernels "implemented by AI"; header says `comm233_phss_tz` FAILED its unit test
  (`IMSRG3Commutators.hh:87`) yet it is the live path for J=0 channel-changing Y (6011-6017). Comments still
  reference an `IMSRG_TZ_FORCE` env var that no longer exists (3138, 6566, 7022, 7304).
- **P-21** `comm133ss` silently returns if both X and Y are channel-changing (4565-4569); known disagreement
  for isospin-changing operators (4514).
- **P-22** Truncation copy-paste bugs: `comm231ss` `d_ed` uses `ec`, `occnat_c/d` from bra (544-546);
  `comm233_pp_hhss` `d_eb` uses orbit a (5535, 5585, 5639; replicated "(sic) as in original" in `_tz` at 5851).
  Only active with finite `dE3max`/`OccNat3Cut`.
- **P-23** Asymmetric cuts: `comm331ss` applies dE3max/OccNat3Cut to `(a,b,j)` only (289-292), then mirrors;
  `comm231ss` uses `>` vs `<` elsewhere for the OccNat cut.
- **P-24** `comm231ss` full 3B `Norm()` per (i,j) pair inside the parallel loop (488) — pure performance bug.
- **P-25** `comm232` hash packs orbit indices in 8 bits (1337-1338): collisions for orbit index ≥256
  (two-species emax ≥ 15); `233_ph`/`333_pph` use 12-bit orbit and 8-bit J fields (6049, 7364).
- **P-26** `comm232` unpack uses `std::find` per output element (O(N2²·dim_klj)) (1964-2032).
- **P-27** `discard_*b_from_3b` erases Z's whole 3B part (`Commutator.cc:474`); likely unintended.
- **P-28** Recent sign/occupation fixes that change numbers vs older outputs: `comm332_ppph` sign (Sep 3 2025,
  3403), `comm331ss` occupation (Oct 2022, 326), `comm333_ppp` occupation (Jul 2022, 7088), `comm132ss` Y3=0 bug
  (Sept 2026, 731), `comm133ss` channel bugfix (Sept 2026, 4562), `comm332_pphhss` twoJ' range (4156),
  factorized fixes (Sept 2026). Validate against the *current* reference, not archived results.
- **P-29** Tensor IMSRG(3): "needs to be modified for beta decay" (isospin-changing tensors) (1780, 3556, 3646);
  6j coverage limited for emax>10 → possible `exit()` inside parallel loops.
- **P-30** Unsupported combinations exit the process (tensor–tensor, dagger–dagger, two isospin-changing
  operators, isospin-changing scalar X in `comm222_pp_hh_221st`): replace with exceptions.
- **P-31** Dead knobs: `imsrg3_dE6max`, 9 of 12 factorized flags; dead functions (§0).

---

## 10. Numerical-equivalence guide for the rewrite

1. **Golden data**: generate random operators with `UnitTest::RandomOp(ms, J, T, parity, rank, herm)` for every
   (hermiticity pair, rank combination) and dump inputs/outputs of each kernel from the reference build
   (both optimized and `ReferenceImplementations`). Compare per component (0B, 1B, each 2B block, each 3B block)
   with absolute tolerance ~1e-10 relative to the operator norm; the reference suite only checks
   `Norm(ΔZ)+ΔZ0 < 1e-6` (`UnitTest.cc:1543-1548`; note the signed `ZeroBody` term can cancel), which is
   too loose for a refactor.
2. **Coverage beyond the existing tests**: emax 3–4, `E3max < 3·emax`, `emax_3body < emax`, finite `dE3max` and
   `OccNat3Cut` (expect deliberate differences where P-22/P-23 are fixed), fractional occupations, parity-odd and
   Tz-changing J=0 operators (`*_tz` vs `*st` paths), tensor λ=1,2, dagger operators, non-Hermitian inputs
   (document unsupported cases per P-9).
3. **Invariants to assert**: hermiticity of Z (both triangles), `[X,Y] = −[Y,X]`, `comm222_pp_hh_221ss ≡
   comm222_pp_hhss + comm221ss`, factorized `comm223_231+232 ≡ comm231ss+comm232ss(comm223ss)` without 3B
   truncation, `comm232ss_slow ≡ comm232ss_srs_optimized_old` on channel-diagonal input, `comm222_phss ≡
   comm222_phst` on reduced copies for J=0 channel-changing inputs.
4. **Non-determinism**: OpenMP dynamic schedules and reductions → compare with tolerances, never bitwise; for
   GPU, fix reduction orders in debug builds to make regressions bisectable.
5. **Ordering assumptions**: if the new storage reorders kets (recommended: contiguous pp/hh/ph classes, or
   SoA by channel), add explicit index maps for every place that relied on lexicographic order (P-12) and on
   `p<=q`, `p<=q<=r` canonical forms.
