# 01 — Core data model of imsrg++ (reference analysis)

Scope: `ModelSpace.hh/.cc`, `TwoBodyChannel.cc`, `ThreeBodyChannel.cc`, `Operator.hh/.cc`,
`TwoBodyME.hh/.cc`, `ThreeBodyME/*`, `ThreeLegME.*`, `AngMom.*`, `AngMomCache.*`,
`PhysicalConstants.hh`, `IMSRG.hh`, `IMSRGProfiler.*` of the reference tree
`imsrg-devel/src/`. All `file:line` citations are relative to `imsrg-devel/src/`.
Sizes quoted below come from a Python re-implementation of the orbit, ket and channel
enumeration, used only for counting (not part of the reference code).

Notation: `j2 = 2j`, `tz2 = 2t_z`, `e = 2n+l`. "Normalized" means antisymmetrized
J-coupled two-body states that carry the factor `1/sqrt(1+δ_pq)`.

---

## 0. Executive summary (most design-relevant facts)

1. **Orbit order is fixed by a closed formula.** The order is shell `N` ascending, `l`
   descending, `j` descending, then proton before neutron (`ModelSpace.cc:381-408`,
   hash `ModelSpace.cc:1154-1164`). Several places depend on three consequences:
   **even index means proton and `index+1` is its neutron partner**; an orbit's index
   does not depend on emax; and orbits are sorted by `e`.
2. **`tz2 = -1` is proton** (`ModelSpace.cc:536`, `1020-1023`). Two-body `Tz = -1, 0, +1`
   means pp, pn, nn.
3. **There is no E2max cut on kets.** Every pair `p<=q` is a ket. E2max is only used by
   the file readers, the Moshinsky precompute and NO2B storage. The 3-body kets do use
   E3max, and its default is `min(14, 3*emax)` (`ModelSpace.cc:164`).
4. **Two-body MEs are stored normalized.** Getters without `_norm` return
   `sqrt((1+δ_ab)(1+δ_cd))·V` (`TwoBodyME.cc:170-176`). **Three-body MEs are stored
   unnormalized** (`ThreeBodyME.hh:38-44`, `Operator.cc:656`).
5. **Tensor (reduced) storage keeps only the channel blocks with `ch_bra <= ch_ket`.**
   The other blocks come from a phase `±(-1)^(J_bra-J_ket)`. `rank_T` means `|ΔTz|`,
   not an isotensor rank.
6. **Operator "move" deep-copies the 2B and 3B parts.** `TwoBodyME` has a user-declared
   destructor and `ThreeBodyME` has user-declared copy operations, so neither gets
   implicit move operations. `Operator(Operator&&)` and `operator=(Operator&&)` therefore
   copy the maps of `arma::mat` and clone the 3B storage (`Operator.cc:110-125`,
   `TwoBodyME.hh:65`, `ThreeBodyME.hh:70,75`).
7. **Every operator built from a ModelSpace allocates a `ThreeLeg` block**
   (Σ_ch nkets_ch × norbits doubles, `ThreeLegME.hh:15`, `Operator.cc:69,86`). That is
   about 0.14 GB at emax=12 and 0.37 GB at emax=14 per operator. It is copied and scaled
   along with the operator even when unused.
8. **The angular-momentum caches are global statics inside `ModelSpace`.** The 6j, 9j
   and Moshinsky hash maps (`ModelSpace.hh:363-366`) are not thread-safe on insertion.
   A 6j cache miss inside a parallel region calls `exit()` (`ModelSpace.cc:1720-1740`).
   The commutators therefore run their first pass single-threaded. `PreCalculateSixJ`
   **clears** the shared static map (`ModelSpace.cc:1770`).
9. **The 3B layer has a strategy-pattern facade with five storage back-ends.** pn uses
   double, isospin uses **float**, no2b uses float or half, mono uses float. There are
   real bugs: `TransformToPN` slices the object and so does nothing useful
   (`ThreeBodyME.cc:336`); iso `Subtract` adds (`ThreeBodyStorage_iso.cc:53`); the binary
   writers write 1/8 (pn) or 1/4 (iso) of the data (`ThreeBodyStorage_pn.cc:372`,
   `ThreeBodyStorage_iso.cc:352`); a precedence bug affects reduced no2b sums
   (`ThreeBodyStorage.cc:389`).
10. **`ModelSpace` copy, move and assignment are unsound.** Copied `Ket`/`Ket3` keep raw
    `Orbit*` pointers into the *source* object. `ThreeBodyChannel::modelspace` is not
    re-pointed. `operator=` drops `OrbitLookup`, `MonopoleKets` and the 3-body
    structures, and it returns a copy by value (`ModelSpace.cc:81-117`, `853-960`).
11. **Hot paths.** Each pn 3B access (`GetME_pn`) sorts, recouples with cached 6j
    lookups, and does `std::map` lookups (`Ket3IndexLookup`, `ThreeBodyChannelLookup`)
    (`ThreeBodyStorage.cc:236-291`). Isospin↔pn conversion calls the GSL CG function four
    times per term (`ThreeBodyStorage_iso.cc:229-249`). Armadillo bounds checking is on
    (no `ARMA_NO_DEBUG`, `ModelSpace.hh:23`).

---

## 1. Orbits

### 1.1 The `Orbit` record (`ModelSpace.hh:41-58`)

| field | type | meaning |
|---|---|---|
| `n, l` | int | radial and orbital quantum numbers |
| `j2` | int | 2j |
| `tz2` | int | 2t_z: **-1 proton, +1 neutron** |
| `occ` | double | reference occupation in [0,1]; fractional means ensemble or equal filling |
| `occ_nat` | double | natural-orbital occupation (IMSRG(3) cuts); initialized to `occ` |
| `cvq` | int | 0 core, 1 valence, 2 q-space (decoupling partition) |
| `index` | int | position in `ModelSpace::Orbits` |

`operator==` compares only `(n,l,j2,tz2)`. The default constructor leaves every field
uninitialized. `ModelSpace::NULL_ORBIT` is a *mutable* static returned by
`GetOrbit(NOT_AN_ORBIT=99999)` (`ModelSpace.cc:15-16, 962-974`).

### 1.2 Index ordering

* Construction loop (`ModelSpace.cc:381-408`): `N=0..Emax`, `l=N,N-2,…,0` (descending),
  `j2=2l+1` then `2l-1`, `tz2=-1` then `+1`.
* Hash (`ModelSpace.cc:1154-1164`), with `N=2n+l`:
  `Index1_hash = N(N+3) + 1 - j2 + (tz2+1)/2`. Single-species spaces use
  `(N(N+3)+1-j2)/2`.
  This equals the sequential index only when no orbit is skipped. `OrbitLookup` maps
  hash to index (`AddOrbit`, `ModelSpace.cc:981-995`). `GetOrbitIndex(n,l,j2,tz2)`
  returns `NOT_AN_ORBIT` when the orbit is absent (`ModelSpace.cc:1145-1152`).
* First orbits: 0 p0s1, 1 n0s1, 2 p0p3, 3 n0p3, 4 p0p1, 5 n0p1, 6 p0d5, 7 n0d5, 8 p0d3,
  9 n0d3, 10 p1s1, 11 n1s1, 12 p0f7, …
* Implicit properties that other code relies on:
  * Index parity equals isospin, and the neutron partner of proton `a` is `a+1`. Used by
    `TwoBodyME::Set_pn_TBME_from_iso` (`TwoBodyME.cc:618-627`), the isospin 3B storage
    (`ThreeBodyStorage_iso.cc:96, 424-426`) and `ThreeBodyStorage_pn::GetME_iso`
    (`ThreeBodyStorage_pn.cc:216-221`). The Lmax and EmaxUnocc cuts treat p and n the same
    way, so pairs stay adjacent. Single-species mode breaks this.
  * An orbit's index is the same in a smaller emax space (prefix property).
    `Operator::Truncate` uses new-space 3B indices directly in the old space
    (`Operator.cc:798,803`).
  * Orbits are sorted by `e`. The isospin allocation loop `break`s on the energy cut
    (`ThreeBodyStorage_iso.cc:100-132`).
* String labels such as `"p0d5"`: `tz` from the first character, a single-digit `n`,
  the `l` letter, then `j2` (`String2Qnumbers`, `ModelSpace.cc:528-543`). The letters are
  `s p d f g h i j k l m n o` for l=0..12. **Note that `j` stands for l=7**; standard
  spectroscopic notation skips `j`. Valence spaces rely on this, e.g. `"n0j15"`
  (`ModelSpace.cc:52`). Two-digit `n` is mis-parsed.

### 1.3 Truncation parameters

| member | default | where it acts |
|---|---|---|
| `Emax` | ctor arg | orbits with `e<=Emax` (`ModelSpace.cc:381`). `Init(emax,…)` **ignores its emax argument** and uses `Emax` (comment at `ModelSpace.cc:226`) |
| `Lmax` | `emax` | orbits with `l<=Lmax` (`ModelSpace.cc:385`). `SetLmax` re-runs Init and exits if a hole or core orbit has `l>Lmax` (`ModelSpace.cc:486-524`) |
| `EmaxUnocc`, `hole_quantum_numbers` | `emax`, empty | drops orbits with `e>EmaxUnocc` whose `(l,j2)` is not "hole-like" (`ModelSpace.cc:394`). `SetEmaxUnocc` fills the set with every `(l,j)` up to `l<=min(Emax,lmax_holes+2)` and re-runs Init (`ModelSpace.cc:1420-1459`) |
| `E2max` | `2*emax` | **not applied to kets or channels** (`TwoBodyChannel.cc:143-152`). Used by readers (`ReadWrite.cc:216-217, 855`), `PreCalculateMoshinsky` (`ModelSpace.cc:1951-1957`), NO2B storage and a few imsrg_util operators |
| `E3max` | `min(14, 3*emax)` | 3-body kets (`ModelSpace.cc:1297`), all 3B storages, the 3B normal-ordering sum (`Operator.cc:649-652`) |
| `emax_3body_` | `emax` | per-orbit cut for the 3-body space (`orbits_3body_space_`, `ModelSpace.cc:1025-1028`). The header default `= Emax` is meaningless because constructors override it (`ModelSpace.hh:266`) |
| `dE3max`, `occnat3cut` | 999, -1 | IMSRG(3) configuration cuts: Σ_i\|e_i−e_F(tz_i)\| <= dE3max and Π n(1−n) >= occnat3cut (natural occupations) (`ModelSpace.cc:1370-1403`) |
| `OneBodyJmax`, `TwoBodyJmax` | derived | `OneBodyJmax = max j2`, `TwoBodyJmax = OneBodyJmax` (the maximum two-body J), `nTwoBodyChannels = 6*(TwoBodyJmax+1)`, or `2*(…)` for single species (`ModelSpace.cc:997-1005`) |
| `Lmax2`, `Lmax3`, `ThreeBodyJmax` | — | unused |
| `hbar_omega`, `target_mass` | 20, 16 | set only by some constructors |

`SetEmax(e)` also resets `EmaxUnocc`, `E2max=2e`, `emax_3body_=e` and `Lmax=e`. It rebuilds
the dense 6j cache when emax grows, but it **does not call Init** (`ModelSpace.cc:1405-1418`).

### 1.4 Orbit sets (all `std::set<index_t>`, iterated in ascending order)

| set | membership | source |
|---|---|---|
| `holes` | `occ >= OCC_CUT` (1e-6) | `ModelSpace.cc:1010-1013` |
| `particles` | `occ < OCC_CUT` (only *completely empty* orbits) | same |
| `core`, `valence`, `qspace` | `cvq` = 0, 1, 2 | `ModelSpace.cc:1014-1019` |
| `proton_orbits`, `neutron_orbits` | `tz2<0`, `tz2>0` | `ModelSpace.cc:1020-1023` |
| `all_orbits` | every orbit | `ModelSpace.cc:1024` |
| `orbits_3body_space_` | `e <= emax_3body_` | `ModelSpace.cc:1025-1028` |
| `OneBodyChannels[{l,j2,tz2}]` | orbits with the same `(l,j2,tz2)` (Hamiltonian-like) | `ModelSpace.cc:1032` |

A partially filled orbit is in `holes`. `SetOcc` and `SetOccNAT` change `occ` without
updating the sets, the ket lists or `Aref`/`Zref` (`ModelSpace.cc:1040-1050`).

### 1.5 Building a space from reference and valence names

Call chain: `ModelSpace(emax, ref, val)` → `Init v1` (`ModelSpace.cc:213`) →
`GetAZfromString` (`568-605`) → `GetOrbitsAZ(A,Z)` (`609-650`) → `Init v2` (`224-290`)
→ `Init v0` (`358-422`) → `SetupKets` and `Setup3bKets`.

* **Reference string.** `"O16"` means element symbol then A. `"vacuum"` maps to `n0`
  (A=Z=0). `"mix_A4.5_Z2.5"` gives fractional A and Z.
* **Filling (`GetOrbitsAZ`).** Within each major shell N the order is
  `g=2N+1, 2N-3, … >= -2N`, `j2=|g|`, with `l=(j2-1)/2` for g>0 and `(j2+1)/2` otherwise.
  So N=2 fills d5/2, s1/2, d3/2 and N=3 fills f7/2, p3/2, p1/2, f5/2. Protons and
  neutrons fill independently. The last orbit gets `occ = (#nucleons)/(2j+1)`. The
  three-argument overload skips orbits listed as valence.
* **Valence string.**
  * `"0hw-shell"`: `Get0hwSpace` (`ModelSpace.cc:659-694`).
  * `"jj-shell"`: `GetjjSpace` (`ModelSpace.cc:696-747`); it takes the largest j of the
    next shell as the intruder.
  * Contains `,`: `"Core,orb,orb,…"` (`ModelSpace.cc:753-770`). The core is filled while
    *skipping* the valence orbits.
  * Contains `"FCI"`: every orbit is valence and there is no core (`ModelSpace.cc:242-265`).
  * A key of the static `ValenceSpaces` table (`ModelSpace.cc:23-70`), e.g. `sd-shell`,
    `fp-shell`, `psd-shell`, `sdfp-shell`, `jj44`, `jj55`, semi-magic variants: the
    first entry is the core and the core is filled *without* skipping valence orbits.
  * Anything else is read as a core nucleus, giving a single reference with no valence
    orbits.
* **Core vs valence.** If an orbit is in both, core wins and a warning is printed
  (`ModelSpace.cc:367-372, 399-402`).
* **Particle counts.** `Aref` and `Zref` are Σ(2j+1)·occ over holes (protons only for
  Z). `Acore` and `Zcore` are Σ(2j+1) over core orbits with no occupation weight
  (`ModelSpace.cc:433-484`). Init sets `target_mass` and `target_Z` from Aref and Zref,
  truncating to int (`ModelSpace.cc:418-419`).
* **`e_fermi[tz2]`** is in oscillator-shell units. It is the largest `e` among holes; if
  that orbit is full (occ>0.99) it is moved halfway to the lowest particle `e`
  (`ModelSpace.cc:1052-1087`).
* **Other entry points.**
  * `Init(emax, vector<string> holes, core, valence)` (`ModelSpace.cc:295-317`).
  * `Init_occ_from_file`, which reads lines `n l j2 tz occ` (`ModelSpace.cc:319-355`).
  * `InitSingleSpecies`, which uses tz=-1 only (`ModelSpace.cc:424-431`).
  * `SetReference(vector|set|map|string)` (`ModelSpace.cc:773-851`). The `map` version
    moves the new holes into the core, except valence orbits.

---

## 2. Two-body channels and kets

### 2.1 Kets (`ModelSpace.hh:63-83`, `ModelSpace.cc:1166-1236`)

* `Ket{Orbit* op, oq; size_t p, q; int dpq; int phase_prefactor}`. The convention is
  **p <= q**.
* Global index `Index2(p,q) = p(2N-1-p)/2 + q` with `N = norbits`. This orders kets
  lexicographically and includes `p==q`. `Kets` has `N(N+1)/2` entries. Calling
  `Index2` with `p>q` returns garbage without any check.
* Exchange phase: `Phase(J) = (-1)^((j2p+j2q)/2 + 1 + J) = -(-1)^(jp+jq-J)`, so
  `|pq;J> = Phase(J)·|qp;J>`. The fermionic sign is included (`ModelSpace.hh:77-80`).

### 2.2 Channel labels and enumeration

* Labels: `(J, parity, Tz)` with `parity = (l_p+l_q)%2` and `Tz = (tz2_p+tz2_q)/2`.
* Index: `ch = 6J + 2(Tz+1) + parity`. The order is J-major, then Tz=-1,0,+1, then
  parity 0,1 (`ModelSpace.cc:1104-1127`). Single species uses `ch = 2J + parity`.
* All `6·(TwoBodyJmax+1)` channels are instantiated, including empty ones
  (`ModelSpace.cc:1238-1249`).
* `SortedTwoBodyChannels(_CC)` lists channel indices by descending ket count with empty
  channels dropped. It exists for OpenMP load balancing (`ModelSpace.cc:1251-1258`).
* Membership (`TwoBodyChannel::CheckChannel_ket`, `TwoBodyChannel.cc:143-152`):
  * Pauli: `p==q` requires even J.
  * Parity and Tz must match.
  * Triangle(jp, jq, J).
* **Cross-coupled (Pandya) channels** `TwoBodyChannel_CC` (`TwoBodyChannel.cc:181-189`)
  use the same index layout but redefine Tz as **`|tz2_p − tz2_q|/2` ∈ {0 (pp/nn), 1 (pn)}**.
  There is no Pauli rule. The slots with Tz=-1 exist but are always empty.

Sizes (scalar operator, two species, J-coupled):

| emax | norbits | kets | non-empty channels | largest channel | one scalar 2B op (GB) |
|---|---|---|---|---|---|
| 8 | 90 | 4,095 | 103 | 648 | 0.052 |
| 10 | 132 | 8,778 | 127 | 1,341 | 0.27 |
| 12 | 182 | 16,653 | 151 | 2,483 | 1.10 |
| 14 | 240 | 28,920 | 175 | 4,234 | 3.67 |

### 2.3 Channel-local indexing (`TwoBodyChannel.cc:19-117`)

* `KetList` maps local to global. `KetMap` maps global to local, with -1 for absent;
  it is a `vector<int>` of size `#kets` in *every* normal and CC channel.
  `NumberKets` is the channel dimension.
* Local order is ascending global order, i.e. lexicographic `(p,q)`.
* `GetLocalIndex(p,q)` (`TwoBodyChannel.cc:70-76`) has two meanings:
  * `p<=q`: returns `KetMap[Index2(p,q)]`, which is `(size_t)-1` when the ket is absent.
  * `p>q`: returns `KetMap[Index2(q,p)] + NumberKets`. This "doubled" index serves
    Pandya matrices of shape `nKets × 2nKets` (`Commutator.cc:1917, 1734`).
    **An absent ket in this branch returns `NumberKets-1`, which looks like a valid index.**
* The constructor `TwoBodyChannel(int N, ModelSpace*)` never sets `modelspace`, so
  `Initialize()` would dereference a null pointer (`TwoBodyChannel.cc:134-139`). It is unused.

### 2.4 Per-channel subsets

The global lists `ModelSpace::KetIndex_*` are `vector<index_t>` (`ModelSpace.cc:1209-1234`).
Per channel they become `arma::uvec` of local indices, obtained by sorted intersection
(`TwoBodyChannel.cc:34-42, 107-117`).

| list | rule |
|---|---|
| `pp` | both `occ < OCC_CUT` |
| `hh` | both `occ > OCC_CUT` |
| `ph` | exactly one `occ > OCC_CUT` |
| `cc` | cvq sum 0 |
| `vc` | cvq sum 1 |
| `qc` | \|cvq diff\| = 2 |
| `vv` | cvq product 1 |
| `qv` | cvq sum 3 |
| `qq` | cvq sum 4 |

* `Ket_occ_hh = n_p n_q` and `Ket_unocc_hh = (1−n_p)(1−n_q)` are aligned with
  `KetIndex_hh`; the ph versions work the same way (`TwoBodyChannel.cc:43-66`). They
  carry fractional occupations into the commutators.
* Edge case: an orbit with `occ == 1e-6` exactly is a hole in `AddOrbit` (`<` test) but
  belongs to neither list here (`>` test).
* `MonopoleKets[Tz+1][parity]` maps a global ket to its index in a J-independent
  (Tz, parity) block (`ModelSpace.cc:1199-1204`). Only HartreeFock uses it.

### 2.5 Normalization, phases, hermiticity, reduced matrix elements

* **Storage.** Stored values are normalized: `V^J_{pq,rs} = <pq;J|V|rs;J>` with
  `|pq;J>` carrying `1/sqrt(1+δ_pq)` (`TwoBodyME.hh:36-41`).
* **Getters.** `GetTBME*` without `_norm` returns
  `Ṽ = sqrt((1+δ_pq)(1+δ_rs))·V` (`TwoBodyME.cc:170-176`), the quantity used in the flow
  equations.
* **Matrix-product identity.** Normalized storage turns half-sums into plain matrix
  products over channel kets with `a<=b` (`TwoBodyME.hh:42-50`):
  `Z̃_ijkl = ½ Σ_ab X̃_ijab Ỹ_abkl  ⇔  Z_ijkl = Σ_{a<=b} X_ijab Y_abkl`.
* **Hermiticity flip for blocks with `ch_bra > ch_ket`.** The formula is
  `<b,J_b||O||k,J_k> = h·(-1)^(J_b−J_k)·<k||O||b>`. `h=+1` if hermitian and `−1`
  otherwise, so a non-hermitian operator is treated as anti-hermitian here
  (`TwoBodyME.cc:198-202`, `Operator.cc:322-335`).
  * One-body (rank_J>0): `O_ji = h·(-1)^((j2_i−j2_j)/2)·O_ij` (`Operator.cc:306-315`).
  * Three-body: `h·(-1)^((2J_b−2J_k)/2)` (`ThreeBodyStorage_pn.cc:431-436`).
* **Reduced convention (Wigner–Eckart).**
  `<JM|T^j_m|J'M'> = <J'M' j m|JM>/sqrt(2J+1)·<J||T^j||J'>` (`Operator.cc:1042-1049`).
  * Scalar operators (rank_J = rank_T = parity = 0) are non-reduced by default; any other
    operator is reduced by default (`Operator.cc:74`, `TwoBodyME.cc:31`,
    `ThreeBodyStorage.cc:26`).
  * `MakeReduced` multiplies one-body row a by `sqrt(2j_a+1)`, each 2B block by
    `sqrt(2J+1)` and each 3B entry by `sqrt(2J+1)` (with `2J = twoJ`). It is refused
    when rank_J>0 (`Operator.cc:1051-1107`).
* **`rank_T`** means `|ΔTz|` for allocation and access (`TwoBodyME.cc:107`).

### 2.6 Three-body kets and channels (ModelSpace side)

* `Ket3{p,q,r; Orbit* op,oq,oR; Jpq}` (`ModelSpace.hh:87-108`).
* `Setup3bKets` (`ModelSpace.cc:1269-1310`) builds kets with:
  * `p<=q<=r` drawn from `orbits_3body_space_`;
  * `e_p+e_q+e_r <= E3max`;
  * `Jpq ∈ [|jp−jq|, jp+jq]`;
  * `p==q` excluded for odd Jpq, and `p==q==r` excluded when j=1/2;
  * no Pauli rule for `q==r`.
* `Ket3IndexHash = (p<<24)+(q<<16)+(r<<8)+Jpq` needs p,q,r,Jpq < 256. With 256 or more
  3-body orbits (`emax_3body >= 15`) keys **collide silently**. The comment says
  "p>=q>=r" but the code builds p<=q<=r (`ModelSpace.cc:1333-1339`).
* **Channels** `(twoJ = 1,3,…,6·emax_3body+1; parity; twoTz ∈ {−3,−1,1,3})` are all
  instantiated, including empty ones, in `std::set` order (twoJ-major, then parity, then
  twoTz) (`ModelSpace.cc:1312-1330`).
  * `ThreeBodyChannelHash = 4(twoJ−1)+(twoTz+3)+parity` feeds a `std::map` lookup
    (`ModelSpace.cc:1360-1364`).
  * `GetThreeBodyChannelIndex` returns `size_t(-1)` when the channel is missing
    (`ModelSpace.cc:1352-1358`).
* `ThreeBodyChannel::CheckChannel_ket` (`ThreeBodyChannel.cc:53-68`) tests parity, Tz,
  Triangle(2Jpq, j2_r, twoJ) and some Pauli-type exclusions. The exclusions remove
  stretched states with repeated orbits and `p==q==r` with `twoJ > 3j2−3`, and are marked
  "TODO … only for testing". **The same rules are coded a second time** in
  `ThreeBodyStorage::GetKetIndex_withRecoupling` (`ThreeBodyStorage.cc:261-267`), and the
  two copies must agree.
* `KetMap` grows lazily to `2·iket+1` (`ThreeBodyChannel.cc:26`). `GetLocalIndex` calls
  `.at()` twice and throws `std::out_of_range` for unsorted or missing kets
  (`ThreeBodyChannel.cc:49`, `ModelSpace.hh:433`).
* Every `Init` rebuilds `Kets3` and all 3-body channels, even for IMSRG(2) runs.
  `ThreeBodyChannel::Initialize` scans all of `Kets3` once per channel.
* `|Kets3|` at E3max=14 is 79,869 (emax 6), 132,445 (emax 8) and 158,797 (emax 12).

---

## 3. Operators

### 3.1 Members and invariants (`Operator.hh:45-68`)

| member | meaning / invariant |
|---|---|
| `ModelSpace* modelspace` | non-owning; every channel and index lookup goes through it |
| `double ZeroBody` | left uninitialized by the default constructor |
| `arma::mat OneBody` | `norb × norb` for every rank. Becomes `norb × 1` for dagger (odd-leg) operators |
| `TwoBodyME TwoBody` | allocated iff `particle_rank >= 2` |
| `ThreeBodyME ThreeBody` | always present, created in **isospin** mode, allocated only on request |
| `ThreeLegME ThreeLeg` | **allocated by the ModelSpace constructors** even for number-conserving operators |
| `rank_J, rank_T, parity` | spherical rank, `|ΔTz|`, and parity (0 even, 1 odd) |
| `particle_rank`, `legs` | `legs = 2·particle_rank` for number-conserving operators. Odd legs mean a dagger operator (`IsNumberConserving()` is `legs%2==0`) |
| `hermitian`, `antihermitian` | both false means non-hermitian. Pushed to TwoBody; the 3B part supports only ±1 |
| `is_reduced` | duplicated in `TwoBodyME::is_reduced` and `ThreeBodyStorage::is_reduced` (three flags that can diverge) |
| `nChannels` | copy of the ModelSpace channel count |
| `OneBodyChannels`, `OneBodyChannels_vec` | per-operator selection rules, from `SetUpOneBodyChannels` (`Operator.cc:251-290`). The flat index is `l*4 + (j2+1−2l) + (tz2+1)/2` (`Operator.cc:295-299`) |
| `Q_space_orbit` | for dagger operators, the orbit whose (l,j,tz) the operator carries. Default `-1` |
| `profiler` | handle to the global static profiler |

**State machine (`SetNumberLegs`, `Operator.cc:1008-1039`).**

* Even legs:
  * TwoBody is (re)allocated if `old_legs<4`.
  * ThreeLeg is deallocated (only when legs actually changes).
  * ThreeBody is allocated in its *current* mode when `legs>5`. That is isospin/float
    unless `SetMode("pn")` ran first.
  * ThreeBody is deallocated when going from 6 to 4 legs.
* Odd legs: TwoBody and ThreeBody are deallocated, OneBody is resized to `norb × 1` and
  zeroed, and ThreeLeg is allocated.

`GetParticleRank()` calls `exit` when the allocation state contradicts `particle_rank`
(`Operator.cc:994-1006`).

**Tensor 2B layout.** `std::map<array<size_t,2>, arma::mat>` holds only `ch_bra <= ch_ket`.
A block is allocated iff Triangle(J_b, J_k, rank_J), `|Tz_b−Tz_k| == rank_T` and the parity
matches (`TwoBodyME.cc:94-113`). Each block is a dense `nkets_bra × nkets_ket` matrix.
Scalar operators have only diagonal blocks, stored with both triangles.

### 3.2 Function catalog (`Operator.hh` / `Operator.cc`)

| group | function | what it does |
|---|---|---|
| construct | `Operator()` | empty; `modelspace=NULL`, ZeroBody uninitialized (57) |
| | `Operator(ms)` | scalar 2-body, hermitian, non-reduced (85) |
| | `Operator(ms,J,T,p,rank)` | general; reduced iff non-scalar; 3B not allocated (66) |
| | copy, move, `=` | the "move" versions deep-copy 2B and 3B (98-125) |
| arithmetic | `*= * / /=`, `operator*(double, Op)` | scale every part; 3B only if `particle_rank>2`; ThreeLeg always (128-170) |
| | `+= + -= -` (Operator) | partwise. `+=` exits on a reduced/non-reduced mismatch and `-=` does not check. Result rank is the max of the two; ThreeLeg is added only if the rhs has odd legs (173-243) |
| | `+= -=` (double), unary `-` | ZeroBody only / multiply by −1 |
| element access | `GetOneBody`, `SetOneBody` | `SetOneBody` also writes the conjugate with the hermiticity phase (306) |
| | `GetTwoBody(chb,chk,i,j)`, `SetTwoBody(J1,p1,T1,J2,p2,T2,i,j,k,l,v)`, `GetTwoBodyDimension` | thin wrappers over TwoBodyME (317-335) |
| flags | `Set/IsHermitian`, `Set/IsAntiHermitian`, `Set/IsNonHermitian` | `SetNonHermitian` leaves ThreeBody untouched (964-985) |
| | `GetJRank/GetTRank/GetParity/GetNumberLegs/GetParticleRank` | getters |
| | `SetParticleRank`, `SetNumberLegs`, `Set/GetQSpaceOrbit` | state changes described above |
| | `MakeReduced/MakeNotReduced/IsReduced` | sqrt(2J+1) conversions; refused for rank_J>0 (1051-1107) |
| | `MakeNormalized/MakeUnNormalized/ChangeNormalization(f)` | scale rows and columns of kets with p==q by 1/√2 or √2. **Row and column are swapped** (bra index scales `.col`), wrong for rectangular tensor blocks; uses the 13-digit `INVSQRT2` (1712-1744) |
| erase / scale | `Erase`, `EraseZero/One/Two/ThreeBody`, `EraseThreeLeg`, `ScaleZero/One/TwoBody` | zero or scale parts |
| | `Symmetrize`, `AntiSymmetrize` | copy the upper triangle to the lower (with phase for rank_J>0) for 1B and 2B; 2B skips non-scalar (1833-1884) |
| normal ordering | `DoNormalOrdering` | w.r.t. `holes`; dispatches to the 2B, 3B or dagger version (380) |
| | `UndoNormalOrdering`, `UndoNormalOrderingCore` | the same with sign −1 (vacuum NO) (390-408) |
| | `DoNormalOrderingCore`, `DoNormalOrderingFilledValence` | NO w.r.t. `core` / `valence` (448-467) |
| | `ReNormalOrderCore(copy_3body)` | sign −1 over holes∖core; optionally keeps the 3B part (413-445) |
| | `DoNormalOrdering2/3/Dagger(sign, occupied)` | implementations; formulas in §3.3 |
| transforms | `Truncate(ms_new)` | copy to a smaller emax by mapping orbits. 3B handles only "pn" plus an `"iso"` branch that **never fires** because the mode is called "isospin"; diagonal 3B channel blocks only (721-831) |
| | `DoIsospinAveraging()` | average p/n one-body; rebuild pp, nn, pn 2B from T=0/1 combinations (834-916) |
| norms / traces | `Norm, OneBodyNorm, TwoBodyNorm, ThreeBodyNorm, OneLegNorm, ThreeLegNorm` | §3.4 |
| | `Trace(A,Z)` | trace over A-body space after vacuum NO (1746-1793) |
| MBPT (misplaced) | `GetMP2_Energy, GetMP3_Energy, GetMP2_3BEnergy, GetPPHH_Ladders, MP1_Eval` | Møller–Plesset style corrections from diagonal OneBody (1185-1637) |
| misc | `ScaleFermiDirac` | unused heuristic; the 2B branch multiplies a row once per ket (bug-like) (1795-1831) |
| | `SetUpOneBodyChannels`, `GetOneBodyChannel` | selection-rule tables |
| | `Size()` | bytes of 0B+1B+2B+3B, ThreeLeg excluded (301) |
| | `WriteBinary/ReadBinary` | raw dump: flags, 0B, 1B, then 2B and 3B if allocated. No version tag; does not store `is_reduced` or `Q_space_orbit` (337-376) |
| | `Print*`, `PrintTimes`, `Get/SetModelSpace` | debugging |

### 3.3 Normal-ordering formulas (needed for numerical reproduction)

`sign` is +1 (NO) or −1 (undo). It multiplies each occupation once.

* **2B, scalar non-reduced** (`Operator.cc:476-571`):
  * `E0 += Σ_k (j2_k+1)·sign·n_k·O1_kk + Σ_{k<=l} Σ_J (2J+1)·n_k n_l·V^J_norm(kl,kl)`.
  * `f_ab += (2J+1)/(2j_a+1)·sign·n_h·Ṽ^J(ah,bh)`, summed over stored blocks and over
    `b` in `OneBodyChannels`. The conjugate is set as `f_ba = h·f_ab`.
* **2B, reduced (tensor)**:
  * `f_ab += sqrt((2J_b+1)(2J_k+1))·sign·n_h·(-1)^(j_a+j_h−J_k−rank_J)·{J_b J_k rank_J; j_b j_a j_h}·Ṽ(ah,bh)`.
  * The value is doubled when `a==b` and `J_b≠J_k` (bug fix of May 2025, `Operator.cc:555-558`).
  * `f_ba = h·(-1)^(j_a−j_b)·f_ab` is written **even when h=0**, so non-hermitian reduced
    operators have their transposed and diagonal entries zeroed (`Operator.cc:561`).
  * The 0-body term assumes non-reduced matrix elements and is only added for scalars.
* **3B** (`Operator.cc:582-674`): rank_J>0 calls `exit`; a reduced input is converted to
  non-reduced and back.
  * `Γ^J_ijkl = [Σ_a sign·n_a·W^{no2b}_{ija,kla}(J)] / [(2J+1)·sqrt((1+δ_ij)(1+δ_kl))]`.
  * `W^{no2b}(J) = Σ_{twoJ} (twoJ+1)·W^{J J twoJ}`; the sum skips terms whose `(ija)` or
    `(kla)` exceed E3max (`ThreeBodyStorage.cc:371-393`).
  * The 0-body and 1-body results from `NO2(Γ)` are scaled by **1/3 and 1/2**, then
    `NO2` of the original operator is added. The 3B part is carried along unchanged, and
    the result's 3B is switched with `SetMode("pn")`. If the 3B input was no2b, this
    **allocates an empty full pn 3B** (possibly many GB).
  * The parallel loop relies on 6j values having been precalculated.
* **Dagger** (`Operator.cc:679-714`):
  `O1(a,0) −= (2J+1)/(2j_a+1)·sign·n_h·ThreeLeg(ch; h,a, h)`, marked "TODO: confirm sign".

### 3.4 Norm definitions (they set flow and BCH thresholds, so replicate exactly)

* `Norm = sqrt(n1² + n2² + n3²)`. Dagger operators use `sqrt(OneLeg² + ThreeLeg²)`
  (`Operator.cc:1644-1664`).
* `n1² = Σ_p Σ_{q∈ch(p)} O_pq² · (j2_p+1)·[(min(j2_q, j2_p+rank_J) − max(−j2_q, j2_p−rank_J))/2 + 1]`.
  For scalars this is `(j2_p+1)` weighting. The formula mixes `j2` with `rank_J` units
  (`Operator.cc:1666-1682`).
* `n2² = Σ_blocks w²·‖M‖_F²`, with `w = 2J_ket+1` (non-reduced) or `1/(2·rank_J+1)`
  (reduced). Off-diagonal blocks count twice. Note the weight is **(2J+1)², not (2J+1)**
  (`TwoBodyME.cc:781-803`).
* `n3` is the plain Frobenius norm of the stored array. For pn each pair of a diagonal
  block is counted once; iso values are floats (`ThreeBodyStorage_pn.cc:341`,
  `ThreeBodyStorage_iso.cc:320`).
* `OneLegNorm` uses the weight `(j2+1)²` (`Operator.cc:1694-1705`).

---

## 4. TwoBodyME (`TwoBodyME.hh:51-184`, `TwoBodyME.cc`)

**Layout.** `MatEl: std::map<{ch_bra,ch_ket}, arma::mat>` with `ch_bra <= ch_ket`
(column-major armadillo). The class also keeps `nChannels`, `hermitian`, `antihermitian`,
`allocated`, `rank_J`, `rank_T`, `parity` and `is_reduced`. `Allocate()` rebuilds every
block as zeros. `GetMatrix(chb,chk)` uses `map::at` and throws for `ch_bra > ch_ket`.

**Accessor families.** Each comes in channel+orbit, channel+`Ket&`, `(J,p,Tz)×2`+orbit,
`_J(Jbra,Jket,a,b,c,d)` and scalar single-channel overloads.

| family | semantics |
|---|---|
| `GetTBME*` | unnormalized Ṽ (×√2 for each repeated pair) |
| `GetTBME*_norm`, `GetTBME_norm(chb,chk,i,j)` | stored normalized value. The index version is raw: no flip, no checks |
| `GetTBME_J`, `GetTBME_J_norm` | infer channels from the orbits. Return 0 on a parity, `|ΔTz|≠rank_T` or triangle failure (`TwoBodyME.cc:367-426`) |
| `SetTBME` (orbit form) | writes `phase·v`; applies the swap phase for `a>b` or `c>d`. For `ch_bra>ch_ket` it writes the transposed block with `(-1)^(J_b−J_k)` and an extra −1 if anti-hermitian. For scalar blocks it also writes the conjugate (`TwoBodyME.cc:206-231`) |
| `AddToTBME` (orbit form) | like Set but **without the anti-hermitian −1** in the flipped case. The conjugate is skipped when `bra==ket` (`TwoBodyME.cc:234-260`) |
| `SetTBME/AddToTBME(chb,chk,i,j)` | index forms; Set does no channel flip, Add flips with `(-1)^ΔJ` only (`TwoBodyME.cc:299-324`) |
| `AddToTBMENonHerm*` | write one element; `NonNormalized` divides by √2 for repeated orbits (`TwoBodyME.cc:262-279, 326-329`) |
| `GetTBME_J(_norm)_twoOps` | fetch the same element from two operators with one lookup (Pandya use) (`TwoBodyME.cc:430-488`) |
| `GetTBMEmonopole(_norm)` | `Σ_J (2J+1) V^J_abcd / ((2j_a+1)(2j_b+1))`; the non-norm version adds the √2 factors (`TwoBodyME.cc:693-760`) |

**Out-of-range handling.**

* Getters on unallocated storage return 0.
* A ket that is not in the channel makes `GetLocalIndex` return `(size_t)-1`, and the
  getter returns 0 (`TwoBodyME.cc:187-190`).
* Setters do **not** check. An absent ket gives index −1, which leads to
  `KetList[-1]` (undefined behaviour) or an armadillo bounds exception. Bounds checking
  is on because `ARMA_NO_DEBUG` is not defined.
* `SetTBME` silently returns when storage is unallocated.
* No m-scheme accessors exist.

**Isospin/pn conversion** (`TwoBodyME.cc:604-651`). Both functions reduce labels to the
proton (even) index and use `index+1` for the neutron.

* `Set_pn_TBME_from_iso(J,T,tz,a,b,c,d,V)` *accumulates* `½·sqrt((1+δ_ab)(1+δ_cd))·V` into
  pn·pn and np·np. The pn·np and np·pn elements get the phase `(2T−1)`. It never sets pp
  or nn; callers must.
* `Get_iso_TBME_from_pn` is the inverse.

**Other.**

* `Erase`, `Scale`, `Eye`.
* `Symmetrize` (scalar only, `symmatu`) and `AntiSymmetrize` (skips rank_J>0) (`TwoBodyME.cc:808-830`).
* `Dimension()` returns Σ N(N+1)/2. `size()` returns bytes.
* Arithmetic:
  * `+=` loops over the lhs blocks and calls `rhs.GetMatrix`, which throws if the block
    structures differ.
  * `+=` with an unallocated lhs copies the rhs.
  * `-=` loops over the rhs blocks (`TwoBodyME.cc:53-90`).
* `Write/ReadBinary` dump the flags and then the matrices in map order. They do not store
  `is_reduced` (`TwoBodyME.cc:908-934`).
* Dead code: the constructor `TwoBodyME(TwoBodyME_ph&)` is declared but never defined
  (`TwoBodyME.hh:28,68`).

---

## 5. ThreeBodyME and its storage back-ends

### 5.1 Facade (`ThreeBodyME.hh:45-180`, `ThreeBodyME.cc`)

* `ThreeBodyME` owns `std::unique_ptr<ThreeBodyStorage>` and a `ModelSpace*`. Its members
  `E3max`, `emax`, `herm (±1)`, `rank_J/T`, `parity` and `ISOSPIN_BLOCK_DIMENSION` are
  mostly vestigial. Allocation always takes E3max from the ModelSpace
  (`ThreeBodyStorage_iso.cc:84`, pn uses ModelSpace channels).
* Element type: `typedef double ThreeBME_type` (`ThreeBodyME.hh:34`).
* The default mode is **isospin** (`ThreeBodyME.cc:23-52`).
* Copy and `operator=` deep-clone the storage. There are no move operations.
* `+=` adds when both sides are allocated, or clones the rhs when the lhs is empty. `-=`
  does the same with a factor −1 (`ThreeBodyME.cc:94-119`).
* **Mode switching.**
  * `SetMode(m)` with `m ∈ {"isospin","pn","no2b","no2bhalf","mono"}` replaces the
    storage, **discarding the data**, and allocates immediately. It does nothing if the
    mode is already active.
  * An unknown name prints a warning and then calls `Allocate()` on the existing storage,
    which wipes it (`ThreeBodyME.cc:365-413`).
  * `no2bhalf` reports `GetStorageMode()=="no2b"`.
  * `SwitchToPN_and_discard()` gives a fresh, allocated pn storage (`ThreeBodyME.cc:351-363`).
  * **`TransformToPN()` is broken.** It fills a new pn storage in parallel and then does
    `*threebody_storage = *TBS_pn` through the base reference
    (`ThreeBodyME.cc:336`). That copies only the base members (`ch_start`, `ch_dim`,
    flags) into the existing isospin object. The mode stays "isospin", the iso data is
    unchanged, and the work is thrown away. It runs from `imsrg++.cc:496-499` when
    `store_3bme_pn=true` (default false).
* `SetME_pn` and `AddToME_pn` with abcdef arguments are not implemented by any back-end.
  When allocated, `NotImplemented` prints and exits; when unallocated it returns 0
  (`ThreeBodyStorage.cc:43-51`). Only the `_pn_ch` setters exist (pn mode).

### 5.2 Base class machinery (`ThreeBodyStorage.hh:71-202`, `ThreeBodyStorage.cc`)

* `enum Permutation {ABC,BCA,CAB,ACB,BAC,CBA}`.
* `SortOrbits` sorts to **a<=b<=c** (base and pn) and returns the permutation
  (`ThreeBodyStorage.cc:66-85`). Its comments say a>=b>=c. The isospin back-end
  overrides it with **descending** order on even indices (`ThreeBodyStorage_iso.cc:421-446`).
* `PermutationPhase`: +1 for ABC, BCA, CAB and −1 for the odd permutations
  (`ThreeBodyStorage.cc:123-126`).
* `RecouplingCoefficient(case, ja,jb,jc, Jab_in, Jab, twoJ)` (`ThreeBodyStorage.cc:97-120`).
  The angular factor is applied on top of `PermutationPhase`, and `J = twoJ/2`:

  ```
  ABC: δ(Jab,Jab_in)                 BAC: δ·(-1)^(ja+jb-Jab)
  BCA: (-1)^(jb+jc+Jab_in+1)  ·ĴĴ'·{ja jb Jab; jc J Jab_in}
  CAB: (-1)^(ja+jb-Jab+1)     ·ĴĴ'·{jb ja Jab; jc J Jab_in}
  ACB: (-1)^(jb+jc+Jab_in-Jab)·ĴĴ'·{jb ja Jab; jc J Jab_in}
  CBA: -1                     ·ĴĴ'·{ja jb Jab; jc J Jab_in}     (ĴĴ' = sqrt((2Jab_in+1)(2Jab+1)))
  ```

  It calls `modelspace->GetSixJ`, so the static 6j cache must already hold the value
  when it runs in parallel.
* `IsKetValid`: emax_3body per orbit, E3max, and the Jab and twoJ coupling ranges.
  `IsKetInEMaxTruncations` and `IsOrbitIn3BodyEMaxTruncation` are helpers
  (`ThreeBodyStorage.cc:160-234`).
* `GetKetIndex_withRecoupling` (`ThreeBodyStorage.cc:236-291`) is the **core 3B lookup**:
  sort, find the channel by (twoJ, parity, twoTz), apply the Pauli filters, then for each
  allowed Jab look up `Tbc.GetLocalIndex(a,b,c,Jab)` and compute the recoupling
  coefficient (dropped if `|c|<1e-10`).
* `GetME_pn_TwoOps` reads the same element from two operators. It assumes a shared pn
  layout and uses `thread_local` scratch vectors (`ThreeBodyStorage.cc:296-339`).
* Default `GetME_pn_no2b(a,b,c,d,e,f,J2b) = Σ_twoJ (twoJ+1)·GetME_pn(J2b,J2b,twoJ,…)`.
  The reduced branch reads `sqrt(2*J2b+1 / (twoJ+1.))`, a **precedence bug** (the iso
  twin `GetME_iso_no2b` is correct, `ThreeBodyStorage.cc:344-393`).
  `GetME_pn_mono = Σ_{J2b} GetME_pn_no2b` (`ThreeBodyStorage.cc:412-427`).
* `ch_start` and `ch_dim` sit in the base class: an `unordered_map` from {ch_bra, ch_ket}
  to an offset, using a boost-style `hash_combine` defined in an anonymous namespace in
  the header (`ThreeBodyStorage.hh:37-59`).

### 5.3 `pn` mode (`ThreeBodyStorage_pn.*`), double precision

* **Layout.** One flat `vector<double>`. The blocks are `(ch_bra <= ch_ket)` over the
  ModelSpace `ThreeBodyChannels`.
  * Allocation condition: twoJ triangle with `rank_J`, parity, and
    `|ΔtwoTz| == 2·rank_T`.
  * Diagonal blocks store the upper triangle row-major:
    `start + (2D − i1 − 1)·i1/2 + i2` with `i1=min, i2=max`.
  * Off-diagonal blocks are full row-major: `start + D2·i1 + i2`
    (`ThreeBodyStorage_pn.cc:75-124, 402-445`).
* **Hermiticity.** `herm_flip = 1` when `ch_bra<ch_ket` or (same channel and
  `ibra >= iket`), otherwise `herm`. Reading `ch_bra>ch_ket` multiplies by
  `(-1)^((2J_b−2J_k)/2)`. With `herm==−1` the diagonal elements are forced to 0.
* **Access.**
  * `GetME_pn(Jab,Jde,twoJ,abcdef)` = `IsKetValid` → recoupled bra and ket index lists →
    `SumRecoupled` (`ThreeBodyStorage_pn.cc:135-157, 238-300`). There is a tensor overload
    with separate `j0, j1`.
  * `GetME_pn_ch`, `SetME_pn_ch` and `AddToME_pn_ch` work on channel-local indices.
  * `GetME_iso` projects back from pn with CGs at fixed `twoTz=1`
    (`ThreeBodyStorage_pn.cc:188-230`).
* `MakeReduced` and `MakeNotReduced` scale by `sqrt(twoJ_bra+1)`.
* **Defects.**
  * `ch_dim` is appended in `Allocate` without being cleared, so `Allocate` is not
    idempotent.
  * `number_allocated` is a non-atomic static counter.
  * `WriteBinary` writes `total_dimension` **bytes** (`ThreeBodyStorage_pn.cc:372`).
* **Use.** Every 3B operator produced during IMSRG(3) and every commutator output.

### 5.4 `isospin` mode (`ThreeBodyStorage_iso.*`), the "full" input format, **float**

* **Keys.** Isospin orbits are the even (proton) indices.
  `KeyHash(a,b,c,d,e,f)` packs `index/2` into 10 bits each, in the order
  `a, d, b, e, c, f` (at most 1024 isospin orbits) (`ThreeBodyStorage_iso.cc:384-393`).
* **Stored set.** `a>=b>=c`, `d>=e>=f` and `(def) <= (abc)` lexicographically, with
  `e_a <= emax_3b`, `E_abc` and `E_def <= E3max`, and parity conserved. There is no Tz
  label (`ThreeBodyStorage_iso.cc:80-165`).
* **Block layout.** For each orbit sextuple the block loops over `Jab → Jde → twoJ`, and
  each `(Jab, Jde, twoJ)` has `ISOSPIN_BLOCK_DIMENSION` entries: 5 for rank_T ∈ {0,2},
  9 for 1, 1 for 3.
  * rank_T=0 isospin index: `2·tab + tde + twoT/2`, giving
    (0,0,½)→0, (0,1,½)→1, (1,0,½)→2, (1,1,½)→3, (1,1,3/2)→4.
  * The index maps for rank_T=1,2,3 are listed at `ThreeBodyStorage_iso.cc:565-577`.
* **AccessME** (`ThreeBodyStorage_iso.cc:462-601`) sorts bra and ket, swaps them if
  needed (factor `herm·(-1)^((2T_abc−2T_def)/2)`), and builds a list of
  (index, coefficient) pairs. The coefficient is a J recoupling times an isospin
  recoupling, both through `RecouplingCoefficient` with `j=½` for isospin.
* **Getters.**
  * `GetME_iso` returns 0 when two isospin labels match and `J+T` is even.
  * `GetME_pn` sums over `tab, tde, T_abc, T_def` using four **GSL CG calls per term**
    (`ThreeBodyStorage_iso.cc:207-256`). For rank_T>0 it also applies the isospin
    Wigner–Eckart factor `CG(T_def … rank_T ΔTz | T_abc)/sqrt(2T_abc+1)`.
* **`SetME_iso`** adds `(V − current)·c_i` to every element in the list. That reproduces
  `V` only when `Σ c_i² = 1` (`ThreeBodyStorage_iso.cc:264-270`).
* **Defects.**
  * `Subtract` adds (`ThreeBodyStorage_iso.cc:53`).
  * The copy constructor sets `total_dimension=0`, so a clone's `size()` and
    `WriteBinary` see 0.
  * `WriteBinary` writes too few bytes (`ThreeBodyStorage_iso.cc:352`).
  * Allocate prints unconditionally.

### 5.5 `no2b` / `no2bhalf` (`ThreeBodyStorage_no2b.*`), float or IEEE half

* **Meaning.** Stores only `<(ab)J2 Tab, c | V | (de)J2 Tde, f>_{T3}` summed over total J
  with weight `(2J+1)`, for `j_c = j_f` and equal `l` parity of c and f. This is exactly
  what NO2B normal ordering needs.
* **Own basis.** `ThreeBodySpaceNO2B` keeps its own isospin orbit list ordered `e↑, l↑, j↑`
  (**different from the ModelSpace order**), mapped through `nlj2idx`
  (`ThreeBodyStorage_no2b.cc:93-147`).
* **Channels** `(J2, P2, J1=j2_c, P1, T3)`. The states `(a<=b, c, Tab)` are subject to
  cuts on emax, pairwise E2max and E3max, and the Pauli rule `(J2+Tab)` even with
  `a==b` excluded. The swapped ordering `ba` has phase `(-1)^(ja+jb+J2+Tab)`
  (`ThreeBodyStorage_no2b.cc:19-67`).
* **Storage.** One `n(n+1)/2` symmetric triangle per channel. Hermitian is assumed and
  `herm` is ignored.
* **Access.** `GetME_pn_no2b` converts pn to isospin with four CG calls per term
  (`ThreeBodyStorage_no2b.cc:362-412`). `GetME_iso_no2b` exits on reduced operators.
* **ReadFile** (`ThreeBodyStorage_no2b.cc:552-810`) reads Miyagi-format files: text
  "me3j", ".gz", or "stream.bin" binary. It runs as an OpenMP region where thread 0
  refills an 8 GB buffer between barriers and the threads take orbits in round-robin.
  An early `break` before a barrier can deadlock. The code checks only
  `IntInputs.size()<2` and then reads `IntInputs[2]`.

### 5.6 `mono` (`ThreeBodyStorage_mono.*`), float

* **Meaning.** Holds the monopole 3B matrix elements for HF: `(a,d)`, `(b,e)` and `(c,f)`
  each share `(l,j)`. Channels are `(j1,j2,j3,l1,l2,l3,twoT)` and states are
  `(a,b,c,Tab)`. There is no antisymmetry handling; all phases are 1, marked
  "TODO check this" (`ThreeBodyStorage_mono.cc:23-85`).
* **Access.** Only `GetME_pn_mono` is available, used by `HartreeFock.cc:361`. Normal
  ordering through `GetME_pn_no2b` would hit `NotImplemented` and exit.

### 5.7 Which mode is used when

| context | mode |
|---|---|
| reading a Darmstadt "full" 3N file | isospin (default), `SetME_iso` (`ReadWrite.cc:1681`) |
| `3bme_type=no2b` | `no2b` or `no2bhalf` + `ReadFile` (`imsrg++.cc:477-485`); normal ordering through `GetME_pn_no2b` (HF, `Operator.cc:654`) |
| `3bme_type=mono` | `mono` + `ReadFile`; HF only |
| IMSRG(3) and commutator outputs | `pn` (`SetMode("pn")` or `SwitchToPN_and_discard`) |

### 5.8 Memory scaling (one operator)

| emax / E3max | pn (double) | isospin (float) |
|---|---|---|
| 4 / 12 | 0.27 GB | 0.049 GB |
| 6 / 12 | 3.5 GB | 0.54 GB |
| 6 / 14 | 9.7 GB | 1.46 GB |
| 8 / 14 | 27.5 GB | — |

pn needs about 6–7× the memory of isospin at equal truncation. The factor comes from
tz-resolved channels, double precision, and storing all `Jab` per `(pqr)`. no2b and mono
grow far more slowly because no total-J or six-index dense blocks are stored.

### 5.9 ThreeLegME (`ThreeLegME.hh/.cc`)

* `map<ch, arma::mat(nkets_ch × norbits)>` holds `<ab;J|O|c>` for dagger operators
  `a†a†a`. It uses the same normalized and phase conventions as TwoBodyME, with
  `GetME` ×√2 for `a==b`.
* `operator+=` throws when the block structures differ.
* The constructor `ThreeLegME(ms)` allocates immediately, which causes the per-operator
  cost noted in §0.7.

---

## 6. AngMom and AngMomCache

### 6.1 Coefficients (`AngMom.hh`, `AngMom.cc`)

| function | algorithm / notes |
|---|---|
| `ThreeJ`, `SixJ`, `NineJ` | GSL `gsl_sf_coupling_{3j,6j,9j}`. Arguments are converted with `int(2*j)`. The default GSL error handler is active (never disabled), so a domain error aborts (`AngMom.cc:108-129`) |
| `CG(ja,ma,jb,mb,J,M)` | `(-1)^(ja−jb+M)·sqrt(2J+1)·3j(ja jb J; ma mb −M)`, Condon–Shortley (`AngMom.cc:114-117`) |
| `NormNineJ`, `TwelveJ_1(_ints)` | normalized 9j; 12j of the first kind as a sum of four 6j |
| `Moshinsky(N,L,n,l,n1,l1,n2,l2,λ,β)` | Buck et al. NPA 600 (1996) algorithm, converted to **Moshinsky's phase** with `(-1)^(l+L+λ)`. `β=π/4` for equal masses; `mosh_beta_2` and `mosh_beta_half` cover mass ratios 2 and ½. Uses the factorial tables (`AngMom.cc:167-241`) |
| `TalmiB`, `Tcoeff`, `Tcoeff_bruteforce` | Talmi B coefficients; lab↔Jacobi 3-body T coefficients (Nogga/Roth conventions with documented corrections) (`AngMom.cc:247-475`) |
| `Triangle(j1,j2,j3)` | uses `round(2·…)` and **does not check that `j1+j2+j3` is an integer** (`AngMom.cc:76-81`) |
| `Jmin/Jmax(initializer_list<{2j,2j}>)` | coupling range in 2j units. An empty list gives Jmax=1000 |
| `phase(x)` | `(-1)^x`, correct for negative x. Callers often pass doubles that convert implicitly to int |
| `factorial`, `double_fact` | lookup in `FactorialList` when filled (`FillFactorialLists(170)`), otherwise computed. Overflows near 171 |

`FactorialList` and `DoubleFactorialList` are declared **`static` in the header**
(`AngMom.hh:41-42`), so every translation unit gets an unused copy. Only the copies in
`AngMom.cc` are filled and read.

### 6.2 Caches

| cache | owner | key | contents and fill |
|---|---|---|---|
| `SixJList` | **static** `ModelSpace` member (`ModelSpace.hh:363`, `ModelSpace.cc:20`) | `SixJHash`: canonicalized by column permutations and upper/lower swaps; 10 bits per 2j (`ModelSpace.cc:1527-1580`) | `PreCalculateSixJ` **clears** the map, inserts every key serially, then computes in a parallel loop. Ranges: `{ja jb J1; jc jd J2}` with b,c,d up to 3-body j (c limited when emax>10), and `{J1 J2 J3; ja jb jc}` (`ModelSpace.cc:1763-1885`) |
| `NineJList` | static | `NineJHash`: canonical permutation (minimum element to the corner), base-91 digits, so 2j <= 90 (`ModelSpace.cc:1594-1655`) | `PreCalculateNineJ` fills only LS–jj `{la lb L; ½ ½ S; ja jb J}`. On a miss `GetNineJ` sums three `GetSixJ` and stores only when single-threaded (`ModelSpace.cc:1887-1939, 2062-2109`) |
| `MoshList` | static | bit-packed (`ModelSpace.cc:1681-1697`) | `PreCalculateMoshinsky` up to E2max. `GetMoshinsky` canonicalizes with symmetry phases; on a miss it **inserts without synchronization**, even in parallel (only a warning is printed) (`ModelSpace.cc:1941-2060`) |
| `T3bList` | static, **declared but never defined** | — | dead; Jacobi3BME uses a local map with the same name |
| `six_j_cache_2b_` (`SixJCache_112112`) | **per-ModelSpace instance**, copied with it | dense 6-index array over `{j1 j2 J3; j4 j5 J6}`, single-particle j and 2-body J (`AngMomCache.hh:9-46`) | Generated serially at construction for `jj_max = 2emax+1`. Size `4(emax+1)^6` doubles: 0.14 GB at emax=12, 0.34 GB at emax=14. Read through `GetCachedSixJ` without bounds checks. A default-constructed ModelSpace has a 4-entry cache, and `Init` does not resize it |

Precalculation flags (`sixj_has_been_precalculated` and the others) are **per instance**,
while the maps are **static**. When a second ModelSpace (for example a smaller IMSRG
space) calls `PreCalculateSixJ`, the shared map is cleared and refilled for *its* emax.
Larger-j symbols needed by the first space are then missing, and a later parallel miss
calls `exit`.

### 6.3 Thread-safety strategy

* `GetSixJ` checks the triangle conditions **only on a miss**, so invalid symbols return
  0 without insertion. A valid miss is stored under `omp critical` only when
  `omp_get_num_threads()<2`; otherwise the program **exits** (`ModelSpace.cc:1699-1744`).
* Commutators therefore run their first call single-threaded, gated by
  `scalar_transform_first_pass`, `scalar3b_transform_first_pass` and
  `tensor_transform_first_pass[40]` (indexed `rank_J*4 + parities`,
  `Commutator.cc:517, 1656`), after which they run in parallel. `ResetFirstPass()`
  re-arms the flags (`ModelSpace.cc:1518-1524`).
* `PreCalculate*` writes to pre-inserted keys from several threads through
  `unordered_map::operator[]`. That works in practice but the standard does not
  guarantee it.

---

## 7. Global state, singletons, OpenMP

**Global or static state.**

* `ModelSpace`: `OCC_CUT=1e-6`, `NOT_AN_ORBIT=99999`, the mutable `NULL_ORBIT`, the
  `ValenceSpaces` table, `SixJList`, `NineJList`, `MoshList`, and the undefined
  `T3bList` (`ModelSpace.cc:14-70`).
* `IMSRGProfiler`: static `std::map<string,double> timer`, `std::map<string,int> counter`
  and `start_time` (`IMSRGProfiler.cc:14-16`). **Every Operator constructor and
  destructor updates `counter["N_Operators"]`** (`Operator.cc:53, 62, 82, 95, 107, 119`).
  `ThreeBodyStorage_pn::Allocate`, `DoNormalOrdering3` and others write `timer[...]`.
  `std::map` is not thread-safe, so creating Operators or allocating storage inside a
  parallel region is a data race. Linux-only `CheckMem` reads `/proc/self/statm`.
* AngMom namespace globals `mosh_beta_*`; per-TU static factorial vectors.
* `ThreeBodyStorage_iso::none_allocated`, `ThreeBodyStorage_pn::none_allocated` and
  `number_allocated` (non-atomic).
* `thread_local` scratch vectors in `ThreeBodyStorage::GetME_pn_TwoOps` and
  `ThreeBodyStorage_pn::GetME_pn`.
* Per-instance mutable algorithm state inside `ModelSpace`: the first-pass flags,
  `PandyaLookup` (lazily filled with no lock, `ModelSpace.cc:2205-2247`), and the
  profiler.

**OpenMP in the in-scope files.**

| location | what runs in parallel |
|---|---|
| `ModelSpace.cc:1871, 1925, 1994` | `PreCalculate*` computation over pre-inserted keys |
| `Operator.cc:622` | `DoNormalOrdering3` over channel blocks (6j values must already be cached) |
| `Operator.cc:1273, 1346, 1453` | MBPT reductions |
| `ThreeBodyME.cc:293` | `TransformToPN` |
| `ThreeBodyStorage_no2b.cc:477, 651` and `ThreeBodyStorage_mono.cc:639, 823` | counting and buffered file reading with barriers |

**Global build flags.** `-frounding-math` is set for the whole build and the standard is
C++14 (`CMakeLists.txt`). Armadillo runs with bounds checks on.

---

## 8. Dependency graph

```
PhysicalConstants.hh        (none)
IMSRGProfiler.{hh,cc}       std, OpenMP, sys/resource
AngMom.{hh,cc}              GSL (gsl_sf_coupling)
AngMomCache.{hh,cc}         AngMom
ModelSpace.hh  (+TwoBodyChannel.cc, ThreeBodyChannel.cc)
                            armadillo, IMSRGProfiler.hh, AngMomCache.hh ; .cc: AngMom, OpenMP
TwoBodyME                   ModelSpace ; .cc: AngMom, PhysicalConstants
ThreeLegME                  ModelSpace, armadillo ; .cc: PhysicalConstants
ThreeBodyStorage(.hh/.cc)   ModelSpace ; .cc: AngMom (also ModelSpace::GetSixJ)
ThreeBodyStorage_pn/_iso    ThreeBodyStorage, AngMom, OpenMP, IMSRGProfiler (via ModelSpace.hh)
ThreeBodyStorage_no2b/_mono ThreeBodyStorage, ModelSpace, boost::iostreams(gzip), half.hpp, OpenMP
ThreeBodyME                 ModelSpace, ThreeBodyStorage ; .cc: all four storage headers, AngMom, OpenMP
Operator                    ModelSpace, TwoBodyME, ThreeBodyME, ThreeLegME, IMSRGProfiler, armadillo ;
                            .cc: AngMom, PhysicalConstants, gsl_math.h (unused), OpenMP
IMSRG.hh                    umbrella over the whole library, no include guard (imsrg++.cc, pyIMSRG.cc)
```

The CMake object libraries mirror this graph. `ThreeBodyME` links PUBLIC to
`ModelSpace`, `IMSRGBoostZip` and `HalfPrecision`, and `Operator` needs the include
directory of `ThreeBodyME/` (`CMakeLists.txt`, `ThreeBodyME/CMakeLists.txt`).

**Circular or awkward couplings.**

* **ModelSpace ↔ channels ↔ kets.** These live in one header and point at each other:
  `Ket`/`Ket3` hold `Orbit*` into `ModelSpace::Orbits`, and every channel holds a
  `ModelSpace*` back-pointer. The source itself comments: "inextricable dependencies …
  What a mess" (`ModelSpace.hh:114-117`).
* **ModelSpace as a god object.** Beyond the basis it holds the angular-momentum caches
  (static and per instance), commutator threading flags, Pandya lookup tables, IMSRG(3)
  cuts (`dE3max`, `occnat3cut`), the Fermi energies, a profiler, valence-space
  definitions and string parsers.
* **3B data depends on ModelSpace caches.** `ThreeBodyStorage` recoupling calls
  `ModelSpace::GetSixJ`, and storage indexing depends on `ModelSpace::ThreeBodyChannels`
  and `Kets3`.
* **I/O inside storage.** NO2B and mono storage classes read files (boost gzip).
* **Operator holds algorithms.** It contains normal ordering, MBPT energies, truncation,
  isospin averaging and binary I/O.
* **Header hygiene.** There are header-level statics (`AngMom.hh`), an anonymous
  namespace in `ThreeBodyStorage.hh`, the duplicate typedefs `ME_half_type` and
  `ME_single_type` in two headers, the non-existent `TwoBodyME_ph`, and widespread
  const-casting in `ModelSpace` getters (`(Ket&)Kets[i]` in const methods,
  `ModelSpace.hh:427-472`).

**Suggested acyclic layering for the rewrite.** Each layer depends only on layers above it.

1. Constants and strong index types.
2. Angular momentum, with explicit cache objects that are thread-safe or built before use.
3. Basis: orbits and orbit sets, two-body basis (kets and channels), three-body basis,
   all index based with no raw pointers.
4. Storage: two-body blocks, the 3B back-ends, three-leg blocks.
5. `Operator`, an aggregate plus a symmetry descriptor.
6. Algorithms as free functions: normal ordering, truncation, norms, MBPT.
7. I/O.
8. Profiling.

---

## 9. Pitfalls

### A. Conventions to replicate exactly (they change numbers)

1. **Orbit order and its hash** (§1.2). The `"j"` letter means l=7 in labels, and
   `tz2=-1` is proton.
2. **Normalized 2B storage and unnormalized 2B getters.** 3B values are unnormalized; the
   3B→2B contraction divides by `(2J+1)·sqrt((1+δ)(1+δ))` (`Operator.cc:656`).
3. **Swap phase** `-(-1)^(jp+jq-J)`. The **tensor flip phase** is `(-1)^(J_b−J_k)` for
   2B, `(-1)^((j2_i−j2_j)/2)` for 1B and `(-1)^((2J_b−2J_k)/2)` for 3B. The reduced-ME
   convention is described in §2.5.
4. **3B recoupling and permutation phases** (§5.2). The canonical order is a<=b<=c for pn
   and descending for isospin.
5. **Isospin↔pn CG conventions** follow Condon–Shortley. The 2B iso→pn factor is
   `½·sqrt((1+δ)(1+δ))` with exchange phase `(2T−1)`.
6. **Default truncations.** `E3max=min(14,3emax)` and there is no E2max cut on kets.
   Partial shells use equal filling `occ=N/(2j+1)` in the naive HO+SO order (§1.5).
   `e_fermi` is in shell units.
7. **Norm definitions** (§3.4), including the `(2J+1)²` weight in the 2B norm. They feed
   convergence and BCH thresholds.
8. **Precision of input 3N data.** The isospin "full" format and mono are stored as float,
   no2b as float or half. Only pn is double. Results depend on this.
9. **`PhysConst::INVSQRT2` has only 13 digits** (`PhysicalConstants.hh:42`). It is used by
   `MakeNormalized`.
10. **GSL 3j, 6j and 9j accuracy and behaviour.** Arguments are `int(2j)` and errors abort.
    Moshinsky values use Moshinsky's phase via the Buck algorithm, and factorial tables
    overflow above 170.

### B. Apparent bugs (fix deliberately and document the deviation)

11. `ThreeBodyME::TransformToPN` slices the storage and leaves it unchanged
    (`ThreeBodyME.cc:336`).
12. `ThreeBodyStorage_iso::Subtract` adds (`ThreeBodyStorage_iso.cc:53`).
13. iso and pn `WriteBinary` write `total_dimension` bytes instead of elements. The iso
    copy constructor zeroes `total_dimension`
    (`ThreeBodyStorage_iso.cc:11, 352`, `ThreeBodyStorage_pn.cc:372`).
14. Reduced `GetME_pn_no2b`: `sqrt(2*J2b+1/(twoJ+1.))` (`ThreeBodyStorage.cc:389`).
    `ThreeBodyStorage(ModelSpace*)` sets `E2max = GetE3max()` (`ThreeBodyStorage.cc:12`).
15. `DoNormalOrdering2`, reduced branch: `OneBody(b,a) = herm·…` is not guarded for
    `herm==0` and wipes non-hermitian entries (`Operator.cc:561`). The `a==b` ×2 fix
    assumes a hermitian operator.
16. `Operator::Truncate`: the `"iso"` branch never fires because the mode is "isospin", so
    3B data is silently dropped for isospin and no2b operators. Only diagonal 3B blocks
    are copied (`Operator.cc:786-822`).
17. `ChangeNormalization` swaps row and column (`Operator.cc:1727-1742`). `ScaleFermiDirac`
    repeatedly scales a row (`Operator.cc:1820-1828`).
18. `AddToTBME` omits the anti-hermitian sign that `SetTBME` applies for
    `ch_bra>ch_ket` (`TwoBodyME.cc:246-252` vs `218-224`). `GetTBME_norm` treats
    non-hermitian operators as anti-hermitian when flipping (`TwoBodyME.cc:200-201`).
19. `TwoBodyChannel::GetLocalIndex(p>q)` returns `NumberKets−1` for an absent ket
    (`TwoBodyChannel.cc:75`). The constructor `TwoBodyChannel(int,ModelSpace*)` never
    sets `modelspace` (`TwoBodyChannel.cc:134-139`).
20. ModelSpace copy and move: `Ket`/`Ket3` `Orbit*` and `ThreeBodyChannel::modelspace`
    still point to the source. The copy constructor drops `Acore`, `Zcore`,
    `MonopoleKets`, `Ket_occ_ph`, `Ket_unocc_ph` and `hole_quantum_numbers`. The move
    constructor leaves `Kets3` empty. `operator=` drops `OrbitLookup`, `Lmax`,
    `e_fermi`, `MonopoleKets` and the 3B structures, and returns a full copy by value
    (`ModelSpace.cc:81-159, 853-960`). For example `imsrg++.cc:601` copies a ModelSpace
    and only works because the source stays alive.
21. `ThreeBodyME::SetMode(<unknown>)` re-allocates and erases the existing data
    (`ThreeBodyME.cc:404-409`). `SetMode("no2b")` followed by `SetMode("no2bhalf")`
    allocates twice (`imsrg++.cc:480-481`).
22. Undefined static `ModelSpace::T3bList`, the undefined `GetT3b`, and the unused
    `TwoBodyME_ph` are dead declarations.

### C. Hard limits and implicit assumptions

23. 3-body ket hash: fewer than 256 orbits in the 3-body space (`emax_3body <= 14`).
    Isospin key: 1024 isospin orbits. 6j hash: `2j <= 1023`. 9j hash: `2j <= 90`.
    Moshinsky packing: n <= 63, l <= 127, L <= 255. NO2B keys: index < 1024.
    `tensor_transform_first_pass` covers rank_J <= 9.
24. Even index means proton and `+1` is the neutron partner, which breaks for single
    species. Indices are emax-independent, which breaks if a smaller space uses
    different Lmax or EmaxUnocc cuts.
25. `Init(emax, …)` ignores emax. A default-constructed ModelSpace has an empty dense 6j
    cache and `GetCachedSixJ` reads out of bounds.
26. 6j values must be precalculated before any parallel region that recouples 3B MEs
    (`DoNormalOrdering3`, `TransformToPN`). A second ModelSpace's `PreCalculateSixJ`
    clears the shared cache.
27. The Pauli and stretched-state exclusions for 3-body kets are coded twice
    (`ThreeBodyChannel.cc:59-66`, `ThreeBodyStorage.cc:261-267`), and lookups `.at()`
    throw if the two disagree.
28. `SetOcc` does not refresh the derived sets or lists. `occ == OCC_CUT` exactly is
    classified inconsistently (§2.4).
29. `rank_T` is `|ΔTz|`. Tensor operators are always stored as hermitian or
    anti-hermitian combinations, so the blocks with `ch_bra>ch_ket` are implied, never
    stored. Non-hermitian 3B operators are unsupported.
30. Setting the default 3B mode with `SetParticleRank(3)` allocates **isospin/float**
    storage unless `SetMode("pn")` came first.

### D. Performance hotspots and hidden costs

31. The Operator move constructor and move assignment deep-copy the 2B and 3B data
    (§0.6). Return-by-value arithmetic such as `operator*` copies twice.
32. ThreeLeg is allocated, copied and scaled for every operator (§0.7).
33. Each 3B ME access goes sort → `std::map` channel lookup → per-Jab `std::map` ket
    lookup, `KetMap.at` and 6j hash lookups (`ThreeBodyStorage.cc:236-291`). Isospin and
    no2b conversions call the GSL CG function four times per term.
34. Every channel stores a `KetMap` sized to the full ket list
    (2 × #channels × #kets ints). `Kets3` and all 3B channels are rebuilt on every
    `Init`, even for IMSRG(2).
35. The dense 6j cache is `4(emax+1)^6` doubles, built serially and copied with every
    ModelSpace. `PreCalculateSixJ` can exceed 1 GB (comment at `ModelSpace.cc:1776-1781`).
36. Global profiler maps are touched in constructors and destructors. Armadillo bounds
    checks are on everywhere. `SortedTwoBodyChannels` sorts by size for dynamic
    scheduling.

---

## Appendix A — Python-exposed core API (`pyIMSRG.cc:22-360`)

The rewrite's bindings should cover at least the following.

* **Orbit, Ket, Ket3** (fields).
* **TwoBodyChannel(_CC):** `GetNumberKets`, `GetLocalIndex`, `GetKetIndex`, `GetKet`,
  `GetKetIndex_{pp,hh,ph,cc,vc,qc,vv,qv,qq}`.
* **ThreeBodyChannel:** `GetNumberKets`, `GetLocalIndex`, `GetKet`.
* **ModelSpace:** constructors; setters for hw, A, Z, emax, e2max, e3max, dE3max, Lmax,
  EmaxUnocc and Emax3Body; `AddOrbit`, `SetupKets`, `Setup3bKets`, `SetOcc`,
  `SetOccNAT`, `FindEFermi`; the channel and orbit getters, `Index2String`, the
  `SetReference*` family, `Init_occ_from_file`, `InitSingleSpecies`, `GetOneBodyChannels`,
  `PreCalculate{SixJ,NineJ,Moshinsky}`, `Get{SixJ,NineJ,Moshinsky}`, `ResetFirstPass`,
  the first-pass setters and `Print`.
* **Operator:** the full set in §3.2 except the arithmetic operators, which are covered by
  the separate `TwoBodyME` and `ArmaMat` bindings.
* **TwoBodyME:** `GetTBME_J(_norm)`, `GetTBMEmonopole(_norm)`, `GetTBME_norm`, channel
  matrix access, `Erase`, `Norm`.
* **ThreeBodyME:** `GetME_iso`, `GetME_pn` (scalar and tensor), `SetME_pn_ch`,
  `GetME_pn_no2b`, `RecouplingCoefficient`, `TransformToPN`, `SwitchToPN_and_discard`,
  `Erase`, `SetMode`, `IsAllocated`, `IsHermitian`, `Is_PN_Mode`, `IsReduced`, `ReadFile`.
* **IMSRGProfiler:** `Print*`, `Clear`, `GetTimer`, `GetCounter`.
