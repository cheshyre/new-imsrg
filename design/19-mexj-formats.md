# 19 — ME1J, ME2JP and ME3JP file formats (owner, D54)

Specification of the owner's proton-neutron resolved operator formats, worked out on 2026-11-16 from
the owner's jIMSRG readers (`jimsrg/files/formats`: `me1j`, `me2jp`, `me3jp`, `helpers/mexj`; the
code was read, not copied into this repository) and checked against the Darmstadt ME2J/ME3J
readers of imsrg++ (`reference/04` §4.2 and the ME3J loop listing there). Implementation: the
milestone after tensor operators (D53, D54); `40-roadmap.md` M15+.

## 1. Scope and uses

* **ME1J** holds the zero- and one-body part, **ME2JP** the two-body part, **ME3JP** the three-body
  part of one operator. Together they describe a **scalar, parity-conserving, charge-conserving**
  operator of definite hermiticity in a proton-neutron resolved single-particle basis. Tensor
  operators (rank > 0) and isospin-changing operators cannot be represented.
* **ME2JP is the NLJJ → NLJJT extension of ME2J** (the owner's statement, verified below): the
  same loops over orbits that now carry the isospin projection, one value per record instead of
  ME2J's four isospin-coupled values. **ME3JP is the same extension of ME3J**, with full three-body
  angular-momentum coupling and one value instead of ME3J's five-value isospin block.
* Uses (D54): the evolved H is written by default, other scalar operators on request; the Magnus
  Omega (anti-Hermitian; with IMSRG(3) also its 3B part) is stored so that a `transform` task can
  apply it to many operators later; checkpoint/restart stores H(s) and the Omega stack in these
  files plus a small solver-state file.

## 2. Single-particle basis (NLJJT)

* Shells e = 2n + l from 0 to emax; inside a shell, l ascending from e mod 2 to e in steps of 2;
  for each l, 2j = 2l − 1 (only for l > 0), then 2l + 1; for each (n, l, j) **the neutron orbit
  first, then the proton orbit**.
* Index: idx(n, l, j, tz) = e(e + 1) + o_l + o_j + [proton], with o_l = 4⌊l/2⌋ − 2·((l + 1) mod 2)
  and o_j = 2j − 2l + 1 ∈ {0, 2} (2j and 2l as integers). Equivalently idx = 2·nlj + [proton],
  where nlj is the orbit index of ME2J (e ascending, then l ascending, then j ascending).
* Single-particle energies are non-decreasing in the index.
* This is **not** our orbit order (proton even, neutron = proton + 1; l and j descending inside a
  shell; AGENTS.md). Readers and writers map orbits by their quantum numbers, never by index
  arithmetic across the two conventions.

## 3. Rules common to the three formats

* **Text layout.** The first line is a header that must contain the format tag (`me1j`, `me2jp`,
  `me3jp`); jIMSRG checks only that substring. Then whitespace-separated decimal numbers in record
  order; line breaks carry no meaning.
* **Not in the file:** emax, E3max and the hermiticity. The reader receives them from its caller
  (the config), as jIMSRG does.
* **Our header line:** the tag, then informational `key=value` tokens:
  `me2jp emax=<e> herm=<+1|-1> op=<name> code=imsrg-<version>` (ME3JP adds `e3max=<E>`). Our
  readers may cross-check these tokens when present and ignore unknown ones; files without them
  (jIMSRG output) stay readable.
* **Values:** matrix elements of the operator (J-scheme, not reduced; m-independent for a scalar)
  between **antisymmetrized, unnormalized** states, as in ME2J and ME3J (owner, 2026-11-16): no
  1/√2 per identical-orbit pair; Pauli-forbidden entries (for example |pp; J odd⟩ with p = q) are
  stored as zeros. Our two-body storage is normalized, so the ME2JP writer multiplies by √2 for
  each identical-orbit pair and the reader divides; our three-body storage (layout A, D43) already
  uses imsrg++'s unnormalized convention.
* **Number format of our writers:** `%.17g` (17 significant digits), so that checkpoints and stored
  Omegas round-trip bit for bit; five values per line.
* **Canonical order and symmetries:** only canonical entries (bra ≥ ket in the order of §5 and §6)
  are stored. Other orderings follow from antisymmetry (§5, §6) and hermiticity (factor +1 or −1).
* **Prefix property:** the outermost loop runs over the first orbit p in index order, and every
  other orbit of a record has an index ≤ p, so the records with p in the emax′ basis form exactly
  the emax′ file. A reader may read the emax′ prefix of a file written for a larger emax (ME3JP:
  same E3max). jIMSRG allows this for ME1J and ME2JP and rejects longer ME3JP files in its
  validator; our readers allow it for all three.
* **Variants and file names:** text `name.me1j`, `name.me2jp`, `name.me3jp`; **gzip** text
  `name.me2jp.gz`, `name.me3jp.gz` (D54; through `io::TextInput` and `GzipWriter`; ME1J stays
  plain text, it is small); **binary** `name.me2jp.bin`, `name.me3jp.bin`: a header made of a
  fixed number of characters, then the values in record order. The binary specification is
  pending from the owner; the implementation separates the record enumeration from the value I/O
  so that the binary variant only adds the header and the raw value source and sink.

## 4. ME1J

```
header line (contains "me1j")
value of the zero-body part
for p in basis:                     # index order
  for q in basis with q <= p:       # dense lower triangle, diagonal included
    <p|O|q>
```

All N(N + 1)/2 pairs of the N = (emax + 1)(emax + 2) NLJJT orbits are listed, including pairs
forbidden by j, parity or charge (stored as zeros); ⟨q|O|p⟩ = h ⟨p|O|q⟩ with the hermiticity
factor h. At emax 12: N = 182, 1 + 16,653 values.

## 5. ME2JP

```
header line (contains "me2jp")
for p in basis:
  for q in basis with q <= p:
    for r in basis with r <= p:
      for s in basis with s <= r, and s <= q if r == p:     # (r, s) <= (p, q)
        skip unless l_p + l_q + l_r + l_s is even           # parity
        skip unless tz_p + tz_q == tz_r + tz_s              # charge conservation
        for J from max(|j_p - j_q|, |j_r - j_s|) to min(j_p + j_q, j_r + j_s):
          <pq; J| O |rs; J>
```

Symmetries (jIMSRG getter): swapping the orbits of a pair gives the phase −(−1)^(j_a + j_b − J);
if (r, s) > (p, q), bra and ket are exchanged with the hermiticity factor.

**Comparison with ME2J** (checked 2026-11-16). Identical: one header line followed by the values;
the orbit order inside a shell; the four loops (p ≥ q; r ≤ p; s ≤ r, and s ≤ q when r = p); the
parity rule; J ascending over the common range. Different: one value per (pqrs, J) instead of
four (T = 0 pn, nn, T = 1 pn, pp); charge conservation is explicit; no e2max and no lmax cut (an
ME2JP file is complete up to 2·emax, emax is its only parameter); values are doubles (ME2J is read
as float); the header tag is checked (the ME2J header is skipped unchecked); jIMSRG reads text
only (its gzip and binary readers are deleted).

Note on the jIMSRG reader (not a format issue): `ME2JPFile::Get2BMatrixElement` computes `jj_max`
with `std::max` where the file layout uses `std::min`, so a J outside the coupling range of one of
the two pairs reads the neighbouring record (or past the end of the array) instead of 0; queries
with a J valid for both pairs are unaffected. Our reader returns 0 for such a J.

## 6. ME3JP

```
header line (contains "me3jp")
for p in basis:
  for q in basis with q <= p:
    for r in basis with r <= q:
      skip if e_p + e_q + e_r > E3max
      for s in basis with s <= p:
        for t in basis with t <= s, and t <= q if s == p:
          for u in basis with u <= t:
            skip if e_s + e_t + e_u > E3max
            skip if s == p and t == q and u > r             # (s, t, u) <= (p, q, r)
            skip unless the parities and the sums of tz of the two triples agree
            for J_pq from |j_p - j_q| to j_p + j_q:
              for J_st from |j_s - j_t| to j_s + j_t:
                for J from max(|J_pq - j_r|, |J_st - j_u|) to min(J_pq + j_r, J_st + j_u):
                  <(pq)J_pq, r; J| O |(st)J_st, u; J>
```

* Every J_pq and J_st is listed, also when orbits coincide (overcomplete; Pauli-forbidden
  combinations are zero). These are the canonical kets of imsrg++ and of our layout A (D43), but
  ordered p ≥ q ≥ r in NLJJT indices instead of a ≤ b ≤ c in our orbit order, so converting
  between the file and our storage needs the recoupling below.
* **Symmetries** (jIMSRG getter; re-derived on 2026-11-16 for antisymmetrized, unnormalized
  states, with Ĵ = √(2J + 1)): a pair swap gives −(−1)^(j_a + j_b − J_ab); exchanging the second
  and the third orbit gives
  |(ab)J_ab, c; J⟩ = −Σ_{J_ac} (−1)^(j_b + j_c + J_ab + J_ac) Ĵ_ab Ĵ_ac {j_b j_a J_ab; j_c J J_ac} |(ac)J_ac, b; J⟩,
  where J_ac runs over the values allowed by both (j_a, j_c) and (J, j_b); bra and ket are
  exchanged with the hermiticity factor when (s, t, u) > (p, q, r).
* **Comparison with ME3J:** the same loop structure (a ≥ b ≥ c with E3max, the second triple
  lexicographically not above the first, parity, then J_ab, J_de, J) with NLJJT orbits; one value
  per (J_pq, J_st, J) instead of the five-value isospin block (T_12, T_45, T); charge conservation
  explicit; no E2max cut on (p, q); doubles; the header tag is checked.
* **Size:** at emax 12 / E3max 16 the 3B part has about 2.2e10 values, comparable to one packed
  layout-A operator (177 GB as doubles); as text about 0.4 TB, hence gzip and the binary variant
  for production sizes.

## 7. Implementation plan (milestone after tensor operators)

* `io`: `ReadMe1j`, `ReadMe2jp`, `ReadMe3jp` and the matching writers; pure record enumerators
  (`ForEachMe2jpRecord`, `ForEachMe3jpRecord`) shared by readers, writers and counts; value
  sources and sinks for text, gzip and (once specified) binary; the recoupling of §6 to map between
  the file's canonical triples and layout A (using `SixJ3Table`).
* Tests: record counts against an independent count; write → read round trips bit for bit (text,
  gzip); ME2J → ME2JP conversion of He4/O16 at emax 2 compared element by element with the ME2J
  reader after normalization; a 3B round trip at emax 2 (E3max 6) against layout A; prefix
  reading; sample files from jIMSRG as goldens when the owner provides them.
* Driver (keys planned, E-PLANNED until implemented): `[output] me2jp = true` writes H (ME1J +
  ME2JP, plus ME3JP when H has a 3B part); `[output] me2jp_operators = [...]` for further scalar
  operators; `[output] omega = "me2jp"` stores the Omega stack; `run.task = "transform"` reads a
  stored Omega and transforms a list of operators; `[checkpoint]` (interval, directory) writes H(s),
  the Omega stack and the solver state (s, step, stage, ds, gather count) and restarts from them.
* Open, from the owner: the binary header and value type of `.bin`.
