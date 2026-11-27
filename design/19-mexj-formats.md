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
* Index: idx(n, l, j, tz) = 2·nlj + [proton], where nlj is the orbit index of ME2J (e ascending,
  then l ascending, then j ascending). (An earlier closed formula, e(e + 1) + o_l + o_j + [proton],
  was wrong: it puts d3/2 at 10 instead of 8; AL-C5, owner-confirmed 2026-11-27.)
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
  plain text, it is small); **binary** `name.me2jp.bin`, `name.me3jp.bin`: a header of exactly
  255 bytes, then the values in record order as 4-byte floats (§3.1; owner via A. Tichai,
  2026-11-16, D57). The implementation separates the record enumeration from the value I/O so
  that the binary variant only adds the header and the raw value source and sink.

### 3.1 Binary variant (D57)

Layout, as A. Tichai's reader (`binary_to_vector`, `#define BINARYHEADERSIZE 255`) reads it:

| Bytes | Content |
|-------|---------|
| 0 … 254 | header, 255 bytes, skipped unread by A. Tichai's reader and imsrg++ |
| 255 … | the values of the text variant in the same record order, IEEE-754 binary32 (`float`), native byte order (little-endian on every supported target), no record markers or padding |

* **Same as imsrg++'s binary ME2J/ME3J** (checked 2026-11-16 in imsrg-devel): `ReadWrite.cc:30`
  `#define HEADERSIZE 255` (a commented-out `500` above it); `ReadBareTBME_Darmstadt` (`.bin`,
  `ReadWrite.cc:687-717`) and `Read_Darmstadt_3body` (`.bin`, `ReadWrite.cc:764-797`) take
  n = (file size − 255) / 4, read and discard the 255 header bytes, read n floats and feed them
  to the text parser through `VectorStream` (`ReadWrite.hh:156-173`), whose `getline` does
  nothing: the binary header replaces the text header line, and the floats are exactly the text
  values.
* **The header content is not interpreted** by either reader. Our writer fills it with our text
  header line (§3: tag and `key=value` tokens), padded with spaces to 255 bytes; our reader checks
  the tag only when the header contains one (files from other writers stay readable).
* **Checks the old readers lack:** the file size must be 255 + 4n with n at least the expected
  value count for the caller's emax (and E3max), longer files being allowed by the prefix rule;
  sizes and offsets are 64-bit (imsrg++'s ME2J path holds the file size in an `int`,
  `ReadWrite.cc:698`, wrong above 2 GiB; its ME3J path uses `size_t`); every read is checked (a
  truncated file is an `InputError`, not zeros); large ME3JP files are read in chunks instead of
  one vector (88 GB of floats at emax 12 / E3max 16).
* **Precision:** binary32 keeps about 7 digits, so a `.bin` file does not round-trip our doubles.
  Until the owner decides otherwise (QUESTIONS O4), `.bin` is for interactions and other inputs;
  checkpoints and stored Omegas use the text or gzip variant (`%.17g`, bit for bit).
* **Implemented for ME2J:** `io::ReadMe2j` (`src/imsrg/io/me2j.cc`, `BinaryInput`) reads
  `.me2j.bin` (dispatch on the file suffix `.bin`; gzip stays detected by content). Checks there:
  the file size must be exactly 255 + 4n bytes with n = `Me2jValueCount` of the file's own
  truncation (also when only an emax prefix is read; stricter than the n-at-least rule above, a longer or shorter file is an
  `InputError`), 64-bit sizes, chunked reads of 2^20 floats, every read checked with the file name
  and byte offset in the message; the header is not interpreted. Tests: `Me2jBinary*` in
  `tests/unit/io/test_me2j.cc`. `tools/reference/me2j_text_to_bin.py` converts a text or gzip
  me2j to this form. The ME3J reader is still to do.

**Verification note (2026-11-17, passed): imsrg++ reads a `.me2j.bin` written by our rule.** Input
`VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` (`file2e1max=14 file2e2max=28 file2lmax=14
hw=16`), converted with `python3 tools/reference/me2j_text_to_bin.py <in>.gz $TMP/vnn_emax14.me2j.bin`
(168 066 656 floats, 672 266 879 bytes = 255 + 4n, 16 s; deleted afterwards). Reference harness
(imsrg++ with the D35 patch, `harness-build-d35/refharness`) at emax 2, He4, 2N only, text/gzip against
`.bin`:

```
refharness hf emax=2 reference=He4 valence_space=He4 hw=16 2bme=<file> file2e1max=14 file2e2max=28 \
    file2lmax=14 dump_hno=true out=<dir>
refharness dump-hamiltonian emax=2 reference=He4 valence_space=He4 hw=16 2bme=<file> file2e1max=14 \
    file2e2max=28 file2lmax=14 out=<dir>
```

imsrg++ printed `n_elem = 168066656` for the `.bin`, i.e. it took the `.bin` branch (`ReadWrite.cc:687-717`)
and counted 255 + 4n. Compared: `hf.json` (EHF -22.989701746458095, e1hf 49.5435562546038, e2hf
-72.5332580010619) and the dumps `V2.dump` (1596 two-body elements), `T.dump`, `H.dump`, `HNO.dump`:
**identical in every token** (max difference 0). That is expected, not a coincidence of tolerance:
imsrg++'s text path also parses the values into `float` (`ReadWrite.cc:882`), so text and `.bin`
reach the same float32 values. Our reader, `imsrg method=HF emax=2 reference=He4 valence_space=He4 hw=16
2bme=<file> file2e1max=14 file2e2max=28 file2lmax=14 intfile=<out>` (release build, text/gzip against
`.bin`): identical log and `results.json` (e_hf -22.989701746458117, e2 -72.53325800106191; 2e-14 from
imsrg++'s value, double-precision HF arithmetic). Not checked: the ME3J `.bin` (reader not written),
emax above 2 (the reading code is the same; the emax-2 prefix of the file is what is read).

### 3.2 Native binary operator format (design/50 C6, 2026-11-26)

Not an exchange format: the exact (float64) storage of our own operators, used for stored Omegas
(`output.write_omega`, `output.omega_format = "native"`) and flow checkpoints (`[checkpoint]`,
`format = "native"`). `io::WriteOperatorBinary` / `io::ReadOperatorBinary`
(`src/imsrg/io/operator_binary.h`). Layout, little-endian: magic `IMSRGOPB`; u32 format version (1);
u32 header length; header: u64 model-space layout hash (FNV-1a over the orbits (n, l, 2j, 2tz), hw and
the two-body channels with their kets), u64 reference hash (occupation bit patterns), i32
particle_rank, rank_J, parity, delta_Tz, hermiticity, delta_particle_number, emax, e2max (= 2 emax),
f64 hw, u32 label length + label bytes, u64 one-body and two-body buffer counts; then f64 zero body,
the raw one-body buffer and the raw two-body block buffer exactly as stored; then a u64 FNV-1a
checksum of everything after the magic. The reader checks the layout hash (always), the reference
hash (optional), counts, exact size and checksum. Files are written to `<path>.tmp.<pid>` and renamed,
so a crash never leaves a partial file under the final name. Round trips are bit for bit; three-body
parts are not supported yet (IMSRG(3) Omegas later).

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
with a J valid for both pairs are unaffected. Our reader returns 0 for such a J (owner requirement
D55, §8).

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
  between the file and our storage needs the recoupling below. The same J-range rules (§8) apply to
  J_pq, J_st and J: a value outside a record's stored ranges is 0.0, never another record's entry.
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
* **J-range safety (D55):** the rules and tests of §8 are part of the acceptance.
* Binary `.bin`: specified in §3.1 (D57); the ME2J reader is done (§3.1), the ME3J reader and the
  writers are open; also open: whether checkpoints need a double-precision
  binary variant (QUESTIONS O4).

## 8. J-range safety (owner requirement, D55)

The owner confirmed the jIMSRG finding of §5 and asked (2026-11-16, verbatim): "Make absolutely sure
this is implemented properly in the new ME2JP and ME3JP implementations." The failure mode: a J
that lies inside the coupling range of one pair (or, in ME3JP, a J_pq, J_st or J allowed for one
side) but outside the record's **stored** range passes a wrong bound check, and the offset reads a
neighbouring record, or past the end of the value array for the last record. The value looks
plausible, so nothing downstream notices. Our implementation makes this impossible by construction
and proves it by tests.

**Rules** (binding for the readers, the writers and every random-access getter):

1. **One source of truth for the ranges.** Each format has one function that computes a record's
   J ranges: ME2JP J ∈ [max(|j_p − j_q|, |j_r − j_s|), min(j_p + j_q, j_r + j_s)]; ME3JP
   J_pq ∈ [|j_p − j_q|, j_p + j_q], J_st ∈ [|j_s − j_t|, j_s + j_t] and, for each (J_pq, J_st),
   J ∈ [max(|J_pq − j_r|, |J_st − j_u|), min(J_pq + j_r, J_st + j_u)]. The record enumerator, the
   value count, the writer, the reader and the lookup table all call it. The lookup table stores
   each record's offset **together with its ranges** (as jIMSRG's `ME3JPBlockInfo` does for
   ME3JP), and the getter uses the stored ranges; it never recomputes a bound.
2. **Outside the stored ranges the answer is exactly 0.0.** Such a state does not exist on at
   least one side. The getter returns 0.0 before it computes any offset.
3. **Every offset is checked.** The raw access layer checks `offset < block size` of the record
   with `IMSRG_CHECK` (negligible next to the hash lookup), so a future range error throws instead
   of returning another record's value.
4. **Two layers.** The raw file layer (records, values, getter with the pair-swap phases,
   hermiticity and, for ME3JP, the recoupling of §6) is separate from the conversion into our
   operators. Rules 1–3 and the tests below apply to the raw layer; the ME3JP recoupling sums call
   the raw getter for every intermediate J, so they inherit the guard.

**Required tests** (part of the milestone's acceptance; none may be skipped or weakened):

* **Sentinel files against an independent oracle.** The test writes files with its own naive loops
  (not the production enumerator), giving every stored entry a unique nonzero value (its ordinal
  + 1): ME2JP at emax 2 and 3, ME3JP at emax 1 (E3max 3) and emax 2 (E3max 6). For every record and
  every J in the **union** of both sides' ranges (ME3JP: every J_pq, J_st and J from the union of
  the two triples' ranges), the production getter must return the sentinel for a stored entry and
  **exactly 0.0** for every other J. Queries in canonical order check the sentinels directly;
  queries in every other orbit order (p↔q, r↔s and bra↔ket for ME2JP; all orderings of both triples
  for ME3JP) must equal the test's own evaluation of the phases (§5) or of the recoupling formula
  (§6) from the sentinels, and must be 0.0 whenever the query's own J values are out of range. Files
  of zeros or of physics values cannot detect this class of bug, so the sentinels are mandatory.
* **The owner's example** at emax 2: ⟨0d5/2π 0d5/2ν; J=3| O |1s1/2π 1s1/2ν; J=3⟩ = 0.0. jIMSRG's
  getter returns the J = 2 entry of the next record, ⟨0d5/2π 0d5/2ν; 2| O |0d3/2ν 0s1/2π; 2⟩.
* **The last record of a file:** a query with an out-of-range J returns 0.0 (jIMSRG would read past
  the end of the array).
* **Lower bounds too:** the union ranges include J below the larger of the two minima; these return
  0.0 as well.
* **AddressSanitizer:** the tests above also run in the `asan` preset before the merge, so any
  out-of-bounds read aborts.
* **Counts and round trips:** the production enumerator's record and value counts equal the
  oracle's for ME1J, ME2JP and ME3JP at several emax and E3max; write → read round trips are bit
  for bit (text and gzip).
* **Review:** the Opus review of the milestone checks rules 1–4 explicitly and reports on them.
