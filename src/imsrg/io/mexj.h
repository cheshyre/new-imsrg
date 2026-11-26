// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Readers and writers of the owner's ME1J and ME2JP formats (design/19).
#ifndef IMSRG_IO_MEXJ_H_
#define IMSRG_IO_MEXJ_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::io {

/// An orbit of the NLJJT file basis (design/19 section 2). `tz2` is -1 for a
/// proton, +1 for a neutron (as `basis::Orbit`). Plain integers: this index
/// space is private to the file formats and distinct from `OrbitIndex`.
struct NljjtOrbit {
  std::int32_t n = 0;
  std::int32_t l = 0;
  std::int32_t j2 = 1;   ///< 2j
  std::int32_t tz2 = 1;  ///< 2 tz

  /// Shell e = 2n + l.
  constexpr std::int32_t e() const noexcept { return 2 * n + l; }
  friend constexpr bool operator==(const NljjtOrbit&,
                                   const NljjtOrbit&) = default;
};

/// The (emax + 1)(emax + 2) orbits of the NLJJT basis in file order: shells
/// ascending; inside a shell l ascending from e mod 2, then 2j = 2l - 1 (l > 0)
/// before 2l + 1; for each (n, l, j) the neutron first, then the proton.
/// O(number of orbits). Precondition: emax >= 0 (IMSRG_CHECK).
std::vector<NljjtOrbit> NljjtOrbits(std::int32_t emax);

/// The file-basis index idx = e(e + 1) + o_l + o_j + [proton] of an orbit
/// (design/19 section 2), equal to its position in `NljjtOrbits`. O(1).
/// Precondition: a valid orbit (n, l >= 0, 2j = 2l +- 1 > 0, tz2 = +-1).
std::int32_t NljjtIndex(std::int32_t n, std::int32_t l, std::int32_t j2,
                        std::int32_t tz2);

/// The J range [jmin, jmax] (integers, scalar operator) of an ME2JP record
/// <pq; J|O|rs; J>, with jmin > jmax meaning "no record": the range is
/// max(|j_p - j_q|, |j_r - j_s|) .. min(j_p + j_q, j_r + j_s), and it is empty
/// when l_p + l_q + l_r + l_s is odd or tz_p + tz_q != tz_r + tz_s. This is the
/// ONE function that computes a record's J range (design/19 section 8 rule 1):
/// the enumerator, the counts, the table, the readers and the writers all use
/// it. O(1).
struct JRange {
  std::int32_t jmin = 0;
  std::int32_t jmax = -1;
  /// Number of J values (0 for an empty range).
  constexpr std::int32_t size() const noexcept {
    return jmax >= jmin ? jmax - jmin + 1 : 0;
  }
  constexpr bool empty() const noexcept { return jmax < jmin; }
  constexpr bool contains(std::int32_t J) const noexcept {
    return J >= jmin && J <= jmax;
  }
};
JRange Me2jpJRange(const NljjtOrbit& p, const NljjtOrbit& q,
                   const NljjtOrbit& r, const NljjtOrbit& s) noexcept;

/// Calls `f(p, q, r, s, range)` (NLJJT indices, `JRange` non-empty) for every
/// record of an ME2JP file of the given emax, in file order (design/19 section
/// 5 loops: p, q <= p, r <= p, s <= r and s <= q if r == p). The values of a
/// record follow in the file for J = range.jmin .. range.jmax. O(number of
/// candidate orbit quadruples).
template <class F>
void ForEachMe2jpRecord(std::int32_t emax, F&& f) {
  const std::vector<NljjtOrbit> orbs = NljjtOrbits(emax);
  const auto count = static_cast<std::int32_t>(orbs.size());
  for (std::int32_t p = 0; p < count; ++p) {
    for (std::int32_t q = 0; q <= p; ++q) {
      for (std::int32_t r = 0; r <= p; ++r) {
        const std::int32_t smax = (r == p) ? q : r;
        for (std::int32_t s = 0; s <= smax; ++s) {
          const JRange range = Me2jpJRange(orbs[static_cast<std::size_t>(p)],
                                           orbs[static_cast<std::size_t>(q)],
                                           orbs[static_cast<std::size_t>(r)],
                                           orbs[static_cast<std::size_t>(s)]);
          if (!range.empty()) f(p, q, r, s, range);
        }
      }
    }
  }
}

/// Number of values of an ME2JP file of the given emax (sum of the J-range
/// sizes of `ForEachMe2jpRecord`). Reference: emax 2 and above are checked
/// against an independent count in the tests.
std::int64_t Me2jpValueCount(std::int32_t emax);
/// Number of values of an ME1J file: 1 + N (N + 1) / 2 with N the number of
/// orbits.
std::int64_t Me1jValueCount(std::int32_t emax);

/// Raw ME2JP data: the values of one file (unnormalized, antisymmetrized
/// <pq; J|O|rs; J>, design/19 section 5) with random access (design/19 section
/// 8). Value type.
class Me2jpTable {
 public:
  /// Location of a record: first value and J range.
  struct Block {
    std::int64_t offset = 0;
    JRange range;
  };

  /// Builds the lookup table of emax `emax` over `values` (in record order).
  /// `h` must be kHermitian or kAntiHermitian.
  /// @throws InputError if `values.size() != Me2jpValueCount(emax)` or `h` is
  ///         non-Hermitian.
  Me2jpTable(std::int32_t emax, Hermiticity h, std::vector<double> values);

  Me2jpTable(const Me2jpTable&) = default;
  Me2jpTable& operator=(const Me2jpTable&) = default;
  Me2jpTable(Me2jpTable&&) = default;
  Me2jpTable& operator=(Me2jpTable&&) = default;

  std::int32_t emax() const noexcept { return emax_; }
  Hermiticity hermiticity() const noexcept { return hermiticity_; }
  const std::vector<double>& values() const noexcept { return values_; }
  /// Number of records.
  std::size_t num_records() const noexcept { return blocks_.size(); }

  /// <pq; J|O|rs; J> for NLJJT orbits in any order and integer J: pair swaps
  /// give -(-1)^(j_a + j_b - J), exchanging bra and ket gives the hermiticity
  /// factor. Exactly 0.0 when the record does not exist (parity, charge) or J
  /// is outside the STORED range of the canonical record (checked before any
  /// offset is computed); every offset is IMSRG_CHECKed. O(1).
  /// Precondition: 0 <= p, q, r, s < number of orbits (IMSRG_CHECK).
  double Get(std::int32_t p, std::int32_t q, std::int32_t r, std::int32_t s,
             std::int32_t J) const;

  /// The record of a CANONICAL quadruple (q <= p, s <= r, (r, s) <= (p, q)), if
  /// any. O(1).
  std::optional<Block> FindBlock(std::int32_t p, std::int32_t q, std::int32_t r,
                                 std::int32_t s) const;

 private:
  std::int32_t emax_;
  Hermiticity hermiticity_;
  std::int32_t num_orbits_;
  std::vector<NljjtOrbit> orbits_;
  std::vector<double> values_;
  std::unordered_map<std::int64_t, Block> blocks_;
};

/// Writes the zero- and one-body part of the scalar operator `op` as an ME1J
/// text file (design/19 section 4): header `me1j emax=<e> herm=<+1|-1>
/// op=<name> code=imsrg` (emax of the operator's model space), then the
/// zero-body value and the dense lower triangle <p|O|q> (q <= p) in NLJJT
/// order, `%.17g`, five per line. File orbits missing from the model space are
/// written as 0. The file is written to a temporary and renamed.
/// @throws NotImplementedError if `op` is not scalar, has particle rank 3 or is
///         non-Hermitian. @throws InputError if the file cannot be written.
void WriteMe1j(const std::filesystem::path& path, const op::Operator& op,
               std::string_view op_name);

/// Writes the two-body part of `op` as an ME2JP file. Values are unnormalized
/// antisymmetrized matrix elements (our normalized value times sqrt(2) for each
/// identical-orbit pair), Pauli-forbidden entries 0, orbits missing from the
/// model space 0. Text (`%.17g`, five per line, header `me2jp emax=... `) or,
/// if `path` ends in `.bin`, the binary variant of design/19 section 3.1: a
/// 255-byte header (the text header line padded with spaces) then little-endian
/// binary32 values. Atomic (temporary file and rename). Same preconditions and
/// exceptions as `WriteMe1j`.
void WriteMe2jp(const std::filesystem::path& path, const op::Operator& op,
                std::string_view op_name);

/// Reads an ME1J and an ME2JP file (text, or `.bin` for the ME2JP) into a new
/// scalar operator of hermiticity `h` (kHermitian or kAntiHermitian) on `ms`
/// with particle rank 2. The files must hold at least the values for the model
/// space's emax; longer files are read as the emax prefix (design/19 section
/// 3). A text header must contain the format tag; `emax=` and `herm=` tokens
/// are cross-checked when present (emax of the file >= the model space's). File
/// orbits missing from `ms` are skipped; nonzero entries that no scalar
/// operator can have are an error. Values are divided by sqrt(2) per identical
/// pair. A `.bin` file must have 255 + 4n bytes with n at least the expected
/// count.
/// @throws InputError (message with path and offset) on any bad file.
op::Operator ReadMe1jMe2jp(const std::filesystem::path& me1j_path,
                           const std::filesystem::path& me2jp_path,
                           std::shared_ptr<const basis::ModelSpace> ms,
                           Hermiticity h);

}  // namespace imsrg::io

#endif  // IMSRG_IO_MEXJ_H_
