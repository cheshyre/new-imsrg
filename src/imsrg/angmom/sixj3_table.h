// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_ANGMOM_SIXJ3_TABLE_H_
#define IMSRG_ANGMOM_SIXJ3_TABLE_H_
/// @file
/// Dense table of the one 6j family the three-body kernels need,
/// {j1 j2 J12; j3 J J23} with three single-particle j, two integer two-body J
/// and one half-integer three-body total J (IMSRG(3) layout decision D43;
/// design/14 section 4 item 0; ported from the benchmark prototype).
///
/// Why a new table: `SixJTable` covers {ja jb J; jc jd J'} with all
/// four j single-particle (2j <= 2 emax + 1); a three-body total J in a
/// j slot is beyond its coverage (the host lookup throws, the view returns NaN;
/// design/benchmarks/imsrg3-layout.md section 7). imsrg++ switches to a
/// hash map for that case (IMSRG3Commutators.cc:5292-5301). This family
/// covers every 6j of the recoupling coefficients (ThreeBodyStorage.cc:97-120:
/// {ja jb Jab; jc J Jab_in} and {jb ja Jab; jc J Jab_in}) and of the [2,2]->3
/// sum ({j_w j_a J1'; j_z J J2'}, IMSRG3Commutators.cc:5294).

#include <cstdint>
#include <type_traits>

#include "imsrg/base/device.h"
#include "imsrg/base/types.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::angmom {

/// Read-only view of a `SixJ3Table`; trivially copyable, captured by value.
class SixJ3View {
 public:
  SixJ3View() = default;
  /// View of `data` (6j values, layout: see `SixJ3Table`) and `hat` (hat[J] =
  /// sqrt(2J + 1), J = 0..2 emax + 1) for shell `emax`; both null or both
  /// consistent with `emax`.
  IMSRG_INLINE_FUNCTION constexpr SixJ3View(const double* data,
                                            const double* hat,
                                            std::int32_t emax) noexcept
      : data_(data), hat_(hat), emax_(emax) {}
  SixJ3View(const SixJ3View&) = default;
  SixJ3View& operator=(const SixJ3View&) = default;
  SixJ3View(SixJ3View&&) = default;
  SixJ3View& operator=(SixJ3View&&) = default;

  /// The 6j values (layout: see `SixJ3Table`); null for a default view.
  IMSRG_INLINE_FUNCTION const double* data() const noexcept { return data_; }
  /// The largest single-particle shell, -1 for a default view.
  IMSRG_INLINE_FUNCTION std::int32_t emax() const noexcept { return emax_; }

  /// {j1 j2 J12; j3 J J23} for single-particle j1, j2, j3 (odd 2j in [1, 2
  /// emax + 1]), integer J12, J23 (`.twice` even, in [0, 2 (2 emax + 1)]) and
  /// half-integer J (odd 2J in [1, 6 emax + 3]); 0.0 outside these ranges and
  /// when a triangle fails. O(1), no allocation, no exceptions.
  IMSRG_INLINE_FUNCTION double operator()(TwoJ j1, TwoJ j2, TwoJ J12, TwoJ j3,
                                          TwoJ J, TwoJ J23) const noexcept {
    const std::int32_t tmax = 2 * emax_ + 1;
    if (data_ == nullptr || !OddIn(j1.twice, tmax) || !OddIn(j2.twice, tmax) ||
        !OddIn(j3.twice, tmax) || !OddIn(J.twice, 6 * emax_ + 3) ||
        !EvenIn(J12.twice, 2 * tmax) || !EvenIn(J23.twice, 2 * tmax)) {
      return 0.0;
    }
    const std::int64_t nj = emax_ + 1;
    const std::int64_t nJ = 2 * emax_ + 2;
    const std::int64_t nJ3 = 3 * emax_ + 2;
    const std::int64_t i =
        (((((j1.twice - 1) / 2 * nj + (j2.twice - 1) / 2) * nJ +
           J12.twice / 2) *
              nj +
          (j3.twice - 1) / 2) *
             nJ3 +
         (J.twice - 1) / 2) *
            nJ +
        J23.twice / 2;
    return data_[i];
  }

  /// sqrt(2J + 1) for an integer two-body J in [0, 2 emax + 1] (no check).
  IMSRG_INLINE_FUNCTION double Hat(std::int32_t J) const noexcept {
    return hat_[J];
  }

 private:
  /// True if `t` is odd and in [1, tmax].
  static IMSRG_INLINE_FUNCTION constexpr bool OddIn(std::int32_t t,
                                                    std::int32_t tmax) {
    return t >= 1 && t <= tmax && (t & 1) == 1;
  }
  /// True if `t` is even and in [0, tmax].
  static IMSRG_INLINE_FUNCTION constexpr bool EvenIn(std::int32_t t,
                                                     std::int32_t tmax) {
    return t >= 0 && t <= tmax && (t & 1) == 0;
  }

  const double* data_ = nullptr;  ///< The 6j values (layout: see SixJ3Table).
  const double* hat_ = nullptr;  ///< hat_[J] = sqrt(2J + 1), J = 0..2 emax + 1.
  std::int32_t emax_ = -1;       ///< Largest single-particle shell.
};
static_assert(std::is_trivially_copyable_v<SixJ3View>);

/// Dense table of {j1 j2 J12; j3 J J23}. Layout: index ((((i1 nj + i2) nJ +
/// J12) nj + i3) nJ3 + iJ) nJ + J23 with i = (2j - 1) / 2, iJ = (2J - 1) / 2,
/// nj = emax + 1, nJ = 2 emax + 2, nJ3 = 3 emax + 2; nj^3 nJ^2 nJ3 doubles
/// (0.36 MB at emax 3, 451 MB at emax 12). Built once on the host with
/// `angmom::Wigner6j` for every entry whose four triangles hold (others 0.0),
/// then copied to the device; immutable afterwards. Needs an initialized
/// `exec::Runtime`.
class SixJ3Table {
 public:
  using View = SixJ3View;

  /// Builds the table for single-particle 2j <= 2 emax + 1. Cost O(nj^3 nJ^2
  /// nJ3) symbol evaluations at most. Precondition: emax >= 0.
  explicit SixJ3Table(int emax);

  SixJ3Table(const SixJ3Table&) = default;
  SixJ3Table& operator=(const SixJ3Table&) = default;
  SixJ3Table(SixJ3Table&&) = default;
  SixJ3Table& operator=(SixJ3Table&&) = default;
  ~SixJ3Table() = default;

  /// The largest single-particle shell.
  int emax() const noexcept { return emax_; }
  /// Bytes of the 6j values plus the hat table.
  std::int64_t size_bytes() const noexcept;
  /// View of the host mirror.
  View host_view() const noexcept;
  /// View of the device data.
  View device_view() const noexcept;

  /// Number of doubles of the 6j part for `emax` (no table built).
  static std::int64_t ValueCount(int emax) noexcept;

  /// Host lookup, see `SixJ3View::operator()` (0.0 outside the ranges and for
  /// failed triangles). O(1).
  double operator()(TwoJ j1, TwoJ j2, TwoJ J12, TwoJ j3, TwoJ J,
                    TwoJ J23) const noexcept {
    return host_view()(j1, j2, J12, j3, J, J23);
  }

 private:
  int emax_;
  exec::DeviceBuffer<double> values_;
  exec::DeviceBuffer<double> hat_;
};

}  // namespace imsrg::angmom

#endif  // IMSRG_ANGMOM_SIXJ3_TABLE_H_
