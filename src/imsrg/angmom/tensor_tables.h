// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_ANGMOM_TENSOR_TABLES_H_
#define IMSRG_ANGMOM_TENSOR_TABLES_H_
/// @file
/// Dense recoupling tables for operators of a fixed spherical tensor rank
/// lambda (milestone M15): the 6j family {J1 J2 lambda; ja jb jc} of the
/// scalar-tensor commutators and the tensor normal ordering, and the 9j family
/// {A B C; D E F; G H lambda} of the tensor Pandya transformation. Built on
/// the host with `Wigner6j` / `Wigner9j`, stored in `exec::DeviceBuffer`s,
/// read in kernels through trivially copyable views (design/12 sections 4.4
/// and 5).
#include <cstdint>
#include <type_traits>

#include "imsrg/base/device.h"
#include "imsrg/base/types.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::angmom {

/// Read-only view of a `LambdaSixJTable`.
struct LambdaSixJView {
  const double* data = nullptr;  ///< nJ^2 nj^3 values (see the table).
  std::int32_t emax = -1;        ///< Single-particle 2j <= 2 emax + 1.

  /// {J1 J2 lambda; ja jb jc} for integer J1, J2 in [0, 2 emax + 1] (`.twice`
  /// even) and single-particle ja, jb, jc (odd 2j <= 2 emax + 1); 0.0 for
  /// arguments outside that range or of the wrong parity. O(1).
  IMSRG_INLINE_FUNCTION double operator()(TwoJ J1, TwoJ J2, TwoJ ja, TwoJ jb,
                                          TwoJ jc) const noexcept {
    const std::int32_t nj = emax + 1;
    const std::int32_t nJ = 2 * emax + 2;
    if (data == nullptr || J1.twice < 0 || J2.twice < 0 ||
        (J1.twice & 1) != 0 || (J2.twice & 1) != 0 || J1.twice / 2 >= nJ ||
        J2.twice / 2 >= nJ || (ja.twice & 1) == 0 || (jb.twice & 1) == 0 ||
        (jc.twice & 1) == 0 || ja.twice < 1 || jb.twice < 1 || jc.twice < 1 ||
        ja.twice > 2 * emax + 1 || jb.twice > 2 * emax + 1 ||
        jc.twice > 2 * emax + 1)
      return 0.0;
    const std::int64_t i =
        (((static_cast<std::int64_t>(J1.twice / 2) * nJ + J2.twice / 2) * nj +
          (ja.twice - 1) / 2) *
             nj +
         (jb.twice - 1) / 2) *
            nj +
        (jc.twice - 1) / 2;
    return data[i];
  }
};
static_assert(std::is_trivially_copyable_v<LambdaSixJView>);

/// Dense table of {J1 J2 lambda; ja jb jc} for one lambda: integer J1, J2 in
/// [0, 2 emax + 1], single-particle j in {1/2, ..., emax + 1/2}. Layout
/// (((J1 nJ + J2) nj + ia) nj + ib) nj + ic with ia = (2ja - 1)/2, nJ = 2 emax
/// + 2, nj = emax + 1 (nJ^2 nj^3 doubles: 0.7 MB at emax 10). Value type.
class LambdaSixJTable {
 public:
  /// Builds the table for rank `lambda` and shell `emax`. Preconditions:
  /// emax >= 0, lambda >= 0. Cost O(nJ^2 nj^3) 6j evaluations.
  LambdaSixJTable(int emax, int lambda);

  LambdaSixJTable(const LambdaSixJTable&) = default;
  LambdaSixJTable& operator=(const LambdaSixJTable&) = default;
  LambdaSixJTable(LambdaSixJTable&&) = default;
  LambdaSixJTable& operator=(LambdaSixJTable&&) = default;

  /// The rank lambda.
  int lambda() const noexcept { return lambda_; }
  /// The shell emax.
  int emax() const noexcept { return emax_; }
  /// View of the host mirror.
  LambdaSixJView host_view() const noexcept;
  /// View of the device data.
  LambdaSixJView device_view() const noexcept;

 private:
  int emax_;
  int lambda_;
  exec::DeviceBuffer<double> values_;
};

/// Read-only view of a `NineJTable`.
struct NineJView {
  const std::int64_t* offsets = nullptr;  ///< nj^4 (A, B, D, E) box offsets.
  const double* data = nullptr;           ///< The values of all boxes.
  std::int32_t emax = -1;                 ///< Single-particle 2j bound.

  /// {A B C; D E F; G H lambda} with single-particle A, B, D, E (odd 2j <= 2
  /// emax + 1) and integer C, F, G, H; 0.0 if C, F, G or H lies outside its
  /// triangle range (C in A x B, F in D x E, G in A x D, H in B x E), if
  /// any argument has the wrong parity, or if |value| was below the table's
  /// skip threshold. O(1).
  IMSRG_INLINE_FUNCTION double operator()(TwoJ A, TwoJ B, TwoJ C, TwoJ D,
                                          TwoJ E, TwoJ F, TwoJ G,
                                          TwoJ H) const noexcept {
    const std::int32_t tmax = 2 * emax + 1;
    if (data == nullptr || (A.twice & 1) == 0 || (B.twice & 1) == 0 ||
        (D.twice & 1) == 0 || (E.twice & 1) == 0 || A.twice > tmax ||
        B.twice > tmax || D.twice > tmax || E.twice > tmax || A.twice < 1 ||
        B.twice < 1 || D.twice < 1 || E.twice < 1 || (C.twice & 1) != 0 ||
        (F.twice & 1) != 0 || (G.twice & 1) != 0 || (H.twice & 1) != 0)
      return 0.0;
    const std::int32_t c0 = Abs(A.twice - B.twice) / 2;
    const std::int32_t nc = (A.twice + B.twice) / 2 - c0 + 1;
    const std::int32_t f0 = Abs(D.twice - E.twice) / 2;
    const std::int32_t nf = (D.twice + E.twice) / 2 - f0 + 1;
    const std::int32_t g0 = Abs(A.twice - D.twice) / 2;
    const std::int32_t ng = (A.twice + D.twice) / 2 - g0 + 1;
    const std::int32_t h0 = Abs(B.twice - E.twice) / 2;
    const std::int32_t nh = (B.twice + E.twice) / 2 - h0 + 1;
    const std::int32_t c = C.twice / 2 - c0;
    const std::int32_t f = F.twice / 2 - f0;
    const std::int32_t g = G.twice / 2 - g0;
    const std::int32_t h = H.twice / 2 - h0;
    if (c < 0 || c >= nc || f < 0 || f >= nf || g < 0 || g >= ng || h < 0 ||
        h >= nh)
      return 0.0;
    const std::int32_t nj = emax + 1;
    const std::int64_t q = ((static_cast<std::int64_t>((A.twice - 1) / 2) * nj +
                             (B.twice - 1) / 2) *
                                nj +
                            (D.twice - 1) / 2) *
                               nj +
                           (E.twice - 1) / 2;
    return data[offsets[q] +
                ((static_cast<std::int64_t>(c) * nf + f) * ng + g) * nh + h];
  }

 private:
  static IMSRG_INLINE_FUNCTION constexpr std::int32_t Abs(
      std::int32_t x) noexcept {
    return x < 0 ? -x : x;
  }
};
static_assert(std::is_trivially_copyable_v<NineJView>);

/// The 9j symbols {A B C; D E F; G H lambda} for one lambda with
/// single-particle A, B, D, E in {1/2, ..., emax + 1/2} and every integer C,
/// F, G, H inside its triangle range (see `NineJView`): one dense box per (A,
/// B, D, E). Entries with |9j| < `skip_threshold` are stored as 0, which is
/// how imsrg++'s tensor commutators skip small 9j symbols (`|9j| < 1e-10` in
/// imsrg-devel TensorCommutators.cc, 1e-8 in the 79e9330 fork). Size about
/// 35 MB at emax 6, 1.1 GB at emax 10 (host and device copies on GPUs).
/// Value type.
class NineJTable {
 public:
  /// Builds the table. Preconditions: emax >= 0, lambda >= 0,
  /// skip_threshold >= 0. Cost: one `Wigner9j` per entry (seconds at emax 6).
  NineJTable(int emax, int lambda, double skip_threshold);

  NineJTable(const NineJTable&) = default;
  NineJTable& operator=(const NineJTable&) = default;
  NineJTable(NineJTable&&) = default;
  NineJTable& operator=(NineJTable&&) = default;

  /// The rank lambda.
  int lambda() const noexcept { return lambda_; }
  /// The skip threshold.
  double skip_threshold() const noexcept { return threshold_; }
  /// View of the host mirror.
  NineJView host_view() const noexcept;
  /// View of the device data.
  NineJView device_view() const noexcept;

 private:
  int emax_;
  int lambda_;
  double threshold_;
  exec::DeviceBuffer<std::int64_t> offsets_;
  exec::DeviceBuffer<double> values_;
};

}  // namespace imsrg::angmom

#endif  // IMSRG_ANGMOM_TENSOR_TABLES_H_
