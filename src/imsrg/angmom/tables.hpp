#pragma once
/// @file
/// Dense, immutable, device-copyable angular-momentum tables (design/12 section 4.4): Pandya 6j
/// symbols, LS-jj 9j recoupling coefficients and Talmi-Moshinsky brackets.
///
/// Each table is built once on the host (serial loops; the cost is stated per class), then
/// stored in `exec::DeviceBuffer`s and never modified. A table offers `host_view()` and
/// `device_view()`: small trivially copyable `View` structs of raw pointers that kernels capture
/// by value; `View::operator()` is `IMSRG_INLINE_FUNCTION`, free of exceptions, and returns 0.0
/// for out-of-range or invalid arguments (cheap integer checks only). The host mirrors are
/// created in the constructors, so const member functions are thread-safe afterwards. Building
/// and destroying a table needs an initialized `exec::Runtime`. Views stay valid while the table
/// is alive and unmoved-from. This header does not include Kokkos.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "imsrg/angmom/moshinsky.hpp"
#include "imsrg/base/device.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::angmom {

/// Read-only view of a `SixJTable`; see there.
struct SixJView {
  const double* data = nullptr;
  std::int32_t emax = -1;

  /// 6j symbol {ja jb J; jc jd J'} (the same value as `wigner_6j(ja, jb, J, jc, jd, Jp)`) for
  /// single-particle ja..jd (odd 2j in [1, 2 emax + 1]) and integer J, J' (`.twice` even, in
  /// [0, 2 (2 emax + 1)]); 0.0 outside this range. O(1).
  IMSRG_INLINE_FUNCTION double operator()(TwoJ ja, TwoJ jb, TwoJ J, TwoJ jc, TwoJ jd,
                                          TwoJ Jp) const noexcept {
    const std::int32_t nj = emax + 1;
    const std::int32_t nJ = 2 * emax + 2;
    const std::int32_t tmax = 2 * emax + 1;
    if (data == nullptr || ja.twice < 1 || ja.twice > tmax || jb.twice < 1 || jb.twice > tmax ||
        jc.twice < 1 || jc.twice > tmax || jd.twice < 1 || jd.twice > tmax || (ja.twice & 1) == 0 ||
        (jb.twice & 1) == 0 || (jc.twice & 1) == 0 || (jd.twice & 1) == 0 || J.twice < 0 ||
        Jp.twice < 0 || (J.twice & 1) != 0 || (Jp.twice & 1) != 0 || J.twice / 2 >= nJ ||
        Jp.twice / 2 >= nJ) {
      return 0.0;
    }
    const std::int64_t i =
        ((((static_cast<std::int64_t>((ja.twice - 1) / 2) * nj + (jb.twice - 1) / 2) * nJ +
           J.twice / 2) *
              nj +
          (jc.twice - 1) / 2) *
             nj +
         (jd.twice - 1) / 2) *
            nJ +
        Jp.twice / 2;
    return data[i];
  }
};
static_assert(std::is_trivially_copyable_v<SixJView>);

/// Dense table of the Pandya-type 6j symbols {ja jb J; jc jd J'} for single-particle 2j in
/// {1, 3, ..., 2 emax + 1} and integer J, J' in [0, 2 emax + 1] (the coverage of imsrg++
/// `SixJCache_112112`). Layout: index ((((ia nj + ib) nJ + J) nj + ic) nj + id) nJ + J' with
/// ia = (2ja - 1)/2, nj = emax + 1, nJ = 2 emax + 2 (so nj^4 nJ^2 doubles: 154 MB at emax 12,
/// 3.8 MB at emax 6). Entries whose triangles fail are 0.
class SixJTable {
 public:
  using View = SixJView;

  /// Builds the table; serial, one `wigner_6j` per entry with satisfied triangles (cost about
  /// O(nj^4 nJ^2) in the worst case; a few ms at emax 6 in Debug). Precondition: emax >= 0.
  explicit SixJTable(int emax);

  /// The maximum single-particle shell: 2j <= 2 emax + 1.
  int emax() const noexcept { return emax_; }
  /// Number of stored doubles.
  std::size_t size() const noexcept { return values_.size(); }
  /// View of the host mirror.
  View host_view() const noexcept;
  /// View of the device data (the same memory as the host view on host backends).
  View device_view() const noexcept;
  /// Host lookup, see `SixJView::operator()`.
  double operator()(TwoJ ja, TwoJ jb, TwoJ J, TwoJ jc, TwoJ jd, TwoJ Jp) const noexcept {
    return host_view()(ja, jb, J, jc, jd, Jp);
  }

 private:
  int emax_;
  exec::DeviceBuffer<double> values_;
};

/// Read-only view of an `NineJLSjjTable`; see there.
struct NineJLSjjView {
  const double* data = nullptr;
  std::int32_t lmax = -1;

  /// LS-jj coefficient `norm_9j(la, 1/2, ja, lb, 1/2, jb, L, S, J)` for 0 <= la, lb <= lmax,
  /// ja = la +- 1/2 and jb = lb +- 1/2 (each >= 1/2), L in [0, 2 lmax], S in {0, 1}, J in
  /// [0, 2 lmax + 1] (J given doubled and even); 0.0 outside this range. O(1).
  IMSRG_INLINE_FUNCTION double operator()(int la, TwoJ ja, int lb, TwoJ jb, int L, int S,
                                          TwoJ J) const noexcept {
    const std::int32_t da = ja.twice - 2 * la + 1;  // 0 for j = l - 1/2, 2 for j = l + 1/2
    const std::int32_t db = jb.twice - 2 * lb + 1;
    if (data == nullptr || la < 0 || la > lmax || lb < 0 || lb > lmax || (da != 0 && da != 2) ||
        (db != 0 && db != 2) || ja.twice < 1 || jb.twice < 1 || L < 0 || L > 2 * lmax || S < 0 ||
        S > 1 || J.twice < 0 || (J.twice & 1) != 0 || J.twice / 2 > 2 * lmax + 1) {
      return 0.0;
    }
    const std::int32_t nL = 2 * lmax + 1;
    const std::int32_t nJ = 2 * lmax + 2;
    const std::int64_t i =
        (((((static_cast<std::int64_t>(la) * 2 + da / 2) * (lmax + 1) + lb) * 2 + db / 2) * nL +
          L) *
             2 +
         S) *
            nJ +
        J.twice / 2;
    return data[i];
  }
};
static_assert(std::is_trivially_copyable_v<NineJLSjjView>);

/// Dense table of LS-jj recoupling coefficients <(la lb) L, (1/2 1/2) S; J | (la 1/2) ja,
/// (lb 1/2) jb; J> = `norm_9j(la, 1/2, ja, lb, 1/2, jb, L, S, J)` (imsrg++ `NormNineJ`). Layout:
/// (((((la 2 + sja) (lmax+1) + lb) 2 + sjb) nL + L) 2 + S) nJ + J, sja = 0 for ja = la - 1/2,
/// 1 for la + 1/2, nL = 2 lmax + 1, nJ = 2 lmax + 2. Entries for ja = -1/2 (la = 0, sja = 0)
/// and invalid couplings are 0.
class NineJLSjjTable {
 public:
  using View = NineJLSjjView;

  /// Builds the table; serial, one `norm_9j` per entry (about 16 (lmax+1)^2 (2 lmax+1)
  /// (2 lmax + 2) evaluations; well under 0.1 s at lmax 6). Precondition: lmax >= 0.
  explicit NineJLSjjTable(int lmax);

  /// The maximum orbital angular momentum l.
  int lmax() const noexcept { return lmax_; }
  /// Number of stored doubles.
  std::size_t size() const noexcept { return values_.size(); }
  /// View of the host mirror.
  View host_view() const noexcept;
  /// View of the device data.
  View device_view() const noexcept;
  /// Host lookup, see `NineJLSjjView::operator()`.
  double operator()(int la, TwoJ ja, int lb, TwoJ jb, int L, int S, TwoJ J) const noexcept {
    return host_view()(la, ja, lb, jb, L, S, J);
  }

 private:
  int lmax_;
  exec::DeviceBuffer<double> values_;
};

/// Read-only view of a `MoshinskyTable`; see there.
struct MoshinskyView {
  const double* data = nullptr;            ///< concatenated K(E) x K(E) blocks
  const std::int32_t* position = nullptr;  ///< (n1,l1,n2,l2) -> local index within its E, or -1
  const std::int64_t* offset = nullptr;    ///< block start per (E, lambda): [E (e2max+1) + lambda]
  const std::int32_t* block_size = nullptr;  ///< K(E)
  std::int32_t e2max = -1;

  /// Bracket `moshinsky(cm, rel, a, b, lambda)`; 0.0 if energy is not conserved, a quantum
  /// number is negative or beyond e2max, or lambda is outside [0, E]. O(1).
  IMSRG_INLINE_FUNCTION double operator()(HoNL cm, HoNL rel, HoNL a, HoNL b,
                                          int lambda) const noexcept {
    const std::int32_t E = 2 * a.n + a.l + 2 * b.n + b.l;
    if (data == nullptr || a.n < 0 || a.l < 0 || b.n < 0 || b.l < 0 || cm.n < 0 || cm.l < 0 ||
        rel.n < 0 || rel.l < 0 || E > e2max || 2 * cm.n + cm.l + 2 * rel.n + rel.l != E ||
        lambda < 0 || lambda > E) {
      return 0.0;
    }
    const std::int32_t nn = e2max / 2 + 1;
    const std::int32_t nl = e2max + 1;
    const std::int32_t ilab = position[((a.n * nl + a.l) * nn + b.n) * nl + b.l];
    const std::int32_t icm = position[((cm.n * nl + cm.l) * nn + rel.n) * nl + rel.l];
    if (ilab < 0 || icm < 0) {
      return 0.0;
    }
    return data[offset[E * (e2max + 1) + lambda] + static_cast<std::int64_t>(icm) * block_size[E] +
                ilab];
  }
};
static_assert(std::is_trivially_copyable_v<MoshinskyView>);

/// All Talmi-Moshinsky brackets with HO energy E = 2N + L + 2n + l = 2 n1 + l1 + 2 n2 + l2 <=
/// e2max. For each E the tuples (n1, l1, n2, l2) with 2 n1 + l1 + 2 n2 + l2 = E are listed in a
/// fixed order (l1 ascending, then n1 ascending, then l2 ascending); K(E) is their number. The
/// cm-rel tuple (N, L, n, l) uses the same list. For each (E, lambda <= E) a dense K(E) x K(E)
/// block M[i_cmrel][i_lab] is stored (many entries are 0). An int32 map over (n1, l1, n2, l2)
/// gives the local index. Memory: `value_count(e2max)` doubles (see `size_bytes()`).
class MoshinskyTable {
 public:
  using View = MoshinskyView;

  /// Builds the table, serially, with one `moshinsky()` call per candidate entry (entries that
  /// violate parity or the lambda triangles are skipped). Cost is dominated by O(e^5) per
  /// bracket; measured about 0.1 s at e2max 12 in a Debug build (parity and triangle pruning).
  /// Precondition: e2max >= 0.
  explicit MoshinskyTable(int e2max);

  /// Largest HO energy E stored.
  int e2max() const noexcept { return e2max_; }
  /// Number of tuples K(E) at energy `E` (0 if E is outside [0, e2max]).
  int block_size(int E) const noexcept;
  /// Number of doubles stored for `e2max` (computed without building the table).
  static std::int64_t value_count(int e2max) noexcept;
  /// Total memory of the table's data, in bytes (values, maps and offsets).
  std::size_t size_bytes() const noexcept;
  /// View of the host mirror.
  View host_view() const noexcept;
  /// View of the device data.
  View device_view() const noexcept;
  /// Host lookup, see `MoshinskyView::operator()`.
  double operator()(HoNL cm, HoNL rel, HoNL a, HoNL b, int lambda) const noexcept {
    return host_view()(cm, rel, a, b, lambda);
  }

 private:
  int e2max_;
  exec::DeviceBuffer<double> values_;
  exec::DeviceBuffer<std::int32_t> position_;
  exec::DeviceBuffer<std::int64_t> offset_;
  exec::DeviceBuffer<std::int32_t> block_size_;
};

}  // namespace imsrg::angmom
