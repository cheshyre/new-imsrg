#pragma once

/// @file
/// Trivially copyable, Kokkos-free views of the data the commutator kernels read (design/12
/// section 5). The views hold raw pointers into device memory; kernels capture them by value.
/// Nothing here includes Kokkos, so the structs can be built in ordinary TUs.

#include <cstdint>
#include <type_traits>

#include "imsrg/base/constants.hpp"
#include "imsrg/base/device.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/basis/device_tables.hpp"
#include "imsrg/basis/two_body_channels.hpp"

namespace imsrg::commutator::detail {

/// CSR of the one-body channels: for orbit p, the orbits with the same (l, 2j, 2tz) as p (p
/// included), ascending, are `orbits[offsets[p] .. offsets[p + 1])`. The same sets index the
/// non-zero elements of a scalar, isospin-conserving one-body operator.
struct OneBodyChannelView {
  const std::int32_t* offsets = nullptr;  ///< Size num_orbits + 1.
  const std::int32_t* orbits = nullptr;   ///< Concatenated orbit lists.
};
static_assert(std::is_trivially_copyable_v<OneBodyChannelView>);

/// Read-only view of the two-body part of a scalar operator: the basis tables, the offset of each
/// channel's dense column-major N x N block in `data`, and the data itself. `data == nullptr`
/// represents an unallocated part, which reads as zero.
struct TwoBodyView {
  basis::TwoBodyTableView tb;                  ///< Channel and ket tables.
  const std::int64_t* block_offset = nullptr;  ///< Size num_channels (+ 1 with the total).
  const double* data = nullptr;                ///< Stored (normalized) elements, or nullptr.

  /// Stored normalized element <pq; J|V|rs; J> for orbits in ANY order (`op::TwoBody::
  /// get_normalized`): a pair with first orbit > second is mapped to the ket (min, max) and
  /// multiplied by `basis::exchange_phase`. Zero if `data` is null, if a ket does not exist for
  /// this J (triangle, Pauli), or if the two kets lie in different channels. `two_J` is 2J.
  /// O(1).
  IMSRG_INLINE_FUNCTION double get_normalized(int p, int q, int r, int s,
                                              int two_J) const noexcept {
    if (data == nullptr) return 0.0;
    const bool swap_bra = p > q;
    const bool swap_ket = r > s;
    const int a = swap_bra ? q : p;
    const int b = swap_bra ? p : q;
    const int c = swap_ket ? s : r;
    const int d = swap_ket ? r : s;
    const std::int32_t ib = tb.local_index(tb.ket_index(OrbitIndex{a}, OrbitIndex{b}), TwoJ{two_J});
    if (ib < 0) return 0.0;
    const std::int32_t ik = tb.local_index(tb.ket_index(OrbitIndex{c}, OrbitIndex{d}), TwoJ{two_J});
    if (ik < 0) return 0.0;
    // Both kets exist for this J, so each lies in the channel (J, parity, Tz) fixed by its orbits.
    const std::int32_t par_bra = (tb.orbit_l[a] + tb.orbit_l[b]) % 2;
    const std::int32_t par_ket = (tb.orbit_l[c] + tb.orbit_l[d]) % 2;
    if (par_bra != par_ket ||
        tb.orbit_tz2[a] + tb.orbit_tz2[b] != tb.orbit_tz2[c] + tb.orbit_tz2[d])
      return 0.0;
    const std::int32_t ch = tb.channel_of(OrbitIndex{a}, OrbitIndex{b}, TwoJ{two_J}).value();
    int phase = 1;
    if (swap_bra)
      phase *= basis::exchange_phase(TwoJ{tb.orbit_j2[p]}, TwoJ{tb.orbit_j2[q]}, TwoJ{two_J});
    if (swap_ket)
      phase *= basis::exchange_phase(TwoJ{tb.orbit_j2[r]}, TwoJ{tb.orbit_j2[s]}, TwoJ{two_J});
    const std::int64_t n = tb.channel_num_kets[ch];
    return phase * data[block_offset[ch] + ib + static_cast<std::int64_t>(ik) * n];
  }

  /// Unnormalized element (`op::TwoBody::get_unnormalized`, imsrg++ `GetTBME_J`): the normalized
  /// element times sqrt(2) for each of the pairs (p, q) and (r, s) with identical orbits. O(1).
  IMSRG_INLINE_FUNCTION double get_unnormalized(int p, int q, int r, int s,
                                                int two_J) const noexcept {
    double v = get_normalized(p, q, r, s, two_J);
    if (p == q) v *= constants::sqrt2;
    if (r == s) v *= constants::sqrt2;
    return v;
  }
};
static_assert(std::is_trivially_copyable_v<TwoBodyView>);

/// Index k of the block with `offsets[k] <= e < offsets[k + 1]` among `num_blocks` blocks
/// (`offsets` has num_blocks + 1 entries, non-decreasing; empty blocks are skipped).
/// Precondition: 0 <= e < offsets[num_blocks]. Binary search, O(log num_blocks).
IMSRG_INLINE_FUNCTION int find_block(const std::int64_t* offsets, int num_blocks,
                                     std::int64_t e) noexcept {
  int lo = 0;
  int hi = num_blocks - 1;  // invariant: offsets[lo] <= e, answer in [lo, hi]
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (offsets[mid] <= e) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

}  // namespace imsrg::commutator::detail
