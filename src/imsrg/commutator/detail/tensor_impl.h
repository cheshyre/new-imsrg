// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_TENSOR_IMPL_H_
#define IMSRG_COMMUTATOR_DETAIL_TENSOR_IMPL_H_

/// @file
/// Internal data and device views of the scalar-tensor commutators
/// (`commutator/tensor.h`). Not part of the public API.

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "imsrg/angmom/tables.h"
#include "imsrg/angmom/tensor_tables.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/device.h"
#include "imsrg/basis/device_tables.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::detail {

/// (-1)^x for an integer x (any sign). O(1).
IMSRG_INLINE_FUNCTION constexpr int Phase(int x) noexcept {
  return (x % 2 == 0) ? 1 : -1;
}

/// Device view of the stored blocks of a tensor (or scalar) two-body part:
/// element access for orbits in any order and any channel pair, with the
/// hermiticity relation for ch_bra > ch_ket (`op::TwoBody::get_normalized(a,
/// b, c, d, J_bra, J_ket)`). Trivially copyable.
struct TensorTwoBodyView {
  basis::TwoBodyTableView tb;  ///< Channel and ket tables.
  /// Offset of the stored block (ch_bra, ch_ket) at ch_bra * num_channels +
  /// ch_ket, -1 if not stored (num_channels^2 entries).
  const std::int64_t* block_offset = nullptr;
  const double* data = nullptr;  ///< Stored normalized elements, or null.
  int herm = 1;                  ///< +1 Hermitian, -1 anti-Hermitian.

  /// Normalized <pq; J1||V||rs; J2>, 0 if a ket does not exist or the block
  /// pair is not stored (either orientation). `two_J1`, `two_J2` are 2J. O(1).
  IMSRG_INLINE_FUNCTION double get_normalized(int p, int q, int r, int s,
                                              int two_J1,
                                              int two_J2) const noexcept {
    if (data == nullptr) return 0.0;
    const bool swap_bra = p > q;
    const bool swap_ket = r > s;
    const int a = swap_bra ? q : p;
    const int b = swap_bra ? p : q;
    const int c = swap_ket ? s : r;
    const int d = swap_ket ? r : s;
    const std::int32_t ib = tb.local_index(
        tb.ket_index(OrbitIndex{a}, OrbitIndex{b}), TwoJ{two_J1});
    if (ib < 0) return 0.0;
    const std::int32_t ik = tb.local_index(
        tb.ket_index(OrbitIndex{c}, OrbitIndex{d}), TwoJ{two_J2});
    if (ik < 0) return 0.0;
    const std::int32_t cb =
        tb.channel_of(OrbitIndex{a}, OrbitIndex{b}, TwoJ{two_J1}).value();
    const std::int32_t ck =
        tb.channel_of(OrbitIndex{c}, OrbitIndex{d}, TwoJ{two_J2}).value();
    int phase = 1;
    if (swap_bra)
      phase *= basis::ExchangePhase(TwoJ{tb.orbit_j2[p]}, TwoJ{tb.orbit_j2[q]},
                                    TwoJ{two_J1});
    if (swap_ket)
      phase *= basis::ExchangePhase(TwoJ{tb.orbit_j2[r]}, TwoJ{tb.orbit_j2[s]},
                                    TwoJ{two_J2});
    const std::int64_t nch = tb.num_channels;
    if (cb <= ck) {
      const std::int64_t off = block_offset[cb * nch + ck];
      if (off < 0) return 0.0;
      const std::int64_t nb = tb.channel_num_kets[cb];
      return phase * data[off + ib + static_cast<std::int64_t>(ik) * nb];
    }
    const std::int64_t off = block_offset[ck * nch + cb];
    if (off < 0) return 0.0;
    const std::int64_t nk = tb.channel_num_kets[ck];
    return herm * Phase((two_J1 - two_J2) / 2) * phase *
           data[off + ik + static_cast<std::int64_t>(ib) * nk];
  }

  /// imsrg++ `GetTBME_J`: the normalized element times sqrt(2) for each pair
  /// of identical orbits. O(1).
  IMSRG_INLINE_FUNCTION double get_unnormalized(int p, int q, int r, int s,
                                                int two_J1,
                                                int two_J2) const noexcept {
    double v = get_normalized(p, q, r, s, two_J1, two_J2);
    if (p == q) v *= constants::kSqrt2;
    if (r == s) v *= constants::kSqrt2;
    return v;
  }
};
static_assert(std::is_trivially_copyable_v<TensorTwoBodyView>);

/// Compressed lists: entries [offsets[i], offsets[i + 1]) of `items` belong to
/// row i. Trivially copyable.
struct CsrView {
  const std::int32_t* offsets = nullptr;
  const std::int32_t* items = nullptr;
};
static_assert(std::is_trivially_copyable_v<CsrView>);

/// Owner of a CSR list (host-built, device copy).
struct Csr {
  exec::DeviceBuffer<std::int32_t> offsets;
  exec::DeviceBuffer<std::int32_t> items;
  /// Device view.
  CsrView view() const {
    return CsrView{offsets.device_data(), items.device_data()};
  }
};

/// Builds a CSR list from host rows.
Csr MakeCsr(const std::vector<std::vector<std::int32_t>>& rows);

/// The cross-coupled ("Pandya") pair structure of a basis: channel c = (J,
/// parity, tz_cc) with tz_cc = |tz2_a - tz2_b| / 2, index 4 J + 2 tz_cc +
/// parity; its pairs are ALL ordered pairs (a, b) with triangle(j_a, j_b, J),
/// parity (-1)^(l_a + l_b) and that tz_cc (a == b included), sorted by (a,
/// b). Independent of occupations and of the operators.
struct CrossCoupled {
  std::int32_t num_channels = 0;
  std::int32_t max_J = 0;
  std::vector<std::int32_t> J;       ///< Per channel.
  std::vector<std::int32_t> parity;  ///< Per channel.
  std::vector<std::int32_t> tz;      ///< Per channel (0 or 1).
  std::vector<std::int32_t> offset;  ///< Pair offsets, num_channels + 1.
  std::vector<std::int32_t> pair_a;  ///< First orbit per pair.
  std::vector<std::int32_t> pair_b;  ///< Second orbit per pair.
  /// Local index of the ordered pair (a, b) in its channel at J:
  /// local[(a * norb + b) * (max_J + 1) + J], -1 if not a pair.
  std::vector<std::int32_t> local;
  /// Device copies of `offset`, `pair_a`, `pair_b`, `local`.
  exec::DeviceBuffer<std::int32_t> d_offset, d_pair_a, d_pair_b, d_local;
};

/// Device view of `CrossCoupled`.
struct CrossCoupledView {
  const std::int32_t* offset = nullptr;
  const std::int32_t* pair_a = nullptr;
  const std::int32_t* pair_b = nullptr;
  const std::int32_t* local = nullptr;
  std::int32_t num_orbits = 0;
  std::int32_t max_J = 0;

  /// Local index of (a, b) at J, -1 if not a pair. O(1).
  IMSRG_INLINE_FUNCTION std::int32_t local_index(int a, int b,
                                                 int J) const noexcept {
    if (J < 0 || J > max_J) return -1;
    return local[(static_cast<std::int64_t>(a) * num_orbits + b) * (max_J + 1) +
                 J];
  }
};
static_assert(std::is_trivially_copyable_v<CrossCoupledView>);

/// Channel index of the ordered pair (a, b) at J. O(1).
IMSRG_INLINE_FUNCTION int CrossCoupledChannel(const std::int32_t* l,
                                              const std::int32_t* tz2, int a,
                                              int b, int J) noexcept {
  const int par = (l[a] + l[b]) % 2;
  const int t = (tz2[a] == tz2[b]) ? 0 : 1;
  return 4 * J + 2 * t + par;
}

/// Entries of a list set: entry e of row r in [offsets[r], offsets[r + 1]).
/// Each entry is "one-body element (from, to) times `coef` times the element
/// in row/column `m` of a two-body block". Trivially copyable.
struct EntryListView {
  const std::int32_t* offsets = nullptr;
  const std::int32_t* m = nullptr;
  const std::int32_t* from = nullptr;
  const std::int32_t* to = nullptr;
  const double* coef = nullptr;
};
static_assert(std::is_trivially_copyable_v<EntryListView>);

/// Owner of an `EntryListView` (host-built, device copy).
struct EntryList {
  exec::DeviceBuffer<std::int32_t> offsets, m, from, to;
  exec::DeviceBuffer<double> coef;
  /// Device view.
  EntryListView view() const {
    return EntryListView{offsets.device_data(), m.device_data(),
                         from.device_data(), to.device_data(),
                         coef.device_data()};
  }
};

/// Host-side builder of an `EntryList`: add the entries of row 0, 1, ... in
/// order, calling `EndRow` after each row; `Finish` uploads the lists.
class EntryListBuilder {
 public:
  /// Appends an entry to the current row.
  void Add(std::int32_t m, std::int32_t from, std::int32_t to, double coef) {
    m_.push_back(m);
    from_.push_back(from);
    to_.push_back(to);
    coef_.push_back(coef);
  }
  /// Closes the current row.
  void EndRow() { offsets_.push_back(static_cast<std::int32_t>(m_.size())); }
  /// Uploads the lists to the device.
  EntryList Finish() const;

 private:
  std::vector<std::int32_t> offsets_{0}, m_, from_, to_;
  std::vector<double> coef_;
};

/// The "lifted one-body" lists of comm122st part 1 (scalar X1, tensor Y2),
/// independent of the tensor rank. With r = `channel_ket_offset[ch] + local`
/// the flat row of the ket (i, j) at the channel's J, list row 2 r is side A
/// and list row 2 r + 1 is side B. Side A: p in the scalar one-body channel of
/// i, the lifted ket (p, j) (from = i, to = p); side B: p in the channel of j,
/// the lifted ket (i, p) (from = j, to = p). The entry coefficient is the
/// exchange phase times sqrt2^[lifted pair has equal orbits]; m is the local
/// index of the lifted ket in the same channel (entries whose ket does not
/// exist are omitted).
struct LiftedKets {
  EntryList list;
};

/// The rank-dependent entry lists of comm122st part 2 (tensor Y1, scalar X2)
/// for the stored blocks of the output (`blocks` = (row channel, column
/// channel) in storage order). Bra lists are indexed by the row of the
/// concatenated block rows, ket lists by the row of the concatenated block
/// columns (`col_offset`).
struct Comm122Lists {
  std::vector<std::int32_t> block_ch_bra, block_ch_ket;  ///< Layout key.
  /// bra1: p in tensor channel of i, X2 row (p, j) (sign and 6j in coef);
  /// bra2: p in channel of j, X2 row (p, i); ket3: p in channel of kk, X2
  /// column (l, p); ket4: p in channel of l, X2 column (kk, p). `from`/`to`
  /// index the one-body tensor Y1.
  EntryList bra1, bra2, ket3, ket4;
  exec::DeviceBuffer<std::int32_t> col_offset;  ///< Per block + total.
  bool built = false;
};

/// Tables that depend on the operator's symmetry (lambda, parity, delta_Tz)
/// and the 9j threshold.
struct TensorRankData {
  angmom::LambdaSixJTable sixj_lambda;        ///< {J1 J2 L; ja jb jc}.
  std::unique_ptr<angmom::NineJTable> ninej;  ///< Built on first ph use.
  /// One-body connections of the tensor: orbits b with ConnectsOrbits(a, b).
  Csr tensor_ch1;
  /// Host copy of `tensor_ch1` (rows).
  std::vector<std::vector<std::int32_t>> tensor_ch1_rows;
  Comm122Lists comm122;  ///< Built on first comm122st call.
};

/// Everything the scalar-tensor kernels share for one model space.
struct TensorContext {
  explicit TensorContext(std::shared_ptr<const basis::ModelSpace> ms);

  std::shared_ptr<const basis::ModelSpace> ms;
  basis::DeviceTables tables;              ///< Orbit, ket and channel tables.
  angmom::SixJTable sixj;                  ///< {ja jb J; jc jd J'}.
  Csr scalar_ch1;                          ///< Orbits of equal (l, 2j, tz2).
  exec::DeviceBuffer<std::int32_t> holes;  ///< n >= 1e-6, ascending.
  exec::DeviceBuffer<std::int32_t> particles;  ///< n < 1e-6, ascending.
  std::int32_t num_holes = 0;
  std::int32_t num_particles = 0;
  CrossCoupled cc;  ///< Built on first ph use.
  bool cc_built = false;
  LiftedKets lifted_kets_data;  ///< Built on first comm122st use.
  bool lifted_built = false;
  /// Per (rank_J, parity, delta_Tz).
  std::map<std::tuple<int, int, int>, std::unique_ptr<TensorRankData>> ranks;

  /// The rank data of `spec` (built on first use).
  TensorRankData& rank(const op::OperatorSpec& spec);
  /// The 9j table of `spec` with `threshold` (rebuilt if the threshold
  /// differs from the cached one).
  const angmom::NineJTable& ninej(const op::OperatorSpec& spec,
                                  double threshold);
  /// The cross-coupled structure (built on first use).
  const CrossCoupled& cross_coupled();
  /// The rank-independent lifted lists of comm122st part 1 (built on first
  /// use).
  const LiftedKets& lifted_kets();
  /// The part-2 lists of comm122st for the stored blocks (`ch_bra[k]`,
  /// `ch_ket[k]`) of the output and the tensor rank data `rd` of `spec`
  /// (rebuilt if the block layout differs from the cached one).
  const Comm122Lists& comm122_lists(const op::OperatorSpec& spec,
                                    const std::vector<std::int32_t>& ch_bra,
                                    const std::vector<std::int32_t>& ch_ket);
};

/// Block offset table of a two-body part for `TensorTwoBodyView` (scalar
/// parts: the diagonal blocks). num_channels^2 entries, -1 for absent blocks.
exec::DeviceBuffer<std::int64_t> BlockOffsetTable(const op::TwoBody& part,
                                                  std::int32_t num_channels);

/// The view of `part` with the given offset table (null data if the part is
/// not allocated).
TensorTwoBodyView MakeTensorView(const op::TwoBody& part,
                                 const basis::TwoBodyTableView& tb,
                                 const exec::DeviceBuffer<std::int64_t>& offs);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_TENSOR_IMPL_H_
