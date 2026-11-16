// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_N7_COST_H_
#define IMSRG_COMMUTATOR_N7_COST_H_
/// @file
/// Exact counts and an analytic cost model of the row-tiled IMSRG(3)-N7
/// commutator terms, for the milestone report (memory and time at emax 12 /
/// E3max 16) and for the per-triple weights of the MPI row partition (M12b).
/// Host only, no Kokkos.
///
/// The model works on the kept three-body space of a `basis::ModelSpace`
/// built `WithThreeBody` and never builds the pair-third space, the context
/// or an operator: the pair-third states of a triple are counted from the
/// orbits (same rule as `op::PairThirdSpace`), so emax 12 / E3max 16 runs in
/// seconds. Quantities per triple t of channel ch (n = dim(ch) rows):
/// * k_t rows (the lines of the triple), r0_t the channel-local first row;
/// * m_t pair-third states, a_t of them active (pair occupation class 0..2 of
///   `kernels/comm232.cc` `PairClass`, i.e. not a particle-hole pair);
/// * mem_t = sum_{k in t} (n - k) stored (packed) doubles per operand;
/// * channel sums S = sum m_t, A = sum a_t, skm = sum k_t m_t.
///
/// The per-term multiply-add counts (one commutator, 3B parts in X and Y)
/// are the formulas of the milestone plan (n7model.py `column_costs`):
///   223 W sums      m_t sum_{T>=t} m_T n_a,   n_a = fa n_orb / 4, fa = 0.35
///   223 epilogue    sum_{T>=t} (m_T nnz(R_t) + nnz(R_T) k_t)
///   H build         2 k_t skm
///   W build         2 S k_t m_t
///   232             4 A m_t n_p,              n_p = fp n_orb / 4, fp = 0.40
///   331             2 k_t S n1
///   133             2 k_t (S n1 + skm)
///   330             2 mem_t
///   231 + 132       4 m_t S n1 / n_orb
/// with n1 the mean one-body channel size. n_a, n_p are the planner's
/// fractions of the orbit space that a W element sum or a particle pair
/// sweep touches (approximations); nnz(R_t) is the exact number of stored
/// entries of the R rows of the triple apart from the |v| < 1e-13 drops;
/// A is exact (the planner used a hole-count estimate).

#include <array>
#include <cstdint>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"

namespace imsrg::commutator {

/// Tunable fractions of the cost model.
struct N7CostParams {
  double fa = 0.35;  ///< Fraction of n_orb / 4 per 223 W element sum.
  double fp = 0.40;  ///< Fraction of n_orb / 4 per 232 particle sweep.
};

/// Terms of the cost table, in the column order of the plan.
enum class N7Term : std::int32_t {
  k223 = 0,    ///< [2,2]->3 W sums plus the R / E epilogue.
  kH,          ///< Build of the H lines (X3 pair-third rows).
  kW,          ///< Build of the W blocks.
  k232,        ///< comm232ss.
  k331,        ///< comm331ss.
  k133,        ///< comm133ss.
  k330,        ///< comm330ss.
  k231And132,  ///< comm231ss + comm132ss.
};
/// Number of `N7Term` values.
inline constexpr std::int32_t kNumCostTerms = 8;

/// Display name of a term ("223", "H", "W", "232", "331", "133", "330",
/// "231+132").
const char* N7TermName(N7Term t) noexcept;

/// Counts and multiply-adds of one triple.
struct N7TripleCost {
  std::int32_t rows = 0;       ///< k_t.
  std::int32_t states = 0;     ///< m_t pair-third states.
  std::int32_t active = 0;     ///< a_t active pair-third states (classes 0-2).
  std::int64_t first_row = 0;  ///< Channel-local first row r0_t.
  std::int64_t nnz_r = 0;      ///< Entries of the R rows of the triple.
  std::int64_t mem = 0;        ///< mem_t, packed doubles per operand.
  double epilogue_223 = 0.0;   ///< Epilogue part of `ma[k223]`.
  /// Multiply-adds per term, indexed by `N7Term`.
  std::array<double, kNumCostTerms> ma{};
  /// Sum of `ma`.
  double Total() const noexcept;
};

/// Per-channel counts.
struct N7ChannelCost {
  std::int32_t dim = 0;          ///< n.
  std::int32_t num_triples = 0;  ///< Triples of the channel.
  std::int64_t states = 0;       ///< S.
  std::int64_t active = 0;       ///< A.
  std::int64_t packed = 0;       ///< n (n + 1) / 2.
};

/// Sizes of the scratch of one batch of lines (see
/// `N7CostModel::BatchScratch`).
struct N7BatchScratch {
  std::int64_t max_doubles = 0;  ///< Largest batch, doubles, per operand.
  std::int64_t max_lines = 0;    ///< Lines of the largest batch.
  std::int64_t max_states = 0;   ///< Pair-third states of the largest batch.
  std::int32_t channel = -1;     ///< Channel of the largest batch, or -1.
};

/// Exact counts and the analytic cost model of the tiled N7 terms of a
/// three-body model space. Immutable value type.
class N7CostModel {
 public:
  /// Counts the space. Precondition: `ms3.three_body_space() != nullptr`
  /// (IMSRG_CHECK). Cost O(triples * J range) time, O(triples) memory (about
  /// 100 bytes per triple).
  explicit N7CostModel(const basis::ModelSpace& ms3,
                       const N7CostParams& params = {});

  N7CostModel(const N7CostModel&) = default;
  N7CostModel& operator=(const N7CostModel&) = default;
  N7CostModel(N7CostModel&&) = default;
  N7CostModel& operator=(N7CostModel&&) = default;

  /// Orbits of the basis.
  std::int32_t num_orbits() const noexcept { return num_orbits_; }
  /// Mean one-body channel size n1 (mean over orbits of the number of orbits
  /// with the same l, 2j, 2tz).
  double OneBodyMean() const noexcept { return n1_; }
  /// Per-triple costs in `Triple3Index` order.
  const std::vector<N7TripleCost>& triples() const noexcept { return triples_; }
  /// Per-channel counts in `Channel3Index` order.
  const std::vector<N7ChannelCost>& channels() const noexcept {
    return channels_;
  }
  /// Number of kept rows (sum of the channel dimensions).
  std::int64_t num_rows() const noexcept { return num_rows_; }
  /// Number of orbit triples.
  std::int64_t num_triples() const noexcept {
    return static_cast<std::int64_t>(triples_.size());
  }
  /// Number of pair-third states.
  std::int64_t num_states() const noexcept { return num_states_; }
  /// Number of active pair-third states.
  std::int64_t NumActive() const noexcept { return num_active_; }
  /// Largest channel dimension.
  std::int32_t MaxDim() const noexcept { return max_dim_; }
  /// Largest number of pair-third states of a channel.
  std::int64_t max_channel_states() const noexcept { return max_states_; }
  /// Sum over channels of n (n + 1) / 2: stored doubles per 3B operand.
  std::int64_t PackedDoubles() const noexcept { return packed_; }
  /// Bytes of one packed 3B operand.
  double PackedBytes() const noexcept {
    return 8.0 * static_cast<double>(packed_);
  }
  /// Doubles of one 2B operator (sum over two-body channels of dim^2).
  std::int64_t TwoBodyDoubles() const noexcept { return two_body_; }

  /// Multiply-adds of one commutator per term, summed over the triples.
  std::array<double, kNumCostTerms> TermTotals() const;
  /// Multiply-adds of one commutator, all terms.
  double TotalMultiplyAdds() const;
  /// Components of the 223 total: {W sums, epilogue}.
  std::array<double, 2> Comm223Parts() const;

  /// Scratch of the batches of one commutator pass over lines, per operand,
  /// in doubles: the lines of a channel are cut greedily into batches of
  /// whole triples holding at most `lines_per_batch` lines (at least one
  /// triple); a batch with L lines, Q pair-third states and S the
  /// states of the channel needs H: S x L, W: S x Q (all S rows of the
  /// channel) and the transposed output sum_{k in batch} (k + 1), k the
  /// channel-local line. Returns the largest batch. Precondition:
  /// `lines_per_batch >= 1` (IMSRG_CHECK). O(triples).
  N7BatchScratch BatchScratch(std::int64_t lines_per_batch) const;

 private:
  std::int32_t num_orbits_ = 0;
  double n1_ = 0.0;
  std::int64_t num_rows_ = 0;
  std::int64_t num_states_ = 0;
  std::int64_t num_active_ = 0;
  std::int64_t packed_ = 0;
  std::int64_t two_body_ = 0;
  std::int64_t max_states_ = 0;
  std::int32_t max_dim_ = 0;
  std::vector<N7TripleCost> triples_;
  std::vector<N7ChannelCost> channels_;
};

/// Per-triple weights for `basis::RowPartition::Balanced`: compute = the sum
/// of the multiply-adds of all terms of the triple (`N7TripleCost::Total`),
/// memory = mem_t (packed doubles). One entry per `Triple3Index`.
/// Precondition: `ms3.three_body_space() != nullptr`. Cost as
/// `N7CostModel`.
basis::LineWeights N7LineWeights(const basis::ModelSpace& ms3);

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_N7_COST_H_
