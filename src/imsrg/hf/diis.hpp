#pragma once
/// @file
/// Internal: Pulay DIIS extrapolation of the Fock matrix for the Hartree-Fock iteration. Not part
/// of the public API of the hf module. Host only.

#include <cstddef>
#include <deque>
#include <vector>

#include "imsrg/linalg/matrix_view.hpp"

namespace imsrg::hf::detail {

/// Pulay's direct inversion in the iterative subspace on pairs (F_k, e_k) of Fock matrices and
/// error matrices e_k = F_k rho_k - rho_k F_k. The overlap of two error matrices is
/// B_ij = sum_ab w_a e_i(a,b) e_j(a,b) with the angular-momentum weights w_a = 2 j_a + 1, and the
/// extrapolated Fock matrix is sum_k c_k F_k with the c minimizing |sum_k c_k e_k| subject to
/// sum_k c_k = 1.
class PulayDiis {
 public:
  /// `weights[a]` = 2 j_a + 1 (size N); `max_vectors >= 1` is the subspace size (the oldest pair
  /// is dropped when it is exceeded).
  PulayDiis(std::vector<double> weights, std::size_t max_vectors);

  /// Appends the pair (F, e), both N x N. Drops the oldest pair if the subspace is full.
  void push(const linalg::Matrix<double>& F, const linalg::Matrix<double>& e);

  /// Forgets all stored pairs.
  void clear() noexcept;

  /// Number of stored pairs.
  std::size_t size() const noexcept { return fock_.size(); }

  /// Weighted Frobenius norm sqrt(sum_ab w_a e(a,b)^2) of the newest error matrix (0 if empty).
  double newest_error_norm() const;

  /// The extrapolated Fock matrix. Solves the bordered system [[B, 1], [1^T, 0]] [c; lambda] =
  /// [0; 1] with B normalized by its largest entry; if it is singular, ill-conditioned
  /// (non-finite coefficients or max |c| > 1e8) the oldest pairs are dropped one at a time, down
  /// to the newest pair alone (c = 1). Precondition: size() >= 1. Complexity O(m^3 + m N^2).
  linalg::Matrix<double> extrapolate() const;

  /// Number of pairs used by the last `extrapolate` call (diagnostics).
  std::size_t last_used() const noexcept { return last_used_; }

 private:
  std::vector<double> weights_;
  std::size_t max_vectors_;
  std::deque<linalg::Matrix<double>> fock_;
  std::deque<linalg::Matrix<double>> err_;
  mutable std::size_t last_used_ = 0;
};

}  // namespace imsrg::hf::detail
