// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/hf/diis.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::hf::detail {

namespace {

double WeightedDot(const std::vector<double>& w,
                   const linalg::Matrix<double>& x,
                   const linalg::Matrix<double>& y) {
  double s = 0.0;
  for (std::int64_t b = 0; b < x.cols(); ++b) {
    for (std::int64_t a = 0; a < x.rows(); ++a) {
      s += w[static_cast<std::size_t>(a)] * x(a, b) * y(a, b);
    }
  }
  return s;
}

/// Returns the weighted Gram matrix of the error matrices `err` and sets
/// `bmax` to its largest absolute element.
linalg::Matrix<double> ErrorGram(const std::vector<double>& weights,
                                 const std::deque<linalg::Matrix<double>>& err,
                                 double& bmax) {
  const std::size_t m = err.size();
  linalg::Matrix<double> B(static_cast<std::int64_t>(m),
                           static_cast<std::int64_t>(m));
  bmax = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = i; j < m; ++j) {
      const double v = WeightedDot(weights, err[i], err[j]);
      B(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)) = v;
      B(static_cast<std::int64_t>(j), static_cast<std::int64_t>(i)) = v;
      bmax = std::max(bmax, std::abs(v));
    }
  }
  return B;
}

/// Solves the bordered Pulay system dropping the oldest pairs one at a time
/// until it is well posed; stores the coefficients in `coeff` (size m, pairs
/// start..m-1 used) and returns the number of pairs used, or 0 if no
/// subsystem was usable (`coeff` unchanged).
std::size_t SolveBordered(const linalg::Matrix<double>& B, double bmax,
                          std::size_t m, std::vector<double>& coeff) {
  for (std::size_t start = 0; start + 1 < m; ++start) {
    const std::size_t k = m - start;
    linalg::Matrix<double> A(static_cast<std::int64_t>(k + 1),
                             static_cast<std::int64_t>(k + 1));
    linalg::Matrix<double> rhs(static_cast<std::int64_t>(k + 1), 1);
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        A(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)) =
            B(static_cast<std::int64_t>(start + i),
              static_cast<std::int64_t>(start + j)) /
            bmax;
      }
      A(static_cast<std::int64_t>(i), static_cast<std::int64_t>(k)) = 1.0;
      A(static_cast<std::int64_t>(k), static_cast<std::int64_t>(i)) = 1.0;
    }
    rhs(static_cast<std::int64_t>(k), 0) = 1.0;
    try {
      linalg::Solve(A.const_view(), rhs.view());
    } catch (const NumericalError&) {
      continue;
    }
    bool ok = true;
    for (std::size_t i = 0; i < k; ++i) {
      const double c = rhs(static_cast<std::int64_t>(i), 0);
      if (!std::isfinite(c) || std::abs(c) > 1.0e8) ok = false;
    }
    if (!ok) continue;
    std::fill(coeff.begin(), coeff.end(), 0.0);
    for (std::size_t i = 0; i < k; ++i)
      coeff[start + i] = rhs(static_cast<std::int64_t>(i), 0);
    return k;
  }
  return 0;
}

}  // namespace

PulayDiis::PulayDiis(std::vector<double> weights, std::size_t max_vectors)
    : weights_(std::move(weights)), max_vectors_(max_vectors) {
  IMSRG_CHECK(max_vectors_ >= 1, "DIIS subspace must hold at least one vector");
}

void PulayDiis::Push(const linalg::Matrix<double>& F,
                     const linalg::Matrix<double>& e) {
  IMSRG_CHECK(F.rows() == e.rows() && F.cols() == e.cols() &&
                  F.rows() == static_cast<std::int64_t>(weights_.size()),
              "DIIS pair has the wrong shape");
  fock_.push_back(F);
  err_.push_back(e);
  while (fock_.size() > max_vectors_) {
    fock_.pop_front();
    err_.pop_front();
  }
}

void PulayDiis::clear() noexcept {
  fock_.clear();
  err_.clear();
}

double PulayDiis::newest_error_norm() const {
  if (err_.empty()) return 0.0;
  return std::sqrt(WeightedDot(weights_, err_.back(), err_.back()));
}

linalg::Matrix<double> PulayDiis::Extrapolate() const {
  IMSRG_CHECK(!fock_.empty(), "DIIS extrapolation needs at least one pair");
  const std::size_t m = fock_.size();

  double bmax = 0.0;
  const linalg::Matrix<double> B = ErrorGram(weights_, err_, bmax);

  std::vector<double> coeff(m, 0.0);
  coeff[m - 1] = 1.0;  // fallback: newest pair only
  last_used_ = 1;
  if (bmax > 0.0) {
    const std::size_t used = SolveBordered(B, bmax, m, coeff);
    if (used > 0) last_used_ = used;
  }

  linalg::Matrix<double> out(fock_.front().rows(), fock_.front().cols());
  for (std::size_t k = 0; k < m; ++k) {
    if (coeff[k] == 0.0) continue;
    for (std::int64_t b = 0; b < out.cols(); ++b) {
      for (std::int64_t a = 0; a < out.rows(); ++a)
        out(a, b) += coeff[k] * fock_[k](a, b);
    }
  }
  return out;
}

}  // namespace imsrg::hf::detail
