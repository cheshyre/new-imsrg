#include "imsrg/operator/compare.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "imsrg/base/error.hpp"

namespace imsrg::op {

namespace {

double max_abs_of(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::max(m, std::abs(x));
  return m;
}

/// max |a - b| over two vectors; a missing (empty) one counts as zero.
double max_abs_diff_of(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.empty()) return max_abs_of(b);
  if (b.empty()) return max_abs_of(a);
  IMSRG_CHECK(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

std::vector<double> two_body_values(const TwoBody& t) {
  if (!t.allocated()) return {};
  return t.matrix().buffer().to_vector();
}

}  // namespace

double PartMaxima::max() const noexcept { return std::max({zero_body, one_body, two_body}); }

PartMaxima max_abs(const Operator& op) {
  PartMaxima m;
  m.zero_body = std::abs(op.zero_body());
  m.one_body = max_abs_of(op.one_body().buffer().to_vector());
  m.two_body = max_abs_of(two_body_values(op.two_body()));
  return m;
}

PartMaxima max_abs_difference(const Operator& a, const Operator& b) {
  IMSRG_CHECK(same_model_space(a, b));
  IMSRG_CHECK(a.spec().same_symmetry(b.spec()));
  PartMaxima m;
  m.zero_body = std::abs(a.zero_body() - b.zero_body());
  m.one_body =
      max_abs_diff_of(a.one_body().buffer().to_vector(), b.one_body().buffer().to_vector());
  m.two_body = max_abs_diff_of(two_body_values(a.two_body()), two_body_values(b.two_body()));
  return m;
}

}  // namespace imsrg::op
