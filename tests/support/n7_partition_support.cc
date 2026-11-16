// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/support/n7_partition_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/n7_cost.h"
#include "tests/support/row_mask.h"

namespace imsrg::testing {

namespace {

/// Accumulates the deviation of the values `a` from `ref` into `d`.
void Accumulate(const std::vector<double>& a, const std::vector<double>& ref,
                OperatorDiff& d) {
  if (a.size() != ref.size()) {
    d.bitwise_equal = false;
    d.max_abs_diff = std::numeric_limits<double>::infinity();
    return;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &ref[i], sizeof(double)) != 0) {
      d.bitwise_equal = false;
    }
    const double diff = std::abs(a[i] - ref[i]);
    d.max_abs_diff = std::isnan(diff) ? std::numeric_limits<double>::infinity()
                                      : std::max(d.max_abs_diff, diff);
    d.max_abs_ref = std::max(d.max_abs_ref, std::abs(ref[i]));
  }
}

/// All values of `x` (0B, 1B, 2B, 3B) in a fixed order, from the device.
std::vector<double> Values(const op::Operator& x) {
  std::vector<double> v = {x.zero_body()};
  const std::vector<double> one = x.one_body().buffer().ToVector();
  const std::vector<double> two = x.two_body().matrix().buffer().ToVector();
  v.insert(v.end(), one.begin(), one.end());
  v.insert(v.end(), two.begin(), two.end());
  if (x.three_body().has_value()) {
    const std::vector<double> three = x.three_body()->buffer().ToVector();
    v.insert(v.end(), three.begin(), three.end());
  }
  return v;
}

/// True iff no value of `x` is NaN.
bool Finite(const op::Operator& x) {
  const std::vector<double> v = Values(x);
  return std::none_of(v.begin(), v.end(),
                      [](double y) { return std::isnan(y); });
}

/// The partition check of one partition `parts` (file comment of the
/// header); returns the relative deviation of the summed parts.
double CheckOnePartition(const op::Operator& X, const op::Operator& Y,
                         const std::function<op::Operator()>& zero,
                         const N7RowsFn& run, const op::Operator& whole,
                         const std::vector<basis::RowSet>& parts, bool bitwise,
                         const std::string& tag) {
  op::Operator sum = zero();
  for (std::size_t p = 0; p < parts.size(); ++p) {
    run(X, Y, sum, parts[p]);
    op::Operator plain = zero();
    run(X, Y, plain, parts[p]);
    op::Operator masked = zero();
    run(MaskedCopy(X, parts[p]), MaskedCopy(Y, parts[p]), masked, parts[p]);
    const OperatorDiff dm = CompareOperators(masked, plain);
    EXPECT_TRUE(dm.bitwise_equal && Finite(masked))
        << tag << " part " << p << ": masked operands change the output (max "
        << dm.max_abs_diff << ")";
  }
  const OperatorDiff d = CompareOperators(sum, whole);
  if (bitwise) {
    EXPECT_TRUE(d.bitwise_equal) << tag << ": parts differ from the whole "
                                 << "run (max " << d.max_abs_diff << ")";
  }
  return d.max_abs_diff / std::max(1.0, d.max_abs_ref);
}

}  // namespace

std::vector<basis::RowSet> ContiguousParts(const basis::ThreeBodySpace& s,
                                           std::int32_t parts) {
  const std::int32_t nt = s.num_triples();
  std::vector<basis::RowSet> out;
  for (std::int32_t p = 0; p < parts; ++p) {
    const auto lo =
        static_cast<std::int32_t>(static_cast<std::int64_t>(p) * nt / parts);
    const auto hi = static_cast<std::int32_t>(static_cast<std::int64_t>(p + 1) *
                                              nt / parts);
    std::vector<basis::RowRange> r;
    std::int32_t t = lo;
    while (t < hi) {
      const basis::OrbitTriple& first = s.triple(basis::Triple3Index{t});
      const Channel3Index ch = first.channel;
      const std::int32_t end = std::min(hi, s.channel(ch).triple_end.value());
      const basis::OrbitTriple& last = s.triple(basis::Triple3Index{end - 1});
      r.push_back(basis::RowRange{ch, basis::Triple3Index{t},
                                  basis::Triple3Index{end}, first.row_begin,
                                  last.row_end});
      t = end;
    }
    out.push_back(basis::RowSet::FromRanges(s, std::move(r)));
  }
  return out;
}

std::vector<basis::RowSet> BalancedParts(const basis::ModelSpace& ms,
                                         std::int32_t parts) {
  const basis::RowPartition p = basis::RowPartition::Balanced(
      *ms.three_body_space(), commutator::N7LineWeights(ms), parts);
  std::vector<basis::RowSet> out;
  for (std::int32_t i = 0; i < p.num_parts(); ++i) out.push_back(p.part(i));
  return out;
}

std::vector<std::pair<std::string, std::vector<basis::RowSet>>>
StandardPartitions(const basis::ModelSpace& ms) {
  std::vector<std::pair<std::string, std::vector<basis::RowSet>>> out;
  for (const std::int32_t p : {2, 3, 7}) {
    out.emplace_back("contiguous " + std::to_string(p),
                     ContiguousParts(*ms.three_body_space(), p));
  }
  for (const std::int32_t p : {4, 16}) {
    out.emplace_back("balanced " + std::to_string(p), BalancedParts(ms, p));
  }
  return out;
}

OperatorDiff CompareOperators(const op::Operator& a, const op::Operator& ref) {
  OperatorDiff d;
  Accumulate(Values(a), Values(ref), d);
  return d;
}

op::Operator MaskedCopy(const op::Operator& x, const basis::RowSet& rows) {
  op::Operator m = x;
  if (m.three_body().has_value()) MaskOutsideLines(*m.three_body(), rows);
  return m;
}

void ExpectPartitionInvariance(const basis::ModelSpace& ms,
                               const op::Operator& X, const op::Operator& Y,
                               const std::function<op::Operator()>& zero,
                               const N7RowsFn& run, double tol, bool bitwise,
                               const std::string& tag) {
  op::Operator whole = zero();
  run(X, Y, whole, basis::RowSet::Whole(*ms.three_body_space()));
  double worst = 0.0;
  for (const auto& [label, parts] : StandardPartitions(ms)) {
    const double dev = CheckOnePartition(X, Y, zero, run, whole, parts, bitwise,
                                         tag + " " + label);
    EXPECT_LE(dev, tol) << tag << " " << label;
    worst = std::max(worst, dev);
  }
  std::printf("[partition] %s: max |dZ| / max(1, max|Z|) = %.3e\n", tag.c_str(),
              worst);
}

}  // namespace imsrg::testing
