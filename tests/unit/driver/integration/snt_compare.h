// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_DRIVER_INTEGRATION_SNT_COMPARE_H_
#define TESTS_UNIT_DRIVER_INTEGRATION_SNT_COMPARE_H_
/// @file
/// Reader and comparison of tokyo-format `.snt` files (the format of imsrg++
/// `int.snt` and of `io::WriteTokyo`): zero body from the "! Zero body term:"
/// comment, the orbit table, one-body elements `a b value` and two-body
/// elements `a b c d J value`.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::it {

/// A parsed `.snt` file with elements in canonical form.
struct SntData {
  double zero_body = 0.0;
  std::vector<std::array<int, 5>> orbits;  ///< idx n l 2j 2tz
  std::map<std::array<int, 2>, double> one_body;
  /// key (a, b, c, d, J) with a <= b, c <= d, (a, b) <= (c, d); value carries
  /// the pair-swap phases.
  std::map<std::array<int, 5>, double> two_body;
  std::size_t two_body_lines = 0;  ///< lines read
  std::size_t duplicate_lines =
      0;  ///< lines whose canonical key was already present

  /// 2j of orbit index `idx` (1-based). @throws std::runtime_error if unknown.
  int two_j(int idx) const {
    for (const auto& o : orbits)
      if (o[0] == idx) return o[3];
    throw std::runtime_error("snt: unknown orbit " + std::to_string(idx));
  }
};

/// Reads `path`. Elements are stored canonically: one-body (a <= b); two-body
/// with each pair ordered a <= b (phase -(-1)^{ja + jb - J} on a swap) and bra
/// <= ket (the matrices are real symmetric). @throws std::runtime_error on a
/// malformed or missing file.
SntData ReadSnt(const std::filesystem::path& path);

/// Result of `CompareSnt`.
struct SntComparison {
  double zero_dev = 0.0;    ///< |zero body difference|
  double max_dev_1b = 0.0;  ///< largest one-body deviation
  double max_dev_2b = 0.0;  ///< largest two-body deviation
  std::size_t missing =
      0;  ///< elements > `threshold` in one file and absent in the other
  std::size_t n1b_a = 0, n1b_b = 0, n2b_a = 0, n2b_b = 0;
  bool orbits_equal = true;
  std::string worst;  ///< description of the largest two-body deviation

  /// Largest deviation of any kind.
  double max_dev() const {
    return std::max({zero_dev, max_dev_1b, max_dev_2b});
  }
};

/// Compares `a` (computed) with `b` (golden). An element absent from one side
/// counts as 0 there and is tallied in `missing` if its magnitude on the other
/// side exceeds `threshold`.
SntComparison CompareSnt(const SntData& a, const SntData& b, double threshold);

}  // namespace imsrg::it

#endif  // TESTS_UNIT_DRIVER_INTEGRATION_SNT_COMPARE_H_
