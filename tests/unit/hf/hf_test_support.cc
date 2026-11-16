// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/hf/hf_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace imsrg::hftest {

Json JsonParser::value() {
  Ws();
  Json j;
  const char c = s_[p_];
  if (c == '{') {
    ++p_;
    Ws();
    while (s_[p_] != '}') {
      Ws();
      std::string k = Str();
      Ws();
      ++p_;  // colon
      j.members.emplace_back(std::move(k), value());
      Ws();
      if (s_[p_] == ',') ++p_;
      Ws();
    }
    ++p_;
  } else if (c == '[') {
    ++p_;
    Ws();
    while (s_[p_] != ']') {
      j.items.push_back(value());
      Ws();
      if (s_[p_] == ',') ++p_;
      Ws();
    }
    ++p_;
  } else if (c == '"') {
    Str();
  } else if (s_.compare(p_, 4, "null") == 0 || s_.compare(p_, 4, "true") == 0) {
    p_ += 4;
  } else if (s_.compare(p_, 5, "false") == 0) {
    p_ += 5;
  } else {
    char* end = nullptr;
    j.number = std::strtod(s_.c_str() + p_, &end);
    p_ = static_cast<std::size_t>(end - s_.c_str());
  }
  return j;
}

Golden LoadGolden(const std::filesystem::path& path) {
  const Json j = ReadJson(path);
  Golden g;
  g.e_hf = j.at("EHF").number;
  g.e1 = j.at("e1hf").number;
  g.e2 = j.at("e2hf").number;
  g.e3 = j.at("e3hf").number;
  g.iterations = static_cast<int>(j.at("iterations").number);
  g.spe = j.at("spe").Numbers();
  g.occupation = j.at("occupation").Numbers();
  for (const auto& row : j.at("C").items) g.C.push_back(row.Numbers());
  return g;
}

void CompareToGolden(const hf::HFResult& r, const Golden& g,
                     const std::string& tag, double e_tol, double spe_tol,
                     double c_tol) {
  ASSERT_EQ(r.spe.size(), g.spe.size());
  double dc = 0.0, docc = 0.0;
  for (std::size_t i = 0; i < g.spe.size(); ++i) {
    docc = std::max(docc, std::abs(r.occupations[i] - g.occupation[i]));
    for (std::size_t a = 0; a < g.spe.size(); ++a) {
      dc = std::max(dc, std::abs(r.C(static_cast<std::int64_t>(i),
                                     static_cast<std::int64_t>(a)) -
                                 g.C[i][a]));
    }
  }
  const double dspe = MaxSpeDiff(r, g);
  std::cout << "[" << tag << "] dE_HF " << r.e_hf - g.e_hf << ", de1 "
            << r.e1 - g.e1 << ", de2 " << r.e2 - g.e2 << ", de3 " << r.e3 - g.e3
            << ", max|dspe| " << dspe << ", max|dC| " << dc << ", max|docc| "
            << docc << ", iterations " << r.iterations << " (golden "
            << g.iterations << ")\n";
  EXPECT_NEAR(r.e_hf, g.e_hf, e_tol);
  EXPECT_NEAR(r.e1, g.e1, 10 * e_tol);
  EXPECT_NEAR(r.e2, g.e2, 10 * e_tol);
  EXPECT_NEAR(r.e3, g.e3, 10 * e_tol);
  EXPECT_LE(dspe, spe_tol);
  EXPECT_LE(dc, c_tol);
  EXPECT_EQ(docc, 0.0);
  EXPECT_EQ(r.iterations, g.iterations);
  EXPECT_TRUE(r.converged);
}

}  // namespace imsrg::hftest
