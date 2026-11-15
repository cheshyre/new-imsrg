// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tests/unit/commutator/commutator_test_cases.h"

#include <memory>
#include <string>
#include <vector>

#include "tests/support/commutator_test_support.h"

namespace imsrg::commutator_test {

const std::vector<GoldenPair>& GoldenPairs() {
  static const auto* const kPairs = new std::vector<GoldenPair>{
      {"emax2_O16", "eta_white", "H"},
      {"emax2_O16", "random_A", "random_H"},
      {"emax1_He4", "random_A1", "random_H1"},
      {"emax1_He4", "random_H1", "random_A1"},
      {"emax1_He4", "random_H1", "random_H2"},
      {"emax1_He4", "random_A1", "random_A2"},
      {"emax1_He6", "random_A1", "random_H1"},
      {"emax1_He6", "random_H1", "random_H2"},
  };
  return *kPairs;
}

std::string DumpPath(const GoldenPair& p, const std::string& name) {
  return "commutator/" + p.dir + "/" + name + ".dump";
}

std::string ResultPath(const GoldenPair& p, const std::string& term) {
  return DumpPath(p, "Z_" + term + "__" + p.x + "__" + p.y);
}

const std::vector<SpaceCase>& SpaceCases() {
  static const auto* const kCases = new std::vector<SpaceCase>{
      {"He4_emax1", 1, "He4", false},
      {"He4_emax1_fractional", 1, "He4", true},
      {"O16_emax2", 2, "O16", false},
      {"O18_emax2", 2, "O18", false},
      {"O16_emax2_fractional", 2, "O16", true},
  };
  return *kCases;
}

std::shared_ptr<const basis::ModelSpace> MakeSpace(const SpaceCase& c) {
  return c.fractional ? testing::FractionalModelSpace(c.emax, c.reference)
                      : testing::MakeModelSpace(c.emax, c.reference);
}

const std::vector<HermPair>& HermPairs() {
  static const auto* const kPairs = new std::vector<HermPair>{
      {"A_H", Hermiticity::kAntiHermitian, Hermiticity::kHermitian, 2, 2},
      {"H_A", Hermiticity::kHermitian, Hermiticity::kAntiHermitian, 2, 2},
      {"H_H", Hermiticity::kHermitian, Hermiticity::kHermitian, 2, 2},
      {"A_A", Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian, 2, 2},
      {"A1_H2", Hermiticity::kAntiHermitian, Hermiticity::kHermitian, 1, 2},
  };
  return *kPairs;
}

}  // namespace imsrg::commutator_test
