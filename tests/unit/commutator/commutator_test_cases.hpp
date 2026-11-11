#pragma once

/// @file
/// Test data shared by the commutator unit tests: the golden pairs, the random-operator model
/// spaces and hermiticity pairs, and the comparison helpers.

#include <memory>
#include <string>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::commutator_test {

/// Tolerance of design/30 section 2 on `testing::relative_deviation`.
inline constexpr double kTol = 1e-10;

/// One golden pair: directory below tests/golden/commutator and the names of the X and Y dumps.
struct GoldenPair {
  std::string dir;
  std::string x;
  std::string y;
};

/// All golden pairs of the three golden directories (emax2_O16, emax1_He4, emax1_He6).
const std::vector<GoldenPair>& golden_pairs();

/// Relative path ("commutator/<dir>/<name>.dump") of a dump of `p.dir`.
std::string dump_path(const GoldenPair& p, const std::string& name);

/// Relative path of the golden result of `term` (an imsrg++ term name or "all") for `p`.
std::string result_path(const GoldenPair& p, const std::string& term);

/// A model space of the random-operator tests.
struct SpaceCase {
  std::string label;
  int emax;
  std::string reference;
  bool fractional;
};

/// The five model spaces of the fast-vs-reference tests.
const std::vector<SpaceCase>& space_cases();

/// Builds the model space of `c` (integer or fractional occupations).
std::shared_ptr<const basis::ModelSpace> make_space(const SpaceCase& c);

/// A pair of hermiticities and the particle ranks of X and Y of a random-operator case.
struct HermPair {
  const char* label;
  Hermiticity x;
  Hermiticity y;
  int rank_x;
  int rank_y;
};

/// (A,H), (H,A), (H,H), (A,A) and a rank-1 X with a rank-2 Y.
const std::vector<HermPair>& herm_pairs();

}  // namespace imsrg::commutator_test
