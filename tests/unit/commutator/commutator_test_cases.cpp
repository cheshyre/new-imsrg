#include "commutator_test_cases.hpp"

#include "commutator_test_support.hpp"

namespace imsrg::commutator_test {

const std::vector<GoldenPair>& golden_pairs() {
  static const std::vector<GoldenPair> pairs = {
      {"emax2_O16", "eta_white", "H"},         {"emax2_O16", "random_A", "random_H"},
      {"emax1_He4", "random_A1", "random_H1"}, {"emax1_He4", "random_H1", "random_A1"},
      {"emax1_He4", "random_H1", "random_H2"}, {"emax1_He4", "random_A1", "random_A2"},
      {"emax1_He6", "random_A1", "random_H1"}, {"emax1_He6", "random_H1", "random_H2"},
  };
  return pairs;
}

std::string dump_path(const GoldenPair& p, const std::string& name) {
  return "commutator/" + p.dir + "/" + name + ".dump";
}

std::string result_path(const GoldenPair& p, const std::string& term) {
  return dump_path(p, "Z_" + term + "__" + p.x + "__" + p.y);
}

const std::vector<SpaceCase>& space_cases() {
  static const std::vector<SpaceCase> cases = {
      {"He4_emax1", 1, "He4", false},           {"He4_emax1_fractional", 1, "He4", true},
      {"O16_emax2", 2, "O16", false},           {"O18_emax2", 2, "O18", false},
      {"O16_emax2_fractional", 2, "O16", true},
  };
  return cases;
}

std::shared_ptr<const basis::ModelSpace> make_space(const SpaceCase& c) {
  return c.fractional ? testing::fractional_model_space(c.emax, c.reference)
                      : testing::model_space(c.emax, c.reference);
}

const std::vector<HermPair>& herm_pairs() {
  static const std::vector<HermPair> pairs = {
      {"A_H", Hermiticity::AntiHermitian, Hermiticity::Hermitian, 2, 2},
      {"H_A", Hermiticity::Hermitian, Hermiticity::AntiHermitian, 2, 2},
      {"H_H", Hermiticity::Hermitian, Hermiticity::Hermitian, 2, 2},
      {"A_A", Hermiticity::AntiHermitian, Hermiticity::AntiHermitian, 2, 2},
      {"A1_H2", Hermiticity::AntiHermitian, Hermiticity::Hermitian, 1, 2},
  };
  return pairs;
}

}  // namespace imsrg::commutator_test
