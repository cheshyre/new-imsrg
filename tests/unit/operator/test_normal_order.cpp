// Tests of normal ordering and its inverse: imsrg-devel goldens and algebraic properties.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/normal_ordering.hpp"
#include "imsrg/operator/operator_dump.hpp"

#ifndef IMSRG_GOLDEN_DIR
#error "IMSRG_GOLDEN_DIR must be defined"
#endif

namespace {

using namespace imsrg;
using namespace imsrg::op;

std::filesystem::path golden(const std::string& rel) {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) / rel;
}

/// Compares `got` with `want` part by part at 1e-10 * max(1, max|want|), prints the deviations
/// and returns the largest one.
double expect_equal_parts(const Operator& got, const Operator& want, double rel_tol,
                          const std::string& label) {
  const PartMaxima ref = max_abs(want);
  const PartMaxima dev = max_abs_difference(got, want);
  const double tol = rel_tol * std::max(1.0, ref.max());
  std::printf("[ dev ] %-40s zero %.3e  one %.3e  two %.3e  (tol %.1e)\n", label.c_str(),
              dev.zero_body, dev.one_body, dev.two_body, tol);
  EXPECT_LE(dev.zero_body, tol) << label << ": zero-body";
  EXPECT_LE(dev.one_body, tol) << label << ": one-body";
  EXPECT_LE(dev.two_body, tol) << label << ": two-body";
  return dev.max();
}

Operator read_into(const std::string& rel, const std::shared_ptr<const basis::ModelSpace>& ms,
                   bool check_occ = true) {
  OperatorDumpReadOptions options;
  options.check_occupations = check_occ;
  return read_operator_dump(golden(rel), ms, options);
}

std::shared_ptr<const basis::ModelSpace> ms_for(const std::string& rel) {
  return model_space_for_dump(read_operator_dump_header(golden(rel)));
}

TEST(NormalOrderGolden, HamiltonianO16) {
  const auto ms = ms_for("hamiltonian/emax2_O16/H.dump");
  const Operator h = read_into("hamiltonian/emax2_O16/H.dump", ms);
  const Operator want = read_into("operator/emax2/H_NO_O16.dump", ms);
  expect_equal_parts(normal_order(h), want, 1e-10, "H -> NO O16");
}

TEST(NormalOrderGolden, RandomHermitianNe20) {
  const auto ms = ms_for("operator/emax2/randomH_NO_Ne20.dump");
  const Operator x = read_into("hamiltonian/emax2_O16/random_H.dump", ms, false);
  const Operator want = read_into("operator/emax2/randomH_NO_Ne20.dump", ms);
  expect_equal_parts(normal_order(x), want, 1e-10, "random_H -> NO Ne20");
}

TEST(NormalOrderGolden, RandomAntiHermitianNe20) {
  const auto ms = ms_for("operator/emax2/randomA_NO_Ne20.dump");
  const Operator x = read_into("commutator/emax2_O16/random_A.dump", ms, false);
  ASSERT_EQ(x.hermiticity(), Hermiticity::AntiHermitian);
  const Operator want = read_into("operator/emax2/randomA_NO_Ne20.dump", ms);
  const Operator got = normal_order(x);
  expect_equal_parts(got, want, 1e-10, "random_A -> NO Ne20");
  EXPECT_EQ(got.zero_body(), 0.0);
}

TEST(NormalOrderGolden, UndoHamiltonianO16) {
  const auto ms = ms_for("hamiltonian/emax2_O16/H.dump");
  const Operator no = read_into("operator/emax2/H_NO_O16.dump", ms);
  const Operator undone = undo_normal_order(no);
  expect_equal_parts(undone, read_into("operator/emax2/H_NO_O16_undo.dump", ms), 1e-10,
                     "undo(H_NO) vs H_NO_O16_undo");
  expect_equal_parts(undone, read_into("hamiltonian/emax2_O16/H.dump", ms), 1e-10,
                     "undo(H_NO) vs H");
}

// ---- Properties -------------------------------------------------------------------------------

class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : s_(seed) {}
  double next() {
    s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(static_cast<std::int64_t>(s_ >> 11)) / 4503599627370496.0 - 1.0;
  }

 private:
  std::uint64_t s_;
};

/// Random scalar operator of the given hermiticity and rank; every element obeys the (l, 2j, tz2)
/// selection rule of the one-body part and the channel structure of the two-body part.
Operator random_operator(const std::shared_ptr<const basis::ModelSpace>& ms, Hermiticity herm,
                         std::int32_t rank, std::uint64_t seed) {
  Lcg rng(seed);
  Operator o(ms, OperatorSpec::scalar(rank, herm));
  const auto& sp = ms->basis();
  const std::int32_t h = hermiticity_sign(herm);
  o.zero_body() = rng.next();
  for (std::int32_t a = 0; a < sp.num_orbits(); ++a) {
    for (const OrbitIndex b : sp.same_ljtz(OrbitIndex{a})) {
      if (h != 0 && b.value() < a) {
        continue;
      }
      if (h < 0 && b.value() == a) {
        continue;  // zero diagonal
      }
      const double v = rng.next();
      if (h == 0) {
        o.one_body().set_element(OrbitIndex{a}, b, v);
      } else {
        o.one_body().set(OrbitIndex{a}, b, v);
      }
    }
  }
  if (rank >= 2) {
    const auto& channels = ms->two_body();
    for (std::int32_t c = 0; c < channels.num_channels(); ++c) {
      const ChannelIndex ch{c};
      const std::int32_t n = channels.channel(ch).num_kets;
      for (std::int32_t i = 0; i < n; ++i) {
        for (std::int32_t j = (h == 0 ? 0 : i); j < n; ++j) {
          if (h < 0 && i == j) {
            continue;
          }
          const double v = rng.next();
          o.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
          if (h != 0 && i != j) {
            o.two_body().set(ch, ch, LocalKetIndex{j}, LocalKetIndex{i}, h * v);
          }
        }
      }
    }
  }
  o.sync_to_device();
  return o;
}

std::shared_ptr<const basis::ModelSpace> ne20() {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 2}, 16.0, "Ne20", "Ne20");
}

/// Reference with fractional occupations 0.3 / 0.7 / 1 / 0 on several orbits.
std::vector<double> custom_occupations(const basis::SingleParticleBasis& sp) {
  std::vector<double> occ(static_cast<std::size_t>(sp.num_orbits()), 0.0);
  const double values[] = {0.7, 0.3, 1.0, 0.0, 0.5};
  for (std::size_t i = 0; i < occ.size(); ++i) {
    occ[i] = values[i % 5];
  }
  return occ;
}

constexpr Hermiticity kAllHerm[] = {Hermiticity::Hermitian, Hermiticity::AntiHermitian,
                                    Hermiticity::NonHermitian};

TEST(NormalOrderProperties, RoundTripsFractionalOccupations) {
  const auto ms = ne20();
  const std::vector<double> occ_custom = custom_occupations(ms->basis());
  for (const Hermiticity herm : kAllHerm) {
    const Operator x = random_operator(ms, herm, 2, 11 + static_cast<unsigned>(herm));
    const std::string tag = std::to_string(static_cast<int>(herm));
    // Model-space reference (Ne20, fractional 1/3 in 0d5/2).
    expect_equal_parts(undo_normal_order(normal_order(x)), x, 1e-12, "undo(NO) herm " + tag);
    expect_equal_parts(normal_order(undo_normal_order(x)), x, 1e-12, "NO(undo) herm " + tag);
    // Explicit occupations.
    const auto to_ref = [&](const Operator& o) {
      return normal_order(o, occ_custom, NormalOrderDirection::ToReference);
    };
    const auto to_vac = [&](const Operator& o) {
      return normal_order(o, occ_custom, NormalOrderDirection::ToVacuum);
    };
    expect_equal_parts(to_vac(to_ref(x)), x, 1e-12, "custom undo(NO) herm " + tag);
    expect_equal_parts(to_ref(to_vac(x)), x, 1e-12, "custom NO(undo) herm " + tag);
  }
}

TEST(NormalOrderProperties, ModelSpaceOverloadUsesReferenceOccupations) {
  const auto ms = ne20();
  const basis::Reference ref =
      basis::Reference::from_occupations(ms->basis(), custom_occupations(ms->basis()));
  const auto ms2 = ms->with_reference(ref);
  const Operator x = random_operator(ms2, Hermiticity::Hermitian, 2, 5);
  const Operator a = normal_order(x);
  const Operator b = normal_order(x, ref.occupations(), NormalOrderDirection::ToReference);
  const PartMaxima d = max_abs_difference(a, b);
  EXPECT_EQ(d.max(), 0.0);
}

TEST(NormalOrderProperties, IsLinear) {
  const auto ms = ne20();
  for (const Hermiticity herm : kAllHerm) {
    const Operator x = random_operator(ms, herm, 2, 21);
    const Operator y = random_operator(ms, herm, 2, 22);
    const double a = -0.37;
    const Operator lhs = normal_order(a * x + y);
    // NO has no constant part for the E0 shift of X alone: NO(X) is linear in the full operator
    // (E0 included), so NO(aX + Y) = a NO(X) + NO(Y) exactly.
    const Operator rhs = a * normal_order(x) + normal_order(y);
    expect_equal_parts(lhs, rhs, 1e-12, "linearity herm " + std::to_string(static_cast<int>(herm)));
  }
}

TEST(NormalOrderProperties, KeepsOneBodySymmetryAndTwoBodyPart) {
  const auto ms = ne20();
  const auto& sp = ms->basis();
  for (const Hermiticity herm : {Hermiticity::Hermitian, Hermiticity::AntiHermitian}) {
    const std::int32_t h = hermiticity_sign(herm);
    const Operator x = random_operator(ms, herm, 2, 31);
    const Operator no = normal_order(x);
    EXPECT_EQ(no.hermiticity(), herm);
    double asym = 0.0;
    for (std::int32_t a = 0; a < sp.num_orbits(); ++a) {
      for (const OrbitIndex b : sp.same_ljtz(OrbitIndex{a})) {
        asym = std::max(asym, std::abs(no.one_body().get(OrbitIndex{a}, b) -
                                       h * no.one_body().get(b, OrbitIndex{a})));
      }
    }
    EXPECT_EQ(asym, 0.0);
    // The two-body part is unchanged.
    Operator diff = no;
    diff.zero_body() = x.zero_body();
    diff.one_body() = x.one_body();
    EXPECT_EQ(max_abs_difference(diff, x).two_body, 0.0);
  }
}

TEST(NormalOrderProperties, OneBodyOperatorOnlyShiftsZeroBody) {
  const auto ms = ne20();
  const auto& sp = ms->basis();
  const auto occ = ms->reference().occupations();
  const Operator x = random_operator(ms, Hermiticity::Hermitian, 1, 41);
  double expected = 0.0;
  for (std::int32_t k = 0; k < sp.num_orbits(); ++k) {
    if (occ[static_cast<std::size_t>(k)] >= basis::kOccupationCut) {
      expected += degeneracy(sp.orbit(OrbitIndex{k}).j2) * occ[static_cast<std::size_t>(k)] *
                  x.one_body().get(OrbitIndex{k}, OrbitIndex{k});
    }
  }
  const Operator no = normal_order(x);
  EXPECT_NEAR(no.zero_body(), x.zero_body() + expected, 1e-13);
  EXPECT_EQ(max_abs_difference(no,
                               [&] {
                                 Operator y = x;
                                 y.zero_body() = no.zero_body();
                                 return y;
                               }())
                .max(),
            0.0);
  const Operator back = undo_normal_order(no);
  EXPECT_NEAR(back.zero_body(), x.zero_body(), 1e-13);
  EXPECT_FALSE(no.two_body().allocated());
}

TEST(NormalOrderProperties, WrongOccupationCountThrows) {
  const auto ms = ne20();
  const Operator x = random_operator(ms, Hermiticity::Hermitian, 2, 51);
  const std::vector<double> occ(static_cast<std::size_t>(ms->num_orbits()) - 1, 0.5);
  EXPECT_THROW(normal_order(x, occ, NormalOrderDirection::ToReference), InvariantError);
  EXPECT_THROW(normal_order(Operator{}), InvariantError);
  EXPECT_THROW(undo_normal_order(Operator{}), InvariantError);
}

}  // namespace
