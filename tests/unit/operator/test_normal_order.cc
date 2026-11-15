// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of normal ordering and its inverse: imsrg-devel goldens and algebraic
// properties.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/reference.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/operator_dump.h"

#ifndef IMSRG_GOLDEN_DIR
#error "IMSRG_GOLDEN_DIR must be defined"
#endif

namespace {

using namespace imsrg;
using namespace imsrg::op;

std::filesystem::path Golden(const std::string& rel) {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) / rel;
}

/// Compares `got` with `want` part by part at 1e-10 * max(1, max|want|), prints
/// the deviations and returns the largest one.
double ExpectEqualParts(const Operator& got, const Operator& want,
                        double rel_tol, const std::string& label) {
  const PartMaxima ref = MaxAbs(want);
  const PartMaxima dev = MaxAbsDifference(got, want);
  const double tol = rel_tol * std::max(1.0, ref.max());
  std::printf("[ dev ] %-40s zero %.3e  one %.3e  two %.3e  (tol %.1e)\n",
              label.c_str(), dev.zero_body, dev.one_body, dev.two_body, tol);
  EXPECT_LE(dev.zero_body, tol) << label << ": zero-body";
  EXPECT_LE(dev.one_body, tol) << label << ": one-body";
  EXPECT_LE(dev.two_body, tol) << label << ": two-body";
  return dev.max();
}

Operator ReadInto(const std::string& rel,
                  const std::shared_ptr<const basis::ModelSpace>& ms,
                  bool check_occ = true) {
  OperatorDumpReadOptions options;
  options.check_occupations = check_occ;
  return ReadOperatorDump(Golden(rel), ms, options);
}

std::shared_ptr<const basis::ModelSpace> MsFor(const std::string& rel) {
  return ModelSpaceForDump(ReadOperatorDumpHeader(Golden(rel)));
}

TEST(NormalOrderGolden, HamiltonianO16) {
  const auto ms = MsFor("hamiltonian/emax2_O16/H.dump");
  const Operator h = ReadInto("hamiltonian/emax2_O16/H.dump", ms);
  const Operator want = ReadInto("operator/emax2/H_NO_O16.dump", ms);
  ExpectEqualParts(NormalOrder(h), want, 1e-10, "H -> NO O16");
}

TEST(NormalOrderGolden, RandomHermitianNe20) {
  const auto ms = MsFor("operator/emax2/randomH_NO_Ne20.dump");
  const Operator x = ReadInto("hamiltonian/emax2_O16/random_H.dump", ms, false);
  const Operator want = ReadInto("operator/emax2/randomH_NO_Ne20.dump", ms);
  ExpectEqualParts(NormalOrder(x), want, 1e-10, "random_H -> NO Ne20");
}

TEST(NormalOrderGolden, RandomAntiHermitianNe20) {
  const auto ms = MsFor("operator/emax2/randomA_NO_Ne20.dump");
  const Operator x = ReadInto("commutator/emax2_O16/random_A.dump", ms, false);
  ASSERT_EQ(x.hermiticity(), Hermiticity::kAntiHermitian);
  const Operator want = ReadInto("operator/emax2/randomA_NO_Ne20.dump", ms);
  const Operator got = NormalOrder(x);
  ExpectEqualParts(got, want, 1e-10, "random_A -> NO Ne20");
  EXPECT_EQ(got.zero_body(), 0.0);
}

TEST(NormalOrderGolden, UndoHamiltonianO16) {
  const auto ms = MsFor("hamiltonian/emax2_O16/H.dump");
  const Operator no = ReadInto("operator/emax2/H_NO_O16.dump", ms);
  const Operator undone = UndoNormalOrder(no);
  ExpectEqualParts(undone, ReadInto("operator/emax2/H_NO_O16_undo.dump", ms),
                   1e-10, "undo(H_NO) vs H_NO_O16_undo");
  ExpectEqualParts(undone, ReadInto("hamiltonian/emax2_O16/H.dump", ms), 1e-10,
                   "undo(H_NO) vs H");
}

// ---- Properties
// -----------------------------------------------------------------------------

class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : s_(seed) {}
  double Next() {
    s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(static_cast<std::int64_t>(s_ >> 11)) /
               4503599627370496.0 -
           1.0;
  }

 private:
  std::uint64_t s_;
};

/// Random scalar operator of the given hermiticity and rank; every element
/// obeys the (l, 2j, tz2) selection rule of the one-body part and the channel
/// structure of the two-body part.
Operator RandomOperator(const std::shared_ptr<const basis::ModelSpace>& ms,
                        Hermiticity herm, std::int32_t rank,
                        std::uint64_t seed) {
  Lcg rng(seed);
  Operator o(ms, OperatorSpec::Scalar(rank, herm));
  const auto& sp = ms->basis();
  const std::int32_t h = HermiticitySign(herm);
  o.zero_body() = rng.Next();
  for (std::int32_t a = 0; a < sp.num_orbits(); ++a) {
    for (const OrbitIndex b : sp.SameLjtz(OrbitIndex{a})) {
      if (h != 0 && b.value() < a) {
        continue;
      }
      if (h < 0 && b.value() == a) {
        continue;  // zero diagonal
      }
      const double v = rng.Next();
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
          const double v = rng.Next();
          o.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
          if (h != 0 && i != j) {
            o.two_body().set(ch, ch, LocalKetIndex{j}, LocalKetIndex{i}, h * v);
          }
        }
      }
    }
  }
  o.SyncToDevice();
  return o;
}

std::shared_ptr<const basis::ModelSpace> Ne20() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "Ne20", "Ne20");
}

/// Reference with fractional occupations 0.3 / 0.7 / 1 / 0 on several orbits.
std::vector<double> CustomOccupations(const basis::SingleParticleBasis& sp) {
  std::vector<double> occ(static_cast<std::size_t>(sp.num_orbits()), 0.0);
  const double values[] = {0.7, 0.3, 1.0, 0.0, 0.5};
  for (std::size_t i = 0; i < occ.size(); ++i) {
    occ[i] = values[i % 5];
  }
  return occ;
}

constexpr Hermiticity kAllHerm[] = {Hermiticity::kHermitian,
                                    Hermiticity::kAntiHermitian,
                                    Hermiticity::kNonHermitian};

TEST(NormalOrderProperties, RoundTripsFractionalOccupations) {
  const auto ms = Ne20();
  const std::vector<double> occ_custom = CustomOccupations(ms->basis());
  for (const Hermiticity herm : kAllHerm) {
    const Operator x =
        RandomOperator(ms, herm, 2, 11 + static_cast<unsigned>(herm));
    const std::string tag = std::to_string(static_cast<int>(herm));
    // Model-space reference (Ne20, fractional 1/3 in 0d5/2).
    ExpectEqualParts(UndoNormalOrder(NormalOrder(x)), x, 1e-12,
                     "undo(NO) herm " + tag);
    ExpectEqualParts(NormalOrder(UndoNormalOrder(x)), x, 1e-12,
                     "NO(undo) herm " + tag);
    // Explicit occupations.
    const auto to_ref = [&](const Operator& o) {
      return NormalOrder(o, occ_custom, NormalOrderDirection::kToReference);
    };
    const auto to_vac = [&](const Operator& o) {
      return NormalOrder(o, occ_custom, NormalOrderDirection::kToVacuum);
    };
    ExpectEqualParts(to_vac(to_ref(x)), x, 1e-12,
                     "custom undo(NO) herm " + tag);
    ExpectEqualParts(to_ref(to_vac(x)), x, 1e-12,
                     "custom NO(undo) herm " + tag);
  }
}

TEST(NormalOrderProperties, ModelSpaceOverloadUsesReferenceOccupations) {
  const auto ms = Ne20();
  const basis::Reference ref = basis::Reference::FromOccupations(
      ms->basis(), CustomOccupations(ms->basis()));
  const auto ms2 = ms->WithReference(ref);
  const Operator x = RandomOperator(ms2, Hermiticity::kHermitian, 2, 5);
  const Operator a = NormalOrder(x);
  const Operator b =
      NormalOrder(x, ref.occupations(), NormalOrderDirection::kToReference);
  const PartMaxima d = MaxAbsDifference(a, b);
  EXPECT_EQ(d.max(), 0.0);
}

TEST(NormalOrderProperties, IsLinear) {
  const auto ms = Ne20();
  for (const Hermiticity herm : kAllHerm) {
    const Operator x = RandomOperator(ms, herm, 2, 21);
    const Operator y = RandomOperator(ms, herm, 2, 22);
    const double a = -0.37;
    const Operator lhs = NormalOrder(a * x + y);
    // NO has no constant part for the E0 shift of X alone: NO(X) is linear in
    // the full operator (E0 included), so NO(aX + Y) = a NO(X) + NO(Y) exactly.
    const Operator rhs = a * NormalOrder(x) + NormalOrder(y);
    ExpectEqualParts(
        lhs, rhs, 1e-12,
        "linearity herm " + std::to_string(static_cast<int>(herm)));
  }
}

TEST(NormalOrderProperties, KeepsOneBodySymmetryAndTwoBodyPart) {
  const auto ms = Ne20();
  const auto& sp = ms->basis();
  for (const Hermiticity herm :
       {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    const std::int32_t h = HermiticitySign(herm);
    const Operator x = RandomOperator(ms, herm, 2, 31);
    const Operator no = NormalOrder(x);
    EXPECT_EQ(no.hermiticity(), herm);
    double asym = 0.0;
    for (std::int32_t a = 0; a < sp.num_orbits(); ++a) {
      for (const OrbitIndex b : sp.SameLjtz(OrbitIndex{a})) {
        asym =
            std::max(asym, std::abs(no.one_body().get(OrbitIndex{a}, b) -
                                    h * no.one_body().get(b, OrbitIndex{a})));
      }
    }
    EXPECT_EQ(asym, 0.0);
    // The two-body part is unchanged.
    Operator diff = no;
    diff.zero_body() = x.zero_body();
    diff.one_body() = x.one_body();
    EXPECT_EQ(MaxAbsDifference(diff, x).two_body, 0.0);
  }
}

TEST(NormalOrderProperties, OneBodyOperatorOnlyShiftsZeroBody) {
  const auto ms = Ne20();
  const auto& sp = ms->basis();
  const auto occ = ms->reference().occupations();
  const Operator x = RandomOperator(ms, Hermiticity::kHermitian, 1, 41);
  double expected = 0.0;
  for (std::int32_t k = 0; k < sp.num_orbits(); ++k) {
    if (occ[static_cast<std::size_t>(k)] >= basis::kOccupationCut) {
      expected += Degeneracy(sp.orbit(OrbitIndex{k}).j2) *
                  occ[static_cast<std::size_t>(k)] *
                  x.one_body().get(OrbitIndex{k}, OrbitIndex{k});
    }
  }
  const Operator no = NormalOrder(x);
  EXPECT_NEAR(no.zero_body(), x.zero_body() + expected, 1e-13);
  EXPECT_EQ(MaxAbsDifference(no,
                             [&] {
                               Operator y = x;
                               y.zero_body() = no.zero_body();
                               return y;
                             }())
                .max(),
            0.0);
  const Operator back = UndoNormalOrder(no);
  EXPECT_NEAR(back.zero_body(), x.zero_body(), 1e-13);
  EXPECT_FALSE(no.two_body().allocated());
}

TEST(NormalOrderProperties, WrongOccupationCountThrows) {
  const auto ms = Ne20();
  const Operator x = RandomOperator(ms, Hermiticity::kHermitian, 2, 51);
  const std::vector<double> occ(static_cast<std::size_t>(ms->num_orbits()) - 1,
                                0.5);
  EXPECT_THROW(NormalOrder(x, occ, NormalOrderDirection::kToReference),
               InvariantError);
  EXPECT_THROW(NormalOrder(Operator{}), InvariantError);
  EXPECT_THROW(UndoNormalOrder(Operator{}), InvariantError);
}

// ---- NO2B three-body input
// --------------------------------------------------------------------

/// Small NO2B container (emax 2) with three nonzero elements, scaled by
/// `factor`: 0s1/2 0s1/2 coupled to J = 0, T12 = 1, with the spectator 0p1/2
/// (2T = 1) or 0p3/2 (2T = 1 and 3).
No2bThreeBody SmallV3(double factor) {
  No2bThreeBody v3(No2bTruncation{2, 4, 6, 2});
  const auto s = *v3.isospin_orbit_index(0, 0, TwoJ{1});
  const auto p1 = *v3.isospin_orbit_index(0, 1, TwoJ{1});
  const auto p3 = *v3.isospin_orbit_index(0, 1, TwoJ{3});
  v3.set_isospin(s, s, p1, 1, s, s, p1, 1, 0, 1, 0.8 * factor);
  v3.set_isospin(s, s, p3, 1, s, s, p3, 1, 0, 3, -0.5 * factor);
  v3.set_isospin(s, s, p3, 1, s, s, p3, 1, 0, 1, 0.3 * factor);
  return v3;
}

TEST(NormalOrderNo2b, EmptyInputEqualsPlainNormalOrder) {
  const auto ms = Ne20();
  const No2bThreeBody empty(No2bTruncation{2, 4, 6, 2});
  for (const Hermiticity herm : {Hermiticity::kHermitian}) {
    const Operator x = RandomOperator(ms, herm, 2, 61);
    const double d =
        MaxAbsDifference(NormalOrder(x, empty), NormalOrder(x)).max();
    EXPECT_EQ(d, 0.0);
  }
}

TEST(NormalOrderNo2b, ZeroOccupationsLeaveOperatorUnchanged) {
  const auto ms = Ne20();
  const Operator x = RandomOperator(ms, Hermiticity::kHermitian, 2, 62);
  const std::vector<double> occ(static_cast<std::size_t>(ms->num_orbits()),
                                0.0);
  const Operator y =
      NormalOrder(x, SmallV3(1.0), occ, NormalOrderDirection::kToReference);
  EXPECT_EQ(MaxAbsDifference(y, x).max(), 0.0);
}

TEST(NormalOrderNo2b, IsLinearInThreeBodyValues) {
  const auto ms = Ne20();
  const Operator x = RandomOperator(ms, Hermiticity::kHermitian, 2, 63);
  const Operator base = NormalOrder(x);
  const Operator one = NormalOrder(x, SmallV3(1.0)) - base;
  const Operator two = NormalOrder(x, SmallV3(2.0)) - base;
  EXPECT_GT(MaxAbs(one).max(), 1e-3);  // the 3N input contributes
  ExpectEqualParts(two, 2.0 * one, 1e-12, "NO2B linearity");
}

TEST(NormalOrderNo2b, RejectsBadArguments) {
  const auto ms = Ne20();
  const No2bThreeBody v3 = SmallV3(1.0);
  const Operator nonherm =
      RandomOperator(ms, Hermiticity::kNonHermitian, 2, 64);
  EXPECT_THROW(NormalOrder(nonherm, v3), InvariantError);
  const Operator x = RandomOperator(ms, Hermiticity::kHermitian, 2, 65);
  const std::vector<double> occ(static_cast<std::size_t>(ms->num_orbits()) - 1,
                                0.5);
  EXPECT_THROW(NormalOrder(x, v3, occ, NormalOrderDirection::kToReference),
               InvariantError);
  EXPECT_THROW(NormalOrder(Operator{}, v3), InvariantError);
}

}  // namespace
