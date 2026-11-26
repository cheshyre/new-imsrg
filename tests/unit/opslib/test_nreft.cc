// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// NREFT one-body tables against the fork 79e9330 dumps
// (tests/golden/nreft/bare_*.txt, tools/reference/gen_golden_nreft.sh), the
// q -> 0 limits of the provenance README section 5 and reference/06 section
// 2.3, and the shorthand parser.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/opslib/detail/nreft_radial.h"
#include "imsrg/opslib/nreft.h"

namespace imsrg::opslib::nreft {
namespace {

namespace fs = std::filesystem;

struct GoldenOp {
  std::string name;
  int J = 0;
  int parity = 0;
  int hermitian = 0;
  int antihermitian = 0;
  std::map<std::pair<int, int>, double> elements;  // nonzero <a||O||b>
};

struct Golden {
  int emax = 0;
  double hw = 0.0;
  std::vector<std::vector<int>> orbits;  // n l j2 tz2
  std::vector<GoldenOp> ops;
};

Golden ReadGolden(const fs::path& p) {
  Golden g;
  std::ifstream in(p);
  std::string line;
  GoldenOp* cur = nullptr;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string w;
    ls >> w;
    if (w == "emax") {
      std::string h;
      ls >> g.emax >> h >> g.hw;
    } else if (w == "orbit") {
      int i, n, l, j2, tz2;
      ls >> i >> n >> l >> j2 >> tz2;
      g.orbits.push_back({n, l, j2, tz2});
    } else if (w == "op") {
      g.ops.emplace_back();
      cur = &g.ops.back();
      ls >> cur->name >> cur->J >> cur->parity >> cur->hermitian >>
          cur->antihermitian;
    } else if (w == "end") {
      cur = nullptr;
    } else if (cur != nullptr) {
      const int a = std::stoi(w);
      int b;
      std::string v;
      ls >> b >> v;
      // strtod parses "nan", "-nan", "inf" as well.
      cur->elements[{a, b}] = std::strtod(v.c_str(), nullptr);
    }
  }
  return g;
}

std::vector<std::string> Split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, c)) out.push_back(item);
  return out;
}

/// Our table for a golden operator name (shorthand or raw block).
OneBodyTable OurTable(const basis::SingleParticleBasis& basis,
                      const std::string& name) {
  if (const auto s = ParseShorthand(name))
    return BuildShorthandTable(basis, *s);
  const std::vector<std::string> f = Split(name, '_');  // raw_<X>_...
  static const std::map<std::string, Family> kFam = {
      {"M", Family::kM},
      {"Phipp", Family::kPhipp},
      {"Phip", Family::kPhip},
      {"Phitp", Family::kPhitp},
      {"Sigma", Family::kSigma},
      {"Sigmap", Family::kSigmap},
      {"Sigmapp", Family::kSigmapp},
      {"Delta", Family::kDelta},
      {"Deltap", Family::kDeltap}};
  if (f[1] == "Ms" || f[1] == "Mg") {
    const int J = std::stoi(f[2]);
    const int L = std::stoi(f[3]);
    const double q = std::stod(f[4]);
    const double y = OscillatorY(q, basis.hbar_omega());
    const Parity par = (f[1] == "Ms") ? ParityOfL(L) : ParityOfL(L + 1);
    OneBodyTable t;
    t.num_orbits = basis.num_orbits();
    t.rank_J = J;
    t.parity = par;
    const std::size_t n = static_cast<std::size_t>(t.num_orbits);
    t.values.assign(n * n, 0.0);
    for (std::size_t a = 0; a < n; ++a) {
      const basis::Orbit& oa = basis.orbits()[a];
      if (oa.tz2.twice > 0) continue;  // protons only
      for (std::size_t b = 0; b < n; ++b) {
        const basis::Orbit& ob = basis.orbits()[b];
        if (!InOneBodyChannel(J, par, oa, ob)) continue;
        t.values[a * n + b] = (f[1] == "Ms") ? MsElement(J, L, oa, ob, y)
                                             : MgElement(J, L, oa, ob, y);
      }
    }
    return t;
  }
  return BuildTable(basis, kFam.at(f[1]), std::stoi(f[2]), Isospin::kProton,
                    std::stod(f[3]));
}

struct Deviation {
  double max_rel = 0.0;  // significant elements, |ref| >= 1e-3 max|table|
  std::string max_rel_at;
  double max_floor = 0.0;  // max |d| / S_q
  std::string max_floor_at;
  int compared = 0;
  int skipped_nonfinite = 0;
  int failures = 0;
};

/// The momentum transfer of a golden name: the last `_` field (shorthands
/// `FM0p_150`, raw blocks `raw_Phip_3_150`).
std::string QOf(const std::string& name) {
  return name.substr(name.rfind('_') + 1);
}

/// Element-by-element comparison with the fork: |ours - ref| <= 1e-12 |ref|,
/// or, near zero, |ours - ref| <= 1e-14 S_q, where S_q is the largest element
/// of all tables in the file at the same q (design/30 section 2: the rule of
/// the Wigner symbols, "1e-12 relative, absolute 1e-14 near 0"). The floor is
/// per q, not per table, because some tables vanish analytically (Deltap at
/// emax 2) or nearly (high J at q = 1050 MeV) and then hold only the rounding
/// noise of the cancelling sums, in the fork as here; the fork's own noise is
/// of this size (its FM0p_15 <0||O||10> and <10||O||0> differ by 6e-14
/// relative).
void CompareGolden(const fs::path& file, double floor = 1.0e-14) {
  if (!fs::exists(file)) GTEST_SKIP() << "golden not found: " << file;
  const Golden g = ReadGolden(file);
  basis::BasisTruncation trunc;
  trunc.emax = g.emax;
  const auto basis =
      basis::SingleParticleBasis::HarmonicOscillator(trunc, g.hw);
  ASSERT_EQ(basis.num_orbits(), static_cast<int>(g.orbits.size()));
  for (std::size_t i = 0; i < g.orbits.size(); ++i) {
    const basis::Orbit& o = basis.orbits()[i];
    ASSERT_EQ(o.n, g.orbits[i][0]);
    ASSERT_EQ(o.l, g.orbits[i][1]);
    ASSERT_EQ(o.j2.twice, g.orbits[i][2]);
    ASSERT_EQ(o.tz2.twice, g.orbits[i][3]);
  }
  std::map<std::string, double> s_q;
  for (const GoldenOp& op : g.ops) {
    double& m = s_q[QOf(op.name)];
    for (const auto& [ab, v] : op.elements) {
      if (std::isfinite(v)) m = std::max(m, std::abs(v));
    }
  }
  Deviation total;
  for (const GoldenOp& op : g.ops) {
    const OneBodyTable t = OurTable(basis, op.name);
    ASSERT_EQ(t.rank_J, op.J) << op.name;
    ASSERT_EQ(static_cast<int>(t.parity), op.parity) << op.name;
    if (op.name.rfind("raw_", 0) != 0) {  // shorthands: Hermitian flags
      EXPECT_EQ(op.hermitian, 1) << op.name;
      EXPECT_EQ(t.hermiticity, Hermiticity::kHermitian) << op.name;
    }
    double scale = 0.0;
    for (const auto& [ab, v] : op.elements) {
      if (std::isfinite(v)) scale = std::max(scale, std::abs(v));
    }
    const double sq = s_q.at(QOf(op.name));
    const int n = t.num_orbits;
    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < n; ++b) {
        const auto it = op.elements.find({a, b});
        const double ref = (it == op.elements.end()) ? 0.0 : it->second;
        const double ours = t.value(OrbitIndex{a}, OrbitIndex{b});
        if (!std::isfinite(ref)) {
          ++total.skipped_nonfinite;
          continue;
        }
        ++total.compared;
        const double d = std::abs(ours - ref);
        const std::string at =
            op.name + " <" + std::to_string(a) + "||" + std::to_string(b) + ">";
        if (std::abs(ref) >= 1.0e-3 * scale &&
            d / std::abs(ref) > total.max_rel) {
          total.max_rel = d / std::abs(ref);
          total.max_rel_at = at;
        }
        if (sq > 0.0 && d / sq > total.max_floor) {
          total.max_floor = d / sq;
          total.max_floor_at = at;
        }
        if (!(d <= 1.0e-12 * std::abs(ref) || d <= floor * sq)) {
          if (++total.failures <= 20) {
            ADD_FAILURE() << at << ": ours " << ours << " fork " << ref
                          << " (max|table| " << scale << ", S_q " << sq << ")";
          }
        }
      }
    }
  }
  std::cout << "[" << file.filename().string() << "] " << g.ops.size()
            << " tables, " << total.compared << " elements compared ("
            << total.skipped_nonfinite << " non-finite in the fork skipped)"
            << "; max relative deviation of significant elements "
            << total.max_rel << " at " << total.max_rel_at << "; max |d|/S_q "
            << total.max_floor << " at " << total.max_floor_at << "; failures "
            << total.failures << "\n";
  EXPECT_EQ(total.failures, 0);
  EXPECT_GT(total.compared, 1000);
}

TEST(NreftGolden, Emax2AllTablesMatchFork) {
  CompareGolden(fs::path(IMSRG_GOLDEN_DIR) / "nreft/bare_emax2.txt");
}

TEST(NreftGolden, Emax4ShorthandsMatchFork) {
  CompareGolden(fs::path(IMSRG_GOLDEN_DIR) / "nreft/bare_emax4_shorthands.txt");
}

/// The production basis size (emax 14, the HF/NAT basis of the archived
/// runs), from the generated goldens (gen_golden_nreft.sh large; skipped if
/// absent). Here the fork itself is not exact: its double-precision radial
/// sums lose up to 8 digits for high radial n (e.g. FPhipp2p_15 <234||232>:
/// fork -0.0173100007660, exact -0.0173099860818 from the mpmath reference,
/// 8.5e-7 relative; design/30 section 6). The port is exact to rounding
/// (EMax14SampleMatchesMpmath below), so this comparison only bounds the
/// fork's error: |d| <= 1e-12 |ref| or |d| <= 1e-8 S_q (observed 1.5e-9 S_q).
/// It still catches any wrong channel, sign or prefactor.
TEST(NreftGolden, Emax14ShorthandsMatchForkWithinItsRoundingError) {
  CompareGolden(
      fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "nreft/bare_emax14_shorthands.txt",
      1.0e-8);
}

/// Elements of every shorthand family at emax 14 (shells >= 10) against the
/// brute-force mpmath evaluation of the same formulas at 50 digits
/// (tools/reference/nreft_mpref.py): 1e-12 relative, or 1e-14 of the largest
/// sampled element at that q.
TEST(NreftGolden, Emax14SampleMatchesMpmath) {
  const fs::path file = fs::path(IMSRG_GOLDEN_DIR) / "nreft/mpref_emax14.txt";
  if (!fs::exists(file)) GTEST_SKIP() << "golden not found: " << file;
  basis::BasisTruncation trunc;
  trunc.emax = 14;
  const auto basis =
      basis::SingleParticleBasis::HarmonicOscillator(trunc, 16.0);
  struct Ref {
    std::string name;
    int a, b;
    double v;
  };
  std::vector<Ref> refs;
  std::map<std::string, double> s_q;
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string w;
    ls >> w;
    if (w == "orbit") {
      int i, n, l, j2, tz2;
      ls >> i >> n >> l >> j2 >> tz2;
      const basis::Orbit& o = basis.orbits()[static_cast<std::size_t>(i)];
      ASSERT_EQ(o.n, n);
      ASSERT_EQ(o.l, l);
      ASSERT_EQ(o.j2.twice, j2);
      ASSERT_EQ(o.tz2.twice, tz2);
    } else if (w == "ref") {
      Ref r;
      ls >> r.name >> r.a >> r.b >> r.v;
      double& m = s_q[QOf(r.name)];
      m = std::max(m, std::abs(r.v));
      refs.push_back(r);
    }
  }
  ASSERT_GT(refs.size(), 500u);
  double max_rel = 0.0;
  int failures = 0;
  for (const Ref& r : refs) {
    const auto s = ParseShorthand(r.name);
    ASSERT_TRUE(s.has_value()) << r.name;
    const double y = OscillatorY(s->q_MeV, 16.0);
    const double ours =
        s->prefactor *
        FamilyElement(s->family, s->L,
                      basis.orbits()[static_cast<std::size_t>(r.a)],
                      basis.orbits()[static_cast<std::size_t>(r.b)], y);
    const double d = std::abs(ours - r.v);
    if (r.v != 0.0) max_rel = std::max(max_rel, d / std::abs(r.v));
    if (!(d <= 1.0e-12 * std::abs(r.v) || d <= 1.0e-14 * s_q.at(QOf(r.name)))) {
      ++failures;
      ADD_FAILURE() << r.name << " <" << r.a << "||" << r.b << ">: ours "
                    << ours << " mpmath " << r.v;
    }
  }
  std::cout << "[mpref_emax14.txt] " << refs.size()
            << " elements, max relative deviation " << max_rel << "\n";
  EXPECT_EQ(failures, 0);
}

// ----- q -> 0 limits (provenance README section 5, reference/06 2.3) -----

basis::SingleParticleBasis Emax2Basis() {
  basis::BasisTruncation trunc;
  trunc.emax = 2;
  return basis::SingleParticleBasis::HarmonicOscillator(trunc, 16.0);
}

// Orbits at emax 2: 0 p0s1, 2 p0p3, 4 p0p1, 6 p0d5, 8 p0d3, 10 p1s1.
double Diag(const OneBodyTable& t, int a) {
  return t.value(OrbitIndex{a}, OrbitIndex{a});
}

TEST(NreftLimits, FM0IsTheNumberOperator) {
  const auto basis = Emax2Basis();
  const auto tp = BuildShorthandTable(basis, *ParseShorthand("FM0p_0"));
  const auto tn = BuildShorthandTable(basis, *ParseShorthand("FM0n_0"));
  for (int a = 0; a < basis.num_orbits(); ++a) {
    const bool proton = basis.orbits()[a].tz2.twice < 0;
    EXPECT_NEAR(Diag(tp, a), proton ? 1.0 : 0.0, 1.0e-14);
    EXPECT_NEAR(Diag(tn, a), proton ? 0.0 : 1.0, 1.0e-14);
    for (int b = 0; b < basis.num_orbits(); ++b) {
      if (a != b) {
        EXPECT_NEAR(tp.value(OrbitIndex{a}, OrbitIndex{b}), 0.0, 1.0e-14);
      }
    }
  }
}

TEST(NreftLimits, FPhipp0IsMinusOneThirdSigmaDotL) {
  // -(1/3) sigma.l: 0d5/2 -> -2/3, 0d3/2 -> +1, 0p3/2 -> -1/3, 0p1/2 -> 2/3.
  const auto t =
      BuildShorthandTable(Emax2Basis(), *ParseShorthand("FPhipp0p_0"));
  EXPECT_NEAR(Diag(t, 6), -2.0 / 3.0, 1.0e-13);
  EXPECT_NEAR(Diag(t, 8), 1.0, 1.0e-13);
  EXPECT_NEAR(Diag(t, 2), -1.0 / 3.0, 1.0e-13);
  EXPECT_NEAR(Diag(t, 4), 2.0 / 3.0, 1.0e-13);
  EXPECT_NEAR(Diag(t, 0), 0.0, 1.0e-13);
  EXPECT_NEAR(Diag(t, 10), 0.0, 1.0e-13);
}

// <j||sigma||j> for l, j = l +- 1/2 (reduced, Edmonds convention):
// j = l + 1/2: sqrt(2j+1) sqrt((j+1)/j); j = l - 1/2: -sqrt(2j+1)
// sqrt(j/(j+1)).
double SigmaReduced(int l, int j2) {
  const double j = 0.5 * j2;
  return (j2 == 2 * l + 1) ? std::sqrt(2 * j + 1) * std::sqrt((j + 1) / j)
                           : -std::sqrt(2 * j + 1) * std::sqrt(j / (j + 1));
}
// <j||l||j> = sqrt(j(j+1)(2j+1)) * <l.j>/(j(j+1)), <l.j> =
// (j(j+1)+l(l+1)-3/4)/2.
double LReduced(int l, int j2) {
  const double j = 0.5 * j2;
  const double ldotj = 0.5 * (j * (j + 1) + l * (l + 1) - 0.75);
  return std::sqrt((2 * j + 1) / (j * (j + 1))) * ldotj;
}

TEST(NreftLimits, Sigmap1AndSigmapp1AreSigma) {
  const auto basis = Emax2Basis();
  const auto tp = BuildShorthandTable(basis, *ParseShorthand("FSigmap1p_0"));
  const auto tpp = BuildShorthandTable(basis, *ParseShorthand("FSigmapp1p_0"));
  for (int a : {0, 2, 4, 6, 8, 10}) {
    const basis::Orbit& o = basis.orbits()[a];
    const double s = SigmaReduced(o.l, o.j2.twice);
    EXPECT_NEAR(Diag(tp, a), s / std::sqrt(6 * std::numbers::pi), 1.0e-13);
    EXPECT_NEAR(Diag(tpp, a), s / (2 * std::sqrt(3 * std::numbers::pi)),
                1.0e-13);
  }
}

TEST(NreftLimits, Delta1IsMinusL) {
  const auto basis = Emax2Basis();
  const auto t = BuildShorthandTable(basis, *ParseShorthand("FDelta1p_0"));
  for (int a : {0, 2, 4, 6, 8, 10}) {
    const basis::Orbit& o = basis.orbits()[a];
    EXPECT_NEAR(
        Diag(t, a),
        -LReduced(o.l, o.j2.twice) / (2 * std::sqrt(6 * std::numbers::pi)),
        1.0e-13);
  }
}

TEST(NreftLimits, Phitp2IsSqrtThreeHalvesPhipp2) {
  // Both -> c [l x sigma]^2 at q -> 0: FPhitp2 / FPhipp2 = sqrt(3/2)
  // (reference/06 2.3); the 0d5/2 diagonal of FPhitp2 is -1.4967 (orchestrator
  // notes, phitp-normalization-notes.md).
  const auto basis = Emax2Basis();
  const auto t = BuildShorthandTable(basis, *ParseShorthand("FPhitp2p_0.0001"));
  const auto u = BuildShorthandTable(basis, *ParseShorthand("FPhipp2p_0.0001"));
  for (int a : {2, 6, 8}) {
    EXPECT_NEAR(Diag(t, a), std::sqrt(1.5) * Diag(u, a), 1.0e-8) << a;
  }
  EXPECT_NEAR(Diag(t, 6), -1.4967, 1.0e-4);
}

// ----- parser -----

TEST(NreftParse, Shorthands) {
  const auto s = ParseShorthand("FSigmapp3n_1050");
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->family, Family::kSigmapp);
  EXPECT_EQ(s->L, 3);
  EXPECT_EQ(s->isospin, Isospin::kNeutron);
  EXPECT_EQ(s->q_MeV, 1050.0);
  EXPECT_EQ(s->prefactor, 1.0);
  const auto m = ParseShorthand("FM0p_15.5");
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(m->q_MeV, 15.5);
  EXPECT_DOUBLE_EQ(m->prefactor, std::sqrt(4 * std::numbers::pi));
  EXPECT_DOUBLE_EQ(ParseShorthand("FPhitp4p_0")->prefactor,
                   PhitpShorthandPrefactor());
  for (const char* bad :
       {"FM1p_10", "FM0x_10", "FM0p10", "FM0p_", "FM0p_1a", "FPhitp1p_0",
        "FSigmap2p_0", "FDelta0n_3", "Rp2", "FM6p_10"}) {
    EXPECT_FALSE(ParseShorthand(bad).has_value()) << bad;
  }
}

TEST(NreftRadial, Hyp1F1KnownValues) {
  // 1F1(-2; 3/2; y) = 1 - 4y/3 + 4y^2/15; 1F1(1; 3/2; y) = 1 + 2y/3 + 4y^2/15 +
  // ...
  const double y = 0.7;
  EXPECT_NEAR(detail::Hyp1F1(-2, 1.5, y), 1 - 4 * y / 3 + 4 * y * y / 15,
              1.0e-15);
  EXPECT_NEAR(detail::Hyp1F1(1, 1.5, 0.0), 1.0, 0.0);
  // 1F1(a; a; y) = e^y is not of this form; use 1F1(1; 2; y) = (e^y - 1)/y
  // generalised: check against the series identity at c = 1.5 numerically:
  // 1F1(1; 3/2; y) = sqrt(pi)/(2 sqrt(y)) e^y erf(sqrt(y)).
  const double ref = std::sqrt(std::numbers::pi) / (2 * std::sqrt(y)) *
                     std::exp(y) * std::erf(std::sqrt(y));
  EXPECT_NEAR(detail::Hyp1F1(1, 1.5, y), ref, 1.0e-14 * ref);
  EXPECT_DOUBLE_EQ(detail::GammaOfHalf(1), std::sqrt(std::numbers::pi));
  EXPECT_DOUBLE_EQ(detail::GammaOfHalf(7),
                   15.0 / 8.0 * std::sqrt(std::numbers::pi));
  EXPECT_DOUBLE_EQ(detail::GammaOfHalf(8), 6.0);
}

}  // namespace
}  // namespace imsrg::opslib::nreft
