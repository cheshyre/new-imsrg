// IMSRG generators (M8 task T1): golden eta(H) of the emax-2 O16 harness, block structure and
// values of every generator kind on a random Hamiltonian, the denominator floor (design/00 D21),
// the denominator tables and the reuse of the generator operator.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/constants.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/flow/generator.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_dump.hpp"
#include "imsrg/operator/random_operator.hpp"

namespace {

using namespace imsrg;
using basis::OrbitClass;
using flow::Denominators;
using flow::GeneratorKind;
using flow::GeneratorOptions;
using flow::Partitioning;
using op::Operator;
using op::OperatorSpec;

constexpr double kPi = 3.14159265358979323846;

std::shared_ptr<const basis::ModelSpace> emax3_space() {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 3}, 16.0, "O16", "sd-shell");
}

Operator random_hamiltonian(const std::shared_ptr<const basis::ModelSpace>& ms,
                            std::uint64_t seed) {
  op::RandomOperatorOptions ro;
  ro.seed = seed;
  return op::random_operator(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian), ro);
}

bool same_ljtz(const basis::Orbit& a, const basis::Orbit& b) {
  return a.l == b.l && a.j2.twice == b.j2.twice && a.tz2.twice == b.tz2.twice;
}

int cls(const basis::ModelSpace& ms, OrbitIndex a) {
  return basis::cvq(ms.partition().orbit_class(a));
}

// Independent class tests of a two-body ket (cvq values 0 core, 1 valence, 2 q-space).
struct KetCvq {
  int cp, cq;
};
KetCvq ket_cvq(const basis::ModelSpace& ms, ChannelIndex ch, std::int32_t local) {
  const auto& tb = ms.two_body();
  const basis::Ket& k = tb.ket(tb.global_ket(ch, LocalKetIndex{local}));
  return {cls(ms, k.p), cls(ms, k.q)};
}
bool is_cc(KetCvq k) { return k.cp == 0 && k.cq == 0; }
bool is_vc(KetCvq k) { return k.cp + k.cq == 1; }
bool is_vv(KetCvq k) { return k.cp == 1 && k.cq == 1; }
bool is_qv(KetCvq k) { return k.cp + k.cq == 3; }
bool is_qq(KetCvq k) { return k.cp == 2 && k.cq == 2; }

bool is_shell_model(GeneratorKind k) {
  return k == GeneratorKind::ShellModel || k == GeneratorKind::ShellModelAtan ||
         k == GeneratorKind::ShellModelWegner || k == GeneratorKind::ShellModelImaginaryTime;
}

// True if eta(x, y) is a "downward" element: x = i in valence u q, y = a in core (u valence for
// the shell-model kinds, x != y).
bool one_body_down(const basis::ModelSpace& ms, bool shell, OrbitIndex x, OrbitIndex y) {
  if (x == y) return false;
  if (!same_ljtz(ms.basis().orbit(x), ms.basis().orbit(y))) return false;
  const int cx = cls(ms, x), cy = cls(ms, y);
  if (cx < 1) return false;
  return shell ? cy <= 1 : cy == 0;
}

// Checks eta (built from H with `opts`) against the documented block structure and f(H, Delta).
void check_generator_structure(const Operator& H_in, const Operator& eta_in,
                               const GeneratorOptions& opts) {
  Operator H(H_in), eta(eta_in);
  H.sync_to_host();
  eta.sync_to_host();
  const basis::ModelSpace& ms = H.model_space();
  const Denominators den(H, opts);
  const bool shell = is_shell_model(opts.kind);
  const int nq = opts.qtransfer_n;
  const std::int32_t norb = ms.num_orbits();
  const std::string label = "kind " + std::to_string(static_cast<int>(opts.kind));

  EXPECT_EQ(eta.hermiticity(), Hermiticity::AntiHermitian) << label;
  double tol_scale = 1.0;
  for (std::int32_t p = 0; p < norb; ++p)
    for (std::int32_t q = 0; q < norb; ++q)
      tol_scale = std::max(tol_scale, std::abs(eta.one_body().get(OrbitIndex{p}, OrbitIndex{q})));

  int nonzero_1b = 0;
  for (std::int32_t p = 0; p < norb; ++p) {
    for (std::int32_t q = 0; q < norb; ++q) {
      const OrbitIndex x{p}, y{q};
      double expected = 0.0;
      if (one_body_down(ms, shell, x, y)) {
        expected = flow::generator_function(opts.kind, H.one_body().get(x, y), den.one_body(x, y), nq);
      } else if (one_body_down(ms, shell, y, x)) {
        expected =
            flow::generator_function(opts.kind, H.one_body().get(x, y), -den.one_body(y, x), nq);
      }
      const double got = eta.one_body().get(x, y);
      EXPECT_NEAR(got, expected, 1e-12 * std::max(1.0, std::abs(expected))) << label << " 1B " << p << "," << q;
      EXPECT_NEAR(got, -eta.one_body().get(y, x), 1e-12 * std::max(1.0, std::abs(got)))
          << label << " 1B antihermitian " << p << "," << q;
      if (got != 0.0) ++nonzero_1b;
    }
  }
  EXPECT_GT(nonzero_1b, 0) << label;

  if (opts.only_1b_eta) {
    // All of the two-body part must be zero.
    EXPECT_EQ(op::max_abs(eta).two_body, 0.0) << label;
    return;
  }
  int nonzero_2b = 0;
  const std::int32_t nch = ms.two_body().num_channels();
  for (std::int32_t c = 0; c < nch; ++c) {
    const ChannelIndex ch{c};
    const auto n = static_cast<std::int32_t>(ms.two_body().channel_kets(ch).size());
    auto bra_ok = [&](KetCvq k) { return is_vv(k) || is_qv(k) || is_qq(k); };
    for (std::int32_t r = 0; r < n; ++r) {
      for (std::int32_t s = 0; s < n; ++s) {
        const KetCvq kr = ket_cvq(ms, ch, r), ks = ket_cvq(ms, ch, s);
        // (bra, ket) allowed for an element (bra = x, ket = y).
        auto allowed = [&](KetCvq bra, KetCvq ket) {
          if (!shell) return is_cc(ket) && (is_qq(bra) || is_vv(bra) || is_qv(bra));
          if (is_cc(ket) || is_vc(ket)) return bra_ok(bra);
          if (is_vv(ket)) return is_qv(bra) || is_qq(bra);
          return false;
        };
        double expected = 0.0;
        if (allowed(kr, ks)) {
          expected = flow::generator_function(
              opts.kind, H.two_body().get(ch, ch, LocalKetIndex{r}, LocalKetIndex{s}),
              den.two_body(ch, LocalKetIndex{r}, LocalKetIndex{s}), nq);
        } else if (allowed(ks, kr)) {
          expected = -flow::generator_function(
              opts.kind, H.two_body().get(ch, ch, LocalKetIndex{s}, LocalKetIndex{r}),
              den.two_body(ch, LocalKetIndex{s}, LocalKetIndex{r}), nq);
        }
        const double got = eta.two_body().get(ch, ch, LocalKetIndex{r}, LocalKetIndex{s});
        ASSERT_NEAR(got, expected, 1e-12 * std::max(1.0, std::abs(expected)))
            << label << " 2B ch " << c << " (" << r << "," << s << ")";
        if (got != 0.0) ++nonzero_2b;
      }
    }
  }
  EXPECT_GT(nonzero_2b, 0) << label;
}

// ---- 1. golden -----------------------------------------------------------------------------

double golden_deviation(const char* eta_file, GeneratorKind kind, const char* name) {
  const std::string dir = std::string(IMSRG_GOLDEN_DIR) + "/commutator/emax2_O16/";
  const auto header = op::read_operator_dump_header(dir + "H.dump");
  const auto ms = op::model_space_for_dump(header);
  const Operator H = op::read_operator_dump(dir + "H.dump", ms);
  const Operator eta_ref = op::read_operator_dump(dir + eta_file, ms);
  GeneratorOptions opts;
  opts.kind = kind;
  const Operator eta = flow::compute_generator(H, opts);
  const op::PartMaxima ref = op::max_abs(eta_ref);
  const op::PartMaxima dev = op::max_abs_difference(eta, eta_ref);
  const double tol = 1e-10 * std::max(1.0, ref.max());
  std::printf("[golden %s] max|d eta|: 0B %.3e  1B %.3e  2B %.3e (tol %.3e, max|eta| %.3e)\n", name,
              dev.zero_body, dev.one_body, dev.two_body, tol, ref.max());
  EXPECT_LE(dev.zero_body, tol) << name;
  EXPECT_LE(dev.one_body, tol) << name;
  EXPECT_LE(dev.two_body, tol) << name;
  EXPECT_GT(ref.one_body, 0.0) << name;
  EXPECT_GT(ref.two_body, 0.0) << name;
  return std::max({dev.zero_body, dev.one_body, dev.two_body});
}

TEST(GeneratorGolden, White) { golden_deviation("eta_white.dump", GeneratorKind::White, "white"); }

TEST(GeneratorGolden, Atan) { golden_deviation("eta_atan.dump", GeneratorKind::Atan, "atan"); }

// ---- 2. structure and values ---------------------------------------------------------------

TEST(GeneratorStructure, AllKindsEpsteinNesbet) {
  const auto ms = emax3_space();
  const Operator H = random_hamiltonian(ms, 11);
  const GeneratorKind kinds[] = {GeneratorKind::White,
                                 GeneratorKind::Atan,
                                 GeneratorKind::ImaginaryTime,
                                 GeneratorKind::Wegner,
                                 GeneratorKind::QtransferAtan,
                                 GeneratorKind::ShellModel,
                                 GeneratorKind::ShellModelAtan,
                                 GeneratorKind::ShellModelWegner,
                                 GeneratorKind::ShellModelImaginaryTime};
  for (GeneratorKind k : kinds) {
    GeneratorOptions opts;
    opts.kind = k;
    const Operator eta = flow::compute_generator(H, opts);
    check_generator_structure(H, eta, opts);
  }
}

TEST(GeneratorStructure, QtransferSecondExponent) {
  const auto ms = emax3_space();
  const Operator H = random_hamiltonian(ms, 12);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::QtransferAtan;
  opts.qtransfer_n = 2;
  const Operator eta = flow::compute_generator(H, opts);
  check_generator_structure(H, eta, opts);
}

TEST(GeneratorStructure, PartitioningsAndDelta) {
  const auto ms = emax3_space();
  const Operator H = random_hamiltonian(ms, 13);
  for (Partitioning p : {Partitioning::MollerPlesset, Partitioning::MpIsospin}) {
    GeneratorOptions opts;
    opts.kind = GeneratorKind::Atan;
    opts.partitioning = p;
    check_generator_structure(H, flow::compute_generator(H, opts), opts);
    opts.kind = GeneratorKind::ShellModelAtan;
    opts.denominator_delta = 0.7;
    opts.denominator_delta_all = true;
    check_generator_structure(H, flow::compute_generator(H, opts), opts);
  }
}

TEST(GeneratorStructure, OnlyOneBodyEta) {
  const auto ms = emax3_space();
  const Operator H = random_hamiltonian(ms, 14);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::White;
  opts.only_1b_eta = true;
  check_generator_structure(H, flow::compute_generator(H, opts), opts);
}

TEST(GeneratorStructure, HartreeFock) {
  const auto ms = emax3_space();
  const Operator H = random_hamiltonian(ms, 15);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::HartreeFock;
  Operator eta = flow::compute_generator(H, opts);
  EXPECT_EQ(eta.spec().particle_rank, 1);
  EXPECT_FALSE(eta.two_body().allocated());
  EXPECT_EQ(eta.hermiticity(), Hermiticity::AntiHermitian);
  Operator Hh(H);
  Hh.sync_to_host();
  eta.sync_to_host();
  const Denominators den(H, opts);
  const std::int32_t norb = ms->num_orbits();
  int nonzero = 0;
  for (std::int32_t i = 0; i < norb; ++i) {
    for (std::int32_t j = 0; j < norb; ++j) {
      const OrbitIndex oi{i}, oj{j};
      double expected = 0.0;
      if (j < i && same_ljtz(ms->basis().orbit(oi), ms->basis().orbit(oj))) {
        expected = Hh.one_body().get(oi, oj) / den.one_body(oi, oj);
      } else if (i < j && same_ljtz(ms->basis().orbit(oi), ms->basis().orbit(oj))) {
        expected = -Hh.one_body().get(oj, oi) / den.one_body(oj, oi);
      }
      EXPECT_NEAR(eta.one_body().get(oi, oj), expected, 1e-12 * std::max(1.0, std::abs(expected)));
      if (expected != 0.0) ++nonzero;
    }
  }
  EXPECT_GT(nonzero, 0);
  for (std::int32_t i = 0; i < norb; ++i)
    EXPECT_EQ(eta.one_body().get(OrbitIndex{i}, OrbitIndex{i}), 0.0);
}

// ---- 3. floor ------------------------------------------------------------------------------

TEST(GeneratorFloor, FloorDenominator) {
  EXPECT_EQ(flow::floor_denominator(-1e-8, 1e-6), -1e-6);
  EXPECT_EQ(flow::floor_denominator(1e-8, 1e-6), 1e-6);
  EXPECT_EQ(flow::floor_denominator(0.0, 1e-6), 1e-6);
  EXPECT_EQ(flow::floor_denominator(-0.0, 1e-6), 1e-6);
  EXPECT_EQ(flow::floor_denominator(-0.5, 1e-6), -0.5);
  EXPECT_EQ(flow::floor_denominator(0.5, 1e-6), 0.5);
  EXPECT_EQ(flow::floor_denominator(-1e-6, 1e-6), -1e-6);
}

TEST(GeneratorFloor, SmallNegativeOneBodyDenominatorKeepsItsSign) {
  const auto ms = emax3_space();
  const basis::SingleParticleBasis& sp = ms->basis();
  const auto a = sp.find(basis::Orbit{.n = 0, .l = 0, .j2 = TwoJ{1}, .tz2 = TwoTz{-1}});
  const auto i = sp.find(basis::Orbit{.n = 1, .l = 0, .j2 = TwoJ{1}, .tz2 = TwoTz{-1}});
  ASSERT_TRUE(a.has_value() && i.has_value());
  ASSERT_EQ(cls(*ms, *a), 0);  // proton 0s1/2: core
  ASSERT_GE(cls(*ms, *i), 1);  // proton 1s1/2: valence
  Operator H(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian));
  H.one_body().set(*a, *a, 0.0);
  H.one_body().set(*i, *i, -1e-8);  // f_ii - f_aa = -1e-8
  H.one_body().set(*i, *a, 0.3);
  H.sync_to_device();

  GeneratorOptions opts;
  opts.kind = GeneratorKind::White;
  opts.partitioning = Partitioning::MollerPlesset;
  Operator eta = flow::compute_generator(H, opts);
  eta.sync_to_host();
  EXPECT_DOUBLE_EQ(eta.one_body().get(*i, *a), 0.3 / -1e-6);
  EXPECT_DOUBLE_EQ(eta.one_body().get(*a, *i), 0.3 / 1e-6);
  EXPECT_LT(eta.one_body().get(*i, *a), 0.0);
  EXPECT_DOUBLE_EQ(Denominators(H, opts).one_body(*i, *a), -1e-6);
}

// ---- 4. denominators -----------------------------------------------------------------------

TEST(GeneratorDenominators, MonopoleEqualsDirectJSum) {
  const auto ms = emax3_space();
  const Operator H0 = random_hamiltonian(ms, 21);
  Operator H(H0);
  H.sync_to_host();
  GeneratorOptions opts;  // Epstein-Nesbet
  const Denominators den(H, opts);
  const std::int32_t norb = ms->num_orbits();
  const std::int32_t pairs[][2] = {{0, 0}, {0, 3}, {2, 5}, {3, 3}, {1, 7}, {norb - 1, norb - 1},
                                   {4, norb - 2}, {5, 15}, {6, 6}, {10, 10}, {0, 10}};
  for (const auto& pr : pairs) {
    const OrbitIndex a{pr[0]}, b{pr[1]};
    const int ja = ms->basis().orbit(a).j2.twice, jb = ms->basis().orbit(b).j2.twice;
    double sum = 0.0;
    for (int tj = std::abs(ja - jb); tj <= ja + jb; tj += 2)
      sum += (tj + 1) * H.two_body().get_unnormalized(a, b, a, b, TwoJ{tj});
    const double direct = sum / ((ja + 1.0) * (jb + 1.0));
    EXPECT_NEAR(den.monopole(a, b), direct, 1e-12 * std::max(1.0, std::abs(direct)))
        << pr[0] << "," << pr[1];
    EXPECT_DOUBLE_EQ(den.monopole(a, b), den.monopole(b, a));
  }
  // Non-trivial values.
  EXPECT_NE(den.monopole(OrbitIndex{0}, OrbitIndex{3}), 0.0);
  // Not built for the other partitionings.
  GeneratorOptions mp;
  mp.partitioning = Partitioning::MollerPlesset;
  EXPECT_EQ(Denominators(H, mp).monopole(OrbitIndex{0}, OrbitIndex{3}), 0.0);
}

TEST(GeneratorDenominators, EpsteinNesbetOneAndTwoBody) {
  const auto ms = emax3_space();
  Operator H = random_hamiltonian(ms, 22);
  H.sync_to_host();
  GeneratorOptions opts;
  const Denominators den(H, opts);
  const auto occ = ms->reference().occupations();
  const auto fdiag = [&](OrbitIndex x) { return H.one_body().get(x, x); };
  const std::int32_t norb = ms->num_orbits();
  for (std::int32_t p = 0; p < norb; p += 3) {
    for (std::int32_t q = 1; q < norb; q += 4) {
      const OrbitIndex i{p}, j{q};
      const double d = fdiag(i) - fdiag(j) +
                       (occ[i.to_size()] - occ[j.to_size()]) * den.monopole(i, j);
      EXPECT_DOUBLE_EQ(den.one_body(i, j), flow::floor_denominator(d, 1e-6));
    }
  }
  const auto& tb = ms->two_body();
  int checked = 0;
  for (std::int32_t c = 0; c < tb.num_channels(); c += 5) {
    const ChannelIndex ch{c};
    const auto n = static_cast<std::int32_t>(tb.channel_kets(ch).size());
    for (std::int32_t r = 0; r < n; r += 3) {
      for (std::int32_t s = 0; s < n; s += 4) {
        const basis::Ket& bra = tb.ket(tb.global_ket(ch, LocalKetIndex{r}));
        const basis::Ket& ket = tb.ket(tb.global_ket(ch, LocalKetIndex{s}));
        const OrbitIndex i = bra.p, j = bra.q, k = ket.p, l = ket.q;
        auto n_of = [&](OrbitIndex x) { return occ[x.to_size()]; };
        const double d =
            fdiag(i) + fdiag(j) - fdiag(k) - fdiag(l) +
            (1 - n_of(i) - n_of(j)) * den.monopole(i, j) -
            (1 - n_of(k) - n_of(l)) * den.monopole(k, l) + (n_of(i) - n_of(k)) * den.monopole(i, k) +
            (n_of(i) - n_of(l)) * den.monopole(i, l) + (n_of(j) - n_of(k)) * den.monopole(j, k) +
            (n_of(j) - n_of(l)) * den.monopole(j, l);
        EXPECT_NEAR(den.two_body(ch, LocalKetIndex{r}, LocalKetIndex{s}),
                    flow::floor_denominator(d, 1e-6), 1e-12 * std::max(1.0, std::abs(d)));
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 50);
}

TEST(GeneratorDenominators, MollerPlessetIsSpeDifference) {
  const auto ms = emax3_space();
  Operator H = random_hamiltonian(ms, 23);
  H.sync_to_host();
  GeneratorOptions opts;
  opts.partitioning = Partitioning::MollerPlesset;
  const Denominators den(H, opts);
  const auto fdiag = [&](OrbitIndex x) { return H.one_body().get(x, x); };
  const std::int32_t norb = ms->num_orbits();
  for (std::int32_t p = 0; p < norb; ++p) {
    for (std::int32_t q = 0; q < norb; ++q) {
      EXPECT_DOUBLE_EQ(den.one_body(OrbitIndex{p}, OrbitIndex{q}),
                       flow::floor_denominator(fdiag(OrbitIndex{p}) - fdiag(OrbitIndex{q}), 1e-6));
    }
  }
  const auto& tb = ms->two_body();
  for (std::int32_t c = 0; c < tb.num_channels(); c += 4) {
    const ChannelIndex ch{c};
    const auto n = static_cast<std::int32_t>(tb.channel_kets(ch).size());
    for (std::int32_t r = 0; r < n; r += 3) {
      for (std::int32_t s = 0; s < n; s += 5) {
        const basis::Ket& bra = tb.ket(tb.global_ket(ch, LocalKetIndex{r}));
        const basis::Ket& ket = tb.ket(tb.global_ket(ch, LocalKetIndex{s}));
        const double d = fdiag(bra.p) + fdiag(bra.q) - fdiag(ket.p) - fdiag(ket.q);
        EXPECT_DOUBLE_EQ(den.two_body(ch, LocalKetIndex{r}, LocalKetIndex{s}),
                         flow::floor_denominator(d, 1e-6));
      }
    }
  }
}

TEST(GeneratorDenominators, MpIsospinAveragesPartners) {
  const auto ms = emax3_space();
  Operator H = random_hamiltonian(ms, 24);
  H.sync_to_host();
  GeneratorOptions opts;
  opts.partitioning = Partitioning::MpIsospin;
  const Denominators den(H, opts);
  const basis::SingleParticleBasis& sp = ms->basis();
  const auto fdiag = [&](OrbitIndex x) { return H.one_body().get(x, x); };
  auto partner = [&](OrbitIndex x) {
    basis::Orbit o = sp.orbit(x);
    o.tz2 = TwoTz{-o.tz2.twice};
    const auto f = sp.find(o);
    return f.value_or(x);
  };
  const std::int32_t norb = ms->num_orbits();
  for (std::int32_t p = 0; p < norb; ++p) {
    for (std::int32_t q = 0; q < norb; ++q) {
      const OrbitIndex i{p}, j{q};
      const double d = 0.5 * ((fdiag(i) - fdiag(j)) + (fdiag(partner(i)) - fdiag(partner(j))));
      EXPECT_NEAR(den.one_body(i, j), flow::floor_denominator(d, 1e-6), 1e-14);
    }
  }
  const auto& tb = ms->two_body();
  for (std::int32_t c = 0; c < tb.num_channels(); c += 4) {
    const ChannelIndex ch{c};
    const auto n = static_cast<std::int32_t>(tb.channel_kets(ch).size());
    for (std::int32_t r = 0; r < n; r += 3) {
      for (std::int32_t s = 0; s < n; s += 5) {
        const basis::Ket& bra = tb.ket(tb.global_ket(ch, LocalKetIndex{r}));
        const basis::Ket& ket = tb.ket(tb.global_ket(ch, LocalKetIndex{s}));
        const double d0 = fdiag(bra.p) + fdiag(bra.q) - fdiag(ket.p) - fdiag(ket.q);
        const double d1 = fdiag(partner(bra.p)) + fdiag(partner(bra.q)) - fdiag(partner(ket.p)) -
                          fdiag(partner(ket.q));
        EXPECT_NEAR(den.two_body(ch, LocalKetIndex{r}, LocalKetIndex{s}),
                    flow::floor_denominator(0.5 * (d0 + d1), 1e-6), 1e-14);
      }
    }
  }
}

TEST(GeneratorDenominators, DeltaAllAndOrbit) {
  const auto ms = emax3_space();
  Operator H = random_hamiltonian(ms, 25);
  H.sync_to_host();
  GeneratorOptions base;
  base.partitioning = Partitioning::MollerPlesset;
  const Denominators d0(H, base);
  GeneratorOptions all = base;
  all.denominator_delta = 0.25;
  all.denominator_delta_all = true;
  const Denominators dall(H, all);
  GeneratorOptions orb = base;
  orb.denominator_delta = 0.25;
  orb.denominator_delta_orbit = OrbitIndex{3};
  const Denominators dorb(H, orb);
  const auto fdiag = [&](OrbitIndex x) { return H.one_body().get(x, x); };
  const OrbitIndex i{5}, j{8}, k{3};
  EXPECT_DOUBLE_EQ(dall.one_body(i, j),
                   flow::floor_denominator(fdiag(i) - fdiag(j) + 0.25, 1e-6));
  EXPECT_DOUBLE_EQ(dorb.one_body(i, j), d0.one_body(i, j));  // orbit 3 not involved
  EXPECT_DOUBLE_EQ(dorb.one_body(k, j),
                   flow::floor_denominator(fdiag(k) - fdiag(j) + 0.25, 1e-6));
  EXPECT_DOUBLE_EQ(dorb.one_body(j, k),
                   flow::floor_denominator(fdiag(j) - fdiag(k) + 0.25, 1e-6));
  // Two-body: only with the "all" flag (imsrg++ quirk P5).
  const auto& tb = ms->two_body();
  const ChannelIndex ch{0};
  const auto n = static_cast<std::int32_t>(tb.channel_kets(ch).size());
  ASSERT_GE(n, 2);
  EXPECT_DOUBLE_EQ(dorb.two_body(ch, LocalKetIndex{0}, LocalKetIndex{1}),
                   d0.two_body(ch, LocalKetIndex{0}, LocalKetIndex{1}));
  const basis::Ket& bra = tb.ket(tb.global_ket(ch, LocalKetIndex{0}));
  const basis::Ket& ket = tb.ket(tb.global_ket(ch, LocalKetIndex{1}));
  const double s = fdiag(bra.p) + fdiag(bra.q) - fdiag(ket.p) - fdiag(ket.q) + 0.25;
  EXPECT_DOUBLE_EQ(dall.two_body(ch, LocalKetIndex{0}, LocalKetIndex{1}),
                   flow::floor_denominator(s, 1e-6));
}

// ---- 5. generator_function -----------------------------------------------------------------

TEST(GeneratorFunction, SpotValues) {
  using flow::generator_function;
  const double w1 = std::sqrt(2.0 * constants::m_nucleon / (constants::hbarc * constants::hbarc));
  const double w2 = 2.0 * constants::m_nucleon / (constants::hbarc * constants::hbarc);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::White, 2.0, 4.0), 0.5);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::White, 2.0, -4.0), -0.5);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::Atan, 1.0, 2.0), 0.5 * std::atan(1.0));
  EXPECT_NEAR(generator_function(GeneratorKind::Atan, 1.0, 2.0), kPi / 8.0, 1e-15);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::Atan, 1.0, -2.0), -kPi / 8.0);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::ImaginaryTime, 3.0, -2.0), -3.0);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::ImaginaryTime, 3.0, 2.0), 3.0);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::Wegner, 3.0, -2.0), -6.0);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::Wegner, 3.0, 2.0), 6.0);
  EXPECT_NEAR(generator_function(GeneratorKind::QtransferAtan, 1.0, 2.0, 1),
              w1 * 0.5 * std::atan(1.0), 1e-15);
  EXPECT_NEAR(generator_function(GeneratorKind::QtransferAtan, 1.0, -2.0, 1),
              -w1 * 0.5 * std::atan(1.0), 1e-15);
  EXPECT_NEAR(generator_function(GeneratorKind::QtransferAtan, 1.0, 2.0, 2),
              w2 * 0.5 * std::atan(1.0), 1e-15);
  // Shell-model kinds use the corresponding single-reference function.
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::ShellModel, 2.0, 4.0), 0.5);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::ShellModelAtan, 1.0, 2.0),
                   generator_function(GeneratorKind::Atan, 1.0, 2.0));
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::ShellModelWegner, 3.0, -2.0), -6.0);
  EXPECT_DOUBLE_EQ(generator_function(GeneratorKind::ShellModelImaginaryTime, 3.0, -2.0), -3.0);
}

TEST(GeneratorFunction, NotImplemented) {
  EXPECT_THROW(flow::generator_function(GeneratorKind::ShellModelAtanNpnh, 1.0, 2.0),
               NotImplementedError);
  EXPECT_THROW(flow::generator_function(GeneratorKind::OnePa, 1.0, 2.0), NotImplementedError);
  const auto ms = emax3_space();
  const Operator H = random_hamiltonian(ms, 31);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::ShellModelAtanNpnh;
  EXPECT_THROW(flow::compute_generator(H, opts), NotImplementedError);
  opts.kind = GeneratorKind::OnePa;
  EXPECT_THROW(flow::compute_generator(H, opts), NotImplementedError);
}

// ---- 6. reuse ------------------------------------------------------------------------------

TEST(GeneratorReuse, SecondCallDoesNotReallocate) {
  const auto ms = emax3_space();
  const Operator H1 = random_hamiltonian(ms, 41);
  const Operator H2 = random_hamiltonian(ms, 42);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::ShellModelAtan;
  Operator eta;
  flow::compute_generator(H1, eta, opts);
  const double* p1 = eta.one_body().buffer().device_data();
  const double* p2 = eta.two_body().matrix().buffer().device_data();

  flow::compute_generator(H1, eta, opts);
  EXPECT_EQ(eta.one_body().buffer().device_data(), p1);
  EXPECT_EQ(eta.two_body().matrix().buffer().device_data(), p2);
  const Operator fresh1 = flow::compute_generator(H1, opts);
  const op::PartMaxima d1 = op::max_abs_difference(eta, fresh1);
  EXPECT_EQ(d1.max(), 0.0);

  // A different H on the same eta, then a kind with fewer blocks: stale elements must vanish.
  flow::compute_generator(H2, eta, opts);
  EXPECT_EQ(eta.one_body().buffer().device_data(), p1);
  EXPECT_EQ(op::max_abs_difference(eta, flow::compute_generator(H2, opts)).max(), 0.0);
  GeneratorOptions white;
  white.kind = GeneratorKind::White;
  flow::compute_generator(H2, eta, white);
  EXPECT_EQ(eta.two_body().matrix().buffer().device_data(), p2);
  EXPECT_EQ(op::max_abs_difference(eta, flow::compute_generator(H2, white)).max(), 0.0);

  // A different spec (Hartree-Fock: particle_rank 1) replaces eta.
  GeneratorOptions hf;
  hf.kind = GeneratorKind::HartreeFock;
  flow::compute_generator(H2, eta, hf);
  EXPECT_EQ(eta.spec().particle_rank, 1);
  EXPECT_FALSE(eta.two_body().allocated());
  EXPECT_EQ(op::max_abs_difference(eta, flow::compute_generator(H2, hf)).max(), 0.0);
}

}  // namespace
