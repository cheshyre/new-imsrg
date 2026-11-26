// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/opslib/nreft.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/opslib/detail/nreft_radial.h"

namespace imsrg::opslib::nreft {

namespace {

using detail::Jdmho;
using detail::Jdpho;
using detail::Jho;

// Wigner symbols with the fork's gsl argument convention (twice the values).
double ThreeJ000(int two_a, int two_b, int two_c) {
  return angmom::Wigner3j(TwoJ{two_a}, TwoJ{two_b}, TwoJ{two_c}, TwoM{0},
                          TwoM{0}, TwoM{0});
}
double SixJ(int a, int b, int c, int d, int e, int f) {
  return angmom::Wigner6j(TwoJ{a}, TwoJ{b}, TwoJ{c}, TwoJ{d}, TwoJ{e}, TwoJ{f});
}
double NineJ(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
  return angmom::Wigner9j(TwoJ{a}, TwoJ{b}, TwoJ{c}, TwoJ{d}, TwoJ{e}, TwoJ{f},
                          TwoJ{g}, TwoJ{h}, TwoJ{i});
}
double MinusOnePow(int k) { return (k % 2 == 0) ? 1.0 : -1.0; }

const double kInvSqrtFourPi = 1.0 / std::sqrt(4 * std::numbers::pi);

// The fork stores J = 0, even-parity building blocks non-reduced.
double ScalarStorage(double v, std::int32_t J, Parity parity, int j2a) {
  if (J == 0 && parity == Parity::kEven) v /= std::sqrt(j2a + 1.0);
  return v;
}

double PhiF(int la, int j2a, int j2b) {
  return MinusOnePow(la + 1) * 6 *
         std::sqrt((2 * la + 1) * (j2a + 1) * (j2b + 1)) /
         std::sqrt(4 * std::numbers::pi);
}

// PhiS1..PhiS4 (DarkMatterNREFT.cc:582-737): the multipole J+1 (S1, S2) and
// J-1 (S3, S4) terms of Phi' and Phi''. `lshift` = +1 (jdmho, l_b + 1) or -1
// (jdpho, l_b - 1); `K` = J + 1 or J - 1; `sign0` is the sign offset of the
// lambda sum.
double PhiS(int na, int la, int j2a, int nb, int lb, int j2b, int J, int K,
            int lshift, int sign0, double y) {
  const int lb2 = lb + lshift;
  double s = 0.0;
  const int lam_lo = (K > J) ? J : J - 1;
  for (int lam = lam_lo; lam <= lam_lo + 1; ++lam) {
    s += MinusOnePow(J - lam + sign0) *
         std::sqrt((2 * lam + 1) * (2 * lam + 1)) *
         SixJ(2 * K, 2, 2 * lam, 2, 2 * J, 2) *
         SixJ(2 * K, 2, 2 * lam, 2 * lb, 2 * la, 2 * lb2) *
         NineJ(2 * la, 2 * lb, 2 * lam, 1, 1, 2, j2a, j2b, 2 * J);
  }
  const double radial =
      (lshift > 0) ? Jdmho(na, la, nb, lb, K, y) : Jdpho(na, la, nb, lb, K, y);
  const double rest = (lshift > 0)
                          ? std::sqrt(2 * (lb + 1) + 1) * std::sqrt(lb + 1)
                          : std::sqrt(2 * (lb - 1) + 1) * std::sqrt(lb);
  s *= rest * ThreeJ000(2 * la, 2 * K, 2 * lb2) * radial;
  return s;
}

struct PhiSums {
  double s12 = 0.0;  // S1 + S2 (multipole J + 1)
  double s34 = 0.0;  // S3 + S4 (multipole J - 1), zero for J = 0
};

PhiSums PhiTerms(const basis::Orbit& a, const basis::Orbit& b, int J,
                 double y) {
  const int na = a.n, la = a.l, j2a = a.j2.twice;
  const int nb = b.n, lb = b.l, j2b = b.j2.twice;
  PhiSums t;
  const double s1 = PhiS(na, la, j2a, nb, lb, j2b, J, J + 1, +1, 1, y);
  const double s2 =
      (lb == 0) ? 0.0 : PhiS(na, la, j2a, nb, lb, j2b, J, J + 1, -1, 0, y);
  t.s12 = s1 + s2;
  if (J > 0) {
    const double s3 = PhiS(na, la, j2a, nb, lb, j2b, J, J - 1, +1, 1, y);
    const double s4 =
        (lb == 0) ? 0.0 : PhiS(na, la, j2a, nb, lb, j2b, J, J - 1, -1, 0, y);
    t.s34 = s3 + s4;
  }
  return t;
}

double PhipElement(int J, const basis::Orbit& a, const basis::Orbit& b,
                   double y) {
  const PhiSums t = PhiTerms(a, b, J, y);
  const double f = PhiF(a.l, a.j2.twice, b.j2.twice);
  double v = f * (-std::sqrt(2 * (J + 1) + 1) * std::sqrt(J) * t.s12);
  if (J > 0) v += f * (std::sqrt(2 * (J - 1) + 1) * std::sqrt(J + 1) * t.s34);
  return ScalarStorage(v, J, ParityOfL(J), a.j2.twice);
}

double PhippElement(int J, const basis::Orbit& a, const basis::Orbit& b,
                    double y) {
  const PhiSums t = PhiTerms(a, b, J, y);
  const double f = PhiF(a.l, a.j2.twice, b.j2.twice);
  double v = f * (std::sqrt(2 * (J + 1) + 1) * std::sqrt(J + 1) * t.s12);
  if (J > 0) v += f * (std::sqrt(2 * (J - 1) + 1) * std::sqrt(J) * t.s34);
  return ScalarStorage(v, J, ParityOfL(J), a.j2.twice);
}

double MElement(int J, const basis::Orbit& a, const basis::Orbit& b, double y) {
  const int la = a.l, j2a = a.j2.twice, lb = b.l, j2b = b.j2.twice;
  const double v = kInvSqrtFourPi * MinusOnePow((2 * J + j2b + 1) / 2) *
                   std::sqrt((2 * la + 1.) * (2 * lb + 1.) * (j2a + 1.) *
                             (j2b + 1.) * (2 * J + 1.)) *
                   SixJ(2 * la, j2a, 1, j2b, 2 * lb, 2 * J) *
                   ThreeJ000(2 * la, 2 * J, 2 * lb) *
                   Jho(a.n, la, b.n, lb, J, y);
  return ScalarStorage(v, J, ParityOfL(J), j2a);
}

// Isospin factor of the bra orbit's species (fork `isofactor`).
double IsoFactor(Isospin iso, const basis::Orbit& a) {
  const bool proton = a.tz2.twice < 0;
  switch (iso) {
    case Isospin::kIsoscalar:
      return 1.0;
    case Isospin::kIsovector:
      return proton ? 1.0 : -1.0;
    case Isospin::kProton:
      return proton ? 1.0 : 0.0;
    case Isospin::kNeutron:
      return proton ? 0.0 : 1.0;
  }
  return 0.0;
}

}  // namespace

Parity FamilyParity(Family family, std::int32_t J) {
  switch (family) {
    case Family::kM:
    case Family::kSigma:
    case Family::kPhip:
    case Family::kPhipp:
    case Family::kPhitp:
    case Family::kDeltap:  // Mg(J, J -+ 1): parity (-1)^J
      return ParityOfL(J);
    case Family::kSigmap:
    case Family::kSigmapp:
    case Family::kDelta:
      return ParityOfL(J + 1);
  }
  return Parity::kEven;
}

Hermiticity FamilyHermiticity(Family family) {
  switch (family) {
    case Family::kSigma:
    case Family::kDeltap:
      return Hermiticity::kAntiHermitian;
    case Family::kPhip:
      return Hermiticity::kNonHermitian;
    default:
      return Hermiticity::kHermitian;
  }
}

double OscillatorY(double q_MeV, double hw_MeV) {
  const double b2 = 1.0 / (hw_MeV * constants::kMNucleon);
  return q_MeV * q_MeV * b2 / 4.0;
}

double MsElement(std::int32_t J, std::int32_t L, const basis::Orbit& a,
                 const basis::Orbit& b, double y) {
  const int la = a.l, j2a = a.j2.twice, lb = b.l, j2b = b.j2.twice;
  const double v = std::sqrt(1.5 / std::numbers::pi) * MinusOnePow(la) *
                   std::sqrt((2 * la + 1.) * (2 * lb + 1.) * (j2a + 1.) *
                             (j2b + 1.) * (2 * L + 1) * (2 * J + 1.)) *
                   NineJ(2 * la, 2 * lb, 2 * L, 1, 1, 2, j2a, j2b, 2 * J) *
                   ThreeJ000(2 * la, 2 * L, 2 * lb) *
                   Jho(a.n, la, b.n, lb, L, y);
  return ScalarStorage(v, J, ParityOfL(L), j2a);
}

double MgElement(std::int32_t J, std::int32_t L, const basis::Orbit& a,
                 const basis::Orbit& b, double y) {
  const int la = a.l, j2a = a.j2.twice, lb = b.l, j2b = b.j2.twice;
  if (la == 0 || lb == 0) return 0.0;
  const double v = 1.0 / std::sqrt(4.0 * std::numbers::pi) *
                   MinusOnePow((2 * L + j2b + 1) / 2) *
                   std::sqrt((2 * la + 1) * (j2a + 1) * (j2b + 1) *
                             (2 * L + 1) * (2 * J + 1)) *
                   SixJ(2 * la, j2a, 1, j2b, 2 * lb, 2 * J) *
                   (-std::sqrt(lb + 1.0) * std::sqrt(2 * (lb + 1) + 1) *
                        SixJ(2 * L, 2, 2 * J, 2 * lb, 2 * la, 2 * (lb + 1)) *
                        ThreeJ000(2 * la, 2 * L, 2 * (lb + 1)) *
                        Jdmho(a.n, la, b.n, lb, L, y) +
                    std::sqrt(lb) * std::sqrt(2 * (lb - 1) + 1) *
                        SixJ(2 * L, 2, 2 * J, 2 * lb, 2 * la, 2 * (lb - 1)) *
                        ThreeJ000(2 * la, 2 * L, 2 * (lb - 1)) *
                        Jdpho(a.n, la, b.n, lb, L, y));
  return ScalarStorage(v, J, ParityOfL(L + 1), j2a);
}

double FamilyElement(Family family, std::int32_t J, const basis::Orbit& a,
                     const basis::Orbit& b, double y) {
  switch (family) {
    case Family::kM:
      return MElement(J, a, b, y);
    case Family::kSigma:
      return MsElement(J, J, a, b, y);
    case Family::kSigmap: {
      double v = -std::sqrt(J / (2 * J + 1.)) * MsElement(J, J + 1, a, b, y);
      if (J > 0) {
        v += std::sqrt((J + 1.) / (2 * J + 1.)) * MsElement(J, J - 1, a, b, y);
      }
      return v;
    }
    case Family::kSigmapp: {
      double v =
          std::sqrt((J + 1.0) / (2 * J + 1.)) * MsElement(J, J + 1, a, b, y);
      if (J > 0) {
        v += std::sqrt(J / (2 * J + 1.)) * MsElement(J, J - 1, a, b, y);
      }
      return v;
    }
    case Family::kDelta:
      return MgElement(J, J, a, b, y);
    case Family::kDeltap: {
      double v = -std::sqrt(J / (2 * J + 1.)) * MgElement(J, J + 1, a, b, y);
      if (J > 0) {
        v += std::sqrt((J + 1.) / (2 * J + 1.)) * MgElement(J, J - 1, a, b, y);
      }
      return v;
    }
    case Family::kPhip:
      return PhipElement(J, a, b, y);
    case Family::kPhipp:
      return PhippElement(J, a, b, y);
    case Family::kPhitp:
      // Operator::operator+ in the fork: Phip + 0.5 * Ms(J, J), elementwise.
      return PhipElement(J, a, b, y) + 0.5 * MsElement(J, J, a, b, y);
  }
  return 0.0;
}

bool InOneBodyChannel(std::int32_t J, Parity parity, const basis::Orbit& a,
                      const basis::Orbit& b) {
  if (a.tz2 != b.tz2) return false;
  if ((a.l + b.l + static_cast<int>(parity)) % 2 != 0) return false;
  if (std::abs(a.l - b.l) > J + 1) return false;
  return std::abs(a.j2.twice - b.j2.twice) <= 2 * J;
}

OneBodyTable BuildTable(const basis::SingleParticleBasis& basis, Family family,
                        std::int32_t J, Isospin isospin, double q_MeV,
                        double prefactor) {
  IMSRG_CHECK(J >= 0);
  OneBodyTable t;
  t.num_orbits = basis.num_orbits();
  t.rank_J = J;
  t.parity = FamilyParity(family, J);
  t.hermiticity = FamilyHermiticity(family);
  const std::size_t n = static_cast<std::size_t>(t.num_orbits);
  t.values.assign(n * n, 0.0);
  const double y = OscillatorY(q_MeV, basis.hbar_omega());
  for (std::size_t a = 0; a < n; ++a) {
    const basis::Orbit& oa = basis.orbits()[a];
    const double iso = IsoFactor(isospin, oa);
    for (std::size_t b = 0; b < n; ++b) {
      const basis::Orbit& ob = basis.orbits()[b];
      if (!InOneBodyChannel(J, t.parity, oa, ob)) continue;
      t.values[a * n + b] =
          prefactor * (FamilyElement(family, J, oa, ob, y) * iso);
    }
  }
  return t;
}

double SqrtFourPi() { return std::sqrt(4 * std::numbers::pi); }

// sqrt(4 pi) as for FPhipp; owner-confirmed, design/50 section 4.
double PhitpShorthandPrefactor() { return SqrtFourPi(); }

std::optional<Shorthand> ParseShorthand(std::string_view name) {
  struct Entry {
    std::string_view prefix;
    Family family;
    int l_first;  // allowed L = l_first, l_first + 2, ... <= l_last
    int l_last;
  };
  // Longest prefixes first (FSigmapp before FSigmap, FPhipp before FPhitp).
  static constexpr Entry kEntries[] = {
      {"FSigmapp", Family::kSigmapp, 1, 5}, {"FSigmap", Family::kSigmap, 1, 5},
      {"FPhipp", Family::kPhipp, 0, 4},     {"FPhitp", Family::kPhitp, 2, 4},
      {"FDelta", Family::kDelta, 1, 5},     {"FM", Family::kM, 0, 4}};
  for (const Entry& e : kEntries) {
    if (!name.starts_with(e.prefix)) continue;
    std::string_view rest = name.substr(e.prefix.size());
    // <L><p|n>_<q>
    if (rest.size() < 4 || rest[0] < '0' || rest[0] > '9') return std::nullopt;
    const int L = rest[0] - '0';
    if (L < e.l_first || L > e.l_last || (L - e.l_first) % 2 != 0) {
      return std::nullopt;
    }
    Shorthand s;
    s.family = e.family;
    s.L = L;
    if (rest[1] == 'p') {
      s.isospin = Isospin::kProton;
    } else if (rest[1] == 'n') {
      s.isospin = Isospin::kNeutron;
    } else {
      return std::nullopt;
    }
    if (rest[2] != '_') return std::nullopt;
    const std::string_view qs = rest.substr(3);
    const auto [ptr, ec] =
        std::from_chars(qs.data(), qs.data() + qs.size(), s.q_MeV);
    if (ec != std::errc() || ptr != qs.data() + qs.size() || s.q_MeV < 0.0) {
      return std::nullopt;
    }
    switch (e.family) {
      case Family::kM:
      case Family::kPhipp:
        s.prefactor = SqrtFourPi();
        break;
      case Family::kPhitp:
        s.prefactor = PhitpShorthandPrefactor();
        break;
      default:
        s.prefactor = 1.0;
    }
    return s;
  }
  return std::nullopt;
}

OneBodyTable BuildShorthandTable(const basis::SingleParticleBasis& basis,
                                 const Shorthand& s) {
  OneBodyTable t =
      BuildTable(basis, s.family, s.L, s.isospin, s.q_MeV, s.prefactor);
  t.hermiticity = Hermiticity::kHermitian;  // all shorthands (owner decision)
  return t;
}

op::Operator BuildScalarShorthand(
    const std::shared_ptr<const basis::ModelSpace>& ms, const Shorthand& s,
    std::int32_t particle_rank) {
  if (s.L != 0) {
    throw InputError(
        "NREFT shorthand with rank L > 0 needs tensor operators (M15)");
  }
  const OneBodyTable t = BuildShorthandTable(ms->basis(), s);
  op::Operator o(
      ms, op::OperatorSpec::Scalar(particle_rank, Hermiticity::kHermitian));
  for (std::int32_t a = 0; a < t.num_orbits; ++a) {
    for (std::int32_t b = 0; b < t.num_orbits; ++b) {
      const double v = t.value(OrbitIndex{a}, OrbitIndex{b});
      if (v != 0.0) o.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
    }
  }
  o.SyncToDevice();
  return o;
}

op::Operator BuildShorthandOperator(
    const std::shared_ptr<const basis::ModelSpace>& ms, const Shorthand& s,
    std::int32_t particle_rank) {
  if (s.L == 0) return BuildScalarShorthand(ms, s, particle_rank);
  const OneBodyTable t = BuildShorthandTable(ms->basis(), s);
  op::OperatorSpec spec =
      op::OperatorSpec::Scalar(particle_rank, Hermiticity::kHermitian);
  spec.rank_J = s.L;
  spec.parity = Parity::kEven;
  spec.delta_Tz = 0;
  op::Operator o(ms, spec);
  for (std::int32_t a = 0; a < t.num_orbits; ++a) {
    for (std::int32_t b = 0; b < t.num_orbits; ++b) {
      const double v = t.value(OrbitIndex{a}, OrbitIndex{b});
      if (v != 0.0) o.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
    }
  }
  o.SyncToDevice();
  return o;
}

}  // namespace imsrg::opslib::nreft
