// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IMSRG generators (M8 task T1); see generator.h and design/13 section 7.
#include "imsrg/flow/generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/flow/detail/generator_three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::flow {

namespace {

bool IsShellModel(GeneratorKind k) {
  return k == GeneratorKind::kShellModel ||
         k == GeneratorKind::kShellModelAtan ||
         k == GeneratorKind::kShellModelWegner ||
         k == GeneratorKind::kShellModelImaginaryTime;
}

// Maps a kind to the single-reference kind that has the same function f.
GeneratorKind FunctionKind(GeneratorKind k) {
  switch (k) {
    case GeneratorKind::kWhite:
    case GeneratorKind::kShellModel:
      return GeneratorKind::kWhite;
    case GeneratorKind::kAtan:
    case GeneratorKind::kShellModelAtan:
      return GeneratorKind::kAtan;
    case GeneratorKind::kImaginaryTime:
    case GeneratorKind::kShellModelImaginaryTime:
      return GeneratorKind::kImaginaryTime;
    case GeneratorKind::kWegner:
    case GeneratorKind::kShellModelWegner:
      return GeneratorKind::kWegner;
    case GeneratorKind::kQtransferAtan:
      return GeneratorKind::kQtransferAtan;
    case GeneratorKind::kShellModelAtanNpnh:
      throw NotImplementedError(
          "generator shell-model-atan-npnh is not implemented");
    case GeneratorKind::kOnePa:
      throw NotImplementedError("generator 1PA is not implemented");
    case GeneratorKind::kHartreeFock:
      throw NotImplementedError(
          "the hartree-fock generator has no generator function");
  }
  throw InvariantError("unknown GeneratorKind");
}

bool SameLjtz(const basis::Orbit& a, const basis::Orbit& b) {
  return a.l == b.l && a.j2.twice == b.j2.twice && a.tz2.twice == b.tz2.twice;
}

// Fills the Epstein-Nesbet monopole table (row-major norb x norb, symmetric)
// from the unnormalized two-body matrix elements <ab|V|ab> of H.
void FillMonopole(const op::Operator& Hh, const basis::SingleParticleBasis& sp,
                  std::int32_t norb, std::vector<double>& monopole) {
  const auto n = static_cast<std::size_t>(norb);
  monopole.assign(n * n, 0.0);
  for (std::int32_t a = 0; a < norb; ++a) {
    for (std::int32_t b = a; b < norb; ++b) {
      const OrbitIndex oa{a}, ob{b};
      const int ja = sp.orbit(oa).j2.twice, jb = sp.orbit(ob).j2.twice;
      double sum = 0.0;
      for (int tj = std::abs(ja - jb); tj <= ja + jb; tj += 2) {
        if (a == b && (tj / 2) % 2 == 1) continue;
        sum +=
            (tj + 1) * Hh.two_body().get_unnormalized(oa, ob, oa, ob, TwoJ{tj});
      }
      const double v =
          sum / (static_cast<double>(ja + 1) * static_cast<double>(jb + 1));
      monopole[static_cast<std::size_t>(a) + static_cast<std::size_t>(b) * n] =
          v;
      monopole[static_cast<std::size_t>(b) + static_cast<std::size_t>(a) * n] =
          v;
    }
  }
}

}  // namespace

double GeneratorFunction(GeneratorKind kind, double h_od, double denominator,
                         int qtransfer_n) {
  switch (FunctionKind(kind)) {
    case GeneratorKind::kWhite:
      return h_od / denominator;
    case GeneratorKind::kAtan:
      return 0.5 * std::atan(2.0 * h_od / denominator);
    case GeneratorKind::kImaginaryTime:
      return h_od * (denominator < 0.0 ? -1.0 : 1.0);
    case GeneratorKind::kWegner:
      return h_od * denominator;
    case GeneratorKind::kQtransferAtan: {
      const double w = std::pow(std::abs(denominator) * constants::kMNucleon /
                                    (constants::kHbarc * constants::kHbarc),
                                0.5 * qtransfer_n);
      return w * 0.5 * std::atan(2.0 * h_od / denominator);
    }
    default:
      break;
  }
  throw InvariantError("unreachable generator kind");
}

double FloorDenominator(double delta, double cutoff) noexcept {
  if (std::abs(delta) < cutoff) return delta < 0.0 ? -cutoff : cutoff;
  return delta;
}

Denominators::Denominators(const op::Operator& H, const GeneratorOptions& opts)
    : ms_(H.model_space_ptr()), opts_(opts) {
  IMSRG_CHECK(!H.empty(), "Denominators: empty operator");
  // cutoff <= 0 would disable the floor and divide by ~0 (inf/NaN).
  if (!(std::isfinite(opts.denominator_cutoff) &&
        opts.denominator_cutoff > 0.0)) {
    throw InputError("denominator_cutoff must be positive and finite");
  }
  H.SyncToHost();
  const op::Operator& Hh = H;
  const basis::SingleParticleBasis& sp = ms_->basis();
  norb_ = sp.num_orbits();
  const auto n = static_cast<std::size_t>(norb_);
  f_diag_.resize(n);
  isospin_partner_.resize(n);
  for (std::int32_t a = 0; a < norb_; ++a) {
    f_diag_[static_cast<std::size_t>(a)] =
        Hh.one_body().get(OrbitIndex{a}, OrbitIndex{a});
    basis::Orbit o = sp.orbit(OrbitIndex{a});
    o.tz2 = TwoTz{-o.tz2.twice};
    isospin_partner_[static_cast<std::size_t>(a)] =
        sp.find(o).value_or(OrbitIndex{a});
  }
  if (opts_.partitioning == Partitioning::kEpsteinNesbet) {
    FillMonopole(Hh, sp, norb_, monopole_);
  }
}

double Denominators::three_body(OrbitIndex a, OrbitIndex b, OrbitIndex c,
                                OrbitIndex i, OrbitIndex j,
                                OrbitIndex k) const {
  auto f = [&](OrbitIndex x) { return f_diag_[x.ToSize()]; };
  // Same sign-keeping floor as the one- and two-body denominators (D21,
  // M10-Q60); imsrg++ has none here.
  return FloorDenominator(f(a) + f(b) + f(c) - f(i) - f(j) - f(k),
                          opts_.denominator_cutoff);
}

double Denominators::monopole(OrbitIndex a, OrbitIndex b) const {
  if (monopole_.empty()) return 0.0;
  return monopole_[a.ToSize() + b.ToSize() * static_cast<std::size_t>(norb_)];
}

double Denominators::one_body(OrbitIndex i, OrbitIndex j) const {
  const auto occ = ms_->reference().occupations();
  double d = f_diag_[i.ToSize()] - f_diag_[j.ToSize()];
  if (opts_.partitioning == Partitioning::kEpsteinNesbet) {
    d += (occ[i.ToSize()] - occ[j.ToSize()]) * monopole(i, j);
  } else if (opts_.partitioning == Partitioning::kMpIsospin) {
    d = 0.5 * (d + f_diag_[isospin_partner_[i.ToSize()].ToSize()] -
               f_diag_[isospin_partner_[j.ToSize()].ToSize()]);
  }
  if (opts_.denominator_delta_all || (opts_.denominator_delta_orbit &&
                                      (*opts_.denominator_delta_orbit == i ||
                                       *opts_.denominator_delta_orbit == j))) {
    d += opts_.denominator_delta;
  }
  return FloorDenominator(d, opts_.denominator_cutoff);
}

double Denominators::two_body(ChannelIndex ch, LocalKetIndex ibra,
                              LocalKetIndex iket) const {
  const basis::TwoBodyChannels& tb = ms_->two_body();
  const basis::Ket& bra = tb.ket(tb.global_ket(ch, ibra));
  const basis::Ket& ket = tb.ket(tb.global_ket(ch, iket));
  const OrbitIndex i = bra.p, j = bra.q, k = ket.p, l = ket.q;
  const auto occ = ms_->reference().occupations();
  auto f = [&](OrbitIndex x) { return f_diag_[x.ToSize()]; };
  auto fp = [&](OrbitIndex x) {
    return f_diag_[isospin_partner_[x.ToSize()].ToSize()];
  };
  double d = f(i) + f(j) - f(k) - f(l);
  if (opts_.partitioning == Partitioning::kMpIsospin) {
    d = 0.5 * (d + fp(i) + fp(j) - fp(k) - fp(l));
  }
  if (opts_.denominator_delta_all) d += opts_.denominator_delta;
  if (opts_.partitioning == Partitioning::kEpsteinNesbet) {
    const double ni = occ[i.ToSize()], nj = occ[j.ToSize()];
    const double nk = occ[k.ToSize()], nl = occ[l.ToSize()];
    d += (1.0 - ni - nj) * monopole(i, j) - (1.0 - nk - nl) * monopole(k, l) +
         (ni - nk) * monopole(i, k) + (ni - nl) * monopole(i, l) +
         (nj - nk) * monopole(j, k) + (nj - nl) * monopole(j, l);
  }
  return FloorDenominator(d, opts_.denominator_cutoff);
}

op::OperatorSpec GeneratorSpec(const op::OperatorSpec& h, GeneratorKind kind) {
  return op::OperatorSpec::Scalar(
      kind == GeneratorKind::kHartreeFock ? 1 : h.particle_rank,
      Hermiticity::kAntiHermitian);
}

namespace {

// Checks that H is a scalar, number-conserving rank-2 or rank-3 operator for
// which generator `opts.kind` is implemented; returns whether H is rank 3.
bool CheckGeneratorInputs(const op::Operator& H, const GeneratorOptions& opts) {
  IMSRG_CHECK(!H.empty(), "compute_generator: empty H");
  IMSRG_CHECK(H.spec().is_scalar() && H.spec().is_number_conserving(),
              "compute_generator: H must be scalar and number conserving");
  IMSRG_CHECK(H.spec().particle_rank == 2 || H.spec().particle_rank == 3,
              "compute_generator: H must have particle_rank 2 or 3");
  const bool rank3 = H.spec().particle_rank == 3;
  IMSRG_CHECK(!rank3 || H.three_body().has_value(),
              "compute_generator: rank-3 H without a three-body part");
  if (opts.kind == GeneratorKind::kShellModelAtanNpnh) {
    throw NotImplementedError(
        "generator shell-model-atan-npnh is not implemented");
  }
  if (opts.kind == GeneratorKind::kOnePa) {
    throw NotImplementedError("generator 1PA is not implemented");
  }

  if (rank3 &&
      (IsShellModel(opts.kind) || opts.kind == GeneratorKind::kHartreeFock)) {
    throw NotImplementedError(
        "shell-model and hartree-fock generators with a rank-3 H are not "
        "implemented");
  }
  return rank3;
}

// Fills the Hartree-Fock generator: eta_ij = f_ij / (f_ii - f_jj) between
// orbits of equal (l, j, tz).
void FillHartreeFockEta(const op::Operator& Hh, const Denominators& den,
                        const basis::SingleParticleBasis& sp,
                        op::Operator& eta) {
  const std::int32_t norb = sp.num_orbits();
  for (std::int32_t i = 0; i < norb; ++i) {
    for (std::int32_t j = 0; j < i; ++j) {
      const OrbitIndex oi{i}, oj{j};
      if (!SameLjtz(sp.orbit(oi), sp.orbit(oj))) continue;
      const double v = Hh.one_body().get(oi, oj) / den.one_body(oi, oj);
      eta.one_body().set_element(oi, oj, v);
      eta.one_body().set_element(oj, oi, -v);
    }
  }
}

// Fills the one-body part of eta for the single-reference or shell-model
// partitions.
void FillOneBodyEta(const op::Operator& Hh, const Denominators& den,
                    const basis::SingleParticleBasis& sp,
                    const basis::Partition& part, GeneratorKind kind,
                    bool shell, int nq, op::Operator& eta) {
  auto one_body_pair = [&](OrbitIndex a, OrbitIndex i) {
    if (a == i || !SameLjtz(sp.orbit(a), sp.orbit(i))) return;
    const double d = den.one_body(i, a);
    eta.one_body().set_element(
        i, a, GeneratorFunction(kind, Hh.one_body().get(i, a), d, nq));
    eta.one_body().set_element(
        a, i, GeneratorFunction(kind, Hh.one_body().get(a, i), -d, nq));
  };
  auto for_ai = [&](std::span<const OrbitIndex> as,
                    std::span<const OrbitIndex> is) {
    for (OrbitIndex a : as)
      for (OrbitIndex i : is) one_body_pair(a, i);
  };
  for_ai(part.core(), part.valence());
  for_ai(part.core(), part.qspace());
  if (shell) {
    for_ai(part.valence(), part.valence());
    for_ai(part.valence(), part.qspace());
  }
}

// Fills the two-body part of eta channel by channel from the ket subsets.
void FillTwoBodyEta(const op::Operator& Hh, const Denominators& den,
                    const basis::ModelSpace& ms, GeneratorKind kind, bool shell,
                    int nq, op::Operator& eta) {
  const basis::TwoBodyKetSubsets& sub = ms.two_body_subsets();
  const std::int32_t nch = ms.two_body().num_channels();
  using basis::KetClass;
  for (std::int32_t c = 0; c < nch; ++c) {
    const ChannelIndex ch{c};
    auto block = [&](KetClass kc_ket,
                     std::initializer_list<KetClass> bra_classes) {
      for (LocalKetIndex k : sub.kets(ch, kc_ket)) {
        for (KetClass bc : bra_classes) {
          for (LocalKetIndex b : sub.kets(ch, bc)) {
            const double v =
                GeneratorFunction(kind, Hh.two_body().get(ch, ch, b, k),
                                  den.two_body(ch, b, k), nq);
            eta.two_body().set(ch, ch, b, k, v);
            eta.two_body().set(ch, ch, k, b, -v);
          }
        }
      }
    };
    if (!shell) {
      block(KetClass::kCc, {KetClass::kQq, KetClass::kVv, KetClass::kQv});
    } else {
      block(KetClass::kCc, {KetClass::kVv, KetClass::kQv, KetClass::kQq});
      block(KetClass::kVc, {KetClass::kVv, KetClass::kQv, KetClass::kQq});
      block(KetClass::kVv, {KetClass::kQv, KetClass::kQq});
    }
  }
}

// Fills the three-body part of eta (imsrg++
// ConstructGenerator_SingleRef_3body) with the device kernel over the stored
// elements of the rows of the three-body space. eta's device data must be
// current (zero) on entry; they are written here and the host mirror of the
// three-body part is stale on return.
void FillThreeBodyEta(const op::Operator& Hh, const Denominators& den,
                      const basis::ModelSpace& ms, const basis::Partition& part,
                      const GeneratorOptions& opts, op::Operator& eta) {
  const basis::ThreeBodySpace& ts = *ms.three_body_space();
  // The rows of this process (all rows if the model space is not
  // distributed).
  const basis::RowSet& rows = ms.three_body_rows();
  const std::int32_t norb = ms.basis().num_orbits();
  std::vector<std::int32_t> is_core(static_cast<std::size_t>(norb));
  for (std::int32_t o = 0; o < norb; ++o) {
    is_core[static_cast<std::size_t>(o)] =
        part.orbit_class(OrbitIndex{o}) == basis::OrbitClass::kCore ? 1 : 0;
  }
  detail::FillThreeBodyGenerator(
      *Hh.three_body(), *eta.three_body(), ts, rows, den.one_body_diagonal(),
      is_core, opts.denominator_cutoff, opts.kind, opts.qtransfer_n);
}

}  // namespace

void ComputeGenerator(const op::Operator& H, op::Operator& eta,
                      const GeneratorOptions& opts) {
  const bool rank3 = CheckGeneratorInputs(H, opts);
  const op::OperatorSpec spec = GeneratorSpec(H.spec(), opts.kind);
  if (eta.empty() || !(eta.spec() == spec) || !op::SameModelSpace(eta, H)) {
    eta = op::Operator(H.model_space_ptr(), spec);
  }
  eta.set_zero();
  eta.SyncToHost();

  H.SyncToHost();  // host reads below (no copy of H, flow review item)
  const op::Operator& Hh = H;
  const Denominators den(H, opts);
  const basis::ModelSpace& ms = H.model_space();
  const basis::SingleParticleBasis& sp = ms.basis();
  const basis::Partition& part = ms.partition();
  const int nq = opts.qtransfer_n;

  if (opts.kind == GeneratorKind::kHartreeFock) {
    FillHartreeFockEta(Hh, den, sp, eta);
    eta.SyncToDevice();
    return;
  }

  const bool shell = IsShellModel(opts.kind);
  const GeneratorKind kind = opts.kind;
  FillOneBodyEta(Hh, den, sp, part, kind, shell, nq, eta);
  if (!opts.only_1b_eta) FillTwoBodyEta(Hh, den, ms, kind, shell, nq, eta);
  eta.SyncToDevice();
  if (rank3 && !opts.only_1b_eta && !opts.only_2b_eta) {
    // After the sync: the kernel writes the device data of eta's 3B part.
    FillThreeBodyEta(Hh, den, ms, part, opts, eta);
  }
}

op::Operator ComputeGenerator(const op::Operator& H,
                              const GeneratorOptions& opts) {
  op::Operator eta;
  ComputeGenerator(H, eta, opts);
  return eta;
}

}  // namespace imsrg::flow
