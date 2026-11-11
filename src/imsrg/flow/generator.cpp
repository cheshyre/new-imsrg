// IMSRG generators (M8 task T1); see generator.hpp and design/13 section 7.
#include "imsrg/flow/generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <span>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/constants.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::flow {

namespace {

bool is_shell_model(GeneratorKind k) {
  return k == GeneratorKind::ShellModel || k == GeneratorKind::ShellModelAtan ||
         k == GeneratorKind::ShellModelWegner || k == GeneratorKind::ShellModelImaginaryTime;
}

// Maps a kind to the single-reference kind that has the same function f.
GeneratorKind function_kind(GeneratorKind k) {
  switch (k) {
    case GeneratorKind::White:
    case GeneratorKind::ShellModel:
      return GeneratorKind::White;
    case GeneratorKind::Atan:
    case GeneratorKind::ShellModelAtan:
      return GeneratorKind::Atan;
    case GeneratorKind::ImaginaryTime:
    case GeneratorKind::ShellModelImaginaryTime:
      return GeneratorKind::ImaginaryTime;
    case GeneratorKind::Wegner:
    case GeneratorKind::ShellModelWegner:
      return GeneratorKind::Wegner;
    case GeneratorKind::QtransferAtan:
      return GeneratorKind::QtransferAtan;
    case GeneratorKind::ShellModelAtanNpnh:
      throw NotImplementedError("generator shell-model-atan-npnh is not implemented");
    case GeneratorKind::OnePa:
      throw NotImplementedError("generator 1PA is not implemented");
    case GeneratorKind::HartreeFock:
      throw NotImplementedError("the hartree-fock generator has no generator function");
  }
  throw InvariantError("unknown GeneratorKind");
}

bool same_ljtz(const basis::Orbit& a, const basis::Orbit& b) {
  return a.l == b.l && a.j2.twice == b.j2.twice && a.tz2.twice == b.tz2.twice;
}

}  // namespace

double generator_function(GeneratorKind kind, double h_od, double denominator, int qtransfer_n) {
  switch (function_kind(kind)) {
    case GeneratorKind::White:
      return h_od / denominator;
    case GeneratorKind::Atan:
      return 0.5 * std::atan(2.0 * h_od / denominator);
    case GeneratorKind::ImaginaryTime:
      return h_od * (denominator < 0.0 ? -1.0 : 1.0);
    case GeneratorKind::Wegner:
      return h_od * denominator;
    case GeneratorKind::QtransferAtan: {
      const double w = std::pow(std::abs(denominator) * constants::m_nucleon /
                                    (constants::hbarc * constants::hbarc),
                                0.5 * qtransfer_n);
      return w * 0.5 * std::atan(2.0 * h_od / denominator);
    }
    default:
      break;
  }
  throw InvariantError("unreachable generator kind");
}

double floor_denominator(double delta, double cutoff) noexcept {
  if (std::abs(delta) < cutoff) return delta < 0.0 ? -cutoff : cutoff;
  return delta;
}

Denominators::Denominators(const op::Operator& H, const GeneratorOptions& opts)
    : ms_(H.model_space_ptr()), opts_(opts) {
  IMSRG_CHECK(!H.empty(), "Denominators: empty operator");
  op::Operator Hh(H);
  Hh.sync_to_host();
  const basis::SingleParticleBasis& sp = ms_->basis();
  norb_ = sp.num_orbits();
  const auto n = static_cast<std::size_t>(norb_);
  f_diag_.resize(n);
  isospin_partner_.resize(n);
  for (std::int32_t a = 0; a < norb_; ++a) {
    f_diag_[static_cast<std::size_t>(a)] = Hh.one_body().get(OrbitIndex{a}, OrbitIndex{a});
    basis::Orbit o = sp.orbit(OrbitIndex{a});
    o.tz2 = TwoTz{-o.tz2.twice};
    isospin_partner_[static_cast<std::size_t>(a)] = sp.find(o).value_or(OrbitIndex{a});
  }
  if (opts_.partitioning == Partitioning::EpsteinNesbet) {
    monopole_.assign(n * n, 0.0);
    for (std::int32_t a = 0; a < norb_; ++a) {
      for (std::int32_t b = a; b < norb_; ++b) {
        const OrbitIndex oa{a}, ob{b};
        const int ja = sp.orbit(oa).j2.twice, jb = sp.orbit(ob).j2.twice;
        double sum = 0.0;
        for (int tj = std::abs(ja - jb); tj <= ja + jb; tj += 2) {
          if (a == b && (tj / 2) % 2 == 1) continue;
          sum += (tj + 1) * Hh.two_body().get_unnormalized(oa, ob, oa, ob, TwoJ{tj});
        }
        const double v = sum / (static_cast<double>(ja + 1) * static_cast<double>(jb + 1));
        monopole_[static_cast<std::size_t>(a) + static_cast<std::size_t>(b) * n] = v;
        monopole_[static_cast<std::size_t>(b) + static_cast<std::size_t>(a) * n] = v;
      }
    }
  }
}

double Denominators::monopole(OrbitIndex a, OrbitIndex b) const {
  if (monopole_.empty()) return 0.0;
  return monopole_[a.to_size() + b.to_size() * static_cast<std::size_t>(norb_)];
}

double Denominators::one_body(OrbitIndex i, OrbitIndex j) const {
  const auto occ = ms_->reference().occupations();
  double d = f_diag_[i.to_size()] - f_diag_[j.to_size()];
  if (opts_.partitioning == Partitioning::EpsteinNesbet) {
    d += (occ[i.to_size()] - occ[j.to_size()]) * monopole(i, j);
  } else if (opts_.partitioning == Partitioning::MpIsospin) {
    d = 0.5 * (d + f_diag_[isospin_partner_[i.to_size()].to_size()] -
               f_diag_[isospin_partner_[j.to_size()].to_size()]);
  }
  if (opts_.denominator_delta_all ||
      (opts_.denominator_delta_orbit && (*opts_.denominator_delta_orbit == i ||
                                         *opts_.denominator_delta_orbit == j))) {
    d += opts_.denominator_delta;
  }
  return floor_denominator(d, opts_.denominator_cutoff);
}

double Denominators::two_body(ChannelIndex ch, LocalKetIndex ibra, LocalKetIndex iket) const {
  const basis::TwoBodyChannels& tb = ms_->two_body();
  const basis::Ket& bra = tb.ket(tb.global_ket(ch, ibra));
  const basis::Ket& ket = tb.ket(tb.global_ket(ch, iket));
  const OrbitIndex i = bra.p, j = bra.q, k = ket.p, l = ket.q;
  const auto occ = ms_->reference().occupations();
  auto f = [&](OrbitIndex x) { return f_diag_[x.to_size()]; };
  auto fp = [&](OrbitIndex x) { return f_diag_[isospin_partner_[x.to_size()].to_size()]; };
  double d = f(i) + f(j) - f(k) - f(l);
  if (opts_.partitioning == Partitioning::MpIsospin) {
    d = 0.5 * (d + fp(i) + fp(j) - fp(k) - fp(l));
  }
  if (opts_.denominator_delta_all) d += opts_.denominator_delta;
  if (opts_.partitioning == Partitioning::EpsteinNesbet) {
    const double ni = occ[i.to_size()], nj = occ[j.to_size()];
    const double nk = occ[k.to_size()], nl = occ[l.to_size()];
    d += (1.0 - ni - nj) * monopole(i, j) - (1.0 - nk - nl) * monopole(k, l) +
         (ni - nk) * monopole(i, k) + (ni - nl) * monopole(i, l) + (nj - nk) * monopole(j, k) +
         (nj - nl) * monopole(j, l);
  }
  return floor_denominator(d, opts_.denominator_cutoff);
}

op::OperatorSpec generator_spec(const op::OperatorSpec& h, GeneratorKind kind) {
  return op::OperatorSpec::scalar(kind == GeneratorKind::HartreeFock ? 1 : h.particle_rank,
                                  Hermiticity::AntiHermitian);
}

void compute_generator(const op::Operator& H, op::Operator& eta, const GeneratorOptions& opts) {
  IMSRG_CHECK(!H.empty(), "compute_generator: empty H");
  IMSRG_CHECK(H.spec().is_scalar() && H.spec().is_number_conserving(),
              "compute_generator: H must be scalar and number conserving");
  IMSRG_CHECK(H.spec().particle_rank == 2, "compute_generator: H must have particle_rank 2");
  if (opts.kind == GeneratorKind::ShellModelAtanNpnh) {
    throw NotImplementedError("generator shell-model-atan-npnh is not implemented");
  }
  if (opts.kind == GeneratorKind::OnePa) {
    throw NotImplementedError("generator 1PA is not implemented");
  }

  const op::OperatorSpec spec = generator_spec(H.spec(), opts.kind);
  if (eta.empty() || !(eta.spec() == spec) || !op::same_model_space(eta, H)) {
    eta = op::Operator(H.model_space_ptr(), spec);
  }
  eta.set_zero();
  eta.sync_to_host();

  op::Operator Hh(H);
  Hh.sync_to_host();
  const Denominators den(H, opts);
  const basis::ModelSpace& ms = H.model_space();
  const basis::SingleParticleBasis& sp = ms.basis();
  const basis::Partition& part = ms.partition();
  const int nq = opts.qtransfer_n;

  if (opts.kind == GeneratorKind::HartreeFock) {
    const std::int32_t norb = sp.num_orbits();
    for (std::int32_t i = 0; i < norb; ++i) {
      for (std::int32_t j = 0; j < i; ++j) {
        const OrbitIndex oi{i}, oj{j};
        if (!same_ljtz(sp.orbit(oi), sp.orbit(oj))) continue;
        const double v = Hh.one_body().get(oi, oj) / den.one_body(oi, oj);
        eta.one_body().set_element(oi, oj, v);
        eta.one_body().set_element(oj, oi, -v);
      }
    }
    eta.sync_to_device();
    return;
  }

  const bool shell = is_shell_model(opts.kind);
  const GeneratorKind kind = opts.kind;

  // One-body part.
  auto one_body_pair = [&](OrbitIndex a, OrbitIndex i) {
    if (a == i || !same_ljtz(sp.orbit(a), sp.orbit(i))) return;
    const double d = den.one_body(i, a);
    eta.one_body().set_element(i, a, generator_function(kind, Hh.one_body().get(i, a), d, nq));
    eta.one_body().set_element(a, i, generator_function(kind, Hh.one_body().get(a, i), -d, nq));
  };
  auto for_ai = [&](std::span<const OrbitIndex> as, std::span<const OrbitIndex> is) {
    for (OrbitIndex a : as)
      for (OrbitIndex i : is) one_body_pair(a, i);
  };
  for_ai(part.core(), part.valence());
  for_ai(part.core(), part.qspace());
  if (shell) {
    for_ai(part.valence(), part.valence());
    for_ai(part.valence(), part.qspace());
  }

  // Two-body part.
  if (!opts.only_1b_eta) {
    const basis::TwoBodyKetSubsets& sub = ms.two_body_subsets();
    const std::int32_t nch = ms.two_body().num_channels();
    using basis::KetClass;
    for (std::int32_t c = 0; c < nch; ++c) {
      const ChannelIndex ch{c};
      auto block = [&](KetClass kc_ket, std::initializer_list<KetClass> bra_classes) {
        for (LocalKetIndex k : sub.kets(ch, kc_ket)) {
          for (KetClass bc : bra_classes) {
            for (LocalKetIndex b : sub.kets(ch, bc)) {
              const double v = generator_function(
                  kind, Hh.two_body().get(ch, ch, b, k), den.two_body(ch, b, k), nq);
              eta.two_body().set(ch, ch, b, k, v);
              eta.two_body().set(ch, ch, k, b, -v);
            }
          }
        }
      };
      if (!shell) {
        block(KetClass::CC, {KetClass::QQ, KetClass::VV, KetClass::QV});
      } else {
        block(KetClass::CC, {KetClass::VV, KetClass::QV, KetClass::QQ});
        block(KetClass::VC, {KetClass::VV, KetClass::QV, KetClass::QQ});
        block(KetClass::VV, {KetClass::QV, KetClass::QQ});
      }
    }
  }
  eta.sync_to_device();
}

op::Operator compute_generator(const op::Operator& H, const GeneratorOptions& opts) {
  op::Operator eta;
  compute_generator(H, eta, opts);
  return eta;
}

}  // namespace imsrg::flow
