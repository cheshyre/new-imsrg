// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/commutator_ref/tensor_mscheme_support.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "imsrg/operator/two_body.h"

namespace imsrg::testing {
namespace {

namespace msc = imsrg::commutator_ref::mscheme;

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}
  double Uniform() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
  }

 private:
  std::uint64_t state_;
};

int PhaseOf(int n) { return n % 2 == 0 ? 1 : -1; }

}  // namespace

op::OperatorSpec SpecOf(const TensorKind& k) {
  op::OperatorSpec s;
  s.particle_rank = 2;
  s.rank_J = k.rank_J;
  s.parity = k.parity;
  s.delta_Tz = k.delta_Tz;
  s.hermiticity = k.hermiticity;
  return s;
}

op::Operator RandomTensor(std::shared_ptr<const basis::ModelSpace> ms,
                          const op::OperatorSpec& spec, std::uint64_t seed) {
  op::Operator o(ms, spec);
  Rng rng(seed);
  const int h = op::HermiticitySign(spec.hermiticity);
  const auto& basis = ms->basis();
  const int n = ms->num_orbits();
  for (int a = 0; a < n; ++a) {
    for (int b = a; b < n; ++b) {
      const auto& oa = basis.orbit(OrbitIndex{a});
      const auto& ob = basis.orbit(OrbitIndex{b});
      if (!op::ConnectsOrbits(spec, oa, ob)) continue;
      if (a == b && h < 0) continue;
      const double v = rng.Uniform();
      o.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
      if (a != b) {
        o.one_body().set_element(
            OrbitIndex{b}, OrbitIndex{a},
            h * PhaseOf((oa.j2.twice - ob.j2.twice) / 2) * v);
      }
    }
  }
  auto& m = o.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto info = m.block_info(k);
    auto blk = m.host_block(k);
    const bool diagonal = info.row_block == info.col_block;
    for (std::int64_t i = 0; i < info.rows; ++i) {
      for (std::int64_t j = 0; j < info.cols; ++j) {
        if (diagonal && j < i) {
          blk(i, j) = h * blk(j, i);
        } else if (diagonal && j == i && h < 0) {
          blk(i, j) = 0.0;
        } else {
          blk(i, j) = rng.Uniform();
        }
      }
    }
  }
  o.SyncToDevice();
  return o;
}

op::Operator MSchemeTensorCommutator(const op::Operator& X,
                                     const op::Operator& Y,
                                     const op::OperatorSpec& z_spec,
                                     TwoM two_mu, const msc::TensorGauge& gauge,
                                     unsigned n_c_mask, unsigned parts_mask) {
  const auto ms = X.model_space_ptr();
  const msc::MBasis mb(*ms);
  const msc::MOperator ax = msc::Expand(X, mb);
  const msc::MOperator ay = msc::ExpandTensor(Y, mb, two_mu, gauge);
  msc::TensorShift shift;
  shift.two_mu = two_mu.twice;
  shift.parity = static_cast<int>(Y.spec().parity);
  shift.two_dtz = 2 * Y.spec().delta_Tz;
  msc::MOperator z(mb.size(), 2);
  for (int n_c = 1; n_c <= 2; ++n_c) {
    if (((n_c_mask >> n_c) & 1U) == 0U) continue;
    for (int n_a = 1; n_a <= 2; ++n_a) {
      for (int n_b = 1; n_b <= 2; ++n_b) {
        if (((parts_mask >> (2 * (n_a - 1) + (n_b - 1))) & 1U) == 0U) continue;
        z.Axpy(1.0, msc::CommutatorPart(ax, n_a, ay, n_b, n_c, mb, shift));
      }
    }
  }
  return msc::ContractTensor(z, ms, z_spec, two_mu, gauge);
}

}  // namespace imsrg::testing
