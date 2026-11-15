// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/random_operator.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"

namespace imsrg::op {

namespace {

/// splitmix64 generator (Vigna), the same on every platform.
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : x_(seed) {}

  std::uint64_t Next() {
    std::uint64_t z = (x_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  /// scale * (2u - 1) with u = (z >> 11) * 2^-53 uniform in [0, 1).
  double Uniform(double scale) {
    const double u = static_cast<double>(Next() >> 11) * 0x1.0p-53;
    return scale * (2.0 * u - 1.0);
  }

 private:
  std::uint64_t x_;
};

/// Fills the elements (a, b) of an n x n matrix in the documented order and
/// calls `store(a, b, value)` for each drawn element, with the triangle rules
/// of `h`.
template <class Store>
void DrawMatrix(SplitMix64& rng, std::int32_t n, Hermiticity h, double scale,
                bool (*allowed)(void*, std::int32_t, std::int32_t), void* ctx,
                Store&& store) {
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = (h == Hermiticity::kNonHermitian ? 0 : a); b < n;
         ++b) {
      if (h == Hermiticity::kAntiHermitian && a == b) continue;
      if (allowed != nullptr && !allowed(ctx, a, b)) continue;
      store(a, b, rng.Uniform(scale));
    }
  }
}

}  // namespace

Operator RandomOperator(std::shared_ptr<const basis::ModelSpace> ms,
                        const OperatorSpec& spec,
                        const RandomOperatorOptions& options) {
  Operator op(std::move(ms), spec);
  if (op.three_body()) {
    throw NotImplementedError(
        "random_operator: operators with a three-body part (particle_rank 3) are not "
        "supported");
  }
  SplitMix64 rng(options.seed);
  const Hermiticity h = spec.hermiticity;
  if (h == Hermiticity::kHermitian && options.zero_body_scale != 0.0) {
    op.zero_body() = rng.Uniform(options.zero_body_scale);
  }

  // One-body: only between orbits with equal (l, 2j, tz2).
  const basis::SingleParticleBasis& sp = op.model_space().basis();
  struct Ctx {
    const basis::SingleParticleBasis* sp;
  } ctx{&sp};
  auto same_class = [](void* c, std::int32_t a, std::int32_t b) {
    const auto& basis = *static_cast<Ctx*>(c)->sp;
    const auto& oa = basis.orbit(OrbitIndex{a});
    const auto& ob = basis.orbit(OrbitIndex{b});
    return oa.l == ob.l && oa.j2 == ob.j2 && oa.tz2 == ob.tz2;
  };
  DrawMatrix(rng, sp.num_orbits(), h, options.one_body_scale, same_class, &ctx,
             [&](std::int32_t a, std::int32_t b, double v) {
               op.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
               if (a != b && h != Hermiticity::kNonHermitian) {
                 op.one_body().set_element(
                     OrbitIndex{b}, OrbitIndex{a},
                     static_cast<double>(HermiticitySign(h)) * v);
               }
             });

  // Two-body: every channel block.
  if (op.two_body().allocated()) {
    const basis::TwoBodyChannels& chans = op.two_body().channels();
    for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
      const ChannelIndex ch{c};
      const std::int32_t n = chans.channel(ch).num_kets;
      DrawMatrix(
          rng, n, h, options.two_body_scale, nullptr, nullptr,
          [&](std::int32_t i, std::int32_t j, double v) {
            op.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
            if (i != j && h != Hermiticity::kNonHermitian) {
              op.two_body().set(ch, ch, LocalKetIndex{j}, LocalKetIndex{i},
                                static_cast<double>(HermiticitySign(h)) * v);
            }
          });
    }
  }
  op.SyncToDevice();
  return op;
}

}  // namespace imsrg::op
