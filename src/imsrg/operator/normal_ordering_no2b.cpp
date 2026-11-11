#include <cmath>
#include <cstdint>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/detail/part_kernels.hpp"
#include "imsrg/operator/normal_ordering.hpp"

namespace imsrg::op {

Operator normal_order(const Operator& op, const No2bThreeBody& v3,
                      std::span<const double> occupations, NormalOrderDirection direction) {
  IMSRG_CHECK(!op.empty(), "normal_order(NO2B): empty operator");
  IMSRG_CHECK(op.hermiticity() == Hermiticity::Hermitian,
              "normal_order(NO2B): the operator must be Hermitian");
  const basis::ModelSpace& ms = op.model_space();
  const basis::SingleParticleBasis& sp = ms.basis();
  IMSRG_CHECK(static_cast<std::int32_t>(occupations.size()) == sp.num_orbits(),
              "normal_order(NO2B): one occupation per orbit required");

  const double s = direction == NormalOrderDirection::ToReference ? 1.0 : -1.0;
  std::vector<std::int32_t> occupied;
  for (std::int32_t k = 0; k < sp.num_orbits(); ++k) {
    if (occupations[static_cast<std::size_t>(k)] >= basis::kOccupationCut) {
      occupied.push_back(k);
    }
  }
  const std::int32_t e3max = v3.truncation().e3max;

  Operator gamma(op.model_space_ptr(), OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const basis::TwoBodyChannels& tbc = gamma.two_body().channels();
  for (std::int32_t c = 0; c < tbc.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const std::int32_t J = tbc.channel(ch).J();
    const auto kets = tbc.channel_kets(ch);
    const auto nk = static_cast<std::int32_t>(kets.size());
    for (std::int32_t i = 0; i < nk; ++i) {
      const basis::Ket& bra = tbc.ket(kets[static_cast<std::size_t>(i)]);
      const basis::Orbit& op_ = sp.orbit(bra.p);
      const basis::Orbit& oq = sp.orbit(bra.q);
      for (std::int32_t k = i; k < nk; ++k) {
        const basis::Ket& ket = tbc.ket(kets[static_cast<std::size_t>(k)]);
        const basis::Orbit& orr = sp.orbit(ket.p);
        const basis::Orbit& ot = sp.orbit(ket.q);
        double g = 0.0;
        for (const std::int32_t a : occupied) {
          const basis::Orbit& oa = sp.orbit(OrbitIndex{a});
          if (op_.e() + oq.e() + oa.e() > e3max || orr.e() + ot.e() + oa.e() > e3max) {
            continue;
          }
          g +=
              s * occupations[static_cast<std::size_t>(a)] * v3.get_pn(op_, oq, oa, orr, ot, oa, J);
        }
        g /= (2 * J + 1) *
             std::sqrt((1.0 + (bra.p == bra.q ? 1.0 : 0.0)) * (1.0 + (ket.p == ket.q ? 1.0 : 0.0)));
        gamma.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{k}, g);
        gamma.two_body().set(ch, ch, LocalKetIndex{k}, LocalKetIndex{i}, g);
      }
    }
  }
  gamma.two_body().sync_to_device();

  Operator result = normal_order(gamma, occupations, direction);
  result.zero_body() *= 1.0 / 3.0;
  detail::scale(0.5, result.one_body().buffer());
  axpy(1.0, normal_order(op, occupations, direction), result);
  result.sync_to_host();
  return result;
}

Operator normal_order(const Operator& op, const No2bThreeBody& v3) {
  IMSRG_CHECK(!op.empty(), "normal_order(NO2B): empty operator");
  return normal_order(op, v3, op.model_space().reference().occupations(),
                      NormalOrderDirection::ToReference);
}

}  // namespace imsrg::op
