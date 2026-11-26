// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/truncate.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"

namespace imsrg::op {

Operator Truncate(const Operator& O,
                  std::shared_ptr<const basis::ModelSpace> target) {
  IMSRG_CHECK(!O.empty(), "Truncate: empty operator");
  IMSRG_CHECK(target != nullptr, "Truncate: null target model space");
  const OperatorSpec& spec = O.spec();
  if (spec.particle_rank > 2) {
    throw NotImplementedError(
        "Truncate: three-body parts are not supported yet");
  }
  if (spec.particle_rank == 2 && !spec.is_scalar()) {
    throw NotImplementedError(
        "Truncate: two-body parts of tensor operators are not supported yet");
  }
  const basis::SingleParticleBasis& src = O.model_space().basis();
  const basis::SingleParticleBasis& dst = target->basis();
  if (src.hbar_omega() != dst.hbar_omega()) {
    throw InputError("Truncate: hw differs between the model spaces (" +
                     std::to_string(src.hbar_omega()) + " vs " +
                     std::to_string(dst.hbar_omega()) + " MeV)");
  }
  const std::int32_t nt = dst.num_orbits();
  std::vector<OrbitIndex> map;
  map.reserve(static_cast<std::size_t>(nt));
  for (std::int32_t a = 0; a < nt; ++a) {
    const basis::Orbit& o = dst.orbit(OrbitIndex(a));
    const auto f = src.find(o);
    if (!f) {
      throw InputError("Truncate: orbit (n=" + std::to_string(o.n) +
                       ", l=" + std::to_string(o.l) +
                       ", 2j=" + std::to_string(o.j2.twice) +
                       ", 2tz=" + std::to_string(o.tz2.twice) +
                       ") of the target is not in the source basis");
    }
    map.push_back(*f);
  }
  auto m = [&map](OrbitIndex a) {
    return map[static_cast<std::size_t>(a.value())];
  };
  O.SyncToHost();
  Operator out(target, spec);
  out.set_hermiticity(O.hermiticity());
  out.zero_body() = O.zero_body();
  for (std::int32_t a = 0; a < nt; ++a) {
    for (std::int32_t b = 0; b < nt; ++b) {
      out.one_body().set_element(
          OrbitIndex(a), OrbitIndex(b),
          O.one_body().get(m(OrbitIndex(a)), m(OrbitIndex(b))));
    }
  }
  if (spec.particle_rank == 2) {
    const basis::TwoBodyChannels& tb = target->two_body();
    const TwoBody& v = O.two_body();
    for (std::int32_t c = 0; c < tb.num_channels(); ++c) {
      const ChannelIndex ch(c);
      const basis::TwoBodyChannel& chan = tb.channel(ch);
      for (std::int32_t i = 0; i < chan.num_kets; ++i) {
        const basis::Ket& bra = tb.ket(tb.global_ket(ch, LocalKetIndex(i)));
        for (std::int32_t j = 0; j < chan.num_kets; ++j) {
          const basis::Ket& ket = tb.ket(tb.global_ket(ch, LocalKetIndex(j)));
          const double x = v.get_normalized(m(bra.p), m(bra.q), m(ket.p),
                                            m(ket.q), chan.two_j);
          out.two_body().set(ch, ch, LocalKetIndex(i), LocalKetIndex(j), x);
        }
      }
    }
  }
  out.SyncToDevice();
  return out;
}

}  // namespace imsrg::op
