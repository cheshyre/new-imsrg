#include "imsrg/operator/normal_ordering.hpp"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"

namespace imsrg::op {

Operator normal_order(const Operator& op, std::span<const double> occupations,
                      NormalOrderDirection direction) {
  IMSRG_CHECK(!op.empty(), "normal_order: empty operator");
  const basis::ModelSpace& ms = op.model_space();
  const basis::SingleParticleBasis& sp = ms.basis();
  const std::int32_t norb = sp.num_orbits();
  IMSRG_CHECK(static_cast<std::int32_t>(occupations.size()) == norb,
              "normal_order: one occupation per orbit required");
  if (!op.spec().is_scalar()) {
    throw NotImplementedError("normal_order: only scalar operators are supported");
  }

  const double s = direction == NormalOrderDirection::ToReference ? 1.0 : -1.0;
  Operator result = op;
  // Host loops below: read the host mirrors of the copy, write the one-body part, sync back.
  result.sync_to_host();

  std::vector<std::int32_t> occupied;  // ascending, n >= kOccupationCut (imsrg++ "holes")
  for (std::int32_t k = 0; k < norb; ++k) {
    if (occupations[static_cast<std::size_t>(k)] >= basis::kOccupationCut) {
      occupied.push_back(k);
    }
  }
  const auto occ = [&](std::int32_t k) { return occupations[static_cast<std::size_t>(k)]; };
  const bool has_two_body = result.two_body().allocated();
  const OneBody& f0 = result.one_body();  // the ORIGINAL one-body part (not modified until the end)
  const TwoBody& v = result.two_body();

  // Zero-body part.
  double e0 = result.zero_body();
  for (const std::int32_t k : occupied) {
    const OrbitIndex ik{k};
    const basis::Orbit& ok = sp.orbit(ik);
    e0 += s * degeneracy(ok.j2) * occ(k) * f0.get(ik, ik);
    if (!has_two_body) {
      continue;
    }
    for (const std::int32_t l : occupied) {
      if (l < k) {
        continue;
      }
      const OrbitIndex il{l};
      const basis::Orbit& ol = sp.orbit(il);
      const std::int32_t jmin = std::abs(ok.j2.twice - ol.j2.twice) / 2;
      const std::int32_t jmax = (ok.j2.twice + ol.j2.twice) / 2;
      for (std::int32_t J = jmin; J <= jmax; ++J) {
        e0 += (2 * J + 1) * occ(k) * occ(l) * v.get_normalized(ik, il, ik, il, TwoJ{2 * J});
      }
    }
  }

  // One-body part from the two-body part. The values are computed from the original parts and
  // collected first, then written.
  struct Entry {
    std::int32_t a;
    std::int32_t b;
    double value;
  };
  std::vector<Entry> updates;
  if (has_two_body) {
    const std::int32_t h_sign = hermiticity_sign(op.hermiticity());
    for (std::int32_t a = 0; a < norb; ++a) {
      const OrbitIndex ia{a};
      const basis::Orbit& oa = sp.orbit(ia);
      for (const OrbitIndex ib : sp.same_ljtz(ia)) {
        const std::int32_t b = ib.value();
        if (h_sign != 0 && b < a) {
          continue;  // (anti-)Hermitian: compute b >= a, mirror below
        }
        double sum = 0.0;
        for (const std::int32_t h : occupied) {
          const OrbitIndex ih{h};
          const basis::Orbit& oh = sp.orbit(ih);
          const std::int32_t jmin = std::abs(oa.j2.twice - oh.j2.twice) / 2;
          const std::int32_t jmax = (oa.j2.twice + oh.j2.twice) / 2;
          double inner = 0.0;
          for (std::int32_t J = jmin; J <= jmax; ++J) {
            inner += (2 * J + 1) * v.get_unnormalized(ia, ih, ib, ih, TwoJ{2 * J});
          }
          sum += s * occ(h) * inner / static_cast<double>(degeneracy(oa.j2));
        }
        const double fab = f0.get(ia, ib) + sum;
        updates.push_back({a, b, fab});
        // imsrg++ also overwrites O_aa = h O_aa for anti-Hermitian operators, which flips the
        // sign of the diagonal at every pass; the diagonal of a valid anti-Hermitian operator is
        // zero, so that quirk is not reproduced here.
        if (a != b && h_sign != 0) {
          updates.push_back({b, a, h_sign * fab});
        }
      }
    }
  }
  for (const Entry& e : updates) {
    result.one_body().set_element(OrbitIndex{e.a}, OrbitIndex{e.b}, e.value);
  }
  result.zero_body() = e0;
  result.sync_to_device();
  return result;
}

Operator normal_order(const Operator& op) {
  IMSRG_CHECK(!op.empty(), "normal_order: empty operator");
  return normal_order(op, op.model_space().reference().occupations(),
                      NormalOrderDirection::ToReference);
}

Operator undo_normal_order(const Operator& op) {
  IMSRG_CHECK(!op.empty(), "undo_normal_order: empty operator");
  return normal_order(op, op.model_space().reference().occupations(),
                      NormalOrderDirection::ToVacuum);
}

}  // namespace imsrg::op
