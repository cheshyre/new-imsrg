#include "imsrg/hf/monopole.hpp"

#include <algorithm>
#include <cstdlib>

#include "imsrg/base/assert.hpp"
#include "imsrg/basis/orbit.hpp"

namespace imsrg::hf::detail {

namespace {

/// (Tz, parity) block of a pair: (Tz + 1) * 2 + parity, Tz = (tz2_p + tz2_q) / 2.
int block_of(const basis::Orbit& p, const basis::Orbit& q) {
  const int tz = (p.tz2.twice + q.tz2.twice) / 2;
  return (tz + 1) * 2 + (p.l + q.l) % 2;
}

}  // namespace

std::vector<std::vector<std::int32_t>> one_body_channels(const basis::SingleParticleBasis& basis) {
  std::vector<std::vector<std::int32_t>> out;
  for (std::int32_t i = 0; i < basis.num_orbits(); ++i) {
    std::vector<std::int32_t> ch;
    for (OrbitIndex j : basis.same_ljtz(OrbitIndex{i})) ch.push_back(j.value());
    std::sort(ch.begin(), ch.end());
    if (ch.front() == i) out.push_back(std::move(ch));
  }
  return out;
}

Monopole2B::Monopole2B(const op::Operator& H) {
  const auto& basis = H.model_space().basis();
  norb_ = basis.num_orbits();
  const auto n = static_cast<std::size_t>(norb_);
  blk_.assign(n * n, -1);
  local_.assign(n * n, -1);
  two_j_.resize(n);
  channel_of_.resize(n);
  for (const auto& ch : one_body_channels(basis)) {
    for (std::int32_t i : ch) channel_of_[static_cast<std::size_t>(i)] = ch;
  }

  std::array<std::vector<std::array<std::int32_t, 2>>, 6> kets;
  for (std::int32_t p = 0; p < norb_; ++p) {
    two_j_[static_cast<std::size_t>(p)] = basis.orbit(OrbitIndex{p}).j2.twice;
    for (std::int32_t q = p; q < norb_; ++q) {
      const int blk = block_of(basis.orbit(OrbitIndex{p}), basis.orbit(OrbitIndex{q}));
      const auto key = static_cast<std::size_t>(p) * n + static_cast<std::size_t>(q);
      blk_[key] = blk;
      local_[key] = static_cast<std::int32_t>(kets[static_cast<std::size_t>(blk)].size());
      kets[static_cast<std::size_t>(blk)].push_back({p, q});
    }
  }

  const auto& tb = H.two_body();
  for (std::size_t blk = 0; blk < 6; ++blk) {
    const auto& ks = kets[blk];
    const auto nk = static_cast<std::int64_t>(ks.size());
    vmon_[blk] = linalg::Matrix<double>(nk, nk);
    vmon_exch_[blk] = linalg::Matrix<double>(nk, nk);
    // Upper triangle (ket >= bra), mirrored: both matrices are symmetric.
    for (std::int64_t bra = 0; bra < nk; ++bra) {
      const OrbitIndex a{ks[static_cast<std::size_t>(bra)][0]};
      const OrbitIndex b{ks[static_cast<std::size_t>(bra)][1]};
      const int ja = two_j_[static_cast<std::size_t>(a.value())];
      const int jb = two_j_[static_cast<std::size_t>(b.value())];
      for (std::int64_t ket = bra; ket < nk; ++ket) {
        const OrbitIndex c{ks[static_cast<std::size_t>(ket)][0]};
        const OrbitIndex d{ks[static_cast<std::size_t>(ket)][1]};
        double v = 0.0;
        double vx = 0.0;
        for (int J2 = std::abs(ja - jb); J2 <= ja + jb; J2 += 2) {
          v += (J2 + 1) * tb.get_unnormalized(a, b, c, d, TwoJ{J2});
          vx += (J2 + 1) * tb.get_unnormalized(a, b, d, c, TwoJ{J2});
        }
        vmon_[blk](bra, ket) = v;
        vmon_[blk](ket, bra) = v;
        vmon_exch_[blk](bra, ket) = vx;
        vmon_exch_[blk](ket, bra) = vx;
      }
    }
  }
}

void Monopole2B::apply(const linalg::Matrix<double>& rho, linalg::Matrix<double>& Vij) const {
  IMSRG_CHECK(rho.rows() == norb_ && rho.cols() == norb_, "rho must be N x N");
  Vij = linalg::Matrix<double>(norb_, norb_);
  const auto n = static_cast<std::size_t>(norb_);
  const auto key = [n](std::int32_t p, std::int32_t q) {
    return static_cast<std::size_t>(std::min(p, q)) * n + static_cast<std::size_t>(std::max(p, q));
  };
  for (std::int32_t i = 0; i < norb_; ++i) {
    for (std::int32_t j : channel_of_[static_cast<std::size_t>(i)]) {
      if (j < i) continue;
      double sum = 0.0;
      for (std::int32_t a = 0; a < norb_; ++a) {
        const std::size_t kia = key(i, a);
        const auto blk = static_cast<std::size_t>(blk_[kia]);
        const std::int64_t li = local_[kia];
        for (std::int32_t b : channel_of_[static_cast<std::size_t>(a)]) {
          const double r = rho(a, b);
          if (r == 0.0) continue;
          const std::size_t kjb = key(j, b);
          const std::int64_t lj = local_[kjb];
          const bool exch = (a > i) != (b > j);
          sum += r * (exch ? vmon_exch_[blk](li, lj) : vmon_[blk](li, lj));
        }
      }
      sum /= static_cast<double>(two_j_[static_cast<std::size_t>(i)] + 1);
      Vij(i, j) = sum;
      Vij(j, i) = sum;
    }
  }
}

Monopole3B::Monopole3B(const op::Operator& H, const op::No2bThreeBody& v3, std::int32_t e3max) {
  const auto& basis = H.model_space().basis();
  norb_ = basis.num_orbits();
  const auto chans = one_body_channels(basis);
  std::vector<const std::vector<std::int32_t>*> channel_of(static_cast<std::size_t>(norb_));
  for (const auto& ch : chans) {
    for (std::int32_t i : ch) channel_of[static_cast<std::size_t>(i)] = &ch;
  }
  const auto orb = [&](std::int32_t i) -> const basis::Orbit& {
    return basis.orbit(OrbitIndex{i});
  };

  for (std::int32_t i = 0; i < norb_; ++i) {
    const auto& oi = orb(i);
    for (std::int32_t j : *channel_of[static_cast<std::size_t>(i)]) {
      if (j < i) continue;
      const auto& oj = orb(j);
      for (std::int32_t a = 0; a < norb_; ++a) {
        const auto& oa = orb(a);
        for (std::int32_t b : *channel_of[static_cast<std::size_t>(a)]) {
          const auto& ob = orb(b);
          for (std::int32_t c = 0; c < norb_; ++c) {
            const auto& oc = orb(c);
            if (oa.e() + oc.e() + oi.e() > e3max) continue;
            for (std::int32_t d : *channel_of[static_cast<std::size_t>(c)]) {
              const auto& od = orb(d);
              if (ob.e() + od.e() + oj.e() > e3max) continue;
              if ((oi.l + oa.l + ob.l + oj.l + oc.l + od.l) % 2 != 0) continue;
              double v = 0.0;
              const int jlo = std::abs(oa.j2.twice - oc.j2.twice) / 2;
              const int jhi = (oa.j2.twice + oc.j2.twice) / 2;
              // imsrg++ passes the integer J itself as the J2 argument (no (2J+1) weight).
              for (int J = jlo; J <= jhi; ++J) v += v3.get_pn(oa, oc, oi, ob, od, oj, J);
              if (v == 0.0) continue;
              keys_.push_back({static_cast<std::uint16_t>(i), static_cast<std::uint16_t>(j),
                               static_cast<std::uint16_t>(a), static_cast<std::uint16_t>(b),
                               static_cast<std::uint16_t>(c), static_cast<std::uint16_t>(d)});
              values_.push_back(v / static_cast<double>(oi.j2.twice + 1));
            }
          }
        }
      }
    }
  }
}

void Monopole3B::apply(const linalg::Matrix<double>& rho, linalg::Matrix<double>& V3ij) const {
  IMSRG_CHECK(rho.rows() == norb_ && rho.cols() == norb_, "rho must be N x N");
  V3ij = linalg::Matrix<double>(norb_, norb_);
  for (std::size_t k = 0; k < keys_.size(); ++k) {
    const Key& key = keys_[k];
    V3ij(key.i, key.j) += rho(key.a, key.b) * rho(key.c, key.d) * values_[k];
  }
  for (std::int32_t i = 0; i < norb_; ++i) {
    for (std::int32_t j = i + 1; j < norb_; ++j) V3ij(j, i) = V3ij(i, j);
  }
}

}  // namespace imsrg::hf::detail
