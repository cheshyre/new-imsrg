// Dense angular-momentum tables: construction on the host, storage in DeviceBuffers.
#include "imsrg/angmom/tables.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "imsrg/angmom/wigner.hpp"
#include "imsrg/base/assert.hpp"

namespace imsrg::angmom {

namespace {

/// Single-particle labels (n1, l1, n2, l2) of the K(E) tuples at one energy, in table order.
using Tuple = std::array<int, 4>;

std::vector<Tuple> tuples_at(int E) {
  std::vector<Tuple> t;
  for (int l1 = 0; l1 <= E; ++l1) {
    for (int n1 = 0; 2 * n1 + l1 <= E; ++n1) {
      for (int l2 = 0; 2 * n1 + l1 + l2 <= E; ++l2) {
        const int rem = E - 2 * n1 - l1 - l2;
        if (rem % 2 == 0) {
          t.push_back({n1, l1, rem / 2, l2});
        }
      }
    }
  }
  return t;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// SixJTable

SixJTable::SixJTable(int emax)
    : emax_(emax), values_([&] {
        IMSRG_CHECK(emax >= 0, "SixJTable: emax must be >= 0");
        const auto nj = static_cast<std::size_t>(emax + 1);
        const auto nJ = static_cast<std::size_t>(2 * emax + 2);
        return exec::DeviceBuffer<double>(nj * nj * nj * nj * nJ * nJ, "imsrg::SixJTable");
      }()) {
  const int nj = emax + 1;
  const int nJ = 2 * emax + 2;
  std::span<double> out = values_.host_span();
  std::size_t i = 0;
  for (int ia = 0; ia < nj; ++ia) {
    for (int ib = 0; ib < nj; ++ib) {
      for (int J = 0; J < nJ; ++J) {
        for (int ic = 0; ic < nj; ++ic) {
          for (int id = 0; id < nj; ++id) {
            for (int Jp = 0; Jp < nJ; ++Jp, ++i) {
              out[i] = wigner_6j(TwoJ{2 * ia + 1}, TwoJ{2 * ib + 1}, TwoJ{2 * J}, TwoJ{2 * ic + 1},
                                 TwoJ{2 * id + 1}, TwoJ{2 * Jp});
            }
          }
        }
      }
    }
  }
  values_.sync_to_device();
}

SixJTable::View SixJTable::host_view() const noexcept {
  return View{values_.host_span().data(), emax_};
}
SixJTable::View SixJTable::device_view() const noexcept {
  return View{values_.device_data(), emax_};
}

// ---------------------------------------------------------------------------------------------
// NineJLSjjTable

NineJLSjjTable::NineJLSjjTable(int lmax)
    : lmax_(lmax), values_([&] {
        IMSRG_CHECK(lmax >= 0, "NineJLSjjTable: lmax must be >= 0");
        const auto nl = static_cast<std::size_t>(lmax + 1);
        return exec::DeviceBuffer<double>(nl * 2 * nl * 2 * static_cast<std::size_t>(2 * lmax + 1) *
                                              2 * static_cast<std::size_t>(2 * lmax + 2),
                                          "imsrg::NineJLSjjTable");
      }()) {
  const int nL = 2 * lmax + 1;
  const int nJ = 2 * lmax + 2;
  std::span<double> out = values_.host_span();
  std::size_t i = 0;
  for (int la = 0; la <= lmax; ++la) {
    for (int sja = 0; sja < 2; ++sja) {
      const int ja2 = 2 * la - 1 + 2 * sja;
      for (int lb = 0; lb <= lmax; ++lb) {
        for (int sjb = 0; sjb < 2; ++sjb) {
          const int jb2 = 2 * lb - 1 + 2 * sjb;
          for (int L = 0; L < nL; ++L) {
            for (int S = 0; S < 2; ++S) {
              for (int J = 0; J < nJ; ++J, ++i) {
                out[i] = (ja2 < 1 || jb2 < 1)
                             ? 0.0
                             : norm_9j(TwoJ{2 * la}, TwoJ{1}, TwoJ{ja2}, TwoJ{2 * lb}, TwoJ{1},
                                       TwoJ{jb2}, TwoJ{2 * L}, TwoJ{2 * S}, TwoJ{2 * J});
              }
            }
          }
        }
      }
    }
  }
  values_.sync_to_device();
}

NineJLSjjTable::View NineJLSjjTable::host_view() const noexcept {
  return View{values_.host_span().data(), lmax_};
}
NineJLSjjTable::View NineJLSjjTable::device_view() const noexcept {
  return View{values_.device_data(), lmax_};
}

// ---------------------------------------------------------------------------------------------
// MoshinskyTable

namespace {

std::int64_t block_count(int E) {
  const auto K = static_cast<std::int64_t>(tuples_at(E).size());
  return static_cast<std::int64_t>(E + 1) * K * K;
}

}  // namespace

std::int64_t MoshinskyTable::value_count(int e2max) noexcept {
  std::int64_t total = 0;
  for (int E = 0; E <= e2max; ++E) {
    total += block_count(E);
  }
  return total;
}

MoshinskyTable::MoshinskyTable(int e2max)
    : e2max_(e2max),
      values_([&] {
        IMSRG_CHECK(e2max >= 0 && e2max <= moshinsky_max_energy,
                    "MoshinskyTable: e2max must be in [0, moshinsky_max_energy]");
        return exec::DeviceBuffer<double>(static_cast<std::size_t>(value_count(e2max)),
                                          "imsrg::MoshinskyTable::values");
      }()),
      position_(static_cast<std::size_t>(e2max / 2 + 1) * static_cast<std::size_t>(e2max + 1) *
                    static_cast<std::size_t>(e2max / 2 + 1) * static_cast<std::size_t>(e2max + 1),
                "imsrg::MoshinskyTable::position"),
      offset_(static_cast<std::size_t>(e2max + 1) * static_cast<std::size_t>(e2max + 1),
              "imsrg::MoshinskyTable::offset"),
      block_size_(static_cast<std::size_t>(e2max + 1), "imsrg::MoshinskyTable::block_size") {
  const int nn = e2max / 2 + 1;
  const int nl = e2max + 1;
  std::span<std::int32_t> pos = position_.host_span();
  std::span<std::int64_t> off = offset_.host_span();
  std::span<std::int32_t> ks = block_size_.host_span();
  std::span<double> out = values_.host_span();
  for (auto& p : pos) {
    p = -1;
  }
  std::int64_t next = 0;
  for (int E = 0; E <= e2max; ++E) {
    const std::vector<Tuple> t = tuples_at(E);
    const auto K = static_cast<std::int64_t>(t.size());
    ks[static_cast<std::size_t>(E)] = static_cast<std::int32_t>(K);
    for (std::size_t j = 0; j < t.size(); ++j) {
      const auto& u = t[j];
      pos[static_cast<std::size_t>(((u[0] * nl + u[1]) * nn + u[2]) * nl + u[3])] =
          static_cast<std::int32_t>(j);
    }
    for (int lambda = 0; lambda <= E; ++lambda) {
      off[static_cast<std::size_t>(E * (e2max + 1) + lambda)] = next;
      for (std::int64_t icm = 0; icm < K; ++icm) {
        const auto& c = t[static_cast<std::size_t>(icm)];
        for (std::int64_t ilab = 0; ilab < K; ++ilab) {
          const auto& u = t[static_cast<std::size_t>(ilab)];
          out[static_cast<std::size_t>(next + icm * K + ilab)] = moshinsky(
              HoNL{c[0], c[1]}, HoNL{c[2], c[3]}, HoNL{u[0], u[1]}, HoNL{u[2], u[3]}, lambda);
        }
      }
      next += K * K;
    }
  }
  IMSRG_CHECK(next == static_cast<std::int64_t>(out.size()), "MoshinskyTable: size mismatch");
  values_.sync_to_device();
  position_.sync_to_device();
  offset_.sync_to_device();
  block_size_.sync_to_device();
}

int MoshinskyTable::block_size(int E) const noexcept {
  if (E < 0 || E > e2max_) {
    return 0;
  }
  return block_size_.host_span()[static_cast<std::size_t>(E)];
}

std::size_t MoshinskyTable::size_bytes() const noexcept {
  return values_.size() * sizeof(double) + position_.size() * sizeof(std::int32_t) +
         offset_.size() * sizeof(std::int64_t) + block_size_.size() * sizeof(std::int32_t);
}

MoshinskyTable::View MoshinskyTable::host_view() const noexcept {
  return View{values_.host_span().data(), position_.host_span().data(), offset_.host_span().data(),
              block_size_.host_span().data(), e2max_};
}

MoshinskyTable::View MoshinskyTable::device_view() const noexcept {
  return View{values_.device_data(), position_.device_data(), offset_.device_data(),
              block_size_.device_data(), e2max_};
}

}  // namespace imsrg::angmom
