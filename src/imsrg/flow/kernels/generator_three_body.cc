// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Kernel TU of the three-body generator (see
// imsrg/flow/detail/generator_three_body.h).
#include "imsrg/flow/detail/generator_three_body.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::flow::detail {

namespace {

// Function families of the generator functions (same as the host
// `GeneratorFunction`).
constexpr int kFamWhite = 0;
constexpr int kFamAtan = 1;
constexpr int kFamImaginaryTime = 2;
constexpr int kFamWegner = 3;
constexpr int kFamQtransferAtan = 4;

int Family(GeneratorKind k) {
  switch (k) {
    case GeneratorKind::kWhite:
    case GeneratorKind::kShellModel:
      return kFamWhite;
    case GeneratorKind::kAtan:
    case GeneratorKind::kShellModelAtan:
      return kFamAtan;
    case GeneratorKind::kImaginaryTime:
    case GeneratorKind::kShellModelImaginaryTime:
      return kFamImaginaryTime;
    case GeneratorKind::kWegner:
    case GeneratorKind::kShellModelWegner:
      return kFamWegner;
    case GeneratorKind::kQtransferAtan:
      return kFamQtransferAtan;
    default:
      break;
  }
  throw InvariantError("three-body generator: kind has no generator function");
}

// Triple classes.
constexpr int kClsOther = 0;
constexpr int kClsParticle = 1;
constexpr int kClsCore = 2;

// Device copy of a host vector.
template <class T>
exec::DeviceBuffer<T> Upload(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> b(v.size(), label);
  if (!v.empty()) b.CopyFromHost(v);
  return b;
}

// Same arithmetic as `FloorDenominator` of generator.cc, device-callable.
IMSRG_INLINE_FUNCTION double FloorDen(double delta, double cutoff) {
  if (Kokkos::abs(delta) < cutoff) return delta < 0.0 ? -cutoff : cutoff;
  return delta;
}

// Same arithmetic as `GeneratorFunction` of generator.cc, device-callable.
IMSRG_INLINE_FUNCTION double GenFunction(int family, double h, double d,
                                         double mn, double hbarc2,
                                         double half_nq) {
  switch (family) {
    case kFamWhite:
      return h / d;
    case kFamAtan:
      return 0.5 * Kokkos::atan(2.0 * h / d);
    case kFamImaginaryTime:
      return h * (d < 0.0 ? -1.0 : 1.0);
    case kFamWegner:
      return h * d;
    default: {
      const double w = Kokkos::pow(Kokkos::abs(d) * mn / hbarc2, half_nq);
      return w * 0.5 * Kokkos::atan(2.0 * h / d);
    }
  }
}

}  // namespace

void FillThreeBodyGenerator(const op::ThreeBody& H3, op::ThreeBody& eta3,
                            const basis::ThreeBodySpace& ts,
                            const basis::RowSet& rows,
                            std::span<const double> f_diag,
                            std::span<const std::int32_t> orbit_is_core,
                            double cutoff, GeneratorKind kind, int nq) {
  IMSRG_CHECK(H3.hermiticity() == Hermiticity::kHermitian &&
                  eta3.hermiticity() == Hermiticity::kAntiHermitian,
              "three-body generator: H3 must be Hermitian, eta3 "
              "anti-Hermitian");
  IMSRG_CHECK(
      H3.size() == eta3.size() && H3.num_channels() == ts.num_channels(),
      "three-body generator: H3 and eta3 differ in layout");
  const int family = Family(kind);
  const std::int64_t nrows_set = rows.NumRows();
  if (nrows_set == 0) return;

  // Per-triple tables: orbits and class (particle = no core orbit in the
  // triple, core = all three orbits core).
  const std::int32_t ntr = ts.num_triples();
  std::vector<std::int32_t> tr_a(static_cast<std::size_t>(ntr));
  std::vector<std::int32_t> tr_b(tr_a.size());
  std::vector<std::int32_t> tr_c(tr_a.size());
  std::vector<std::int32_t> tr_cls(tr_a.size());
  for (std::int32_t t = 0; t < ntr; ++t) {
    const basis::OrbitTriple& o = ts.triple(basis::Triple3Index{t});
    const auto s = static_cast<std::size_t>(t);
    tr_a[s] = o.a.value();
    tr_b[s] = o.b.value();
    tr_c[s] = o.c.value();
    const int ncore = (orbit_is_core[o.a.ToSize()] != 0 ? 1 : 0) +
                      (orbit_is_core[o.b.ToSize()] != 0 ? 1 : 0) +
                      (orbit_is_core[o.c.ToSize()] != 0 ? 1 : 0);
    tr_cls[s] = ncore == 0 ? kClsParticle : ncore == 3 ? kClsCore : kClsOther;
  }
  // Per flat row: its triple.
  const std::int64_t nflat = ts.NumRows();
  std::vector<std::int32_t> row_triple(static_cast<std::size_t>(nflat));
  for (std::int64_t r = 0; r < nflat; ++r) {
    row_triple[static_cast<std::size_t>(r)] = ts.row_triple(r).value();
  }
  // Per channel: flat row offset, dimension, block offset in the buffers.
  const std::int32_t nch = ts.num_channels();
  std::vector<std::int64_t> ch_row_off(static_cast<std::size_t>(nch));
  std::vector<std::int32_t> ch_dim(ch_row_off.size());
  for (std::int32_t c = 0; c < nch; ++c) {
    const basis::KeptChannel3& kc = ts.channel(Channel3Index{c});
    ch_row_off[static_cast<std::size_t>(c)] = kc.row_offset;
    ch_dim[static_cast<std::size_t>(c)] = kc.dim;
  }
  const std::span<const std::int64_t> blk = eta3.block_offsets();
  const std::vector<std::int64_t> ch_blk(blk.begin(), blk.end());
  // Per row of the set: channel and local row.
  std::vector<std::int32_t> set_ch;
  std::vector<std::int32_t> set_k;
  set_ch.reserve(static_cast<std::size_t>(nrows_set));
  set_k.reserve(static_cast<std::size_t>(nrows_set));
  for (const basis::RowRange& rg : rows.ranges()) {
    for (std::int32_t k = rg.row_begin.value(); k < rg.row_end.value(); ++k) {
      set_ch.push_back(rg.channel.value());
      set_k.push_back(k);
    }
  }
  const std::vector<double> f_host(f_diag.begin(), f_diag.end());

  const auto d_tr_a = Upload(tr_a, "flow::tr_a");
  const auto d_tr_b = Upload(tr_b, "flow::tr_b");
  const auto d_tr_c = Upload(tr_c, "flow::tr_c");
  const auto d_tr_cls = Upload(tr_cls, "flow::tr_cls");
  const auto d_row_triple = Upload(row_triple, "flow::row_triple");
  const auto d_ch_row_off = Upload(ch_row_off, "flow::ch_row_off");
  const auto d_ch_dim = Upload(ch_dim, "flow::ch_dim");
  const auto d_ch_blk = Upload(ch_blk, "flow::ch_blk");
  const auto d_set_ch = Upload(set_ch, "flow::set_ch");
  const auto d_set_k = Upload(set_k, "flow::set_k");
  const auto d_f = Upload(f_host, "flow::f_diag");

  const auto v_a = exec::View(d_tr_a);
  const auto v_b = exec::View(d_tr_b);
  const auto v_c = exec::View(d_tr_c);
  const auto v_cls = exec::View(d_tr_cls);
  const auto v_rt = exec::View(d_row_triple);
  const auto v_cro = exec::View(d_ch_row_off);
  const auto v_dim = exec::View(d_ch_dim);
  const auto v_blk = exec::View(d_ch_blk);
  const auto v_sch = exec::View(d_set_ch);
  const auto v_sk = exec::View(d_set_k);
  const auto v_f = exec::View(d_f);
  const double* h_data = H3.device_data();
  double* z_data = eta3.device_data();
  const double mn = constants::kMNucleon;
  const double hbarc2 = constants::kHbarc * constants::kHbarc;
  const double half_nq = 0.5 * nq;

  exec::ParallelFor(
      "flow::three_body_generator", static_cast<exec::Index>(nrows_set),
      IMSRG_LAMBDA(const exec::Index r) {
        const int ch = v_sch(r);
        const std::int64_t k = v_sk(r);
        const std::int64_t n = v_dim(ch);
        const std::int64_t base = v_cro(ch);
        // Column k of the packed lower triangle: z(m, k), m = k..n-1.
        const std::int64_t col =
            v_blk(ch) + op::ThreeBody::PackedIndex(n, k, k);
        const int clk = v_cls(v_rt(base + k));
        z_data[col] = 0.0;  // the diagonal of an anti-Hermitian part
        for (std::int64_t m = k + 1; m < n; ++m) {
          const int tm = v_rt(base + m);
          const int clm = v_cls(tm);
          double out = 0.0;
          if ((clm == kClsParticle && clk == kClsCore) ||
              (clm == kClsCore && clk == kClsParticle)) {
            const bool m_is_bra = clm == kClsParticle;
            const int tk = v_rt(base + k);
            const int tb = m_is_bra ? tm : tk;  // bra: all-particle triple
            const int tt = m_is_bra ? tk : tm;  // ket: all-core triple
            const double h = h_data[col + (m - k)];
            const double delta = v_f(v_a(tb)) + v_f(v_b(tb)) + v_f(v_c(tb)) -
                                 v_f(v_a(tt)) - v_f(v_b(tt)) - v_f(v_c(tt));
            const double d = FloorDen(delta, cutoff);
            const double v = GenFunction(family, h, d, mn, hbarc2, half_nq);
            out = m_is_bra ? v : -v;
          }
          z_data[col + (m - k)] = out;
        }
      });
  exec::Fence("flow::three_body_generator");
}

}  // namespace imsrg::flow::detail
