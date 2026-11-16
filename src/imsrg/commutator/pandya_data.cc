// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Construction of the Pandya data of the particle-hole term (design/12 section
// 4.2). Host-side only; no Kokkos.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/pandya.h"

namespace imsrg::commutator::detail {

namespace {

/// Device buffer holding a copy of `v` (empty buffer for an empty vector).
template <class T>
exec::DeviceBuffer<T> Upload(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> b(v.size(), label);
  if (!v.empty()) b.CopyFromHost(std::span<const T>(v));
  return b;
}

}  // namespace

namespace {

/// Host tables of the cc channels uploaded by UploadChannelTables.
struct ChannelHostTables {
  std::vector<std::int32_t> ket_off, k_off, ket_p, ket_q, k_a, k_b;
  std::vector<double> k_d;
};

/// Fills the per-channel host sizes and offsets of `d` (ncc_host, nk_host,
/// xk_off_host, yk_off_host, w_off_host) and builds the ket and nonzero-
/// occupation-difference tables of every particle-hole channel.
ChannelHostTables BuildChannelTables(const basis::ModelSpace& ms,
                                     PandyaData& d) {
  const auto& ph = ms.particle_hole();
  const auto& tb = ms.two_body();
  const auto occ = ms.reference().occupations();
  const std::int32_t ncc = ph.num_channels();
  d.ncc_host.assign(static_cast<std::size_t>(ncc), 0);
  d.nk_host.assign(static_cast<std::size_t>(ncc), 0);
  d.xk_off_host.assign(static_cast<std::size_t>(ncc) + 1, 0);
  d.yk_off_host.assign(static_cast<std::size_t>(ncc) + 1, 0);
  d.w_off_host.assign(static_cast<std::size_t>(ncc) + 1, 0);

  std::vector<std::int32_t> ket_off(static_cast<std::size_t>(ncc) + 1, 0);
  std::vector<std::int32_t> k_off(static_cast<std::size_t>(ncc) + 1, 0);
  std::vector<std::int32_t> ket_p, ket_q, k_a, k_b;
  std::vector<double> k_d;

  for (std::int32_t c = 0; c < ncc; ++c) {
    const auto kets = ph.channel_kets(ChannelIndex{c});
    const std::int64_t n = static_cast<std::int64_t>(kets.size());
    std::int32_t nk = 0;
    for (const KetIndex k : kets) {
      const auto& ket = tb.ket(k);
      const std::int32_t p = ket.p.value();
      const std::int32_t q = ket.q.value();
      ket_p.push_back(p);
      ket_q.push_back(q);
      const double np = occ[static_cast<std::size_t>(p)];
      const double nq = occ[static_cast<std::size_t>(q)];
      if (np != nq) {
        k_a.push_back(p);
        k_b.push_back(q);
        k_d.push_back(np - nq);
        k_a.push_back(q);
        k_b.push_back(p);
        k_d.push_back(nq - np);
        nk += 2;
      }
    }
    const auto cs = static_cast<std::size_t>(c);
    d.ncc_host[cs] = static_cast<std::int32_t>(n);
    d.nk_host[cs] = nk;
    ket_off[cs + 1] = ket_off[cs] + static_cast<std::int32_t>(n);
    k_off[cs + 1] = k_off[cs] + nk;
    d.xk_off_host[cs + 1] = d.xk_off_host[cs] + n * nk;
    d.yk_off_host[cs + 1] =
        d.yk_off_host[cs] + static_cast<std::int64_t>(nk) * n;
    d.w_off_host[cs + 1] = d.w_off_host[cs] + n * 2 * n;
  }
  return {std::move(ket_off), std::move(k_off), std::move(ket_p),
          std::move(ket_q),   std::move(k_a),   std::move(k_b),
          std::move(k_d)};
}

/// Uploads the channel tables and the ket-J offsets of the particle-hole
/// channels to device buffers of `d`.
void UploadChannelTables(const basis::ModelSpace& ms,
                         const ChannelHostTables& t, PandyaData& d) {
  const auto& ph = ms.particle_hole();
  d.ncc = Upload(d.ncc_host, "pandya::ncc");
  d.nk = Upload(d.nk_host, "pandya::nk");
  d.ket_off = Upload(t.ket_off, "pandya::ket_off");
  d.k_off = Upload(t.k_off, "pandya::k_off");
  d.ket_p = Upload(t.ket_p, "pandya::ket_p");
  d.ket_q = Upload(t.ket_q, "pandya::ket_q");
  d.k_a = Upload(t.k_a, "pandya::k_a");
  d.k_b = Upload(t.k_b, "pandya::k_b");
  d.k_d = Upload(t.k_d, "pandya::k_d");
  d.xk_off = Upload(d.xk_off_host, "pandya::xk_off");
  d.yk_off = Upload(d.yk_off_host, "pandya::yk_off");
  d.w_off = Upload(d.w_off_host, "pandya::w_off");
  const auto jo = ph.ket_J_offsets();
  const auto lt = ph.local_index_table();
  d.ket_J_offsets = Upload(std::vector<std::int32_t>(jo.begin(), jo.end()),
                           "pandya::ket_J_offsets");
  d.local_index_table = Upload(std::vector<std::int32_t>(lt.begin(), lt.end()),
                               "pandya::local_index_table");
}

/// Builds the kets-by-class tables of `d` (and checks the ascending ket order
/// of every two-body channel).
void BuildClassKets(const basis::ModelSpace& ms, PandyaData& d) {
  const auto& tb = ms.two_body();
  // Kets by class (parity, Tz): class = 3 parity + Tz + 1, ascending ket
  // order inside each class. The inverse transform over pairs of these kets
  // needs every two-body channel to list its kets in ascending global order
  // (then b >= a is the upper triangle of each block): checked here.
  for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
    const auto kets = tb.channel_kets(ChannelIndex{ch});
    for (std::size_t i = 1; i < kets.size(); ++i)
      IMSRG_CHECK(kets[i - 1].value() < kets[i].value(),
                  "BuildPandyaData: two-body channel kets are not in "
                  "ascending global order");
  }
  {
    constexpr std::int32_t kNumClasses = 6;
    std::vector<std::vector<std::int32_t>> lists(kNumClasses);
    const auto& orbits = ms.basis();
    for (std::int32_t k = 0; k < tb.num_kets(); ++k) {
      const auto& ket = tb.ket(KetIndex{k});
      const auto& a = orbits.orbit(ket.p);
      const auto& b = orbits.orbit(ket.q);
      const std::int32_t parity = (a.l + b.l) % 2;
      const std::int32_t tz = (a.tz2.twice + b.tz2.twice) / 2;
      lists[static_cast<std::size_t>(3 * parity + tz + 1)].push_back(k);
    }
    std::vector<std::int32_t> flat;
    std::vector<std::int32_t> offsets{0};
    for (const auto& l : lists) {
      flat.insert(flat.end(), l.begin(), l.end());
      offsets.push_back(static_cast<std::int32_t>(flat.size()));
    }
    d.num_class_kets = static_cast<std::int64_t>(flat.size());
    d.class_kets = Upload(flat, "pandya::class_kets");
    d.class_offset = Upload(offsets, "pandya::class_offset");
    d.kernel_view.num_classes = kNumClasses;
  }
}

/// Builds the cc-ket descriptors of every ordered orbit pair of `d`.
void BuildCcPairs(const basis::ModelSpace& ms, PandyaData& d) {
  const auto& ph = ms.particle_hole();
  // Descriptors of the cc ket of every ordered orbit pair (inverse transform).
  {
    const auto& sp = ms.basis();
    const std::int32_t norb = sp.num_orbits();
    const auto jo_host = ph.ket_J_offsets();
    std::vector<std::int32_t> pairs(4 * static_cast<std::size_t>(norb) *
                                    static_cast<std::size_t>(norb));
    for (std::int32_t p = 0; p < norb; ++p) {
      for (std::int32_t q = 0; q < norb; ++q) {
        const auto& op = sp.orbit(OrbitIndex{p});
        const auto& oq = sp.orbit(OrbitIndex{q});
        const std::int32_t jp = op.j2.twice, jq = oq.j2.twice;
        const std::int32_t jmin = (jp > jq ? jp - jq : jq - jp) / 2;
        const std::int32_t k =
            basis::ImsrgppKetIndex(OrbitIndex{std::min(p, q)},
                                   OrbitIndex{std::max(p, q)}, norb)
                .value();
        const std::int32_t dtz = op.tz2.twice - oq.tz2.twice;
        const std::int32_t cls =
            (dtz < 0 ? -dtz : dtz) + 2 + ((op.l + oq.l) & 1);
        std::int32_t* e =
            pairs.data() +
            4 * (static_cast<std::size_t>(p) * static_cast<std::size_t>(norb) +
                 static_cast<std::size_t>(q));
        e[0] = jo_host[static_cast<std::size_t>(k)] - jmin;
        e[1] = jmin;
        e[2] = (jp + jq) / 2;
        e[3] = cls + 256 * (((jp + jq) / 2) & 1);
      }
    }
    d.cc_pair = Upload(pairs, "pandya::cc_pair");
  }
}

/// Points the kernel view of `d` at its device buffers.
void FillKernelView(PandyaData& d, std::int32_t ncc) {
  PandyaView& v = d.kernel_view;
  v.num_cc = ncc;
  v.ncc = d.ncc.device_data();
  v.nk = d.nk.device_data();
  v.ket_off = d.ket_off.device_data();
  v.k_off = d.k_off.device_data();
  v.ket_p = d.ket_p.device_data();
  v.ket_q = d.ket_q.device_data();
  v.k_a = d.k_a.device_data();
  v.k_b = d.k_b.device_data();
  v.k_d = d.k_d.device_data();
  v.xk_off = d.xk_off.device_data();
  v.yk_off = d.yk_off.device_data();
  v.w_off = d.w_off.device_data();
  v.ket_J_offsets = d.ket_J_offsets.device_data();
  v.local_index_table = d.local_index_table.device_data();
  v.xk = d.xk.device_data();
  v.yk = d.yk.device_data();
  v.w = d.w.device_data();
  v.class_kets = d.class_kets.device_data();
  v.class_offset = d.class_offset.device_data();
  v.cc_pair = d.cc_pair.device_data();
}

/// Builds the per-channel GEMM problems of `d` over the kernel-view buffers.
void BuildGemmProblems(PandyaData& d, std::int32_t ncc) {
  const PandyaView& v = d.kernel_view;
  for (std::int32_t c = 0; c < ncc; ++c) {
    const auto cs = static_cast<std::size_t>(c);
    const std::int64_t n = d.ncc_host[cs];
    const std::int64_t nk = d.nk_host[cs];
    // k == 0 (nk == 0) blocks are skipped: their W stays zero
    // (zero-initialized, never written). The problems run on host BLAS (see the
    // host_is_device check in comm222_phss, design/12 section 3).
    if (n == 0 || nk == 0) continue;
    // W(:, 0 .. Ncc) = XK YK, and after the in-place row-pair flip of YK
    // W(:, Ncc .. 2 Ncc) = XK YK' (see the scratch layout in pandya.h).
    linalg::GemmProblem g;
    g.alpha = 1.0;
    g.A = linalg::MatrixView<const double>{v.xk + d.xk_off_host[cs], n, nk, n};
    g.opA = linalg::Op::kNoTrans;
    g.B = linalg::MatrixView<const double>{v.yk + d.yk_off_host[cs], nk, n, nk};
    g.opB = linalg::Op::kNoTrans;
    g.beta = 0.0;
    g.C = linalg::MatrixView<double>{v.w + d.w_off_host[cs], n, n, n};
    d.gemms.push_back(g);
    g.C = linalg::MatrixView<double>{v.w + d.w_off_host[cs] + n * n, n, n, n};
    d.gemms_right.push_back(g);
  }
}

}  // namespace

PandyaData BuildPandyaData(const basis::ModelSpace& ms) {
  IMSRG_CHECK(2 * ms.basis().emax() + 2 <= kPandyaMaxJp,
              "BuildPandyaData: emax too large for the J' buffers of the "
              "particle-hole inverse transform (kPandyaMaxJp)");
  const std::int32_t ncc = ms.particle_hole().num_channels();

  PandyaData d;
  d.num_cc = ncc;
  const ChannelHostTables t = BuildChannelTables(ms, d);
  UploadChannelTables(ms, t, d);
  BuildClassKets(ms, d);
  BuildCcPairs(ms, d);

  const auto sz = [](const std::vector<std::int64_t>& off) {
    return static_cast<std::size_t>(off.back());
  };
  d.xk = exec::DeviceBuffer<double>(sz(d.xk_off_host), "pandya::XK");
  d.yk = exec::DeviceBuffer<double>(sz(d.yk_off_host), "pandya::YK");
  d.w = exec::DeviceBuffer<double>(sz(d.w_off_host), "pandya::W");

  FillKernelView(d, ncc);
  BuildGemmProblems(d, ncc);
  return d;
}

void EnsureGeneralScratch(PandyaData& pd) {
  if (pd.general.allocated()) return;
  // Build into a local object and move-assign at the end: a throwing
  // allocation leaves `pd.general` unallocated.
  PandyaGeneralData g;
  const auto ncc = static_cast<std::size_t>(pd.num_cc);
  std::vector<std::int64_t> g_off(ncc + 1, 0);
  std::vector<std::int64_t> gz_off(ncc + 1, 0);
  for (std::size_t c = 0; c < ncc; ++c) {
    const std::int64_t n = pd.ncc_host[c];
    const std::int64_t nk = pd.nk_host[c];
    g_off[c + 1] = g_off[c] + 2 * n * nk;
    gz_off[c + 1] = gz_off[c] + 4 * n * n;
  }
  g.g_off = Upload(g_off, "pandya_general::g_off");
  g.gz_off = Upload(gz_off, "pandya_general::gz_off");
  g.a = exec::DeviceBuffer<double>(static_cast<std::size_t>(g_off.back()),
                                   "pandya_general::A");
  g.b = exec::DeviceBuffer<double>(static_cast<std::size_t>(g_off.back()),
                                   "pandya_general::B");
  g.z = exec::DeviceBuffer<double>(static_cast<std::size_t>(gz_off.back()),
                                   "pandya_general::Z");
  for (std::size_t c = 0; c < ncc; ++c) {
    const std::int64_t n = pd.ncc_host[c];
    const std::int64_t nk = pd.nk_host[c];
    // nk == 0 blocks are skipped: their Z stays zero (never written).
    if (n == 0 || nk == 0) continue;
    linalg::GemmProblem p;
    p.alpha = 1.0;
    p.A = linalg::MatrixView<const double>{g.a.device_data() + g_off[c], 2 * n,
                                           nk, 2 * n};
    p.opA = linalg::Op::kNoTrans;
    p.B = linalg::MatrixView<const double>{g.b.device_data() + g_off[c], nk,
                                           2 * n, nk};
    p.opB = linalg::Op::kNoTrans;
    p.beta = 0.0;
    p.C = linalg::MatrixView<double>{g.z.device_data() + gz_off[c], 2 * n,
                                     2 * n, 2 * n};
    g.gemms.push_back(p);
  }
  g.g_off_host = std::move(g_off);
  g.gz_off_host = std::move(gz_off);
  pd.general = std::move(g);
}

}  // namespace imsrg::commutator::detail
