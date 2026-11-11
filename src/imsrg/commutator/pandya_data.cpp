// Construction of the Pandya data of the particle-hole term (design/12 section 4.2). Host-side
// only; no Kokkos.

#include <cstdint>
#include <vector>

#include "imsrg/commutator/detail/pandya.hpp"

namespace imsrg::commutator::detail {

namespace {

/// Device buffer holding a copy of `v` (empty buffer for an empty vector).
template <class T>
exec::DeviceBuffer<T> upload(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> b(v.size(), label);
  if (!v.empty()) b.copy_from_host(std::span<const T>(v));
  return b;
}

}  // namespace

PandyaData build_pandya_data(const basis::ModelSpace& ms) {
  const auto& ph = ms.particle_hole();
  const auto& tb = ms.two_body();
  const auto occ = ms.reference().occupations();
  const std::int32_t ncc = ph.num_channels();

  PandyaData d;
  d.num_cc = ncc;
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
    d.yk_off_host[cs + 1] = d.yk_off_host[cs] + static_cast<std::int64_t>(nk) * 2 * n;
    d.w_off_host[cs + 1] = d.w_off_host[cs] + n * 2 * n;
  }

  d.ncc = upload(d.ncc_host, "pandya::ncc");
  d.nk = upload(d.nk_host, "pandya::nk");
  d.ket_off = upload(ket_off, "pandya::ket_off");
  d.k_off = upload(k_off, "pandya::k_off");
  d.ket_p = upload(ket_p, "pandya::ket_p");
  d.ket_q = upload(ket_q, "pandya::ket_q");
  d.k_a = upload(k_a, "pandya::k_a");
  d.k_b = upload(k_b, "pandya::k_b");
  d.k_d = upload(k_d, "pandya::k_d");
  d.xk_off = upload(d.xk_off_host, "pandya::xk_off");
  d.yk_off = upload(d.yk_off_host, "pandya::yk_off");
  d.w_off = upload(d.w_off_host, "pandya::w_off");
  const auto jo = ph.ket_J_offsets();
  const auto lt = ph.local_index_table();
  d.ket_J_offsets =
      upload(std::vector<std::int32_t>(jo.begin(), jo.end()), "pandya::ket_J_offsets");
  d.local_index_table =
      upload(std::vector<std::int32_t>(lt.begin(), lt.end()), "pandya::local_index_table");

  const auto sz = [](const std::vector<std::int64_t>& off) {
    return static_cast<std::size_t>(off.back());
  };
  d.xk = exec::DeviceBuffer<double>(sz(d.xk_off_host), "pandya::XK");
  d.yk = exec::DeviceBuffer<double>(sz(d.yk_off_host), "pandya::YK");
  d.w = exec::DeviceBuffer<double>(sz(d.w_off_host), "pandya::W");
  d.zb = exec::DeviceBuffer<double>(sz(d.w_off_host), "pandya::ZB");

  PandyaView& v = d.view_;
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
  v.zb = d.zb.device_data();

  for (std::int32_t c = 0; c < ncc; ++c) {
    const auto cs = static_cast<std::size_t>(c);
    const std::int64_t n = d.ncc_host[cs];
    const std::int64_t nk = d.nk_host[cs];
    if (n == 0 || nk == 0) continue;
    linalg::GemmProblem g;
    g.alpha = 1.0;
    g.A = linalg::MatrixView<const double>{v.xk + d.xk_off_host[cs], n, nk, n};
    g.opA = linalg::Op::NoTrans;
    g.B = linalg::MatrixView<const double>{v.yk + d.yk_off_host[cs], nk, 2 * n, nk};
    g.opB = linalg::Op::NoTrans;
    g.beta = 0.0;
    g.C = linalg::MatrixView<double>{v.w + d.w_off_host[cs], n, 2 * n, n};
    d.gemms.push_back(g);
  }
  return d;
}

}  // namespace imsrg::commutator::detail
