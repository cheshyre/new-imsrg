// Kernels evaluating the device accessors of DeviceTables. Free functions only (design/12 s5);
// lambdas capture the (trivially copyable) views and raw output pointers by value.
#include "device_tables_kernels.hpp"

#include "imsrg/exec/kokkos_backend.hpp"

namespace imsrg::basis::test {

void eval_ket_tables(const TwoBodyTableView& tb_in, std::int32_t max_J,
                     exec::DeviceBuffer<std::int32_t>& local,
                     exec::DeviceBuffer<std::int32_t>& channel,
                     exec::DeviceBuffer<std::int32_t>& ket_idx) {
  const TwoBodyTableView tb = tb_in;
  std::int32_t* out_local = local.device_data();
  std::int32_t* out_channel = channel.device_data();
  std::int32_t* out_ket = ket_idx.device_data();
  const std::int32_t nj = max_J + 1;
  exec::parallel_for(
      "test::eval_ket_tables", static_cast<exec::Index>(tb.num_kets) * nj,
      IMSRG_LAMBDA(const exec::Index t) {
        const std::int32_t k = static_cast<std::int32_t>(t / nj);
        const std::int32_t J = static_cast<std::int32_t>(t % nj);
        const OrbitIndex p{tb.ket_p[k]};
        const OrbitIndex q{tb.ket_q[k]};
        out_local[t] = tb.local_index(KetIndex{k}, TwoJ{2 * J});
        out_channel[t] = tb.channel_of(p, q, TwoJ{2 * J}).value();
        if (J == 0) out_ket[k] = tb.ket_index(p, q).value();
      });
  exec::fence("test::eval_ket_tables");
}

void eval_invalid_two_j(const TwoBodyTableView& tb_in, std::int32_t max_J,
                        exec::DeviceBuffer<std::int32_t>& out) {
  const TwoBodyTableView tb = tb_in;
  std::int32_t* o = out.device_data();
  exec::parallel_for(
      "test::eval_invalid_two_j", static_cast<exec::Index>(tb.num_kets),
      IMSRG_LAMBDA(const exec::Index k) {
        const KetIndex ki{static_cast<std::int32_t>(k)};
        o[3 * k + 0] = tb.local_index(ki, TwoJ{-2});
        o[3 * k + 1] = tb.local_index(ki, TwoJ{1});
        o[3 * k + 2] = tb.local_index(ki, TwoJ{2 * (max_J + 1)});
      });
  exec::fence("test::eval_invalid_two_j");
}

void eval_global_kets(const TwoBodyTableView& tb_in, exec::DeviceBuffer<std::int32_t>& out) {
  const TwoBodyTableView tb = tb_in;
  std::int32_t* o = out.device_data();
  exec::parallel_for(
      "test::eval_global_kets", static_cast<exec::Index>(tb.num_channels),
      IMSRG_LAMBDA(const exec::Index c) {
        const ChannelIndex ch{static_cast<std::int32_t>(c)};
        for (std::int32_t i = 0; i < tb.channel_num_kets[c]; ++i) {
          o[tb.channel_ket_offset[c] + i] = tb.global_ket(ch, LocalKetIndex{i}).value();
        }
      });
  exec::fence("test::eval_global_kets");
}

void eval_subsets(const KetSubsetTableView& sv_in, exec::DeviceBuffer<std::int32_t>& begin,
                  exec::DeviceBuffer<std::int32_t>& end, exec::DeviceBuffer<std::int32_t>& size,
                  exec::DeviceBuffer<std::int32_t>& kets, exec::DeviceBuffer<double>& occ,
                  exec::DeviceBuffer<double>& vac) {
  const KetSubsetTableView sv = sv_in;
  std::int32_t* ob = begin.device_data();
  std::int32_t* oe = end.device_data();
  std::int32_t* os = size.device_data();
  std::int32_t* ok = kets.device_data();
  double* oo = occ.device_data();
  double* ov = vac.device_data();
  exec::parallel_for(
      "test::eval_subsets", static_cast<exec::Index>(sv.num_channels) * kNumKetClasses,
      IMSRG_LAMBDA(const exec::Index t) {
        const ChannelIndex ch{static_cast<std::int32_t>(t / kNumKetClasses)};
        const KetClass c = static_cast<KetClass>(t % kNumKetClasses);
        const std::int32_t b = sv.begin(ch, c);
        const std::int32_t e = sv.end(ch, c);
        ob[t] = b;
        oe[t] = e;
        os[t] = sv.size(ch, c);
        for (std::int32_t i = b; i < e; ++i) {
          ok[i] = sv.local_kets[i];
          oo[i] = sv.occupation_products[i];
          ov[i] = sv.vacancy_products[i];
        }
      });
  exec::fence("test::eval_subsets");
}

void eval_orbits(const OrbitTableView& ov_in, exec::DeviceBuffer<std::int32_t>& n,
                 exec::DeviceBuffer<std::int32_t>& l, exec::DeviceBuffer<std::int32_t>& j2,
                 exec::DeviceBuffer<std::int32_t>& tz2, exec::DeviceBuffer<std::int32_t>& cvq,
                 exec::DeviceBuffer<double>& occ) {
  const OrbitTableView ov = ov_in;
  std::int32_t* on = n.device_data();
  std::int32_t* ol = l.device_data();
  std::int32_t* oj = j2.device_data();
  std::int32_t* ot = tz2.device_data();
  std::int32_t* oc = cvq.device_data();
  double* oo = occ.device_data();
  exec::parallel_for(
      "test::eval_orbits", static_cast<exec::Index>(ov.num_orbits),
      IMSRG_LAMBDA(const exec::Index i) {
        on[i] = ov.n[i];
        ol[i] = ov.l[i];
        oj[i] = ov.j2[i];
        ot[i] = ov.tz2[i];
        oc[i] = ov.cvq[i];
        oo[i] = ov.occupation[i];
      });
  exec::fence("test::eval_orbits");
}

}  // namespace imsrg::basis::test
