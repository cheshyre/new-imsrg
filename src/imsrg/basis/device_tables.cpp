#include "imsrg/basis/device_tables.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace imsrg::basis {

namespace {

template <class T>
exec::DeviceBuffer<T> upload(std::span<const T> src, const char* label) {
  exec::DeviceBuffer<T> buf(src.size(), label);
  if (!src.empty()) buf.copy_from_host(src);
  return buf;
}

template <class T>
exec::DeviceBuffer<T> upload(const std::vector<T>& src, const char* label) {
  return upload<T>(std::span<const T>(src), label);
}

}  // namespace

DeviceTables::DeviceTables(const SingleParticleBasis& basis, const Reference& reference,
                           const Partition& partition, const TwoBodyChannels& channels,
                           const TwoBodyKetSubsets& subsets) {
  const auto orbs = basis.orbits();
  const std::size_t no = orbs.size();
  std::vector<std::int32_t> n(no), l(no), j2(no), tz2(no), cvq_v(no);
  for (std::size_t i = 0; i < no; ++i) {
    n[i] = orbs[i].n;
    l[i] = orbs[i].l;
    j2[i] = orbs[i].j2.twice;
    tz2[i] = orbs[i].tz2.twice;
    cvq_v[i] = static_cast<std::int32_t>(partition.classes()[i]);
  }
  orb_n_ = upload(n, "basis::orb_n");
  orb_l_ = upload(l, "basis::orb_l");
  orb_j2_ = upload(j2, "basis::orb_j2");
  orb_tz2_ = upload(tz2, "basis::orb_tz2");
  orb_cvq_ = upload(cvq_v, "basis::orb_cvq");
  orb_occ_ = upload<double>(reference.occupations(), "basis::orb_occ");

  const auto kets = channels.kets();
  std::vector<std::int32_t> kp(kets.size()), kq(kets.size());
  for (std::size_t i = 0; i < kets.size(); ++i) {
    kp[i] = kets[i].p.value();
    kq[i] = kets[i].q.value();
  }
  ket_p_ = upload(kp, "basis::ket_p");
  ket_q_ = upload(kq, "basis::ket_q");

  const auto chs = channels.channels();
  const std::size_t nc = chs.size();
  std::vector<std::int32_t> cj(nc), cp(nc), ct(nc), cn(nc), co(nc);
  for (std::size_t i = 0; i < nc; ++i) {
    cj[i] = chs[i].two_j.twice;
    cp[i] = static_cast<std::int32_t>(chs[i].parity);
    ct[i] = chs[i].two_tz.twice;
    cn[i] = chs[i].num_kets;
    co[i] = chs[i].ket_offset;
  }
  ch_two_j_ = upload(cj, "basis::ch_two_j");
  ch_parity_ = upload(cp, "basis::ch_parity");
  ch_two_tz_ = upload(ct, "basis::ch_two_tz");
  ch_num_kets_ = upload(cn, "basis::ch_num_kets");
  ch_offset_ = upload(co, "basis::ch_ket_offset");

  const auto ckt = channels.channel_ket_table();
  std::vector<std::int32_t> ck(ckt.size());
  std::transform(ckt.begin(), ckt.end(), ck.begin(), [](KetIndex k) { return k.value(); });
  ch_kets_ = upload(ck, "basis::channel_kets");
  ket_j_offsets_ = upload<std::int32_t>(channels.ket_J_offsets(), "basis::ket_J_offsets");
  local_index_ = upload<std::int32_t>(channels.local_index_table(), "basis::local_index");

  const auto st = subsets.subset_table();
  std::vector<std::int32_t> sk(st.size());
  std::transform(st.begin(), st.end(), sk.begin(), [](LocalKetIndex k) { return k.value(); });
  sub_offsets_ = upload<std::int32_t>(subsets.subset_offsets(), "basis::subset_offsets");
  sub_kets_ = upload(sk, "basis::subset_kets");
  sub_occ_ = upload<double>(subsets.occupation_product_table(), "basis::subset_occ_products");
  sub_vac_ = upload<double>(subsets.vacancy_product_table(), "basis::subset_vac_products");

  const auto ni = static_cast<std::int32_t>(no);
  orbits_view_ = {orb_n_.device_data(),
                  orb_l_.device_data(),
                  orb_j2_.device_data(),
                  orb_tz2_.device_data(),
                  orb_occ_.device_data(),
                  orb_cvq_.device_data(),
                  ni};
  two_body_view_.ket_p = ket_p_.device_data();
  two_body_view_.ket_q = ket_q_.device_data();
  two_body_view_.num_kets = channels.num_kets();
  two_body_view_.num_orbits = ni;
  two_body_view_.channel_two_j = ch_two_j_.device_data();
  two_body_view_.channel_parity = ch_parity_.device_data();
  two_body_view_.channel_two_tz = ch_two_tz_.device_data();
  two_body_view_.channel_num_kets = ch_num_kets_.device_data();
  two_body_view_.channel_ket_offset = ch_offset_.device_data();
  two_body_view_.num_channels = channels.num_channels();
  two_body_view_.channel_kets = ch_kets_.device_data();
  two_body_view_.ket_J_offsets = ket_j_offsets_.device_data();
  two_body_view_.local_index_table = local_index_.device_data();
  two_body_view_.orbit_j2 = orb_j2_.device_data();
  two_body_view_.orbit_l = orb_l_.device_data();
  two_body_view_.orbit_tz2 = orb_tz2_.device_data();
  subsets_view_ = {sub_offsets_.device_data(), sub_kets_.device_data(), sub_occ_.device_data(),
                   sub_vac_.device_data(), subsets.num_channels()};
}

}  // namespace imsrg::basis
