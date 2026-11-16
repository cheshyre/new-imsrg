// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/basis/device_tables.h"

#include <algorithm>
#include <span>
#include <vector>

namespace imsrg::basis {

namespace {

template <class T>
exec::DeviceBuffer<T> Upload(std::span<const T> src, const char* label) {
  exec::DeviceBuffer<T> buf(src.size(), label);
  if (!src.empty()) buf.CopyFromHost(src);
  return buf;
}

template <class T>
exec::DeviceBuffer<T> Upload(const std::vector<T>& src, const char* label) {
  return Upload<T>(std::span<const T>(src), label);
}

// Host copies of the per-orbit columns uploaded to the device.
struct OrbitColumns {
  std::vector<std::int32_t> n, l, j2, tz2, cvq;
};

// Builds the per-orbit columns (n, l, 2j, 2tz, core/valence/qspace class).
OrbitColumns MakeOrbitColumns(const SingleParticleBasis& basis,
                              const Partition& partition) {
  const auto orbs = basis.orbits();
  const std::size_t no = orbs.size();
  OrbitColumns c;
  c.n.resize(no);
  c.l.resize(no);
  c.j2.resize(no);
  c.tz2.resize(no);
  c.cvq.resize(no);
  for (std::size_t i = 0; i < no; ++i) {
    c.n[i] = orbs[i].n;
    c.l[i] = orbs[i].l;
    c.j2[i] = orbs[i].j2.twice;
    c.tz2[i] = orbs[i].tz2.twice;
    c.cvq[i] = static_cast<std::int32_t>(partition.classes()[i]);
  }
  return c;
}

// Host copies of the per-channel columns uploaded to the device.
struct ChannelColumns {
  std::vector<std::int32_t> two_j, parity, two_tz, num_kets, ket_offset;
};

// Builds the per-channel columns (2J, parity, 2Tz, ket count, ket offset).
ChannelColumns MakeChannelColumns(std::span<const TwoBodyChannel> chs) {
  const std::size_t nc = chs.size();
  ChannelColumns c;
  c.two_j.resize(nc);
  c.parity.resize(nc);
  c.two_tz.resize(nc);
  c.num_kets.resize(nc);
  c.ket_offset.resize(nc);
  for (std::size_t i = 0; i < nc; ++i) {
    c.two_j[i] = chs[i].two_j.twice;
    c.parity[i] = static_cast<std::int32_t>(chs[i].parity);
    c.two_tz[i] = chs[i].two_tz.twice;
    c.num_kets[i] = chs[i].num_kets;
    c.ket_offset[i] = chs[i].ket_offset;
  }
  return c;
}

}  // namespace

DeviceTables::DeviceTables(const SingleParticleBasis& basis,
                           const Reference& reference,
                           const Partition& partition,
                           const TwoBodyChannels& channels,
                           const TwoBodyKetSubsets& subsets) {
  const OrbitColumns oc = MakeOrbitColumns(basis, partition);
  orb_n_ = Upload(oc.n, "basis::orb_n");
  orb_l_ = Upload(oc.l, "basis::orb_l");
  orb_j2_ = Upload(oc.j2, "basis::orb_j2");
  orb_tz2_ = Upload(oc.tz2, "basis::orb_tz2");
  orb_cvq_ = Upload(oc.cvq, "basis::orb_cvq");
  orb_occ_ = Upload<double>(reference.occupations(), "basis::orb_occ");

  const std::size_t no = oc.n.size();
  const auto kets = channels.kets();
  std::vector<std::int32_t> kp(kets.size()), kq(kets.size());
  for (std::size_t i = 0; i < kets.size(); ++i) {
    kp[i] = kets[i].p.value();
    kq[i] = kets[i].q.value();
  }
  ket_p_ = Upload(kp, "basis::ket_p");
  ket_q_ = Upload(kq, "basis::ket_q");

  const ChannelColumns cc = MakeChannelColumns(channels.channels());
  ch_two_j_ = Upload(cc.two_j, "basis::ch_two_j");
  ch_parity_ = Upload(cc.parity, "basis::ch_parity");
  ch_two_tz_ = Upload(cc.two_tz, "basis::ch_two_tz");
  ch_num_kets_ = Upload(cc.num_kets, "basis::ch_num_kets");
  ch_offset_ = Upload(cc.ket_offset, "basis::ch_ket_offset");

  const auto ckt = channels.channel_ket_table();
  std::vector<std::int32_t> ck(ckt.size());
  std::transform(ckt.begin(), ckt.end(), ck.begin(),
                 [](KetIndex k) { return k.value(); });
  ch_kets_ = Upload(ck, "basis::channel_kets");
  ket_j_offsets_ =
      Upload<std::int32_t>(channels.ket_J_offsets(), "basis::ket_J_offsets");
  local_index_ =
      Upload<std::int32_t>(channels.local_index_table(), "basis::local_index");

  const auto st = subsets.subset_table();
  std::vector<std::int32_t> sk(st.size());
  std::transform(st.begin(), st.end(), sk.begin(),
                 [](LocalKetIndex k) { return k.value(); });
  sub_offsets_ =
      Upload<std::int32_t>(subsets.subset_offsets(), "basis::subset_offsets");
  sub_kets_ = Upload(sk, "basis::subset_kets");
  sub_occ_ = Upload<double>(subsets.occupation_product_table(),
                            "basis::subset_occ_products");
  sub_vac_ = Upload<double>(subsets.vacancy_product_table(),
                            "basis::subset_vac_products");

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
  subsets_view_ = {sub_offsets_.device_data(), sub_kets_.device_data(),
                   sub_occ_.device_data(), sub_vac_.device_data(),
                   subsets.num_channels()};
}

}  // namespace imsrg::basis
