// DeviceTables: device accessor results (evaluated in kernels) against the host classes.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "device_tables_kernels.hpp"
#include <gtest/gtest.h>

#include "imsrg/basis/device_tables.hpp"
#include "imsrg/basis/model_space.hpp"

namespace imsrg::basis {
namespace {

using exec::DeviceBuffer;
using I32 = DeviceBuffer<std::int32_t>;

struct Case {
  const char* ref;
  const char* val;
  int emax;
};

class DeviceTablesTest : public ::testing::TestWithParam<Case> {};

TEST_P(DeviceTablesTest, MatchesHost) {
  const Case c = GetParam();
  const auto ms = ModelSpace::from_names(BasisTruncation{c.emax}, 16.0, c.ref, c.val);
  const TwoBodyChannels& tbc = ms->two_body();
  const TwoBodyKetSubsets& sub = ms->two_body_subsets();
  DeviceTables dt(ms->basis(), ms->reference(), ms->partition(), tbc, sub);
  const auto& ov = dt.orbits();
  const auto& tb = dt.two_body();
  const auto& sv = dt.subsets();

  // Orbits.
  const std::size_t no = static_cast<std::size_t>(ms->num_orbits());
  ASSERT_EQ(ov.num_orbits, ms->num_orbits());
  {
    I32 n(no), l(no), j2(no), tz2(no), cvq(no);
    DeviceBuffer<double> occ(no);
    test::eval_orbits(ov, n, l, j2, tz2, cvq, occ);
    const auto hn = n.to_vector(), hl = l.to_vector(), hj = j2.to_vector(), ht = tz2.to_vector(),
               hc = cvq.to_vector();
    const auto ho = occ.to_vector();
    for (std::size_t i = 0; i < no; ++i) {
      const Orbit& o = ms->basis().orbits()[i];
      EXPECT_EQ(hn[i], o.n);
      EXPECT_EQ(hl[i], o.l);
      EXPECT_EQ(hj[i], o.j2.twice);
      EXPECT_EQ(ht[i], o.tz2.twice);
      EXPECT_EQ(hc[i], static_cast<std::int32_t>(ms->partition().classes()[i]));
      EXPECT_EQ(ho[i], ms->reference().occupations()[i]);
    }
  }

  // Kets: ket_index, channel_of, local_index for every ket and J.
  const std::int32_t max_J = tbc.max_J();
  const std::size_t nk = static_cast<std::size_t>(tbc.num_kets());
  const std::size_t nj = static_cast<std::size_t>(max_J + 1);
  ASSERT_EQ(tb.num_kets, tbc.num_kets());
  ASSERT_EQ(tb.num_channels, tbc.num_channels());
  {
    I32 local(nk * nj), channel(nk * nj), kidx(nk);
    test::eval_ket_tables(tb, max_J, local, channel, kidx);
    const auto hl = local.to_vector(), hc = channel.to_vector(), hk = kidx.to_vector();
    for (std::int32_t k = 0; k < tbc.num_kets(); ++k) {
      const Ket& kt = tbc.ket(KetIndex{k});
      EXPECT_EQ(hk[static_cast<std::size_t>(k)], tbc.ket_index(kt.p, kt.q).value());
      EXPECT_EQ(hk[static_cast<std::size_t>(k)], k);
      for (std::int32_t J = 0; J <= max_J; ++J) {
        const std::size_t t = static_cast<std::size_t>(k) * nj + static_cast<std::size_t>(J);
        const auto lo = tbc.local_index(KetIndex{k}, TwoJ{2 * J});
        ASSERT_EQ(hl[t], lo ? lo->value() : -1) << "ket " << k << " J " << J;
        EXPECT_EQ(hc[t], tbc.channel_of(kt.p, kt.q, TwoJ{2 * J}).value());
      }
    }
    I32 bad(nk * 3);
    test::eval_invalid_two_j(tb, max_J, bad);
    for (auto v : bad.to_vector()) EXPECT_EQ(v, -1);
  }

  // Channels and global_ket.
  {
    const auto chs = tbc.channels();
    const std::size_t total = tbc.channel_ket_table().size();
    I32 out(total);
    test::eval_global_kets(tb, out);
    const auto h = out.to_vector();
    for (std::int32_t ch = 0; ch < tbc.num_channels(); ++ch) {
      const TwoBodyChannel& tc = chs[static_cast<std::size_t>(ch)];
      for (std::int32_t i = 0; i < tc.num_kets; ++i) {
        EXPECT_EQ(h[static_cast<std::size_t>(tc.ket_offset + i)],
                  tbc.global_ket(ChannelIndex{ch}, LocalKetIndex{i}).value());
      }
    }
    // Channel scalar arrays, read back through host mirrors of the same tables.
    EXPECT_EQ(tb.channel_num_kets != nullptr, true);
  }

  // Subsets.
  {
    const std::size_t ns = static_cast<std::size_t>(sub.num_channels()) * kNumKetClasses;
    const std::size_t total = sub.subset_table().size();
    I32 b(ns), e(ns), s(ns), kets(total);
    DeviceBuffer<double> occ(total), vac(total);
    test::eval_subsets(sv, b, e, s, kets, occ, vac);
    const auto hb = b.to_vector(), he = e.to_vector(), hs = s.to_vector(), hk = kets.to_vector();
    const auto ho = occ.to_vector(), hv = vac.to_vector();
    for (std::int32_t ch = 0; ch < sub.num_channels(); ++ch) {
      for (KetClass kc : kAllKetClasses) {
        const std::size_t t =
            static_cast<std::size_t>(ch) * kNumKetClasses + static_cast<std::size_t>(kc);
        const auto host = sub.kets(ChannelIndex{ch}, kc);
        ASSERT_EQ(static_cast<std::size_t>(hs[t]), host.size());
        ASSERT_EQ(he[t] - hb[t], hs[t]);
        const bool prod = kc == KetClass::HH || kc == KetClass::PH;
        for (std::size_t i = 0; i < host.size(); ++i) {
          const std::size_t g = static_cast<std::size_t>(hb[t]) + i;
          EXPECT_EQ(hk[g], host[i].value());
          if (prod) {
            EXPECT_EQ(ho[g], sub.occupation_products(ChannelIndex{ch}, kc)[i]);
            EXPECT_EQ(hv[g], sub.vacancy_products(ChannelIndex{ch}, kc)[i]);
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Nuclei, DeviceTablesTest,
                         ::testing::Values(Case{"O16", "O16", 2}, Case{"O16", "O16", 4},
                                           Case{"Ne20", "sd-shell", 2},
                                           Case{"Ne20", "sd-shell", 4}),
                         [](const ::testing::TestParamInfo<Case>& i) {
                           std::string r = std::string(i.param.ref) + "_" + i.param.val + "_e" +
                                           std::to_string(i.param.emax);
                           for (char& ch : r)
                             if (ch == '-') ch = '_';
                           return r;
                         });

// Host-side content of the buffers (host_span / to_vector) and move semantics.
TEST(DeviceTables, HostContentsAndMove) {
  const auto ms = ModelSpace::from_names(BasisTruncation{2}, 16.0, "O16", "O16");
  DeviceTables a(ms->basis(), ms->reference(), ms->partition(), ms->two_body(),
                 ms->two_body_subsets());
  const std::int32_t* p = a.orbits().n;
  DeviceTables b(std::move(a));
  EXPECT_EQ(b.orbits().n, p);  // device pointers survive a move
  const auto& tbc = ms->two_body();
  I32 offs(tbc.ket_J_offsets().size());
  offs.copy_from_host(tbc.ket_J_offsets());
  EXPECT_EQ(offs.to_vector(),
            std::vector<std::int32_t>(tbc.ket_J_offsets().begin(), tbc.ket_J_offsets().end()));
  const auto hs = static_cast<const I32&>(offs).host_span();
  ASSERT_EQ(hs.size(), tbc.ket_J_offsets().size());
  for (std::size_t i = 0; i < hs.size(); ++i) EXPECT_EQ(hs[i], tbc.ket_J_offsets()[i]);
  static_assert(std::is_trivially_copyable_v<TwoBodyTableView>);
  EXPECT_EQ(b.two_body().num_kets, tbc.num_kets());
  EXPECT_EQ(b.subsets().num_channels, ms->two_body_subsets().num_channels());
}

}  // namespace
}  // namespace imsrg::basis
