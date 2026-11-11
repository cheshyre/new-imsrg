// Record-by-record comparison of the basis bookkeeping (orbits, reference, partition, two-body
// channels and ket subsets) with every imsrg-basis-dump v1 golden file. Only the header records
// of each file are used to build our objects; every comparison is exact (integers equal, doubles
// with ==, since the files carry %.17g round-trip values). The ph channels and the three-body
// records are compared by later tasks.
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "golden_basis_dump.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/partition.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/basis/two_body_channels.hpp"
#include "imsrg/basis/valence_space.hpp"

namespace imsrg::basis {
namespace {

using testing::GoldenBasisDump;
using testing::read_golden_basis_dump;

// Golden subset names in the order of KetClass (PP HH PH CC VC QC VV QV QQ).
constexpr const char* kSubsetNames[kNumKetClasses] = {"pp", "hh", "ph", "cc", "vc",
                                                      "qc", "vv", "qv", "qq"};

// Everything built from the header records of one golden file.
struct Built {
  SingleParticleBasis basis;
  Reference ref;
  Partition part;
  TwoBodyChannels ch;
  TwoBodyKetSubsets sub;
};

Built build(const GoldenBasisDump& g) {
  SingleParticleBasis basis =
      SingleParticleBasis::harmonic_oscillator(BasisTruncation{g.emax}, g.hw);
  const Nucleus nucleus = g.reference == "default" ? default_reference_nucleus(g.valence_space)
                                                   : Nucleus::parse(g.reference);
  Reference ref = Reference::from_nucleus(basis, nucleus);
  const ValenceSpaceDefinition def = define_valence_space(g.valence_space, basis, nucleus);
  Partition part = Partition::from_definition(basis, def);
  TwoBodyChannels ch(basis);
  TwoBodyKetSubsets sub(ch, ref, part);
  return Built{std::move(basis), std::move(ref), std::move(part), std::move(ch), std::move(sub)};
}

// Index lists of a span of strong indices.
template <class Span>
std::vector<int> values_of(const Span& s) {
  std::vector<int> v;
  for (const auto& x : s) v.push_back(static_cast<int>(x.value()));
  return v;
}

class GoldenBasis : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    path_ = testing::golden_basis_dir() / GetParam();
    ASSERT_NO_THROW(g_ = read_golden_basis_dump(path_)) << path_;
    ASSERT_NO_THROW(b_ = std::make_unique<Built>(build(g_))) << GetParam();
  }
  std::filesystem::path path_;
  GoldenBasisDump g_;
  std::unique_ptr<Built> b_;
};

TEST_P(GoldenBasis, HeaderAndCounts) {
  EXPECT_EQ(g_.lmax, b_->basis.lmax()) << "record 'basis' lmax";
  EXPECT_EQ(g_.emax, b_->basis.emax()) << "record 'basis' emax";
  EXPECT_EQ(g_.Aref, b_->ref.A()) << "record 'counts' Aref";
  EXPECT_EQ(g_.Zref, b_->ref.Z()) << "record 'counts' Zref";
  EXPECT_EQ(g_.Acore, b_->part.A_core()) << "record 'counts' Acore";
  EXPECT_EQ(g_.Zcore, b_->part.Z_core()) << "record 'counts' Zcore";
  EXPECT_EQ(g_.target_mass, b_->ref.target_mass()) << "record 'counts' target_mass";
  EXPECT_EQ(g_.target_Z, b_->ref.target_Z()) << "record 'counts' target_Z";
}

TEST_P(GoldenBasis, Orbits) {
  ASSERT_EQ(static_cast<int>(g_.orbits.size()), b_->basis.num_orbits()) << "record 'norbits'";
  for (const auto& go : g_.orbits) {
    ASSERT_EQ(go.index, static_cast<int>(&go - g_.orbits.data())) << "orbit record order";
    const OrbitIndex a{go.index};
    const Orbit& o = b_->basis.orbit(a);
    const std::string where = "orbit " + std::to_string(go.index);
    ASSERT_EQ(go.n, o.n) << where << " n";
    ASSERT_EQ(go.l, o.l) << where << " l";
    ASSERT_EQ(go.j2, o.j2.twice) << where << " 2j";
    ASSERT_EQ(go.tz2, o.tz2.twice) << where << " 2tz";
    ASSERT_EQ(go.occupation, b_->ref.occupation(a)) << where << " occupation";
    ASSERT_EQ(go.cvq, cvq(b_->part.orbit_class(a))) << where << " cvq";
  }
}

TEST_P(GoldenBasis, OrbitLists) {
  EXPECT_EQ(g_.holes, values_of(b_->ref.holes())) << "record 'holes'";
  EXPECT_EQ(g_.particles, values_of(b_->ref.particles())) << "record 'particles'";
  EXPECT_EQ(g_.core, values_of(b_->part.core())) << "record 'core'";
  EXPECT_EQ(g_.valence, values_of(b_->part.valence())) << "record 'valence'";
  EXPECT_EQ(g_.qspace, values_of(b_->part.qspace())) << "record 'qspace'";
}

TEST_P(GoldenBasis, Kets) {
  ASSERT_EQ(static_cast<int>(g_.kets.size()), b_->ch.num_kets()) << "record 'nkets'";
  for (std::size_t k = 0; k < g_.kets.size(); ++k) {
    const Ket& ket = b_->ch.ket(KetIndex{static_cast<int>(k)});
    ASSERT_EQ(g_.kets[k].first, ket.p.value()) << "ket " << k << " p";
    ASSERT_EQ(g_.kets[k].second, ket.q.value()) << "ket " << k << " q";
  }
}

TEST_P(GoldenBasis, TwoBodyChannels) {
  ASSERT_EQ(static_cast<int>(g_.channels.size()), b_->ch.num_channels())
      << "record 'ntwobody_channels'";
  for (const auto& gc : g_.channels) {
    const std::string where = "channel " + std::to_string(gc.ch);
    const ChannelIndex c{gc.ch};
    const TwoBodyChannel& ours = b_->ch.channel(c);
    ASSERT_EQ(gc.J, ours.J()) << where << " J";
    ASSERT_EQ(gc.parity, static_cast<int>(ours.parity)) << where << " parity";
    ASSERT_EQ(gc.Tz, ours.Tz()) << where << " Tz";
    ASSERT_EQ(gc.nkets, ours.num_kets) << where << " nkets";
    // The global ket list in local order.
    ASSERT_EQ(gc.kets, values_of(b_->ch.channel_kets(c))) << where << " channel_kets";
  }
}

TEST_P(GoldenBasis, KetSubsets) {
  ASSERT_EQ(static_cast<int>(g_.channels.size()), b_->sub.num_channels());
  for (const auto& gc : g_.channels) {
    const ChannelIndex c{gc.ch};
    for (const KetClass k : kAllKetClasses) {
      const std::string name = kSubsetNames[static_cast<int>(k)];
      const auto it = gc.subsets.find(name);
      ASSERT_NE(it, gc.subsets.end()) << "channel " << gc.ch << " has no subset " << name;
      ASSERT_EQ(it->second, values_of(b_->sub.kets(c, k)))
          << "channel " << gc.ch << " (J=" << gc.J << " parity=" << gc.parity << " Tz=" << gc.Tz
          << ") record 'channel_subset' " << name;
    }
  }
}

std::string case_name(const ::testing::TestParamInfo<std::string>& info) {
  std::string s = info.param.substr(0, info.param.size() - 4);  // strip ".txt"
  for (char& c : s) {
    if (c == '-') c = '_';
  }
  return s;
}

INSTANTIATE_TEST_SUITE_P(Files, GoldenBasis, ::testing::ValuesIn(testing::golden_basis_files()),
                         case_name);

}  // namespace
}  // namespace imsrg::basis
