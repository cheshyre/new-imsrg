// Tests of the ModelSpace bundle: the imsrg++-driver construction path against every golden file,
// sharing of the occupation-independent parts, the reference-changing derivations, and errors.
#include <memory>
#include <string>
#include <vector>

#include "golden_basis_dump.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::basis;
using imsrg::basis::testing::golden_basis_dir;
using imsrg::basis::testing::golden_basis_files;
using imsrg::basis::testing::read_golden_basis_dump;

std::shared_ptr<const ModelSpace> o16_emax2() {
  return ModelSpace::from_names(BasisTruncation{2}, 16.0, "O16", "O16");
}

// from_names reproduces the golden orbits (occupation, cvq), counts and 2B channel sizes; the
// record-by-record comparison of all structures is in test_golden_basis.cpp.
TEST(BasisModelSpace, FromNamesMatchesEveryGoldenFile) {
  for (const std::string& name : golden_basis_files()) {
    SCOPED_TRACE(name);
    const auto g = read_golden_basis_dump(golden_basis_dir() / name);
    const auto ms =
        ModelSpace::from_names(BasisTruncation{g.emax}, g.hw, g.reference, g.valence_space);
    ASSERT_EQ(ms->num_orbits(), static_cast<int>(g.orbits.size()));
    for (const auto& go : g.orbits) {
      const OrbitIndex a{go.index};
      EXPECT_EQ(ms->reference().occupation(a), go.occupation) << "orbit " << go.index;
      EXPECT_EQ(cvq(ms->partition().orbit_class(a)), go.cvq) << "orbit " << go.index;
    }
    EXPECT_EQ(ms->reference().A(), g.Aref);
    EXPECT_EQ(ms->reference().Z(), g.Zref);
    EXPECT_EQ(ms->partition().A_core(), g.Acore);
    EXPECT_EQ(ms->partition().Z_core(), g.Zcore);
    EXPECT_EQ(ms->reference().target_mass(), g.target_mass);
    EXPECT_EQ(ms->reference().target_Z(), g.target_Z);
    ASSERT_EQ(ms->two_body().num_channels(), static_cast<int>(g.channels.size()));
    for (const auto& gc : g.channels) {
      const ChannelIndex ch{gc.ch};
      EXPECT_EQ(ms->two_body().channel(ch).num_kets, gc.nkets) << "channel " << gc.ch;
      EXPECT_EQ(static_cast<int>(ms->two_body_subsets().kets(ch, KetClass::HH).size()),
                static_cast<int>(gc.subsets.at("hh").size()))
          << "channel " << gc.ch;
    }
  }
}

TEST(BasisModelSpace, WithReferenceSharesBasisAndChannels) {
  const auto ms = o16_emax2();
  const auto ne20 = Reference::from_nucleus(ms->basis(), Nucleus::parse("Ne20"));
  const auto ms2 = ms->with_reference(ne20);
  EXPECT_EQ(&ms2->basis(), &ms->basis());
  EXPECT_EQ(&ms2->two_body(), &ms->two_body());
  EXPECT_EQ(ms2->partition(), ms->partition());
  EXPECT_EQ(ms2->reference(), ne20);
  EXPECT_FALSE(*ms2 == *ms);
  // The subsets follow the new occupations: the d5/2 orbits (6, 7) are now holes.
  const ChannelIndex ch0{0};  // J=0, even, Tz=-1: kets 0 10 23 42 57 68 75 -> (0,0) ... (6,6)
  const auto hh = ms2->two_body_subsets().kets(ch0, KetClass::HH);
  EXPECT_EQ(hh.size(), 4u);  // (0,0), (2,2), (4,4), (6,6)
  EXPECT_EQ(ms->two_body_subsets().kets(ch0, KetClass::HH).size(), 3u);
  // The original is unchanged.
  EXPECT_EQ(ms->reference().name(), "O16");
}

TEST(BasisModelSpace, WithReferenceAndPartitionAppliesHolesAsCore) {
  const auto ms = ModelSpace::from_names(BasisTruncation{2}, 16.0, "Ne20", "sd-shell");
  const auto o16 = Reference::from_nucleus(ms->basis(), Nucleus::parse("O16"));
  const auto part = Partition::with_holes_as_core(ms->partition(), o16, ms->basis());
  const auto ms2 = ms->with_reference_and_partition(o16, part);
  EXPECT_EQ(&ms2->two_body(), &ms->two_body());
  EXPECT_EQ(ms2->partition().core().size(), 6u);
  EXPECT_EQ(ms2->partition().valence().size(), 6u);
  EXPECT_EQ(ms2->reference().A(), 16.0);
}

TEST(BasisModelSpace, EqualityIsStructural) {
  const auto a = o16_emax2();
  const auto b = o16_emax2();
  EXPECT_NE(a.get(), b.get());
  EXPECT_TRUE(*a == *b);
  const auto c = ModelSpace::from_names(BasisTruncation{2}, 16.0, "O16", "sd-shell");
  EXPECT_FALSE(*a == *c);  // same occupations, different partition
  const auto d = ModelSpace::from_names(BasisTruncation{2}, 20.0, "O16", "O16");
  EXPECT_FALSE(*a == *d);  // different hbar*omega
}

TEST(BasisModelSpace, SizeMismatchThrows) {
  const auto ms = o16_emax2();
  const auto big = SingleParticleBasis::harmonic_oscillator(BasisTruncation{4}, 16.0);
  const auto ref4 = Reference::from_nucleus(big, Nucleus::parse("O16"));
  EXPECT_THROW((void)ms->with_reference(ref4), InputError);
  EXPECT_THROW((void)ModelSpace::create(ms->basis(), ref4, ms->partition()), InputError);
}

TEST(BasisModelSpace, BadNamesThrow) {
  EXPECT_THROW((void)ModelSpace::from_names(BasisTruncation{2}, 16.0, "Xx16", "O16"), InputError);
  EXPECT_THROW((void)ModelSpace::from_names(BasisTruncation{2}, 16.0, "default", "0hw-shell"),
               InputError);
  EXPECT_THROW((void)ModelSpace::from_names(BasisTruncation{2}, 16.0, "Ca48", "Ca48"), InputError);
}

}  // namespace
