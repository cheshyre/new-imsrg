// Self-consistency of the golden basis files (imsrg-basis-dump v1) through the strict parser, and
// agreement of the harmonic-oscillator orbit list with them.
#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "golden_basis_dump.hpp"
#include <gtest/gtest.h>

#include "imsrg/basis/single_particle_basis.hpp"

namespace imsrg::basis::testing {
namespace {

// True if the union of the index lists is exactly {0..n-1} with no index twice.
bool is_partition(const std::vector<const std::vector<int>*>& parts, int n) {
  std::vector<int> all;
  for (const auto* p : parts) all.insert(all.end(), p->begin(), p->end());
  std::sort(all.begin(), all.end());
  if (static_cast<int>(all.size()) != n) return false;
  for (int i = 0; i < n; ++i) {
    if (all[static_cast<std::size_t>(i)] != i) return false;
  }
  return true;
}

void check_dump(const GoldenBasisDump& d) {
  const int norb = static_cast<int>(d.orbits.size());
  ASSERT_GT(norb, 0);
  EXPECT_EQ(norb, (d.emax + 1) * (d.emax + 2));
  EXPECT_EQ(d.hw, 16.0);

  int max_j2 = 0;
  for (int i = 0; i < norb; ++i) {
    const GoldenOrbit& o = d.orbits[static_cast<std::size_t>(i)];
    EXPECT_EQ(o.index, i);
    EXPECT_TRUE(o.j2 == 2 * o.l + 1 || o.j2 == 2 * o.l - 1);
    EXPECT_TRUE(o.tz2 == -1 || o.tz2 == 1);
    EXPECT_GE(o.occupation, 0.0);
    EXPECT_LE(o.occupation, 1.0);
    max_j2 = std::max(max_j2, o.j2);
  }

  // holes/particles partition the orbits, and follow the occupation cut of imsrg++.
  EXPECT_TRUE(is_partition({&d.holes, &d.particles}, norb));
  for (const int h : d.holes) EXPECT_GE(d.orbits[static_cast<std::size_t>(h)].occupation, 1e-6);
  for (const int p : d.particles) EXPECT_LT(d.orbits[static_cast<std::size_t>(p)].occupation, 1e-6);

  // core/valence/qspace partition the orbits consistently with cvq.
  EXPECT_TRUE(is_partition({&d.core, &d.valence, &d.qspace}, norb));
  for (const int i : d.core) EXPECT_EQ(d.orbits[static_cast<std::size_t>(i)].cvq, 0);
  for (const int i : d.valence) EXPECT_EQ(d.orbits[static_cast<std::size_t>(i)].cvq, 1);
  for (const int i : d.qspace) EXPECT_EQ(d.orbits[static_cast<std::size_t>(i)].cvq, 2);

  // Kets: all p <= q, index = Index2(p, q) = p (2N - 1 - p) / 2 + q, complete.
  ASSERT_EQ(static_cast<int>(d.kets.size()), norb * (norb + 1) / 2);
  for (std::size_t k = 0; k < d.kets.size(); ++k) {
    const auto [p, q] = d.kets[k];
    ASSERT_LE(p, q);
    ASSERT_GE(p, 0);
    ASSERT_LT(q, norb);
    EXPECT_EQ(static_cast<int>(k), p * (2 * norb - 1 - p) / 2 + q);
  }

  // Two-body channels: 6 (Jmax + 1), ch = 6J + 2(Tz + 1) + parity, ascending kets, subsets.
  ASSERT_EQ(static_cast<int>(d.channels.size()), 6 * (max_j2 + 1));
  int total_kets = 0;
  for (std::size_t c = 0; c < d.channels.size(); ++c) {
    const GoldenChannel& ch = d.channels[c];
    EXPECT_EQ(ch.ch, static_cast<int>(c));
    EXPECT_EQ(ch.ch, 6 * ch.J + 2 * (ch.Tz + 1) + ch.parity);
    EXPECT_TRUE(ch.parity == 0 || ch.parity == 1);
    EXPECT_GE(ch.Tz, -1);
    EXPECT_LE(ch.Tz, 1);
    EXPECT_EQ(ch.nkets, static_cast<int>(ch.kets.size()));
    EXPECT_TRUE(std::is_sorted(ch.kets.begin(), ch.kets.end()));
    EXPECT_TRUE(std::adjacent_find(ch.kets.begin(), ch.kets.end()) == ch.kets.end());
    total_kets += ch.nkets;
    for (const int k : ch.kets) {
      ASSERT_GE(k, 0);
      ASSERT_LT(k, static_cast<int>(d.kets.size()));
      const auto [p, q] = d.kets[static_cast<std::size_t>(k)];
      const GoldenOrbit& op = d.orbits[static_cast<std::size_t>(p)];
      const GoldenOrbit& oq = d.orbits[static_cast<std::size_t>(q)];
      EXPECT_EQ((op.tz2 + oq.tz2) / 2, ch.Tz);
      EXPECT_EQ((op.l + oq.l) % 2, ch.parity);
      // J couples |jp - jq| <= 2J <= jp + jq (in units of 1/2), same parity of 2J + jp + jq.
      EXPECT_LE(std::abs(op.j2 - oq.j2), 2 * ch.J);
      EXPECT_LE(2 * ch.J, op.j2 + oq.j2);
    }
    ASSERT_EQ(ch.subsets.size(), 9u);
    for (const char* name : {"pp", "hh", "ph", "cc", "vc", "qc", "vv", "qv", "qq"}) {
      ASSERT_TRUE(ch.subsets.count(name)) << name;
      const auto& s = ch.subsets.at(name);
      EXPECT_TRUE(std::is_sorted(s.begin(), s.end())) << name;
    }
    EXPECT_TRUE(
        is_partition({&ch.subsets.at("pp"), &ch.subsets.at("hh"), &ch.subsets.at("ph")}, ch.nkets))
        << "pp/hh/ph, channel " << ch.ch;
    EXPECT_TRUE(is_partition({&ch.subsets.at("cc"), &ch.subsets.at("vc"), &ch.subsets.at("qc"),
                              &ch.subsets.at("vv"), &ch.subsets.at("qv"), &ch.subsets.at("qq")},
                             ch.nkets))
        << "cvq classes, channel " << ch.ch;
  }
  // Every ket is in exactly one channel or in none (J out of the allowed range); never in two.
  EXPECT_LE(total_kets, 6 * (max_j2 + 1) * static_cast<int>(d.kets.size()));

  // Cross-coupled channels.
  EXPECT_EQ(d.ph_channels.size(), d.channels.size());
  for (std::size_t c = 0; c < d.ph_channels.size(); ++c) {
    const GoldenPhChannel& ch = d.ph_channels[c];
    EXPECT_EQ(ch.ch, static_cast<int>(c));
    EXPECT_EQ(ch.ch, 6 * ch.J + 2 * (ch.Tz + 1) + ch.parity);
    EXPECT_EQ(ch.nkets, static_cast<int>(ch.kets.size()));
    EXPECT_TRUE(std::is_sorted(ch.kets.begin(), ch.kets.end()));
    for (const int k : ch.kets) {
      ASSERT_GE(k, 0);
      ASSERT_LT(k, static_cast<int>(d.kets.size()));
    }
  }

  // Three-body channels.
  long long sum3 = 0;
  for (std::size_t c = 0; c < d.channels3.size(); ++c) {
    EXPECT_EQ(d.channels3[c].ch, static_cast<int>(c));
    EXPECT_GE(d.channels3[c].nkets, 0);
    sum3 += d.channels3[c].nkets;
  }
  EXPECT_EQ(sum3, d.sum_channel_kets3);
  EXPECT_GT(d.nkets3, 0);
}

TEST(GoldenBasisDump, ThereAreTwelveFiles) {
  const auto files = golden_basis_files();
  ASSERT_EQ(files.size(), 12u);
  EXPECT_TRUE(std::is_sorted(files.begin(), files.end()));
  for (const auto& f : files) {
    EXPECT_TRUE(std::filesystem::exists(golden_basis_dir() / f)) << f;
  }
}

TEST(GoldenBasisDump, SelfConsistentAndMatchesHarmonicOscillatorBasis) {
  for (const auto& name : golden_basis_files()) {
    SCOPED_TRACE(name);
    const GoldenBasisDump d = read_golden_basis_dump(golden_basis_dir() / name);
    check_dump(d);
    EXPECT_TRUE(d.source.find("imsrg-devel") != std::string::npos);

    const auto spb = SingleParticleBasis::harmonic_oscillator(BasisTruncation{d.emax}, d.hw);
    ASSERT_EQ(spb.num_orbits(), static_cast<int>(d.orbits.size()));
    for (std::size_t i = 0; i < d.orbits.size(); ++i) {
      const Orbit& o = spb.orbits()[i];
      EXPECT_EQ(o.n, d.orbits[i].n) << i;
      EXPECT_EQ(o.l, d.orbits[i].l) << i;
      EXPECT_EQ(o.j2.twice, d.orbits[i].j2) << i;
      EXPECT_EQ(o.tz2.twice, d.orbits[i].tz2) << i;
    }
    EXPECT_EQ(static_cast<int>(d.channels.size()), 6 * (spb.max_j2().twice + 1));
  }
}

TEST(GoldenBasisDump, KnownContentOfNe20Emax2) {
  const GoldenBasisDump d = read_golden_basis_dump(golden_basis_dir() / "emax2_Ne20.txt");
  EXPECT_EQ(d.emax, 2);
  EXPECT_EQ(d.e3max, 12);
  EXPECT_EQ(d.reference, "Ne20");
  EXPECT_EQ(d.valence_space, "Ne20");
  EXPECT_EQ(d.Aref, 20.0);
  EXPECT_EQ(d.Zref, 10.0);
  EXPECT_EQ(d.target_mass, 20);
  EXPECT_EQ(d.target_Z, 10);
  ASSERT_EQ(d.orbits.size(), 12u);
  EXPECT_EQ(d.orbits[6].occupation, 0.33333333333333331);  // 0d5/2 proton, exact round trip
}

// ---- strictness of the parser ---------------------------------------------------------------

std::vector<std::string> read_lines(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) lines.push_back(l);
  return lines;
}

std::filesystem::path write_lines(const std::vector<std::string>& lines, const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / "imsrg_golden_dump_test";
  std::filesystem::create_directories(dir);
  const auto path = dir / name;
  std::ofstream out(path);
  for (const auto& l : lines) out << l << '\n';
  return path;
}

// Expects the parser to throw a runtime_error whose message names the file and line `line`.
void expect_rejected(const std::vector<std::string>& lines, const std::string& file, int line,
                     const std::string& reason_part) {
  const auto path = write_lines(lines, file);
  try {
    (void)read_golden_basis_dump(path);
    ADD_FAILURE() << "no exception for " << file;
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find(file), std::string::npos) << msg;
    EXPECT_NE(msg.find(":" + std::to_string(line) + ":"), std::string::npos) << msg;
    EXPECT_NE(msg.find(reason_part), std::string::npos) << msg;
  }
  std::filesystem::remove(path);
}

TEST(GoldenBasisDump, RejectsMalformedInput) {
  const auto good = read_lines(golden_basis_dir() / "emax2_He4.txt");
  ASSERT_GT(good.size(), 100u);

  auto find_line = [&](const std::string& prefix) {
    for (std::size_t i = 0; i < good.size(); ++i) {
      if (good[i].rfind(prefix, 0) == 0) return i;
    }
    ADD_FAILURE() << prefix;
    return std::size_t{0};
  };

  {  // wrong first line
    auto l = good;
    l[0] = "# imsrg-basis-dump v2";
    expect_rejected(l, "bad_first.txt", 1, "first line");
  }
  {  // fewer orbit records than norbits
    auto l = good;
    const std::size_t i = find_line("orbit 11 ");
    l.erase(l.begin() + static_cast<std::ptrdiff_t>(i));
    expect_rejected(l, "short_orbits.txt", static_cast<int>(i) + 1, "holes");
  }
  {  // holes count != number of indices
    auto l = good;
    const std::size_t i = find_line("holes ");
    l[i] = "holes 3 0 1";
    expect_rejected(l, "bad_holes.txt", static_cast<int>(i) + 1, "count");
  }
  {  // channel nkets != channel_kets entries
    auto l = good;
    const std::size_t i = find_line("channel 0 ");
    l[i] = "channel 0 0 0 -1 99";
    expect_rejected(l, "bad_nkets.txt", static_cast<int>(i) + 2, "kets");
  }
  {  // records out of order (ket before nkets)
    auto l = good;
    const std::size_t i = find_line("nkets ");
    std::swap(l[i], l[i + 1]);
    expect_rejected(l, "bad_order.txt", static_cast<int>(i) + 1, "expected record 'nkets'");
  }
  {  // not a number
    auto l = good;
    const std::size_t i = find_line("basis ");
    l[i] = "basis emax two e2max 4 e3max 12 lmax 2 hw 16";
    expect_rejected(l, "bad_number.txt", static_cast<int>(i) + 1, "not an integer");
  }
  {  // missing end
    auto l = good;
    l.pop_back();
    expect_rejected(l, "no_end.txt", static_cast<int>(l.size()) + 1, "end");
  }
  {  // content after end
    auto l = good;
    l.push_back("junk");
    expect_rejected(l, "after_end.txt", static_cast<int>(l.size()), "after 'end'");
  }
  {  // 3B sum record with a wrong field name
    auto l = good;
    const std::size_t i = find_line("nkets3 ");
    l[i] = "nkets3 5 sum 7";
    expect_rejected(l, "bad_nkets3.txt", static_cast<int>(i) + 1, "sum_channel_kets3");
  }
  EXPECT_THROW((void)read_golden_basis_dump(golden_basis_dir() / "does_not_exist.txt"),
               std::runtime_error);
}

}  // namespace
}  // namespace imsrg::basis::testing
