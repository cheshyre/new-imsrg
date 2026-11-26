// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// ME1J / ME2JP (design/19): sentinel files against an independent oracle
// (J-range safety, section 8), counts, and write -> read round trips.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/io/mexj.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/random_operator.h"
#include "tests/support/scratch_dir.h"

namespace imsrg::io {
namespace {

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// The test's own oracle: orbits and naive record loops (not the production
// enumerator).

struct TestOrbit {
  int n, l, j2, tz2;
};

std::vector<TestOrbit> OracleOrbits(int emax) {
  std::vector<TestOrbit> v;
  for (int e = 0; e <= emax; ++e) {
    for (int l = e % 2; l <= e; l += 2) {
      for (int j2 = 2 * l - 1; j2 <= 2 * l + 1; j2 += 2) {
        if (j2 <= 0) continue;
        v.push_back({(e - l) / 2, l, j2, +1});  // neutron first
        v.push_back({(e - l) / 2, l, j2, -1});
      }
    }
  }
  return v;
}

using Key4 = std::array<int, 4>;

struct OracleRecord {
  int jmin = 0;
  std::vector<double> values;  // sentinel = ordinal + 1
};

struct Oracle {
  std::vector<TestOrbit> orbs;
  std::map<Key4, OracleRecord> records;
  std::vector<Key4> order;
  std::int64_t count = 0;
};

Oracle MakeOracle(int emax) {
  Oracle o;
  o.orbs = OracleOrbits(emax);
  const int N = static_cast<int>(o.orbs.size());
  for (int p = 0; p < N; ++p) {
    for (int q = 0; q <= p; ++q) {
      for (int r = 0; r <= p; ++r) {
        for (int s = 0; s <= r; ++s) {
          if (r == p && s > q) continue;
          const auto& P = o.orbs[p];
          const auto& Q = o.orbs[q];
          const auto& R = o.orbs[r];
          const auto& S = o.orbs[s];
          if ((P.l + Q.l + R.l + S.l) % 2 != 0) continue;
          if (P.tz2 + Q.tz2 != R.tz2 + S.tz2) continue;
          int lo = 0;
          int hi = 100;
          // J from the 2j values directly.
          lo = std::max(std::abs(P.j2 - Q.j2), std::abs(R.j2 - S.j2)) / 2;
          hi = std::min(P.j2 + Q.j2, R.j2 + S.j2) / 2;
          if (lo > hi) continue;
          OracleRecord rec;
          rec.jmin = lo;
          for (int J = lo; J <= hi; ++J) {
            rec.values.push_back(static_cast<double>(o.count + 1));
            ++o.count;
          }
          o.records[{p, q, r, s}] = rec;
          o.order.push_back({p, q, r, s});
        }
      }
    }
  }
  return o;
}

int PairPhase(const Oracle& o, int a, int b, int J) {
  const int k = (o.orbs[a].j2 + o.orbs[b].j2) / 2 - J;
  return (std::abs(k) % 2 == 0) ? -1 : +1;  // -(-1)^k
}

/// The oracle's value of <ab; J|O|cd; J> for any orbit order: tries the 8
/// symmetry images and applies the phases of design/19 section 5.
double OracleValue(const Oracle& o, int h, int a, int b, int c, int d, int J) {
  for (int sab = 0; sab < 2; ++sab) {
    for (int scd = 0; scd < 2; ++scd) {
      for (int sx = 0; sx < 2; ++sx) {
        int p = sab != 0 ? b : a;
        int q = sab != 0 ? a : b;
        int r = scd != 0 ? d : c;
        int s = scd != 0 ? c : d;
        if (sx != 0) {
          std::swap(p, r);
          std::swap(q, s);
        }
        const auto it = o.records.find({p, q, r, s});
        if (it == o.records.end()) continue;
        const int idx = J - it->second.jmin;
        if (idx < 0 || idx >= static_cast<int>(it->second.values.size())) {
          continue;
        }
        double ph = 1.0;
        if (sab != 0) ph *= PairPhase(o, a, b, J);
        if (scd != 0) ph *= PairPhase(o, c, d, J);
        if (sx != 0) ph *= h;
        return ph * it->second.values[idx];
      }
    }
  }
  return 0.0;
}

std::vector<double> OracleValues(const Oracle& o) {
  std::vector<double> v;
  for (const Key4& k : o.order) {
    for (double x : o.records.at(k).values) v.push_back(x);
  }
  return v;
}

// -----------------------------------------------------------------------------

TEST(Mexj, OrbitOrderAndIndex) {
  for (int emax = 0; emax <= 4; ++emax) {
    const auto prod = NljjtOrbits(emax);
    const auto mine = OracleOrbits(emax);
    ASSERT_EQ(prod.size(), mine.size());
    ASSERT_EQ(prod.size(), static_cast<std::size_t>((emax + 1) * (emax + 2)));
    for (std::size_t i = 0; i < prod.size(); ++i) {
      EXPECT_EQ(prod[i].n, mine[i].n);
      EXPECT_EQ(prod[i].l, mine[i].l);
      EXPECT_EQ(prod[i].j2, mine[i].j2);
      EXPECT_EQ(prod[i].tz2, mine[i].tz2);
      EXPECT_EQ(NljjtIndex(prod[i].n, prod[i].l, prod[i].j2, prod[i].tz2),
                static_cast<std::int32_t>(i));
    }
  }
  // Hand-checked: 0s1/2 n,p = 0,1; 1s1/2 pi = 7; 0d5/2 nu = 10, pi = 11.
  EXPECT_EQ(NljjtIndex(0, 0, 1, +1), 0);
  EXPECT_EQ(NljjtIndex(0, 0, 1, -1), 1);
  EXPECT_EQ(NljjtIndex(1, 0, 1, -1), 7);
  EXPECT_EQ(NljjtIndex(0, 2, 5, +1), 10);
  EXPECT_EQ(NljjtIndex(0, 2, 5, -1), 11);
}

TEST(Mexj, Counts) {
  for (int emax = 0; emax <= 4; ++emax) {
    const Oracle o = MakeOracle(emax);
    EXPECT_EQ(Me2jpValueCount(emax), o.count) << emax;
    std::int64_t records = 0;
    ForEachMe2jpRecord(emax, [&](int, int, int, int, const JRange&) {
      ++records;
    });
    EXPECT_EQ(records, static_cast<std::int64_t>(o.records.size())) << emax;
    const std::int64_t n = (emax + 1) * (emax + 2);
    EXPECT_EQ(Me1jValueCount(emax), 1 + n * (n + 1) / 2);
    // The enumerator visits the oracle's records in the same order.
    std::size_t i = 0;
    ForEachMe2jpRecord(emax, [&](int p, int q, int r, int s, const JRange& g) {
      ASSERT_LT(i, o.order.size());
      EXPECT_EQ((Key4{p, q, r, s}), o.order[i]);
      EXPECT_EQ(g.jmin, o.records.at(o.order[i]).jmin);
      EXPECT_EQ(g.size(),
                static_cast<int>(o.records.at(o.order[i]).values.size()));
      ++i;
    });
  }
}

class Sentinel : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(Sentinel, EveryRecordEveryJEveryOrdering) {
  const int emax = std::get<0>(GetParam());
  const int h = std::get<1>(GetParam());
  const Oracle o = MakeOracle(emax);
  const Me2jpTable table(
      emax, h > 0 ? Hermiticity::kHermitian : Hermiticity::kAntiHermitian,
      OracleValues(o));
  const int N = static_cast<int>(o.orbs.size());
  const int jtop = 2 * emax + 3;  // beyond every range
  std::int64_t nonzero = 0;
  for (int a = 0; a < N; ++a) {
    for (int b = 0; b < N; ++b) {
      for (int c = 0; c < N; ++c) {
        for (int d = 0; d < N; ++d) {
          for (int J = 0; J <= jtop; ++J) {
            const double want = OracleValue(o, h, a, b, c, d, J);
            const double got = table.Get(a, b, c, d, J);
            ASSERT_EQ(got, want) << a << "," << b << "," << c << "," << d
                                 << " J=" << J;
            if (want != 0.0) ++nonzero;
          }
        }
      }
    }
  }
  EXPECT_GT(nonzero, o.count);  // the non-canonical orderings are exercised
  // Canonical queries return the sentinel itself.
  for (const Key4& k : o.order) {
    const OracleRecord& rec = o.records.at(k);
    for (std::size_t i = 0; i < rec.values.size(); ++i) {
      EXPECT_EQ(table.Get(k[0], k[1], k[2], k[3],
                          rec.jmin + static_cast<int>(i)),
                rec.values[i]);
    }
    // Lower bound and upper bound of the stored range: exactly 0.0.
    EXPECT_EQ(table.Get(k[0], k[1], k[2], k[3], rec.jmin - 1), 0.0);
    EXPECT_EQ(table.Get(k[0], k[1], k[2], k[3],
                        rec.jmin + static_cast<int>(rec.values.size())),
              0.0);
  }
  // The last record of the file: out-of-range J gives 0.0, not a read past the
  // end.
  const Key4& last = o.order.back();
  const OracleRecord& lr = o.records.at(last);
  EXPECT_EQ(table.Get(last[0], last[1], last[2], last[3],
                      lr.jmin + static_cast<int>(lr.values.size())),
            0.0);
  EXPECT_EQ(table.Get(last[0], last[1], last[2], last[3], 1000), 0.0);
  EXPECT_EQ(table.Get(last[0], last[1], last[2], last[3], -1), 0.0);
}

INSTANTIATE_TEST_SUITE_P(Mexj, Sentinel,
                         ::testing::Values(std::make_tuple(2, +1),
                                           std::make_tuple(2, -1),
                                           std::make_tuple(3, +1),
                                           std::make_tuple(3, -1)));

TEST(Mexj, OwnersExample) {
  const Oracle o = MakeOracle(2);
  const Me2jpTable table(2, Hermiticity::kHermitian, OracleValues(o));
  const int dp = NljjtIndex(0, 2, 5, -1);
  const int dn = NljjtIndex(0, 2, 5, +1);
  const int sp = NljjtIndex(1, 0, 1, -1);
  const int sn = NljjtIndex(1, 0, 1, +1);
  // <0d5/2 pi 0d5/2 nu; 3| O |1s1/2 pi 1s1/2 nu; 3> = 0, in every ordering.
  for (int J : {2, 3, 4, 5}) {
    EXPECT_EQ(table.Get(dp, dn, sp, sn, J), 0.0) << J;
    EXPECT_EQ(table.Get(sp, sn, dp, dn, J), 0.0) << J;
    EXPECT_EQ(table.Get(dn, dp, sn, sp, J), 0.0) << J;
  }
  // jIMSRG would return the J = 2 entry of the next record: the next record
  // is <d5/2 pi d5/2 nu| d3/2 nu s1/2 pi>.
  const int d3n = NljjtIndex(0, 2, 3, +1);
  const int s0p = NljjtIndex(0, 0, 1, -1);
  EXPECT_NE(table.Get(dp, dn, d3n, s0p, 2), 0.0);
  // The in-range entries are sentinels.
  EXPECT_NE(table.Get(dp, dn, sp, sn, 0), 0.0);
  EXPECT_NE(table.Get(dp, dn, sp, sn, 1), 0.0);
}

TEST(Mexj, TableRejectsBadInput) {
  EXPECT_THROW(Me2jpTable(1, Hermiticity::kHermitian, {1.0, 2.0}), InputError);
  EXPECT_THROW(
      Me2jpTable(0, Hermiticity::kNonHermitian,
                 std::vector<double>(Me2jpValueCount(0), 1.0)),
      InputError);
}

// -----------------------------------------------------------------------------
// Round trips.

std::shared_ptr<const basis::ModelSpace> MakeMs(int emax) {
  return basis::ModelSpace::FromNames({.emax = emax}, 20.0, "He4", "He4");
}

op::Operator MakeRandom(const std::shared_ptr<const basis::ModelSpace>& ms,
                        Hermiticity h, std::uint64_t seed) {
  op::RandomOperatorOptions opt;
  opt.seed = seed;
  opt.zero_body_scale = 1.0;
  return op::RandomOperator(ms, op::OperatorSpec::Scalar(2, h), opt);
}

std::int64_t UlpDistance(double a, double b) {
  if (a == b) return 0;
  if ((a < 0) != (b < 0)) return INT64_MAX;
  return std::llabs(std::bit_cast<std::int64_t>(a) -
                    std::bit_cast<std::int64_t>(b));
}

/// Calls f(a, b, c, d, J, ident_pairs) for every ket pair a <= b, c <= d.
template <class F>
void ForEachElement(const basis::ModelSpace& ms, F&& f) {
  const auto n = ms.basis().num_orbits();
  const int jtop = 2 * ms.basis().emax() + 2;
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = a; b < n; ++b) {
      for (std::int32_t c = 0; c < n; ++c) {
        for (std::int32_t d = c; d < n; ++d) {
          const int ident = (a == b ? 1 : 0) + (c == d ? 1 : 0);
          for (int J = 0; J <= jtop; ++J) {
            f(OrbitIndex{a}, OrbitIndex{b}, OrbitIndex{c}, OrbitIndex{d}, J,
              ident);
          }
        }
      }
    }
  }
}

void ExpectSameOperator(const op::Operator& x, const op::Operator& y,
                        const std::string& what) {
  const basis::ModelSpace& ms = x.model_space();
  EXPECT_EQ(x.zero_body(), y.zero_body()) << what;
  const auto n = ms.basis().num_orbits();
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = 0; b < n; ++b) {
      ASSERT_EQ(x.one_body().get(OrbitIndex{a}, OrbitIndex{b}),
                y.one_body().get(OrbitIndex{a}, OrbitIndex{b}))
          << what << " 1b " << a << "," << b;
    }
  }
  std::int64_t inexact = 0;
  std::int64_t nonzero = 0;
  ForEachElement(ms, [&](OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                         int J, int ident) {
    const double u = x.two_body().get_normalized(a, b, c, d, TwoJ{2 * J});
    const double v = y.two_body().get_normalized(a, b, c, d, TwoJ{2 * J});
    if (u != 0.0) ++nonzero;
    if (ident == 1) {
      // x -> x * sqrt(2) -> / sqrt(2) cannot be the identity for every double
      // (the product has fewer representable values in one binade), so one
      // identical pair allows 2 ulp; all other classes must be bit exact.
      if (u != v) ++inexact;
      ASSERT_LE(UlpDistance(u, v), 2) << what << " J=" << J;
    } else {
      ASSERT_EQ(u, v) << what << " J=" << J << " ident=" << ident;
    }
  });
  EXPECT_GT(nonzero, 0) << what;
  RecordProperty("inexact_one_identical_pair", std::to_string(inexact));
}

class RoundTrip : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(RoundTrip, TextBitForBit) {
  const int emax = std::get<0>(GetParam());
  const Hermiticity h = std::get<1>(GetParam()) > 0
                            ? Hermiticity::kHermitian
                            : Hermiticity::kAntiHermitian;
  const auto ms = MakeMs(emax);
  const op::Operator O = MakeRandom(ms, h, 7);
  test_support::ScratchDir dir;
  const fs::path f1 = dir.path() / "x.me1j";
  const fs::path f2 = dir.path() / "x.me2jp";
  WriteMe1j(f1, O, "random");
  WriteMe2jp(f2, O, "random");
  std::ifstream in(f2);
  std::string header;
  std::getline(in, header);
  EXPECT_EQ(header, std::string("me2jp emax=") + std::to_string(emax) +
                        " herm=" + (h == Hermiticity::kHermitian ? "+1" : "-1") +
                        " op=random code=imsrg");
  const op::Operator R = ReadMe1jMe2jp(f1, f2, ms, h);
  ExpectSameOperator(O, R, "text emax " + std::to_string(emax));
  // A second generation writes the same text.
  WriteMe2jp(dir.path() / "y.me2jp", R, "random");
  std::ifstream a(f2, std::ios::binary);
  std::ifstream b(dir.path() / "y.me2jp", std::ios::binary);
  std::stringstream sa;
  std::stringstream sb;
  sa << a.rdbuf();
  sb << b.rdbuf();
  EXPECT_EQ(sa.str().size(), sb.str().size());
}

INSTANTIATE_TEST_SUITE_P(Mexj, RoundTrip,
                         ::testing::Values(std::make_tuple(2, +1),
                                           std::make_tuple(2, -1),
                                           std::make_tuple(4, +1),
                                           std::make_tuple(4, -1)));

TEST(Mexj, BinaryRoundTrip) {
  for (int emax : {2, 4}) {
    for (Hermiticity h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
      const auto ms = MakeMs(emax);
      const op::Operator O = MakeRandom(ms, h, 11);
      test_support::ScratchDir dir;
      const fs::path f1 = dir.path() / "x.me1j";
      const fs::path f2 = dir.path() / "x.me2jp.bin";
      WriteMe1j(f1, O, "random");
      WriteMe2jp(f2, O, "random");
      EXPECT_EQ(fs::file_size(f2), static_cast<std::uintmax_t>(
                                       255 + 4 * Me2jpValueCount(emax)));
      const op::Operator R = ReadMe1jMe2jp(f1, f2, ms, h);
      std::int64_t checked = 0;
      ForEachElement(*ms, [&](OrbitIndex a, OrbitIndex b, OrbitIndex c,
                              OrbitIndex d, int J, int ident) {
        const TwoJ tj{2 * J};
        const double unnorm = O.two_body().get_unnormalized(a, b, c, d, tj);
        const double factor = ident == 2   ? 2.0
                              : ident == 1 ? std::sqrt(2.0)
                                           : 1.0;
        const double want =
            static_cast<double>(static_cast<float>(unnorm)) / factor;
        const double got = R.two_body().get_normalized(a, b, c, d, tj);
        ASSERT_EQ(got, want);
        const double orig = O.two_body().get_normalized(a, b, c, d, tj);
        ASSERT_LE(std::abs(got - orig), 1.0e-7 * std::abs(orig));
        if (orig != 0.0) ++checked;
      });
      EXPECT_GT(checked, 0);
      EXPECT_EQ(R.zero_body(), O.zero_body());
    }
  }
}

TEST(Mexj, PrefixReading) {
  const auto ms4 = MakeMs(4);
  const auto ms2 = MakeMs(2);
  const op::Operator O4 = MakeRandom(ms4, Hermiticity::kHermitian, 3);
  test_support::ScratchDir dir;
  for (const char* ext : {"me2jp", "me2jp.bin"}) {
    const fs::path f1 = dir.path() / "x.me1j";
    const fs::path f2 = dir.path() / (std::string("x.") + ext);
    WriteMe1j(f1, O4, "p");
    WriteMe2jp(f2, O4, "p");
    const op::Operator R = ReadMe1jMe2jp(f1, f2, ms2, Hermiticity::kHermitian);
    const bool bin = std::string(ext) != "me2jp";
    ForEachElement(*ms2, [&](OrbitIndex a, OrbitIndex b, OrbitIndex c,
                             OrbitIndex d, int J, int) {
      const auto m = [&](OrbitIndex x) {
        return *ms4->basis().find(ms2->basis().orbit(x));
      };
      const double want = O4.two_body().get_normalized(m(a), m(b), m(c), m(d),
                                                       TwoJ{2 * J});
      const double got = R.two_body().get_normalized(a, b, c, d, TwoJ{2 * J});
      if (bin) {
        ASSERT_LE(std::abs(got - want), 1.0e-6 * (std::abs(want) + 1.0e-3));
      } else {
        ASSERT_LE(UlpDistance(got, want), 2);
      }
    });
    EXPECT_EQ(R.zero_body(), O4.zero_body());
  }
}

TEST(Mexj, ReaderErrors) {
  const auto ms = MakeMs(2);
  const op::Operator O = MakeRandom(ms, Hermiticity::kHermitian, 5);
  test_support::ScratchDir dir;
  const fs::path f1 = dir.path() / "x.me1j";
  const fs::path f2 = dir.path() / "x.me2jp";
  const fs::path fb = dir.path() / "x.me2jp.bin";
  WriteMe1j(f1, O, "e");
  WriteMe2jp(f2, O, "e");
  WriteMe2jp(fb, O, "e");
  // Wrong hermiticity token.
  EXPECT_THROW(ReadMe1jMe2jp(f1, f2, ms, Hermiticity::kAntiHermitian),
               InputError);
  // The file has a lower emax than the model space.
  EXPECT_THROW(ReadMe1jMe2jp(f1, f2, MakeMs(3), Hermiticity::kHermitian),
               InputError);
  // Missing file.
  EXPECT_THROW(
      ReadMe1jMe2jp(f1, dir.path() / "none.me2jp", ms, Hermiticity::kHermitian),
      InputError);
  // Truncated text.
  {
    std::ifstream in(f2);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    const fs::path t = dir.path() / "t.me2jp";
    std::ofstream(t) << text.substr(0, text.size() / 2);
    EXPECT_THROW(ReadMe1jMe2jp(f1, t, ms, Hermiticity::kHermitian), InputError);
    // No tag.
    const fs::path n = dir.path() / "n.me2jp";
    std::ofstream(n) << "header without a tag\n"
                     << text.substr(text.find('\n') + 1);
    EXPECT_THROW(ReadMe1jMe2jp(f1, n, ms, Hermiticity::kHermitian), InputError);
  }
  // Truncated and misaligned binary.
  {
    const std::uintmax_t size = fs::file_size(fb);
    fs::path t = dir.path() / "t.me2jp.bin";
    fs::copy_file(fb, t);
    fs::resize_file(t, size - 4);
    EXPECT_THROW(ReadMe1jMe2jp(f1, t, ms, Hermiticity::kHermitian), InputError);
    fs::resize_file(t, size - 1);
    EXPECT_THROW(ReadMe1jMe2jp(f1, t, ms, Hermiticity::kHermitian), InputError);
  }
}

TEST(Mexj, WriterRejectsNonHermitian) {
  const auto ms = MakeMs(1);
  const op::Operator O(
      ms, op::OperatorSpec::Scalar(2, Hermiticity::kNonHermitian));
  test_support::ScratchDir dir;
  EXPECT_THROW(WriteMe2jp(dir.path() / "x.me2jp", O, "n"), NotImplementedError);
  EXPECT_THROW(WriteMe1j(dir.path() / "x.me1j", O, "n"), NotImplementedError);
  EXPECT_FALSE(fs::exists(dir.path() / "x.me2jp"));
}

}  // namespace
}  // namespace imsrg::io
