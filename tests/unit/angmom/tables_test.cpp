// Tests of SixJTable, NineJLSjjTable and MoshinskyTable (host lookups and device views).
#include "imsrg/angmom/tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tables_kernels.hpp"
#include <gtest/gtest.h>

#include "imsrg/angmom/moshinsky.hpp"
#include "imsrg/angmom/wigner.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::angmom;

TEST(AngmomTables, SixJEqualsWignerExhaustively) {
  const int emax = 6;
  const SixJTable t(emax);
  EXPECT_EQ(t.emax(), emax);
  const int tmax = 2 * emax + 1;
  long checked = 0;
  for (int a = 1; a <= tmax; a += 2) {
    for (int b = 1; b <= tmax; b += 2) {
      for (int J = 0; J <= tmax; J += 2) {
        for (int c = 1; c <= tmax; c += 2) {
          for (int d = 1; d <= tmax; d += 2) {
            for (int Jp = 0; Jp <= tmax; Jp += 2) {
              const double ref = wigner_6j(TwoJ{a}, TwoJ{b}, TwoJ{J}, TwoJ{c}, TwoJ{d}, TwoJ{Jp});
              ASSERT_EQ(t(TwoJ{a}, TwoJ{b}, TwoJ{J}, TwoJ{c}, TwoJ{d}, TwoJ{Jp}), ref);
              ++checked;
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 0);
}

TEST(AngmomTables, SixJOutOfRangeIsZero) {
  const SixJTable t(3);
  EXPECT_EQ(t(TwoJ{9}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);   // ja too large
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{20}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);  // J too large
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{-2}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);  // J negative
  EXPECT_EQ(t(TwoJ{2}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);   // even 2j
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{1}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);   // odd 2J
  EXPECT_EQ(t(TwoJ{-1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);  // negative 2j
}

TEST(AngmomTables, NineJEqualsNormNineJ) {
  const int lmax = 6;
  const NineJLSjjTable t(lmax);
  for (int la = 0; la <= lmax; ++la) {
    for (int sa = -1; sa <= 1; sa += 2) {
      const int ja = 2 * la + sa;
      if (ja < 1) {
        continue;
      }
      for (int lb = 0; lb <= lmax; ++lb) {
        for (int sb = -1; sb <= 1; sb += 2) {
          const int jb = 2 * lb + sb;
          if (jb < 1) {
            continue;
          }
          for (int L = 0; L <= 2 * lmax; ++L) {
            for (int S = 0; S <= 1; ++S) {
              for (int J = 0; J <= 2 * lmax + 1; ++J) {
                const double ref = norm_9j(TwoJ{2 * la}, TwoJ{1}, TwoJ{ja}, TwoJ{2 * lb}, TwoJ{1},
                                           TwoJ{jb}, TwoJ{2 * L}, TwoJ{2 * S}, TwoJ{2 * J});
                ASSERT_EQ(t(la, TwoJ{ja}, lb, TwoJ{jb}, L, S, TwoJ{2 * J}), ref);
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(t(0, TwoJ{-1}, 0, TwoJ{1}, 0, 0, TwoJ{0}), 0.0);
  EXPECT_EQ(t(lmax + 1, TwoJ{2 * lmax + 3}, 0, TwoJ{1}, 0, 0, TwoJ{0}), 0.0);
  EXPECT_EQ(t(1, TwoJ{2}, 0, TwoJ{1}, 0, 0, TwoJ{0}), 0.0);  // ja not l +- 1/2
}

TEST(AngmomTables, MoshinskyEqualsFunctionForEveryStoredEntry) {
  const int e2max = 12;
  const MoshinskyTable t(e2max);
  EXPECT_EQ(t.e2max(), e2max);
  std::int64_t count = 0;
  for (int E = 0; E <= e2max; ++E) {
    std::vector<std::array<int, 4>> tup;
    for (int l1 = 0; l1 <= E; ++l1) {
      for (int n1 = 0; 2 * n1 + l1 <= E; ++n1) {
        for (int l2 = 0; 2 * n1 + l1 + l2 <= E; ++l2) {
          const int rem = E - 2 * n1 - l1 - l2;
          if (rem % 2 == 0) {
            tup.push_back({n1, l1, rem / 2, l2});
          }
        }
      }
    }
    ASSERT_EQ(static_cast<int>(tup.size()), t.block_size(E));
    for (int lam = 0; lam <= E; ++lam) {
      for (const auto& c : tup) {
        for (const auto& u : tup) {
          const HoNL cm{c[0], c[1]}, rel{c[2], c[3]}, a{u[0], u[1]}, b{u[2], u[3]};
          ASSERT_EQ(t(cm, rel, a, b, lam), moshinsky(cm, rel, a, b, lam));
          ++count;
        }
      }
    }
  }
  EXPECT_EQ(count, MoshinskyTable::value_count(e2max));
  EXPECT_EQ(t(HoNL{0, 0}, HoNL{0, 0}, HoNL{0, 1}, HoNL{0, 0}, 0), 0.0);  // energy mismatch
  EXPECT_EQ(t(HoNL{0, 0}, HoNL{0, 0}, HoNL{0, 0}, HoNL{0, 0}, 1), 0.0);  // lambda > E
  EXPECT_EQ(t(HoNL{0, 0}, HoNL{0, 0}, HoNL{-1, 0}, HoNL{0, 0}, 0), 0.0);
  EXPECT_EQ(t(HoNL{7, 0}, HoNL{0, 0}, HoNL{7, 0}, HoNL{0, 0}, 0), 0.0);  // E > e2max
  EXPECT_GT(t.size_bytes(), 0u);
}

TEST(AngmomTables, DeviceViewsReproduceHostSums) {
  const int emax = 3;
  const SixJTable s(emax);
  double dev_abs = 0.0;
  const double dev = test::sum_sixj_device(s, dev_abs);
  const auto hv = s.host_view();
  double host = 0.0;
  double host_abs = 0.0;
  const int nj = emax + 1;
  const int nJ = 2 * emax + 2;
  for (int ia = 0; ia < nj; ++ia) {
    for (int ib = 0; ib < nj; ++ib) {
      for (int J = 0; J < nJ; ++J) {
        for (int ic = 0; ic < nj; ++ic) {
          for (int id = 0; id < nj; ++id) {
            for (int Jp = 0; Jp < nJ; ++Jp) {
              const double x = hv(TwoJ{2 * ia + 1}, TwoJ{2 * ib + 1}, TwoJ{2 * J}, TwoJ{2 * ic + 1},
                                  TwoJ{2 * id + 1}, TwoJ{2 * Jp});
              host += x;
              host_abs += std::abs(x);
            }
          }
        }
      }
    }
  }
  EXPECT_GT(host_abs, 0.0);
  EXPECT_NEAR(dev, host, 1e-9 * host_abs);
  EXPECT_NEAR(dev_abs, host_abs, 1e-9 * host_abs);

  const int lmax = 4;
  const NineJLSjjTable nt(lmax);
  const auto h9 = nt.host_view();
  double host9 = 0.0;
  for (int la = 0; la <= lmax; ++la) {
    for (int sja = 0; sja < 2; ++sja) {
      for (int lb = 0; lb <= lmax; ++lb) {
        for (int sjb = 0; sjb < 2; ++sjb) {
          for (int L = 0; L <= 2 * lmax; ++L) {
            for (int S = 0; S < 2; ++S) {
              for (int J = 0; J <= 2 * lmax + 1; ++J) {
                host9 += h9(la, TwoJ{2 * la - 1 + 2 * sja}, lb, TwoJ{2 * lb - 1 + 2 * sjb}, L, S,
                            TwoJ{2 * J});
              }
            }
          }
        }
      }
    }
  }
  EXPECT_NEAR(test::sum_ninej_device(nt), host9, 1e-9 * std::max(1.0, std::abs(host9)));

  const MoshinskyTable m(6);
  std::vector<std::int32_t> q;
  std::vector<double> expect;
  for (int a1 = 0; a1 <= 3; ++a1) {
    for (int l1 = 0; l1 <= 4; ++l1) {
      for (int a2 = 0; a2 <= 3; ++a2) {
        for (int l2 = 0; l2 <= 4; ++l2) {
          const int E = 2 * a1 + l1 + 2 * a2 + l2;
          for (int lam = 0; lam <= std::min(E, 6); ++lam) {
            for (int L = 0; L <= E; ++L) {
              const int rem = E - L;  // N = 0, n = rem/2, l = 0
              if (rem % 2 != 0) {
                continue;
              }
              for (int v : {0, L, rem / 2, 0, a1, l1, a2, l2, lam}) {
                q.push_back(v);
              }
              expect.push_back(m(HoNL{0, L}, HoNL{rem / 2, 0}, HoNL{a1, l1}, HoNL{a2, l2}, lam));
            }
          }
        }
      }
    }
  }
  ASSERT_FALSE(expect.empty());
  imsrg::exec::DeviceBuffer<std::int32_t> qb(q.size());
  qb.copy_from_host(q);
  imsrg::exec::DeviceBuffer<double> out(expect.size());
  test::eval_moshinsky_device(m, qb, out);
  const auto got = out.to_vector();
  double maxdiff = 0.0;
  double maxabs = 0.0;
  for (std::size_t i = 0; i < expect.size(); ++i) {
    maxdiff = std::max(maxdiff, std::abs(got[i] - expect[i]));
    maxabs = std::max(maxabs, std::abs(expect[i]));
  }
  EXPECT_EQ(maxdiff, 0.0);
  EXPECT_GT(maxabs, 0.0);
}

}  // namespace
