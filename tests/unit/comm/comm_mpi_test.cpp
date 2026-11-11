/// @file
/// Multi-rank tests of imsrg::comm. CTest runs this under mpiexec with 2 and 4 ranks; every rank
/// runs the tests, only rank 0 prints, and the exit code is the maximum over ranks.

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/comm/communicator.hpp"
#include "imsrg/comm/environment.hpp"

namespace {

using imsrg::comm::Communicator;
using imsrg::comm::Environment;

// Set in main() after MPI is up.
Communicator g_world = Communicator::serial();

TEST(CommMpi, RankAndSize) {
  EXPECT_GE(g_world.size(), 2);
  EXPECT_GE(g_world.rank(), 0);
  EXPECT_LT(g_world.rank(), g_world.size());
  EXPECT_EQ(g_world.is_root(), g_world.rank() == 0);
  EXPECT_FALSE(g_world.is_serial());
  EXPECT_TRUE(Environment::is_initialized());
  EXPECT_TRUE(Environment::mpi_enabled());
  g_world.barrier();
}

TEST(CommMpi, ExactIntegerReductions) {
  const std::int64_t n = g_world.size();
  const std::int64_t r = g_world.rank();
  EXPECT_EQ(g_world.allreduce_sum(r + 1), n * (n + 1) / 2);
  EXPECT_EQ(g_world.allreduce_max(r * 10 - 3), (n - 1) * 10 - 3);
  EXPECT_EQ(g_world.allreduce_min(r * 10 - 3), -3);
  std::array<std::int64_t, 3> v{r, 1, -r};
  g_world.allreduce_sum(v);
  EXPECT_EQ(v[0], n * (n - 1) / 2);
  EXPECT_EQ(v[1], n);
  EXPECT_EQ(v[2], -n * (n - 1) / 2);
  std::array<std::int32_t, 2> w{static_cast<std::int32_t>(r), static_cast<std::int32_t>(-r)};
  g_world.allreduce_max(w);
  EXPECT_EQ(w[0], n - 1);
  EXPECT_EQ(w[1], 0);
  g_world.allreduce_min(w);  // now identical on all ranks
  EXPECT_EQ(w[0], n - 1);
  std::array<std::int32_t, 1> s32{static_cast<std::int32_t>(r + 1)};
  g_world.allreduce_sum(s32);
  EXPECT_EQ(s32[0], n * (n + 1) / 2);
  // 64-bit values beyond 32-bit range.
  EXPECT_EQ(g_world.allreduce_sum(std::int64_t{1} << 40), n * (std::int64_t{1} << 40));
}

TEST(CommMpi, FloatingPointReductions) {
  const int n = g_world.size();
  const double r = g_world.rank();
  // Sums of dyadic rationals are exact.
  EXPECT_DOUBLE_EQ(g_world.allreduce_sum(0.25 * (r + 1)), 0.25 * n * (n + 1) / 2);
  EXPECT_EQ(g_world.allreduce_max(r * 0.5), (n - 1) * 0.5);
  EXPECT_EQ(g_world.allreduce_min(r * 0.5 + 1.0), 1.0);
  std::array<double, 2> d{r, 2.0};
  g_world.allreduce_sum(d);
  EXPECT_DOUBLE_EQ(d[0], n * (n - 1) / 2.0);
  EXPECT_DOUBLE_EQ(d[1], 2.0 * n);
  std::array<float, 2> f{static_cast<float>(r), 1.0F};
  g_world.allreduce_max(f);
  EXPECT_EQ(f[0], static_cast<float>(n - 1));
  g_world.allreduce_min(f);
  EXPECT_EQ(f[1], 1.0F);
  // Identical on all ranks: compare the result with rank 0's via allgather.
  const auto all = g_world.allgather(g_world.allreduce_sum(0.1 * (r + 1)));
  for (double x : all) EXPECT_EQ(x, all[0]);
}

TEST(CommMpi, BroadcastFromNonZeroRoot) {
  const int root = g_world.size() - 1;
  const bool is_root = g_world.rank() == root;
  std::array<double, 3> d =
      is_root ? std::array<double, 3>{1.5, 2.5, -3.5} : std::array<double, 3>{0, 0, 0};
  g_world.broadcast(d, root);
  EXPECT_EQ(d, (std::array<double, 3>{1.5, 2.5, -3.5}));

  std::array<float, 2> f{is_root ? 4.0F : 0.0F, is_root ? 5.0F : 0.0F};
  g_world.broadcast(f, root);
  EXPECT_EQ(f[1], 5.0F);

  std::array<std::int32_t, 2> i{is_root ? 11 : 0, is_root ? -12 : 0};
  g_world.broadcast(i, root);
  EXPECT_EQ(i[1], -12);

  std::array<std::int64_t, 2> l{is_root ? (std::int64_t{1} << 45) : 0, is_root ? 3 : 0};
  g_world.broadcast(l, root);
  EXPECT_EQ(l[0], std::int64_t{1} << 45);

  std::array<std::byte, 4> b{};
  if (is_root) b = {std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}};
  g_world.broadcast(b, root);
  EXPECT_EQ(b[0], std::byte{0xde});
  EXPECT_EQ(b[3], std::byte{0xef});

  std::string s = is_root ? "a string from the last rank" : "";
  g_world.broadcast(s, root);
  EXPECT_EQ(s, "a string from the last rank");

  std::string longer = is_root ? "" : "stale content that is longer than the root's";
  g_world.broadcast(longer, root);
  EXPECT_EQ(longer, "");

  EXPECT_THROW(g_world.broadcast(d, g_world.size()), imsrg::InvariantError);
}

TEST(CommMpi, AllgatherOrderedByRank) {
  const auto n = static_cast<std::size_t>(g_world.size());
  const auto di = g_world.allgather(static_cast<double>(g_world.rank()) * 2.0);
  ASSERT_EQ(di.size(), n);
  for (std::size_t k = 0; k < n; ++k) EXPECT_EQ(di[k], 2.0 * static_cast<double>(k));
  const auto li = g_world.allgather(static_cast<std::int64_t>(g_world.rank()) * 100);
  ASSERT_EQ(li.size(), n);
  for (std::size_t k = 0; k < n; ++k) EXPECT_EQ(li[k], static_cast<std::int64_t>(k) * 100);
}

TEST(CommMpi, Alltoallv) {
  // Rank r sends (r + 1 + dest) doubles of value 100 r + dest to each dest; so rank d receives
  // from rank s (s + 1 + d) doubles of value 100 s + d.
  const int me = g_world.rank();
  const auto n = static_cast<std::size_t>(g_world.size());
  std::vector<int> sc(n), sd(n), rc(n), rd(n);
  std::vector<double> send;
  int total_recv = 0;
  for (std::size_t u = 0; u < n; ++u) {
    const int k = static_cast<int>(u);
    sc[u] = me + 1 + k;
    sd[u] = static_cast<int>(send.size());
    send.insert(send.end(), static_cast<std::size_t>(sc[u]), 100.0 * me + k);
    rc[u] = k + 1 + me;
    rd[u] = total_recv;
    total_recv += rc[u];
  }
  std::vector<double> recv(static_cast<std::size_t>(total_recv), -1.0);
  g_world.alltoallv(send, sc, sd, recv, rc, rd);
  for (std::size_t u = 0; u < n; ++u) {
    for (int j = 0; j < rc[u]; ++j) {
      EXPECT_EQ(recv[static_cast<std::size_t>(rd[u] + j)], 100.0 * static_cast<double>(u) + me);
    }
  }
}

TEST(CommMpi, SplitEvenOdd) {
  const int me = g_world.rank();
  const int n = g_world.size();
  const auto sub = g_world.split(me % 2, me);
  const int expected_size = (me % 2 == 0) ? (n + 1) / 2 : n / 2;
  EXPECT_EQ(sub.size(), expected_size);
  EXPECT_EQ(sub.rank(), me / 2);
  EXPECT_FALSE(sub.is_serial());
  // Sum of world ranks within the group.
  std::int64_t expected = 0;
  for (int k = me % 2; k < n; k += 2) expected += k;
  EXPECT_EQ(sub.allreduce_sum(static_cast<std::int64_t>(me)), expected);
  const auto gathered = sub.allgather(static_cast<std::int64_t>(me));
  ASSERT_EQ(gathered.size(), static_cast<std::size_t>(expected_size));
  for (std::size_t k = 0; k < gathered.size(); ++k) {
    EXPECT_EQ(gathered[k], static_cast<std::int64_t>(2 * k + static_cast<std::size_t>(me % 2)));
  }
  std::string s = sub.is_root() ? "group " + std::to_string(me % 2) : "";
  sub.broadcast(s, 0);
  EXPECT_EQ(s, "group " + std::to_string(me % 2));
  sub.barrier();
  // The reversed key reverses the order inside the group.
  const auto rev = g_world.split(0, -me);
  EXPECT_EQ(rev.rank(), n - 1 - me);
}

TEST(CommMpi, SubcommunicatorOutlivesCopies) {
  Communicator kept = Communicator::serial();
  {
    const auto sub = g_world.split(0, g_world.rank());
    kept = sub;  // the handle outlives `sub`
  }
  EXPECT_EQ(kept.size(), g_world.size());
  EXPECT_EQ(kept.allreduce_sum(std::int64_t{1}), g_world.size());
}

TEST(CommMpi, Duplicate) {
  const auto dup = g_world.duplicate();
  EXPECT_EQ(dup.size(), g_world.size());
  EXPECT_EQ(dup.rank(), g_world.rank());
  EXPECT_EQ(dup.allreduce_sum(std::int64_t{1}), g_world.size());
}

TEST(CommMpi, SerialCommunicatorWorksAlongside) {
  const auto s = Communicator::serial();
  EXPECT_EQ(s.size(), 1);
  EXPECT_EQ(s.allreduce_sum(static_cast<double>(g_world.rank())), g_world.rank());
}

}  // namespace

int main(int argc, char** argv) {
  int exit_code = 0;
  {
    Environment env(argc, argv);
    g_world = env.world();
    ::testing::InitGoogleTest(&argc, argv);
    if (g_world.rank() != 0) {  // only rank 0 prints the full GoogleTest report
      auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
      delete listeners.Release(listeners.default_result_printer());
    }
    const int failed = RUN_ALL_TESTS();
    if (failed != 0 && g_world.rank() != 0) {
      std::fprintf(stderr, "comm_mpi: tests failed on rank %d\n", g_world.rank());
    }
    exit_code = static_cast<int>(g_world.allreduce_max(static_cast<std::int64_t>(failed)));
    g_world = Communicator::serial();
  }
  return exit_code;
}
