/// @file
/// Smoke test of the MPI toolchain. CTest runs it as `mpiexec -n N smoke_mpi_hello
/// --expected-ranks=N`; it checks the size of MPI_COMM_WORLD, the rank range, and an allreduce.
/// Every rank runs the tests; the exit code is the maximum over ranks, so a failure on any rank
/// fails the CTest test. Uses MPI directly because the imsrg::comm facade arrives in M2.

#include <cstdio>
#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include <mpi.h>

namespace {

// Set once in main() from --expected-ranks=N, before any test runs.
int expected_ranks = -1;

TEST(MpiSmoke, WorldSizeMatchesLauncher) {
  int size = 0;
  ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
  ASSERT_GT(expected_ranks, 0) << "pass --expected-ranks=N";
  EXPECT_EQ(size, expected_ranks);
}

TEST(MpiSmoke, AllreduceSumsRanks) {
  int rank = -1;
  int size = 0;
  ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
  ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
  EXPECT_GE(rank, 0);
  EXPECT_LT(rank, size);
  long long mine = rank + 1;
  long long total = 0;
  ASSERT_EQ(MPI_Allreduce(&mine, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), MPI_SUCCESS);
  const long long n = size;
  EXPECT_EQ(total, n * (n + 1) / 2);
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  constexpr const char* kFlag = "--expected-ranks=";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], kFlag, std::strlen(kFlag)) == 0) {
      expected_ranks = std::stoi(std::string(argv[i] + std::strlen(kFlag)));
    }
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {  // only rank 0 prints the full GoogleTest report
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }
  int failed = RUN_ALL_TESTS();
  if (failed != 0 && rank != 0) {
    std::fprintf(stderr, "smoke_mpi_hello: tests failed on rank %d\n", rank);
  }
  int any_failed = 0;
  MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return any_failed;
}
