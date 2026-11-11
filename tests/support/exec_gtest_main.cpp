// GoogleTest main for tests that need the execution runtime. Link with NO_GTEST_MAIN and add this
// source to the test: it initializes the exec::Runtime (Kokkos, honouring --kokkos-* arguments)
// around RUN_ALL_TESTS. Other modules' kernel tests reuse it.
#include <gtest/gtest.h>

#include "imsrg/exec/runtime.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  imsrg::exec::Runtime runtime(argc, argv);
  return RUN_ALL_TESTS();
}
