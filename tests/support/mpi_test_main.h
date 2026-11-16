// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_SUPPORT_MPI_TEST_MAIN_H_
#define TESTS_SUPPORT_MPI_TEST_MAIN_H_
/// @file
/// A reusable `main()` for MPI GoogleTest executables that need Kokkos
/// (`mpi_test_main.cc`). Define the test with `NO_GTEST_MAIN` and add that
/// source to its `SOURCES`.

#include "imsrg/comm/communicator.h"

namespace imsrg::test_support {

/// The communicator over all ranks of the test job (the serial communicator
/// when run without mpiexec). Valid inside the tests only, i.e. between the
/// MPI initialization and finalization of `main()`.
const comm::Communicator& TestWorld();

}  // namespace imsrg::test_support

#endif  // TESTS_SUPPORT_MPI_TEST_MAIN_H_
