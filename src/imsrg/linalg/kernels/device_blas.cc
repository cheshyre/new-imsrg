// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/linalg/device_blas.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/detail/gemm_check.h"

#if defined(IMSRG_DEVICE_BLAS_ROCBLAS) && IMSRG_DEVICE_BLAS_ROCBLAS
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif
#if defined(IMSRG_DEVICE_BLAS_KOKKOSKERNELS) && IMSRG_DEVICE_BLAS_KOKKOSKERNELS
#include <KokkosBlas3_gemm.hpp>
#endif

namespace imsrg::linalg {
namespace {

#if !(defined(IMSRG_DEVICE_BLAS_HOST) && IMSRG_DEVICE_BLAS_HOST)

[[maybe_unused]] int Lead(std::int64_t ld) {
  return static_cast<int>(std::max<std::int64_t>(ld, 1));
}

struct Dims {
  int m = 0;
  int n = 0;
  int k = 0;
};

Dims GemmDims(const GemmProblem& p) {
  const bool ta = p.opA == Op::kTrans;
  const bool tb = p.opB == Op::kTrans;
  return Dims{static_cast<int>(ta ? p.A.cols : p.A.rows),
              static_cast<int>(tb ? p.B.rows : p.B.cols),
              static_cast<int>(ta ? p.A.rows : p.A.cols)};
}

#endif  // !IMSRG_DEVICE_BLAS_HOST

#if defined(IMSRG_DEVICE_BLAS_ROCBLAS) && IMSRG_DEVICE_BLAS_ROCBLAS

void CheckHip(hipError_t e, const char* what) {
  IMSRG_CHECK(e == hipSuccess, std::string("device_blas: ") + what + ": " +
                                   hipGetErrorString(e));
}

void CheckRocblas(rocblas_status s, const char* what) {
  IMSRG_CHECK(
      s == rocblas_status_success,
      std::string("device_blas: ") + what + ": " + rocblas_status_to_string(s));
}

// Side streams for GroupedGemm: problems are spread over them and joined
// back into the Kokkos stream by events, so small GEMMs of different channels
// overlap on the device.
constexpr std::size_t kSideStreams = 4;

// rocBLAS handles and streams, created on first use and never destroyed (the
// process-wide equivalent of a BLAS library's internal state; a mutex makes
// the calls from several host threads safe).
struct RocblasState {
  std::mutex mutex;
  rocblas_handle main = nullptr;  // runs on the Kokkos stream
  std::array<hipStream_t, kSideStreams> streams{};
  std::array<rocblas_handle, kSideStreams> handles{};
  std::array<hipEvent_t, kSideStreams> done{};
  hipEvent_t start = nullptr;
};

RocblasState& State() {
  static auto* const kState = [] {
    auto* s = new RocblasState;  // never destroyed
    CheckRocblas(rocblas_create_handle(&s->main), "rocblas_create_handle");
    CheckHip(hipEventCreateWithFlags(&s->start, hipEventDisableTiming),
             "hipEventCreate");
    for (std::size_t i = 0; i < kSideStreams; ++i) {
      CheckHip(hipStreamCreateWithFlags(&s->streams[i], hipStreamNonBlocking),
               "hipStreamCreate");
      CheckRocblas(rocblas_create_handle(&s->handles[i]),
                   "rocblas_create_handle");
      CheckRocblas(rocblas_set_stream(s->handles[i], s->streams[i]),
                   "rocblas_set_stream");
      CheckHip(hipEventCreateWithFlags(&s->done[i], hipEventDisableTiming),
               "hipEventCreate");
    }
    return s;
  }();
  return *kState;
}

hipStream_t KokkosStream() { return exec::DefaultExecSpace().hip_stream(); }

rocblas_operation RocOp(Op op) {
  return op == Op::kTrans ? rocblas_operation_transpose
                          : rocblas_operation_none;
}

void RocGemm(rocblas_handle h, const GemmProblem& p) {
  const Dims d = GemmDims(p);
  if (d.m == 0 || d.n == 0) return;
  // k == 0 is valid BLAS (C = beta C; beta == 0 does not read C).
  CheckRocblas(rocblas_dgemm(h, RocOp(p.opA), RocOp(p.opB), d.m, d.n, d.k,
                             &p.alpha, p.A.data, Lead(p.A.ld), p.B.data,
                             Lead(p.B.ld), &p.beta, p.C.data, Lead(p.C.ld)),
               "rocblas_dgemm");
}

void RunGemms(std::span<const GemmProblem> problems,
              const std::vector<double>& flops) {
  RocblasState& s = State();
  const std::lock_guard<std::mutex> lock(s.mutex);
  const hipStream_t ks = KokkosStream();
  if (problems.size() == 1) {
    CheckRocblas(rocblas_set_stream(s.main, ks), "rocblas_set_stream");
    RocGemm(s.main, problems[0]);
    return;
  }
  // Fork: the side streams wait for everything queued on the Kokkos stream.
  CheckHip(hipEventRecord(s.start, ks), "hipEventRecord");
  for (std::size_t i = 0; i < kSideStreams; ++i)
    CheckHip(hipStreamWaitEvent(s.streams[i], s.start, 0),
             "hipStreamWaitEvent");
  // Largest first, each to the least loaded stream (deterministic; each C
  // is written by one call, so the bits do not depend on the assignment).
  std::vector<std::size_t> order(problems.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::size_t a, std::size_t b) { return flops[a] > flops[b]; });
  std::array<double, kSideStreams> load{};
  for (const std::size_t i : order) {
    const auto it = std::min_element(load.begin(), load.end());
    const auto si = static_cast<std::size_t>(it - load.begin());
    *it += flops[i] + 1.0e5;  // + launch overhead in flop units
    RocGemm(s.handles[si], problems[i]);
  }
  // Join: the Kokkos stream waits for every side stream.
  for (std::size_t i = 0; i < kSideStreams; ++i) {
    CheckHip(hipEventRecord(s.done[i], s.streams[i]), "hipEventRecord");
    CheckHip(hipStreamWaitEvent(ks, s.done[i], 0), "hipStreamWaitEvent");
  }
}

#elif defined(IMSRG_DEVICE_BLAS_KOKKOSKERNELS) && \
    IMSRG_DEVICE_BLAS_KOKKOSKERNELS

using ConstMat =
    Kokkos::View<const double**, Kokkos::LayoutLeft, exec::DeviceMemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using Mat = Kokkos::View<double**, Kokkos::LayoutLeft, exec::DeviceMemorySpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// An ld x cols LayoutLeft view restricted to the first `rows` rows: the
// subview keeps LayoutLeft with stride ld (the last column's padding is
// never accessed).
template <class V, class T>
auto Wrap(const MatrixView<T>& a) {
  const auto ld = static_cast<std::size_t>(std::max<std::int64_t>(a.ld, 1));
  V full(a.data, ld, static_cast<std::size_t>(a.cols));
  return Kokkos::subview(full,
                         Kokkos::pair<std::size_t, std::size_t>(
                             0, static_cast<std::size_t>(a.rows)),
                         Kokkos::ALL);
}

void KkGemm(const GemmProblem& p) {
  const Dims d = GemmDims(p);
  if (d.m == 0 || d.n == 0) return;
  const char* ta = p.opA == Op::kTrans ? "T" : "N";
  const char* tb = p.opB == Op::kTrans ? "T" : "N";
  KokkosBlas::gemm(exec::DefaultExecSpace(), ta, tb, p.alpha,
                   Wrap<ConstMat>(p.A), Wrap<ConstMat>(p.B), p.beta,
                   Wrap<Mat>(p.C));
}

void RunGemms(std::span<const GemmProblem> problems,
              const std::vector<double>& /*flops*/) {
  for (const GemmProblem& p : problems) KkGemm(p);
}

#endif

}  // namespace

void DeviceGemm(double alpha, MatrixView<const double> A, Op opA,
                MatrixView<const double> B, Op opB, double beta,
                MatrixView<double> C) {
  const GemmProblem p{alpha, A, opA, B, opB, beta, C};
  DeviceGroupedGemm(std::span<const GemmProblem>(&p, 1));
}

void DeviceGroupedGemm(std::span<const GemmProblem> problems) {
#if defined(IMSRG_DEVICE_BLAS_HOST) && IMSRG_DEVICE_BLAS_HOST
  // Host BLAS reads the data directly: pending kernels must finish first.
  exec::Fence("linalg::device_grouped_gemm");
  GroupedGemm(problems);
#else
  if (problems.empty()) return;
  std::vector<double> flops(problems.size());
  for (std::size_t i = 0; i < problems.size(); ++i)
    flops[i] = detail::CheckGemmProblem(problems[i]);
  RunGemms(problems, flops);
#endif
}

std::string_view DeviceBlasBackend() {
#if defined(IMSRG_DEVICE_BLAS_ROCBLAS) && IMSRG_DEVICE_BLAS_ROCBLAS
  return "rocblas";
#elif defined(IMSRG_DEVICE_BLAS_KOKKOSKERNELS) && \
    IMSRG_DEVICE_BLAS_KOKKOSKERNELS
  return "kokkos-kernels";
#else
  return "host";
#endif
}

}  // namespace imsrg::linalg
