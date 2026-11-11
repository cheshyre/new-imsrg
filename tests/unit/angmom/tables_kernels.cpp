// Kernels reading the angmom table device views. Free functions only (design/12 section 5).
#include "tables_kernels.hpp"

#include <cstdint>

#include "imsrg/exec/kokkos_backend.hpp"

namespace imsrg::angmom::test {

double sum_sixj_device(const SixJTable& t, double& abs_sum) {
  const SixJView v = t.device_view();
  const std::int64_t nj = v.emax + 1;
  const std::int64_t nJ = 2 * static_cast<std::int64_t>(v.emax) + 2;
  const std::int64_t n = nj * nj * nj * nj * nJ * nJ;
  double result = 0.0;
  exec::parallel_reduce(
      "test::sum_sixj", n,
      IMSRG_LAMBDA(const exec::Index i, double& partial) {
        std::int64_t r = i;
        const std::int64_t jp = r % nJ;
        r /= nJ;
        const std::int64_t id = r % nj;
        r /= nj;
        const std::int64_t ic = r % nj;
        r /= nj;
        const std::int64_t J = r % nJ;
        r /= nJ;
        const std::int64_t ib = r % nj;
        const std::int64_t ia = r / nj;
        partial += v(TwoJ{static_cast<int>(2 * ia + 1)}, TwoJ{static_cast<int>(2 * ib + 1)},
                     TwoJ{static_cast<int>(2 * J)}, TwoJ{static_cast<int>(2 * ic + 1)},
                     TwoJ{static_cast<int>(2 * id + 1)}, TwoJ{static_cast<int>(2 * jp)});
      },
      result);
  double abs_result = 0.0;
  exec::parallel_reduce(
      "test::sum_sixj_abs", n,
      IMSRG_LAMBDA(const exec::Index i, double& partial) {
        std::int64_t r = i;
        const std::int64_t jp = r % nJ;
        r /= nJ;
        const std::int64_t id = r % nj;
        r /= nj;
        const std::int64_t ic = r % nj;
        r /= nj;
        const std::int64_t J = r % nJ;
        r /= nJ;
        const std::int64_t ib = r % nj;
        const std::int64_t ia = r / nj;
        const double x = v(TwoJ{static_cast<int>(2 * ia + 1)}, TwoJ{static_cast<int>(2 * ib + 1)},
                           TwoJ{static_cast<int>(2 * J)}, TwoJ{static_cast<int>(2 * ic + 1)},
                           TwoJ{static_cast<int>(2 * id + 1)}, TwoJ{static_cast<int>(2 * jp)});
        partial += x < 0 ? -x : x;
      },
      abs_result);
  abs_sum = abs_result;
  return result;
}

double sum_ninej_device(const NineJLSjjTable& t) {
  const NineJLSjjView v = t.device_view();
  const std::int64_t nl = v.lmax + 1;
  const std::int64_t nL = 2 * static_cast<std::int64_t>(v.lmax) + 1;
  const std::int64_t nJ = 2 * static_cast<std::int64_t>(v.lmax) + 2;
  const std::int64_t n = nl * 2 * nl * 2 * nL * 2 * nJ;
  double result = 0.0;
  exec::parallel_reduce(
      "test::sum_ninej", n,
      IMSRG_LAMBDA(const exec::Index i, double& partial) {
        std::int64_t r = i;
        const std::int64_t J = r % nJ;
        r /= nJ;
        const std::int64_t S = r % 2;
        r /= 2;
        const std::int64_t L = r % nL;
        r /= nL;
        const std::int64_t sjb = r % 2;
        r /= 2;
        const std::int64_t lb = r % nl;
        r /= nl;
        const std::int64_t sja = r % 2;
        const std::int64_t la = r / 2;
        partial += v(static_cast<int>(la), TwoJ{static_cast<int>(2 * la - 1 + 2 * sja)},
                     static_cast<int>(lb), TwoJ{static_cast<int>(2 * lb - 1 + 2 * sjb)},
                     static_cast<int>(L), static_cast<int>(S), TwoJ{static_cast<int>(2 * J)});
      },
      result);
  return result;
}

void eval_moshinsky_device(const MoshinskyTable& t, const exec::DeviceBuffer<std::int32_t>& queries,
                           exec::DeviceBuffer<double>& out) {
  const MoshinskyView v = t.device_view();
  const auto q = exec::view(queries);
  auto o = exec::view(out);
  exec::parallel_for(
      "test::eval_moshinsky", static_cast<exec::Index>(o.extent(0)),
      IMSRG_LAMBDA(const exec::Index i) {
        const exec::Index b = 9 * i;
        o(i) = v(HoNL{q(b), q(b + 1)}, HoNL{q(b + 2), q(b + 3)}, HoNL{q(b + 4), q(b + 5)},
                 HoNL{q(b + 6), q(b + 7)}, q(b + 8));
      });
  exec::fence("test::eval_moshinsky");
}

}  // namespace imsrg::angmom::test
