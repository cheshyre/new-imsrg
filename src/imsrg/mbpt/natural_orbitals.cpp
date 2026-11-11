#include "imsrg/mbpt/natural_orbitals.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "imsrg/base/error.hpp"
#include "imsrg/linalg/blas.hpp"
#include "imsrg/operator/basis_transform.hpp"

namespace imsrg::mbpt {

namespace {

constexpr double kEps = 1e-8;

struct Ctx {
  op::Operator& H;
  const basis::ModelSpace& ms;
  const basis::Reference& ref;

  double e(OrbitIndex a) const { return H.one_body().get(a, a); }
  double n(OrbitIndex a) const { return ref.occupation(a); }
  const auto& o(OrbitIndex a) const { return ms.basis().orbit(a); }
  int j2(OrbitIndex a) const { return o(a).j2.twice; }

  /// Unnormalized <ab|V|cd>_J with parity / charge guards (0 when the channel does not exist).
  double V(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, int J) const {
    const auto &A = o(a), &B = o(b), &C = o(c), &D = o(d);
    if ((A.l + B.l + C.l + D.l) % 2 != 0) return 0.0;
    if (A.tz2.twice + B.tz2.twice != C.tz2.twice + D.tz2.twice) return 0.0;
    return H.two_body().get_unnormalized(a, b, c, d, TwoJ{2 * J});
  }
};

/// imsrg++ regularization: 0.5 (sqrt(t + eps^2) - eps)/sqrt(t + eps^2).
double regulated(double tbme, double eps) {
  const double s = std::sqrt(tbme + eps * eps);
  return 0.5 * (s - eps) / s;
}

}  // namespace

linalg::Matrix<double> mbpt2_density_matrix(op::Operator& H) {
  H.sync_to_host();
  const auto& ms = H.model_space();
  const auto& ref = ms.reference();
  const Ctx x{H, ms, ref};
  const auto N = ms.num_orbits();
  linalg::Matrix<double> rho(N, N);
  for (const OrbitIndex i : ref.holes()) rho(i.value(), i.value()) = ref.occupation(i);

  const auto particles = ref.particles();
  const auto holes = ref.holes();
  linalg::Matrix<double> rho_pp(N, N), rho_hh(N, N), rho_ph(N, N);

  // PP block: a, b particles in the same channel, b <= a.
  for (const OrbitIndex a : particles) {
    for (const OrbitIndex b : ms.basis().same_ljtz(a)) {
      if (b.value() > a.value() || ref.occupation(b) >= basis::kOccupationCut) continue;
      double r = 0.0;
      for (const OrbitIndex c : particles)
        for (const OrbitIndex i : holes)
          for (const OrbitIndex j : holes) {
            const double eac = x.e(a) + x.e(c) - x.e(i) - x.e(j);
            const double ebc = x.e(b) + x.e(c) - x.e(i) - x.e(j);
            if (std::abs(eac * ebc) < kEps) continue;
            const int jmin = std::max({std::abs(x.j2(a) - x.j2(c)), std::abs(x.j2(i) - x.j2(j)),
                                       std::abs(x.j2(b) - x.j2(c))}) / 2;
            const int jmax =
                std::min({x.j2(a) + x.j2(c), x.j2(i) + x.j2(j), x.j2(b) + x.j2(c)}) / 2;
            double t = 0.0;
            for (int J = jmin; J <= jmax; ++J)
              t += (2 * J + 1) * x.V(a, c, i, j, J) * x.V(i, j, b, c, J);
            const double nc = 1.0 - x.n(c);
            t *= (1 - x.n(a)) * (1 - x.n(b)) * nc * nc * x.n(i) * x.n(i) * x.n(j) * x.n(j);
            r += regulated(t, 0.5 * std::sqrt(std::abs(eac * ebc)));
          }
      const double v = r * 0.5 / (x.j2(a) + 1);
      rho_pp(a.value(), b.value()) = v;
      rho_pp(b.value(), a.value()) = v;
    }
  }

  // HH block: i, j holes in the same channel, j <= i.
  for (const OrbitIndex i : holes) {
    for (const OrbitIndex j : ms.basis().same_ljtz(i)) {
      if (j.value() > i.value() || ref.occupation(j) < basis::kOccupationCut) continue;
      double r = 0.0;
      for (const OrbitIndex a : particles)
        for (const OrbitIndex b : particles)
          for (const OrbitIndex k : holes) {
            const double eik = x.e(a) + x.e(b) - x.e(i) - x.e(k);
            const double ejk = x.e(a) + x.e(b) - x.e(k) - x.e(j);
            if (std::abs(eik * ejk) < kEps) continue;
            const int jmin = std::max({std::abs(x.j2(a) - x.j2(b)), std::abs(x.j2(i) - x.j2(k)),
                                       std::abs(x.j2(j) - x.j2(k))}) / 2;
            const int jmax =
                std::min({x.j2(a) + x.j2(b), x.j2(i) + x.j2(k), x.j2(j) + x.j2(k)}) / 2;
            double t = 0.0;
            for (int J = jmin; J <= jmax; ++J)
              t += (2 * J + 1) * x.V(a, b, i, k, J) * x.V(j, k, a, b, J);
            const double na = 1.0 - x.n(a), nb = 1.0 - x.n(b);
            t *= na * na * nb * nb * x.n(k) * x.n(k) * x.n(i) * x.n(j);
            r += regulated(t, 0.5 * std::sqrt(std::abs(eik * ejk)));
          }
      const double v = -r * 0.5 / (x.j2(i) + 1);
      rho_hh(i.value(), j.value()) = v;
      rho_hh(j.value(), i.value()) = v;
    }
  }

  // PH block, two terms.
  for (const OrbitIndex i : holes) {
    for (const OrbitIndex a : ms.basis().same_ljtz(i)) {
      if (ref.occupation(a) >= basis::kOccupationCut) continue;  // a particle
      const double eai = x.e(a) - x.e(i);
      double r1 = 0.0;
      for (const OrbitIndex b : particles)
        for (const OrbitIndex c : particles)
          for (const OrbitIndex j : holes) {
            const double ebcij = x.e(b) + x.e(c) - x.e(i) - x.e(j);
            if (eai * ebcij < kEps) continue;
            const int jmin = std::max({std::abs(x.j2(a) - x.j2(j)), std::abs(x.j2(b) - x.j2(c)),
                                       std::abs(x.j2(i) - x.j2(j))}) / 2;
            const int jmax =
                std::min({x.j2(a) + x.j2(j), x.j2(b) + x.j2(c), x.j2(i) + x.j2(j)}) / 2;
            double t = 0.0;
            for (int J = jmin; J <= jmax; ++J)
              t += (2 * J + 1) * x.V(a, j, b, c, J) * x.V(b, c, i, j, J);
            t *= (1 - x.n(a)) * (1 - x.n(b)) * (1 - x.n(c)) * x.n(i) * x.n(j);
            r1 += t / (eai * ebcij);
          }
      double r2 = 0.0;
      for (const OrbitIndex b : particles)
        for (const OrbitIndex j : holes)
          for (const OrbitIndex k : holes) {
            const double eabkj = x.e(a) + x.e(b) - x.e(k) - x.e(j);
            if (eai * eabkj < kEps) continue;
            const int jmin = std::max({std::abs(x.j2(k) - x.j2(j)), std::abs(x.j2(i) - x.j2(b)),
                                       std::abs(x.j2(a) - x.j2(b))}) / 2;
            const int jmax =
                std::min({x.j2(k) + x.j2(j), x.j2(i) + x.j2(b), x.j2(a) + x.j2(b)}) / 2;
            double t = 0.0;
            for (int J = jmin; J <= jmax; ++J)
              t += (2 * J + 1) * x.V(k, j, i, b, J) * x.V(a, b, k, j, J);
            t *= (1 - x.n(a)) * x.n(i) * x.n(j) * x.n(k) * (1 - x.n(b));
            r2 += t / (eai * eabkj);
          }
      const double v = (r1 - r2) * 0.5 / (x.j2(a) + 1);
      rho_ph(a.value(), i.value()) += v;
      rho_ph(i.value(), a.value()) += v;
    }
  }

  for (std::int64_t p = 0; p < N; ++p)
    for (std::int64_t q = 0; q < N; ++q) rho(p, q) += rho_pp(p, q) + rho_hh(p, q) + rho_ph(p, q);
  return rho;
}

NaturalOrbitals natural_orbitals(const linalg::Matrix<double>& rho_hf,
                                 const basis::ModelSpace& ms) {
  const auto& basis = ms.basis();
  const std::int64_t N = ms.num_orbits();
  NaturalOrbitals out{linalg::Matrix<double>(N, N), std::vector<double>(static_cast<size_t>(N), 0.0)};
  double trace = 0.0;
  for (std::int64_t a = 0; a < N; ++a) {
    const OrbitIndex oa(static_cast<std::int32_t>(a));
    const auto idx = basis.same_ljtz(oa);
    if (idx.empty() || idx.front() != oa) continue;  // process each channel once
    const std::int64_t n = static_cast<std::int64_t>(idx.size());
    linalg::Matrix<double> blk(n, n);
    for (std::int64_t r = 0; r < n; ++r)
      for (std::int64_t c = 0; c < n; ++c) blk(r, c) = rho_hf(idx[r].value(), idx[c].value());
    const auto eig = linalg::eigh(blk.const_view());
    const int j2 = basis.orbit(oa).j2.twice;
    for (std::int64_t c = 0; c < n; ++c) {
      const std::int64_t src = n - 1 - c;  // descending occupations
      const double sign = (eig.eigenvectors(c, src) < 0.0) ? -1.0 : 1.0;  // C(i,i) >= 0
      for (std::int64_t r = 0; r < n; ++r)
        out.c_hf2nat(idx[r].value(), idx[c].value()) = sign * eig.eigenvectors(r, src);
      out.occupations[static_cast<size_t>(idx[c].value())] = eig.eigenvalues[static_cast<size_t>(src)];
      trace += (j2 + 1) * eig.eigenvalues[static_cast<size_t>(src)];
    }
  }
  if (std::abs(trace - ms.reference().A()) > 1e-6)
    throw InvariantError("natural_orbitals: density-matrix trace " + std::to_string(trace) +
                         " differs from A = " + std::to_string(ms.reference().A()));
  return out;
}

linalg::Matrix<double> matmul(const linalg::Matrix<double>& a, const linalg::Matrix<double>& b) {
  linalg::Matrix<double> c(a.rows(), b.cols());
  for (std::int64_t i = 0; i < a.rows(); ++i)
    for (std::int64_t k = 0; k < a.cols(); ++k) {
      const double v = a(i, k);
      if (v == 0.0) continue;
      for (std::int64_t j = 0; j < b.cols(); ++j) c(i, j) += v * b(k, j);
    }
  return c;
}

linalg::Matrix<double> reference_density_ho(const linalg::Matrix<double>& c_ho2nat,
                                            const basis::ModelSpace& ms) {
  const auto& ref = ms.reference();
  const std::int64_t N = ms.num_orbits();
  linalg::Matrix<double> rho(N, N);
  for (const OrbitIndex al : ref.holes()) {
    const double n = ref.occupation(al);
    const std::int64_t k = al.value();
    for (std::int64_t a = 0; a < N; ++a) {
      const double ca = n * c_ho2nat(a, k);
      if (ca == 0.0) continue;
      for (std::int64_t b = 0; b < N; ++b) rho(a, b) += ca * c_ho2nat(b, k);
    }
  }
  return rho;
}

op::Operator to_nat_basis(op::Operator& H, const linalg::Matrix<double>& c_hf2nat) {
  H.sync_to_host();
  return op::transform_basis(H, c_hf2nat.const_view());
}

}  // namespace imsrg::mbpt
