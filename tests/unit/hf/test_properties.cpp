// Structural properties of the converged Hartree-Fock state (emax 2, O16, 2N).
#include "hf_test_support.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

struct State {
  op::Operator H;
  hf::HFResult r;
};

// Owned by the fixture so that the device buffers die before Kokkos is finalized.
std::unique_ptr<State> g_state;

const State& state() { return *g_state; }

class HfProperties : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    op::Operator H = load_hamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
    hf::HFResult r = hf::solve(H, nullptr);
    g_state = std::make_unique<State>(State{std::move(H), std::move(r)});
  }
  static void TearDownTestSuite() { g_state.reset(); }
};

const basis::Orbit& orbit(std::int64_t i) {
  return state().H.model_space().basis().orbit(OrbitIndex{static_cast<std::int32_t>(i)});
}

TEST_F(HfProperties, DensityIsHermitianWithCorrectTrace) {
  const auto& r = state().r;
  const auto n = r.rho.rows();
  double trace = 0.0, asym = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    trace += (orbit(i).j2.twice + 1.0) * r.rho(i, i);
    for (std::int64_t j = 0; j < n; ++j) asym = std::max(asym, std::abs(r.rho(i, j) - r.rho(j, i)));
  }
  EXPECT_LT(asym, 1e-13);
  EXPECT_NEAR(trace, 16.0, 1e-10);
}

TEST_F(HfProperties, DensityIsIdempotentAtConvergence) {
  const auto& r = state().r;
  const auto n = r.rho.rows();
  double dev = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::int64_t k = 0; k < n; ++k) s += r.rho(i, k) * r.rho(k, j);
      dev = std::max(dev, std::abs(s - r.rho(i, j)));
    }
  }
  std::cout << "[properties] max|rho^2 - rho| " << dev << "\n";
  EXPECT_LT(dev, 1e-10);
}

TEST_F(HfProperties, FockMatrixIsHermitian) {
  const auto& r = state().r;
  double asym = 0.0;
  for (std::int64_t i = 0; i < r.F.rows(); ++i) {
    for (std::int64_t j = 0; j < r.F.cols(); ++j) {
      asym = std::max(asym, std::abs(r.F(i, j) - r.F(j, i)));
    }
  }
  EXPECT_LT(asym, 1e-12);
}

TEST_F(HfProperties, ChannelEnergiesAreAscending) {
  const auto& r = state().r;
  const auto n = static_cast<std::int64_t>(r.spe.size());
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = i + 1; j < n; ++j) {
      const auto &a = orbit(i), &b = orbit(j);
      if (a.l == b.l && a.j2.twice == b.j2.twice && a.tz2.twice == b.tz2.twice) {
        EXPECT_LE(r.spe[static_cast<std::size_t>(i)], r.spe[static_cast<std::size_t>(j)]);
      }
    }
  }
}

TEST_F(HfProperties, CoefficientsAreOrthogonalWithPositiveDiagonal) {
  const auto& r = state().r;
  const auto n = r.C.rows();
  double dev = 0.0;
  for (std::int64_t a = 0; a < n; ++a) {
    EXPECT_GE(r.C(a, a), 0.0);
    for (std::int64_t b = 0; b < n; ++b) {
      double s = 0.0;
      for (std::int64_t i = 0; i < n; ++i) s += r.C(i, a) * r.C(i, b);
      dev = std::max(dev, std::abs(s - (a == b ? 1.0 : 0.0)));
    }
  }
  EXPECT_LT(dev, 1e-12);
}

TEST_F(HfProperties, EnergyEqualsSumOfParts) {
  const auto& r = state().r;
  EXPECT_DOUBLE_EQ(r.e_hf, r.e1 + r.e2 + r.e3);
}

}  // namespace
