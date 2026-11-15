// rh_imsrg3.cpp -- IMSRG(3) [2,2]->3 term (comm223ss) of the OLD imsrg++ for the M10 layout
// benchmark (design/reference/02-commutators.md sections 5.4 and 5.9).
//
//   refharness comm223 emax=N e3max=M reference=O16 x=X.dump y=Y.dump [epsilon=eps.txt] [reps=5]
//
// X (anti-Hermitian) and Y (Hermitian) are 1B/2B OperatorDumps written by the new code. The
// model space is built as for the other subcommands (hw 16, valence = reference, E3max = e3max,
// emax_3body = emax, no dE3max / OccNat cut).
//   Full mode: Z = particle-rank-3 scalar operator with pn three-body storage (Hermitian);
//     Commutator::comm223ss(X, Y, Z) once as warm-up plus `reps` timed calls (the 3B part of Z is
//     zeroed before each call because the term adds). Prints the 3B norm and min/mean times.
//   Perturbative-triples mode (epsilon=FILE, one value per orbit in orbit-index order, '#' starts
//     a comment): Commutator::perturbative_triples = true, Z particle rank 2 with
//     Z.OneBody(i,i) = epsilon_i; the energy comes back in Z.ZeroBody. The flag is reset after.
//     NOTE: imsrg++ itself (IMSRGSolver, HFMBPT) raises E3max to 3 emax for this mode; here
//     e3max is used as given so both modes see the same 3B channels.
// The 6j table is filled by comm223ss itself through ModelSpace::PreCalculateSixJ() before its
// parallel region (a no-op afterwards), so no extra warm-up is needed.

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

namespace {

struct Timing {
  double min_ms = std::numeric_limits<double>::max();
  double mean_ms = 0.0;
};

std::vector<double> read_epsilon(const std::string& path, size_t norbits) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<double> e;
  std::string line;
  while (std::getline(f, line)) {
    auto h = line.find('#');
    if (h != std::string::npos) line.resize(h);
    std::istringstream is(line);
    double v;
    while (is >> v) e.push_back(v);
  }
  if (e.size() != norbits)
    throw std::runtime_error("epsilon file has " + std::to_string(e.size()) + " values, model space has " +
                             std::to_string(norbits) + " orbits");
  return e;
}

}  // namespace

int cmd_comm223(const Args& a) {
  SpaceParams p = space_params(a);
  p.reference = a.str("reference", "O16");
  p.valence_space = a.str("valence_space", p.reference);
  const int reps = a.integer("reps", 5);
  if (reps < 1) throw std::runtime_error("reps must be >= 1");
  if (!a.has("x") || !a.has("y"))
    throw std::runtime_error(
        "usage: refharness comm223 emax=N e3max=M reference=O16 x=X.dump y=Y.dump [epsilon=eps.txt] [reps=5]");

  auto ms = make_modelspace(p);
  Commutator::SetUseIMSRG3(false);
  Operator X = read_opdump(a.str("x", ""), *ms);
  Operator Y = read_opdump(a.str("y", ""), *ms);
  if (X.IsReduced()) X.MakeNotReduced();
  if (Y.IsReduced()) Y.MakeNotReduced();

  std::printf("# refharness comm223 (imsrg++)\n");
  std::printf("emax %d e3max %d reference %s orbits %zu three_body_channels %zu version %s\n", p.emax, p.e3max,
              p.reference.c_str(), ms->all_orbits.size(), (size_t)ms->GetNumberThreeBodyChannels(),
              reference_version().c_str());

  // ---- full mode
  {
    Operator Z(*ms, 0, X.GetTRank() + Y.GetTRank(), (X.GetParity() + Y.GetParity()) % 2, 3);
    Z.SetHermitian();  // anti-Hermitian x Hermitian
    if (Z.IsReduced()) Z.MakeNotReduced();
    Z.ThreeBody.SetMode("pn");  // allocates

    auto call = [&]() {
      Z.ThreeBody.Erase();  // the term adds into Z
      Commutator::comm223ss(X, Y, Z);
    };
    call();  // warm-up (also fills the 6j table)
    Timing t;
    for (int r = 0; r < reps; ++r) {
      const double t0 = omp_get_wtime();
      call();
      const double ms_ = (omp_get_wtime() - t0) * 1e3;
      t.min_ms = std::min(t.min_ms, ms_);
      t.mean_ms += ms_ / reps;
    }
    std::printf("comm223ss_three_body_norm %s\n", fmt17(Z.ThreeBodyNorm()).c_str());
    std::printf("comm223ss_ms %.6g %.6g\n", t.min_ms, t.mean_ms);
  }

  // ---- perturbative triples
  if (a.has("epsilon")) {
    const std::vector<double> eps = read_epsilon(a.str("epsilon", ""), ms->all_orbits.size());
    Operator Z(*ms, 0, X.GetTRank() + Y.GetTRank(), (X.GetParity() + Y.GetParity()) % 2, 2);
    Z.SetHermitian();
    if (Z.IsReduced()) Z.MakeNotReduced();
    for (size_t i = 0; i < eps.size(); ++i) Z.OneBody(i, i) = eps[i];

    auto call = [&]() {
      Z.ZeroBody = 0;
      Commutator::perturbative_triples = true;
      Commutator::comm223ss(X, Y, Z);
      Commutator::perturbative_triples = false;
      return Z.ZeroBody;
    };
    double e = call();  // warm-up
    Timing t;
    for (int r = 0; r < reps; ++r) {
      const double t0 = omp_get_wtime();
      e = call();
      const double ms_ = (omp_get_wtime() - t0) * 1e3;
      t.min_ms = std::min(t.min_ms, ms_);
      t.mean_ms += ms_ / reps;
    }
    std::printf("triples_energy %s\n", fmt17(e).c_str());
    std::printf("triples_ms %.6g %.6g\n", t.min_ms, t.mean_ms);
  }
  std::printf("omp_get_max_threads %d\n", omp_get_max_threads());
  return 0;
}

}  // namespace rh
