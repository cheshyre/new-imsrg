// rh_bench.cpp -- IMSRG(2) commutator micro-benchmark of the OLD imsrg++ (design/30 section 5).
//
//   refharness bench-commutator emax=N reps=R [reference=O16] [seed=S]
//
// Counterpart of tools/bench/commutator_bench.cpp: random scalar rank-2 X (anti-Hermitian) and
// Y (Hermitian) from UnitTest::RandomOp, IMSRG(3) off, one warm-up call per term, then R timed
// calls (omp_get_wtime) of every Commutator:: term function and of Commutator::Commutator(X, Y).
// The term functions add into Z, so Z accumulates over the repetitions. Same table format.

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>

#include "IMSRG.hh"
#include "UnitTest.hh"
#include "rh_common.hh"

namespace rh {

namespace {

using TermFn = std::function<void(const Operator&, const Operator&, Operator&)>;

struct Row {
  std::string name;
  double mean_ms = 0.0;
  double min_ms = 0.0;
};

Row time_it(const std::string& name, int reps, const std::function<void()>& fn) {
  fn();  // warm-up
  Row r;
  r.name = name;
  r.min_ms = std::numeric_limits<double>::max();
  double total = 0.0;
  for (int i = 0; i < reps; ++i) {
    const double t0 = omp_get_wtime();
    fn();
    const double ms = (omp_get_wtime() - t0) * 1e3;
    total += ms;
    r.min_ms = std::min(r.min_ms, ms);
  }
  r.mean_ms = total / reps;
  return r;
}

}  // namespace

int cmd_bench_commutator(const Args& a) {
  SpaceParams p = space_params(a);
  p.reference = a.str("reference", "O16");
  p.valence_space = a.str("valence_space", p.reference);
  const int reps = a.integer("reps", 10);
  if (reps < 1) throw std::runtime_error("reps must be >= 1");
  const int seed = a.integer("seed", 1);

  auto ms = make_modelspace(p);
  Commutator::SetUseIMSRG3(false);

  UnitTest ut(*ms);
  ut.SetRandomSeed(static_cast<uint64_t>(seed));
  Operator X = ut.RandomOp(*ms, 0, 0, 0, 2, -1);
  Operator Y = ut.RandomOp(*ms, 0, 0, 0, 2, +1);

  // Z as in Commutator::CommutatorScalarScalar (see rh_commutator.cpp).
  Operator Z(*ms, 0, X.GetTRank() + Y.GetTRank(), (X.GetParity() + Y.GetParity()) % 2, 2);
  Z.SetHermitian();  // anti-Hermitian x Hermitian
  if (Z.IsReduced()) Z.MakeNotReduced();

  size_t n2b = 0;
  for (auto& kv : X.TwoBody.MatEl) n2b += kv.second.n_elem;
  const char* omp = std::getenv("OMP_NUM_THREADS");
  std::printf("# refharness bench-commutator (imsrg++)\n");
  std::printf("emax %d  reference %s  orbits %zu  2B elements %zu  OMP_NUM_THREADS %s  version %s\n",
              p.emax, p.reference.c_str(), ms->all_orbits.size(), n2b, omp ? omp : "(unset)",
              reference_version().c_str());
  std::printf("reps %d\n", reps);

  const std::vector<std::pair<std::string, TermFn>> table = {
      {"comm110ss", Commutator::comm110ss},
      {"comm220ss", Commutator::comm220ss},
      {"comm111ss", Commutator::comm111ss},
      {"comm121ss", Commutator::comm121ss},
      {"comm221ss", Commutator::comm221ss},
      {"comm122ss", Commutator::comm122ss},
      {"comm222_pp_hhss", Commutator::comm222_pp_hhss},
      {"comm222_phss", Commutator::comm222_phss},
      {"comm222_pp_hh_221ss", Commutator::comm222_pp_hh_221ss},
  };
  std::vector<Row> rows;
  for (const auto& t : table) {
    Z.Erase();
    rows.push_back(time_it(t.first, reps, [&] { t.second(X, Y, Z); }));
  }
  rows.push_back(time_it("commutator (full)", reps, [&] {
    Operator W = Commutator::Commutator(X, Y);
    (void)W;
  }));

  std::printf("%-22s %12s %12s\n", "term", "mean ms", "min ms");
  for (const Row& r : rows) std::printf("%-22s %12.4f %12.4f\n", r.name.c_str(), r.mean_ms, r.min_ms);
  return 0;
}

}  // namespace rh
