// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// M12b phase 1, task D5: IT-11, the end-to-end MPI invariance tests
// (design/30 section 4). Flows run on 2 and 4 ranks reproduce the serial run
// of the same configuration with identical step counts: every number of
// results.json and every .snt element agrees to 1e-12 max(1, |ref|) (the
// reduce+broadcast of the commutator gives all ranks the same bits, the
// serial run differs only by the summation order of the shares). A second
// part checks at library level that after every step of a Magnus flow all
// ranks hold the same H(s) bit for bit.
//
// Built twice from this source: `driver_mpi_it11_<n>ranks` (fast cases) and,
// with IMSRG_IT11_SLOW, `driver_mpi_it11_slow_<n>ranks` (N7, triples). Skips
// without the 2N input file.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/driver/pipeline/mpi_sync.h"
#include "imsrg/driver/run.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/operator.h"
#include "tests/support/mpi_test_main.h"
#include "tests/support/scratch_dir.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"
#include "tests/unit/driver/integration/snt_compare.h"

namespace imsrg::driver {
namespace {

namespace fs = std::filesystem;
using comm::Communicator;
using it::CompareSnt;
using it::DataFile;
using it::Json;
using it::ParseJson;
using it::ReadSnt;
using it::Slurp;
using it::SntComparison;
using it::TwoBodyName;
using test_support::ScratchDir;
using test_support::TestWorld;

constexpr double kTol = 1.0e-12;

double WallSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

/// Rank 0 creates a scratch directory and broadcasts its path.
std::string BroadcastBase(const Communicator& world,
                          const std::unique_ptr<ScratchDir>& scratch) {
  std::string base = scratch ? scratch->path().string() : std::string();
  world.Broadcast(base, 0);
  return base;
}

/// Config text: `nucleus` at `emax`, hw 16, the 2N file; `tables` are the
/// further TOML tables (complete, each ending with a blank line); output
/// into `dir`.
std::string Toml(const fs::path& dir, const std::string& nucleus, int emax,
                 const std::string& tables) {
  std::ostringstream t;
  t << "[basis]\nemax = " << emax << "\nhw = 16.0\n\n[reference]\nnucleus = \""
    << nucleus << "\"\n\n[interaction.two_body]\nfile = \""
    << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << tables << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  return t.str();
}

RunConfig Load(const std::string& text) {
  ConfigSources src;
  src.config_text = text;
  auto loaded = LoadConfig(src);
  if (!loaded.has_value()) {
    std::string msg = "config rejected:";
    for (const auto& issue : loaded.error()) msg += "\n  " + issue.ToString();
    throw std::runtime_error(msg);
  }
  return loaded.value();
}

/// Accumulates the deviations of two results.json trees.
struct Deviation {
  double max_rel = 0.0;     ///< max |d| / max(1, |ref|) over all numbers
  std::string worst;        ///< path of the largest
  std::size_t numbers = 0;  ///< numbers compared
  std::vector<std::string> mismatches;  ///< structure/exact-count failures
  std::string dir_a, dir_ref;  ///< output directories, masked in strings
};

/// `text` with every occurrence of `dir` replaced by "<out>".
std::string MaskDir(std::string text, const std::string& dir) {
  if (dir.empty()) return text;
  for (std::size_t p = text.find(dir); p != std::string::npos;
       p = text.find(dir, p)) {
    text.replace(p, dir.size(), "<out>");
  }
  return text;
}

bool IsExactKey(const std::string& key) {
  return key.find("steps") != std::string::npos ||
         key.find("gather") != std::string::npos ||
         key.find("count") != std::string::npos ||
         key.find("num_") != std::string::npos;
}

bool IsTimeKey(const std::string& key) {
  return key.find("wall") != std::string::npos ||
         key.find("time") != std::string::npos ||
         key.find("seconds") != std::string::npos;
}

void CompareJson(const Json& a, const Json& ref, const std::string& path,
                 const std::string& key, Deviation& dev) {
  if (a.kind != ref.kind) {
    dev.mismatches.push_back(path + ": kind differs");
    return;
  }
  switch (ref.kind) {
    case Json::Kind::kNumber: {
      if (IsTimeKey(key)) return;
      ++dev.numbers;
      const double d = std::abs(a.number - ref.number);
      if (IsExactKey(key) && d != 0.0) {
        dev.mismatches.push_back(path + ": count " + std::to_string(a.number) +
                                 " vs " + std::to_string(ref.number));
      }
      const double rel = d / std::max(1.0, std::abs(ref.number));
      if (rel > dev.max_rel || std::isnan(rel)) {
        dev.max_rel = rel;
        dev.worst = path;
      }
      return;
    }
    case Json::Kind::kString:
      if (MaskDir(a.text, dev.dir_a) != MaskDir(ref.text, dev.dir_ref) &&
          !IsTimeKey(key)) {
        dev.mismatches.push_back(path + ": '" + a.text + "' vs '" + ref.text +
                                 "'");
      }
      return;
    case Json::Kind::kBool:
      if (a.boolean != ref.boolean) dev.mismatches.push_back(path + ": bool");
      return;
    case Json::Kind::kArray:
      if (a.items.size() != ref.items.size()) {
        dev.mismatches.push_back(path + ": array length");
        return;
      }
      for (std::size_t i = 0; i < ref.items.size(); ++i) {
        CompareJson(a.items[i], ref.items[i],
                    path + "[" + std::to_string(i) + "]", key, dev);
      }
      return;
    case Json::Kind::kObject:
      if (a.fields.size() != ref.fields.size()) {
        dev.mismatches.push_back(path + ": member count");
        return;
      }
      for (const auto& f : ref.fields) {
        if (!a.has(f.first)) {
          dev.mismatches.push_back(path + "." + f.first + ": missing");
          continue;
        }
        CompareJson(a[f.first], f.second, path + "." + f.first, f.first, dev);
      }
      return;
    case Json::Kind::kNull:
      return;
  }
}

/// Compares every .snt file of `serial` with the one of the same name in
/// `mpi`: max|d| <= tol max(1, max|ref|), no missing elements, same orbits.
/// Returns the largest relative deviation; failures are recorded.
double CompareSntFiles(const fs::path& mpi, const fs::path& serial,
                       std::size_t& files) {
  double worst = 0.0;
  files = 0;
  if (!fs::exists(serial)) return worst;
  for (const auto& e : fs::directory_iterator(serial)) {
    if (e.path().extension() != ".snt") continue;
    ++files;
    const fs::path other = mpi / e.path().filename();
    EXPECT_TRUE(fs::exists(other)) << other;
    if (!fs::exists(other)) continue;
    const it::SntData ref = ReadSnt(e.path());
    const it::SntData got = ReadSnt(other);
    double max_ref = std::abs(ref.zero_body);
    for (const auto& kv : ref.one_body) {
      max_ref = std::max(max_ref, std::abs(kv.second));
    }
    for (const auto& kv : ref.two_body) {
      max_ref = std::max(max_ref, std::abs(kv.second));
    }
    const SntComparison c = CompareSnt(got, ref, 0.0);
    const double rel = c.max_dev() / std::max(1.0, max_ref);
    worst = std::max(worst, rel);
    EXPECT_LE(rel, kTol) << e.path().filename() << ": " << c.worst;
    EXPECT_EQ(c.missing, 0u) << e.path().filename();
    EXPECT_TRUE(c.orbits_equal) << e.path().filename();
  }
  return worst;
}

std::int64_t TotalSteps(const Json& results) {
  std::int64_t n = 0;
  for (const auto& st : results["imsrg"]["stages"].items) {
    n += static_cast<std::int64_t>(st["steps"].number);
  }
  return n;
}

std::size_t CountFiles(const fs::path& dir) {
  std::size_t n = 0;
  if (!fs::exists(dir)) return 0;
  for (const auto& e : fs::recursive_directory_iterator(dir)) {
    if (e.is_regular_file()) ++n;
  }
  return n;
}

/// Runs `Run` and turns an exception into a flag (all ranks must agree on
/// failure before anybody waits in a collective).
bool TryRun(const RunConfig& cfg, const Communicator& comm, int rank) {
  try {
    Run(cfg, comm);
    return true;
  } catch (const std::exception& e) {
    ADD_FAILURE() << "rank " << rank << ": " << e.what();
    return false;
  }
}

/// One IT-11 case: all ranks run the configuration `tables` for `nucleus`
/// at `emax` into <scratch>/mpi/rank<r>, rank 0 runs it serially into
/// <scratch>/serial; rank 0 compares and prints a table row.
void RunCase(const char* name, const std::string& nucleus, int emax,
             const std::string& tables) {
  const Communicator& world = TestWorld();
  if (!fs::exists(DataFile(TwoBodyName()))) {
    GTEST_SKIP() << "missing 2N input file";
  }
  std::unique_ptr<ScratchDir> scratch;
  if (world.is_root()) scratch = std::make_unique<ScratchDir>();
  const fs::path base = BroadcastBase(world, scratch);
  const fs::path mdir = base / "mpi" / ("rank" + std::to_string(world.rank()));

  const RunConfig cfg = Load(Toml(mdir, nucleus, emax, tables));
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = TryRun(cfg, world, world.rank());
  const double wall_mpi = WallSince(t0);
  if (world.AllreduceMax(ok ? std::int64_t{0} : std::int64_t{1}) != 0) return;
  world.Barrier();

  if (world.is_root()) {
    const fs::path sdir = base / "serial";
    const RunConfig scfg = Load(Toml(sdir, nucleus, emax, tables));
    const auto t1 = std::chrono::steady_clock::now();
    ASSERT_TRUE(TryRun(scfg, Communicator::Serial(), 0));
    const double wall_serial = WallSince(t1);

    const fs::path m0 = base / "mpi" / "rank0";
    const Json got = ParseJson(Slurp(m0 / "results.json"));
    const Json ref = ParseJson(Slurp(sdir / "results.json"));
    Deviation dev;
    dev.dir_a = m0.string();
    dev.dir_ref = sdir.string();
    CompareJson(got, ref, "results", "", dev);
    for (const auto& m : dev.mismatches) ADD_FAILURE() << name << ": " << m;
    EXPECT_LE(dev.max_rel, kTol) << name << ": worst " << dev.worst;
    EXPECT_GT(dev.numbers, 5u);
    std::size_t nsnt = 0;
    const double snt_dev = CompareSntFiles(m0, sdir, nsnt);
    for (int r = 1; r < world.size(); ++r) {
      EXPECT_EQ(CountFiles(base / "mpi" / ("rank" + std::to_string(r))), 0u)
          << "files written by rank " << r;
    }
    const std::int64_t sm = TotalSteps(got), ss = TotalSteps(ref);
    EXPECT_EQ(sm, ss) << name << ": steps";
    std::printf("[IT-11] %-22s ranks %d  steps %" PRId64 "/%" PRId64
                "  numbers %zu  max dev %.2e "
                "(%s)  snt %zu files dev %.2e  wall mpi %.1f s serial %.1f s\n",
                name, world.size(), sm, ss, dev.numbers, dev.max_rel,
                dev.worst.c_str(), nsnt, snt_dev, wall_mpi, wall_serial);
  }
  world.Barrier();
}

#ifndef IMSRG_IT11_SLOW

const char kMagnus[] =
    "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = \"magnus\"\n\n"
    "[operators]\nlist = [\"Rp2\"]\n\n";

TEST(DriverMpiIt11, He4Emax2Magnus) { RunCase("He4_emax2", "He4", 2, kMagnus); }

TEST(DriverMpiIt11, O16Emax3Magnus) { RunCase("O16_emax3", "O16", 3, kMagnus); }

TEST(DriverMpiIt11, He4Emax2HunterGatherer) {
  RunCase("He4_emax2_HG", "He4", 2,
          "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = "
          "\"magnus\"\n\n[imsrg.step]\nomega_norm_max = 0.1\n\n"
          "[imsrg.omega]\nhunter_gatherer = true\n\n");
}

TEST(DriverMpiIt11, He4Emax2Imsrg3f2) {
  RunCase("He4_emax2_3f2", "He4", 2,
          "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = "
          "\"magnus\"\n\n[imsrg.step]\nsmax = 200\n\n[imsrg3]\nfactorized = "
          "true\n\n");
}

TEST(DriverMpiIt11, He4Emax2FlowRk4) {
  RunCase("He4_emax2_rk4", "He4", 2,
          "[imsrg]\nmethod = \"flow_rk4\"\n\n[operators]\nlist = "
          "[\"Rp2\"]\n\n");
}

TEST(DriverMpiIt11, He4Emax2PShellSnt) {
  RunCase("He4_emax2_pshell", "He4", 2,
          "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = "
          "\"magnus\"\n\n[valence]\nspace = \"p-shell\"\n\n");
}

#else  // IMSRG_IT11_SLOW

const char kN7[] =
    "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = \"magnus\"\n\n"
    "[imsrg.step]\nsmax = 200\n\n[imsrg3]\nin_flow = true\nvariant = "
    "\"n7\"\n\n[operators]\nlist = [\"Rp2\"]\n\n";

TEST(DriverMpiIt11Slow, He4Emax2N7) { RunCase("He4_emax2_N7", "He4", 2, kN7); }

TEST(DriverMpiIt11Slow, O16Emax2N7) { RunCase("O16_emax2_N7", "O16", 2, kN7); }

TEST(DriverMpiIt11Slow, He4Emax2PerturbativeTriples) {
  RunCase("He4_emax2_triples", "He4", 2,
          "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = "
          "\"magnus\"\nperturbative_triples = true\n\n");
}

TEST(DriverMpiIt11Slow, O16Emax3N7TwoRanksOnly) {
  if (TestWorld().size() != 2) GTEST_SKIP() << "2 ranks only";
  RunCase("O16_emax3_N7", "O16", 3, kN7);
}

#endif  // IMSRG_IT11_SLOW

// ---------------------------------------------------------------------------
// Library level: identical bits on all ranks after every step.

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

/// 64-bit FNV-1a of `n` bytes, continuing from `h`.
std::uint64_t Fnv(std::uint64_t h, const void* data, std::size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= kFnvPrime;
  }
  return h;
}

std::uint64_t HashSpan(std::uint64_t h, std::span<const double> s) {
  return Fnv(h, s.data(), s.size_bytes());
}

/// Hash of the zero-body value and all buffers of `h` (host mirrors).
std::uint64_t HashOperator(const op::Operator& h) {
  h.SyncToHost();
  std::uint64_t x = kFnvOffset;
  const double e0 = h.zero_body();
  x = Fnv(x, &e0, sizeof(e0));
  x = HashSpan(x, h.one_body().buffer().host_span());
  x = HashSpan(x, h.two_body().matrix().buffer().host_span());
  if (h.three_body().has_value()) {
    x = HashSpan(x, h.three_body()->buffer().host_span());
  }
  return x;
}

/// Hash of the norms and energy of one flow-log line (not the wall time).
std::uint64_t HashStatus(std::uint64_t x, const flow::FlowStatus& st) {
  const double v[] = {st.s,
                      st.ds,
                      st.e0,
                      st.norm_h,
                      st.norm_eta_1b,
                      st.norm_eta_2b,
                      st.norm_eta,
                      st.norm_eta_3b,
                      st.norm_omega_1b,
                      st.norm_omega_2b,
                      st.norm_omega_3b,
                      st.norm_omega};
  return Fnv(x, v, sizeof(v));
}

/// One step record of a solver run.
struct StepLog {
  std::vector<std::uint64_t> hashes;
  std::vector<flow::FlowStatus> lines;
  int steps = 0;
};

/// Runs a Magnus solver on `h0` with `comm`; every status line records the
/// hash of (H(s), norms).
StepLog RunSolver(const op::Operator& h0,
                  const commutator::CommutatorOptions& c,
                  const Communicator& comm) {
  const commutator::WorkspaceOptions wopts{.comm = comm};
  flow::MagnusSolver solver(h0, flow::SolverOptions{}, flow::GeneratorOptions{},
                            flow::BchOptions{}, c, wopts);
  StepLog log;
  const auto r = solver.Solve([&](const flow::FlowStatus& st) {
    log.hashes.push_back(HashStatus(HashOperator(solver.hamiltonian()), st));
    log.lines.push_back(st);
  });
  log.steps = r.steps;
  return log;
}

std::int64_t AsInt(std::uint64_t h) {
  std::int64_t v = 0;
  std::memcpy(&v, &h, sizeof(v));
  return v;
}

/// The He4 emax 2 flow start (N7: three-body space as the driver does).
op::Operator He4Start(bool n7, commutator::CommutatorOptions& copts) {
  const RunConfig cfg =
      Load(Toml("unused", "He4", 2, "[hartree_fock]\nmixing = \"linear\"\n\n"));
  pipeline::PreparedHamiltonian ph = pipeline::PrepareHamiltonian(cfg);
  pipeline::BroadcastOperator(ph.hno, TestWorld());
  if (!n7) return ph.hno;
  basis::ThreeBodyCuts cuts;
  cuts.e3max = cfg.basis.e3max;
  cuts.emax_3body = -1;
  const std::shared_ptr<const basis::ModelSpace> ms3 =
      ph.hno.model_space_ptr()->WithThreeBody(cuts);
  copts.terms = commutator::TermMask::Imsrg3N7();
  return op::WithModelSpace(ph.hno, ms3);
}

/// After every step all ranks hold the same bits of H(s), and the step count
/// equals that of the serial solver; E0 and the norms agree with it to the
/// tolerance of the end-to-end tests.
void CheckStepIdentity(bool n7, const char* name) {
  const Communicator& world = TestWorld();
  if (!fs::exists(DataFile(TwoBodyName()))) {
    GTEST_SKIP() << "missing 2N input file";
  }
  commutator::CommutatorOptions copts;
  const op::Operator h0 = He4Start(n7, copts);
  const auto t0 = std::chrono::steady_clock::now();
  const StepLog mpi = RunSolver(h0, copts, world);
  const double wall_mpi = WallSince(t0);
  const auto t1 = std::chrono::steady_clock::now();
  const StepLog ser = RunSolver(h0, copts, Communicator::Serial());
  const double wall_ser = WallSince(t1);

  ASSERT_EQ(mpi.hashes.size(), mpi.lines.size());
  std::int64_t first_bad = -1;
  for (std::size_t i = 0; i < mpi.hashes.size(); ++i) {
    const std::vector<std::int64_t> all = world.Allgather(AsInt(mpi.hashes[i]));
    for (const std::int64_t v : all) {
      if (v != all[0] && first_bad < 0)
        first_bad = static_cast<std::int64_t>(i);
    }
  }
  EXPECT_EQ(first_bad, -1) << "ranks diverge at line " << first_bad;
  EXPECT_EQ(mpi.steps, ser.steps);
  ASSERT_EQ(mpi.lines.size(), ser.lines.size());
  double dev = 0.0;
  for (std::size_t i = 0; i < ser.lines.size(); ++i) {
    const flow::FlowStatus &a = mpi.lines[i], &b = ser.lines[i];
    for (const auto& [x, y] : {std::pair{a.e0, b.e0},
                               {a.norm_h, b.norm_h},
                               {a.norm_eta, b.norm_eta},
                               {a.norm_omega, b.norm_omega},
                               {a.ds, b.ds}}) {
      dev = std::max(dev, std::abs(x - y) / std::max(1.0, std::abs(y)));
    }
  }
  EXPECT_LE(dev, kTol);
  if (world.is_root()) {
    std::printf(
        "[IT-11 steps] %-14s ranks %d  steps %d/%d  lines %zu  ranks "
        "identical: %s  max dev vs serial %.2e  wall mpi %.1f s serial %.1f "
        "s\n",
        name, world.size(), mpi.steps, ser.steps, mpi.lines.size(),
        first_bad < 0 ? "yes" : "NO", dev, wall_mpi, wall_ser);
  }
}

#ifndef IMSRG_IT11_SLOW
TEST(FlowMpiSteps, Imsrg2He4Emax2) { CheckStepIdentity(false, "IMSRG2 He4"); }
#else
TEST(FlowMpiSteps, N7He4Emax2) { CheckStepIdentity(true, "N7 He4"); }
#endif

}  // namespace
}  // namespace imsrg::driver
