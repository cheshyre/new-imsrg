// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Flow checkpoint/restart and Omega dumps through `driver::Run` (design/50
// C5-C7): the 27Al pipeline in miniature (NAT basis with HF holes at emax 4,
// [imsrg.space] emax 2, two Magnus stages with Omega splits). A run killed
// after a checkpoint (testing aid checkpoint.abort_after_steps) and restarted
// reproduces the uninterrupted run bit for bit with native checkpoints, and
// within float32 precision with ME2JP checkpoints. D75: the restart reads the
// prepared Hamiltonian of the checkpoint (no readers, HF or NAT: checked on
// the captured log), SIGTERM writes a checkpoint and stops, and
// checkpoint.prepared_from starts another flow from a stored H_0. Skips
// without the 2N file.
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/termination.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;
namespace fs = std::filesystem;

/// The miniature 27Al configuration; `checkpoint` is appended as given.
std::string Al27Toml(const fs::path& dir, const std::string& extra) {
  return TomlText(dir, "Al27",
                  "[valence]\nspace = \"sd-shell\"\n\n"
                  "[natural_orbitals]\nuse_hf_reference = true\n\n"
                  "[imsrg.space]\nemax = 2\n\n" +
                      extra,
                  false, "0.6\nomega_norm_max = 0.1\neta_criterion = 1e-3",
                  "nat") +
         "write_omega = true\n";
}

RunResult RunWith(const std::string& tag, const std::string& extra,
                  fs::path* out_dir = nullptr) {
  const fs::path dir = FreshDir(tag);
  if (out_dir) *out_dir = dir;
  driver::ConfigSources src;
  src.config_text = Al27Toml(dir, extra);
  return Execute(src, dir);
}

std::string Checkpoint(const fs::path& d, const std::string& format,
                       int abort_after, bool restart) {
  return "[checkpoint]\ndirectory = \"" + d.string() +
         "\"\ninterval_minutes = 0.0\nformat = \"" + format +
         "\"\nabort_after_steps = " + std::to_string(abort_after) +
         "\nrestart = " + (restart ? "true" : "false") + "\n\n";
}

const Json& Stages(const RunResult& r) { return r.results["imsrg"]["stages"]; }

TEST(Checkpoint, NativeRestartIsBitForBit) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  fs::path dir_a;
  const RunResult a = RunWith("ckpt_ref", "", &dir_a);
  ASSERT_FALSE(a.planned) << a.note;
  const int steps1 = static_cast<int>(Stages(a).at(0)["steps"].number);
  const int steps2 = static_cast<int>(Stages(a).at(1)["steps"].number);
  std::printf("[checkpoint] reference: %d + %d steps, %g Omegas, E0 %.12f\n",
              steps1, steps2, a.results["imsrg"]["num_omegas"].number,
              a.results["imsrg"]["e0"].number);
  ASSERT_GE(steps1, 4);
  ASSERT_GE(steps2, 4);
  ASSERT_GE(a.results["imsrg"]["num_omegas"].number, 3.0);
  // Omega manifest and files of write_omega.
  EXPECT_TRUE(fs::exists(dir_a / (a.config.output.stem + "_omegas.toml")));
  EXPECT_TRUE(fs::exists(dir_a / (a.config.output.stem + "_omega_000.opb")));
  EXPECT_TRUE(fs::exists(dir_a / (a.config.output.stem + "_basis.bin")));

  // Kill in stage 1 (step 3) and in stage 2 (2 steps into it); restart.
  for (const int kill : {3, steps1 + 2}) {
    const fs::path ck = FreshDir("ckpt_dir_" + std::to_string(kill));
    EXPECT_THROW(RunWith("ckpt_killed", Checkpoint(ck, "native", kill, false)),
                 std::runtime_error);
    ASSERT_TRUE(fs::exists(ck / "state.txt"));
    fs::path dir_c;
    const RunResult c =
        RunWith("ckpt_restart", Checkpoint(ck, "native", 0, true), &dir_c);
    ASSERT_FALSE(c.planned) << c.note;
    ASSERT_EQ(Stages(c).items.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(Stages(c).at(i)["steps"].number,
                Stages(a).at(i)["steps"].number);
      EXPECT_EQ(Stages(c).at(i)["s"].number, Stages(a).at(i)["s"].number);
      EXPECT_EQ(Stages(c).at(i)["e0"].number, Stages(a).at(i)["e0"].number);
    }
    EXPECT_EQ(c.results["imsrg"]["e0"].number, a.results["imsrg"]["e0"].number);
    // The valence interaction and every Omega file are identical.
    const std::string stem = a.config.output.stem;
    EXPECT_EQ(Slurp(dir_c / (stem + ".snt")), Slurp(dir_a / (stem + ".snt")));
    const int n = static_cast<int>(a.results["imsrg"]["num_omegas"].number);
    for (int k = 0; k < n; ++k) {
      char name[64];
      std::snprintf(name, sizeof(name), "_omega_%03d.opb", k);
      EXPECT_EQ(Slurp(dir_c / (stem + name)), Slurp(dir_a / (stem + name)))
          << "Omega " << k << " after a kill at step " << kill;
    }
  }
}

/// The lines of a text file.
std::vector<std::string> Lines(const fs::path& path) {
  std::vector<std::string> lines;
  std::istringstream in(Slurp(path));
  std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  return lines;
}

/// A flow-file line without its last column (wall_s, not reproducible).
std::string WithoutWall(const std::string& line) {
  return line.substr(0, line.rfind(','));
}

/// Runs the configuration in the existing directory `dir` (no cleaning).
RunResult RunIn(const fs::path& dir, const std::string& extra) {
  driver::ConfigSources src;
  src.config_text = Al27Toml(dir, extra);
  return Execute(src, dir);
}

/// WE-R-1: the flow file of a killed and restarted run (same output
/// directory) equals that of the uninterrupted run, but for wall_s: kept rows
/// up to the checkpointed step, rows written after it dropped.
TEST(Checkpoint, RestartedFlowFileIsContinuous) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  fs::path dir_a;
  const RunResult a = RunWith("flow_ref", "", &dir_a);
  ASSERT_FALSE(a.planned) << a.note;
  const int steps1 = static_cast<int>(Stages(a).at(0)["steps"].number);
  ASSERT_GE(steps1, 4);
  const std::string flow_name = a.config.output.stem + ".flow";
  const std::vector<std::string> ref = Lines(dir_a / flow_name);
  ASSERT_GT(ref.size(), static_cast<std::size_t>(steps1 + 4));

  // Stage 1 (step 3), the end of stage 1, the stage-2 start and its middle.
  for (const int kill : {3, steps1, steps1 + 1, steps1 + 2}) {
    const fs::path ck = FreshDir("flow_ck_" + std::to_string(kill));
    const fs::path dir = FreshDir("flow_run_" + std::to_string(kill));
    EXPECT_THROW(RunIn(dir, Checkpoint(ck, "native", kill, false)),
                 std::runtime_error);
    const std::vector<std::string> killed = Lines(dir / flow_name);
    ASSERT_GT(killed.size(), 1u);
    {
      // A row past the checkpoint, and a torn row.
      std::ofstream out(dir / flow_name, std::ios::app);
      const std::string& last = killed.back();
      const std::size_t c1 = last.find(',');
      const std::size_t c2 = last.find(',', c1 + 1);
      out << last.substr(0, c1 + 1) << "99" << last.substr(c2) << "\n"
          << last.substr(0, 7);
    }
    const RunResult c = RunIn(dir, Checkpoint(ck, "native", 0, true));
    ASSERT_FALSE(c.planned) << c.note;
    const std::vector<std::string> got = Lines(dir / flow_name);
    ASSERT_EQ(got.size(), ref.size()) << "kill at step " << kill;
    EXPECT_EQ(got[0], ref[0]);
    for (std::size_t i = 1; i < ref.size(); ++i) {
      EXPECT_EQ(WithoutWall(got[i]), WithoutWall(ref[i]))
          << "row " << i << " after a kill at step " << kill;
    }
  }
}

/// Kills a run after `kill` steps, rewrites its flow file (`lines`, header
/// first) with `mutate` (setting `*remove` deletes the file instead), restarts
/// and returns the restarted flow file; `log` receives the captured stdout of
/// the restart.
std::vector<std::string> RestartWithFlowFile(
    const std::string& tag, const std::string& flow_name, int kill,
    const std::function<void(std::vector<std::string>*, bool*)>& mutate,
    std::string* log) {
  const fs::path ck = FreshDir(tag + "_ck");
  const fs::path dir = FreshDir(tag + "_run");
  EXPECT_THROW(RunIn(dir, Checkpoint(ck, "native", kill, false)),
               std::runtime_error);
  std::vector<std::string> lines = Lines(dir / flow_name);
  bool remove = false;
  mutate(&lines, &remove);
  if (remove) {
    fs::remove(dir / flow_name);
  } else {
    std::ofstream out(dir / flow_name);
    for (std::size_t i = 0; i < lines.size(); ++i) {
      out << lines[i] << (i + 1 < lines.size() ? "\n" : "");  // keeps a torn
    }
  }
  driver::ConfigSources src;
  src.config_text = Al27Toml(dir, Checkpoint(ck, "native", 0, true));
  testing::internal::CaptureStdout();
  try {
    const RunResult c = Execute(src, dir);
    EXPECT_FALSE(c.planned) << c.note;
  } catch (...) {
    *log = testing::internal::GetCapturedStdout();
    throw;
  }
  *log = testing::internal::GetCapturedStdout();
  return Lines(dir / flow_name);
}

/// The torn row directly after the last valid row (the resume step) is
/// dropped by the field-count check; the restarted file equals the straight
/// one and nothing is warned. A missing file, the file of another run (other
/// s at the resume step) and a file with a gap (no row for the resume step)
/// start afresh at the resume step with a warning.
TEST(Checkpoint, RestartedFlowFileProvenance) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  fs::path dir_a;
  const RunResult a = RunWith("prov_ref", "", &dir_a);
  ASSERT_FALSE(a.planned) << a.note;
  const std::string flow_name = a.config.output.stem + ".flow";
  const std::vector<std::string> ref = Lines(dir_a / flow_name);
  const std::string kWarning = "afresh at the resume step";
  const int kill = 3;

  {  // A torn row right after the last valid row.
    std::string log;
    const auto got = RestartWithFlowFile(
        "prov_torn", flow_name, kill,
        [](std::vector<std::string>* lines, bool*) {
          lines->push_back(lines->back().substr(0, 7));
        },
        &log);
    EXPECT_EQ(log.find(kWarning), std::string::npos) << log;
    ASSERT_EQ(got.size(), ref.size());
    EXPECT_EQ(got[0], ref[0]);
    for (std::size_t i = 1; i < ref.size(); ++i) {
      EXPECT_EQ(WithoutWall(got[i]), WithoutWall(ref[i])) << "row " << i;
    }
  }

  // Each of these: warning, fresh file whose first row is the step after the
  // resume step (stage 1, step 4).
  using Mutation = std::function<void(std::vector<std::string>*, bool*)>;
  const auto expect_fresh = [&](const std::string& tag,
                                const Mutation& mutate) {
    std::string log;
    const auto got = RestartWithFlowFile(tag, flow_name, kill, mutate, &log);
    EXPECT_NE(log.find(kWarning), std::string::npos) << tag << "\n" << log;
    ASSERT_GT(got.size(), 1u) << tag;
    EXPECT_EQ(got[0], ref[0]) << tag;
    EXPECT_EQ(got[1].substr(0, 4), "1,4,") << tag;
    // The rows after the resume step equal the straight run's.
    ASSERT_LT(got.size(), ref.size()) << tag;
    for (std::size_t i = 1; i < got.size(); ++i) {
      EXPECT_EQ(WithoutWall(got[got.size() - i]),
                WithoutWall(ref[ref.size() - i]))
          << tag << " row from the end " << i;
    }
  };
  expect_fresh("prov_missing",
               [](std::vector<std::string>*, bool* remove) { *remove = true; });
  expect_fresh("prov_other_run", [](std::vector<std::string>* lines, bool*) {
    std::string& last = lines->back();
    const std::size_t c1 = last.find(',');
    const std::size_t c2 = last.find(',', c1 + 1);
    const std::size_t c3 = last.find(',', c2 + 1);
    last = last.substr(0, c2 + 1) + "0.123" + last.substr(c3);
  });
  expect_fresh("prov_gap", [](std::vector<std::string>* lines, bool*) {
    lines->pop_back();  // no row for the resume step
  });
}

/// A run without restart truncates an existing flow file; a restart whose
/// flow file has another header starts it afresh.
TEST(Checkpoint, FlowFileTruncatedUnlessRestart) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  fs::path dir_a;
  const RunResult a = RunWith("flow_trunc_ref", "", &dir_a);
  ASSERT_FALSE(a.planned) << a.note;
  const std::string flow_name = a.config.output.stem + ".flow";
  const std::vector<std::string> ref = Lines(dir_a / flow_name);

  const fs::path dir = FreshDir("flow_trunc");
  {
    std::ofstream out(dir / flow_name);
    for (int i = 0; i < 400; ++i) out << "stale," << i << "\n";
  }
  const RunResult b = RunIn(dir, "");
  ASSERT_FALSE(b.planned) << b.note;
  const std::vector<std::string> got = Lines(dir / flow_name);
  ASSERT_EQ(got.size(), ref.size());
  EXPECT_EQ(got[0], ref[0]);

  // Restart over a file with a foreign header: fresh file, restarted rows.
  const fs::path ck = FreshDir("flow_trunc_ck");
  const fs::path dir_k = FreshDir("flow_trunc_run");
  EXPECT_THROW(RunIn(dir_k, Checkpoint(ck, "native", 3, false)),
               std::runtime_error);
  {
    std::ofstream out(dir_k / flow_name);
    out << "a,b,c\n1,2,3\n";
  }
  const RunResult c = RunIn(dir_k, Checkpoint(ck, "native", 0, true));
  ASSERT_FALSE(c.planned) << c.note;
  const std::vector<std::string> fresh = Lines(dir_k / flow_name);
  EXPECT_EQ(fresh[0], ref[0]);
  EXPECT_LT(fresh.size(), ref.size());
  EXPECT_EQ(fresh[1].substr(0, 4), "1,4,");
}

TEST(Checkpoint, OmegaSecondFormatWritesBoth) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const fs::path dir = FreshDir("ckpt_both");
  driver::ConfigSources src;
  src.config_text = Al27Toml(dir, "") + "omega_second_format = true\n";
  const RunResult r = Execute(src, dir);
  ASSERT_FALSE(r.planned) << r.note;
  const std::string stem = r.config.output.stem;
  EXPECT_TRUE(fs::exists(dir / (stem + "_omegas.toml")));
  EXPECT_TRUE(fs::exists(dir / (stem + "_omega_000.opb")));
  EXPECT_TRUE(fs::exists(dir / (stem + "_me2jp_omegas.toml")));
  EXPECT_TRUE(fs::exists(dir / (stem + "_me2jp_omega_000.me2jp.bin")));
  EXPECT_TRUE(fs::exists(dir / (stem + "_me2jp_omega_000.me1j")));
}

TEST(Checkpoint, Me2jpRestartWithinFloat32) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const RunResult a = RunWith("ckpt32_ref", "");
  ASSERT_FALSE(a.planned) << a.note;
  const int steps1 = static_cast<int>(Stages(a).at(0)["steps"].number);
  const fs::path ck = FreshDir("ckpt32_dir");
  EXPECT_THROW(
      RunWith("ckpt32_killed", Checkpoint(ck, "me2jp", steps1 + 2, false)),
      std::runtime_error);
  ASSERT_TRUE(fs::exists(ck / "state.txt"));
  const RunResult c =
      RunWith("ckpt32_restart", Checkpoint(ck, "me2jp", 0, true));
  ASSERT_FALSE(c.planned) << c.note;
  const double e0a = a.results["imsrg"]["e0"].number;
  const double e0c = c.results["imsrg"]["e0"].number;
  std::printf(
      "[checkpoint me2jp] E0 %.12f vs %.12f (diff %.3e), steps %g vs %g\n", e0c,
      e0a, e0c - e0a, Stages(c).at(1)["steps"].number,
      Stages(a).at(1)["steps"].number);
  // float32 two-body parts (~6e-8 relative per element): owner O4 accepts a
  // restart within float32 precision, E0 to ~1e-6 MeV (observed 7e-9 MeV).
  EXPECT_NEAR(e0c, e0a, 1.0e-6);
  EXPECT_NEAR(Stages(c).at(1)["steps"].number, Stages(a).at(1)["steps"].number,
              3.0);
}

TEST(Checkpoint, RestartRejectsAnotherConfiguration) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const fs::path ck = FreshDir("ckpt_cfg_dir");
  EXPECT_THROW(RunWith("ckpt_cfg_killed", Checkpoint(ck, "native", 2, false)),
               std::runtime_error);
  // Another eta_criterion: refused.
  const fs::path dir = FreshDir("ckpt_cfg_restart");
  driver::ConfigSources src;
  std::string text = Al27Toml(dir, Checkpoint(ck, "native", 0, true));
  const std::string from = "eta_criterion = 1e-3";
  text.replace(text.find(from), from.size(), "eta_criterion = 2e-3");
  src.config_text = text;
  EXPECT_THROW(Execute(src, dir), std::runtime_error);
}

// ---- Prepared Hamiltonian in the checkpoint, SIGTERM (design/00 D75).

/// The miniature 27Al configuration with basis.kind `kind` ("nat" as
/// `Al27Toml`, or "hf") and the Rp2 operator (it exercises C and the
/// normal-ordering space); `omega_norm_max` and `extra` as given.
std::string Al27KindToml(const fs::path& dir, const std::string& kind,
                         const std::string& extra,
                         const std::string& omega_norm_max = "0.1") {
  const std::string nat =
      kind == "nat" ? "[natural_orbitals]\nuse_hf_reference = true\n\n" : "";
  return TomlText(dir, "Al27",
                  "[valence]\nspace = \"sd-shell\"\n\n" + nat +
                      "[imsrg.space]\nemax = 2\n\n"
                      "[operators]\nlist = [\"Rp2\"]\n\n" +
                      extra,
                  false,
                  "0.6\nomega_norm_max = " + omega_norm_max +
                      "\neta_criterion = 1e-3",
                  kind) +
         "write_omega = true\n";
}

/// Runs `toml` (written for `dir`); with `log` the captured stdout.
RunResult RunToml(const fs::path& dir, const std::string& toml,
                  std::string* log = nullptr) {
  driver::ConfigSources src;
  src.config_text = toml;
  if (log == nullptr) return Execute(src, dir);
  testing::internal::CaptureStdout();
  try {
    RunResult r = Execute(src, dir);
    *log = testing::internal::GetCapturedStdout();
    return r;
  } catch (...) {
    *log = testing::internal::GetCapturedStdout();
    throw;
  }
}

/// `text` with every occurrence of `from` replaced by `to`.
std::string ReplaceAll(std::string text, const std::string& from,
                       const std::string& to) {
  for (std::size_t p = text.find(from); p != std::string::npos;
       p = text.find(from, p + to.size())) {
    text.replace(p, from.size(), to);
  }
  return text;
}

/// Expects the outputs of run `c` (in `dir_c`) to equal those of `a` (in
/// `dir_a`) bit for bit: results.json (output paths mapped), the valence
/// interaction, the Rp2 file and every Omega file.
void ExpectSameOutputs(const RunResult& a, const fs::path& dir_a,
                       const RunResult& c, const fs::path& dir_c) {
  EXPECT_EQ(c.results["imsrg"]["e0"].number, a.results["imsrg"]["e0"].number);
  EXPECT_EQ(
      ReplaceAll(Slurp(dir_c / "results.json"), dir_c.string(), dir_a.string()),
      Slurp(dir_a / "results.json"));
  const std::string stem = a.config.output.stem;
  EXPECT_EQ(Slurp(dir_c / (stem + ".snt")), Slurp(dir_a / (stem + ".snt")));
  EXPECT_EQ(Slurp(dir_c / (stem + "_Rp2.snt")),
            Slurp(dir_a / (stem + "_Rp2.snt")));
  const int n = static_cast<int>(a.results["imsrg"]["num_omegas"].number);
  for (int k = 0; k < n; ++k) {
    char name[64];
    std::snprintf(name, sizeof(name), "_omega_%03d.opb", k);
    EXPECT_EQ(Slurp(dir_c / (stem + name)), Slurp(dir_a / (stem + name)))
        << "Omega " << k;
  }
}

/// True if the log shows that stage 1 (readers, HF, NAT) ran.
bool PreparedHere(const std::string& log) {
  return log.find("HF: E =") != std::string::npos ||
         log.find("EMP2 =") != std::string::npos;
}

void KillAndRestartFromPreparedState(const std::string& kind) {
  const fs::path dir_a = FreshDir("prep_ref_" + kind);
  const RunResult a = RunToml(dir_a, Al27KindToml(dir_a, kind, ""));
  ASSERT_FALSE(a.planned) << a.note;

  const fs::path ck = FreshDir("prep_ckpt_" + kind);
  const fs::path dir_k = FreshDir("prep_killed_" + kind);
  EXPECT_THROW(RunToml(dir_k, Al27KindToml(dir_k, kind,
                                           Checkpoint(ck, "native", 3, false))),
               std::runtime_error);
  ASSERT_TRUE(fs::exists(ck / "state.txt"));
  const fs::path prepared = ck / "prepared";
  EXPECT_TRUE(fs::exists(prepared / "prepared.toml"));
  EXPECT_TRUE(fs::exists(prepared / "hno.opb"));
  EXPECT_TRUE(fs::exists(prepared / "c_hf.bin"));
  EXPECT_EQ(fs::exists(prepared / "c_nat.bin"), kind == "nat");
  EXPECT_FALSE(fs::exists(ck / "prepared.tmp"));

  const fs::path dir_c = FreshDir("prep_restart_" + kind);
  std::string log;
  const RunResult c = RunToml(
      dir_c, Al27KindToml(dir_c, kind, Checkpoint(ck, "native", 0, true)),
      &log);
  ASSERT_FALSE(c.planned) << c.note;
  // No preparation on restart: the log line of the stored state, and none of
  // the HF / EMP2 lines that stage 1 always prints.
  EXPECT_NE(log.find("restart: prepared Hamiltonian read from"),
            std::string::npos);
  EXPECT_FALSE(PreparedHere(log)) << log.substr(0, 4000);
  ExpectSameOutputs(a, dir_a, c, dir_c);
}

TEST(Checkpoint, PreparedRestartNatSkipsPreparation) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  KillAndRestartFromPreparedState("nat");
}

TEST(Checkpoint, PreparedRestartHfSkipsPreparation) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  KillAndRestartFromPreparedState("hf");
}

TEST(Checkpoint, RestartWithoutPreparedStatePrepares) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  // A checkpoint of an older build has no prepared/: the restart prepares.
  const fs::path ck = FreshDir("prep_old_ckpt");
  const fs::path dir_k = FreshDir("prep_old_killed");
  EXPECT_THROW(RunToml(dir_k, Al27KindToml(dir_k, "hf",
                                           Checkpoint(ck, "native", 2, false))),
               std::runtime_error);
  fs::remove_all(ck / "prepared");
  const fs::path dir_c = FreshDir("prep_old_restart");
  std::string log;
  const RunResult c = RunToml(
      dir_c, Al27KindToml(dir_c, "hf", Checkpoint(ck, "native", 0, true)),
      &log);
  ASSERT_FALSE(c.planned) << c.note;
  EXPECT_NE(log.find("restart: no prepared Hamiltonian in"), std::string::npos);
  EXPECT_TRUE(PreparedHere(log));
  EXPECT_TRUE(fs::exists(ck / "prepared" / "prepared.toml"));
}

TEST(Checkpoint, SigtermWritesCheckpointAndRestartIsBitForBit) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const fs::path dir_a = FreshDir("term_ref");
  const RunResult a = RunToml(dir_a, Al27KindToml(dir_a, "nat", ""));
  ASSERT_FALSE(a.planned) << a.note;

  // SIGTERM arrives during the preparation: the flow stops after its first
  // step. interval_minutes = 60 and no abort_after_steps: only the
  // termination request writes the checkpoint.
  const fs::path ck = FreshDir("term_ckpt");
  const std::string ckpt_keys = "[checkpoint]\ndirectory = \"" + ck.string() +
                                "\"\ninterval_minutes = 60.0\n\n";
  driver::InstallTerminationHandler();
  std::raise(SIGTERM);
  EXPECT_TRUE(driver::TerminationRequested());
  const fs::path dir_k = FreshDir("term_killed");
  EXPECT_THROW(RunToml(dir_k, Al27KindToml(dir_k, "nat", ckpt_keys)),
               InterruptedError);
  driver::RestoreDefaultTerminationHandler();
  driver::ClearTerminationRequest();
  ASSERT_TRUE(fs::exists(ck / "state.txt"));
  EXPECT_NE(Slurp(ck / "state.txt").find("loop 1 1 "), std::string::npos);

  const fs::path dir_c = FreshDir("term_restart");
  const std::string restart_keys = "[checkpoint]\ndirectory = \"" +
                                   ck.string() +
                                   "\"\ninterval_minutes = 60.0\n"
                                   "restart = true\n\n";
  const RunResult c = RunToml(dir_c, Al27KindToml(dir_c, "nat", restart_keys));
  ASSERT_FALSE(c.planned) << c.note;
  ExpectSameOutputs(a, dir_a, c, dir_c);
}

TEST(Checkpoint, SigtermWithoutCheckpointDirectoryStops) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  driver::RequestTermination();
  const fs::path dir = FreshDir("term_nockpt");
  EXPECT_THROW(RunToml(dir, Al27KindToml(dir, "hf", "")), InterruptedError);
  driver::ClearTerminationRequest();
  EXPECT_FALSE(fs::exists(dir / "results.json"));
}

TEST(Checkpoint, PreparedFromAnotherRunIsBitForBit) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  // Run A (omega_norm_max 0.1) writes its prepared state.
  const fs::path ck_a = FreshDir("from_ckpt_a");
  const fs::path dir_a = FreshDir("from_run_a");
  const std::string ckpt_a = "[checkpoint]\ndirectory = \"" + ck_a.string() +
                             "\"\ninterval_minutes = 60.0\n\n";
  const RunResult a = RunToml(dir_a, Al27KindToml(dir_a, "nat", ckpt_a));
  ASSERT_FALSE(a.planned) << a.note;
  ASSERT_TRUE(fs::exists(ck_a / "prepared" / "prepared.toml"));

  // Run B (omega_norm_max 0.2) from scratch, and from A's prepared state.
  const fs::path dir_b = FreshDir("from_run_b");
  const RunResult b = RunToml(dir_b, Al27KindToml(dir_b, "nat", "", "0.2"));
  ASSERT_FALSE(b.planned) << b.note;
  const fs::path ck_c = FreshDir("from_ckpt_c");
  const fs::path dir_c = FreshDir("from_run_c");
  const std::string ckpt_c = "[checkpoint]\ndirectory = \"" + ck_c.string() +
                             "\"\ninterval_minutes = 60.0\nprepared_from = \"" +
                             ck_a.string() + "\"\n\n";
  std::string log;
  const RunResult c =
      RunToml(dir_c, Al27KindToml(dir_c, "nat", ckpt_c, "0.2"), &log);
  ASSERT_FALSE(c.planned) << c.note;
  EXPECT_NE(log.find("(checkpoint.prepared_from)"), std::string::npos);
  EXPECT_FALSE(PreparedHere(log)) << log.substr(0, 4000);
  ExpectSameOutputs(b, dir_b, c, dir_c);
  // C keeps its own prepared state (restartable on its own), equal to A's.
  EXPECT_EQ(Slurp(ck_c / "prepared" / "hno.opb"),
            Slurp(ck_a / "prepared" / "hno.opb"));
  EXPECT_EQ(Slurp(ck_c / "prepared" / "prepared.toml"),
            Slurp(ck_a / "prepared" / "prepared.toml"));

  // Another Hamiltonian (hartree_fock.mixing changes the hash): refused.
  const fs::path dir_d = FreshDir("from_run_d");
  std::string toml = Al27KindToml(dir_d, "nat", ckpt_c, "0.2");
  toml = ReplaceAll(toml, "mixing = \"linear\"", "mixing = \"diis\"");
  EXPECT_THROW(RunToml(dir_d, toml), InputError);
}

}  // namespace
}  // namespace imsrg::it
