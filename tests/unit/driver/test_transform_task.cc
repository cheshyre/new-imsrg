// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// run.task = "transform" (design/50 C10) against the in-process transform:
// the 27Al pipeline in miniature (NAT basis with HF holes at emax 4,
// [imsrg.space] emax 2, ensemble reference re-normal ordered to the 16O core,
// two Magnus stages with Omega splits, fork compat profile) writes its Omegas
// and the .snt of scalar and tensor operators; the transform task on the
// stored Omegas writes the same files: byte for byte with native Omegas,
// within float32 precision with ME2JP Omegas. Skips without the 2N file.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/run.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;
namespace fs = std::filesystem;

/// Operators of the test: scalars (TCM has a two-body part) and tensors of
/// rank 1 and 2 (NREFT shorthands, one-body at basis.emax).
const std::vector<std::string>& OperatorNames() {
  static const auto* const kNames =
      new std::vector<std::string>{"TCM",      "FM0p_150",      "FPhipp0n_150",
                                   "FM2p_150", "FSigmap1p_150", "FDelta1p_150"};
  return *kNames;
}

bool IsTensor(const std::string& name) {
  return name != "TCM" && name.find("0p_") == std::string::npos &&
         name.find("0n_") == std::string::npos;
}

std::string OperatorList() {
  std::string s = "[operators]\nlist = [";
  for (std::size_t i = 0; i < OperatorNames().size(); ++i) {
    s += (i ? ", \"" : "\"") + OperatorNames()[i] + "\"";
  }
  return s + "]\n\n";
}

/// The miniature 27Al configuration of test_checkpoint.cc plus operators and
/// the fork profile; `run_extra` goes into `[run]`, `tail` after `[output]`.
std::string Al27Toml(const fs::path& dir, const std::string& run_extra,
                     const std::string& tail) {
  return "[run]\ncompat = \"imsrg_fork_79e9330\"\n" + run_extra + "\n" +
         TomlText(dir, "Al27",
                  "[valence]\nspace = \"sd-shell\"\n\n"
                  "[natural_orbitals]\nuse_hf_reference = true\n\n"
                  "[imsrg.space]\nemax = 2\n\n" +
                      OperatorList(),
                  false, "0.6\nomega_norm_max = 0.1\neta_criterion = 1e-3",
                  "nat") +
         "stem = \"int\"\n" + tail;
}

void RunToml(const std::string& toml) {
  driver::ConfigSources src;
  src.config_text = toml;
  auto loaded = driver::LoadConfig(src);
  ASSERT_TRUE(loaded.has_value())
      << (loaded.has_value() ? std::string()
                             : loaded.error().front().ToString());
  driver::Run(loaded.value());
}

/// Numeric rows of a .snt file: key = all tokens but the last, value = last
/// token (comment lines and count lines skipped).
std::map<std::string, double> SntRows(const fs::path& path) {
  std::map<std::string, double> rows;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '!' || line[0] == '#') continue;
    std::istringstream s(line);
    std::vector<std::string> tok;
    for (std::string t; s >> t;) tok.push_back(t);
    if (tok.size() < 3 || tok.back().find_first_of(".eE") == std::string::npos)
      continue;
    std::string key;
    for (std::size_t i = 0; i + 1 < tok.size(); ++i) key += tok[i] + " ";
    rows[key] = std::stod(tok.back());
  }
  return rows;
}

TEST(TransformTask, MatchesInProcessTransform) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  for (const std::string format : {"native", "me2jp"}) {
    SCOPED_TRACE(format);
    const fs::path flow_dir = FreshDir("transform_flow_" + format);
    RunToml(
        Al27Toml(flow_dir, "",
                 "write_omega = true\nomega_format = \"" + format + "\"\n"));
    const fs::path manifest = flow_dir / "int_omegas.toml";
    ASSERT_TRUE(fs::exists(manifest));
    ASSERT_TRUE(fs::exists(flow_dir / "int_occ_no.dat"));

    const fs::path farm_dir = FreshDir("transform_farm_" + format);
    RunToml(
        Al27Toml(farm_dir, "task = \"transform\"\n",
                 "\n[transform]\nmanifest = \"" + manifest.string() + "\"\n"));
    // The failure manifest lists nothing.
    const auto failed = SntRows(farm_dir / "failed.txt");
    EXPECT_TRUE(failed.empty());

    for (const std::string& name : OperatorNames()) {
      SCOPED_TRACE(name);
      const std::string suffix = IsTensor(name) ? "_2b.snt" : ".snt";
      const fs::path in_process = flow_dir / ("int_" + name + suffix);
      const fs::path farm = farm_dir / ("int" + name + suffix);
      ASSERT_TRUE(fs::exists(in_process)) << in_process;
      ASSERT_TRUE(fs::exists(farm)) << farm;
      if (format == "native") {
        EXPECT_EQ(Slurp(farm), Slurp(in_process));
        continue;
      }
      // ME2JP stores the Omegas in float32 (relative rounding 6e-8 per
      // element); the transformed elements move by the same relative amount
      // times the BCH amplification. Report the deviation and bound it far
      // below the 1e-4 reproduction target of design/50 section 2.
      const auto a = SntRows(farm);
      const auto b = SntRows(in_process);
      double dmax = 0.0;
      std::size_t missing = 0;
      for (const auto& [k, v] : b) {
        const auto it = a.find(k);
        if (it == a.end()) {
          missing += std::abs(v) > 2.0e-7 ? 1 : 0;
          continue;
        }
        dmax = std::max(dmax, std::abs(it->second - v));
      }
      std::printf("[transform] me2jp %-14s rows %zu max|d| %.3e\n",
                  name.c_str(), b.size(), dmax);
      EXPECT_EQ(missing, 0U);
      EXPECT_LE(dmax, 1.0e-5);
    }
  }
}

TEST(TransformTask, FailureManifestListsBrokenItems) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const fs::path flow_dir = FreshDir("transform_fail_flow");
  RunToml(Al27Toml(flow_dir, "", "write_omega = true\n"));
  const fs::path farm_dir = FreshDir("transform_fail_farm");
  const fs::path work = farm_dir / "work.txt";
  {
    std::ofstream w(work);
    // A good item, a name that does not parse, an operator that parses but
    // throws while it is built (E0 is not implemented in the pipeline), and
    // a q that is not a number.
    w << "# work list\nFM0n_150\n\nFNoSuchFamily0p_150\nE0\nFM0p_abc\n";
  }
  // operators.list stays as in the flow; the work list is appended.
  RunToml(Al27Toml(farm_dir, "task = \"transform\"\n",
                   "\n[transform]\nmanifest = \"" +
                       (flow_dir / "int_omegas.toml").string() +
                       "\"\nwork_list = \"" + work.string() + "\"\n"));
  std::ifstream in(farm_dir / "failed.txt");
  std::vector<std::string> failed;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    failed.push_back(line.substr(0, line.find('\t')));
  }
  EXPECT_EQ(failed, (std::vector<std::string>{"FNoSuchFamily0p_150", "E0",
                                              "FM0p_abc"}));
  EXPECT_TRUE(fs::exists(farm_dir / "intFM0n_150.snt"));
}

}  // namespace
}  // namespace imsrg::it
