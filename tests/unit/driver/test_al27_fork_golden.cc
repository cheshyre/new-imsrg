// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The 27Al NREFT pipeline (design/50, design/reference/06) end to end against
// the goldens of the owner's imsrg++ fork 79e9330 (tests/golden/nreft/flow/
// fork-79e9330): the legacy command line of the golden's cmd.txt with
// run.compat = "imsrg_fork_79e9330" (Al27 ensemble reference, sd-shell, NAT
// basis with the HF holes, NSmagnus, the 11 operators incl. 6 tensors). Every
// number of the written `.snt` files (H, TCM, 4 scalar and 6 tensor NREFT
// operators) is compared with the golden: same set of index rows, relative
// 5e-7 + absolute 1.5e-7 (7 printed significant digits; the fork drops
// |me| < 1e-7), a row present in one file only must be below 2e-7. Skips
// without the input files.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace fs = std::filesystem;

/// The rows of a tokyo-format `.snt` file of the fork (scalar or tensor).
struct ForkSnt {
  double zero_body = 0.0;
  std::map<std::vector<int>, double> one_body;
  std::map<std::vector<int>, double> two_body;
};

ForkSnt ReadFork(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  ForkSnt s;
  std::string line;
  bool data = false;
  int header_lines = 0;
  while (std::getline(in, line)) {
    if (line.rfind("! Zero body term:", 0) == 0) {
      s.zero_body = std::stod(line.substr(17));
      continue;
    }
    if (line.rfind("! interaction", 0) == 0 ||
        line.rfind("! reduced matrix elements", 0) == 0 ||
        line.rfind("! operator", 0) == 0) {
      data = true;
      continue;
    }
    if (!data || line.empty() || line[0] == '!') continue;
    std::istringstream ls(line);
    std::vector<std::string> tok;
    for (std::string t; ls >> t;) tok.push_back(t);
    if (tok.size() == 3 && tok[1] == "0") {  // "<count> 0 <hw>"
      ++header_lines;
      continue;
    }
    std::vector<int> key;
    for (std::size_t i = 0; i + 1 < tok.size(); ++i)
      key.push_back(std::stoi(tok[i]));
    const double v = std::stod(tok.back());
    (header_lines <= 1 ? s.one_body : s.two_body)[key] = v;
  }
  return s;
}

struct Deviation {
  double max_abs = 0.0;
  double max_ratio = 0.0;  // max |a-b| / (5e-7|b| + 1.5e-7)
  std::size_t rows = 0, only_one = 0;
  std::string worst;
};

constexpr double kRel = 5.0e-7, kAbs = 1.5e-7, kOnly = 2.0e-7;

void CompareRows(const std::map<std::vector<int>, double>& a,
                 const std::map<std::vector<int>, double>& b,
                 const std::string& tag, Deviation& d) {
  auto key = [](const std::vector<int>& k) {
    std::string s;
    for (int x : k) s += std::to_string(x) + " ";
    return s;
  };
  for (const auto& [k, va] : a) {
    const auto it = b.find(k);
    const double vb = it == b.end() ? 0.0 : it->second;
    ++d.rows;
    if (it == b.end()) {
      ++d.only_one;
      EXPECT_LT(std::abs(va), kOnly) << tag << " row only computed: " << key(k);
      continue;
    }
    const double dev = std::abs(va - vb);
    const double ratio = dev / (kRel * std::abs(vb) + kAbs);
    if (ratio > d.max_ratio) d.worst = tag + " " + key(k);
    d.max_abs = std::max(d.max_abs, dev);
    d.max_ratio = std::max(d.max_ratio, ratio);
    EXPECT_LE(dev, kRel * std::abs(vb) + kAbs)
        << tag << " row " << key(k) << " computed " << va << " golden " << vb;
  }
  for (const auto& [k, vb] : b) {
    if (a.find(k) == a.end()) {
      ++d.only_one;
      EXPECT_LT(std::abs(vb), kOnly) << tag << " row only golden: " << key(k);
    }
  }
}

/// Runs `driver::Run` with the legacy command line of the golden's cmd.txt
/// and compares every written file.
void RunCase(const std::string& golden_dir, const std::string& tag) {
  fs::path gold = fs::path(IMSRG_GOLDEN_DIR) / "nreft" / "flow" /
                  "fork-79e9330" / golden_dir;
  if (!fs::exists(DataFile(TwoBodyName())) ||
      !fs::exists(DataFile(ThreeBodyName()))) {
    GTEST_SKIP() << "missing input files in " << IMSRG_TEST_DATA_DIR;
  }
  const fs::path dir = FreshDir(tag);
  std::ifstream cmd(gold / "cmd.txt");
  std::string text;
  std::getline(cmd, text);
  std::istringstream ts(text);
  driver::ConfigSources src;
  std::string tok;
  ts >> tok;  // the executable
  while (ts >> tok) {
    const auto eq = tok.find('=');
    const std::string key = tok.substr(0, eq);
    if (key == "scratch" || key == "flowfile" || key == "intfile") continue;
    if (key == "2bme") tok = "2bme=" + DataFile(tok.substr(eq + 1)).string();
    if (key == "3bme") tok = "3bme=" + DataFile(tok.substr(eq + 1)).string();
    src.legacy_args.push_back(tok);
  }
  src.set_overrides = {"run.compat=imsrg_fork_79e9330",
                       "output.directory=" + dir.string(),
                       "output.stem=int"};
  src.command_line = "test_al27_fork_golden " + tag;
  const RunResult r = Execute(src, dir);
  if (r.planned) GTEST_SKIP() << "driver::run not ready: " << r.note;

  std::vector<std::pair<std::string, std::string>> files = {
      {"int.snt", "int.snt"}};
  // Golden file names: int<name>.snt (scalar) and int<name>_2b.snt (tensor);
  // ours: int_<name>.snt / int_<name>_2b.snt.
  for (const auto& entry : fs::directory_iterator(gold)) {
    const std::string f = entry.path().filename().string();
    if (f.rfind("int", 0) != 0 || f.size() < 5 ||
        f.substr(f.size() - 4) != ".snt" || f == "int.snt")
      continue;
    files.emplace_back("int_" + f.substr(3), f);
  }
  EXPECT_EQ(files.size(), 12U);
  for (const auto& [mine, golden] : files) {
    ASSERT_TRUE(fs::exists(dir / mine)) << "not written: " << mine;
    const ForkSnt a = ReadFork(dir / mine);
    const ForkSnt b = ReadFork(gold / golden);
    Deviation d;
    CompareRows(a.one_body, b.one_body, golden + " 1b", d);
    CompareRows(a.two_body, b.two_body, golden + " 2b", d);
    // The zero body is printed with 6 significant digits.
    EXPECT_LE(std::abs(a.zero_body - b.zero_body),
              5.0e-6 * std::abs(b.zero_body) + 1.0e-9)
        << golden << " zero body";
    std::printf("[fork-golden] %-28s rows %5zu (only one side %zu) max dev %.2e"
                " (%.2f of the tolerance) %s\n",
                golden.c_str(), d.rows, d.only_one, d.max_abs, d.max_ratio,
                d.worst.c_str());
  }
}

TEST(Al27ForkGolden, NatEmax2) { RunCase("nat_emax2", "al27_fork_emax2"); }

TEST(Al27ForkGolden, NatEmax4Imsrg3) {
  RunCase("nat_emax4_imsrg3", "al27_fork_emax4");
}

}  // namespace
}  // namespace imsrg::it
