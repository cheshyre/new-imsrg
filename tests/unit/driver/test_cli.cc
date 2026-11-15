// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// App-level tests of the `imsrg` executable (design/16 section 6.3 item 10):
// spawn it in a fresh temporary directory and check stdout, stderr, files and
// exit codes.
#include <sys/wait.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace fs = std::filesystem;

namespace {

constexpr const char* kMe2j =
    "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
constexpr const char* kStream =
    "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";

std::string Slurp(const fs::path& p) {
  std::ifstream in(p);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Result {
  int code = -1;
  std::string out;
  std::string err;
};

/// A fresh temporary working directory, removed on destruction.
class TempDir {
 public:
  explicit TempDir(bool with_inputs = true) {
    static std::atomic<int> counter{0};
    path_ = fs::temp_directory_path() /
            ("imsrg_cli_test_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "_" + std::to_string(counter++));
    fs::create_directories(path_);
    if (with_inputs) {
      std::ofstream(path_ / kMe2j).put('\n');
      std::ofstream(path_ / kStream).put('\n');
    }
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  const fs::path& path() const { return path_; }

  /// Runs the executable with `args` (a shell fragment) in this directory.
  Result Run(const std::string& args) const {
    const std::string cmd = "cd '" + path_.string() + "' && '" +
                            IMSRG_EXE_PATH + "' " + args +
                            " > out.txt 2> err.txt";
    const int status = std::system(cmd.c_str());
    Result r;
    r.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    r.out = Slurp(path_ / "out.txt");
    r.err = Slurp(path_ / "err.txt");
    return r;
  }

 private:
  fs::path path_;
};

std::string Example(const char* name) {
  return std::string(IMSRG_SOURCE_DIR) + "/examples/config/" + name;
}

/// stdout up to (not including) the `[provenance]` line.
std::string BeforeProvenance(const std::string& s) {
  const auto pos = s.find("[provenance]");
  return pos == std::string::npos ? s : s.substr(0, pos);
}

bool Contains(const std::string& s, const std::string& what) {
  return s.find(what) != std::string::npos;
}

struct ExampleCase {
  const char* file;
  std::string legacy;
};

std::string Common2n(const std::string& emax, const std::string& hw) {
  return std::string("2bme=") + kMe2j +
         " file2e1max=14 file2e2max=28 file2lmax=14 emax=" + emax +
         " hw=" + hw + " reference=O16 ";
}

std::string ThreeN() {
  return std::string(" 3bme=") + kStream +
         " 3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18";
}

}  // namespace

TEST(DriverCli, ExamplesTomlAndLegacyResolveIdentically) {
  const ExampleCase cases[] = {
      {"a_o16_imsrg2.toml",
       Common2n("4", "16") + "valence_space=O16 Operators=Rp2"},
      {"b_o16_imsrg2_3n.toml",
       Common2n("4", "16") + "valence_space=O16" + ThreeN()},
      {"c_o16_sd_shell.toml",
       Common2n("6", "20") + "valence_space=sd-shell Operators=Rp2"},
      {"d_o16_imsrg3_n7.toml", Common2n("2", "16") +
                                   "e3max=6 valence_space=O16" + ThreeN() +
                                   " IMSRG3=true imsrg3_n7=true"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.file);
    TempDir d;
    const Result t = d.Run("--print-config --config " + Example(c.file));
    const Result l = d.Run("--print-config " + c.legacy);
    EXPECT_EQ(t.code, 0) << t.err;
    EXPECT_EQ(l.code, 0) << l.err;
    EXPECT_FALSE(t.out.empty());
    EXPECT_EQ(BeforeProvenance(t.out), BeforeProvenance(l.out));
    EXPECT_TRUE(Contains(t.out, "[provenance]"));
  }
}

TEST(DriverCli, ValidateWritesNothingAndRunStopsOnPlaceholderInput) {
  TempDir d;
  const std::string a = Example("a_o16_imsrg2.toml");
  const Result v = d.Run("--validate --config " + a);
  EXPECT_EQ(v.code, 0) << v.err;
  EXPECT_TRUE(Contains(v.out, "configuration is valid"));
  EXPECT_FALSE(fs::exists(d.path() / "output"));

  // Since M8 the pipeline runs: the one-byte placeholder me2j file passes the
  // V-14 preflight but is not an interaction, so the reader stops the run with
  // an input error (exit 3) after the resolved config was written.
  const Result r = d.Run("--config " + a);
  EXPECT_EQ(r.code, 3) << r.err;
  EXPECT_TRUE(Contains(r.err, "data ended")) << r.err;
  EXPECT_TRUE(fs::exists(d.path() / "output" / "resolved_config.toml"));

  const Result again = d.Run("--validate --config output/resolved_config.toml");
  EXPECT_EQ(again.code, 0) << again.err;
}

TEST(DriverCli, SetOverrides) {
  TempDir d;
  const std::string a = Example("a_o16_imsrg2.toml");
  const std::string base = "--print-config --config " + a + " ";

  Result r = d.Run(base + "--set basis.emax=2");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "emax = 2\n")) << r.out;

  r = d.Run(base + "--set imsrg.method=magnus");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "method = \"magnus\""));

  r = d.Run(base + "--set imsrg.step.smax=100");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "smax = 100.0")) << r.out;

  r = d.Run(base + "--set 'operators.list=[\"Rp2\",\"Rm2\"]'");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "\"Rp2\"") && Contains(r.out, "\"Rm2\""))
      << r.out;

  r = d.Run(base + "--set basis.emax=2 --set basis.emax=3");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "emax = 3\n")) << r.out;
  EXPECT_FALSE(Contains(r.out, "emax = 2\n"));

  r = d.Run(base + "--set basis.emxa=2");
  EXPECT_EQ(r.code, 2);
  EXPECT_TRUE(Contains(r.err, "E-KEY-UNKNOWN")) << r.err;
  EXPECT_TRUE(Contains(r.err, "emax")) << r.err;
}

TEST(DriverCli, CommandLineErrors) {
  TempDir d;
  Result r = d.Run("--config " + Example("a_o16_imsrg2.toml") + " emax=4");
  EXPECT_EQ(r.code, 2);
  EXPECT_TRUE(Contains(r.err, "E-CLI-MIX")) << r.err;

  r = d.Run("--foo");
  EXPECT_EQ(r.code, 2);

  r = d.Run("");
  EXPECT_EQ(r.code, 2);
}

TEST(DriverCli, ExitCodes) {
  {
    TempDir d;
    {
      std::ofstream(d.path() / "bad.toml") << "[basis]\nemxa = 4\n";
    }
    const Result r = d.Run("--validate --config bad.toml");
    EXPECT_EQ(r.code, 2);
    EXPECT_TRUE(Contains(r.err, "E-KEY-UNKNOWN")) << r.err;
  }
  {
    TempDir d(/*with_inputs=*/false);
    const Result r =
        d.Run("--validate --config " + Example("a_o16_imsrg2.toml"));
    EXPECT_EQ(r.code, 3) << r.err;
    EXPECT_TRUE(Contains(r.err, "V-14")) << r.err;
  }
  {
    TempDir d;
    const Result r = d.Run("--validate --config does_not_exist.toml");
    EXPECT_EQ(r.code, 3) << r.err;
    EXPECT_TRUE(Contains(r.err, "E-FILE")) << r.err;
  }
  {
    TempDir d;
    const Result r =
        d.Run("--validate " + Common2n("4.5", "16") + "valence_space=O16");
    EXPECT_EQ(r.code, 2) << r.err;
  }
}

TEST(DriverCli, InformationOutputs) {
  TempDir d;
  Result r = d.Run("--legacy-keys");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "2bme"));
  EXPECT_TRUE(Contains(r.out, "OperatorsTDA"));
  // 85 keys: one table row per key; the header and the notes make up the rest.
  std::ifstream keys(std::string(IMSRG_SOURCE_DIR) +
                     "/tests/unit/driver/data/legacy_keys.txt");
  std::string key;
  int n = 0;
  while (keys >> key) {
    ++n;
    EXPECT_TRUE(Contains(r.out, key)) << key;
  }
  EXPECT_EQ(n, 85);

  r = d.Run("--list-operators");
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_TRUE(Contains(r.out, "Rp2"));
  EXPECT_TRUE(Contains(r.out, "rhop_<r>"));

  r = d.Run("help");
  EXPECT_EQ(r.code, 0);
  EXPECT_TRUE(Contains(r.out, "--legacy-keys"));
  r = d.Run("-help");
  EXPECT_EQ(r.code, 0);
  r = d.Run("--help");
  EXPECT_EQ(r.code, 0);
  EXPECT_TRUE(Contains(r.out, "--config"));
  EXPECT_TRUE(Contains(r.out, "Legacy key=value mode"));
}
