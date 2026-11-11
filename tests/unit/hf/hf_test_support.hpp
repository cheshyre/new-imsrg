#pragma once
// Shared helpers of the hf tests: a tiny JSON reader for the golden hf.json files, loading of
// the Hamiltonian / NO2B inputs, and the comparison of an HFResult with the golden values.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/hf/hartree_fock.hpp"
#include "imsrg/io/no2b.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_dump.hpp"

namespace imsrg::hftest {

namespace fs = std::filesystem;

/// A JSON value reduced to what the golden files contain: numbers, arrays, objects.
struct Json {
  double number = 0.0;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> members;

  const Json& at(const std::string& key) const {
    for (const auto& [k, v] : members) {
      if (k == key) return v;
    }
    throw std::runtime_error("json key not found: " + key);
  }
  std::vector<double> numbers() const {
    std::vector<double> out;
    for (const auto& it : items) out.push_back(it.number);
    return out;
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : s_(std::move(text)) {}
  Json parse() { return value(); }

 private:
  void ws() {
    while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
  }
  std::string str() {
    std::string out;
    ++p_;  // opening quote
    while (s_[p_] != '"') {
      if (s_[p_] == '\\') ++p_;
      out += s_[p_++];
    }
    ++p_;
    return out;
  }
  Json value() {
    ws();
    Json j;
    const char c = s_[p_];
    if (c == '{') {
      ++p_;
      ws();
      while (s_[p_] != '}') {
        ws();
        std::string k = str();
        ws();
        ++p_;  // colon
        j.members.emplace_back(std::move(k), value());
        ws();
        if (s_[p_] == ',') ++p_;
        ws();
      }
      ++p_;
    } else if (c == '[') {
      ++p_;
      ws();
      while (s_[p_] != ']') {
        j.items.push_back(value());
        ws();
        if (s_[p_] == ',') ++p_;
        ws();
      }
      ++p_;
    } else if (c == '"') {
      str();
    } else if (s_.compare(p_, 4, "null") == 0 || s_.compare(p_, 4, "true") == 0) {
      p_ += 4;
    } else if (s_.compare(p_, 5, "false") == 0) {
      p_ += 5;
    } else {
      char* end = nullptr;
      j.number = std::strtod(s_.c_str() + p_, &end);
      p_ = static_cast<std::size_t>(end - s_.c_str());
    }
    return j;
  }
  std::string s_;
  std::size_t p_ = 0;
};

inline Json read_json(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return JsonParser(ss.str()).parse();
}

/// Golden values of one hf.json.
struct Golden {
  double e_hf = 0, e1 = 0, e2 = 0, e3 = 0;
  int iterations = 0;
  std::vector<double> spe, occupation;
  std::vector<std::vector<double>> C;  // rows: C[ho][hf]
};

inline Golden load_golden(const fs::path& path) {
  const Json j = read_json(path);
  Golden g;
  g.e_hf = j.at("EHF").number;
  g.e1 = j.at("e1hf").number;
  g.e2 = j.at("e2hf").number;
  g.e3 = j.at("e3hf").number;
  g.iterations = static_cast<int>(j.at("iterations").number);
  g.spe = j.at("spe").numbers();
  g.occupation = j.at("occupation").numbers();
  for (const auto& row : j.at("C").items) g.C.push_back(row.numbers());
  return g;
}

/// Max |spe - golden|, printed with the other deviations by `compare_to_golden`.
inline double max_spe_diff(const hf::HFResult& r, const Golden& g) {
  double d = 0.0;
  for (std::size_t i = 0; i < g.spe.size(); ++i) d = std::max(d, std::abs(r.spe[i] - g.spe[i]));
  return d;
}

/// Compares and prints every deviation. Tolerances: E_HF `e_tol`, spe `spe_tol`, C `c_tol`.
inline void compare_to_golden(const hf::HFResult& r, const Golden& g, const std::string& tag,
                              double e_tol, double spe_tol, double c_tol) {
  ASSERT_EQ(r.spe.size(), g.spe.size());
  double dc = 0.0, docc = 0.0;
  for (std::size_t i = 0; i < g.spe.size(); ++i) {
    docc = std::max(docc, std::abs(r.occupations[i] - g.occupation[i]));
    for (std::size_t a = 0; a < g.spe.size(); ++a) {
      dc = std::max(dc, std::abs(r.C(static_cast<std::int64_t>(i), static_cast<std::int64_t>(a)) -
                                 g.C[i][a]));
    }
  }
  const double dspe = max_spe_diff(r, g);
  std::cout << "[" << tag << "] dE_HF " << r.e_hf - g.e_hf << ", de1 " << r.e1 - g.e1 << ", de2 "
            << r.e2 - g.e2 << ", de3 " << r.e3 - g.e3 << ", max|dspe| " << dspe << ", max|dC| "
            << dc << ", max|docc| " << docc << ", iterations " << r.iterations << " (golden "
            << g.iterations << ")\n";
  EXPECT_NEAR(r.e_hf, g.e_hf, e_tol);
  EXPECT_NEAR(r.e1, g.e1, 10 * e_tol);
  EXPECT_NEAR(r.e2, g.e2, 10 * e_tol);
  EXPECT_NEAR(r.e3, g.e3, 10 * e_tol);
  EXPECT_LE(dspe, spe_tol);
  EXPECT_LE(dc, c_tol);
  EXPECT_EQ(docc, 0.0);
  EXPECT_EQ(r.iterations, g.iterations);
  EXPECT_TRUE(r.converged);
}

/// Hamiltonian from `dir/H.dump`.
inline op::Operator load_hamiltonian(const fs::path& dir) {
  const fs::path dump = dir / "H.dump";
  const auto ms = op::model_space_for_dump(op::read_operator_dump_header(dump));
  return op::read_operator_dump(dump, ms);
}

/// The NO2B file of the 2N+3N cases (outside the repo); empty path if absent.
inline fs::path no2b_file() {
  const fs::path p = fs::path(IMSRG_TEST_DATA_DIR) /
                     "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";
  return fs::exists(p) ? p : fs::path{};
}

/// Reads the NO2B file (stored with truncation {6,12,18,6}) into a container with `trunc`.
inline op::No2bThreeBody load_v3(const fs::path& file, const op::No2bTruncation& trunc) {
  op::No2bThreeBody v3(trunc);
  io::read_no2b(file, {.file = {6, 12, 18, 6}}, v3);
  return v3;
}

}  // namespace imsrg::hftest
