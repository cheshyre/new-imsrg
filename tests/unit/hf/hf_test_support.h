// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_HF_HF_TEST_SUPPORT_H_
#define TESTS_UNIT_HF_HF_TEST_SUPPORT_H_
// Shared helpers of the hf tests: a tiny JSON reader for the golden hf.json
// files, loading of the Hamiltonian / NO2B inputs, and the comparison of an
// HFResult with the golden values.
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

#include "imsrg/hf/hartree_fock.h"
#include "imsrg/io/no2b.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"

namespace imsrg::hftest {

struct Json;

/// Object members in file order. A class (not a bare vector alias) with
/// special members defined after `Json` is complete: clang with libstdc++
/// instantiates the vector members eagerly while the element is incomplete.
class JsonMembers : public std::vector<std::pair<std::string, Json>> {
 public:
  JsonMembers();
  ~JsonMembers();
  JsonMembers(const JsonMembers& other);
  JsonMembers(JsonMembers&& other) noexcept;
  JsonMembers& operator=(const JsonMembers& other);
  JsonMembers& operator=(JsonMembers&& other) noexcept;
};

/// A JSON value reduced to what the golden files contain: numbers, arrays,
/// objects.
struct Json {
  double number = 0.0;
  std::vector<Json> items;
  JsonMembers members;

  const Json& at(const std::string& key) const {
    for (const auto& [k, v] : members) {
      if (k == key) return v;
    }
    throw std::runtime_error("json key not found: " + key);
  }
  std::vector<double> Numbers() const {
    std::vector<double> out;
    for (const auto& it : items) out.push_back(it.number);
    return out;
  }
};

inline JsonMembers::JsonMembers() = default;
inline JsonMembers::~JsonMembers() = default;
inline JsonMembers::JsonMembers(const JsonMembers& other) = default;
inline JsonMembers::JsonMembers(JsonMembers&& other) noexcept = default;
inline JsonMembers& JsonMembers::operator=(const JsonMembers& other) = default;
inline JsonMembers& JsonMembers::operator=(JsonMembers&& other) noexcept =
    default;

class JsonParser {
 public:
  explicit JsonParser(std::string text) : s_(std::move(text)) {}

  JsonParser(const JsonParser&) = default;
  JsonParser& operator=(const JsonParser&) = default;
  JsonParser(JsonParser&&) = default;
  JsonParser& operator=(JsonParser&&) = default;
  Json Parse() { return value(); }

 private:
  void Ws() {
    while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_])))
      ++p_;
  }
  std::string Str() {
    std::string out;
    ++p_;  // opening quote
    while (s_[p_] != '"') {
      if (s_[p_] == '\\') ++p_;
      out += s_[p_++];
    }
    ++p_;
    return out;
  }
  Json value();
  std::string s_;
  std::size_t p_ = 0;
};

inline Json ReadJson(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return JsonParser(ss.str()).Parse();
}

/// Golden values of one hf.json.
struct Golden {
  double e_hf = 0, e1 = 0, e2 = 0, e3 = 0;
  int iterations = 0;
  std::vector<double> spe, occupation;
  std::vector<std::vector<double>> C;  // rows: C[ho][hf]
};

Golden LoadGolden(const std::filesystem::path& path);

/// Max |spe - golden|, printed with the other deviations by
/// `CompareToGolden`.
inline double MaxSpeDiff(const hf::HFResult& r, const Golden& g) {
  double d = 0.0;
  for (std::size_t i = 0; i < g.spe.size(); ++i)
    d = std::max(d, std::abs(r.spe[i] - g.spe[i]));
  return d;
}

/// Compares and prints every deviation. Tolerances: E_HF `e_tol`, spe
/// `spe_tol`, C `c_tol`.
void CompareToGolden(const hf::HFResult& r, const Golden& g,
                     const std::string& tag, double e_tol, double spe_tol,
                     double c_tol);

/// Hamiltonian from `dir/H.dump`.
inline op::Operator LoadHamiltonian(const std::filesystem::path& dir) {
  const std::filesystem::path dump = dir / "H.dump";
  const auto ms = op::ModelSpaceForDump(op::ReadOperatorDumpHeader(dump));
  return op::ReadOperatorDump(dump, ms);
}

/// The NO2B file of the 2N+3N cases (outside the repo); empty path if absent.
inline std::filesystem::path No2bFile() {
  const std::filesystem::path p =
      std::filesystem::path(IMSRG_TEST_DATA_DIR) /
      "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";
  return std::filesystem::exists(p) ? p : std::filesystem::path{};
}

/// Reads the NO2B file (stored with truncation {6,12,18,6}) into a container
/// with `trunc`.
inline op::No2bThreeBody LoadV3(const std::filesystem::path& file,
                                const op::No2bTruncation& trunc) {
  op::No2bThreeBody v3(trunc);
  io::ReadNo2b(file, {.file = {6, 12, 18, 6}}, v3);
  return v3;
}

}  // namespace imsrg::hftest

#endif  // TESTS_UNIT_HF_HF_TEST_SUPPORT_H_
