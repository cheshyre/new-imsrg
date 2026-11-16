// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/driver/integration/snt_compare.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::it {

SntData ReadSnt(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  SntData d;
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line[0] == '!') {
      const std::string tag = "! Zero body term:";
      if (line.compare(0, tag.size(), tag) == 0)
        d.zero_body = std::strtod(line.c_str() + tag.size(), nullptr);
      continue;
    }
    std::istringstream ss(line);
    std::vector<std::string> tok;
    for (std::string t; ss >> t;) tok.push_back(t);
    if (!tok.empty()) rows.push_back(std::move(tok));
  }
  std::size_t r = 0;
  auto need = [&](std::size_t ntok) -> const std::vector<std::string>& {
    if (r >= rows.size() || rows[r].size() < ntok)
      throw std::runtime_error("snt: malformed line " + std::to_string(r) +
                               " in " + path.string());
    return rows[r++];
  };
  const auto& ms = need(4);
  const int norb = std::atoi(ms[0].c_str()) + std::atoi(ms[1].c_str());
  for (int i = 0; i < norb; ++i) {
    const auto& t = need(5);
    d.orbits.push_back({std::atoi(t[0].c_str()), std::atoi(t[1].c_str()),
                        std::atoi(t[2].c_str()), std::atoi(t[3].c_str()),
                        std::atoi(t[4].c_str())});
  }
  const auto n1 = static_cast<std::size_t>(std::atoi(need(3)[0].c_str()));
  for (std::size_t i = 0; i < n1; ++i) {
    const auto& t = need(3);
    int a = std::atoi(t[0].c_str());
    int b = std::atoi(t[1].c_str());
    if (a > b) std::swap(a, b);
    d.one_body[{a, b}] = std::strtod(t[2].c_str(), nullptr);
  }
  const auto n2 = static_cast<std::size_t>(std::atoi(need(3)[0].c_str()));
  for (std::size_t i = 0; i < n2; ++i) {
    const auto& t = need(6);
    int idx[4];
    for (std::size_t k = 0; k < 4; ++k) idx[k] = std::atoi(t[k].c_str());
    const int J = std::atoi(t[4].c_str());
    double v = std::strtod(t[5].c_str(), nullptr);
    for (int p = 0; p < 4; p += 2) {
      if (idx[p] > idx[p + 1]) {
        std::swap(idx[p], idx[p + 1]);
        const int e = (d.two_j(idx[p]) + d.two_j(idx[p + 1])) / 2 - J;
        if (((e % 2) + 2) % 2 == 0) v = -v;  // -(-1)^e
      }
    }
    if (std::array<int, 2>{idx[0], idx[1]} >
        std::array<int, 2>{idx[2], idx[3]}) {
      std::swap(idx[0], idx[2]);
      std::swap(idx[1], idx[3]);
    }
    ++d.two_body_lines;
    const bool fresh =
        d.two_body
            .emplace(std::array<int, 5>{idx[0], idx[1], idx[2], idx[3], J}, v)
            .second;
    if (!fresh) ++d.duplicate_lines;
  }
  return d;
}

SntComparison CompareSnt(const SntData& a, const SntData& b, double threshold) {
  SntComparison c;
  c.zero_dev = std::abs(a.zero_body - b.zero_body);
  c.n1b_a = a.one_body.size();
  c.n1b_b = b.one_body.size();
  c.n2b_a = a.two_body.size();
  c.n2b_b = b.two_body.size();
  c.orbits_equal = (a.orbits == b.orbits);
  auto walk = [&](const auto& ma, const auto& mb, double& maxdev,
                  auto describe) {
    auto ia = ma.begin();
    auto ib = mb.begin();
    while (ia != ma.end() || ib != mb.end()) {
      double va = 0.0, vb = 0.0;
      auto key = (ia != ma.end()) ? ia->first : ib->first;
      if (ib == mb.end() || (ia != ma.end() && ia->first < ib->first)) {
        key = ia->first;
        va = ia->second;
        ++ia;
      } else if (ia == ma.end() || ib->first < ia->first) {
        key = ib->first;
        vb = ib->second;
        ++ib;
      } else {
        key = ia->first;
        va = ia->second;
        vb = ib->second;
        ++ia;
        ++ib;
      }
      const double dev = std::abs(va - vb);
      if ((va == 0.0 || vb == 0.0) && dev > threshold) ++c.missing;
      if (dev > maxdev) {
        maxdev = dev;
        describe(key, va, vb);
      }
    }
  };
  walk(a.one_body, b.one_body, c.max_dev_1b,
       [](const auto&, double, double) {});
  walk(a.two_body, b.two_body, c.max_dev_2b,
       [&](const auto& k, double va, double vb) {
         std::ostringstream os;
         os << "(" << k[0] << " " << k[1] << " " << k[2] << " " << k[3]
            << ") J=" << k[4] << " computed " << va << " golden " << vb;
         c.worst = os.str();
       });
  return c;
}

}  // namespace imsrg::it
