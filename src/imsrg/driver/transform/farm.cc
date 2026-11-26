// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/transform/farm.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/io/operator_binary.h"
#include "imsrg/opslib/nreft.h"

namespace imsrg::driver::transform {

std::vector<std::string> ReadWorkList(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw InputError("cannot read the work list '" + path.string() + "'");
  std::vector<std::string> names;
  std::string line;
  while (std::getline(in, line)) {
    if (const auto tab = line.find('\t'); tab != std::string::npos) {
      line.resize(tab);
    }
    const auto b = line.find_first_not_of(" \r");
    if (b == std::string::npos || line[b] == '#') continue;
    const auto e = line.find_last_not_of(" \r");
    names.push_back(line.substr(b, e - b + 1));
  }
  return names;
}

double CommutatorWeight(const basis::ModelSpace& ms, std::int32_t rank_J,
                        Parity parity) {
  const auto channels = ms.two_body().channels();
  double w = 0.0;
  for (std::size_t a = 0; a < channels.size(); ++a) {
    for (std::size_t b = a; b < channels.size(); ++b) {
      const auto& ca = channels[a];
      const auto& cb = channels[b];
      if (ca.two_tz.twice != cb.two_tz.twice) continue;
      const bool parity_ok =
          (static_cast<int>(ca.parity) + static_cast<int>(cb.parity) +
           static_cast<int>(parity)) %
              2 ==
          0;
      if (!parity_ok) continue;
      if (std::abs(ca.J() - cb.J()) > rank_J || rank_J > ca.J() + cb.J()) {
        continue;
      }
      const double na = ca.num_kets;
      const double nb = cb.num_kets;
      w += na * nb * (na + nb);
    }
  }
  return w;
}

double ItemCost(const OperatorRequest& request, const basis::ModelSpace& ms,
                std::size_t num_omegas) {
  std::int32_t rank_J = request.shape.rank_j;
  Parity parity = request.shape.parity == 0 ? Parity::kEven : Parity::kOdd;
  if (request.kind == OperatorKind::kNreft) {
    if (const auto s = opslib::nreft::ParseShorthand(request.canonical_name)) {
      rank_J = s->L;
      parity = Parity::kEven;
    }
  }
  return static_cast<double>(num_omegas) * CommutatorWeight(ms, rank_J, parity);
}

std::vector<int> AssignStatic(std::span<const double> costs, int num_ranks) {
  IMSRG_CHECK(num_ranks >= 1, "AssignStatic: num_ranks must be >= 1");
  std::vector<std::size_t> order(costs.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t i, std::size_t j) {
                     return costs[i] > costs[j];
                   });
  std::vector<double> load(static_cast<std::size_t>(num_ranks), 0.0);
  std::vector<int> owner(costs.size(), 0);
  for (const std::size_t i : order) {
    const auto r = static_cast<std::size_t>(
        std::min_element(load.begin(), load.end()) - load.begin());
    owner[i] = static_cast<int>(r);
    load[r] += costs[i];
  }
  return owner;
}

void WriteFailureManifest(const std::filesystem::path& path,
                          std::span<const Failure> failures) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  io::WriteFileAtomically(path, [&](std::ostream& out) {
    out << "# operators of run.task = \"transform\" that did not finish: "
        << failures.size() << "\n# name<TAB>reason (usable as "
        << "transform.work_list)\n";
    for (const Failure& f : failures) {
      std::string reason = f.reason;
      std::replace(reason.begin(), reason.end(), '\n', ' ');
      std::replace(reason.begin(), reason.end(), '\t', ' ');
      out << f.name << '\t' << reason << '\n';
    }
  });
}

}  // namespace imsrg::driver::transform
