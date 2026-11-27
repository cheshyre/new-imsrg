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
#include <string_view>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/io/operator_binary.h"
#include "imsrg/opslib/nreft.h"

namespace imsrg::driver::transform {

std::vector<std::string> ReadWorkList(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in)
    throw InputError("cannot read the work list '" + path.string() + "'");
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

namespace {
// The work-list name of the intrinsic kinetic energy of the center of mass
// (opslib catalog), the one item with a host preparation that dominates its
// cost (kTcmPreparationCost).
constexpr std::string_view kTcmItemName = "TCM";
}  // namespace

double TensorCostFactor(std::int32_t rank_J) {
  // Fit (AL-F3, 2026-11-26): t(L) / CommutatorWeight(L) at emax 10 on one
  // GCD is 4.55 (L = 0) and 7.6 + 6.5 L (L = 1..5, within 1 %), in units of
  // seconds per 1e10 for two Omegas; normalized to L = 0.
  if (rank_J <= 0) return 1.0;
  return 1.67 + 1.43 * static_cast<double>(rank_J);
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
  const double n = static_cast<double>(num_omegas);
  double cost =
      n * CommutatorWeight(ms, rank_J, parity) * TensorCostFactor(rank_J);
  if (request.canonical_name == kTcmItemName) {
    cost += kTcmPreparationCost * CommutatorWeight(ms, 0, Parity::kEven);
  }
  return cost;
}

std::vector<int> AssignStatic(std::span<const double> costs, int num_ranks) {
  IMSRG_CHECK(num_ranks >= 1, "AssignStatic: num_ranks must be >= 1");
  std::vector<std::size_t> order(costs.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::size_t i, std::size_t j) { return costs[i] > costs[j]; });
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
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
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
