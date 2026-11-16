// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/pipeline/charge_radius.h"

#include <cmath>
#include <optional>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"

namespace imsrg::driver::pipeline {

double ChargeRadiusSquared(double rp2, double rso, int mass_number,
                           int proton_number) {
  IMSRG_CHECK(proton_number > 0 && mass_number >= proton_number,
              "charge radius needs 0 < Z <= A");
  const double z = static_cast<double>(proton_number);
  // Same evaluation order as imsrg++.cc:1367 (NEUTRON_RCH2 * (A - Z) / Z).
  return rp2 + rso / z + constants::kProtonRch2 +
         constants::kNeutronRch2 * (mass_number - proton_number) / z +
         constants::kDarwinFoldy;
}

std::optional<double> ChargeRadius(double rch2) {
  if (!std::isfinite(rch2) || rch2 < 0.0) return std::nullopt;
  return std::sqrt(rch2);
}

}  // namespace imsrg::driver::pipeline
