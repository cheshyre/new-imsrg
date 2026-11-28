// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/config/compat.h"

#include <array>
#include <span>
#include <string_view>

namespace imsrg::driver {
namespace {

struct Entry {
  std::string_view name;
  bool CompatOptions::* field;
};

constexpr std::array<Entry, 7> kEntries = {{
    {"fork_step_defaults", &CompatOptions::fork_step_defaults},
    {"tokyo_writer_fork_79e9330", &CompatOptions::tokyo_writer_fork_79e9330},
    {"tensor_no_fork_diagonal_bug",
     &CompatOptions::tensor_no_fork_diagonal_bug},
    {"tensor_ninej_threshold_1e8", &CompatOptions::tensor_ninej_threshold_1e8},
    {"tensor_fork_ph_missing_rank_phase",
     &CompatOptions::tensor_fork_ph_missing_rank_phase},
    {"tensor_one_body_norm_imsrgpp",
     &CompatOptions::tensor_one_body_norm_imsrgpp},
    {"fork_operator_file_names", &CompatOptions::fork_operator_file_names},
}};

}  // namespace

std::span<const std::string_view> CompatOptionNames() noexcept {
  static const auto* const kNames = [] {
    auto* a = new std::array<std::string_view, kEntries.size()>();
    for (std::size_t i = 0; i < kEntries.size(); ++i)
      (*a)[i] = kEntries[i].name;
    return a;
  }();
  return {kNames->data(), kNames->size()};
}

bool SetCompatOption(CompatOptions& o, std::string_view name, bool value) {
  for (const Entry& e : kEntries) {
    if (e.name == name) {
      o.*(e.field) = value;
      return true;
    }
  }
  return false;
}

CompatOptions EffectiveCompatOptions(const TaskConfig& run) {
  CompatOptions o = CompatOptionsFor(run.compat);
  for (const auto& n : run.compat_enable) SetCompatOption(o, n, true);
  for (const auto& n : run.compat_disable) SetCompatOption(o, n, false);
  return o;
}

CompatOptions EffectiveCompatOptions(const RunConfig& config) {
  return EffectiveCompatOptions(config.run);
}

}  // namespace imsrg::driver
