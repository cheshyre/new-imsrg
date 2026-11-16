// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/options.h"

#include <array>

namespace imsrg::commutator {

namespace {
constexpr std::array<std::string_view, kNumTerms> kTermNames = {
    "comm110ss",       "comm220ss",         "comm111ss",
    "comm121ss",       "comm221ss",         "comm122ss",
    "comm222_pp_hhss", "comm222_phss",      "comm133ss",
    "comm330ss",       "comm223ss",         "comm232ss",
    "comm331ss",       "comm231ss",         "comm132ss",
    "comm233_pp_hhss", "comm233_phss",      "comm332_ppph_hhhpss",
    "comm332_pphhss",  "comm333_ppp_hhhss", "comm333_pph_hhpss"};
}  // namespace

std::string_view TermName(Term t) noexcept {
  const auto i = static_cast<std::size_t>(t);
  return i < kTermNames.size() ? kTermNames[i] : std::string_view{};
}

std::optional<Term> TermFromName(std::string_view name) noexcept {
  for (std::size_t i = 0; i < kTermNames.size(); ++i)
    if (kTermNames[i] == name) return static_cast<Term>(i);
  return std::nullopt;
}

}  // namespace imsrg::commutator
