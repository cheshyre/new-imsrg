#include "imsrg/commutator/options.hpp"

#include <array>

namespace imsrg::commutator {

namespace {
constexpr std::array<std::string_view, kNumTerms> kTermNames = {
    "comm110ss", "comm220ss", "comm111ss",       "comm121ss",
    "comm221ss", "comm122ss", "comm222_pp_hhss", "comm222_phss"};
}  // namespace

std::string_view term_name(Term t) noexcept {
  const auto i = static_cast<std::size_t>(t);
  return i < kTermNames.size() ? kTermNames[i] : std::string_view{};
}

std::optional<Term> term_from_name(std::string_view name) noexcept {
  for (std::size_t i = 0; i < kTermNames.size(); ++i)
    if (kTermNames[i] == name) return static_cast<Term>(i);
  return std::nullopt;
}

}  // namespace imsrg::commutator
