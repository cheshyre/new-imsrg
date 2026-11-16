// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/basis/orbit.h"

#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>

#include "imsrg/base/error.h"

namespace imsrg::basis {

namespace {

// imsrg++ l letters (ModelSpace.cc:532): note 'j' for l = 7.
constexpr std::string_view kLetters = "spdfghijklmno";

bool IsDigit(char c) { return c >= '0' && c <= '9'; }

// Parses the run of decimal digits of `s` starting at `pos`; advances `pos`.
// Returns false if there is no digit or the value overflows.
bool ParseNumber(std::string_view s, std::size_t& pos, std::int32_t& value) {
  const std::size_t start = pos;
  while (pos < s.size() && IsDigit(s[pos])) ++pos;
  if (pos == start) return false;
  const auto [ptr, ec] =
      std::from_chars(s.data() + start, s.data() + pos, value);
  return ec == std::errc() && ptr == s.data() + pos;
}

}  // namespace

std::string OrbitLabel(const Orbit& o) {
  if (!IsValid(o) || o.l >= static_cast<std::int32_t>(kLetters.size())) {
    throw InputError("orbit_label: invalid orbit (n=" + std::to_string(o.n) +
                     ", l=" + std::to_string(o.l) +
                     ", 2j=" + std::to_string(o.j2.twice) +
                     ", 2tz=" + std::to_string(o.tz2.twice) + ")");
  }
  std::string s;
  s += o.is_proton() ? 'p' : 'n';
  s += std::to_string(o.n);
  s += kLetters[static_cast<std::size_t>(o.l)];
  s += std::to_string(o.j2.twice);
  return s;
}

Orbit ParseOrbitLabel(std::string_view label) {
  const auto fail = [&](const char* why) {
    return InputError("invalid orbit label '" + std::string(label) +
                      "': " + why);
  };
  if (label.size() < 4)
    throw fail("expected <p|n><n><l letter><2j>, e.g. p0d5");
  Orbit o;
  if (label[0] == 'p') {
    o.tz2 = TwoTz::Proton();
  } else if (label[0] == 'n') {
    o.tz2 = TwoTz::Neutron();
  } else {
    throw fail("the first character must be 'p' or 'n'");
  }
  std::size_t pos = 1;
  if (!ParseNumber(label, pos, o.n))
    throw fail("missing radial quantum number n");
  if (pos >= label.size()) throw fail("missing l letter");
  const std::size_t l = kLetters.find(label[pos]);
  if (l == std::string_view::npos)
    throw fail("unknown l letter (expected one of spdfghijklmno)");
  o.l = static_cast<std::int32_t>(l);
  ++pos;
  if (!ParseNumber(label, pos, o.j2.twice)) throw fail("missing 2j");
  if (pos != label.size()) throw fail("trailing characters after 2j");
  if (!IsValid(o)) throw fail("2j must be 2l+1 or 2l-1 and positive");
  return o;
}

}  // namespace imsrg::basis
