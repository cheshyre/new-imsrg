#include "imsrg/basis/nucleus.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "imsrg/base/error.hpp"

namespace imsrg::basis {

namespace {

// The periodic table of imsrg++ (ModelSpace.cc:570-576), index = Z; "n" is the neutron.
constexpr std::array<std::string_view, 119> kElements = {
    "n",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Parses a complete decimal number (strtod syntax, no leading or trailing junk).
bool parse_decimal(std::string_view s, double& value) {
  if (s.empty() || !(is_digit(s.front()) || s.front() == '.')) return false;
  const std::string copy(s);
  char* end = nullptr;
  value = std::strtod(copy.c_str(), &end);
  return end == copy.c_str() + copy.size() && std::isfinite(value);
}

Nucleus checked(Nucleus nuc, std::string_view input) {
  if (!(nuc.Z >= 0.0) || !(nuc.A >= nuc.Z) || !std::isfinite(nuc.A)) {
    throw InputError("invalid nucleus '" + std::string(input) + "': need A >= Z >= 0");
  }
  return nuc;
}

}  // namespace

std::string_view element_symbol(std::int32_t Z) {
  if (Z < 0 || Z >= static_cast<std::int32_t>(kElements.size())) {
    throw InputError("element_symbol: no element with Z = " + std::to_string(Z));
  }
  return kElements[static_cast<std::size_t>(Z)];
}

std::int32_t element_proton_number(std::string_view symbol) {
  const auto it = std::find(kElements.begin(), kElements.end(), symbol);
  if (it == kElements.end()) {
    throw InputError("unknown element symbol '" + std::string(symbol) + "'");
  }
  return static_cast<std::int32_t>(it - kElements.begin());
}

Nucleus Nucleus::parse(std::string_view name) {
  if (name == "vacuum") return Nucleus{0.0, 0.0};
  constexpr std::string_view kMix = "mix_A";
  if (name.substr(0, kMix.size()) == kMix) {
    const std::size_t zpos = name.find("_Z");
    double A = 0.0;
    double Z = 0.0;
    if (zpos == std::string_view::npos ||
        !parse_decimal(name.substr(kMix.size(), zpos - kMix.size()), A) ||
        !parse_decimal(name.substr(zpos + 2), Z)) {
      throw InputError("invalid nucleus mixture '" + std::string(name) +
                       "': expected mix_A<A>_Z<Z>, e.g. mix_A4.5_Z2.5");
    }
    return checked(Nucleus{A, Z}, name);
  }
  std::size_t i = 0;
  while (i < name.size() && !is_digit(name[i])) ++i;
  const std::string_view symbol = name.substr(0, i);
  const std::string_view digits = name.substr(i);
  if (symbol.empty() || digits.empty() ||
      !std::all_of(digits.begin(), digits.end(), [](char c) { return is_digit(c); })) {
    throw InputError("invalid nucleus name '" + std::string(name) +
                     "': expected an element symbol and a mass number, e.g. O16");
  }
  const auto it = std::find(kElements.begin(), kElements.end(), symbol);
  if (it == kElements.end()) {
    throw InputError("invalid nucleus name '" + std::string(name) + "': unknown element '" +
                     std::string(symbol) + "'");
  }
  double A = 0.0;
  if (!parse_decimal(digits, A)) {
    throw InputError("invalid nucleus name '" + std::string(name) + "': bad mass number");
  }
  return checked(Nucleus{A, static_cast<double>(it - kElements.begin())}, name);
}

std::string Nucleus::name() const {
  if (A == 0.0 && Z == 0.0) return "vacuum";
  const bool integral = std::floor(A) == A && std::floor(Z) == Z && Z >= 0.0 &&
                        Z < static_cast<double>(kElements.size()) && A >= Z;
  if (integral) {
    return std::string(kElements[static_cast<std::size_t>(Z)]) + fmt::format("{}", A);
  }
  return fmt::format("mix_A{}_Z{}", A, Z);
}

std::vector<OrbitOccupation> naive_filling(const Nucleus& nucleus, std::int32_t emax,
                                           std::span<const Orbit> skip) {
  if (emax < 0) throw InputError("naive_filling: emax must be >= 0");
  const double A = nucleus.A;
  const double Z = nucleus.Z;
  if (!(Z >= 0.0) || !(A >= Z) || !std::isfinite(A)) {
    throw InputError("naive_filling: invalid nucleus (need A >= Z >= 0)");
  }
  const auto skipped = [&](const Orbit& o) {
    return std::find(skip.begin(), skip.end(), o) != skip.end();
  };
  // ModelSpace::GetOrbitsAZ, ModelSpace.cc:617-650; same operations, same order.
  double zz = 0;
  double nn = 0;
  std::vector<OrbitOccupation> filled;
  for (std::int32_t N = 0; N <= emax; ++N) {
    for (std::int32_t g = 2 * N + 1; g >= -2 * N; g -= 4) {
      const std::int32_t j2 = std::abs(g);
      const std::int32_t l = g < 0 ? (j2 + 1) / 2 : (j2 - 1) / 2;
      const std::int32_t n = (N - l) / 2;
      const Orbit proton{n, l, TwoJ{j2}, TwoTz::proton()};
      const Orbit neutron{n, l, TwoJ{j2}, TwoTz::neutron()};
      if (zz < Z && !skipped(proton)) {
        const double dz = std::min(Z - zz, j2 + 1.);
        filled.push_back({proton, dz / (j2 + 1.0)});
        zz += dz;
      }
      if (nn < A - Z && !skipped(neutron)) {
        const double dn = std::min(A - Z - nn, j2 + 1.);
        filled.push_back({neutron, dn / (j2 + 1.0)});
        nn += dn;
      }
      if (zz == Z && nn == A - Z) return filled;
    }
  }
  throw InputError("naive_filling: " + nucleus.name() + " (A=" + fmt::format("{}", A) +
                   ", Z=" + fmt::format("{}", Z) +
                   ") does not fit into the shells e <= " + std::to_string(emax));
}

}  // namespace imsrg::basis
