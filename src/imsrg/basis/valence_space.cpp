#include "imsrg/basis/valence_space.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <fmt/format.h>

#include "imsrg/base/error.hpp"

namespace imsrg::basis {

namespace {

// Copied from ModelSpace.cc:23-70 (ModelSpace::ValenceSpaces), sorted by name (std::map order).
constexpr std::string_view kOrbits0[] = {"n0d5", "n0d3", "n1s1"};
constexpr std::string_view kOrbits1[] = {"n0f7", "n0f5", "n1p3", "n1p1"};
constexpr std::string_view kOrbits2[] = {"n0f7", "n0f5", "n1p3", "n1p1", "n0g9"};
constexpr std::string_view kOrbits3[] = {"p0f7", "p0f5", "p1p3", "p1p1"};
constexpr std::string_view kOrbits4[] = {"n2p3", "n2p1", "n1f5", "n1f7", "n0h9", "n0i13"};
constexpr std::string_view kOrbits5[] = {"n3s1", "n2d5", "n2d3", "n1g9", "n1g7", "n0i11", "n0j15"};
constexpr std::string_view kOrbits6[] = {"n0g9", "n0f5", "n1p3", "n1p1"};
constexpr std::string_view kOrbits7[] = {"n0h11", "n0g7", "n1d5", "n1d3", "n2s1"};
constexpr std::string_view kOrbits8[] = {"p0f7", "n0f7", "p0f5", "n0f5",
                                         "p1p3", "n1p3", "p1p1", "n1p1"};
constexpr std::string_view kOrbits9[] = {"p0f7", "n0f7", "p0f5", "n0f5", "p1p3",
                                         "n1p3", "p1p1", "n1p1", "p0g9", "n0g9"};
constexpr std::string_view kOrbits10[] = {"p0f7", "n0f7", "p0f5", "n0f5", "p1p3",
                                          "n1p3", "p1p1", "n1p1", "n0g9"};
constexpr std::string_view kOrbits11[] = {"p0f7", "p0f5", "p1p3", "p1p1", "n0g9",
                                          "n0g7", "n1d5", "n1d3", "n2s1"};
constexpr std::string_view kOrbits12[] = {"p0g9", "n0g9", "p0g7", "n0g7", "p1d5",
                                          "n1d5", "p1d3", "n1d3", "p2s1", "n2s1"};
constexpr std::string_view kOrbits13[] = {"p0f5", "n0f5", "p1p3", "n1p3",
                                          "p1p1", "n1p1", "p0g9", "n0g9"};
constexpr std::string_view kOrbits14[] = {"p0f5", "p1p3", "p1p1", "p0g9", "n0g7",
                                          "n1d5", "n1d3", "n2s1", "n0h11"};
constexpr std::string_view kOrbits15[] = {"p0g7", "p1d5", "p1d3", "p2s1", "p0h11",
                                          "n0g7", "n1d5", "n1d3", "n2s1", "n0h11"};
constexpr std::string_view kOrbits16[] = {"p0g7", "p1d5", "p1d3", "p2s1", "p0h11", "n0h9",
                                          "n1f7", "n1f5", "n2p3", "n2p1", "n0i13"};
constexpr std::string_view kOrbits17[] = {"p0p3", "n0p3", "p0p1", "n0p1"};
constexpr std::string_view kOrbits18[] = {"p2p3", "n2p3", "p2p1", "n2p1", "p1f5",  "n1f5",
                                          "p1f7", "n1f7", "p0h9", "n0h9", "p0i13", "n0i13"};
constexpr std::string_view kOrbits19[] = {"n2p3", "n2p1", "n1f5", "n1f7", "n0h9", "n0i13"};
constexpr std::string_view kOrbits20[] = {"p2p3", "p2p1", "p1f5", "p1f7", "p0h9", "p0i13"};
constexpr std::string_view kOrbits21[] = {"p2p3", "n3s1", "p2p1", "n2d5",  "p1f5",  "n2d3", "p1f7",
                                          "n1g9", "p0h9", "n1g7", "p0i13", "n0i11", "n0j15"};
constexpr std::string_view kOrbits22[] = {"p0p3", "n0p3", "p0p1", "n0p1", "p0d5",
                                          "n0d5", "p0d3", "n0d3", "p1s1", "n1s1"};
constexpr std::string_view kOrbits23[] = {"p0p3", "p0p1", "n0d5", "n0d3", "n1s1"};
constexpr std::string_view kOrbits24[] = {"n0p3", "n0p1", "p0d5", "p0d3", "p1s1"};
constexpr std::string_view kOrbits25[] = {"p0s1", "n0s1"};
constexpr std::string_view kOrbits26[] = {"p0d5", "n0d5", "p0d3", "n0d3", "p1s1", "n1s1"};
constexpr std::string_view kOrbits27[] = {"p0d3", "n0d3", "p1s1", "n1s1",
                                          "p0f7", "n0f7", "p1p3", "n1p3"};
constexpr std::string_view kOrbits28[] = {"p0d5", "n0d5", "p0d3", "n0d3", "p1s1", "n1s1", "p0f7",
                                          "n0f7", "p0f5", "n0f5", "p1p3", "n1p3", "p1p1", "n1p1"};
constexpr std::string_view kOrbits29[] = {"p0d5", "p0d3", "p1s1", "n0f7", "n0f5", "n1p3", "n1p1"};
constexpr std::string_view kOrbits30[] = {"n0d5", "n0d3", "n1s1", "p0f7", "p0f5", "p1p3", "p1p1"};
constexpr std::string_view kOrbits31[] = {"n0h11", "n0g7", "n1d5", "n1d3", "n2s1"};
constexpr std::string_view kOrbits32[] = {"p0h11", "p0g7", "p1d5", "p1d3", "p2s1"};
constexpr std::string_view kOrbits33[] = {"p0h11", "p0g7", "p1d5", "p1d3", "p2s1", "n2p3",
                                          "n2p1",  "n1f5", "n1f7", "n0h9", "n0i13"};
constexpr std::string_view kOrbits34[] = {"p0h11", "p0g7", "p1d5", "p1d3", "p2s1",
                                          "n2p3",  "n2p1", "n1f5", "n0h9", "n0i13"};
constexpr std::string_view kOrbits35[] = {"p0h11", "p0g7", "p1d5", "p1d3", "p2s1",  "n3s1",
                                          "n2d5",  "n2d3", "n1g9", "n1g7", "n0i11", "n0j15"};
constexpr std::string_view kOrbits36[] = {"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1"};
constexpr std::string_view kOrbits37[] = {"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1",
                                          "p0d5", "n0d5", "p0d3", "n0d3", "p1s1", "n1s1"};
constexpr std::string_view kOrbits38[] = {"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1", "p0d5",
                                          "n0d5", "p0d3", "n0d3", "p1s1", "n1s1", "p0f7", "n0f7",
                                          "p0f5", "n0f5", "p1p3", "n1p3", "p1p1", "n1p1"};

constexpr NamedValenceSpace kTable[] = {
    {"Ca28_sdSemiMagic-shell", "Ca28", kOrbits0},
    {"Ca40_fpSemiMagic-shell", "Ca40", kOrbits1},
    {"Ca40_fpg9SemiMagic-shell", "Ca40", kOrbits2},
    {"Ca70_fpProtonSemiMagic-shell", "Ca70", kOrbits3},
    {"Pb164pfhi13NRSemiMagic-shell", "Pb164", kOrbits4},
    {"Pb208sdgij15NRSemiMagic-shell", "Pb208", kOrbits5},
    {"Zr68_pfg9_NeutronSemiMagic", "Zr68", kOrbits6},
    {"Zr90_sdg9_NeutronSemiMagic", "Zr90", kOrbits7},
    {"fp-shell", "Ca40", kOrbits8},
    {"fpg9-shell", "Ca40", kOrbits9},
    {"fpg9NR-shell", "Ca40", kOrbits10},
    {"fpgdsNR-shell", "Ca60", kOrbits11},
    {"gds-shell", "Zr80", kOrbits12},
    {"jj44", "Ni56", kOrbits13},
    {"jj45", "Ni78", kOrbits14},
    {"jj55", "Sn100", kOrbits15},
    {"jj56", "Sn132", kOrbits16},
    {"p-shell", "He4", kOrbits17},
    {"pfhi13NR-shell", "Pb164", kOrbits18},
    {"pfhi13NR_NeutronSemiMagic", "Sn132", kOrbits19},
    {"pfhi13_ProtonSemiMagic", "Pb208", kOrbits20},
    {"pfhi13sdgij15NR-shell", "Pb208", kOrbits21},
    {"psd-shell", "He4", kOrbits22},
    {"psdNR-shell", "He10", kOrbits23},
    {"psdPR-shell", "O10", kOrbits24},
    {"s-shell", "vacuum", kOrbits25},
    {"sd-shell", "O16", kOrbits26},
    {"sd3f7p3-shell", "Si28", kOrbits27},
    {"sdfp-shell", "O16", kOrbits28},
    {"sdfpNR-shell", "O28", kOrbits29},
    {"sdfpPR-shell", "Ca28", kOrbits30},
    {"sdgh11_NeutronSn100SemiMagic", "Sn100", kOrbits31},
    {"sdgh11_ProtonSemiMagic", "Sn176", kOrbits32},
    {"sdgh11pfhi13NR-shell", "Sn132", kOrbits33},
    {"sdgh11pfhi13NRfr-shell", "Sn140", kOrbits34},
    {"sdgh11sdgij15NR-shell", "Sn176", kOrbits35},
    {"sp-shell", "vacuum", kOrbits36},
    {"spsd-shell", "vacuum", kOrbits37},
    {"spsdpf-shell", "vacuum", kOrbits38},
};

// Appends the naive filling of `core` (skipping `skip`) as core orbits.
void append_core(std::vector<Orbit>& out, const Nucleus& core, std::int32_t emax,
                 std::span<const Orbit> skip = {}) {
  for (const OrbitOccupation& oo : naive_filling(core, emax, skip)) {
    out.push_back(oo.orbit);
  }
}

void add_unique(std::vector<Orbit>& v, const Orbit& o) {
  if (std::find(v.begin(), v.end(), o) == v.end()) v.push_back(o);
}

Orbit make_orbit(std::int32_t n, std::int32_t l, std::int32_t j2, std::int32_t tz2) {
  return Orbit{n, l, TwoJ{j2}, TwoTz{tz2}};
}

// Get0hwSpace (ModelSpace.cc:659-694).
ValenceSpaceDefinition zero_hw_space(const SingleParticleBasis& basis, const Nucleus& reference) {
  const int Aref = static_cast<int>(reference.A);
  const int Zref = static_cast<int>(reference.Z);
  const int Nref = Aref - Zref;
  int osc_p = 0;
  int osc_n = 0;
  while ((osc_p + 1) * (osc_p + 2) * (osc_p + 3) / 3 <= Zref) osc_p++;
  while ((osc_n + 1) * (osc_n + 2) * (osc_n + 3) / 3 <= Nref) osc_n++;
  const int Zcore = osc_p * (osc_p + 1) * (osc_p + 2) / 3;
  const int Ncore = osc_n * (osc_n + 1) * (osc_n + 2) / 3;

  ValenceSpaceDefinition def;
  const Nucleus core{static_cast<double>(Zcore + Ncore), static_cast<double>(Zcore)};
  def.core_name = core.name();
  append_core(def.core, core, basis.emax());
  auto add_species = [&](int osc, int tz2) {
    for (int L = osc; L >= 0; L -= 2) {
      for (int j2 = 2 * L + 1; j2 > std::max(2 * L - 2, 0); j2 -= 2) {
        add_unique(def.valence, make_orbit((osc - L) / 2, L, j2, tz2));
      }
    }
  };
  if (Zref > Zcore) add_species(osc_p, -1);
  if (Nref > Ncore) add_species(osc_n, +1);
  return def;
}

// GetjjSpace (ModelSpace.cc:696-747).
ValenceSpaceDefinition jj_space(const SingleParticleBasis& basis, const Nucleus& reference) {
  constexpr std::array<int, 9> jjmagic = {2, 6, 14, 28, 50, 82, 126, 184, 258};
  const int Aref = static_cast<int>(reference.A);
  const int Zref = static_cast<int>(reference.Z);
  const int Nref = Aref - Zref;
  auto closure_index = [&](int count, const char* what) {
    // imsrg++ indexes outside its table for count < 2 or count >= 258; we refuse instead.
    if (count < jjmagic.front() || count >= jjmagic.back()) {
      throw InputError(fmt::format("jj-shell: {} number {} is outside the supported range [2, 258)",
                                   what, count));
    }
    int i = 0;
    while (count >= jjmagic[static_cast<std::size_t>(i)]) i++;
    return i - 1;
  };
  const int iZ = closure_index(Zref, "proton");
  const int iN = closure_index(Nref, "neutron");
  const int Zcore = jjmagic[static_cast<std::size_t>(iZ)];
  const int Ncore = jjmagic[static_cast<std::size_t>(iN)];

  ValenceSpaceDefinition def;
  const Nucleus core{static_cast<double>(Zcore + Ncore), static_cast<double>(Zcore)};
  def.core_name = core.name();
  append_core(def.core, core, basis.emax());
  auto add_species = [&](int idx, int tz2) {
    for (int L = idx; L >= 0; L -= 2) {
      for (int j2 = 2 * L + 1; j2 > std::max(2 * L - 2, 0); j2 -= 2) {
        if (j2 == 2 * idx + 1) {
          add_unique(def.valence, make_orbit(0, L + 1, j2 + 2, tz2));  // largest j of next shell
        } else {
          add_unique(def.valence, make_orbit((idx - L) / 2, L, j2, tz2));
        }
      }
    }
  };
  if (Zref > Zcore) add_species(iZ, -1);
  if (Nref > Ncore) add_species(iN, +1);
  return def;
}

// getline(ss, tok, ',') semantics: a trailing empty field is not returned.
std::vector<std::string_view> split_commas(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    const std::size_t c = s.find(',', pos);
    if (c == std::string_view::npos) {
      out.push_back(s.substr(pos));
      break;
    }
    out.push_back(s.substr(pos, c - pos));
    pos = c + 1;
  }
  return out;
}

}  // namespace

std::span<const NamedValenceSpace> named_valence_spaces() noexcept { return kTable; }

const NamedValenceSpace* find_named_valence_space(std::string_view name) noexcept {
  for (const NamedValenceSpace& s : kTable) {
    if (s.name == name) return &s;
  }
  return nullptr;
}

ValenceSpaceDefinition define_valence_space(std::string_view spec, const SingleParticleBasis& basis,
                                            const Nucleus& reference) {
  if (spec == "0hw-shell") return zero_hw_space(basis, reference);
  if (spec == "jj-shell") return jj_space(basis, reference);

  ValenceSpaceDefinition def;
  if (spec.find(',') != std::string_view::npos) {
    const std::vector<std::string_view> tokens = split_commas(spec);
    const Nucleus core = Nucleus::parse(tokens.front());
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      add_unique(def.valence, parse_orbit_label(tokens[i]));
    }
    def.core_name = core.name();
    append_core(def.core, core, basis.emax(), def.valence);
    return def;
  }
  if (spec.find("FCI") != std::string_view::npos) {
    def.valence.assign(basis.orbits().begin(), basis.orbits().end());
    return def;
  }
  if (const NamedValenceSpace* named = find_named_valence_space(spec)) {
    for (const std::string_view label : named->orbits) {
      add_unique(def.valence, parse_orbit_label(label));
    }
    const Nucleus core = Nucleus::parse(named->core);
    def.core_name = core.name();
    append_core(def.core, core, basis.emax());
    return def;
  }
  const Nucleus core = Nucleus::parse(spec);
  def.core_name = core.name();
  append_core(def.core, core, basis.emax());
  return def;
}

Nucleus default_reference_nucleus(std::string_view spec) {
  if (spec.find(',') != std::string_view::npos) {
    return Nucleus::parse(split_commas(spec).front());
  }
  if (spec == "0hw-shell" || spec == "jj-shell" || spec.find("FCI") != std::string_view::npos) {
    throw InputError(fmt::format(
        "valence space '{}' needs an explicit reference nucleus (reference=default is not "
        "defined for it)",
        spec));
  }
  if (const NamedValenceSpace* named = find_named_valence_space(spec)) {
    return Nucleus::parse(named->core);
  }
  return Nucleus::parse(spec);
}

}  // namespace imsrg::basis
