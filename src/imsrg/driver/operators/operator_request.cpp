// Operator catalog and the table-driven operator-name grammar (design/16 section 4).
#include "imsrg/driver/operators/operator_request.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <system_error>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"

namespace imsrg::driver {

namespace {

// --------------------------------------------------------------------------------------------
// The catalog
// --------------------------------------------------------------------------------------------

constexpr OpParamSpec kHw[] = {{"hw", OpParamType::Number, true, "oscillator frequency in MeV"}};
constexpr OpParamSpec kZ[] = {{"z", OpParamType::Integer, false, "proton number"}};
constexpr OpParamSpec kAZ[] = {{"a", OpParamType::Integer, false, "mass number"},
                               {"z", OpParamType::Integer, false, "proton number"}};
constexpr OpParamSpec kOrbit[] = {{"orbit", OpParamType::Orbit, false, "orbit label, e.g. p0d5"}};
constexpr OpParamSpec kOrbit2[] = {{"orbit1", OpParamType::Orbit, false, "bra orbit, e.g. p0s1"},
                                   {"orbit2", OpParamType::Orbit, false, "ket orbit, e.g. p0s1"}};
constexpr OpParamSpec kR[] = {{"r", OpParamType::Number, true, "radius in fm"}};
constexpr OpParamSpec kRmax[] = {{"rmax", OpParamType::Number, true, "sphere radius in fm"}};

using Params = std::span<const OpParamSpec>;

/// A P1 row (scalar: J = 0, dTz = 0, even parity). Particle ranks are a best guess, to be
/// confirmed by the operator-library milestone.
OperatorCatalogEntry p1(std::string_view name, Params params, OperatorKind kind, int rank,
                        std::string_view doc, int min_params = -1, bool macro = false) {
  OperatorCatalogEntry e;
  e.name = name;
  e.params = params;
  e.min_params = min_params < 0 ? static_cast<int>(params.size()) : min_params;
  e.macro = macro;
  e.shape = OperatorShape{0, 0, 0, rank};
  e.phase = Phase::P1;
  e.kind = kind;
  e.doc = doc;
  return e;
}

/// A planned or not-supported row: variadic, no kind.
OperatorCatalogEntry later(std::string_view name, Phase phase, OperatorShape shape,
                           std::string_view doc, bool macro = false) {
  OperatorCatalogEntry e;
  e.name = name;
  e.variadic = true;
  e.macro = macro;
  e.shape = shape;
  e.phase = phase;
  e.doc = doc;
  return e;
}

void add_p1_rows(std::vector<OperatorCatalogEntry>& v) {
  const Params none;
  using K = OperatorKind;
  v.push_back(p1("Trel", none, K::Trel, 2, "intrinsic kinetic energy T - T_cm"));
  v.push_back(p1("TCM", none, K::Tcm, 2, "center-of-mass kinetic energy"));
  v.push_back(p1("Tlab", none, K::Tlab, 1, "lab-frame kinetic energy (one-body)"));
  v.push_back(p1("TrelMassCorrection", none, K::TrelMassCorrection, 2,
                 "proton-neutron mass-difference correction to Trel"));
  v.push_back(p1("HCM", Params(kHw), K::Hcm, 2,
                 "Lawson center-of-mass Hamiltonian; without <hw> uses the basis hw", 0));
  v.push_back(p1("VCM", Params(kHw), K::Vcm, 2, "oscillator term of HCM at frequency <hw>"));
  v.push_back(p1("R2CM", none, K::R2cm, 2, "center-of-mass R^2"));
  v.push_back(p1("Rp2", none, K::Rp2, 2, "intrinsic point-proton radius squared"));
  v.push_back(p1("Rn2", none, K::Rn2, 2, "intrinsic point-neutron radius squared"));
  v.push_back(p1("Rm2", none, K::Rm2, 2, "intrinsic point-matter radius squared"));
  v.push_back(p1("Rm2lab", none, K::Rm2lab, 1, "lab-frame matter radius squared (one-body)"));
  v.push_back(p1("Rp2Z", Params(kZ), K::Rp2Z, 2, "Rp2 with the proton number overridden"));
  v.push_back(p1("Rp2AZ", Params(kAZ), K::Rp2AZ, 2, "Rp2 with mass and proton number overridden"));
  v.push_back(p1("Rn2Z", Params(kZ), K::Rn2Z, 2, "Rn2 with the proton number overridden"));
  v.push_back(p1("R2_p1", none, K::R2p1, 1, "one-body part of the proton r^2"));
  v.push_back(p1("R2_n1", none, K::R2n1, 1, "one-body part of the neutron r^2"));
  v.push_back(p1("R2_p2", none, K::R2p2, 2, "two-body part (r_i.r_j) of the proton r^2"));
  v.push_back(p1("R2_n2", none, K::R2n2, 2, "two-body part (r_i.r_j) of the neutron r^2"));
  v.push_back(p1("Rso", none, K::Rso, 1, "spin-orbit correction to the proton radius"));
  v.push_back(p1("E0", none, K::E0, 1, "isoscalar monopole operator"));
  v.push_back(p1("OccRef", none, K::OccRef, 1, "number operator on the reference orbits"));
  v.push_back(p1("OneOcc", Params(kOrbit), K::OneOcc, 1, "occupation number of one orbit"));
  v.push_back(p1("AllOcc", Params(kOrbit), K::AllOcc, 1,
                 "occupation summed over n for the (l, j, tz) of the orbit"));
  v.push_back(
      p1("OBD", Params(kOrbit2), K::Obd, 1, "one-body density matrix element <orbit1|.|orbit2>"));
  v.push_back(p1("rhop", Params(kR), K::RhoP, 1, "point-proton density at radius r"));
  v.push_back(p1("rhon", Params(kR), K::RhoN, 1, "point-neutron density at radius r"));
  v.push_back(p1("rhop_all", none, K::RhoP, 1,
                 "macro: rhop_<r> for r = 0, 0.2, ..., 10 (51 operators)", -1, true));
  v.push_back(p1("rhon_all", none, K::RhoN, 1,
                 "macro: rhon_<r> for r = 0, 0.2, ..., 10 (51 operators)", -1, true));
  v.push_back(p1("rhocentralp", Params(kRmax), K::RhoCentralP, 1,
                 "central proton density averaged on a sphere of radius rmax"));
  v.push_back(p1("rhocentraln", Params(kRmax), K::RhoCentralN, 1,
                 "central neutron density averaged on a sphere of radius rmax"));
  v.push_back(p1("Iso2", none, K::Iso2, 2, "total isospin squared T^2"));
  v.push_back(p1("Tz2", none, K::Tz2, 2, "isospin projection squared Tz^2"));
  v.push_back(p1("LdotS", none, K::LdotS, 1, "one-body l.s"));
}

void add_planned_rows(std::vector<OperatorCatalogEntry>& v) {
  const auto pl = [&v](std::string_view name, OperatorShape shape, std::string_view doc,
                       bool macro = false) {
    v.push_back(later(name, Phase::Planned, shape, doc, macro));
  };
  // Shape order: rank_j, dtz, parity, particle_rank. dTz is in imsrg++ units (Operator(ms, J,
  // dTz, parity, rank)): 1 for beta decay, 2 for 0vbb.
  // Scalar potentials.
  pl("VCoul", {0, 0, 0, 2}, "Coulomb interaction between protons");
  pl("VCentralCoul", {0, 0, 0, 2}, "central Coulomb interaction");
  pl("VMinnesota", {0, 0, 0, 2}, "Minnesota NN potential");
  pl("VGaus", {0, 0, 0, 2}, "Gaussian two-body potential, VGaus_<s>");
  pl("VGausCSB", {0, 0, 0, 2}, "Gaussian charge-symmetry-breaking potential, VGausCSB_<s>");
  pl("VBareDelta", {0, 0, 0, 2}, "contact (delta) interaction");
  pl("VSDI", {0, 0, 0, 2}, "surface delta interaction, VSDI_<V0>_<R>");
  pl("VWS", {0, 0, 0, 1}, "Woods-Saxon potential, VWS_<V0>_<R0>_<a0>");
  pl("HOtrap", {0, 0, 0, 1}, "harmonic-oscillator trap, HOtrap_<hw>");
  pl("QdotQ", {0, 0, 0, 2}, "Q.Q two-body operator (any name starting with QdotQ)");
  pl("VQQ", {0, 0, 0, 2}, "quadrupole-quadrupole interaction");
  pl("L2rel", {0, 0, 0, 2}, "relative orbital angular momentum squared");
  pl("FFp", {0, 0, 0, 1}, "point-proton form factor, FFp_<q>");
  pl("FFn", {0, 0, 0, 1}, "point-neutron form factor, FFn_<q>");
  pl("protonFBC", {0, 0, 0, 1}, "proton Fourier-Bessel coefficient, protonFBC_<nu>");
  pl("neutronFBC", {0, 0, 0, 1}, "neutron Fourier-Bessel coefficient, neutronFBC_<nu>");
  // Tensors: electric and magnetic multipoles.
  static constexpr std::string_view kE[] = {"E1", "E2", "E3", "E4", "E5", "E6"};
  static constexpr std::string_view kM[] = {"M1", "M2", "M3", "M4", "M5"};
  for (int k = 1; k <= 6; ++k) {
    pl(kE[k - 1], {k, 0, k % 2, 1}, "electric multipole of rank k");
  }
  pl("E2int", {2, 0, 0, 2}, "intrinsic (CM-corrected) electric quadrupole");
  pl("nE2", {2, 0, 0, 1}, "neutron electric quadrupole");
  for (int k = 1; k <= 5; ++k) {
    pl(kM[k - 1], {k, 0, (k + 1) % 2, 1}, "magnetic multipole of rank k");
  }
  pl("M1p", {1, 0, 0, 1}, "proton magnetic dipole");
  pl("M1n", {1, 0, 0, 1}, "neutron magnetic dipole");
  pl("M1S", {1, 0, 0, 1}, "spin part of the magnetic dipole");
  pl("M1L", {1, 0, 0, 1}, "orbital part of the magnetic dipole");
  pl("Sigma", {1, 0, 0, 1}, "spin operator");
  pl("Sigma_p", {1, 0, 0, 1}, "proton spin operator");
  pl("Sigma_n", {1, 0, 0, 1}, "neutron spin operator");
  pl("ISM", {0, 0, 0, 1}, "isoscalar monopole response operator");
  pl("IVM", {0, 0, 0, 1}, "isovector monopole response operator");
  pl("ISQ", {2, 0, 0, 1}, "isoscalar quadrupole response operator");
  pl("IVQ", {2, 0, 0, 1}, "isovector quadrupole response operator");
  pl("ISO", {3, 0, 1, 1}, "isoscalar octupole response operator");
  pl("IVO", {3, 0, 1, 1}, "isovector octupole response operator");
  pl("IVD", {1, 0, 1, 1}, "isovector dipole response operator");
  pl("ISD", {1, 0, 1, 1}, "isoscalar dipole response operator, ISD_<Rms>");
  pl("Schiff", {1, 0, 1, 1}, "Schiff moment, Schiff_<R>");
  pl("Anapole", {1, 0, 1, 1}, "anapole moment");
  pl("GamowTeller", {1, 1, 0, 2}, "allowed Gamow-Teller operator");
  // Charge changing and parity odd.
  pl("Fermi", {0, 1, 0, 2}, "allowed Fermi operator");
  pl("AxialCharge", {0, 1, 1, 2}, "axial charge");
  pl("RadialOverlap", {0, 1, 0, 1}, "radial overlap between protons and neutrons");
  pl("VPT", {0, 0, 1, 2}, "parity- and time-reversal-violating potential, VPT_<c1>_<c2>_<c3>...");
  // Dagger operators.
  pl("Dagger", {0, 0, 0, 1}, "particle-adding operator, Dagger_<orb>");
  pl("DaggerHF", {0, 0, 0, 1}, "particle-adding operator in the HF basis, DaggerHF_<orb>");
  pl("DaggerAlln", {0, 0, 0, 1}, "particle-adding operator summed over n, DaggerAlln_<orb>");
  pl("DaggerHF_valence", {0, 0, 0, 1}, "macro: DaggerHF_<orb> for each valence orbit", true);
  pl("DaggerAlln_valence", {0, 0, 0, 1}, "macro: DaggerAlln_<orb> for each valence orbit", true);
  // Neutrinoless double beta decay and double Gamow-Teller.
  pl("M0nu_GT", {0, 2, 0, 2}, "0vbb Gamow-Teller part, M0nu_GT_<Ecl>_<src>[_<ff>]");
  pl("M0nu_F", {0, 2, 0, 2}, "0vbb Fermi part, M0nu_F_<Ecl>_<src>");
  pl("M0nu_T", {0, 2, 0, 2}, "0vbb tensor part, M0nu_T_<Ecl>_<src>[_<ff>]");
  pl("M0nu_C", {0, 2, 0, 2}, "0vbb contact part, M0nu_C_<cutoff>_<power>");
  pl("M0nuHeavy", {0, 2, 0, 2}, "0vbb heavy-neutrino exchange, M0nuHeavy_<GT|F|T>_<src>_<ff>");
  pl("M0nuIntegrand", {0, 2, 0, 2}, "0vbb integrand at one momentum-mesh point");
  pl("M0nuR", {0, 2, 0, 2}, "0vbb Gamow-Teller at fixed relative distance, M0nuR_<Ecl>_<r12>");
  pl("DGT", {0, 2, 0, 2}, "double Gamow-Teller operator");
  pl("DGTR", {0, 2, 0, 2}, "radial dependence of DGT, DGTR_<r12>");
  pl("DGTRLocal", {0, 2, 0, 2}, "DGT with surface localization, DGTRLocal_<r12>");
  pl("DMNREFT", {0, 0, 0, 1}, "dark-matter NR-EFT operator, DMNREFT_<Op><IsoSV>_<q>_<J>");
}

void add_not_supported_rows(std::vector<OperatorCatalogEntry>& v) {
  v.push_back(later("hfsNMS", Phase::NotSupported, {0, 0, 0, 1}, "atomic normal mass shift"));
  v.push_back(later("hfsSMS", Phase::NotSupported, {0, 0, 0, 2}, "atomic specific mass shift"));
}

const std::vector<OperatorCatalogEntry>& catalog_rows() {
  static const std::vector<OperatorCatalogEntry> rows = [] {
    std::vector<OperatorCatalogEntry> v;
    v.reserve(160);
    add_p1_rows(v);
    add_planned_rows(v);
    add_not_supported_rows(v);
    return v;
  }();
  return rows;
}

// --------------------------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------------------------

std::vector<std::string_view> split_underscore(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t pos = s.find('_', start);
    if (pos == std::string_view::npos) {
      out.push_back(s.substr(start));
      return out;
    }
    out.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
}

/// Shortest round-trip decimal form of `d` (3.0 -> "3", 0.2 -> "0.2"); negative zero prints "0".
std::string format_number(double d) {
  if (d == 0.0) d = 0.0;
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), d);
  return std::string(buf, res.ptr);
}

std::string format_param(const OpParam& p) {
  if (const auto* i = std::get_if<std::int64_t>(&p)) return std::to_string(*i);
  if (const auto* d = std::get_if<double>(&p)) return format_number(*d);
  return std::get<std::string>(p);
}

std::string canonical_of(std::string_view name, const std::vector<OpParam>& params) {
  std::string s(name);
  for (const auto& p : params) {
    s += '_';
    s += format_param(p);
  }
  return s;
}

std::string type_text(const OpParamSpec& p) {
  switch (p.type) {
    case OpParamType::Integer:
      return "integer";
    case OpParamType::Number:
      return p.non_negative ? "number >= 0" : "number";
    case OpParamType::Orbit:
      return "orbit";
    case OpParamType::Word:
      return "word";
  }
  return "?";
}

std::string schema_text(const OperatorCatalogEntry& row) {
  std::string s;
  for (const auto& p : row.params) {
    if (!s.empty()) s += ", ";
    s += std::string(p.name) + ": " + type_text(p);
  }
  return s;
}

std::string plural_params(std::size_t n) {
  return n == 1 ? "1 parameter" : std::to_string(n) + " parameters";
}

/// "expects 1 parameter (r: number >= 0)", "expects 0 or 1 parameters (...)", "expects no
/// parameters".
std::string expects_text(const OperatorCatalogEntry& row) {
  const auto n = row.params.size();
  const auto m = static_cast<std::size_t>(row.min_params);
  if (n == 0) return "expects no parameters";
  std::string s = "expects ";
  s += m == n ? plural_params(n) : std::to_string(m) + " to " + plural_params(n);
  s += " (" + schema_text(row) + ")";
  return s;
}

struct ParamError {
  std::string code;
  std::string detail;
};

bool is_word(std::string_view s) {
  if (s.empty()) return false;
  return std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' ||
           c == '-';
  });
}

std::optional<ParamError> check_number(double d, const OpParamSpec& spec) {
  if (!std::isfinite(d)) {
    return ParamError{"E-OP-PARAMS", "'" + std::string(spec.name) + "' must be finite"};
  }
  if (spec.non_negative && d < 0.0) {
    return ParamError{"E-OP-PARAMS", "'" + format_number(d) + "' is negative (" +
                                         std::string(spec.name) + " must be >= 0)"};
  }
  return std::nullopt;
}

/// Parses an orbit label and checks e <= emax. Malformed labels are E-OP-PARAMS, labels outside
/// the model space E-OP-ORBIT.
std::optional<ParamError> make_orbit(std::string_view label, int emax, OpParam& out) {
  basis::Orbit o;
  try {
    o = basis::parse_orbit_label(label);
  } catch (const InputError&) {
    return ParamError{"E-OP-PARAMS",
                      "'" + std::string(label) + "' is not an orbit label (expected e.g. p0d5)"};
  }
  if (o.e() > emax) {
    return ParamError{"E-OP-ORBIT",
                      "orbit '" + std::string(label) + "' (e = " + std::to_string(o.e()) +
                          ") is outside the model space (emax = " + std::to_string(emax) + ")"};
  }
  out = basis::orbit_label(o);
  return std::nullopt;
}

/// Converts one string token per the schema.
std::optional<ParamError> convert_token(const OpParamSpec& spec, std::string_view tok, int emax,
                                        OpParam& out) {
  const auto quoted = "'" + std::string(tok) + "'";
  switch (spec.type) {
    case OpParamType::Integer: {
      std::int64_t v = 0;
      const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
      if (tok.empty() || res.ec != std::errc{} || res.ptr != tok.data() + tok.size()) {
        return ParamError{"E-OP-PARAMS", quoted + " is not an integer"};
      }
      out = v;
      return std::nullopt;
    }
    case OpParamType::Number: {
      double v = 0.0;
      const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
      if (tok.empty() || res.ec != std::errc{} || res.ptr != tok.data() + tok.size() ||
          !std::isfinite(v)) {
        return ParamError{"E-OP-PARAMS", quoted + " is not a number"};
      }
      if (auto err = check_number(v, spec)) return err;
      out = v == 0.0 ? 0.0 : v;
      return std::nullopt;
    }
    case OpParamType::Orbit:
      return make_orbit(tok, emax, out);
    case OpParamType::Word:
      if (!is_word(tok)) {
        return ParamError{"E-OP-PARAMS", quoted + " is not a word ([A-Za-z0-9+-]+)"};
      }
      out = std::string(tok);
      return std::nullopt;
  }
  return std::nullopt;
}

/// Converts one inline-table value per the schema.
std::optional<ParamError> convert_value(const OpParamSpec& spec, const RawValue& val, int emax,
                                        OpParam& out) {
  const auto key = "'" + std::string(spec.name) + "'";
  const auto wrong_type = [&](std::string_view expected) {
    return ParamError{"E-OP-PARAMS", "parameter " + key + " must be " + std::string(expected) +
                                         ", got " + std::string(val.type_name())};
  };
  switch (spec.type) {
    case OpParamType::Integer: {
      const auto* i = val.get_if<std::int64_t>();
      if (!i) return wrong_type("an integer");
      out = *i;
      return std::nullopt;
    }
    case OpParamType::Number: {
      double d = 0.0;
      if (const auto* i = val.get_if<std::int64_t>()) {
        d = static_cast<double>(*i);
      } else if (const auto* f = val.get_if<double>()) {
        d = *f;
      } else {
        return wrong_type("a number");
      }
      if (auto err = check_number(d, spec)) return err;
      out = d == 0.0 ? 0.0 : d;
      return std::nullopt;
    }
    case OpParamType::Orbit: {
      const auto* s = val.get_if<std::string>();
      if (!s) return wrong_type("an orbit label string");
      return make_orbit(*s, emax, out);
    }
    case OpParamType::Word: {
      const auto* s = val.get_if<std::string>();
      if (!s) return wrong_type("a string");
      if (!is_word(*s)) {
        return ParamError{"E-OP-PARAMS", "parameter " + key + ": '" + *s + "' is not a word"};
      }
      out = *s;
      return std::nullopt;
    }
  }
  return std::nullopt;
}

OperatorRequest make_request(const OperatorCatalogEntry& row, std::vector<OpParam> params) {
  OperatorRequest r;
  r.kind = *row.kind;
  r.canonical_name = canonical_of(row.name, params);
  r.params = std::move(params);
  r.shape = row.shape;
  r.phase = Phase::P1;
  return r;
}

/// The 51 requests of rhop_all / rhon_all: r = i / 5.0, i = 0..50 (exact decimals).
void expand_macro(const OperatorCatalogEntry& row, std::vector<OperatorRequest>& out) {
  const std::string_view base = row.name.substr(0, row.name.find('_'));
  const OperatorCatalogEntry* target = nullptr;
  for (const auto& e : catalog_rows()) {
    if (e.name == base) target = &e;
  }
  if (target == nullptr) throw InvariantError("operator macro without a target row");
  for (int i = 0; i <= 50; ++i) {
    out.push_back(make_request(*target, {OpParam{i / 5.0}}));
  }
}

class ListParser {
 public:
  ListParser(const ModelSpaceInfo& info, std::string_view key_path)
      : emax_(info.emax), key_path_(key_path) {}

  void parse_element(const RawValue& spec) {
    if (const auto* s = spec.get_if<std::string>()) {
      parse_string(*s, spec.where);
    } else if (const auto* t = spec.get_if<RawTable>()) {
      parse_table(*t, spec.where);
    } else {
      error("E-OP-PARAMS",
            "an operator spec must be a string or an inline table, got " +
                std::string(spec.type_name()),
            spec.where);
    }
  }

  Expected<std::vector<OperatorRequest>, ConfigIssues> finish() {
    if (has_errors(issues_)) return Unexpected<ConfigIssues>{std::move(issues_)};
    return std::move(requests_);
  }

 private:
  void error(std::string code, std::string message, const SourceRef& where,
             std::optional<std::string> suggestion = std::nullopt) {
    issues_.push_back(
        make_error(std::move(code), key_path_, std::move(message), where, std::move(suggestion)));
  }

  /// Adds the requests of one spec; reports duplicates against everything added before.
  void add(std::vector<OperatorRequest> reqs, const SourceRef& where) {
    for (auto& r : reqs) {
      if (!seen_.insert(r.canonical_name).second) {
        error("E-OP-DUPLICATE", "operator '" + r.canonical_name + "' is requested more than once",
              where);
        continue;
      }
      requests_.push_back(std::move(r));
    }
  }

  std::optional<std::string> suggest(const std::vector<std::string_view>& tokens) {
    std::vector<std::string_view> names;
    for (const auto& e : catalog_rows()) names.push_back(e.name);
    if (tokens.size() >= 2) {
      const std::string joined = std::string(tokens[0]) + "_" + std::string(tokens[1]);
      if (auto s = did_you_mean(joined, names)) return s;
    }
    return did_you_mean(tokens[0], names);
  }

  /// Handles the phase of a matched row; returns true if parsing must stop (planned / NS).
  bool reject_phase(const OperatorCatalogEntry& row, const std::string& shown,
                    const SourceRef& where) {
    if (row.phase == Phase::Planned) {
      error("E-PLANNED", "operator '" + shown + "' is not available in this build", where);
      return true;
    }
    if (row.phase == Phase::NotSupported) {
      error("E-OP-NS", "operator '" + shown + "' is not supported (" + std::string(row.doc) + ")",
            where);
      return true;
    }
    return false;
  }

  void parse_string(const std::string& spec, const SourceRef& where) {
    if (spec.empty()) {
      error("E-OP-PARAMS", "empty operator spec", where);
      return;
    }
    const auto tokens = split_underscore(spec);
    const OperatorCatalogEntry* row = nullptr;
    std::size_t name_tokens = 0;
    for (const auto& e : catalog_rows()) {
      const auto nt = split_underscore(e.name);
      if (nt.size() <= name_tokens || nt.size() > tokens.size()) continue;
      if (std::equal(nt.begin(), nt.end(), tokens.begin())) {
        row = &e;
        name_tokens = nt.size();
      }
    }
    if (row == nullptr && tokens[0].starts_with("QdotQ")) {
      for (const auto& e : catalog_rows()) {
        if (e.name == "QdotQ") row = &e;
      }
      name_tokens = 1;
    }
    if (row == nullptr) {
      const std::string name(tokens[0]);
      error("E-OP-UNKNOWN", "operator '" + name + "' is not known", where, suggest(tokens));
      return;
    }
    if (reject_phase(*row, spec, where)) return;

    const std::size_t given = tokens.size() - name_tokens;
    const auto n = row->params.size();
    if (given < static_cast<std::size_t>(row->min_params) || given > n) {
      error("E-OP-PARAMS",
            "'" + std::string(row->name) + "' " + expects_text(*row) + ", got " +
                std::to_string(given),
            where);
      return;
    }
    std::vector<OpParam> params;
    bool ok = true;
    for (std::size_t i = 0; i < given; ++i) {
      OpParam value;
      if (auto err = convert_token(row->params[i], tokens[name_tokens + i], emax_, value)) {
        error(err->code, "'" + spec + "': " + err->detail, where);
        ok = false;
      } else {
        params.push_back(std::move(value));
      }
    }
    if (!ok) return;
    finish_row(*row, std::move(params), where);
  }

  void finish_row(const OperatorCatalogEntry& row, std::vector<OpParam> params,
                  const SourceRef& where) {
    std::vector<OperatorRequest> reqs;
    if (row.macro) {
      expand_macro(row, reqs);
    } else {
      reqs.push_back(make_request(row, std::move(params)));
    }
    add(std::move(reqs), where);
  }

  void parse_table(const RawTable& table, const SourceRef& where) {
    const RawValue* name_val = nullptr;
    for (const auto& [k, v] : table) {
      if (k == "name") name_val = &v;
    }
    const auto* name = name_val ? name_val->get_if<std::string>() : nullptr;
    if (name == nullptr) {
      error("E-OP-PARAMS", "an inline operator table needs a string key 'name'", where);
      return;
    }
    const OperatorCatalogEntry* row = nullptr;
    for (const auto& e : catalog_rows()) {
      if (e.name == *name) row = &e;
    }
    if (row == nullptr) {
      error("E-OP-UNKNOWN", "operator '" + *name + "' is not known", where,
            suggest(split_underscore(*name)));
      return;
    }
    if (reject_phase(*row, *name, where)) return;

    bool ok = true;
    std::set<std::string_view> keys;
    for (const auto& [k, v] : table) {
      if (k == "name") continue;
      const bool known = std::any_of(row->params.begin(), row->params.end(),
                                     [&](const OpParamSpec& p) { return p.name == k; });
      if (!known) {
        const auto sch = schema_text(*row);
        error("E-OP-PARAMS",
              "'" + *name + "': unexpected key '" + k +
                  "' (parameters: " + (sch.empty() ? "none" : sch) + ")",
              where);
        ok = false;
      } else if (!keys.insert(k).second) {
        error("E-OP-PARAMS", "'" + *name + "': key '" + k + "' given more than once", where);
        ok = false;
      }
    }
    std::vector<OpParam> params;
    bool gap = false;
    for (std::size_t i = 0; i < row->params.size(); ++i) {
      const auto& spec = row->params[i];
      const RawValue* val = nullptr;
      for (const auto& [k, v] : table) {
        if (k == spec.name) val = &v;
      }
      if (val == nullptr) {
        if (static_cast<int>(i) < row->min_params) {
          error("E-OP-PARAMS",
                "'" + *name + "' " + expects_text(*row) + ": missing parameter '" +
                    std::string(spec.name) + "'",
                where);
          ok = false;
        }
        gap = true;
        continue;
      }
      if (gap) {
        error("E-OP-PARAMS",
              "'" + *name + "': parameter '" + std::string(spec.name) +
                  "' given after a missing optional parameter",
              where);
        ok = false;
        continue;
      }
      OpParam value;
      if (auto err = convert_value(spec, *val, emax_, value)) {
        error(err->code, "'" + *name + "': " + err->detail, where);
        ok = false;
      } else {
        params.push_back(std::move(value));
      }
    }
    if (!ok) return;
    finish_row(*row, std::move(params), where);
  }

  int emax_;
  std::string key_path_;
  ConfigIssues issues_;
  std::vector<OperatorRequest> requests_;
  std::set<std::string> seen_;
};

std::string shape_text(const OperatorShape& s) {
  std::ostringstream os;
  os << "J=" << s.rank_j << " dTz=" << s.dtz << " P=" << (s.parity == 0 ? '+' : '-')
     << " rank=" << s.particle_rank;
  return os.str();
}

std::string display_name(const OperatorCatalogEntry& e) {
  if (e.name == "QdotQ") return "QdotQ*";
  std::string s(e.name);
  if (e.variadic) {
    // Planned / unsupported families: parameters are not checked yet.
    return s;
  }
  for (std::size_t i = 0; i < e.params.size(); ++i) {
    const std::string p = "<" + std::string(e.params[i].name) + ">";
    if (static_cast<int>(i) >= e.min_params) {
      s += "[_" + p + "]";
    } else {
      s += "_" + p;
    }
  }
  return s;
}

}  // namespace

std::span<const OperatorCatalogEntry> operator_catalog() noexcept {
  try {
    return catalog_rows();
  } catch (...) {
    return {};
  }
}

Expected<std::vector<OperatorRequest>, ConfigIssues> parse_operator_list(
    std::span<const RawValue> specs, const ModelSpaceInfo& info, std::string_view key_path) {
  ListParser parser(info, key_path);
  for (const auto& spec : specs) parser.parse_element(spec);
  return parser.finish();
}

Expected<std::vector<OperatorRequest>, ConfigIssues> parse_operator_spec(
    std::string_view spec, const ModelSpaceInfo& info) {
  const RawValue value{RawValue::Data{std::string(spec)}, SourceRef{}};
  return parse_operator_list(std::span<const RawValue>(&value, 1), info, "operators.list");
}

std::string operator_catalog_text() {
  std::ostringstream os;
  const struct {
    Phase phase;
    const char* heading;
  } groups[] = {{Phase::P1, "P1 (implemented)"},
                {Phase::Planned, "planned (recognised, not available in this build)"},
                {Phase::NotSupported, "not supported"}};
  for (const auto& g : groups) {
    os << g.heading << ":\n";
    for (const auto& e : catalog_rows()) {
      if (e.phase != g.phase) continue;
      os << "  " << display_name(e);
      if (!e.params.empty()) {
        os << "  [";
        for (std::size_t i = 0; i < e.params.size(); ++i) {
          if (i > 0) os << "; ";
          os << e.params[i].name << ": " << type_text(e.params[i]) << ", " << e.params[i].doc;
        }
        os << "]";
      }
      os << "  " << shape_text(e.shape) << "  " << e.doc << "\n";
    }
    os << "\n";
  }
  return os.str();
}

}  // namespace imsrg::driver
