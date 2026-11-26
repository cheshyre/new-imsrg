// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Legacy imsrg++ `key=value` command line -> raw tree in new-schema paths
// (design/16 section 3).
#include "imsrg/driver/legacy/legacy_parameters.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/basis/nucleus.h"
#include "imsrg/basis/valence_space.h"

namespace imsrg::driver {

namespace {

using K = LegacyKind;
constexpr LegacyStatus kS = LegacyStatus::kSupported;
constexpr LegacyStatus kC = LegacyStatus::kChanged;
constexpr LegacyStatus kNS = LegacyStatus::kNotSupported;
constexpr LegacyStatus kPL = LegacyStatus::kPlanned;

// The 85 keys of Parameters.hh in declaration order, plus the fork 79e9330 keys
// use_HF_reference_in_NAT, only_2b_omega_at_end, reduced_232_impl and
// imsrg3_commutator_depth (89) (design/16 sections
// 3.1-3.4).
constexpr std::array<LegacyKeyInfo, 89> kKeys = {{
    {"2bme", K::kString, "none", kS, "interaction.two_body.file",
     "none = no 2N input (error unless fmt2=schematic, 2bme=Minnesota); "
     "oakridge sp,vnn -> 2 "
     "files"},
    {"3bme", K::kString, "none", kS, "interaction.three_body.file",
     "none = no 3N table"},
    {"3bme_type", K::kString, "full", kS, "interaction.three_body.type",
     "full|no2b|mono; dropped (W-L1) without 3bme"},
    {"no2b_precision", K::kString, "single", kS,
     "interaction.three_body.no2b_precision",
     "single|half; dropped (W-L1) without 3bme"},
    {"core_generator", K::kString, "atan", kS, "imsrg.generator.core",
     "hyphenated names mapped; qtransfer-atan_<n> -> qtransfer_atan + "
     "qtransfer_n"},
    {"valence_generator", K::kString, "shell-model-atan", kS,
     "imsrg.generator.valence", "dropped (W-NSTEPS1) with nsteps=1"},
    {"flowfile", K::kString, "default", kC, "output.flow_file",
     "default -> unset"},
    {"intfile", K::kString, "default", kC, "output.directory, output.stem",
     "default -> unset; dir/stem split at the last '/'"},
    {"fmt2", K::kString, "me2j", kS, "interaction.two_body.format",
     "nushellx also sets hamiltonian.kinetic=none; schematic only with "
     "2bme=Minnesota"},
    {"fmt3", K::kString, "me3j", kS, "interaction.three_body.format",
     "me3j|navratil; hdf5 rejected (design/00 D24)"},
    {"input_op_fmt", K::kString, "navratil", kPL, "operators.from_file",
     "applied to every OperatorsFromFile entry as its format"},
    {"reference", K::kString, "default", kC, "reference.nucleus",
     "default -> unset"},
    {"valence_space", K::kString, "", kC, "valence.space, reference.nucleus",
     "named space/0hw-shell/jj-shell/FCI/comma list -> valence.space; a "
     "nucleus -> reference"},
    {"custom_valence_space", K::kString, "", kS, "valence.custom",
     "wins over valence_space (warning V-30)"},
    {"basis", K::kString, "HF", kS, "basis.kind",
     "HF|oscillator|NAT -> hf|oscillator|nat"},
    {"method", K::kString, "magnus", kC, "imsrg.method, run.task",
     "shorthands HF MP3 FCI NSmagnus brueckner brueckner1step (brueckner2 "
     "planned)"},
    {"denominator_delta_orbit", K::kString, "none", kS,
     "imsrg.denominator.delta_orbit", "none -> unset"},
    {"LECs", K::kString, "EM2.0_2.0", kPL, "interaction.two_body.lecs",
     "explicit value ignored with a warning"},
    {"scratch", K::kString, "", kC, "imsrg.omega.storage, run.scratch_dir",
     "/dev/null -> discard; otherwise scratch + scratch_dir"},
    {"use_brueckner_bch", K::kString, "false", kS, "imsrg.use_brueckner_bch",
     "bool"},
    {"valence_file_format", K::kString, "tokyo", kS,
     "output.valence_file_format", "tokyo|nushellx|antoine"},
    {"occ_file", K::kString, "none", kS, "reference.occupations_file",
     "none -> unset"},
    {"goose_tank", K::kString, "false", kPL, "imsrg.goose_tank", "bool"},
    {"write_omega", K::kString, "false", kC, "output.write_omega",
     "bool; always writes (Q16)"},
    {"nucleon_mass_correction", K::kString, "false", kS,
     "hamiltonian.nucleon_mass_correction", "bool"},
    {"hunter_gatherer", K::kString, "false", kS, "imsrg.omega.hunter_gatherer",
     "bool"},
    {"relativistic_correction", K::kString, "false", kPL,
     "hamiltonian.relativistic_correction", "bool"},
    {"IMSRG3", K::kString, "false", kS, "imsrg3.in_flow", "bool"},
    {"imsrg3_n7", K::kString, "false", kS, "imsrg3.variant",
     "true -> variant n7 (rule L2)"},
    {"imsrg3_mp4", K::kString, "false", kPL, "imsrg3.variant",
     "true -> variant mp4 (rule L2)"},
    {"imsrg3_at_end", K::kString, "false", kS, "imsrg3.at_end",
     "needs a magnus method (V-52); Q11"},
    {"imsrg3_no_qqq", K::kString, "false", kPL, "imsrg3.no_qqq",
     "bool (rule L2)"},
    {"physical_system", K::kString, "nuclear", kS, "-",
     "nuclear accepted and dropped; atomic rejected"},
    {"freeze_occupations", K::kString, "false", kS,
     "hartree_fock.freeze_occupations", "bool"},
    {"discard_no2b_from_3n", K::kString, "false", kPL,
     "interaction.three_body.discard_no2b_after_hf",
     "bool; dropped (W-L1) without 3bme"},
    {"use_NAT_occupations", K::kString, "false", kS,
     "natural_orbitals.use_occupations", "dropped (V-21) unless basis=NAT"},
    {"use_HF_reference_in_NAT", K::kString, "false", kS,
     "natural_orbitals.use_hf_reference",
     "fork 79e9330 key; dropped (V-21) unless basis=NAT"},
    {"order_NAT_by_energy", K::kString, "false", kC, "natural_orbitals.order",
     "true -> order=energy, overriding NAT_order"},
    {"NAT_order", K::kString, "occupation", kS, "natural_orbitals.order",
     "occupation|energy; dropped (V-21) unless basis=NAT"},
    {"store_3bme_pn", K::kString, "false", kS,
     "interaction.three_body.store_pn", "bool; dropped (W-L1) without 3bme"},
    {"discard_residual_input3N", K::kString, "false", kS,
     "imsrg3.discard_residual_input_3n", "bool"},
    {"only_2b_eta", K::kString, "false", kS, "imsrg3.only_2b_eta",
     "bool (rule L2)"},
    {"only_2b_omega", K::kString, "false", kS, "imsrg3.only_2b_omega",
     "bool (rule L2)"},
    {"perturbative_triples", K::kString, "false", kS,
     "imsrg.perturbative_triples",
     "bool; dropped (W-Q10) unless method=magnus"},
    {"write_HO_ops", K::kString, "false", kPL, "output.write_ho_ops", "bool"},
    {"write_HF_ops", K::kString, "false", kPL, "output.write_hf_ops", "bool"},
    {"denominator_partitioning", K::kString, "Epstein_Nesbet", kS,
     "imsrg.denominator.partitioning",
     "Epstein_Nesbet|Moller_Plesset|MP_isospin"},
    {"hw", K::kDouble, "20.0", kS, "basis.hw", "supplied default 20.0"},
    {"smax", K::kDouble, "200", kS, "imsrg.step.smax", "0 = no flow"},
    {"dsmax", K::kDouble, "0.5", kS, "imsrg.step.dsmax", ""},
    {"ds_0", K::kDouble, "0.5", kC, "imsrg.step.ds_0",
     "unused by magnus (W-Q2)"},
    {"domega", K::kDouble, "0.1", kS, "imsrg.step.domega", ""},
    {"omega_norm_max", K::kDouble, "0.25", kS, "imsrg.step.omega_norm_max",
     "overridden by NSmagnus/brueckner* (W-Q15)"},
    {"ode_tolerance", K::kDouble, "1e-6", kPL, "imsrg.step.ode_tolerance",
     "flow methods"},
    {"denominator_delta", K::kDouble, "0", kC, "imsrg.denominator.delta",
     "dropped (W-Q3) without denominator_delta_orbit"},
    {"BetaCM", K::kDouble, "0", kS, "hamiltonian.beta_cm", ""},
    {"hwBetaCM", K::kDouble, "-1", kS, "hamiltonian.beta_cm_hw",
     "< 0 -> unset"},
    {"eta_criterion", K::kDouble, "1e-6", kS, "imsrg.step.eta_criterion", ""},
    {"hw_trap", K::kDouble, "-1", kC,
     "hamiltonian.trap_hw, hamiltonian.kinetic",
     "<= 0 -> unset; > 0 -> trap_hw and kinetic=lab"},
    {"dE3max", K::kDouble, "99", kS, "imsrg3.de3max",
     "dropped (W-L2) without IMSRG(3)/triples"},
    {"OccNat3Cut", K::kDouble, "-1", kS, "imsrg3.occ_nat3_cut",
     "<= 0 -> unset"},
    {"threebody_threshold", K::kDouble, "0", kS, "imsrg3.threebody_threshold",
     "nonzero dropped (W-L2) without IMSRG(3)"},
    {"A", K::kInt, "-1", kS, "basis.mass_number", "<= 0 -> unset"},
    {"e3max", K::kInt, "12", kS, "basis.e3max", ""},
    {"emax", K::kInt, "6", kS, "basis.emax", "supplied default 6"},
    {"lmax", K::kInt, "99999", kS, "basis.lmax", "99999 or >= emax -> unset"},
    {"lmax3", K::kInt, "-1", kS, "basis.lmax3", "<= 0 -> unset"},
    {"nsteps", K::kInt, "-1", kC, "imsrg.steps",
     "-1 -> unset; only 1 or 2 (W-Q9, W-NSTEPS1)"},
    {"file2e1max", K::kInt, "12", kS, "interaction.two_body.e1max",
     "supplied default 12"},
    {"file2e2max", K::kInt, "-1", kC, "interaction.two_body.e2max",
     "<= 0 -> unset"},
    {"file2lmax", K::kInt, "-1", kC, "interaction.two_body.lmax",
     "<= 0 -> unset; explicit 0 warns"},
    {"file3e1max", K::kInt, "12", kS, "interaction.three_body.e1max",
     "supplied default 12 with 3bme"},
    {"file3e2max", K::kInt, "-1", kS, "interaction.three_body.e2max",
     "-1 -> unset"},
    {"file3e3max", K::kInt, "12", kS, "interaction.three_body.e3max",
     "supplied default 12 with 3bme"},
    {"atomicZ", K::kInt, "-1", kNS, "",
     "atomic systems are not supported by imsrg"},
    {"emax_unocc", K::kInt, "-1", kS, "basis.emax_unocc", "<= 0 -> unset"},
    {"emax_imsrg", K::kInt, "-1", kS, "imsrg.space.emax", "< 0 -> unset"},
    {"e2max_imsrg", K::kInt, "-1", kS, "imsrg.space.e2max", "< 0 -> unset"},
    {"e3max_imsrg", K::kInt, "-1", kS, "imsrg.space.e3max", "< 0 -> unset"},
    {"emax_3body_imsrg", K::kInt, "-1", kS, "imsrg.space.emax_3body",
     "< 0 -> unset"},
    {"Operators", K::kVector, "", kS, "operators.list",
     "strings, parsed later"},
    {"OperatorsFromFile", K::kVector, "", kPL, "operators.from_file",
     "Name^J_T_P_R^file2b[^file3b] -> table"},
    {"OperatorsPT1", K::kVector, "", kPL, "operators.first_order",
     "strings, parsed later"},
    {"OperatorsRPA", K::kVector, "", kNS, "",
     "RPA-resummed operator corrections are not supported"},
    {"OperatorsTDA", K::kVector, "", kNS, "",
     "TDA-resummed operator corrections are not supported"},
    {"SPWF", K::kVector, "", kS, "output.spwf", "strings, parsed later"},
    // Keys of the fork imsrg++ 79e9330 (27Al archive, design/50) that only
    // concern IMSRG(3) variants: accepted at their defaults and dropped.
    {"only_2b_omega_at_end", K::kString, "false", kS, "-",
     "fork 79e9330 key; only false accepted (dropped)"},
    {"reduced_232_impl", K::kString, "false", kS, "-",
     "fork 79e9330 key; only false accepted (dropped)"},
    {"imsrg3_commutator_depth", K::kInt, "100", kS, "-",
     "fork 79e9330 key; only 100 accepted (dropped)"},
}};

// -----------------------------------------------------------------------------
// Spelling tables for enumerations (legacy spelling -> canonical schema
// spelling). The canonical spelling itself is always accepted too (rule L3).

struct Spelling {
  std::string_view spelling;
  std::string_view canonical;
};

constexpr std::array kGenerators = std::to_array<Spelling>({
    {"white", "white"},
    {"atan", "atan"},
    {"imaginary-time", "imaginary_time"},
    {"wegner", "wegner"},
    {"qtransfer-atan", "qtransfer_atan"},
    {"shell-model", "shell_model"},
    {"shell-model-atan", "shell_model_atan"},
    {"shell-model-wegner", "shell_model_wegner"},
    {"shell-model-atan-npnh", "shell_model_atan_npnh"},
    {"shell-model-imaginary-time", "shell_model_imaginary_time"},
    {"hartree-fock", "hartree_fock"},
    {"1PA", "one_pa"},
});
constexpr std::array kBasis = std::to_array<Spelling>({
    {"HF", "hf"},
    {"oscillator", "oscillator"},
    {"NAT", "nat"},
});
constexpr std::array kPartitioning = std::to_array<Spelling>({
    {"Epstein_Nesbet", "epstein_nesbet"},
    {"Moller_Plesset", "moller_plesset"},
    {"Moller-Plesset", "moller_plesset"},
    {"MollerPlesset", "moller_plesset"},
    {"MP", "moller_plesset"},
    {"MP_isospin", "mp_isospin"},
});
constexpr std::array k3bmeType = std::to_array<Spelling>({
    {"full", "full"},
    {"no2b", "no2b"},
    {"mono", "mono"},
});
constexpr std::array kNo2bPrecision = std::to_array<Spelling>({
    {"single", "single"},
    {"half", "half"},
});
constexpr std::array kFmt2 = std::to_array<Spelling>({
    {"me2j", "me2j"},
    {"navratil", "navratil"},
    {"Navratil", "navratil"},
    {"oslo", "oslo"},
    {"oakridge", "oakridge"},
    {"oakridgebin", "oakridge_bin"},
    {"oakridge_bin", "oakridge_bin"},
    {"takayuki", "takayuki"},
    {"nushellx", "nushellx"},
    {"schematic", "schematic_minnesota"},
});
constexpr std::array kFmt3 = std::to_array<Spelling>({
    {"me3j", "me3j"},
    {"navratil", "navratil"},
    {"Navratil", "navratil"},
    {"hdf5", "hdf5"},
});
constexpr std::array kValenceFileFormat = std::to_array<Spelling>({
    {"tokyo", "tokyo"},
    {"nushellx", "nushellx"},
    {"antoine", "antoine"},
});
constexpr std::array kNatOrder = std::to_array<Spelling>({
    {"occupation", "occupation"},
    {"energy", "energy"},
});
constexpr std::array kOpFmt = std::to_array<Spelling>({
    {"navratil", "navratil"},
    {"Navratil", "navratil"},
    {"miyagi", "miyagi"},
});

/// Pass-through method values (design/16 3.5) and the full list of accepted
/// spellings.
constexpr std::array kVerbatimMethods = std::to_array<std::string_view>({
    "magnus",
    "magnus_backoff",
    "magnus_modified_euler",
    "magnus_adaptive",
    "flow",
    "flow_euler",
    "restore_4th_order",
});
constexpr std::array kMethodSpellings = std::to_array<std::string_view>({
    "HF",
    "MP3",
    "FCI",
    "NSmagnus",
    "brueckner",
    "brueckner1step",
    "brueckner2",
    "magnus_euler",
    "flow_adaptive",
    "flow_RK4",
    "flow_rk4",
    "magnus",
    "magnus_backoff",
    "magnus_modified_euler",
    "magnus_adaptive",
    "flow",
    "flow_euler",
    "restore_4th_order",
});

// -----------------------------------------------------------------------------
// Direct (1:1) key tables.

enum class Absent {
  kNever,  // always emitted
  kLE0,    // absent if <= 0
  kLT0,    // absent if < 0
  kEqM1,   // absent if == -1
};

struct DirectNumber {
  std::string_view key;
  std::string_view path;
  Absent absent;
  bool three_body_only;
};

constexpr std::array kDirectInts = std::to_array<DirectNumber>({
    {"emax", "basis.emax", Absent::kNever, false},
    {"e3max", "basis.e3max", Absent::kNever, false},
    {"A", "basis.mass_number", Absent::kLE0, false},
    {"lmax3", "basis.lmax3", Absent::kLE0, false},
    {"emax_unocc", "basis.emax_unocc", Absent::kLE0, false},
    {"file2e1max", "interaction.two_body.e1max", Absent::kNever, false},
    {"file2e2max", "interaction.two_body.e2max", Absent::kLE0, false},
    {"file3e1max", "interaction.three_body.e1max", Absent::kNever, true},
    {"file3e2max", "interaction.three_body.e2max", Absent::kEqM1, true},
    {"file3e3max", "interaction.three_body.e3max", Absent::kNever, true},
    {"emax_imsrg", "imsrg.space.emax", Absent::kLT0, false},
    {"e2max_imsrg", "imsrg.space.e2max", Absent::kLT0, false},
    {"e3max_imsrg", "imsrg.space.e3max", Absent::kLT0, false},
    {"emax_3body_imsrg", "imsrg.space.emax_3body", Absent::kLT0, false},
});

constexpr std::array kDirectDoubles = std::to_array<DirectNumber>({
    {"hw", "basis.hw", Absent::kNever, false},
    {"smax", "imsrg.step.smax", Absent::kNever, false},
    {"dsmax", "imsrg.step.dsmax", Absent::kNever, false},
    {"ds_0", "imsrg.step.ds_0", Absent::kNever, false},
    {"domega", "imsrg.step.domega", Absent::kNever, false},
    {"ode_tolerance", "imsrg.step.ode_tolerance", Absent::kNever, false},
    {"denominator_delta", "imsrg.denominator.delta", Absent::kNever, false},
    {"BetaCM", "hamiltonian.beta_cm", Absent::kNever, false},
    {"hwBetaCM", "hamiltonian.beta_cm_hw", Absent::kLT0, false},
    {"eta_criterion", "imsrg.step.eta_criterion", Absent::kNever, false},
    {"dE3max", "imsrg3.de3max", Absent::kNever, false},
    {"OccNat3Cut", "imsrg3.occ_nat3_cut", Absent::kLE0, false},
    {"threebody_threshold", "imsrg3.threebody_threshold", Absent::kNever,
     false},
});

struct DirectBool {
  std::string_view key;
  std::string_view path;
  bool three_body_only;
};

constexpr std::array kDirectBools = std::to_array<DirectBool>({
    {"goose_tank", "imsrg.goose_tank", false},
    {"write_omega", "output.write_omega", false},
    {"nucleon_mass_correction", "hamiltonian.nucleon_mass_correction", false},
    {"hunter_gatherer", "imsrg.omega.hunter_gatherer", false},
    {"relativistic_correction", "hamiltonian.relativistic_correction", false},
    {"IMSRG3", "imsrg3.in_flow", false},
    {"imsrg3_at_end", "imsrg3.at_end", false},
    {"imsrg3_no_qqq", "imsrg3.no_qqq", false},
    {"freeze_occupations", "hartree_fock.freeze_occupations", false},
    {"discard_no2b_from_3n", "interaction.three_body.discard_no2b_after_hf",
     true},
    {"use_NAT_occupations", "natural_orbitals.use_occupations", false},
    {"use_HF_reference_in_NAT", "natural_orbitals.use_hf_reference", false},
    {"store_3bme_pn", "interaction.three_body.store_pn", true},
    {"discard_residual_input3N", "imsrg3.discard_residual_input_3n", false},
    {"only_2b_eta", "imsrg3.only_2b_eta", false},
    {"only_2b_omega", "imsrg3.only_2b_omega", false},
    {"perturbative_triples", "imsrg.perturbative_triples", false},
    {"write_HO_ops", "output.write_ho_ops", false},
    {"write_HF_ops", "output.write_hf_ops", false},
});

bool AbsentBySentinel(Absent rule, double v) {
  switch (rule) {
    case Absent::kNever:
      return false;
    case Absent::kLE0:
      return v <= 0.0;
    case Absent::kLT0:
      return v < 0.0;
    case Absent::kEqM1:
      return v == -1.0;
  }
  return false;
}

/// Drops one leading '+' (imsrg++ read numbers with istringstream, which
/// accepts it); "+-1" stays invalid.
std::string_view StripPlus(std::string_view s) {
  if (s.size() > 1 && s.front() == '+' && s[1] != '-') s.remove_prefix(1);
  return s;
}

std::optional<std::int64_t> TryInt(std::string_view s) {
  s = StripPlus(s);
  std::int64_t v = 0;
  const char* b = s.data();
  const char* e = s.data() + s.size();
  const auto r = std::from_chars(b, e, v);
  if (r.ec != std::errc{} || r.ptr != e || s.empty()) return std::nullopt;
  return v;
}

/// Finite doubles only ("inf" and "nan" are not valid parameter values).
std::optional<double> TryDouble(std::string_view s) {
  s = StripPlus(s);
  double v = 0.0;
  const char* b = s.data();
  const char* e = s.data() + s.size();
  const auto r = std::from_chars(b, e, v);
  if (r.ec != std::errc{} || r.ptr != e || s.empty() || !std::isfinite(v))
    return std::nullopt;
  return v;
}

std::vector<std::string> Split(std::string_view s, char sep) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (true) {
    const std::size_t p = s.find(sep, start);
    if (p == std::string_view::npos) {
      out.emplace_back(s.substr(start));
      return out;
    }
    out.emplace_back(s.substr(start, p - start));
    start = p + 1;
  }
}

/// One collected `key=value` (or one element of a vector key).
struct Entry {
  std::string value;
  std::string token;  ///< the verbatim legacy token
};

struct Gen {
  std::string canon;
  std::optional<std::int64_t> n;  ///< qtransfer n
};

class Translator {
 public:
  LegacyTranslation Run(std::span<const std::string> args) {
    Collect(args);
    ParseEnums();
    DoMethod();
    DoGenerators();
    DoBasisAndMisc();
    DoTwoBody();
    DoThreeBody();
    DoNumbers();
    DoValenceAndReference();
    DoOutputs();
    DoVectors();
    DoCrossRules();
    return std::move(out_);
  }

 private:
  // ------------------------------------------------------------------ plumbing
  std::map<std::string, Entry, std::less<>> scalars_;
  std::map<std::string, std::vector<Entry>, std::less<>> vectors_;
  LegacyTranslation out_;

  // Parsed enums shared between groups.
  std::optional<std::string> fmt2_;
  std::optional<std::string> basis_;
  std::optional<std::string> op_fmt_;
  std::optional<Gen> core_gen_;
  std::optional<Gen> val_gen_;
  std::optional<double> forced_omega_;
  std::optional<std::int64_t> forced_steps_;
  bool forced_brueckner_ = false;
  bool core_from_valence_ = false;
  bool has_3n_ = false;

  const Entry* get(std::string_view key) const {
    const auto it = scalars_.find(key);
    return it == scalars_.end() ? nullptr : &it->second;
  }
  bool given(std::string_view key) const {
    return get(key) != nullptr || vectors_.contains(key);
  }
  // Given with a parsable value. Keys whose old default is supplied get it also
  // when the given value is invalid: that value is already reported
  // (E-LEGACY-VALUE), and a missing required key would only add a consequential
  // V-03.
  bool given_valid_int(std::string_view key) const {
    const Entry* e = get(key);
    return e != nullptr && TryInt(e->value).has_value();
  }
  bool given_valid_double(std::string_view key) const {
    const Entry* e = get(key);
    return e != nullptr && TryDouble(e->value).has_value();
  }

  static SourceRef ArgRef(const Entry& e) {
    return {SourceRef::Kind::kLegacyArg, e.token, 0, 0};
  }
  static SourceRef DefaultRef(std::string_view key) {
    return {SourceRef::Kind::kLegacyDefault, std::string(key), 0, 0};
  }
  SourceRef ref_of(std::string_view key) const {
    const Entry* e = get(key);
    return e ? ArgRef(*e) : SourceRef{};
  }

  void AddError(std::string code, std::string key, std::string msg,
                SourceRef where = {},
                std::optional<std::string> sug = std::nullopt) {
    out_.issues.push_back(MakeError(std::move(code), std::move(key),
                                    std::move(msg), std::move(where),
                                    std::move(sug)));
  }
  void Warn(std::string code, std::string key, std::string msg,
            SourceRef where = {}) {
    out_.issues.push_back(MakeWarning(std::move(code), std::move(key),
                                      std::move(msg), std::move(where)));
  }
  void Rule(std::string text) { out_.applied_rules.push_back(std::move(text)); }

  void Put(std::string_view path, RawValue::Data d, SourceRef where) {
    out_.tree.set(std::string(path), RawValue{std::move(d), std::move(where)});
  }
  void PutS(std::string_view path, std::string_view v, const SourceRef& where) {
    Put(path, std::string(v), where);
  }
  void PutI(std::string_view path, std::int64_t v, const SourceRef& where) {
    Put(path, v, where);
  }
  void PutD(std::string_view path, double v, const SourceRef& where) {
    Put(path, v, where);
  }
  void PutB(std::string_view path, bool v, const SourceRef& where) {
    Put(path, v, where);
  }

  bool in_tree(std::string_view path) const { return out_.tree.contains(path); }
  const RawValue* tree_find(std::string_view path) const {
    return out_.tree.find(path);
  }
  bool tree_true(std::string_view path) const {
    const RawValue* v = tree_find(path);
    if (v == nullptr) return false;
    const bool* b = std::get_if<bool>(&v->data);
    return b != nullptr && *b;
  }
  std::optional<std::string> tree_str(std::string_view path) const {
    const RawValue* v = tree_find(path);
    if (v == nullptr) return std::nullopt;
    const std::string* s = std::get_if<std::string>(&v->data);
    if (s == nullptr) return std::nullopt;
    return *s;
  }
  std::optional<std::int64_t> tree_int(std::string_view path) const {
    const RawValue* v = tree_find(path);
    if (v == nullptr) return std::nullopt;
    const auto* i = std::get_if<std::int64_t>(&v->data);
    if (i == nullptr) return std::nullopt;
    return *i;
  }
  std::optional<double> tree_double(std::string_view path) const {
    const RawValue* v = tree_find(path);
    if (v == nullptr) return std::nullopt;
    const auto* d = std::get_if<double>(&v->data);
    if (d == nullptr) return std::nullopt;
    return *d;
  }

  // ------------------------------------------------------------------ value
  // parsing
  std::optional<std::int64_t> ParseInt(const Entry& e, std::string_view key) {
    auto v = TryInt(e.value);
    if (!v)
      AddError("E-LEGACY-VALUE", std::string(key),
               "expected an integer for '" + std::string(key) + "', got '" +
                   e.value + "'",
               ArgRef(e));
    return v;
  }
  std::optional<double> ParseDouble(const Entry& e, std::string_view key) {
    auto v = TryDouble(e.value);
    if (!v)
      AddError("E-LEGACY-VALUE", std::string(key),
               "expected a number for '" + std::string(key) + "', got '" +
                   e.value + "'",
               ArgRef(e));
    return v;
  }
  std::optional<bool> ParseBool(const Entry& e, std::string_view key) {
    if (e.value == "true") return true;
    if (e.value == "false") return false;
    AddError("E-LEGACY-VALUE", std::string(key),
             "expected 'true' or 'false' for '" + std::string(key) +
                 "', got '" + e.value + "'",
             ArgRef(e));
    return std::nullopt;
  }

  /// Maps a spelling through `table`; the canonical spellings are accepted too.
  /// Unknown -> V-01.
  std::optional<std::string> MapEnum(const Entry& e, std::string_view path,
                                     std::span<const Spelling> table) {
    for (const Spelling& s : table)
      if (e.value == s.spelling) return std::string(s.canonical);
    for (const Spelling& s : table)
      if (e.value == s.canonical) return std::string(s.canonical);
    std::vector<std::string> cands;
    for (const Spelling& s : table) {
      for (std::string_view c : {s.spelling, s.canonical}) {
        if (std::find(cands.begin(), cands.end(), c) == cands.end())
          cands.emplace_back(c);
      }
    }
    AddError("V-01", std::string(path),
             "unknown value '" + e.value + "' for " + std::string(path),
             ArgRef(e), DidYouMean(e.value, cands));
    return std::nullopt;
  }

  std::optional<Gen> ParseGenerator(const Entry& e, std::string_view path) {
    for (std::string_view prefix : {std::string_view("qtransfer-atan_"),
                                    std::string_view("qtransfer_atan_")}) {
      if (e.value.starts_with(prefix)) {
        const auto n = TryInt(std::string_view(e.value).substr(prefix.size()));
        if (!n || *n < 1) {
          AddError("E-LEGACY-VALUE", std::string(path),
                   "qtransfer generator needs a positive integer n, got '" +
                       e.value + "'",
                   ArgRef(e));
          return std::nullopt;
        }
        return Gen{"qtransfer_atan", *n};
      }
    }
    auto canon = MapEnum(e, path, kGenerators);
    if (!canon) return std::nullopt;
    Gen g{*canon, std::nullopt};
    if (g.canon == "qtransfer_atan") g.n = 1;
    return g;
  }

  // ------------------------------------------------------------------ token
  // collection
  void Collect(std::span<const std::string> args) {
    std::vector<std::string_view> names;
    for (const auto& k : kKeys) names.push_back(k.key);
    for (const std::string& tok : args) {
      const std::size_t eq = tok.find('=');
      const SourceRef where{SourceRef::Kind::kLegacyArg, tok, 0, 0};
      if (eq == std::string::npos) {
        AddError("E-LEGACY-SYNTAX", tok,
                 "expected key=value, got '" + tok + "'", where);
        continue;
      }
      const std::string key = tok.substr(0, eq);
      const std::string value = tok.substr(eq + 1);
      const LegacyKeyInfo* info = FindLegacyKey(key);
      if (info == nullptr) {
        AddError("E-KEY-UNKNOWN", key, "unknown legacy key '" + key + "'",
                 where,
                 DidYouMean(key, std::span<const std::string_view>(names)));
        continue;
      }
      if (value.empty()) {
        AddError("E-LEGACY-SYNTAX", key,
                 "empty value for '" + key + "' (key= is not allowed)", where);
        continue;
      }
      if (info->kind == LegacyKind::kVector) {
        for (const std::string& part : Split(value, ',')) {
          if (part.empty()) {
            AddError("E-LEGACY-VALUE", key,
                     "empty element in the list value of '" + key + "'", where);
            continue;
          }
          vectors_[key].push_back(Entry{part, tok});
        }
        continue;
      }
      if (scalars_.contains(key))
        Warn("W-DUPLICATE", key,
             "key '" + key + "' given more than once; the last value wins",
             where);
      scalars_[key] = Entry{value, tok};
    }
  }

  // ------------------------------------------------------------------ enum
  // pre-pass
  void ParseEnums() {
    if (const Entry* e = get("fmt2"))
      fmt2_ = MapEnum(*e, "interaction.two_body.format", kFmt2);
    if (const Entry* e = get("basis"))
      basis_ = MapEnum(*e, "basis.kind", kBasis);
    if (const Entry* e = get("input_op_fmt"))
      op_fmt_ = MapEnum(*e, "operators.from_file", kOpFmt);
    if (const Entry* e = get("core_generator"))
      core_gen_ = ParseGenerator(*e, "imsrg.generator.core");
    if (const Entry* e = get("valence_generator"))
      val_gen_ = ParseGenerator(*e, "imsrg.generator.valence");
    const Entry* b3 = get("3bme");
    has_3n_ = b3 != nullptr && b3->value != "none";
  }

  // ------------------------------------------------------------------ method
  // (design/16 3.5)
  /// method = NSmagnus, brueckner or brueckner1step: Magnus with a forced
  /// omega_norm_max (and Brueckner BCH / one step).
  void ForceNsMagnusMethod(const std::string& v, const SourceRef& ref) {
    PutS("imsrg.method", "magnus", ref);
    std::string text = "method=" + v + " -> imsrg.method=magnus";
    const double onm = v == "NSmagnus" ? 50000.0 : 500.0;
    forced_omega_ = onm;
    text += ", imsrg.step.omega_norm_max=" +
            std::string(v == "NSmagnus" ? "50000" : "500");
    PutD("imsrg.step.omega_norm_max", onm, ref);
    if (const Entry* o = get("omega_norm_max")) {
      const auto given_v = ParseDouble(*o, "omega_norm_max");
      if (given_v && std::abs(*given_v - onm) > 1.0e-12)
        Warn("W-Q15", "imsrg.step.omega_norm_max",
             "method=" + v + " overrides omega_norm_max=" + o->value,
             ArgRef(*o));
    }
    if (v != "NSmagnus") {
      forced_brueckner_ = true;
      PutB("imsrg.use_brueckner_bch", true, ref);
      text += ", imsrg.use_brueckner_bch=true";
    }
    if (v == "brueckner1step") {
      forced_steps_ = 1;
      PutI("imsrg.steps", 1, ref);
      core_from_valence_ = true;
      text += ", imsrg.steps=1, imsrg.generator.core=<valence_generator>";
    }
    Rule(text);
  }

  void DoMethod() {
    const Entry* m = get("method");
    if (m == nullptr) return;
    const std::string& v = m->value;
    const SourceRef ref = ArgRef(*m);
    const auto verbatim =
        std::find(kVerbatimMethods.begin(), kVerbatimMethods.end(), v);
    if (v == "HF" || v == "MP3" || v == "FCI") {
      const char* task = v == "HF" ? "hf" : (v == "MP3" ? "mp3" : "fci_export");
      PutS("run.task", task, ref);
      Rule("method=" + v + " -> run.task=" + task);
    } else if (v == "NSmagnus" || v == "brueckner" || v == "brueckner1step") {
      ForceNsMagnusMethod(v, ref);
    } else if (v == "brueckner2") {
      AddError(
          "E-PLANNED", "imsrg.method",
          "method=brueckner2 (HF-basis rotation restart) is not available in "
          "this build",
          ref);
    } else if (v == "magnus_euler") {
      PutS("imsrg.method", "magnus", ref);
      Rule("method=magnus_euler -> imsrg.method=magnus");
    } else if (v == "flow_adaptive") {
      PutS("imsrg.method", "flow", ref);
      Rule("method=flow_adaptive -> imsrg.method=flow");
    } else if (v == "flow_RK4" || v == "flow_rk4") {
      PutS("imsrg.method", "flow_rk4", ref);
      Rule("method=" + v + " -> imsrg.method=flow_rk4");
    } else if (verbatim != kVerbatimMethods.end()) {
      PutS("imsrg.method", v, ref);
    } else {
      std::vector<std::string> cands(kMethodSpellings.begin(),
                                     kMethodSpellings.end());
      AddError("V-01", "imsrg.method", "unknown method '" + v + "'", ref,
               DidYouMean(v, cands));
    }
  }

  // ------------------------------------------------------------------
  // generators
  void DoGenerators() {
    std::optional<Gen> core = core_gen_;
    SourceRef core_ref = ref_of("core_generator");
    if (core_from_valence_) {
      core = val_gen_
                 ? val_gen_
                 : std::optional<Gen>(Gen{"shell_model_atan", std::nullopt});
      core_ref = ref_of("method");
    }
    if (core) PutS("imsrg.generator.core", core->canon, core_ref);
    if (val_gen_)
      PutS("imsrg.generator.valence", val_gen_->canon,
           ref_of("valence_generator"));
    // qtransfer_n: both generators must agree.
    std::optional<std::int64_t> n;
    SourceRef n_ref;
    bool conflict = false;
    const auto consider = [&](const std::optional<Gen>& g, const SourceRef& r) {
      if (!g || !g->n) return;
      if (n && *n != *g->n) conflict = true;
      if (!n) {
        n = g->n;
        n_ref = r;
      }
    };
    consider(core, core_ref);
    consider(val_gen_, ref_of("valence_generator"));
    if (conflict)
      AddError(
          "E-LEGACY-VALUE", "imsrg.generator.qtransfer_n",
          "core_generator and valence_generator name different qtransfer n",
          n_ref);
    else if (n)
      PutI("imsrg.generator.qtransfer_n", *n, n_ref);
  }

  // ------------------------------------------------------------------ basis,
  // bools, enums
  void DoBasisAndMisc() {
    if (basis_) PutS("basis.kind", *basis_, ref_of("basis"));
    DoDirectBools();
    DoImsrg3Variant();
    DoEnumKeys();
    DoUnsupportedSystem();
  }

  /// Bools mapped 1:1, and use_brueckner_bch.
  void DoDirectBools() {
    // Bools mapped 1:1.
    for (const DirectBool& d : kDirectBools) {
      const Entry* e = get(d.key);
      if (e == nullptr) continue;
      if (d.three_body_only && !has_3n_) {
        Drop3nOnly(d.key, *e);
        continue;
      }
      const auto b = ParseBool(*e, d.key);
      if (b) PutB(d.path, *b, ArgRef(*e));
    }
    if (const Entry* e = get("use_brueckner_bch")) {
      const auto b = ParseBool(*e, "use_brueckner_bch");
      if (b && !forced_brueckner_)
        PutB("imsrg.use_brueckner_bch", *b, ArgRef(*e));
    }
  }

  /// imsrg3_n7 / imsrg3_mp4 -> imsrg3.variant.
  void DoImsrg3Variant() {
    // imsrg3 variant.
    const Entry* n7 = get("imsrg3_n7");
    const Entry* mp4 = get("imsrg3_mp4");
    // An unparsable value was reported by ParseBool; it counts as false here.
    const bool n7v =
        n7 != nullptr && ParseBool(*n7, "imsrg3_n7").value_or(false);
    const bool mp4v =
        mp4 != nullptr && ParseBool(*mp4, "imsrg3_mp4").value_or(false);
    if (n7v && mp4v)
      AddError("E-LEGACY-VALUE", "imsrg3.variant",
               "imsrg3_n7=true and imsrg3_mp4=true are exclusive",
               ArgRef(*mp4));
    else if (n7v)
      PutS("imsrg3.variant", "n7", ArgRef(*n7));
    else if (mp4v)
      PutS("imsrg3.variant", "mp4", ArgRef(*mp4));
  }

  /// Enum-valued and string keys: NAT order, valence file format, denominator
  /// partitioning and delta orbit, occupations file.
  void DoEnumKeys() {
    // order_NAT_by_energy overrides NAT_order.
    if (const Entry* e = get("NAT_order")) {
      const auto o = MapEnum(*e, "natural_orbitals.order", kNatOrder);
      if (o) PutS("natural_orbitals.order", *o, ArgRef(*e));
    }
    if (const Entry* e = get("order_NAT_by_energy")) {
      const auto b = ParseBool(*e, "order_NAT_by_energy");
      if (b && *b) {
        PutS("natural_orbitals.order", "energy", ArgRef(*e));
        Rule("order_NAT_by_energy=true -> natural_orbitals.order=energy");
      }
    }

    if (const Entry* e = get("valence_file_format")) {
      const auto o =
          MapEnum(*e, "output.valence_file_format", kValenceFileFormat);
      if (o) PutS("output.valence_file_format", *o, ArgRef(*e));
    }
    if (const Entry* e = get("denominator_partitioning")) {
      const auto o =
          MapEnum(*e, "imsrg.denominator.partitioning", kPartitioning);
      if (o) PutS("imsrg.denominator.partitioning", *o, ArgRef(*e));
    }
    if (const Entry* e = get("denominator_delta_orbit");
        e && e->value != "none")
      PutS("imsrg.denominator.delta_orbit", e->value, ArgRef(*e));
    if (const Entry* e = get("occ_file"); e && e->value != "none")
      PutS("reference.occupations_file", e->value, ArgRef(*e));
  }

  /// Rejects atomic systems; warns about the ignored LECs key.
  void DoUnsupportedSystem() {
    if (const Entry* e = get("physical_system")) {
      if (e->value == "atomic")
        AddError("E-LEGACY-NS", "physical_system",
                 "atomic systems are not supported by imsrg", ArgRef(*e));
      else if (e->value != "nuclear")
        AddError("E-LEGACY-VALUE", "physical_system",
                 "physical_system must be 'nuclear' or 'atomic', got '" +
                     e->value + "'",
                 ArgRef(*e));
    }
    // Fork 79e9330 IMSRG(3) keys: only their defaults (dropped).
    for (const auto& [key, def] :
         {std::pair<std::string_view, std::string_view>{"only_2b_omega_at_end",
                                                        "false"},
          {"reduced_232_impl", "false"},
          {"imsrg3_commutator_depth", "100"}}) {
      if (const Entry* e = get(key); e && e->value != def) {
        AddError("E-LEGACY-NS", std::string(key),
                 "the fork 79e9330 option '" + std::string(key) +
                     "' is supported only at its default " + std::string(def),
                 ArgRef(*e));
      }
    }
    if (const Entry* e = get("atomicZ"))
      AddError("E-LEGACY-NS", "atomicZ",
               "atomic systems are not supported by imsrg", ArgRef(*e));
    if (const Entry* e = get("LECs"))
      Warn("W-L1", "LECs",
           "LECs=" + e->value + " is ignored (only used by hdf5 readers)",
           ArgRef(*e));
  }

  void Drop3nOnly(std::string_view key, const Entry& e) {
    Warn("W-L1", std::string(key),
         "'" + std::string(key) + "' is ignored: no 3N table (3bme) is given",
         ArgRef(e));
  }

  // ------------------------------------------------------------------ two-body
  // input
  void DoTwoBody() {
    const Entry* b2 = get("2bme");
    const bool none = b2 == nullptr || b2->value == "none";
    const SourceRef fmt_ref = ref_of("fmt2");
    if (fmt2_ && *fmt2_ == "schematic_minnesota") {
      if (b2 != nullptr && b2->value == "Minnesota") {
        PutS("interaction.two_body.file", "Minnesota", ArgRef(*b2));
        PutS("interaction.two_body.format", "schematic_minnesota", fmt_ref);
      } else {
        AddError("E-LEGACY-VALUE", "fmt2",
                 "fmt2=schematic is only valid with 2bme=Minnesota",
                 b2 ? ArgRef(*b2) : fmt_ref);
      }
    } else if (none) {
      AddError("V-03", "interaction.two_body.file",
               "no two-body interaction: '2bme' is required (a file path)",
               b2 ? ArgRef(*b2) : SourceRef{});
    } else {
      const bool oak =
          fmt2_ && (*fmt2_ == "oakridge" || *fmt2_ == "oakridge_bin");
      if (oak && b2->value.find(',') != std::string::npos) {
        const auto parts = Split(b2->value, ',');
        if (parts.size() != 2 || parts[0].empty() || parts[1].empty()) {
          AddError("E-LEGACY-VALUE", "2bme",
                   "oakridge formats take 'sp_file,vnn_file'", ArgRef(*b2));
        } else {
          RawArray arr;
          for (const auto& p : parts) arr.push_back(RawValue{p, ArgRef(*b2)});
          Put("interaction.two_body.file", std::move(arr), ArgRef(*b2));
        }
      } else {
        PutS("interaction.two_body.file", b2->value, ArgRef(*b2));
      }
      if (fmt2_) PutS("interaction.two_body.format", *fmt2_, fmt_ref);
    }
    if (fmt2_ && *fmt2_ == "nushellx") {
      PutS("hamiltonian.kinetic", "none", fmt_ref);
      Rule("fmt2=nushellx -> hamiltonian.kinetic=none");
    }
  }

  // ------------------------------------------------------------------
  // three-body input
  void DoThreeBody() {
    const Entry* b3 = get("3bme");
    if (has_3n_) {
      PutS("interaction.three_body.file", b3->value, ArgRef(*b3));
      if (!given_valid_int("file3e1max"))
        PutI("interaction.three_body.e1max", 12, DefaultRef("file3e1max"));
      if (!given_valid_int("file3e3max"))
        PutI("interaction.three_body.e3max", 12, DefaultRef("file3e3max"));
    }
    struct EnumKey {
      std::string_view key;
      std::string_view path;
      std::span<const Spelling> table;
    };
    const std::array<EnumKey, 3> enums = {{
        {"3bme_type", "interaction.three_body.type", k3bmeType},
        {"no2b_precision", "interaction.three_body.no2b_precision",
         kNo2bPrecision},
        {"fmt3", "interaction.three_body.format", kFmt3},
    }};
    for (const EnumKey& k : enums) {
      const Entry* e = get(k.key);
      if (e == nullptr) continue;
      const auto v = MapEnum(*e, k.path, k.table);
      if (!v) continue;
      if (k.key == "fmt3" && *v == "hdf5") {
        AddError("E-LEGACY-NS", "fmt3",
                 "HDF5 3N files are not supported (design/00 D24)", ArgRef(*e));
        continue;
      }
      if (!has_3n_) {
        Drop3nOnly(k.key, *e);
        continue;
      }
      PutS(k.path, *v, ArgRef(*e));
    }
  }

  // ------------------------------------------------------------------ numbers
  std::int64_t FinalEmax() {
    if (const Entry* e = get("emax"))
      if (const auto v = TryInt(e->value)) return *v;
    return 6;
  }

  void DoNumbers() {
    // Supplied defaults.
    if (!given_valid_int("emax")) PutI("basis.emax", 6, DefaultRef("emax"));
    if (!given_valid_double("hw")) PutD("basis.hw", 20.0, DefaultRef("hw"));
    if (!given_valid_int("file2e1max"))
      PutI("interaction.two_body.e1max", 12, DefaultRef("file2e1max"));
    DoDirectNumbers();
    DoCutNumbers();
    DoStepsAndTrap();
  }

  /// Integer and double keys mapped 1:1.
  void DoDirectNumbers() {
    for (const DirectNumber& d : kDirectInts) {
      const Entry* e = get(d.key);
      if (e == nullptr) continue;
      if (d.three_body_only && !has_3n_) {
        Drop3nOnly(d.key, *e);
        continue;
      }
      const auto v = ParseInt(*e, d.key);
      if (!v || AbsentBySentinel(d.absent, static_cast<double>(*v))) continue;
      PutI(d.path, *v, ArgRef(*e));
    }
    for (const DirectNumber& d : kDirectDoubles) {
      const Entry* e = get(d.key);
      if (e == nullptr) continue;
      const auto v = ParseDouble(*e, d.key);
      if (!v || AbsentBySentinel(d.absent, *v)) continue;
      PutD(d.path, *v, ArgRef(*e));
    }
  }

  /// omega_norm_max, lmax and file2lmax.
  void DoCutNumbers() {
    if (const Entry* e = get("omega_norm_max"); e && !forced_omega_) {
      if (const auto v = ParseDouble(*e, "omega_norm_max"))
        PutD("imsrg.step.omega_norm_max", *v, ArgRef(*e));
    }
    if (const Entry* e = get("lmax")) {
      const auto v = ParseInt(*e, "lmax");
      if (v && *v != 99999 && *v < FinalEmax())
        PutI("basis.lmax", *v, ArgRef(*e));
    }
    if (const Entry* e = get("file2lmax")) {
      if (const auto v = ParseInt(*e, "file2lmax")) {
        if (*v > 0) {
          PutI("interaction.two_body.lmax", *v, ArgRef(*e));
        } else if (*v == 0) {
          Warn("W-Q4", "file2lmax",
               "file2lmax=0 is treated as the default (= file2e1max) by "
               "imsrg++; ignored",
               ArgRef(*e));
        }
      }
    }
  }

  /// nsteps and hw_trap.
  void DoStepsAndTrap() {
    if (const Entry* e = get("nsteps"); e && !forced_steps_) {
      if (const auto v = ParseInt(*e, "nsteps")) {
        if (*v == 1 || *v == 2)
          PutI("imsrg.steps", *v, ArgRef(*e));
        else if (*v != -1)
          AddError("E-LEGACY-VALUE", "nsteps",
                   "nsteps must be -1, 1 or 2, got " + e->value, ArgRef(*e));
      }
    }
    if (const Entry* e = get("hw_trap")) {
      if (const auto v = ParseDouble(*e, "hw_trap"); v && *v > 0.0) {
        PutD("hamiltonian.trap_hw", *v, ArgRef(*e));
        PutS("hamiltonian.kinetic", "lab", ArgRef(*e));
        Rule("hw_trap=" + e->value +
             " -> hamiltonian.trap_hw, hamiltonian.kinetic=lab");
      }
    }
  }

  // ------------------------------------------------------------------ valence
  // space, reference
  void DoValenceAndReference() {
    const Entry* ref = get("reference");
    const bool ref_set = ref != nullptr && ref->value != "default";
    if (ref_set) PutS("reference.nucleus", ref->value, ArgRef(*ref));

    if (const Entry* c = get("custom_valence_space")) {
      PutS("valence.custom", c->value, ArgRef(*c));
      if (const Entry* v = get("valence_space"))
        Warn("V-30", "valence.space",
             "custom_valence_space overrides valence_space=" + v->value +
                 " (ignored)",
             ArgRef(*v));
      return;
    }
    const Entry* v = get("valence_space");
    if (v == nullptr) return;
    const std::string& s = v->value;
    if (basis::FindNamedValenceSpace(s) != nullptr || s == "0hw-shell" ||
        s == "jj-shell" || s.find("FCI") != std::string::npos ||
        s.find(',') != std::string::npos) {
      PutS("valence.space", s, ArgRef(*v));
      return;
    }
    bool is_nucleus = true;
    try {
      (void)basis::Nucleus::Parse(s);
    } catch (const Error&) {
      is_nucleus = false;
    }
    if (!is_nucleus) {
      std::vector<std::string> names;
      for (const auto& n : basis::NamedValenceSpaces())
        names.emplace_back(n.name);
      AddError(
          "V-01", "valence.space",
          "valence_space '" + s + "' is neither a valence space nor a nucleus",
          ArgRef(*v), DidYouMean(s, names));
      return;
    }
    if (!ref_set) {
      PutS("reference.nucleus", s, ArgRef(*v));
      Rule("valence_space=" + s + " -> reference.nucleus=" + s);
    } else if (ref->value != s) {
      AddError("E-LEGACY-NS", "valence_space",
               "valence_space=" + s +
                   " names a nucleus different from reference=" + ref->value +
                   "; not supported",
               ArgRef(*v));
    }
  }

  // ------------------------------------------------------------------ output /
  // misc paths
  void DoOutputs() {
    if (const Entry* e = get("intfile"); e && e->value != "default") {
      const std::size_t p = e->value.rfind('/');
      std::string dir = ".";
      std::string stem = e->value;
      if (p != std::string::npos) {
        dir = p == 0 ? "/" : e->value.substr(0, p);
        stem = e->value.substr(p + 1);
      }
      if (stem.empty()) {
        AddError("E-LEGACY-VALUE", "intfile", "intfile must end in a file stem",
                 ArgRef(*e));
      } else {
        PutS("output.directory", dir, ArgRef(*e));
        PutS("output.stem", stem, ArgRef(*e));
        Rule("intfile=" + e->value + " -> output.directory=" + dir +
             ", output.stem=" + stem);
      }
    }
    if (const Entry* e = get("flowfile"); e && e->value != "default")
      PutS("output.flow_file", e->value, ArgRef(*e));
    if (const Entry* e = get("scratch")) {
      if (e->value == "/dev/null" || e->value == "/dev/null/") {
        PutS("imsrg.omega.storage", "discard", ArgRef(*e));
        Rule("scratch=" + e->value + " -> imsrg.omega.storage=discard");
      } else {
        PutS("imsrg.omega.storage", "scratch", ArgRef(*e));
        PutS("run.scratch_dir", e->value, ArgRef(*e));
        Rule("scratch=" + e->value +
             " -> imsrg.omega.storage=scratch, run.scratch_dir");
      }
    }
  }

  // ------------------------------------------------------------------ vector
  // keys
  RawArray StringArray(const std::vector<Entry>& v) {
    RawArray arr;
    for (const Entry& e : v) arr.push_back(RawValue{e.value, ArgRef(e)});
    return arr;
  }

  void DoVectors() {
    struct V {
      std::string_view key;
      std::string_view path;
    };
    for (const V& k : {V{"Operators", "operators.list"},
                       V{"OperatorsPT1", "operators.first_order"},
                       V{"SPWF", "output.spwf"}}) {
      const auto it = vectors_.find(k.key);
      if (it == vectors_.end()) continue;
      Put(k.path, StringArray(it->second), ArgRef(it->second.front()));
    }
    for (std::string_view key :
         {std::string_view("OperatorsRPA"), std::string_view("OperatorsTDA")}) {
      const auto it = vectors_.find(key);
      if (it == vectors_.end()) continue;
      AddError("E-LEGACY-NS", std::string(key),
               std::string(key == "OperatorsRPA" ? "RPA" : "TDA") +
                   "-resummed operator corrections are not supported",
               ArgRef(it->second.front()));
    }
    if (const auto it = vectors_.find("OperatorsFromFile");
        it != vectors_.end()) {
      RawArray tables;
      for (const Entry& e : it->second) {
        if (auto t = ParseFromFileTag(e)) tables.push_back(std::move(*t));
      }
      if (!tables.empty())
        Put("operators.from_file", std::move(tables),
            ArgRef(it->second.front()));
    }
  }

  std::optional<RawValue> ParseFromFileTag(const Entry& e) {
    const SourceRef ref = ArgRef(e);
    const auto bad = [&](const std::string& why) {
      AddError("E-LEGACY-VALUE", "OperatorsFromFile",
               "malformed OperatorsFromFile tag '" + e.value +
                   "' (expected Name^J_T_P_R^file2b[^file3b]): " + why,
               ref);
      return std::optional<RawValue>{};
    };
    const auto parts = Split(e.value, '^');
    if (parts.size() < 3 || parts.size() > 4)
      return bad("wrong number of '^' fields");
    if (parts[0].empty() || parts[2].empty() ||
        (parts.size() == 4 && parts[3].empty()))
      return bad("empty field");
    const auto jtpr = Split(parts[1], '_');
    if (jtpr.size() != 4) return bad("J_T_P_R needs four integers");
    std::array<std::int64_t, 4> n{};
    for (std::size_t i = 0; i < 4; ++i) {
      const auto v = TryInt(jtpr[i]);
      if (!v) return bad("'" + jtpr[i] + "' is not an integer");
      n[i] = *v;
    }
    RawTable t;
    t.emplace_back("name", RawValue{parts[0], ref});
    t.emplace_back("rank", RawValue{n[0], ref});
    t.emplace_back("dtz", RawValue{n[1], ref});
    t.emplace_back("parity", RawValue{n[2], ref});
    t.emplace_back("particle_rank", RawValue{n[3], ref});
    t.emplace_back("file_2b", RawValue{parts[2], ref});
    if (parts.size() == 4) t.emplace_back("file_3b", RawValue{parts[3], ref});
    if (op_fmt_)
      t.emplace_back("format", RawValue{*op_fmt_, ref_of("input_op_fmt")});
    return RawValue{std::move(t), ref};
  }

  // ------------------------------------------------------------------
  // cross-key rules
  void Drop(std::string_view path) { out_.tree.erase(path); }

  /// L2: IMSRG(3)-only keys are dropped (with a warning) when IMSRG(3) is not
  /// active.
  void DropInertImsrg3Keys() {
    const bool imsrg3_active =
        tree_true("imsrg3.in_flow") || tree_true("imsrg3.at_end");
    if (!imsrg3_active) {
      const std::array<std::pair<std::string_view, std::string_view>, 5> keys =
          {{
              {"imsrg3_n7", "imsrg3.variant"},
              {"imsrg3_mp4", "imsrg3.variant"},
              {"imsrg3_no_qqq", "imsrg3.no_qqq"},
              {"only_2b_eta", "imsrg3.only_2b_eta"},
              {"only_2b_omega", "imsrg3.only_2b_omega"},
          }};
      for (const auto& [key, path] : keys) {
        if (!given(key) || !in_tree(path)) continue;
        Drop(path);
        Warn("W-L2", std::string(key),
             "'" + std::string(key) +
                 "' has no effect without IMSRG3 or imsrg3_at_end; ignored",
             ref_of(key));
      }
      if (const auto tv = tree_double("imsrg3.threebody_threshold");
          tv && *tv != 0.0) {
        Drop("imsrg3.threebody_threshold");
        Warn("W-L2", "threebody_threshold",
             "threebody_threshold has no effect without IMSRG3 or "
             "imsrg3_at_end; ignored",
             ref_of("threebody_threshold"));
      }
      if (!tree_true("imsrg.perturbative_triples")) {
        for (const auto& [key, path] :
             {std::pair{std::string_view("dE3max"),
                        std::string_view("imsrg3.de3max")},
              std::pair{std::string_view("OccNat3Cut"),
                        std::string_view("imsrg3.occ_nat3_cut")}}) {
          if (!given(key) || !in_tree(path)) continue;
          Drop(path);
          Warn("W-L2", std::string(key),
               "'" + std::string(key) +
                   "' has no effect without IMSRG3, imsrg3_at_end or "
                   "perturbative_triples; "
                   "ignored",
               ref_of(key));
        }
      }
    }
  }

  /// V-21: natural-orbital keys are dropped (with a warning) unless basis=NAT.
  void DropInertNaturalOrbitalKeys() {
    if (tree_str("basis.kind").value_or("hf") != "nat") {
      for (std::string_view key : {std::string_view("use_NAT_occupations"),
                                   std::string_view("NAT_order"),
                                   std::string_view("use_HF_reference_in_NAT"),
                                   std::string_view("order_NAT_by_energy")}) {
        if (!given(key)) continue;
        Warn("V-21", std::string(key),
             "'" + std::string(key) + "' is ignored unless basis=NAT",
             ref_of(key));
      }
      Drop("natural_orbitals.use_occupations");
      Drop("natural_orbitals.order");
      Drop("natural_orbitals.use_hf_reference");
    }
  }

  void DoCrossRules() {
    const bool has_valence =
        in_tree("valence.space") || in_tree("valence.custom");
    const std::string method = tree_str("imsrg.method").value_or("magnus");
    const auto steps = tree_int("imsrg.steps");

    // Q1.
    if (steps && *steps == 1 && has_valence && given("valence_generator")) {
      Drop("imsrg.generator.valence");
      if (!core_from_valence_)
        Warn("W-NSTEPS1", "valence_generator",
             "nsteps=1 uses core_generator only; valence_generator is ignored "
             "(design/16 Q1)",
             ref_of("valence_generator"));
    }
    // Q9.
    if (steps && *steps == 2 && !has_valence) {
      Drop("imsrg.steps");
      Warn("W-Q9", "nsteps",
           "nsteps=2 without a valence space: the second stage is skipped",
           ref_of("nsteps"));
    }
    // Q3.
    if (const auto dv = tree_double("imsrg.denominator.delta");
        dv && *dv != 0.0 && !in_tree("imsrg.denominator.delta_orbit")) {
      Drop("imsrg.denominator.delta");
      Warn("W-Q3", "denominator_delta",
           "denominator_delta is ignored without denominator_delta_orbit",
           ref_of("denominator_delta"));
    }
    // Q10.
    if (tree_true("imsrg.perturbative_triples") && method != "magnus") {
      Drop("imsrg.perturbative_triples");
      Warn("W-Q10", "perturbative_triples",
           "perturbative_triples runs only with method=magnus; ignored",
           ref_of("perturbative_triples"));
    }
    // V-52.
    if (tree_true("imsrg3.at_end") && method != "magnus") {
      Drop("imsrg3.at_end");
      Warn("V-52", "imsrg3_at_end",
           "imsrg3_at_end requires a magnus method; ignored",
           ref_of("imsrg3_at_end"));
    }
    DropInertImsrg3Keys();
    // Q2.
    if (given("ds_0") && method == "magnus") {
      const auto fast = [](const std::optional<Gen>& g) {
        return g && (g->canon.find("imaginary") != std::string::npos ||
                     g->canon.find("wegner") != std::string::npos);
      };
      if (!fast(core_gen_) && !fast(val_gen_))
        Warn("W-Q2", "ds_0",
             "ds_0 has no effect with method=magnus (design/16 Q2)",
             ref_of("ds_0"));
    }
    DropInertNaturalOrbitalKeys();
  }
};

std::string KindText(LegacyKind k) {
  switch (k) {
    case LegacyKind::kString:
      return "string";
    case LegacyKind::kDouble:
      return "double";
    case LegacyKind::kInt:
      return "int";
    case LegacyKind::kVector:
      return "vector";
  }
  return "?";
}

std::string StatusText(LegacyStatus s) {
  switch (s) {
    case LegacyStatus::kSupported:
      return "S";
    case LegacyStatus::kChanged:
      return "C";
    case LegacyStatus::kNotSupported:
      return "NS";
    case LegacyStatus::kPlanned:
      return "PL";
  }
  return "?";
}

std::string Pad(std::string s, std::size_t w) {
  if (s.size() < w) s.append(w - s.size(), ' ');
  return s;
}

}  // namespace

std::span<const LegacyKeyInfo> LegacyKeys() noexcept { return kKeys; }

const LegacyKeyInfo* FindLegacyKey(std::string_view key) noexcept {
  for (const LegacyKeyInfo& k : kKeys)
    if (k.key == key) return &k;
  return nullptr;
}

LegacyTranslation TranslateLegacy(std::span<const std::string> args) {
  return Translator{}.Run(args);
}

std::string LegacyKeysText() {
  std::size_t wk = 3, wd = 7, wn = 7;
  for (const LegacyKeyInfo& k : kKeys) {
    wk = std::max(wk, k.key.size());
    wd = std::max(wd, std::max<std::size_t>(k.old_default.size(), 2));
    wn = std::max(wn, k.new_key.empty() ? std::size_t{1} : k.new_key.size());
  }
  std::string out = Pad("key", wk) + "  " + Pad("kind", 6) + "  " +
                    Pad("default", wd) + "  " + Pad("st", 2) + "  " +
                    Pad("new key", wn) + "  note\n";
  for (const LegacyKeyInfo& k : kKeys) {
    out +=
        Pad(std::string(k.key), wk) + "  " + Pad(KindText(k.kind), 6) + "  " +
        Pad(k.old_default.empty() ? "\"\"" : std::string(k.old_default), wd) +
        "  " + Pad(StatusText(k.status), 2) + "  " +
        Pad(k.new_key.empty() ? "-" : std::string(k.new_key), wn) + "  " +
        std::string(k.note) + "\n";
  }
  return out;
}

}  // namespace imsrg::driver
