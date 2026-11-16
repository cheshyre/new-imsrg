// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Resolution of a raw tree into a RunConfig: typed reading through the schema
// visitor, defaults, derived values (design/16 sections 1 and 6.1), then
// validation (validate.cc).
#include "imsrg/driver/config/resolve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "imsrg/basis/nucleus.h"
#include "imsrg/basis/valence_space.h"
#include "imsrg/driver/config/schema_keys.h"

namespace imsrg::driver {

namespace {

template <class T>
struct IsOptional : std::false_type {};
template <class T>
struct IsOptional<std::optional<T>> : std::true_type {};

/// Typed reader: the visitor of `VisitSchema` that fills a RunConfig from a
/// raw tree.
class Reader {
 public:
  Reader(const RawTree& tree, RunConfig& cfg, ConfigIssues& issues)
      : tree_(tree), cfg_(cfg), issues_(issues) {}

  /// Reads one schema field. Absent keys keep their default (or raise V-03 when
  /// required).
  template <class F>
  void Field(const KeyInfo& key, F& f) {
    const std::string path(key.path);
    const RawValue* v = tree_.find(path);
    if (v == nullptr) {
      if (key.required) {
        issues_.push_back(
            MakeError("V-03", path, "required key '" + path + "' is missing"));
      }
      return;
    }
    const bool ok = Read(path, *v, f, key.string_or_array);
    if constexpr (std::is_same_v<F, std::vector<std::string>>) {
      if (ok && path == "interaction.two_body.file" &&
          (f.empty() || f.size() > 2)) {
        issues_.push_back(MakeError(
            "V-02", path,
            path + " must have 1 or 2 entries, got " + std::to_string(f.size()),
            v->where));
      }
    }
  }

  /// Reads an optional table: the section exists iff the table is present in
  /// the tree.
  template <class S, class Body>
  void OptionalTable(std::string_view table, std::optional<S>& section,
                     Body&& body) {
    if (!tree_.has_table(table)) return;
    section.emplace();
    body(*section);
  }

 private:
  void TypeError(const std::string& path, const RawValue& v,
                 std::string_view expected) {
    issues_.push_back(MakeError("E-TYPE", path,
                                "expected " + std::string(expected) + ", got " +
                                    std::string(v.type_name()),
                                v.where));
  }

  template <class F>
  bool Read(const std::string& path, const RawValue& v, F& f,
            bool string_or_array = false) {
    if constexpr (std::is_same_v<F, bool>) {
      if (const bool* b = v.get_if<bool>()) {
        f = *b;
        return true;
      }
      TypeError(path, v, "a boolean");
      return false;
    } else if constexpr (std::is_same_v<F, int>) {
      return ReadInt(path, v, f);
    } else if constexpr (std::is_same_v<F, double>) {
      return ReadDouble(path, v, f);
    } else if constexpr (std::is_same_v<F, std::string>) {
      if (const auto* s = v.get_if<std::string>()) {
        f = *s;
        return true;
      }
      TypeError(path, v, "a string");
      return false;
    } else if constexpr (IsOptional<F>::value) {
      typename F::value_type tmp{};
      if (Read(path, v, tmp, string_or_array)) {
        f = std::move(tmp);
        return true;
      }
      return false;
    } else if constexpr (detail::ConfigEnum<F>) {
      return ReadEnum(path, v, f);
    } else if constexpr (std::is_same_v<F, std::vector<std::string>>) {
      return ReadStringList(path, v, f, string_or_array);
    } else if constexpr (std::is_same_v<F, std::vector<OperatorRequest>>) {
      return ReadOperatorList(path, v, f);
    } else if constexpr (std::is_same_v<F, std::vector<OperatorFromFile>>) {
      return ReadFromFile(path, v, f);
    } else {
      static_assert(!sizeof(F), "unsupported schema field type");
    }
  }

  /// Reads an int field; V-02 if the TOML integer does not fit.
  bool ReadInt(const std::string& path, const RawValue& v, int& f) {
    if (const auto* i = v.get_if<std::int64_t>()) {
      if (*i < std::numeric_limits<int>::min() ||
          *i > std::numeric_limits<int>::max()) {
        issues_.push_back(MakeError(
            "V-02", path,
            path + " = " + std::to_string(*i) + " does not fit into an int",
            v.where));
        return false;
      }
      f = static_cast<int>(*i);
      return true;
    }
    TypeError(path, v, "an integer");
    return false;
  }

  /// Reads a double field (a TOML integer is accepted); V-02 if not finite.
  bool ReadDouble(const std::string& path, const RawValue& v, double& f) {
    if (const double* d = v.get_if<double>()) {
      if (!std::isfinite(
              *d)) {  // TOML allows inf and nan; no parameter means them
        issues_.push_back(MakeError(
            "V-02", path, path + " must be a finite number", v.where));
        return false;
      }
      f = *d;
      return true;
    }
    if (const auto* i = v.get_if<std::int64_t>()) {
      f = static_cast<double>(*i);
      return true;
    }
    TypeError(path, v, "a number");
    return false;
  }

  /// Reads an enum field from its string name; V-01 lists the allowed names.
  template <class F>
  bool ReadEnum(const std::string& path, const RawValue& v, F& f) {
    const auto* s = v.get_if<std::string>();
    if (s == nullptr) {
      TypeError(path, v, "a string");
      return false;
    }
    if (auto e = EnumFromString<F>(*s)) {
      f = *e;
      return true;
    }
    const auto names = EnumNames<F>();
    std::string allowed;
    for (const auto n : names) {
      if (!allowed.empty()) allowed += ", ";
      allowed += n;
    }
    issues_.push_back(MakeError(
        "V-01", path,
        "unknown " + std::string(EnumTraits<F>::type_name) + " '" + *s +
            "'; allowed: " + allowed,
        v.where, DidYouMean(*s, std::span<const std::string_view>(names))));
    return false;
  }

  /// Reads a list of strings (a single string if `string_or_array`).
  bool ReadStringList(const std::string& path, const RawValue& v,
                      std::vector<std::string>& f, bool string_or_array) {
    if (const auto* s = v.get_if<std::string>();
        s != nullptr && string_or_array) {
      f = {*s};
      return true;
    }
    const auto* a = v.get_if<RawArray>();
    if (a == nullptr) {
      TypeError(path, v,
                string_or_array ? "a string or an array of strings"
                                : "an array of strings");
      return false;
    }
    std::vector<std::string> out;
    bool ok = true;
    for (std::size_t i = 0; i < a->size(); ++i) {
      const auto* s = (*a)[i].get_if<std::string>();
      if (s == nullptr) {
        TypeError(path + "[" + std::to_string(i) + "]", (*a)[i], "a string");
        ok = false;
      } else {
        out.push_back(*s);
      }
    }
    if (ok) f = std::move(out);
    return ok;
  }

  /// Reads the `operators.list` entries through `ParseOperatorList`.
  bool ReadOperatorList(const std::string& path, const RawValue& v,
                        std::vector<OperatorRequest>& f) {
    const auto* a = v.get_if<RawArray>();
    if (a == nullptr) {
      TypeError(path, v, "an array");
      return false;
    }
    auto parsed = ParseOperatorList(std::span<const RawValue>(*a),
                                    ModelSpaceInfo{cfg_.basis.emax}, path);
    if (parsed.has_value()) {
      f = std::move(parsed.value());
      return true;
    }
    for (auto& issue : parsed.error()) issues_.push_back(std::move(issue));
    return false;
  }

  /// Reads one `operators.from_file` entry at `base` into `e`; returns false
  /// if any key is wrong or required keys are missing.
  bool ReadFromFileEntry(const std::string& base, const RawValue& entry,
                         const RawTable& tbl, OperatorFromFile& e) {
    static const auto* const kKeys = new std::vector<std::string>{
        "name",          "rank",    "dtz",     "parity",
        "particle_rank", "file_2b", "file_3b", "format"};
    bool ok = true;
    bool have_name = false;
    bool have_file = false;
    for (const auto& [key, val] : tbl) {
      const std::string ep = base + "." + key;
      bool r = true;
      if (key == "name") {
        r = Read(ep, val, e.name);
        have_name = true;
      } else if (key == "rank") {
        r = Read(ep, val, e.rank);
      } else if (key == "dtz") {
        r = Read(ep, val, e.dtz);
      } else if (key == "parity") {
        r = Read(ep, val, e.parity);
      } else if (key == "particle_rank") {
        r = Read(ep, val, e.particle_rank);
      } else if (key == "file_2b") {
        r = Read(ep, val, e.file_2b);
        have_file = true;
      } else if (key == "file_3b") {
        r = Read(ep, val, e.file_3b);
      } else if (key == "format") {
        r = Read(ep, val, e.format);
      } else {
        issues_.push_back(MakeError("E-KEY-UNKNOWN", ep,
                                    "unknown key '" + key + "' in " + base,
                                    val.where, DidYouMean(key, *kKeys)));
        r = false;
      }
      ok = ok && r;
    }
    if (!have_name) {
      issues_.push_back(MakeError("V-03", base + ".name",
                                  "required key '" + base + ".name' is missing",
                                  entry.where));
      ok = false;
    }
    if (!have_file) {
      issues_.push_back(MakeError(
          "V-03", base + ".file_2b",
          "required key '" + base + ".file_2b' is missing", entry.where));
      ok = false;
    }
    return ok;
  }

  bool ReadFromFile(const std::string& path, const RawValue& v,
                    std::vector<OperatorFromFile>& out) {
    const auto* a = v.get_if<RawArray>();
    if (a == nullptr) {
      TypeError(path, v, "an array of tables");
      return false;
    }
    std::vector<OperatorFromFile> result;
    bool ok = true;
    for (std::size_t i = 0; i < a->size(); ++i) {
      const std::string base = path + "[" + std::to_string(i) + "]";
      const auto* tbl = (*a)[i].get_if<RawTable>();
      if (tbl == nullptr) {
        TypeError(base, (*a)[i], "a table");
        ok = false;
        continue;
      }
      OperatorFromFile e;
      ok = ReadFromFileEntry(base, (*a)[i], *tbl, e) && ok;
      result.push_back(std::move(e));
    }
    if (ok) out = std::move(result);
    return ok;
  }

  const RawTree& tree_;
  RunConfig& cfg_;
  ConfigIssues& issues_;
};

std::string FormatG(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", x);
  return buf;
}

/// Replaces characters that do not belong into a file name.
std::string Sanitize(const std::string& s) {
  std::string out = s;
  for (char& c : out) {
    const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z') || c == '.' || c == '_' ||
                    c == '+' || c == '-';
    if (!ok) c = '_';
  }
  return out;
}

/// Derives the mass number from the reference nucleus or the valence core
/// (V-23 if neither fixes it).
void DeriveMassNumber(const RawTree& tree, RunConfig& cfg, ConfigIssues& issues,
                      const std::optional<basis::Nucleus>& reference,
                      bool reference_bad,
                      const std::optional<basis::Nucleus>& core) {
  if (!tree.contains("basis.mass_number")) {
    const std::optional<basis::Nucleus>& src = reference ? reference : core;
    if (src && src->A == std::floor(src->A) && src->A >= 1.0) {
      cfg.basis.mass_number = static_cast<int>(src->A);
    } else if (!reference_bad) {
      issues.push_back(MakeError(
          "V-23", "basis.mass_number",
          "cannot determine the mass number; set basis.mass_number or "
          "reference.nucleus",
          SourceRef{SourceRef::Kind::kDerived, "basis.mass_number", 0, 0}));
    }
  }
}

/// Derives the output stem and the flow file from the run's identity.
void DeriveOutputNames(const RawTree& tree, RunConfig& cfg,
                       const std::optional<basis::Nucleus>& core) {
  if (!tree.contains("output.stem")) {
    const std::string method = cfg.run.task == Task::kImsrg
                                   ? std::string(ToString(cfg.imsrg.method))
                                   : std::string(ToString(cfg.run.task));
    std::string ref = "none";
    if (cfg.reference.nucleus) {
      ref = *cfg.reference.nucleus;
    } else if (core) {
      ref = core->name();
    }
    std::string val = ref;
    if (cfg.valence.space) {
      val = *cfg.valence.space;
    } else if (cfg.valence.custom) {
      val = *cfg.valence.custom;
    }
    cfg.output.stem =
        Sanitize(method + "_" + ref + "_" + val + "_hw" +
                 FormatG(cfg.basis.hw) + "_e" + std::to_string(cfg.basis.emax) +
                 "_A" + std::to_string(cfg.basis.mass_number));
  }
  if (!tree.contains("output.flow_file")) {
    cfg.output.flow_file = (std::filesystem::path(cfg.output.directory) /
                            (cfg.output.stem + ".flow"))
                               .generic_string();
  }
}

/// Derived values (design/16 section 2). Never throws and never assumes reading
/// succeeded.
void Derive(const RawTree& tree, const ResolveOptions& options, RunConfig& cfg,
            ConfigIssues& issues) {
  // Effective e3max.
  if (cfg.basis.emax >= 0)
    cfg.basis.e3max = std::min(cfg.basis.e3max, 3 * cfg.basis.emax);

  // Two-body file cuts.
  auto& tb = cfg.interaction.two_body;
  if (!tree.contains("interaction.two_body.e2max")) tb.e2max = 2 * tb.e1max;
  if (!tree.contains("interaction.two_body.lmax")) tb.lmax = tb.e1max;

  // Reference nucleus and valence core.
  std::optional<basis::Nucleus> reference;
  bool reference_bad = false;
  if (cfg.reference.nucleus) {
    try {
      reference = basis::Nucleus::Parse(*cfg.reference.nucleus);
    } catch (const std::exception&) {
      reference_bad = true;  // V-31 reports it
    }
  }
  std::optional<basis::Nucleus> core;
  if (cfg.valence.space) {
    try {
      core = basis::DefaultReferenceNucleus(*cfg.valence.space);
    } catch (const std::exception&) {
    }
  }

  DeriveMassNumber(tree, cfg, issues, reference, reference_bad, core);

  // D37: DIIS for TOML configs, imsrg++'s linear mixing for legacy command
  // lines (so that old command lines reproduce imsrg++ iteration counts).
  if (options.legacy && !tree.contains("hartree_fock.mixing"))
    cfg.hartree_fock.mixing = HartreeFockMixing::kLinear;

  if (!tree.contains("hamiltonian.beta_cm_hw"))
    cfg.hamiltonian.beta_cm_hw = cfg.basis.hw;

  if (!tree.contains("hamiltonian.kinetic")) {
    if (tb.format == TwoBodyFormat::kNushellx) {
      cfg.hamiltonian.kinetic = KineticEnergy::kNone;
    } else if (cfg.hamiltonian.trap_hw) {
      cfg.hamiltonian.kinetic = KineticEnergy::kLab;
    } else {
      cfg.hamiltonian.kinetic = KineticEnergy::kIntrinsic;
    }
  }

  // IMSRG(3f2) recipe (owner decision D32): factorized = true implies
  // hunter-gatherer and omega_norm_max = 0.1 unless given explicitly.
  if (cfg.imsrg3.factorized) {
    if (!tree.contains("imsrg.omega.hunter_gatherer"))
      cfg.imsrg.omega.hunter_gatherer = true;
    if (!tree.contains("imsrg.step.omega_norm_max"))
      cfg.imsrg.step.omega_norm_max = 0.1;
  }

  if (!tree.contains("imsrg.steps"))
    cfg.imsrg.steps = cfg.valence.has_valence_space() ? 2 : 1;

  if (auto& sp = cfg.imsrg.space) {
    if (!tree.contains("imsrg.space.emax")) sp->emax = cfg.basis.emax;
    if (!tree.contains("imsrg.space.e2max")) sp->e2max = 2 * sp->emax;
    if (!tree.contains("imsrg.space.e3max"))
      sp->e3max = std::min(cfg.basis.e3max, 3 * sp->emax);
    if (!tree.contains("imsrg.space.emax_3body")) sp->emax_3body = sp->emax;
  }

  DeriveOutputNames(tree, cfg, core);

  // The stage-2 generator is inert for one stage: report the default.
  if (cfg.imsrg.steps == 1)
    cfg.imsrg.generator.valence = GeneratorConfig{}.valence;
}

}  // namespace

ResolveResult Resolve(const RawTree& tree, const ResolveOptions& options) {
  ResolveResult result;
  Reader reader(tree, result.config, result.issues);
  VisitSchema(result.config, reader);
  Derive(tree, options, result.config, result.issues);
  auto found = Validate(result.config, tree, options);
  for (auto& issue : found) result.issues.push_back(std::move(issue));
  return result;
}

}  // namespace imsrg::driver
