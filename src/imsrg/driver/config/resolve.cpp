// Resolution of a raw tree into a RunConfig: typed reading through the schema visitor, defaults,
// derived values (design/16 sections 1 and 6.1), then validation (validate.cpp).
#include "imsrg/driver/config/resolve.hpp"

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

#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/valence_space.hpp"
#include "imsrg/driver/config/schema_keys.hpp"

namespace imsrg::driver {

namespace {

template <class T>
struct is_optional : std::false_type {};
template <class T>
struct is_optional<std::optional<T>> : std::true_type {};

/// Typed reader: the visitor of `visit_schema` that fills a RunConfig from a raw tree.
class Reader {
 public:
  Reader(const RawTree& tree, RunConfig& cfg, ConfigIssues& issues)
      : tree_(tree), cfg_(cfg), issues_(issues) {}

  /// Reads one schema field. Absent keys keep their default (or raise V-03 when required).
  template <class F>
  void field(const KeyInfo& key, F& f) {
    const std::string path(key.path);
    const RawValue* v = tree_.find(path);
    if (v == nullptr) {
      if (key.required) {
        issues_.push_back(make_error("V-03", path, "required key '" + path + "' is missing"));
      }
      return;
    }
    const bool ok = read(path, *v, f, key.string_or_array);
    if constexpr (std::is_same_v<F, std::vector<std::string>>) {
      if (ok && path == "interaction.two_body.file" && (f.empty() || f.size() > 2)) {
        issues_.push_back(make_error(
            "V-02", path, path + " must have 1 or 2 entries, got " + std::to_string(f.size()),
            v->where));
      }
    }
  }

  /// Reads an optional table: the section exists iff the table is present in the tree.
  template <class S, class Body>
  void optional_table(std::string_view table, std::optional<S>& section, Body&& body) {
    if (!tree_.has_table(table)) return;
    section.emplace();
    body(*section);
  }

 private:
  void type_error(const std::string& path, const RawValue& v, std::string_view expected) {
    issues_.push_back(make_error(
        "E-TYPE", path, "expected " + std::string(expected) + ", got " + std::string(v.type_name()),
        v.where));
  }

  template <class F>
  bool read(const std::string& path, const RawValue& v, F& f, bool string_or_array = false) {
    if constexpr (std::is_same_v<F, bool>) {
      if (const bool* b = v.get_if<bool>()) {
        f = *b;
        return true;
      }
      type_error(path, v, "a boolean");
      return false;
    } else if constexpr (std::is_same_v<F, int>) {
      if (const auto* i = v.get_if<std::int64_t>()) {
        if (*i < std::numeric_limits<int>::min() || *i > std::numeric_limits<int>::max()) {
          issues_.push_back(
              make_error("V-02", path,
                         path + " = " + std::to_string(*i) + " does not fit into an int", v.where));
          return false;
        }
        f = static_cast<int>(*i);
        return true;
      }
      type_error(path, v, "an integer");
      return false;
    } else if constexpr (std::is_same_v<F, double>) {
      if (const double* d = v.get_if<double>()) {
        if (!std::isfinite(*d)) {  // TOML allows inf and nan; no parameter means them
          issues_.push_back(make_error("V-02", path, path + " must be a finite number", v.where));
          return false;
        }
        f = *d;
        return true;
      }
      if (const auto* i = v.get_if<std::int64_t>()) {
        f = static_cast<double>(*i);
        return true;
      }
      type_error(path, v, "a number");
      return false;
    } else if constexpr (std::is_same_v<F, std::string>) {
      if (const auto* s = v.get_if<std::string>()) {
        f = *s;
        return true;
      }
      type_error(path, v, "a string");
      return false;
    } else if constexpr (is_optional<F>::value) {
      typename F::value_type tmp{};
      if (read(path, v, tmp, string_or_array)) {
        f = std::move(tmp);
        return true;
      }
      return false;
    } else if constexpr (ConfigEnum<F>) {
      const auto* s = v.get_if<std::string>();
      if (s == nullptr) {
        type_error(path, v, "a string");
        return false;
      }
      if (auto e = enum_from_string<F>(*s)) {
        f = *e;
        return true;
      }
      const auto names = enum_names<F>();
      std::string allowed;
      for (const auto n : names) {
        if (!allowed.empty()) allowed += ", ";
        allowed += n;
      }
      issues_.push_back(make_error(
          "V-01", path,
          "unknown " + std::string(EnumTraits<F>::type_name) + " '" + *s + "'; allowed: " + allowed,
          v.where, did_you_mean(*s, std::span<const std::string_view>(names))));
      return false;
    } else if constexpr (std::is_same_v<F, std::vector<std::string>>) {
      if (const auto* s = v.get_if<std::string>(); s != nullptr && string_or_array) {
        f = {*s};
        return true;
      }
      const auto* a = v.get_if<RawArray>();
      if (a == nullptr) {
        type_error(path, v,
                   string_or_array ? "a string or an array of strings" : "an array of strings");
        return false;
      }
      std::vector<std::string> out;
      bool ok = true;
      for (std::size_t i = 0; i < a->size(); ++i) {
        const auto* s = (*a)[i].get_if<std::string>();
        if (s == nullptr) {
          type_error(path + "[" + std::to_string(i) + "]", (*a)[i], "a string");
          ok = false;
        } else {
          out.push_back(*s);
        }
      }
      if (ok) f = std::move(out);
      return ok;
    } else if constexpr (std::is_same_v<F, std::vector<OperatorRequest>>) {
      const auto* a = v.get_if<RawArray>();
      if (a == nullptr) {
        type_error(path, v, "an array");
        return false;
      }
      auto parsed =
          parse_operator_list(std::span<const RawValue>(*a), ModelSpaceInfo{cfg_.basis.emax}, path);
      if (parsed.has_value()) {
        f = std::move(parsed.value());
        return true;
      }
      for (auto& issue : parsed.error()) issues_.push_back(std::move(issue));
      return false;
    } else if constexpr (std::is_same_v<F, std::vector<OperatorFromFile>>) {
      return read_from_file(path, v, f);
    } else {
      static_assert(!sizeof(F), "unsupported schema field type");
    }
  }

  bool read_from_file(const std::string& path, const RawValue& v,
                      std::vector<OperatorFromFile>& out) {
    const auto* a = v.get_if<RawArray>();
    if (a == nullptr) {
      type_error(path, v, "an array of tables");
      return false;
    }
    static const std::vector<std::string> keys = {"name",          "rank",    "dtz",     "parity",
                                                  "particle_rank", "file_2b", "file_3b", "format"};
    std::vector<OperatorFromFile> result;
    bool ok = true;
    for (std::size_t i = 0; i < a->size(); ++i) {
      const std::string base = path + "[" + std::to_string(i) + "]";
      const auto* tbl = (*a)[i].get_if<RawTable>();
      if (tbl == nullptr) {
        type_error(base, (*a)[i], "a table");
        ok = false;
        continue;
      }
      OperatorFromFile e;
      bool have_name = false;
      bool have_file = false;
      for (const auto& [key, val] : *tbl) {
        const std::string ep = base + "." + key;
        bool r = true;
        if (key == "name") {
          r = read(ep, val, e.name);
          have_name = true;
        } else if (key == "rank") {
          r = read(ep, val, e.rank);
        } else if (key == "dtz") {
          r = read(ep, val, e.dtz);
        } else if (key == "parity") {
          r = read(ep, val, e.parity);
        } else if (key == "particle_rank") {
          r = read(ep, val, e.particle_rank);
        } else if (key == "file_2b") {
          r = read(ep, val, e.file_2b);
          have_file = true;
        } else if (key == "file_3b") {
          r = read(ep, val, e.file_3b);
        } else if (key == "format") {
          r = read(ep, val, e.format);
        } else {
          issues_.push_back(make_error("E-KEY-UNKNOWN", ep, "unknown key '" + key + "' in " + base,
                                       val.where, did_you_mean(key, keys)));
          r = false;
        }
        ok = ok && r;
      }
      if (!have_name) {
        issues_.push_back(make_error("V-03", base + ".name",
                                     "required key '" + base + ".name' is missing", (*a)[i].where));
        ok = false;
      }
      if (!have_file) {
        issues_.push_back(make_error("V-03", base + ".file_2b",
                                     "required key '" + base + ".file_2b' is missing",
                                     (*a)[i].where));
        ok = false;
      }
      result.push_back(std::move(e));
    }
    if (ok) out = std::move(result);
    return ok;
  }

  const RawTree& tree_;
  RunConfig& cfg_;
  ConfigIssues& issues_;
};

std::string format_g(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", x);
  return buf;
}

/// Replaces characters that do not belong into a file name.
std::string sanitize(const std::string& s) {
  std::string out = s;
  for (char& c : out) {
    const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    c == '.' || c == '_' || c == '+' || c == '-';
    if (!ok) c = '_';
  }
  return out;
}

/// Derived values (design/16 section 2). Never throws and never assumes reading succeeded.
void derive(const RawTree& tree, RunConfig& cfg, ConfigIssues& issues) {
  // Effective e3max.
  if (cfg.basis.emax >= 0) cfg.basis.e3max = std::min(cfg.basis.e3max, 3 * cfg.basis.emax);

  // Two-body file cuts.
  auto& tb = cfg.interaction.two_body;
  if (!tree.contains("interaction.two_body.e2max")) tb.e2max = 2 * tb.e1max;
  if (!tree.contains("interaction.two_body.lmax")) tb.lmax = tb.e1max;

  // Reference nucleus and valence core.
  std::optional<basis::Nucleus> reference;
  bool reference_bad = false;
  if (cfg.reference.nucleus) {
    try {
      reference = basis::Nucleus::parse(*cfg.reference.nucleus);
    } catch (const std::exception&) {
      reference_bad = true;  // V-31 reports it
    }
  }
  std::optional<basis::Nucleus> core;
  if (cfg.valence.space) {
    try {
      core = basis::default_reference_nucleus(*cfg.valence.space);
    } catch (const std::exception&) {
    }
  }

  // Mass number.
  if (!tree.contains("basis.mass_number")) {
    const std::optional<basis::Nucleus>& src = reference ? reference : core;
    if (src && src->A == std::floor(src->A) && src->A >= 1.0) {
      cfg.basis.mass_number = static_cast<int>(src->A);
    } else if (!reference_bad) {
      issues.push_back(
          make_error("V-23", "basis.mass_number",
                     "cannot determine the mass number; set basis.mass_number or reference.nucleus",
                     SourceRef{SourceRef::Kind::Derived, "basis.mass_number", 0, 0}));
    }
  }

  if (!tree.contains("hamiltonian.beta_cm_hw")) cfg.hamiltonian.beta_cm_hw = cfg.basis.hw;

  if (!tree.contains("hamiltonian.kinetic")) {
    if (tb.format == TwoBodyFormat::Nushellx) {
      cfg.hamiltonian.kinetic = KineticEnergy::None;
    } else if (cfg.hamiltonian.trap_hw) {
      cfg.hamiltonian.kinetic = KineticEnergy::Lab;
    } else {
      cfg.hamiltonian.kinetic = KineticEnergy::Intrinsic;
    }
  }

  if (!tree.contains("imsrg.steps")) cfg.imsrg.steps = cfg.valence.has_valence_space() ? 2 : 1;

  if (auto& sp = cfg.imsrg.space) {
    if (!tree.contains("imsrg.space.emax")) sp->emax = cfg.basis.emax;
    if (!tree.contains("imsrg.space.e2max")) sp->e2max = 2 * sp->emax;
    if (!tree.contains("imsrg.space.e3max")) sp->e3max = std::min(cfg.basis.e3max, 3 * sp->emax);
    if (!tree.contains("imsrg.space.emax_3body")) sp->emax_3body = sp->emax;
  }

  // Output stem and flow file.
  if (!tree.contains("output.stem")) {
    const std::string method = cfg.run.task == Task::Imsrg
                                   ? std::string(to_string(cfg.imsrg.method))
                                   : std::string(to_string(cfg.run.task));
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
        sanitize(method + "_" + ref + "_" + val + "_hw" + format_g(cfg.basis.hw) + "_e" +
                 std::to_string(cfg.basis.emax) + "_A" + std::to_string(cfg.basis.mass_number));
  }
  if (!tree.contains("output.flow_file")) {
    cfg.output.flow_file =
        (std::filesystem::path(cfg.output.directory) / (cfg.output.stem + ".flow"))
            .generic_string();
  }

  // The stage-2 generator is inert for one stage: report the default.
  if (cfg.imsrg.steps == 1) cfg.imsrg.generator.valence = GeneratorConfig{}.valence;
}

}  // namespace

ResolveResult resolve(const RawTree& tree, const ResolveOptions& options) {
  ResolveResult result;
  Reader reader(tree, result.config, result.issues);
  visit_schema(result.config, reader);
  derive(tree, result.config, result.issues);
  auto found = validate(result.config, tree, options);
  for (auto& issue : found) result.issues.push_back(std::move(issue));
  return result;
}

}  // namespace imsrg::driver
