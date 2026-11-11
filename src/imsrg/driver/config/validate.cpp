// Validation rules V-xx of design/16 section 5 on a resolved RunConfig.
#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <unistd.h>

#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/valence_space.hpp"
#include "imsrg/driver/config/resolve.hpp"
#include "imsrg/driver/config/schema_keys.hpp"

namespace imsrg::driver {

namespace {

namespace fs = std::filesystem;

/// Collects rule violations; knows the tree (for source locations) and the strictness mode.
class Checker {
 public:
  Checker(const RunConfig& cfg, const RawTree& tree, const ResolveOptions& opt)
      : cfg_(cfg), tree_(tree), opt_(opt) {}

  ConfigIssues take() { return std::move(issues_); }

  SourceRef where(std::string_view path) const {
    if (const RawValue* v = tree_.find(path)) return v->where;
    return {};
  }

  void error(std::string code, std::string path, std::string msg,
             std::optional<std::string> suggestion = std::nullopt) {
    auto w = where(path);
    issues_.push_back(
        make_error(std::move(code), std::move(path), std::move(msg), w, std::move(suggestion)));
  }

  void warning(std::string code, std::string path, std::string msg) {
    auto w = where(path);
    issues_.push_back(make_warning(std::move(code), std::move(path), std::move(msg), w));
  }

  /// "TOML error / legacy warning": same code, severity depends on the mode.
  void strict(std::string code, std::string path, std::string msg) {
    if (opt_.legacy) {
      warning(std::move(code), std::move(path), std::move(msg));
    } else {
      error(std::move(code), std::move(path), std::move(msg));
    }
  }

  void run();

 private:
  void ranges();
  void planned();
  void valence();
  void cuts();
  void files();
  void hamiltonian_basis();
  void imsrg_rules();
  void imsrg3_rules();
  void operators();
  void output();

  void need_pos(std::string_view path, double v) {
    if (!(v > 0)) {
      error("V-02", std::string(path), std::string(path) + " must be > 0, got " + format(v));
    }
  }
  void need_ge(std::string_view path, double v, double lo) {
    if (!(v >= lo)) {
      error("V-02", std::string(path),
            std::string(path) + " must be >= " + format(lo) + ", got " + format(v));
    }
  }
  static std::string format(double v) {
    std::string s = std::to_string(v);
    if (v == static_cast<double>(static_cast<long long>(v))) {
      return std::to_string(static_cast<long long>(v));
    }
    while (s.size() > 1 && s.back() == '0') s.pop_back();
    return s;
  }

  bool in_tree(std::string_view p) const { return tree_.contains(p); }
  int effective_e2max() const { return cfg_.interaction.two_body.e2max; }

  void check_readable(const std::string& path, const std::string& file);
  void check_writable_dir(const std::string& path, const std::string& dir);

  const RunConfig& cfg_;
  const RawTree& tree_;
  const ResolveOptions& opt_;
  ConfigIssues issues_;
};

/// Visitor over a copy of the config that reports enum fields with Phase::Planned.
class PlannedEnumVisitor {
 public:
  explicit PlannedEnumVisitor(Checker& c) : c_(c) {}

  template <class F>
  void field(const KeyInfo& key, F& f) {
    if constexpr (ConfigEnum<F>) {
      if (enum_phase(f) == Phase::Planned) {
        c_.error("E-PLANNED", std::string(key.path),
                 "'" + std::string(to_string(f)) + "' is not available in this build");
      }
    }
  }

  template <class S, class Body>
  void optional_table(std::string_view, std::optional<S>& section, Body&& body) {
    if (section) body(*section);
  }

 private:
  Checker& c_;
};

void Checker::run() {
  ranges();
  planned();
  valence();
  cuts();
  hamiltonian_basis();
  imsrg_rules();
  imsrg3_rules();
  operators();
  output();
  if (opt_.check_files) files();
}

void Checker::ranges() {
  const auto& b = cfg_.basis;
  need_ge("basis.emax", b.emax, 0);
  need_pos("basis.hw", b.hw);
  if (const RawValue* v = tree_.find("basis.e3max")) {
    if (const auto* i = v->get_if<std::int64_t>())
      need_ge("basis.e3max", static_cast<double>(*i), 0);
  }
  if (b.lmax) need_ge("basis.lmax", *b.lmax, 0);
  if (b.lmax3) need_ge("basis.lmax3", *b.lmax3, 1);
  if (b.emax_unocc) need_ge("basis.emax_unocc", *b.emax_unocc, 1);
  if (in_tree("basis.mass_number")) need_ge("basis.mass_number", b.mass_number, 1);
  if (cfg_.hamiltonian.trap_hw) need_pos("hamiltonian.trap_hw", *cfg_.hamiltonian.trap_hw);
  need_pos("hamiltonian.beta_cm_hw", cfg_.hamiltonian.beta_cm_hw);

  const auto& t2 = cfg_.interaction.two_body;
  need_ge("interaction.two_body.e1max", t2.e1max, 0);
  need_ge("interaction.two_body.e2max", t2.e2max, 0);
  need_ge("interaction.two_body.lmax", t2.lmax, 0);
  if (const auto& t3 = cfg_.interaction.three_body) {
    need_ge("interaction.three_body.e1max", t3->e1max, 0);
    need_ge("interaction.three_body.e3max", t3->e3max, 0);
    if (t3->e2max) need_ge("interaction.three_body.e2max", *t3->e2max, 0);
  }

  const auto& im = cfg_.imsrg;
  if (im.steps != 1 && im.steps != 2) {
    error("V-02", "imsrg.steps", "imsrg.steps must be 1 or 2, got " + std::to_string(im.steps));
  }
  need_ge("imsrg.step.smax", im.step.smax, 0);
  need_pos("imsrg.step.dsmax", im.step.dsmax);
  need_pos("imsrg.step.ds_0", im.step.ds_0);
  need_pos("imsrg.step.domega", im.step.domega);
  need_pos("imsrg.step.omega_norm_max", im.step.omega_norm_max);
  need_pos("imsrg.step.eta_criterion", im.step.eta_criterion);
  need_pos("imsrg.step.ode_tolerance", im.step.ode_tolerance);
  need_ge("imsrg.generator.qtransfer_n", im.generator.qtransfer_n, 1);
  if (im.space) {
    need_ge("imsrg.space.emax", im.space->emax, 0);
    need_ge("imsrg.space.e2max", im.space->e2max, 0);
    need_ge("imsrg.space.e3max", im.space->e3max, 0);
    need_ge("imsrg.space.emax_3body", im.space->emax_3body, 0);
  }
  const auto& i3 = cfg_.imsrg3;
  need_ge("imsrg3.threebody_threshold", i3.threebody_threshold, 0);
  if (i3.occ_nat3_cut) need_pos("imsrg3.occ_nat3_cut", *i3.occ_nat3_cut);
}

void Checker::planned() {
  RunConfig copy = cfg_;
  PlannedEnumVisitor v(*this);
  visit_schema(copy, v);

  const RunConfig def{};
  auto flag = [&](std::string_view path) {
    error("E-PLANNED", std::string(path),
          "'" + std::string(path) + "' is not available in this build");
  };
  if (cfg_.hamiltonian.relativistic_correction != def.hamiltonian.relativistic_correction) {
    flag("hamiltonian.relativistic_correction");
  }
  if (cfg_.interaction.two_body.lecs) flag("interaction.two_body.lecs");
  if (cfg_.interaction.three_body && cfg_.interaction.three_body->discard_no2b_after_hf) {
    flag("interaction.three_body.discard_no2b_after_hf");
  }
  if (cfg_.imsrg.goose_tank != def.imsrg.goose_tank) flag("imsrg.goose_tank");
  if (cfg_.imsrg3.no_qqq != def.imsrg3.no_qqq) flag("imsrg3.no_qqq");
  if (!cfg_.operators.first_order.empty()) flag("operators.first_order");
  if (cfg_.output.write_ho_ops != def.output.write_ho_ops) flag("output.write_ho_ops");
  if (cfg_.output.write_hf_ops != def.output.write_hf_ops) flag("output.write_hf_ops");
  if (cfg_.valence.space && cfg_.valence.space->find("FCI") != std::string::npos) {
    error("E-PLANNED", "valence.space",
          "'" + *cfg_.valence.space + "' is not available in this build");
  }
  for (std::size_t i = 0; i < cfg_.operators.from_file.size(); ++i) {
    const auto& e = cfg_.operators.from_file[i];
    if (e.rank > 0) {
      const std::string p = "operators.from_file[" + std::to_string(i) + "].rank";
      error("E-PLANNED", p,
            "'" + e.name + "' has rank " + std::to_string(e.rank) +
                "; rank > 0 operators are not available in this build");
    }
  }
}

void Checker::valence() {
  const auto& v = cfg_.valence;
  if (v.space) {
    const std::string& s = *v.space;
    const bool ok = basis::find_named_valence_space(s) != nullptr || s == "0hw-shell" ||
                    s == "jj-shell" || s.find("FCI") != std::string::npos ||
                    s.find(',') != std::string::npos;
    if (!ok) {
      std::vector<std::string> names;
      for (const auto& n : basis::named_valence_spaces()) names.emplace_back(n.name);
      auto hint = did_you_mean(s, names);
      std::string sug = hint ? *hint + "; " : std::string();
      sug += "for a single-reference run set reference.nucleus";
      error("V-01", "valence.space", "unknown valence space '" + s + "'", sug);
    }
  }
  if (v.space && v.custom) {
    strict("V-30", "valence.custom", "valence.space and valence.custom are both set");
  }
  if (cfg_.reference.nucleus) {
    try {
      (void)basis::Nucleus::parse(*cfg_.reference.nucleus);
    } catch (const std::exception& e) {
      error("V-31", "reference.nucleus",
            "cannot parse reference.nucleus '" + *cfg_.reference.nucleus + "': " + e.what());
    }
  }
  if (cfg_.reference.occupations_file && !cfg_.reference.nucleus && !v.has_valence_space()) {
    error("V-32", "reference.occupations_file",
          "reference.occupations_file requires reference.nucleus or a valence space");
  }
}

void Checker::cuts() {
  const auto& b = cfg_.basis;
  const auto& t2 = cfg_.interaction.two_body;
  if (b.emax > t2.e1max) {
    error("V-10", "basis.emax",
          "basis.emax = " + std::to_string(b.emax) +
              " exceeds interaction.two_body.e1max = " + std::to_string(t2.e1max));
  }
  if (effective_e2max() < 2 * b.emax) {
    error("V-11", "interaction.two_body.e2max",
          "interaction.two_body.e2max = " + std::to_string(effective_e2max()) +
              " is smaller than 2*basis.emax = " + std::to_string(2 * b.emax));
  }
  if (b.lmax && *b.lmax > t2.lmax) {
    error("V-12", "basis.lmax",
          "basis.lmax = " + std::to_string(*b.lmax) +
              " exceeds interaction.two_body.lmax = " + std::to_string(t2.lmax));
  }
  if (const auto& t3 = cfg_.interaction.three_body) {
    const int lim = std::min(b.emax, b.e3max);
    if (lim > t3->e1max) {
      error("V-13", "interaction.three_body.e1max",
            "min(basis.emax, basis.e3max) = " + std::to_string(lim) +
                " exceeds interaction.three_body.e1max = " + std::to_string(t3->e1max));
    }
    if (b.e3max > t3->e3max) {
      error("V-13", "interaction.three_body.e3max",
            "basis.e3max = " + std::to_string(b.e3max) +
                " exceeds interaction.three_body.e3max = " + std::to_string(t3->e3max));
    }
    if (t3->e2max && *t3->e2max < std::min(2 * b.emax, b.e3max)) {
      error("V-13", "interaction.three_body.e2max",
            "interaction.three_body.e2max = " + std::to_string(*t3->e2max) +
                " is smaller than min(2*basis.emax, basis.e3max) = " +
                std::to_string(std::min(2 * b.emax, b.e3max)));
    }
  }
}

void Checker::hamiltonian_basis() {
  const auto& h = cfg_.hamiltonian;
  if (h.trap_hw && h.kinetic != KineticEnergy::Lab) {
    strict("V-20", "hamiltonian.trap_hw",
           "hamiltonian.trap_hw requires hamiltonian.kinetic = \"lab\", got \"" +
               std::string(to_string(h.kinetic)) + "\"");
  }
  const NaturalOrbitalsConfig nat_def{};
  if (cfg_.basis.kind != BasisKind::Nat) {
    if (cfg_.natural_orbitals.use_occupations != nat_def.use_occupations) {
      strict("V-21", "natural_orbitals.use_occupations",
             "natural_orbitals.use_occupations requires basis.kind = \"nat\"");
    }
    if (cfg_.natural_orbitals.order != nat_def.order) {
      strict("V-21", "natural_orbitals.order",
             "natural_orbitals.order requires basis.kind = \"nat\"");
    }
  }
  if (cfg_.basis.kind == BasisKind::Oscillator && in_tree("hartree_fock.freeze_occupations") &&
      cfg_.hartree_fock.freeze_occupations) {
    warning("V-22", "hartree_fock.freeze_occupations",
            "hartree_fock.freeze_occupations has no effect with basis.kind = \"oscillator\"");
  }
}

void Checker::imsrg_rules() {
  const auto& im = cfg_.imsrg;
  const bool has_val = cfg_.valence.has_valence_space();
  if (im.steps == 2 && !has_val) {
    strict("V-40", "imsrg.steps", "imsrg.steps = 2 requires a valence space (valence.space)");
  }
  if (im.steps == 1 && has_val) {
    if (in_tree("imsrg.generator.valence")) {
      strict("V-41", "imsrg.generator.valence",
             "imsrg.generator.valence has no effect with imsrg.steps = 1");
    }
    if (!is_shell_model_generator(im.generator.core)) {
      warning("V-41", "imsrg.generator.core",
              "imsrg.steps = 1 with a valence space but a core generator '" +
                  std::string(to_string(im.generator.core)) +
                  "' that is not a shell-model generator");
    }
  }
  if (is_shell_model_generator(im.generator.core) && !has_val) {
    strict("V-42", "imsrg.generator.core",
           "imsrg.generator.core = \"" + std::string(to_string(im.generator.core)) +
               "\" is a shell-model generator and needs a valence space");
  }
  if (im.denominator.delta != 0 && !im.denominator.delta_orbit) {
    strict("V-43", "imsrg.denominator.delta_orbit",
           "imsrg.denominator.delta is nonzero but imsrg.denominator.delta_orbit is not set");
  }
  if (im.denominator.delta_orbit && *im.denominator.delta_orbit != "all") {
    const std::string& lab = *im.denominator.delta_orbit;
    try {
      const auto o = basis::parse_orbit_label(lab);
      if (o.e() > cfg_.basis.emax) {
        error("V-43", "imsrg.denominator.delta_orbit",
              "orbit '" + lab + "' (e = " + std::to_string(o.e()) +
                  ") lies outside basis.emax = " + std::to_string(cfg_.basis.emax));
      }
    } catch (const std::exception&) {
      error("V-43", "imsrg.denominator.delta_orbit",
            "imsrg.denominator.delta_orbit '" + lab + "' is neither \"all\" nor an orbit label");
    }
  }
  if (cfg_.imsrg.omega.storage == OmegaStorage::Discard && cfg_.operators.any()) {
    error("V-44", "imsrg.omega.storage",
          "imsrg.omega.storage = \"discard\" is incompatible with extra operators");
  }
  if (cfg_.imsrg.omega.storage == OmegaStorage::Scratch && cfg_.run.scratch_dir.empty()) {
    error("V-44", "run.scratch_dir", "imsrg.omega.storage = \"scratch\" requires run.scratch_dir");
  }
  // design/16 V-45 and V-52 (and imsrg++, which tests method == "magnus") mean exactly magnus.
  if (im.perturbative_triples && im.method != Method::Magnus) {
    strict("V-45", "imsrg.perturbative_triples",
           "imsrg.perturbative_triples requires imsrg.method = \"magnus\", got \"" +
               std::string(to_string(im.method)) + "\"");
  }
  if (is_flow_method(im.method) && cfg_.imsrg3.active()) {
    error("V-46", "imsrg.method",
          "flow method \"" + std::string(to_string(im.method)) +
              "\" cannot be combined with IMSRG(3)");
  }
  if (im.space) {
    const auto& s = *im.space;
    if (s.emax > cfg_.basis.emax) {
      error("V-48", "imsrg.space.emax",
            "imsrg.space.emax = " + std::to_string(s.emax) +
                " exceeds basis.emax = " + std::to_string(cfg_.basis.emax));
    }
    if (s.e2max > 2 * s.emax) {
      error("V-48", "imsrg.space.e2max",
            "imsrg.space.e2max = " + std::to_string(s.e2max) +
                " exceeds 2*imsrg.space.emax = " + std::to_string(2 * s.emax));
    }
    if (s.e3max > cfg_.basis.e3max) {
      error("V-48", "imsrg.space.e3max",
            "imsrg.space.e3max = " + std::to_string(s.e3max) +
                " exceeds basis.e3max = " + std::to_string(cfg_.basis.e3max));
    }
    if (s.emax_3body > s.emax) {
      error("V-48", "imsrg.space.emax_3body",
            "imsrg.space.emax_3body = " + std::to_string(s.emax_3body) +
                " exceeds imsrg.space.emax = " + std::to_string(s.emax));
    }
  }
}

void Checker::imsrg3_rules() {
  const auto& i3 = cfg_.imsrg3;
  const Imsrg3Config def{};
  if (!i3.active()) {
    auto bad = [&](std::string_view key) {
      strict("V-50", "imsrg3." + std::string(key),
             "imsrg3." + std::string(key) + " is set but IMSRG(3) is not active");
    };
    if (i3.variant != def.variant) bad("variant");
    if (i3.no_qqq != def.no_qqq) bad("no_qqq");
    if (i3.only_2b_eta != def.only_2b_eta) bad("only_2b_eta");
    if (i3.only_2b_omega != def.only_2b_omega) bad("only_2b_omega");
    if (i3.threebody_threshold != def.threebody_threshold) bad("threebody_threshold");
  }
  if (i3.at_end) {
    if (cfg_.imsrg.method != Method::Magnus) {
      strict("V-52", "imsrg3.at_end", "imsrg3.at_end requires imsrg.method = \"magnus\"");
    }
    if (!cfg_.interaction.three_body) {
      warning("V-52", "imsrg3.at_end",
              "imsrg3.at_end is set but no three-body interaction is given");
    }
    if (!i3.in_flow) {
      warning("W-Q11", "imsrg3.at_end", "input 3N is not included at IMSRG(3) level");
    }
  }
  if ((i3.de3max || i3.occ_nat3_cut) && !(i3.active() || cfg_.imsrg.perturbative_triples)) {
    strict("V-54", i3.de3max ? "imsrg3.de3max" : "imsrg3.occ_nat3_cut",
           "imsrg3.de3max / imsrg3.occ_nat3_cut require IMSRG(3) or imsrg.perturbative_triples");
  }
}

void Checker::operators() {
  for (std::size_t i = 0; i < cfg_.operators.from_file.size(); ++i) {
    const auto& e = cfg_.operators.from_file[i];
    const std::string base = "operators.from_file[" + std::to_string(i) + "]";
    if (e.rank < 0) {
      error("V-62", base + ".rank", base + ".rank must be >= 0, got " + std::to_string(e.rank));
    }
    if (e.dtz < -2 || e.dtz > 2) {
      error("V-62", base + ".dtz", base + ".dtz must be in -2..2, got " + std::to_string(e.dtz));
    }
    if (e.parity != 0 && e.parity != 1) {
      error("V-62", base + ".parity",
            base + ".parity must be 0 or 1, got " + std::to_string(e.parity));
    }
    if (e.particle_rank < 1 || e.particle_rank > 3) {
      error("V-62", base + ".particle_rank",
            base + ".particle_rank must be 1, 2 or 3, got " + std::to_string(e.particle_rank));
    }
  }
}

void Checker::output() {
  const auto& s = cfg_.output.stem;
  if (s.find('/') != std::string::npos || s.find('\\') != std::string::npos) {
    error("V-63", "output.stem", "output.stem '" + s + "' must not contain path separators");
  }
}

void Checker::check_readable(const std::string& path, const std::string& file) {
  std::error_code ec;
  if (!fs::is_regular_file(file, ec) || ::access(file.c_str(), R_OK) != 0) {
    error("V-14", path, "file '" + file + "' does not exist or is not readable");
  }
}

void Checker::check_writable_dir(const std::string& path, const std::string& dir) {
  std::error_code ec;
  fs::path p = fs::absolute(dir, ec);
  fs::path probe = p;
  while (!probe.empty() && !fs::exists(probe, ec)) {
    const fs::path parent = probe.parent_path();
    if (parent == probe) break;
    probe = parent;
  }
  if (!fs::is_directory(probe, ec) || ::access(probe.c_str(), W_OK) != 0) {
    error("V-14", path,
          "directory '" + dir + "' is not writable (checked '" + probe.generic_string() + "')");
  }
}

void Checker::files() {
  const auto& t2 = cfg_.interaction.two_body;
  if (t2.format != TwoBodyFormat::SchematicMinnesota) {
    for (std::size_t i = 0; i < t2.file.size(); ++i) {
      check_readable("interaction.two_body.file", t2.file[i]);
    }
  }
  if (cfg_.interaction.three_body) {
    check_readable("interaction.three_body.file", cfg_.interaction.three_body->file);
  }
  if (cfg_.reference.occupations_file) {
    check_readable("reference.occupations_file", *cfg_.reference.occupations_file);
  }
  for (std::size_t i = 0; i < cfg_.operators.from_file.size(); ++i) {
    const auto& e = cfg_.operators.from_file[i];
    const std::string base = "operators.from_file[" + std::to_string(i) + "]";
    check_readable(base + ".file_2b", e.file_2b);
    if (e.file_3b) check_readable(base + ".file_3b", *e.file_3b);
  }
  check_writable_dir("output.directory", cfg_.output.directory);
  if (cfg_.imsrg.omega.storage == OmegaStorage::Scratch && !cfg_.run.scratch_dir.empty()) {
    check_writable_dir("run.scratch_dir", cfg_.run.scratch_dir);
  }
}

}  // namespace

ConfigIssues validate(const RunConfig& cfg, const RawTree& tree, const ResolveOptions& options) {
  Checker c(cfg, tree, options);
  c.run();
  return c.take();
}

}  // namespace imsrg::driver
