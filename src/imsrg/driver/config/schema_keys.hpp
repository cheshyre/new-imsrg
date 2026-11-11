#pragma once
/// @file
/// The single schema table (design/16 section 6.1): every TOML key of `RunConfig`, once, with its
/// path, doc, default text, requiredness and phase, bound to its `RunConfig` field.
///
/// `visit_schema(config, visitor)` calls, in canonical order (the order of design/16 section 2 and
/// of `resolved_config.toml`),
///   * `visitor.field(const KeyInfo&, Field&)` for every leaf key, where `Field` is the field's
///     type (`bool`, `int`, `double`, `std::string`, `std::optional<int|double|std::string>`, a
///     `ConfigEnum`, `std::vector<std::string>`, `std::vector<OperatorRequest>` or
///     `std::vector<OperatorFromFile>`), const-qualified when `config` is const;
///   * `visitor.optional_table(std::string_view table, std::optional<Section>& section, body)`
///     for the optional tables `interaction.three_body` and `imsrg.space`; the visitor decides
///     whether to call `body(Section&)` (reader: emplace and call if the raw tree has the table;
///     writer: call if engaged; key collector: call on a temporary).
/// The TOML reader (unknown-key detection, did-you-mean), the resolver (typed reading), the writer
/// and the docs all walk this one function, so a key exists in exactly one place.
///
/// Defaults are the default member initializers of the `RunConfig` sections; `default_text` only
/// documents them (and the derived defaults computed by `resolve`).

#include <span>
#include <string_view>
#include <vector>

#include "imsrg/driver/config/phase.hpp"
#include "imsrg/driver/config/schema.hpp"

namespace imsrg::driver {

/// Static description of one schema key.
struct KeyInfo {
  std::string_view path;          ///< full dotted path, e.g. "interaction.two_body.e1max"
  std::string_view doc;           ///< one line
  std::string_view default_text;  ///< documented default: "12", "2*e1max", "required", "none"
  bool required = false;          ///< V-03; keys of an optional table only if the table is present
  bool derived = false;           ///< not given -> computed by `resolve` from other keys
  Phase phase = Phase::P1;        ///< Planned: any non-default value is E-PLANNED
  bool string_or_array = false;   ///< accepts a string or an array of strings (two_body.file)
};

/// Visits every schema key of `config` (a `RunConfig` or `const RunConfig`) in canonical order;
/// see the file comment for the visitor interface.
template <class Config, class Visitor>
void visit_schema(Config& c, Visitor&& v) {
  constexpr Phase PL = Phase::Planned;
  // [run]
  v.field(KeyInfo{"run.task", "where the run stops: imsrg, hf, mp3, fci_export", "imsrg"},
          c.run.task);
  v.field(KeyInfo{"run.scratch_dir", "directory for Omega spill files (storage = scratch)", "\"\""},
          c.run.scratch_dir);
  v.field(KeyInfo{"run.log_level", "stdout verbosity: quiet, normal, debug", "normal"},
          c.run.log_level);
  // [basis]
  v.field(KeyInfo{"basis.emax", "single-particle truncation e = 2n + l", "required", true},
          c.basis.emax);
  v.field(KeyInfo{"basis.hw", "oscillator frequency (MeV)", "required", true}, c.basis.hw);
  v.field(KeyInfo{"basis.e3max", "three-body truncation; effective min(e3max, 3*emax)", "12"},
          c.basis.e3max);
  v.field(KeyInfo{"basis.lmax", "l cut of the whole calculation", "none"}, c.basis.lmax);
  v.field(KeyInfo{"basis.lmax3", "l cut of the three-body interaction", "none"}, c.basis.lmax3);
  v.field(KeyInfo{"basis.emax_unocc", "emax for (l, j) unoccupied in the reference", "none"},
          c.basis.emax_unocc);
  v.field(KeyInfo{"basis.mass_number", "A in Trel and radius operators",
                  "A of the reference nucleus (or valence core)", false, true},
          c.basis.mass_number);
  v.field(KeyInfo{"basis.kind", "normal-ordering basis: hf, oscillator, nat", "hf"}, c.basis.kind);
  // [reference], [valence]
  v.field(
      KeyInfo{"reference.nucleus", "reference nucleus, e.g. \"O16\"", "core of the valence space"},
      c.reference.nucleus);
  v.field(KeyInfo{"reference.occupations_file", "per-orbit occupations: n l 2j 2tz occ", "none"},
          c.reference.occupations_file);
  v.field(KeyInfo{"valence.space", "named valence space, 0hw-shell, jj-shell, FCI or comma list",
                  "none (single reference)"},
          c.valence.space);
  v.field(KeyInfo{"valence.custom", "custom comma-separated core and valence orbit lists", "none"},
          c.valence.custom);
  // [hamiltonian]
  v.field(KeyInfo{"hamiltonian.kinetic", "kinetic energy added: intrinsic, lab, none",
                  "intrinsic (none for nushellx files, lab with trap_hw)", false, true},
          c.hamiltonian.kinetic);
  v.field(KeyInfo{"hamiltonian.trap_hw", "lab-frame HO trap frequency (MeV); needs kinetic = lab",
                  "none"},
          c.hamiltonian.trap_hw);
  v.field(KeyInfo{"hamiltonian.beta_cm", "Lawson term strength beta (0 = off)", "0.0"},
          c.hamiltonian.beta_cm);
  v.field(KeyInfo{"hamiltonian.beta_cm_hw", "frequency of the Lawson term (MeV)", "basis.hw", false,
                  true},
          c.hamiltonian.beta_cm_hw);
  v.field(KeyInfo{"hamiltonian.nucleon_mass_correction", "proton-neutron mass splitting", "false"},
          c.hamiltonian.nucleon_mass_correction);
  v.field(KeyInfo{"hamiltonian.relativistic_correction", "p^4 kinetic correction", "false", false,
                  false, PL},
          c.hamiltonian.relativistic_correction);
  // [interaction.two_body]
  v.field(KeyInfo{"interaction.two_body.file", "2N file (oakridge: [sp, vnn])", "required", true,
                  false, Phase::P1, true},
          c.interaction.two_body.file);
  v.field(KeyInfo{"interaction.two_body.format", "2N file format", "me2j"},
          c.interaction.two_body.format);
  v.field(KeyInfo{"interaction.two_body.e1max", "emax stored in the 2N file", "required", true},
          c.interaction.two_body.e1max);
  v.field(KeyInfo{"interaction.two_body.e2max", "e2max cut of the 2N file", "2*e1max", false, true},
          c.interaction.two_body.e2max);
  v.field(KeyInfo{"interaction.two_body.lmax", "lmax cut of the 2N file", "e1max", false, true},
          c.interaction.two_body.lmax);
  v.field(
      KeyInfo{"interaction.two_body.lecs", "LEC preset (hdf5 formats)", "none", false, false, PL},
      c.interaction.two_body.lecs);
  // [interaction.three_body]
  v.optional_table("interaction.three_body", c.interaction.three_body, [&](auto& t) {
    v.field(KeyInfo{"interaction.three_body.file", "3N file", "required", true}, t.file);
    v.field(KeyInfo{"interaction.three_body.type", "3N content: full, no2b, mono", "full"}, t.type);
    v.field(KeyInfo{"interaction.three_body.format", "3N file family for type = full", "me3j"},
            t.format);
    v.field(
        KeyInfo{"interaction.three_body.no2b_precision", "NO2B storage: single, half", "single"},
        t.no2b_precision);
    v.field(KeyInfo{"interaction.three_body.e1max", "e1max of the 3N file", "required", true},
            t.e1max);
    v.field(KeyInfo{"interaction.three_body.e3max", "e3max of the 3N file", "required", true},
            t.e3max);
    v.field(KeyInfo{"interaction.three_body.e2max", "e2max cut of the 3N file", "none"}, t.e2max);
    v.field(KeyInfo{"interaction.three_body.store_pn", "keep 3N in proton-neutron form", "false"},
            t.store_pn);
    v.field(KeyInfo{"interaction.three_body.discard_no2b_after_hf",
                    "drop the 3N normal-ordered 2B part after HF (diagnostic)", "false", false,
                    false, PL},
            t.discard_no2b_after_hf);
  });
  // [hartree_fock], [natural_orbitals]
  v.field(KeyInfo{"hartree_fock.freeze_occupations", "keep reference occupations fixed", "false"},
          c.hartree_fock.freeze_occupations);
  v.field(
      KeyInfo{"natural_orbitals.use_occupations", "use NAT occupations for the reference", "false"},
      c.natural_orbitals.use_occupations);
  v.field(
      KeyInfo{"natural_orbitals.order", "NAT orbit labelling: occupation, energy", "occupation"},
      c.natural_orbitals.order);
  // [imsrg]
  v.field(KeyInfo{"imsrg.method", "solver", "magnus"}, c.imsrg.method);
  v.field(KeyInfo{"imsrg.steps", "decoupling stages (1 or 2)", "2 if valence orbits exist, else 1",
                  false, true},
          c.imsrg.steps);
  v.field(KeyInfo{"imsrg.use_brueckner_bch", "Brueckner-style BCH", "false"},
          c.imsrg.use_brueckner_bch);
  v.field(KeyInfo{"imsrg.goose_tank", "IMSRG(2)* goose-tank correction", "false", false, false, PL},
          c.imsrg.goose_tank);
  v.field(
      KeyInfo{"imsrg.perturbative_triples", "perturbative [2,2]->3 energy after stage 1", "false"},
      c.imsrg.perturbative_triples);
  v.field(KeyInfo{"imsrg.generator.core", "generator of stage 1", "atan"}, c.imsrg.generator.core);
  v.field(KeyInfo{"imsrg.generator.valence", "generator of stage 2", "shell_model_atan"},
          c.imsrg.generator.valence);
  v.field(KeyInfo{"imsrg.generator.qtransfer_n", "index n of qtransfer_atan (exponent 0.5 n)", "1"},
          c.imsrg.generator.qtransfer_n);
  v.field(KeyInfo{"imsrg.denominator.partitioning", "generator denominators", "epstein_nesbet"},
          c.imsrg.denominator.partitioning);
  v.field(KeyInfo{"imsrg.denominator.delta", "offset added to denominators", "0.0"},
          c.imsrg.denominator.delta);
  v.field(KeyInfo{"imsrg.denominator.delta_orbit", "\"all\" or an orbit label for delta", "none"},
          c.imsrg.denominator.delta_orbit);
  v.field(KeyInfo{"imsrg.step.smax", "max flow parameter per stage", "200.0"}, c.imsrg.step.smax);
  v.field(KeyInfo{"imsrg.step.dsmax", "max step size", "0.5"}, c.imsrg.step.dsmax);
  v.field(KeyInfo{"imsrg.step.ds_0", "initial step size", "0.5"}, c.imsrg.step.ds_0);
  v.field(KeyInfo{"imsrg.step.domega", "max norm of eta*ds", "0.1"}, c.imsrg.step.domega);
  v.field(KeyInfo{"imsrg.step.omega_norm_max", "Omega norm at which Omega is split", "0.25"},
          c.imsrg.step.omega_norm_max);
  v.field(KeyInfo{"imsrg.step.eta_criterion", "convergence threshold on norm of eta", "1e-6"},
          c.imsrg.step.eta_criterion);
  v.field(KeyInfo{"imsrg.step.ode_tolerance", "ODE solver tolerance (flow methods)", "1e-6"},
          c.imsrg.step.ode_tolerance);
  v.field(
      KeyInfo{"imsrg.omega.storage", "where split Omegas live: memory, scratch, discard", "memory"},
      c.imsrg.omega.storage);
  v.field(KeyInfo{"imsrg.omega.hunter_gatherer", "merge Omega pieces into one", "false"},
          c.imsrg.omega.hunter_gatherer);
  v.optional_table("imsrg.space", c.imsrg.space, [&](auto& s) {
    v.field(KeyInfo{"imsrg.space.emax", "emax of the IMSRG part", "basis.emax", false, true},
            s.emax);
    v.field(
        KeyInfo{"imsrg.space.e2max", "e2max of the IMSRG part", "2*imsrg.space.emax", false, true},
        s.e2max);
    v.field(KeyInfo{"imsrg.space.e3max", "e3max of the IMSRG part",
                    "min(basis.e3max, 3*imsrg.space.emax)", false, true},
            s.e3max);
    v.field(KeyInfo{"imsrg.space.emax_3body", "emax of 3-body operators in the IMSRG part",
                    "imsrg.space.emax", false, true},
            s.emax_3body);
  });
  // [imsrg3]
  v.field(KeyInfo{"imsrg3.in_flow", "3-body terms in the flow commutators", "false"},
          c.imsrg3.in_flow);
  v.field(KeyInfo{"imsrg3.at_end", "replay Omega at IMSRG(3) level after the flow", "false"},
          c.imsrg3.at_end);
  v.field(KeyInfo{"imsrg3.variant", "3-body terms: full, n7, mp4", "full"}, c.imsrg3.variant);
  v.field(KeyInfo{"imsrg3.factorized", "IMSRG(3f2) factorized BCH correction", "false"},
          c.imsrg3.factorized);
  v.field(KeyInfo{"imsrg3.no_qqq", "drop qqq terms", "false", false, false, PL}, c.imsrg3.no_qqq);
  v.field(KeyInfo{"imsrg3.only_2b_eta", "keep the generator 2-body", "false"},
          c.imsrg3.only_2b_eta);
  v.field(KeyInfo{"imsrg3.only_2b_omega", "keep Omega 2-body", "false"}, c.imsrg3.only_2b_omega);
  v.field(
      KeyInfo{"imsrg3.discard_residual_input_3n", "drop input 3N after normal ordering", "false"},
      c.imsrg3.discard_residual_input_3n);
  v.field(
      KeyInfo{"imsrg3.threebody_threshold", "skip 3-body terms below this operator norm", "0.0"},
      c.imsrg3.threebody_threshold);
  v.field(KeyInfo{"imsrg3.de3max", "energy cut on 3-body states", "none"}, c.imsrg3.de3max);
  v.field(KeyInfo{"imsrg3.occ_nat3_cut", "natural-orbital occupation cut on 3-body states", "none"},
          c.imsrg3.occ_nat3_cut);
  // [operators]
  v.field(KeyInfo{"operators.list", "operators transformed with the flow", "[]"}, c.operators.list);
  v.field(KeyInfo{"operators.first_order", "first-order perturbative operator corrections", "[]",
                  false, false, PL},
          c.operators.first_order);
  v.field(KeyInfo{"operators.from_file", "operators read from files", "[]"}, c.operators.from_file);
  // [output]
  v.field(KeyInfo{"output.directory", "output directory (created if missing)", "\"output\""},
          c.output.directory);
  v.field(KeyInfo{"output.stem", "base name of output files",
                  "<method>_<reference>_<valence>_hw<hw>_e<emax>_A<A>", false, true},
          c.output.stem);
  v.field(KeyInfo{"output.flow_file", "flow table", "<directory>/<stem>.flow", false, true},
          c.output.flow_file);
  v.field(KeyInfo{"output.valence_file_format", "valence interaction format", "tokyo"},
          c.output.valence_file_format);
  v.field(KeyInfo{"output.write_omega", "write Omega operators", "false"}, c.output.write_omega);
  v.field(KeyInfo{"output.write_ho_ops", "write operators before HF", "false", false, false, PL},
          c.output.write_ho_ops);
  v.field(
      KeyInfo{"output.write_hf_ops", "write operators before the flow", "false", false, false, PL},
      c.output.write_hf_ops);
  v.field(KeyInfo{"output.spwf", "orbits whose single-particle wave functions are written", "[]"},
          c.output.spwf);
}

/// All schema keys in canonical order (keys of optional tables included).
std::span<const KeyInfo> schema_keys();

/// The key with exactly this path, or nullptr.
const KeyInfo* find_schema_key(std::string_view path);

/// All table paths that may appear in TOML, parents included, in canonical order: "run",
/// "basis", ..., "interaction", "interaction.two_body", "interaction.three_body", ..., "imsrg",
/// "imsrg.generator", ..., "output". (`provenance` is not a schema table; readers skip it.)
std::span<const std::string_view> schema_tables();

/// Direct children (last path component) of `table` that are keys or tables, for did-you-mean:
/// schema_children("imsrg") = {"method", "steps", ..., "generator", "denominator", ...}.
std::vector<std::string_view> schema_children(std::string_view table);

}  // namespace imsrg::driver
