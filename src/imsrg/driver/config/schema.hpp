#pragma once
/// @file
/// The typed run configuration of the `imsrg` executable (design/16-config-and-cli.md section 2):
/// the `RunConfig` tree, its enums with their canonical TOML spellings and phases, and the
/// provenance record.
///
/// A `RunConfig` is always *resolved*: defaults are filled and derived values computed (effective
/// `basis.e3max`, file e2max/lmax, mass number, number of stages, output stem and flow file); only
/// keys whose absence has a meaning of its own stay `std::optional`. It is built by `load_config`
/// (imsrg/driver/config/load_config.hpp) and treated as immutable afterwards: pipeline stages
/// receive it, or one section, by const reference. The keys, their order, docs and defaults are
/// described once, in `visit_schema` (imsrg/driver/config/schema_keys.hpp).
///
/// Generator and solver-method enums are defined here for now; M8 reconciles them with the `flow`
/// module (QUESTIONS M8a-Q1).

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "imsrg/driver/config/config_issue.hpp"
#include "imsrg/driver/config/phase.hpp"
#include "imsrg/driver/operators/operator_request.hpp"

namespace imsrg::driver {

// ---- Enums
// ---------------------------------------------------------------------------------------

/// `run.task`: where the run stops.
enum class Task { Imsrg, Hf, Mp3, FciExport };
/// `run.log_level`: stdout verbosity.
enum class LogLevel { Quiet, Normal, Debug };
/// `basis.kind`: basis the Hamiltonian is normal ordered in.
enum class BasisKind { Hf, Oscillator, Nat };
/// `hamiltonian.kinetic`: kinetic-energy term added to the input interaction.
enum class KineticEnergy { Intrinsic, Lab, None };
/// `interaction.two_body.format`.
enum class TwoBodyFormat {
  Me2j,
  Navratil,
  Oslo,
  Oakridge,
  OakridgeBin,
  Takayuki,
  Nushellx,
  SchematicMinnesota
};
/// `interaction.three_body.type`.
enum class ThreeBodyType { Full, No2b, Mono };
/// `interaction.three_body.format` (HDF5 dropped, design/00 D24).
enum class ThreeBodyFormat { Me3j, Navratil };
/// `interaction.three_body.no2b_precision`.
enum class No2bPrecision { Single, Half };
/// `natural_orbitals.order`.
enum class NatOrder { Occupation, Energy };
/// `imsrg.method`: the solver.
enum class Method {
  Magnus,
  MagnusBackoff,
  MagnusModifiedEuler,
  MagnusAdaptive,
  Flow,
  FlowEuler,
  FlowRk4,
  Restore4thOrder
};
/// `imsrg.generator.core` / `.valence`.
enum class Generator {
  White,
  Atan,
  ImaginaryTime,
  Wegner,
  QtransferAtan,
  ShellModel,
  ShellModelAtan,
  ShellModelWegner,
  ShellModelImaginaryTime,
  ShellModelAtanNpnh,
  HartreeFock,
  OnePa
};
/// `imsrg.denominator.partitioning`.
enum class Partitioning { EpsteinNesbet, MollerPlesset, MpIsospin };
/// `imsrg.omega.storage`.
enum class OmegaStorage { Memory, Scratch, Discard };
/// `imsrg3.variant`.
enum class Imsrg3Variant { Full, N7, Mp4 };
/// `output.valence_file_format`.
enum class ValenceFileFormat { Tokyo, Nushellx, Antoine };
/// `operators.from_file[i].format`.
enum class OperatorFileFormat { Navratil, Miyagi };

// ---- Enum spellings
// ------------------------------------------------------------------------------

/// One enumerator with its canonical TOML spelling and phase.
template <class E>
struct EnumEntry {
  E value;
  std::string_view name;
  Phase phase = Phase::P1;
};

/// Spellings of a config enum: specialisations provide `type_name` and `entries` (all
/// enumerators, in declaration order). Only canonical spellings live here; legacy spellings are
/// the legacy layer's business.
template <class E>
struct EnumTraits;

/// An enum with `EnumTraits`.
template <class E>
concept ConfigEnum = std::is_enum_v<E> && requires {
  EnumTraits<E>::type_name;
  EnumTraits<E>::entries;
};

/// Canonical spelling of `value` ("" if it is not an enumerator).
template <ConfigEnum E>
constexpr std::string_view to_string(E value) noexcept {
  for (const auto& entry : EnumTraits<E>::entries) {
    if (entry.value == value) return entry.name;
  }
  return {};
}

/// The enumerator spelled exactly `name`, or nullopt.
template <ConfigEnum E>
constexpr std::optional<E> enum_from_string(std::string_view name) noexcept {
  for (const auto& entry : EnumTraits<E>::entries) {
    if (entry.name == name) return entry.value;
  }
  return std::nullopt;
}

/// Phase of `value`.
template <ConfigEnum E>
constexpr Phase enum_phase(E value) noexcept {
  for (const auto& entry : EnumTraits<E>::entries) {
    if (entry.value == value) return entry.phase;
  }
  return Phase::NotSupported;
}

/// All canonical spellings of `E`, in declaration order (did-you-mean candidates, docs).
template <ConfigEnum E>
std::vector<std::string_view> enum_names() {
  std::vector<std::string_view> names;
  for (const auto& entry : EnumTraits<E>::entries) names.push_back(entry.name);
  return names;
}

#define IMSRG_DRIVER_ENUM_TRAITS(Enum, TypeName, ...)                              \
  template <>                                                                      \
  struct EnumTraits<Enum> {                                                        \
    static constexpr std::string_view type_name = TypeName;                        \
    static constexpr auto entries = std::to_array<EnumEntry<Enum>>({__VA_ARGS__}); \
  }

IMSRG_DRIVER_ENUM_TRAITS(Task, "task", {Task::Imsrg, "imsrg"}, {Task::Hf, "hf"}, {Task::Mp3, "mp3"},
                         {Task::FciExport, "fci_export", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(LogLevel, "log level", {LogLevel::Quiet, "quiet"},
                         {LogLevel::Normal, "normal"}, {LogLevel::Debug, "debug"});
IMSRG_DRIVER_ENUM_TRAITS(BasisKind, "basis kind", {BasisKind::Hf, "hf"},
                         {BasisKind::Oscillator, "oscillator"}, {BasisKind::Nat, "nat"});
IMSRG_DRIVER_ENUM_TRAITS(KineticEnergy, "kinetic energy", {KineticEnergy::Intrinsic, "intrinsic"},
                         {KineticEnergy::Lab, "lab"}, {KineticEnergy::None, "none"});
IMSRG_DRIVER_ENUM_TRAITS(TwoBodyFormat, "two-body format", {TwoBodyFormat::Me2j, "me2j"},
                         {TwoBodyFormat::Navratil, "navratil", Phase::Planned},
                         {TwoBodyFormat::Oslo, "oslo", Phase::Planned},
                         {TwoBodyFormat::Oakridge, "oakridge", Phase::Planned},
                         {TwoBodyFormat::OakridgeBin, "oakridge_bin", Phase::Planned},
                         {TwoBodyFormat::Takayuki, "takayuki", Phase::Planned},
                         {TwoBodyFormat::Nushellx, "nushellx", Phase::Planned},
                         {TwoBodyFormat::SchematicMinnesota, "schematic_minnesota",
                          Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(ThreeBodyType, "three-body type", {ThreeBodyType::Full, "full"},
                         {ThreeBodyType::No2b, "no2b"},
                         {ThreeBodyType::Mono, "mono", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(ThreeBodyFormat, "three-body format", {ThreeBodyFormat::Me3j, "me3j"},
                         {ThreeBodyFormat::Navratil, "navratil", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(No2bPrecision, "NO2B precision", {No2bPrecision::Single, "single"},
                         {No2bPrecision::Half, "half"});
IMSRG_DRIVER_ENUM_TRAITS(NatOrder, "natural-orbital order", {NatOrder::Occupation, "occupation"},
                         {NatOrder::Energy, "energy"});
IMSRG_DRIVER_ENUM_TRAITS(Method, "method", {Method::Magnus, "magnus"},
                         {Method::MagnusBackoff, "magnus_backoff", Phase::Planned},
                         {Method::MagnusModifiedEuler, "magnus_modified_euler", Phase::Planned},
                         {Method::MagnusAdaptive, "magnus_adaptive", Phase::Planned},
                         {Method::Flow, "flow"}, {Method::FlowEuler, "flow_euler", Phase::Planned},
                         {Method::FlowRk4, "flow_rk4"},
                         {Method::Restore4thOrder, "restore_4th_order", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(Generator, "generator", {Generator::White, "white"},
                         {Generator::Atan, "atan"}, {Generator::ImaginaryTime, "imaginary_time"},
                         {Generator::Wegner, "wegner"},
                         {Generator::QtransferAtan, "qtransfer_atan"},
                         {Generator::ShellModel, "shell_model"},
                         {Generator::ShellModelAtan, "shell_model_atan"},
                         {Generator::ShellModelWegner, "shell_model_wegner", Phase::Planned},
                         {Generator::ShellModelImaginaryTime, "shell_model_imaginary_time"},
                         {Generator::ShellModelAtanNpnh, "shell_model_atan_npnh", Phase::Planned},
                         {Generator::HartreeFock, "hartree_fock"},
                         {Generator::OnePa, "one_pa", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(Partitioning, "denominator partitioning",
                         {Partitioning::EpsteinNesbet, "epstein_nesbet"},
                         {Partitioning::MollerPlesset, "moller_plesset"},
                         {Partitioning::MpIsospin, "mp_isospin", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(OmegaStorage, "Omega storage", {OmegaStorage::Memory, "memory"},
                         {OmegaStorage::Scratch, "scratch"}, {OmegaStorage::Discard, "discard"});
IMSRG_DRIVER_ENUM_TRAITS(Imsrg3Variant, "IMSRG(3) variant", {Imsrg3Variant::Full, "full"},
                         {Imsrg3Variant::N7, "n7"}, {Imsrg3Variant::Mp4, "mp4", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(ValenceFileFormat, "valence file format",
                         {ValenceFileFormat::Tokyo, "tokyo"},
                         {ValenceFileFormat::Nushellx, "nushellx", Phase::Planned},
                         {ValenceFileFormat::Antoine, "antoine", Phase::Planned});
IMSRG_DRIVER_ENUM_TRAITS(OperatorFileFormat, "operator file format",
                         {OperatorFileFormat::Navratil, "navratil"},
                         {OperatorFileFormat::Miyagi, "miyagi"});

#undef IMSRG_DRIVER_ENUM_TRAITS

/// True for the shell-model generators (they need valence orbits, design/16 V-41, V-42).
constexpr bool is_shell_model_generator(Generator g) noexcept {
  return g == Generator::ShellModel || g == Generator::ShellModelAtan ||
         g == Generator::ShellModelWegner || g == Generator::ShellModelImaginaryTime ||
         g == Generator::ShellModelAtanNpnh;
}

/// True for the ODE flow methods (no IMSRG(3) algebra, design/16 V-46).
constexpr bool is_flow_method(Method m) noexcept {
  return m == Method::Flow || m == Method::FlowEuler || m == Method::FlowRk4 ||
         m == Method::Restore4thOrder;
}

// ---- Sections
// ------------------------------------------------------------------------------------ Field docs
// are short; the authoritative meaning, defaults and rules are in design/16 section 2 and in
// visit_schema. "Resolved" = always set after load_config (derived if not given).

/// `[run]`.
struct TaskConfig {
  Task task = Task::Imsrg;
  std::string scratch_dir;  ///< Omega spill directory (used iff imsrg.omega.storage = scratch)
  LogLevel log_level = LogLevel::Normal;
  friend bool operator==(const TaskConfig&, const TaskConfig&) = default;
};

/// `[basis]`: model space and single-particle basis.
struct BasisConfig {
  int emax = 0;                   ///< required; e = 2n + l <= emax
  double hw = 0.0;                ///< required; oscillator frequency in MeV
  int e3max = 12;                 ///< resolved to the effective value min(e3max, 3 * emax)
  std::optional<int> lmax;        ///< l cut of the whole calculation; none = no cut
  std::optional<int> lmax3;       ///< l cut of the 3N interaction; none = no cut
  std::optional<int> emax_unocc;  ///< emax for (l, j) unoccupied in the reference; none = emax
  int mass_number = 0;            ///< resolved: A of reference.nucleus, else of the valence core
  BasisKind kind = BasisKind::Hf;
  friend bool operator==(const BasisConfig&, const BasisConfig&) = default;
};

/// `[reference]`.
struct ReferenceConfig {
  std::optional<std::string> nucleus;           ///< e.g. "O16"; none = core of the valence space
  std::optional<std::string> occupations_file;  ///< rows "n l 2j 2tz occ"
  friend bool operator==(const ReferenceConfig&, const ReferenceConfig&) = default;
};

/// `[valence]`. Neither set = single-reference decoupling.
struct ValenceConfig {
  std::optional<std::string> space;   ///< named space ("sd-shell"), 0hw-shell, jj-shell, comma list
  std::optional<std::string> custom;  ///< custom orbit lists; exclusive with `space`
  /// True if a valence space is defined (valence orbits exist; stage 2 possible).
  bool has_valence_space() const noexcept { return space.has_value() || custom.has_value(); }
  friend bool operator==(const ValenceConfig&, const ValenceConfig&) = default;
};

/// `[hamiltonian]`: terms added to the input interaction.
struct HamiltonianConfig {
  KineticEnergy kinetic = KineticEnergy::Intrinsic;  ///< resolved (nushellx -> none, trap -> lab)
  std::optional<double> trap_hw;  ///< lab-frame HO trap frequency (MeV); requires kinetic = lab
  double beta_cm = 0.0;           ///< Lawson term strength; active iff |beta_cm| > 1e-6
  double beta_cm_hw = 0.0;        ///< resolved: basis.hw if not given
  bool nucleon_mass_correction = false;
  bool relativistic_correction = false;  ///< planned
  friend bool operator==(const HamiltonianConfig&, const HamiltonianConfig&) = default;
};

/// `[interaction.two_body]`.
struct TwoBodyInputConfig {
  std::vector<std::string> file;  ///< required; 1 path (2 for oakridge sp + vnn files)
  TwoBodyFormat format = TwoBodyFormat::Me2j;
  int e1max = 0;                    ///< required; single-particle emax stored in the file
  int e2max = 0;                    ///< resolved: 2 * e1max if not given
  int lmax = 0;                     ///< resolved: e1max if not given
  std::optional<std::string> lecs;  ///< planned (hdf5 formats only)
  friend bool operator==(const TwoBodyInputConfig&, const TwoBodyInputConfig&) = default;
};

/// `[interaction.three_body]` (present iff a 3N interaction is read).
struct ThreeBodyInputConfig {
  std::string file;  ///< required
  ThreeBodyType type = ThreeBodyType::Full;
  ThreeBodyFormat format = ThreeBodyFormat::Me3j;
  No2bPrecision no2b_precision = No2bPrecision::Single;
  int e1max = 0;             ///< required
  int e3max = 0;             ///< required
  std::optional<int> e2max;  ///< none = no extra e2 cut
  bool store_pn = false;
  bool discard_no2b_after_hf = false;  ///< planned (diagnostic)
  friend bool operator==(const ThreeBodyInputConfig&, const ThreeBodyInputConfig&) = default;
};

/// `[interaction]`.
struct InteractionConfig {
  TwoBodyInputConfig two_body;
  std::optional<ThreeBodyInputConfig> three_body;
  friend bool operator==(const InteractionConfig&, const InteractionConfig&) = default;
};

/// `[hartree_fock]`.
struct HartreeFockConfig {
  bool freeze_occupations = false;
  friend bool operator==(const HartreeFockConfig&, const HartreeFockConfig&) = default;
};

/// `[natural_orbitals]`.
struct NaturalOrbitalsConfig {
  bool use_occupations = false;
  NatOrder order = NatOrder::Occupation;
  friend bool operator==(const NaturalOrbitalsConfig&, const NaturalOrbitalsConfig&) = default;
};

/// `[imsrg.generator]`.
struct GeneratorConfig {
  Generator core = Generator::Atan;               ///< stage 1 (the only stage if steps = 1)
  Generator valence = Generator::ShellModelAtan;  ///< stage 2; reset to default if steps = 1
  int qtransfer_n = 1;                            ///< exponent index of qtransfer_atan (0.5 n)
  friend bool operator==(const GeneratorConfig&, const GeneratorConfig&) = default;
};

/// `[imsrg.denominator]`.
struct DenominatorConfig {
  Partitioning partitioning = Partitioning::EpsteinNesbet;
  double delta = 0.0;                      ///< requires delta_orbit if nonzero
  std::optional<std::string> delta_orbit;  ///< "all" or an orbit label
  friend bool operator==(const DenominatorConfig&, const DenominatorConfig&) = default;
};

/// `[imsrg.step]` (numerics-defining step control).
struct StepConfig {
  double smax = 200.0;
  double dsmax = 0.5;
  double ds_0 = 0.5;
  double domega = 0.1;
  double omega_norm_max = 0.25;
  double eta_criterion = 1e-6;
  double ode_tolerance = 1e-6;
  friend bool operator==(const StepConfig&, const StepConfig&) = default;
};

/// `[imsrg.omega]`.
struct OmegaConfig {
  OmegaStorage storage = OmegaStorage::Memory;
  bool hunter_gatherer = false;
  friend bool operator==(const OmegaConfig&, const OmegaConfig&) = default;
};

/// `[imsrg.space]` (present iff any of its keys was given; all four resolved then).
struct ImsrgSpaceConfig {
  int emax = 0;        ///< default basis.emax
  int e2max = 0;       ///< default 2 * emax
  int e3max = 0;       ///< default min(basis.e3max, 3 * emax)
  int emax_3body = 0;  ///< default emax
  friend bool operator==(const ImsrgSpaceConfig&, const ImsrgSpaceConfig&) = default;
};

/// `[imsrg]`.
struct ImsrgConfig {
  Method method = Method::Magnus;
  int steps = 1;  ///< resolved: 2 if valence orbits exist, else 1
  bool use_brueckner_bch = false;
  bool goose_tank = false;  ///< planned
  bool perturbative_triples = false;
  GeneratorConfig generator;
  DenominatorConfig denominator;
  StepConfig step;
  OmegaConfig omega;
  std::optional<ImsrgSpaceConfig> space;
  friend bool operator==(const ImsrgConfig&, const ImsrgConfig&) = default;
};

/// `[imsrg3]`. All defaults = IMSRG(2).
struct Imsrg3Config {
  bool in_flow = false;
  bool at_end = false;
  Imsrg3Variant variant = Imsrg3Variant::Full;
  bool factorized = false;  ///< IMSRG(3f2): BchOptions::factorized
  bool no_qqq = false;      ///< planned
  bool only_2b_eta = false;
  bool only_2b_omega = false;
  bool discard_residual_input_3n = false;
  double threebody_threshold = 0.0;
  std::optional<double> de3max;        ///< none = no energy cut (old default 99)
  std::optional<double> occ_nat3_cut;  ///< none = no cut (old default -1)
  /// True if any IMSRG(3) evaluation is requested (in the flow or at the end).
  bool active() const noexcept { return in_flow || at_end; }
  friend bool operator==(const Imsrg3Config&, const Imsrg3Config&) = default;
};

/// One `operators.from_file` entry (old `OperatorsFromFile` tag `Name^J_T_P_R^file2b[^file3b]`).
struct OperatorFromFile {
  std::string name;
  int rank = 0;           ///< J; rank > 0 is planned
  int dtz = 0;            ///< -2..2
  int parity = 0;         ///< 0 or 1
  int particle_rank = 2;  ///< 1, 2 or 3
  std::string file_2b;
  std::optional<std::string> file_3b;
  OperatorFileFormat format = OperatorFileFormat::Navratil;
  friend bool operator==(const OperatorFromFile&, const OperatorFromFile&) = default;
};

/// `[operators]`.
struct OperatorsConfig {
  std::vector<OperatorRequest> list;         ///< transformed with the flow (macros expanded)
  std::vector<OperatorRequest> first_order;  ///< planned
  std::vector<OperatorFromFile> from_file;
  /// True if any operator is requested.
  bool any() const noexcept { return !list.empty() || !first_order.empty() || !from_file.empty(); }
  friend bool operator==(const OperatorsConfig&, const OperatorsConfig&) = default;
};

/// `[output]`.
struct OutputConfig {
  std::string directory = "output";
  std::string stem;       ///< resolved: <method>_<reference>_<valence>_hw<hw>_e<emax>_A<A>
  std::string flow_file;  ///< resolved: <directory>/<stem>.flow
  ValenceFileFormat valence_file_format = ValenceFileFormat::Tokyo;
  bool write_omega = false;
  bool write_ho_ops = false;      ///< planned
  bool write_hf_ops = false;      ///< planned
  std::vector<std::string> spwf;  ///< orbit labels or "valence"
  friend bool operator==(const OutputConfig&, const OutputConfig&) = default;
};

/// Where a configuration came from (design/16 section 1.7). Written as `[provenance]` into
/// `resolved_config.toml`, ignored when that file is read back, and not part of equality.
struct Provenance {
  std::string version;                     ///< imsrg version
  std::string git_hash;                    ///< "unknown" if not built from git
  std::string date;                        ///< load time, ISO 8601 UTC
  std::string command_line;                ///< original command line
  std::vector<std::string> legacy_args;    ///< legacy tokens, verbatim
  ConfigIssues warnings;                   ///< warnings of the load (also printed)
  std::vector<std::string> applied_rules;  ///< shorthand/sentinel rewrites, human readable
};

/// The complete resolved configuration of one run.
struct RunConfig {
  TaskConfig run;
  BasisConfig basis;
  ReferenceConfig reference;
  ValenceConfig valence;
  HamiltonianConfig hamiltonian;
  InteractionConfig interaction;
  HartreeFockConfig hartree_fock;
  NaturalOrbitalsConfig natural_orbitals;
  ImsrgConfig imsrg;
  Imsrg3Config imsrg3;
  OperatorsConfig operators;
  OutputConfig output;
  Provenance provenance;  ///< metadata only: excluded from operator==

  /// Field-by-field equality of every section except `provenance`.
  friend bool operator==(const RunConfig& a, const RunConfig& b) {
    return a.run == b.run && a.basis == b.basis && a.reference == b.reference &&
           a.valence == b.valence && a.hamiltonian == b.hamiltonian &&
           a.interaction == b.interaction && a.hartree_fock == b.hartree_fock &&
           a.natural_orbitals == b.natural_orbitals && a.imsrg == b.imsrg && a.imsrg3 == b.imsrg3 &&
           a.operators == b.operators && a.output == b.output;
  }
};

}  // namespace imsrg::driver
