// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_CONFIG_SCHEMA_H_
#define IMSRG_DRIVER_CONFIG_SCHEMA_H_
/// @file
/// The typed run configuration of the `imsrg` executable
/// (design/16-config-and-cli.md section 2): the `RunConfig` tree, its enums
/// with their canonical TOML spellings and phases, and the provenance record.
///
/// A `RunConfig` is always *resolved*: defaults are filled and derived values
/// computed (effective `basis.e3max`, file e2max/lmax, mass number, number of
/// stages, output stem and flow file); only keys whose absence has a meaning of
/// its own stay `std::optional`. It is built by `LoadConfig`
/// (imsrg/driver/config/load_config.h) and treated as immutable afterwards:
/// pipeline stages receive it, or one section, by const reference. The keys,
/// their order, docs and defaults are described once, in `VisitSchema`
/// (imsrg/driver/config/schema_keys.h).
///
/// Generator and solver-method enums are defined here for now; M8 reconciles
/// them with the `flow` module (QUESTIONS M8a-Q1).

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "imsrg/driver/config/config_issue.h"
#include "imsrg/driver/config/phase.h"
#include "imsrg/driver/operators/operator_request.h"

namespace imsrg::driver {

// ---- Enums
// -----------------------------------------------------------------------------

/// `run.task`: where the run stops.
enum class Task { kImsrg, kHf, kMp3, kFciExport, kTransform };
/// `run.log_level`: stdout verbosity.
enum class LogLevel { kQuiet, kNormal, kDebug };
/// `basis.kind`: basis the Hamiltonian is normal ordered in.
enum class BasisKind { kHf, kOscillator, kNat };
/// `hamiltonian.kinetic`: kinetic-energy term added to the input interaction.
enum class KineticEnergy { kIntrinsic, kLab, kNone };
/// `interaction.two_body.format`.
enum class TwoBodyFormat {
  kMe2j,
  kNavratil,
  kOslo,
  kOakridge,
  kOakridgeBin,
  kTakayuki,
  kNushellx,
  kSchematicMinnesota
};
/// `interaction.three_body.type`.
enum class ThreeBodyType { kFull, kNo2b, kMono };
/// `interaction.three_body.format` (HDF5 dropped, design/00 D24).
enum class ThreeBodyFormat { kMe3j, kNavratil };
/// `interaction.three_body.no2b_precision`.
enum class No2bPrecision { kSingle, kHalf };
/// `hartree_fock.mixing`: how the HF iteration combines successive iterates.
enum class HartreeFockMixing { kDiis, kLinear };
/// `natural_orbitals.order`.
enum class NatOrder { kOccupation, kEnergy };
/// `natural_orbitals.sort`: order inside one (l, j, tz) channel.
enum class NatSort { kAbsOccupation, kSignedOccupation };
/// `run.compat`: a named compatibility profile that switches on the internal
/// options of `CompatOptions` (driver/config/compat.h; design/50 section 5).
enum class Compat { kNone, kImsrgFork79e9330 };
/// `imsrg.method`: the solver.
enum class Method {
  kMagnus,
  kMagnusBackoff,
  kMagnusModifiedEuler,
  kMagnusAdaptive,
  kFlow,
  kFlowEuler,
  kFlowRk4,
  kRestore4thOrder
};
/// `imsrg.generator.core` / `.valence`.
enum class Generator {
  kWhite,
  kAtan,
  kImaginaryTime,
  kWegner,
  kQtransferAtan,
  kShellModel,
  kShellModelAtan,
  kShellModelWegner,
  kShellModelImaginaryTime,
  kShellModelAtanNpnh,
  kHartreeFock,
  kOnePa
};
/// `imsrg.denominator.partitioning`.
enum class Partitioning { kEpsteinNesbet, kMollerPlesset, kMpIsospin };
/// `imsrg.omega.storage`.
enum class OmegaStorage { kMemory, kScratch, kDiscard };
/// File format of stored operators (Omegas, checkpoints; design/50 C6):
/// native float64 binary (exact) or ME1J + binary ME2JP (float32 two-body
/// part, design/19 section 3.1).
enum class OperatorFileKind { kNative, kMe2jp };
/// `imsrg3.variant`.
enum class Imsrg3Variant { kFull, kN7, kMp4 };
/// `output.valence_file_format`.
enum class ValenceFileFormat { kTokyo, kNushellx, kAntoine };
/// `operators.from_file[i].format`.
enum class OperatorFileFormat { kNavratil, kMiyagi };

// ---- Enum spellings
// -----------------------------------------------------------------------------

/// One enumerator with its canonical TOML spelling and phase.
template <class E>
struct EnumEntry {
  E value;
  std::string_view name;
  Phase phase = Phase::kP1;
};

/// Spellings of a config enum: specialisations provide `type_name` and
/// `entries` (all enumerators, in declaration order). Only canonical spellings
/// live here; legacy spellings are the legacy layer's business.
template <class E>
struct EnumTraits;

namespace detail {

/// An enum with `EnumTraits`. Internal: concepts stay out of the public API
/// (Google style); the functions below state it as a requires-clause.
template <class E>
concept ConfigEnum = std::is_enum_v<E> && requires {
  EnumTraits<E>::type_name;
  EnumTraits<E>::entries;
};

}  // namespace detail

/// Canonical spelling of `value` ("" if it is not an enumerator).
template <class E>
  requires detail::ConfigEnum<E>
constexpr std::string_view ToString(E value) noexcept {
  for (const auto& entry : EnumTraits<E>::entries) {
    if (entry.value == value) return entry.name;
  }
  return {};
}

/// The enumerator spelled exactly `name`, or nullopt.
template <class E>
  requires detail::ConfigEnum<E>
constexpr std::optional<E> EnumFromString(std::string_view name) noexcept {
  for (const auto& entry : EnumTraits<E>::entries) {
    if (entry.name == name) return entry.value;
  }
  return std::nullopt;
}

/// Phase of `value`.
template <class E>
  requires detail::ConfigEnum<E>
constexpr Phase EnumPhase(E value) noexcept {
  for (const auto& entry : EnumTraits<E>::entries) {
    if (entry.value == value) return entry.phase;
  }
  return Phase::kNotSupported;
}

/// All canonical spellings of `E`, in declaration order (did-you-mean
/// candidates, docs).
template <class E>
  requires detail::ConfigEnum<E>
std::vector<std::string_view> EnumNames() {
  std::vector<std::string_view> names;
  for (const auto& entry : EnumTraits<E>::entries) names.push_back(entry.name);
  return names;
}

#define IMSRG_DRIVER_ENUM_TRAITS(Enum, TypeName, ...)       \
  template <>                                               \
  struct EnumTraits<Enum> {                                 \
    static constexpr std::string_view type_name = TypeName; \
    static constexpr auto entries =                         \
        std::to_array<EnumEntry<Enum>>({__VA_ARGS__});      \
  }

IMSRG_DRIVER_ENUM_TRAITS(Task, "task", {Task::kImsrg, "imsrg"},
                         {Task::kHf, "hf"}, {Task::kMp3, "mp3"},
                         {Task::kFciExport, "fci_export", Phase::kPlanned},
                         {Task::kTransform, "transform"});
IMSRG_DRIVER_ENUM_TRAITS(LogLevel, "log level", {LogLevel::kQuiet, "quiet"},
                         {LogLevel::kNormal, "normal"},
                         {LogLevel::kDebug, "debug"});
IMSRG_DRIVER_ENUM_TRAITS(BasisKind, "basis kind", {BasisKind::kHf, "hf"},
                         {BasisKind::kOscillator, "oscillator"},
                         {BasisKind::kNat, "nat"});
IMSRG_DRIVER_ENUM_TRAITS(KineticEnergy, "kinetic energy",
                         {KineticEnergy::kIntrinsic, "intrinsic"},
                         {KineticEnergy::kLab, "lab"},
                         {KineticEnergy::kNone, "none"});
IMSRG_DRIVER_ENUM_TRAITS(
    TwoBodyFormat, "two-body format", {TwoBodyFormat::kMe2j, "me2j"},
    {TwoBodyFormat::kNavratil, "navratil", Phase::kPlanned},
    {TwoBodyFormat::kOslo, "oslo", Phase::kPlanned},
    {TwoBodyFormat::kOakridge, "oakridge", Phase::kPlanned},
    {TwoBodyFormat::kOakridgeBin, "oakridge_bin", Phase::kPlanned},
    {TwoBodyFormat::kTakayuki, "takayuki", Phase::kPlanned},
    {TwoBodyFormat::kNushellx, "nushellx", Phase::kPlanned},
    {TwoBodyFormat::kSchematicMinnesota, "schematic_minnesota",
     Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(ThreeBodyType, "three-body type",
                         {ThreeBodyType::kFull, "full"},
                         {ThreeBodyType::kNo2b, "no2b"},
                         {ThreeBodyType::kMono, "mono", Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(ThreeBodyFormat, "three-body format",
                         {ThreeBodyFormat::kMe3j, "me3j"},
                         {ThreeBodyFormat::kNavratil, "navratil",
                          Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(No2bPrecision, "NO2B precision",
                         {No2bPrecision::kSingle, "single"},
                         {No2bPrecision::kHalf, "half"});
IMSRG_DRIVER_ENUM_TRAITS(HartreeFockMixing, "Hartree-Fock mixing",
                         {HartreeFockMixing::kDiis, "diis"},
                         {HartreeFockMixing::kLinear, "linear"});
IMSRG_DRIVER_ENUM_TRAITS(Compat, "compatibility profile",
                         {Compat::kNone, "none"},
                         {Compat::kImsrgFork79e9330, "imsrg_fork_79e9330"});
IMSRG_DRIVER_ENUM_TRAITS(NatOrder, "natural-orbital order",
                         {NatOrder::kOccupation, "occupation"},
                         {NatOrder::kEnergy, "energy"});
IMSRG_DRIVER_ENUM_TRAITS(NatSort, "natural-orbital sorting",
                         {NatSort::kAbsOccupation, "abs_occupation"},
                         {NatSort::kSignedOccupation, "signed_occupation"});
IMSRG_DRIVER_ENUM_TRAITS(
    Method, "method", {Method::kMagnus, "magnus"},
    {Method::kMagnusBackoff, "magnus_backoff", Phase::kPlanned},
    {Method::kMagnusModifiedEuler, "magnus_modified_euler", Phase::kPlanned},
    {Method::kMagnusAdaptive, "magnus_adaptive", Phase::kPlanned},
    {Method::kFlow, "flow"},
    {Method::kFlowEuler, "flow_euler", Phase::kPlanned},
    {Method::kFlowRk4, "flow_rk4"},
    {Method::kRestore4thOrder, "restore_4th_order", Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(
    Generator, "generator", {Generator::kWhite, "white"},
    {Generator::kAtan, "atan"}, {Generator::kImaginaryTime, "imaginary_time"},
    {Generator::kWegner, "wegner"},
    {Generator::kQtransferAtan, "qtransfer_atan"},
    {Generator::kShellModel, "shell_model"},
    {Generator::kShellModelAtan, "shell_model_atan"},
    {Generator::kShellModelWegner, "shell_model_wegner", Phase::kPlanned},
    {Generator::kShellModelImaginaryTime, "shell_model_imaginary_time"},
    {Generator::kShellModelAtanNpnh, "shell_model_atan_npnh", Phase::kPlanned},
    {Generator::kHartreeFock, "hartree_fock"},
    {Generator::kOnePa, "one_pa", Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(Partitioning, "denominator partitioning",
                         {Partitioning::kEpsteinNesbet, "epstein_nesbet"},
                         {Partitioning::kMollerPlesset, "moller_plesset"},
                         {Partitioning::kMpIsospin, "mp_isospin",
                          Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(OperatorFileKind, "operator file format",
                         {OperatorFileKind::kNative, "native"},
                         {OperatorFileKind::kMe2jp, "me2jp"});
IMSRG_DRIVER_ENUM_TRAITS(OmegaStorage, "Omega storage",
                         {OmegaStorage::kMemory, "memory"},
                         {OmegaStorage::kScratch, "scratch", Phase::kPlanned},
                         {OmegaStorage::kDiscard, "discard", Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(Imsrg3Variant, "IMSRG(3) variant",
                         {Imsrg3Variant::kFull, "full"},
                         {Imsrg3Variant::kN7, "n7"},
                         {Imsrg3Variant::kMp4, "mp4", Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(ValenceFileFormat, "valence file format",
                         {ValenceFileFormat::kTokyo, "tokyo"},
                         {ValenceFileFormat::kNushellx, "nushellx",
                          Phase::kPlanned},
                         {ValenceFileFormat::kAntoine, "antoine",
                          Phase::kPlanned});
IMSRG_DRIVER_ENUM_TRAITS(OperatorFileFormat, "operator file format",
                         {OperatorFileFormat::kNavratil, "navratil"},
                         {OperatorFileFormat::kMiyagi, "miyagi"});

#undef IMSRG_DRIVER_ENUM_TRAITS

/// True for the shell-model generators (they need valence orbits, design/16
/// V-41, V-42).
constexpr bool IsShellModelGenerator(Generator g) noexcept {
  return g == Generator::kShellModel || g == Generator::kShellModelAtan ||
         g == Generator::kShellModelWegner ||
         g == Generator::kShellModelImaginaryTime ||
         g == Generator::kShellModelAtanNpnh;
}

/// True for the ODE flow methods (no IMSRG(3) algebra, design/16 V-46).
constexpr bool IsFlowMethod(Method m) noexcept {
  return m == Method::kFlow || m == Method::kFlowEuler ||
         m == Method::kFlowRk4 || m == Method::kRestore4thOrder;
}

// ---- Sections
// -----------------------------------------------------------------------------
// Field docs are short; the authoritative meaning, defaults and rules are in
// design/16 section 2 and in visit_schema. "Resolved" = always set after
// load_config (derived if not given).

/// `[run]`.
struct TaskConfig {
  Task task = Task::kImsrg;
  std::string scratch_dir;  ///< Omega spill directory (used iff
                            ///< imsrg.omega.storage = scratch)
  LogLevel log_level = LogLevel::kNormal;
  /// Compatibility profile (default: none = the fixed behaviour).
  Compat compat = Compat::kNone;
  friend bool operator==(const TaskConfig&, const TaskConfig&) = default;
};

/// `[basis]`: model space and single-particle basis.
struct BasisConfig {
  int emax = 0;     ///< required; e = 2n + l <= emax
  double hw = 0.0;  ///< required; oscillator frequency in MeV
  int e3max = 12;   ///< resolved to the effective value min(e3max, 3 * emax)
  std::optional<int> lmax;   ///< l cut of the whole calculation; none = no cut
  std::optional<int> lmax3;  ///< l cut of the 3N interaction; none = no cut
  std::optional<int>
      emax_unocc;  ///< emax for (l, j) unoccupied in the reference; none = emax
  int mass_number =
      0;  ///< resolved: A of reference.nucleus, else of the valence core
  BasisKind kind = BasisKind::kHf;
  friend bool operator==(const BasisConfig&, const BasisConfig&) = default;
};

/// `[reference]`.
struct ReferenceConfig {
  std::optional<std::string>
      nucleus;  ///< e.g. "O16"; none = core of the valence space
  std::optional<std::string> occupations_file;  ///< rows "n l 2j 2tz occ"
  friend bool operator==(const ReferenceConfig&,
                         const ReferenceConfig&) = default;
};

/// `[valence]`. Neither set = single-reference decoupling.
struct ValenceConfig {
  std::optional<std::string>
      space;  ///< named space ("sd-shell"), 0hw-shell, jj-shell, comma list
  std::optional<std::string>
      custom;  ///< custom orbit lists; exclusive with `space`
  /// True if a valence space is defined (valence orbits exist; stage 2
  /// possible).
  bool has_valence_space() const noexcept {
    return space.has_value() || custom.has_value();
  }
  friend bool operator==(const ValenceConfig&, const ValenceConfig&) = default;
};

/// `[hamiltonian]`: terms added to the input interaction.
struct HamiltonianConfig {
  KineticEnergy kinetic =
      KineticEnergy::kIntrinsic;  ///< resolved (nushellx -> none, trap -> lab)
  std::optional<double>
      trap_hw;  ///< lab-frame HO trap frequency (MeV); requires kinetic = lab
  double beta_cm = 0.0;  ///< Lawson term strength; active iff |beta_cm| > 1e-6
  double beta_cm_hw = 0.0;  ///< resolved: basis.hw if not given
  bool nucleon_mass_correction = false;
  bool relativistic_correction = false;  ///< planned
  friend bool operator==(const HamiltonianConfig&,
                         const HamiltonianConfig&) = default;
};

/// `[interaction.two_body]`.
struct TwoBodyInputConfig {
  std::vector<std::string>
      file;  ///< required; 1 path (2 for oakridge sp + vnn files)
  TwoBodyFormat format = TwoBodyFormat::kMe2j;
  int e1max = 0;  ///< required; single-particle emax stored in the file
  int e2max = 0;  ///< resolved: 2 * e1max if not given
  int lmax = 0;   ///< resolved: e1max if not given
  std::optional<std::string> lecs;  ///< planned (hdf5 formats only)
  friend bool operator==(const TwoBodyInputConfig&,
                         const TwoBodyInputConfig&) = default;
};

/// `[interaction.three_body]` (present iff a 3N interaction is read).
struct ThreeBodyInputConfig {
  std::string file;  ///< required
  ThreeBodyType type = ThreeBodyType::kFull;
  ThreeBodyFormat format = ThreeBodyFormat::kMe3j;
  No2bPrecision no2b_precision = No2bPrecision::kSingle;
  int e1max = 0;             ///< required
  int e3max = 0;             ///< required
  std::optional<int> e2max;  ///< none = no extra e2 cut
  bool store_pn = false;
  bool discard_no2b_after_hf = false;  ///< planned (diagnostic)
  friend bool operator==(const ThreeBodyInputConfig&,
                         const ThreeBodyInputConfig&) = default;
};

/// `[interaction]`.
struct InteractionConfig {
  TwoBodyInputConfig two_body;
  std::optional<ThreeBodyInputConfig> three_body;
  friend bool operator==(const InteractionConfig&,
                         const InteractionConfig&) = default;
};

/// `[hartree_fock]`.
struct HartreeFockConfig {
  /// Pulay DIIS in TOML; the legacy layer resolves an absent key to `kLinear`
  /// (imsrg++ density mixing, D37).
  HartreeFockMixing mixing = HartreeFockMixing::kDiis;
  bool freeze_occupations = false;
  friend bool operator==(const HartreeFockConfig&,
                         const HartreeFockConfig&) = default;
};

/// `[natural_orbitals]`.
struct NaturalOrbitalsConfig {
  bool use_occupations = false;
  NatOrder order = NatOrder::kOccupation;
  /// Order of the natural orbitals inside a channel: descending |occupation|
  /// (default) or descending signed occupation (imsrg-devel comparisons).
  NatSort sort = NatSort::kAbsOccupation;
  /// Reset the occupied natural orbitals to the HF orbitals (fork imsrg++
  /// `use_HF_reference_in_NAT`, `mbpt::UseHfForHoleStates`).
  bool use_hf_reference = false;
  friend bool operator==(const NaturalOrbitalsConfig&,
                         const NaturalOrbitalsConfig&) = default;
};

/// `[imsrg.generator]`.
struct GeneratorConfig {
  Generator core = Generator::kAtan;  ///< stage 1 (the only stage if steps = 1)
  Generator valence =
      Generator::kShellModelAtan;  ///< stage 2; reset to default if steps = 1
  int qtransfer_n = 1;             ///< exponent index of qtransfer_atan (0.5 n)
  friend bool operator==(const GeneratorConfig&,
                         const GeneratorConfig&) = default;
};

/// `[imsrg.denominator]`.
struct DenominatorConfig {
  Partitioning partitioning = Partitioning::kEpsteinNesbet;
  double delta = 0.0;                      ///< requires delta_orbit if nonzero
  std::optional<std::string> delta_orbit;  ///< "all" or an orbit label
  friend bool operator==(const DenominatorConfig&,
                         const DenominatorConfig&) = default;
};

/// `[imsrg.step]` (numerics-defining step control).
struct StepConfig {
  double smax = 200.0;
  double dsmax = 0.5;
  double ds_0 = 0.5;
  double domega = 0.1;
  double omega_norm_max = 0.25;
  double eta_criterion = 1.0e-6;
  double ode_tolerance = 1.0e-6;
  friend bool operator==(const StepConfig&, const StepConfig&) = default;
};

/// `[imsrg.omega]`.
struct OmegaConfig {
  OmegaStorage storage = OmegaStorage::kMemory;
  bool hunter_gatherer = false;
  friend bool operator==(const OmegaConfig&, const OmegaConfig&) = default;
};

/// `[imsrg.space]` (present iff any of its keys was given; all four resolved
/// then).
struct ImsrgSpaceConfig {
  int emax = 0;        ///< default basis.emax
  int e2max = 0;       ///< default 2 * emax
  int e3max = 0;       ///< default min(basis.e3max, 3 * emax)
  int emax_3body = 0;  ///< default emax
  friend bool operator==(const ImsrgSpaceConfig&,
                         const ImsrgSpaceConfig&) = default;
};

/// `[imsrg]`.
struct ImsrgConfig {
  Method method = Method::kMagnus;
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
  Imsrg3Variant variant = Imsrg3Variant::kFull;
  bool factorized = false;  ///< IMSRG(3f2): BchOptions::factorized
  /// 3f2: one-body intermediates of the factorized double commutator
  /// (`BchOptions::factorized_terms.one_body_intermediates`).
  bool factorized_1b_intermediates = true;
  /// 3f2: two-body intermediates (`...two_body_intermediates`; not
  /// implemented, E-PLANNED together with `factorized`).
  bool factorized_2b_intermediates = false;
  bool no_qqq = false;  ///< planned
  bool only_2b_eta = false;
  bool only_2b_omega = false;
  bool discard_residual_input_3n = false;
  double threebody_threshold = 0.0;
  std::optional<double> de3max;  ///< none = no energy cut (old default 99)
  std::optional<double> occ_nat3_cut;  ///< none = no cut (old default -1)
  /// True if any IMSRG(3) evaluation is requested (in the flow or at the end).
  bool active() const noexcept { return in_flow || at_end; }
  friend bool operator==(const Imsrg3Config&, const Imsrg3Config&) = default;
};

/// One `operators.from_file` entry (old `OperatorsFromFile` tag
/// `Name^J_T_P_R^file2b[^file3b]`).
struct OperatorFromFile {
  std::string name;
  int rank = 0;           ///< J; rank > 0 is planned
  int dtz = 0;            ///< -2..2
  int parity = 0;         ///< 0 or 1
  int particle_rank = 2;  ///< 1, 2 or 3
  std::string file_2b;
  std::optional<std::string> file_3b;
  OperatorFileFormat format = OperatorFileFormat::kNavratil;
  friend bool operator==(const OperatorFromFile&,
                         const OperatorFromFile&) = default;
};

/// `[operators]`.
struct OperatorsConfig {
  std::vector<OperatorRequest>
      list;  ///< transformed with the flow (macros expanded)
  std::vector<OperatorRequest> first_order;  ///< planned
  std::vector<OperatorFromFile> from_file;
  /// True if any operator is requested.
  bool any() const noexcept {
    return !list.empty() || !first_order.empty() || !from_file.empty();
  }
  friend bool operator==(const OperatorsConfig&,
                         const OperatorsConfig&) = default;
};

/// `[output]`.
struct OutputConfig {
  std::string directory = "output";
  std::string
      stem;  ///< resolved: <method>_<reference>_<valence>_hw<hw>_e<emax>_A<A>
  std::string flow_file;  ///< resolved: <directory>/<stem>.flow
  ValenceFileFormat valence_file_format = ValenceFileFormat::kTokyo;
  bool write_omega = false;
  /// Format of the Omega files of `write_omega`.
  OperatorFileKind omega_format = OperatorFileKind::kNative;
  /// With `write_omega`: also write the stack in the other format, under the
  /// stem `<stem>_me2jp` (or `<stem>_native`), for comparisons in the farm.
  bool omega_second_format = false;
  bool write_ho_ops = false;      ///< planned
  bool write_hf_ops = false;      ///< planned
  std::vector<std::string> spwf;  ///< orbit labels or "valence"
  friend bool operator==(const OutputConfig&, const OutputConfig&) = default;
};

/// `[checkpoint]`: flow checkpoints (Magnus, design/13 section 7.3).
struct CheckpointConfig {
  std::string directory;           ///< empty: no checkpoints
  double interval_minutes = 30.0;  ///< wall time between checkpoints; 0:
                                   ///< after every step
  OperatorFileKind format = OperatorFileKind::kNative;
  bool restart = false;       ///< continue from the checkpoint in `directory`
  int abort_after_steps = 0;  ///< testing aid: stop with an error right after
                              ///< the checkpoint of this flow step (0: never)
  friend bool operator==(const CheckpointConfig&,
                         const CheckpointConfig&) = default;
};

/// `[transform]`: `run.task = "transform"`, the operator farm over stored
/// Omegas (design/50 C10, design/16 section 2).
struct TransformConfig {
  std::string manifest;   ///< `<stem>_omegas.toml` of output.write_omega
  std::string work_list;  ///< file: one operator name per line (# comments)
  std::string failure_manifest;  ///< empty: <output.directory>/failed.txt
  bool skip_existing = false;    ///< skip items whose output file exists
  friend bool operator==(const TransformConfig&,
                         const TransformConfig&) = default;
};

/// Where a configuration came from (design/16 section 1.7). Written as
/// `[provenance]` into `resolved_config.toml`, ignored when that file is read
/// back, and not part of equality.
struct Provenance {
  std::string version;                   ///< imsrg version
  std::string git_hash;                  ///< "unknown" if not built from git
  std::string date;                      ///< load time, ISO 8601 UTC
  std::string command_line;              ///< original command line
  std::vector<std::string> legacy_args;  ///< legacy tokens, verbatim
  ConfigIssues warnings;  ///< warnings of the load (also printed)
  std::vector<std::string>
      applied_rules;  ///< shorthand/sentinel rewrites, human readable
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
  CheckpointConfig checkpoint;
  TransformConfig transform;
  Provenance provenance;  ///< metadata only: excluded from operator==

  /// Field-by-field equality of every section except `provenance`.
  friend bool operator==(const RunConfig& a, const RunConfig& b) {
    return a.run == b.run && a.basis == b.basis && a.reference == b.reference &&
           a.valence == b.valence && a.hamiltonian == b.hamiltonian &&
           a.interaction == b.interaction && a.hartree_fock == b.hartree_fock &&
           a.natural_orbitals == b.natural_orbitals && a.imsrg == b.imsrg &&
           a.imsrg3 == b.imsrg3 && a.operators == b.operators &&
           a.output == b.output && a.checkpoint == b.checkpoint &&
           a.transform == b.transform;
  }
};

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_CONFIG_SCHEMA_H_
