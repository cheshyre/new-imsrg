// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The pipeline of the `imsrg` driver (design/13 section 9, design/16 section
// 6.1, imsrg++.cc): stage 1 (pipeline/hamiltonian.cc) -> IMSRG flow (1 or 2
// Magnus stages) -> re-normal ordering for ensemble references -> transformed
// operators -> outputs (results.json, flow CSV, .snt).
#include "imsrg/driver/run.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/options.h"
#include "imsrg/driver/config/writer.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/charge_radius.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/flow/ode.h"
#include "imsrg/flow/triples.h"
#include "imsrg/io/valence_writers.h"
#include "imsrg/mbpt/energies.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/normal_ordering.h"

namespace imsrg::driver {

namespace {

namespace fs = std::filesystem;

[[noreturn]] void Planned(const std::string& what) {
  throw ConfigError("E-PLANNED: " + what +
                    " is not implemented yet in this build");
}

/// Options of the pipeline that are not implemented yet (design/16 phases):
/// fail before any work.
void RejectPlanned(const RunConfig& c) {
  if (c.run.task == Task::kMp3) Planned("run.task = \"mp3\"");
  if (c.run.task == Task::kFciExport) Planned("run.task = \"fci_export\"");
  if (c.imsrg.method != Method::kMagnus && c.imsrg.method != Method::kFlow &&
      c.imsrg.method != Method::kFlowRk4) {
    Planned("imsrg.method other than \"magnus\", \"flow\", \"flow_rk4\"");
  }
  if (c.imsrg.method != Method::kMagnus && c.imsrg.omega.hunter_gatherer) {
    throw ConfigError(
        "imsrg.omega.hunter_gatherer requires imsrg.method = \"magnus\" (the "
        "flow methods keep no Omega)");
  }
  if (c.imsrg.use_brueckner_bch) Planned("imsrg.use_brueckner_bch");
  if (c.imsrg.goose_tank) Planned("imsrg.goose_tank");
  if (c.imsrg.perturbative_triples && c.imsrg3.occ_nat3_cut) {
    // imsrg++.cc:570 switches to natural orbitals for the cut.
    Planned("imsrg3.occ_nat3_cut with imsrg.perturbative_triples");
  }
  if (c.imsrg.space) Planned("[imsrg.space] (IMSRG model-space truncation)");
  if (c.imsrg3.at_end) Planned("imsrg3.at_end");
  if (c.imsrg3.in_flow) {
    if (c.imsrg3.variant != Imsrg3Variant::kN7) {
      Planned("imsrg3.variant other than \"n7\"");
    }
    if (c.imsrg3.only_2b_omega) Planned("imsrg3.only_2b_omega");
    if (c.imsrg3.occ_nat3_cut) Planned("imsrg3.occ_nat3_cut with IMSRG(3)");
    if (c.imsrg.perturbative_triples) {
      Planned("imsrg.perturbative_triples with imsrg3.in_flow");
    }
  }
  if (c.imsrg3.factorized && c.imsrg3.factorized_2b_intermediates) {
    Planned("imsrg3.factorized_2b_intermediates");
  }
  if (c.output.valence_file_format == ValenceFileFormat::kAntoine) {
    Planned("output.valence_file_format = \"antoine\"");
  }
  if (c.output.write_omega) Planned("output.write_omega");
  if (!c.operators.first_order.empty()) Planned("operators.first_order");
  if (!c.operators.from_file.empty()) Planned("operators.from_file");
}

/// The zero-, one- and two-body parts of `O` as a particle-rank-2 operator
/// (imsrg++.cc:1177-1180, 1256-1260 throw the residual 3-body part away);
/// `O` itself if its rank is below 3.
op::Operator TwoBodyPart(const op::Operator& O) {
  if (O.spec().particle_rank < 3) return O;
  O.SyncToHost();
  op::OperatorSpec spec = O.spec();
  spec.particle_rank = 2;
  op::Operator out(O.model_space_ptr(), spec);
  out.zero_body() = O.zero_body();
  out.one_body() = O.one_body();
  out.two_body() = O.two_body();
  out.SyncToDevice();
  return out;
}

/// driver::Generator -> flow::GeneratorKind (the enums list the same
/// generators, M8a-Q1).
flow::GeneratorKind ToGeneratorKind(Generator g) {
  switch (g) {
    case Generator::kWhite:
      return flow::GeneratorKind::kWhite;
    case Generator::kAtan:
      return flow::GeneratorKind::kAtan;
    case Generator::kImaginaryTime:
      return flow::GeneratorKind::kImaginaryTime;
    case Generator::kWegner:
      return flow::GeneratorKind::kWegner;
    case Generator::kQtransferAtan:
      return flow::GeneratorKind::kQtransferAtan;
    case Generator::kShellModel:
      return flow::GeneratorKind::kShellModel;
    case Generator::kShellModelAtan:
      return flow::GeneratorKind::kShellModelAtan;
    case Generator::kShellModelWegner:
      return flow::GeneratorKind::kShellModelWegner;
    case Generator::kShellModelImaginaryTime:
      return flow::GeneratorKind::kShellModelImaginaryTime;
    case Generator::kShellModelAtanNpnh:
      return flow::GeneratorKind::kShellModelAtanNpnh;
    case Generator::kHartreeFock:
      return flow::GeneratorKind::kHartreeFock;
    case Generator::kOnePa:
      return flow::GeneratorKind::kOnePa;
  }
  throw InvariantError("unknown driver::Generator");
}

flow::GeneratorOptions MakeGeneratorOptions(const RunConfig& c, Generator g,
                                            const basis::ModelSpace& ms) {
  flow::GeneratorOptions o;
  o.kind = ToGeneratorKind(g);
  if (o.kind == flow::GeneratorKind::kShellModelAtanNpnh ||
      o.kind == flow::GeneratorKind::kOnePa) {
    Planned("generator \"" + std::string(ToString(g)) + "\"");
  }
  switch (c.imsrg.denominator.partitioning) {
    case Partitioning::kEpsteinNesbet:
      o.partitioning = flow::Partitioning::kEpsteinNesbet;
      break;
    case Partitioning::kMollerPlesset:
      o.partitioning = flow::Partitioning::kMollerPlesset;
      break;
    case Partitioning::kMpIsospin:
      o.partitioning = flow::Partitioning::kMpIsospin;
      break;
  }
  o.qtransfer_n = c.imsrg.generator.qtransfer_n;
  o.only_2b_eta = c.imsrg3.in_flow && c.imsrg3.only_2b_eta;
  o.denominator_delta = c.imsrg.denominator.delta;
  if (const auto& orb = c.imsrg.denominator.delta_orbit) {
    if (*orb == "all") {
      o.denominator_delta_all = true;
    } else {
      o.denominator_delta_orbit =
          ms.basis().index_of(basis::ParseOrbitLabel(*orb));
    }
  }
  return o;
}

/// imsrg++.cc:1080-1090 and 1135-1140: imaginary-time and Wegner generators use
/// small steps.
void SmallStepsForStiffGenerators(Generator g, flow::SolverOptions& so) {
  const std::string_view name = ToString(g);
  const bool stiff = name.find("imaginary") != std::string_view::npos ||
                     name.find("wegner") != std::string_view::npos;
  if (stiff && so.ds_0 > 1.0e-2) {
    so.ds_0 = 1.0e-4;
    so.dsmax = 1.0e-2;
  }
}

std::string Fmt17(double x) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

std::string JsonString(const std::string& s) {
  std::string out = "\"";
  for (const char ch : s) {
    if (ch == '"' || ch == '\\') out += '\\';
    out += ch;
  }
  return out + "\"";
}

const char* StopName(flow::StopReason r) {
  switch (r) {
    case flow::StopReason::kConverged:
      return "converged";
    case flow::StopReason::kSmaxReached:
      return "smax";
    case flow::StopReason::kDiverged:
      return "diverged";
  }
  return "unknown";
}

/// The flow log: a CSV file (design/13 section 7) plus one stdout line per
/// step.
class FlowLog {
 public:
  /// `three_body`: add the columns norm_eta_3b and norm_omega_3b (IMSRG(3)).
  FlowLog(const fs::path& path, bool three_body) : three_body_(three_body) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    out_.open(path);
    if (!out_)
      throw InputError("cannot write the flow file '" + path.string() + "'");
    out_ << "stage,istep,s,ds,E0,norm_H,norm_eta_1b,norm_eta_2b,"
         << (three_body_ ? "norm_eta_3b," : "")
         << "norm_eta,norm_omega_1b,norm_omega_2b,"
         << (three_body_ ? "norm_omega_3b," : "")
         << "norm_omega,num_omegas,E_MP2,wall_s\n";
    log::Info(
        "{:>5} {:>5} {:>12} {:>18} {:>14} {:>14} {:>14} {:>14} {:>4} {:>10}",
        "stage", "step", "s", "E0", "|eta_1b|", "|eta_2b|", "|Omega_1b|",
        "|Omega_2b|", "nOm", "wall(s)");
  }
  /// Gives the current Hamiltonian that the `E_MP2` column evaluates.
  void set_hamiltonian(std::function<const op::Operator&()> current) {
    current_h_ = std::move(current);
  }

  void operator()(const flow::FlowStatus& st) {
    // imsrg++'s flow file column E(MP2): the MBPT(2) correlation energy of the
    // current Hamiltonian H(s) (IMSRGSolver.cc:1273). Mp2Energy syncs its
    // argument to the host, so it works on a copy.
    double e_mp2 = std::nan("");
    if (current_h_) {
      op::Operator h = current_h_();
      e_mp2 = mbpt::Mp2Energy(h);
    }
    out_ << st.stage << ',' << st.istep << ',' << Fmt17(st.s) << ','
         << Fmt17(st.ds) << ',' << Fmt17(st.e0) << ',' << Fmt17(st.norm_h)
         << ',' << Fmt17(st.norm_eta_1b) << ',' << Fmt17(st.norm_eta_2b) << ',';
    if (three_body_) out_ << Fmt17(st.norm_eta_3b) << ',';
    out_ << Fmt17(st.norm_eta) << ',' << Fmt17(st.norm_omega_1b) << ','
         << Fmt17(st.norm_omega_2b) << ',';
    if (three_body_) out_ << Fmt17(st.norm_omega_3b) << ',';
    out_ << Fmt17(st.norm_omega) << ',' << st.num_omegas << ',' << Fmt17(e_mp2)
         << ',' << Fmt17(st.wall_seconds) << '\n';
    out_.flush();
    log::Info(
        "{:>5} {:>5} {:>12.5f} {:>18.9f} {:>14.9f} {:>14.9f} {:>14.9f} "
        "{:>14.9f} {:>4} "
        "{:>10.3f}",
        st.stage, st.istep, st.s, st.e0, st.norm_eta_1b, st.norm_eta_2b,
        st.norm_omega_1b, st.norm_omega_2b, st.num_omegas, st.wall_seconds);
  }

 private:
  bool three_body_ = false;
  std::ofstream out_;
  std::function<const op::Operator&()> current_h_;
};

/// The solver of a run: Magnus (an Omega stack, operators replayed after the
/// flow) or an ODE flow method (operators evolve alongside H, imsrg++.cc:1074).
class Engine {
 public:
  explicit Engine(flow::MagnusSolver&& s) : v_(std::move(s)) {}
  explicit Engine(flow::OdeSolver&& s) : v_(std::move(s)) {}

  bool is_magnus() const { return v_.index() == 0; }
  flow::SolveResult Solve(const flow::MagnusSolver::StatusCallback& cb) {
    return std::visit([&](auto& s) { return s.Solve(cb); }, v_);
  }
  void set_generator(const flow::GeneratorOptions& g) {
    std::visit([&](auto& s) { s.set_generator(g); }, v_);
  }
  void set_solver_options(const flow::SolverOptions& o) {
    std::visit([&](auto& s) { s.set_solver_options(o); }, v_);
  }
  const flow::SolverOptions& solver_options() const {
    return std::visit(
        [](const auto& s) -> const flow::SolverOptions& {
          return s.solver_options();
        },
        v_);
  }
  const op::Operator& hamiltonian() const {
    return std::visit(
        [](const auto& s) -> const op::Operator& { return s.hamiltonian(); },
        v_);
  }
  /// The Magnus solver (precondition: `is_magnus()`).
  flow::MagnusSolver& magnus_solver() {
    return std::get<flow::MagnusSolver>(v_);
  }
  /// Size of the Omega stack (1 for the ODE methods, which keep none).
  std::size_t num_omegas() const {
    if (const auto* m = std::get_if<flow::MagnusSolver>(&v_))
      return m->omegas().size();
    return 1;
  }
  int num_gathers() const {
    if (const auto* m = std::get_if<flow::MagnusSolver>(&v_))
      return m->num_gathers();
    return 0;
  }
  /// The i-th requested operator transformed to the end of the flow: replayed
  /// Omega stack (Magnus, `prepared` is the HF-basis operator) or the operator
  /// that flowed alongside H (ODE methods).
  op::Operator Transformed(std::size_t i, const op::Operator& prepared) {
    if (auto* m = std::get_if<flow::MagnusSolver>(&v_))
      return m->Transform(prepared);
    return std::get<flow::OdeSolver>(v_).flowing_operator(i);
  }

 private:
  std::variant<flow::MagnusSolver, flow::OdeSolver> v_;
};

struct StageRecord {
  int stage = 0;
  std::string generator;
  flow::SolveResult result;
};

/// The perturbative-triples correction of stage 1 (imsrg++.cc:1100-1108).
struct TriplesRecord {
  double de = 0.0;
  double e0_plus_de = 0.0;  ///< E0 of the stage-1 H(s) plus de.
};

struct OperatorRecord {
  std::string name;
  double reference = 0.0;
  double imsrg = 0.0;
  std::string file;
};

/// The charge radius of the output stage (D35): the zero-body parts of Rp2 and
/// Rso before the flow (HF) and after it, and the Rch2 built from them.
struct ChargeRadiusRecord {
  double rso_reference = 0.0;
  double rso_imsrg = 0.0;
  double rch2_reference = 0.0;
  double rch2_imsrg = 0.0;
};

/// Positions of Rp2 and Rso in the operator list the flow evolves.
struct ChargeRadiusSlots {
  std::size_t rp2 = 0;
  std::size_t rso = 0;
};

/// The charge radius is produced whenever the plain `Rp2` is requested (D35).
/// Finds Rp2 and Rso in `requests` and appends an Rso request if there is
/// none. Returns nullopt (and leaves `requests` alone) without a requested Rp2
/// or without protons.
std::optional<ChargeRadiusSlots> PlanChargeRadius(
    const RunConfig& config, std::vector<OperatorRequest>& requests) {
  std::optional<std::size_t> rp2;
  std::optional<std::size_t> rso;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (requests[i].kind == OperatorKind::kRp2 && !rp2) rp2 = i;
    if (requests[i].kind == OperatorKind::kRso && !rso) rso = i;
  }
  if (!rp2) return std::nullopt;
  if (!rso) {
    auto parsed =
        ParseOperatorSpec("Rso", ModelSpaceInfo{.emax = config.basis.emax});
    if (!parsed.has_value() || parsed.value().size() != 1) {
      throw InvariantError("the catalog operator Rso does not parse");
    }
    requests.push_back(parsed.value().front());
    rso = requests.size() - 1;
  }
  return ChargeRadiusSlots{*rp2, *rso};
}

/// Occupations of the core (1 for core orbits, 0 otherwise).
std::vector<double> CoreOccupations(const basis::ModelSpace& ms) {
  std::vector<double> occ(static_cast<std::size_t>(ms.num_orbits()), 0.0);
  for (const OrbitIndex a : ms.partition().core()) occ[a.ToSize()] = 1.0;
  return occ;
}

/// imsrg++.cc:1221-1277: with a valence space, re-normal order w.r.t. the core
/// if the reference used for normal ordering is not the core (ensemble /
/// targeted normal ordering).
bool NeedsRenormalOrdering(const basis::ModelSpace& ms) {
  if (ms.partition().is_single_reference()) return false;
  const std::vector<double> core = CoreOccupations(ms);
  const auto occ = ms.reference().occupations();
  for (std::size_t a = 0; a < core.size(); ++a) {
    if (std::abs(occ[a] - core[a]) > 1.0e-12) return true;
  }
  return false;
}

/// Undo the normal ordering w.r.t. the reference, then normal order w.r.t. the
/// core.
op::Operator RenormalOrderToCore(const op::Operator& O) {
  const op::Operator vacuum = op::UndoNormalOrder(O);
  const std::vector<double> core = CoreOccupations(O.model_space());
  return op::NormalOrder(vacuum, core, op::NormalOrderDirection::kToReference);
}

io::TokyoWriteOptions TokyoOptions(const RunConfig& c,
                                   const pipeline::PreparedHamiltonian& ph,
                                   io::ValenceWriteMode mode) {
  io::TokyoWriteOptions o;
  o.mode = mode;
  o.file_2n = c.interaction.two_body.file.empty()
                  ? ""
                  : c.interaction.two_body.file.front();
  o.file_3n =
      c.interaction.three_body ? c.interaction.three_body->file : "none";
  o.e1max = c.basis.emax;
  o.e2max = 2 * c.basis.emax;
  o.e3max = c.basis.e3max;
  o.hbar_omega = c.basis.hw;
  o.a_ref = static_cast<std::int32_t>(ph.ms_bare->reference().target_mass());
  o.z_ref = static_cast<std::int32_t>(ph.ms_bare->reference().target_Z());
  o.a_target = static_cast<double>(ph.mass_number);
  o.version =
      c.provenance.version.empty() ? "imsrg" : "imsrg " + c.provenance.version;
  return o;
}

void WriteValenceFile(const fs::path& path, op::Operator O, const RunConfig& c,
                      const pipeline::PreparedHamiltonian& ph,
                      io::ValenceWriteMode mode) {
  O.SyncToHost();
  const io::ValenceWriteOptions o = TokyoOptions(c, ph, mode);
  if (c.output.valence_file_format == ValenceFileFormat::kNushellx) {
    fs::path base = path;
    io::WriteNushellxInt(base.replace_extension(".int"), O, o);
    io::WriteNushellxSps(base.replace_extension(".sp"), O, o);
  } else {
    io::WriteTokyo(path, O, o);
  }
}

/// Version of the `results.json` layout (design/16 section 2.10: the schema
/// table and its changelog). Bump it with every change of the written keys.
constexpr int kResultsSchemaVersion = 3;

void WriteResults(const fs::path& path, const pipeline::PreparedHamiltonian& ph,
                  const std::vector<StageRecord>& stages, double e0,
                  std::size_t num_omegas, int num_gathers,
                  const std::vector<OperatorRecord>& ops,
                  const std::string& valence_file,
                  const std::optional<TriplesRecord>& triples = {},
                  const std::optional<ChargeRadiusRecord>& radius = {}) {
  std::ofstream out(path);
  if (!out) throw InputError("cannot write '" + path.string() + "'");
  out << "{\n  \"schema_version\": " << kResultsSchemaVersion
      << ",\n  \"e_reference\": " << Fmt17(ph.e_reference)
      << ",\n  \"e_mp2\": " << Fmt17(ph.e_mp2) << ",\n";
  if (ph.hf) {
    out << "  \"hf\": {\"e_hf\": " << Fmt17(ph.hf->e_hf)
        << ", \"e1\": " << Fmt17(ph.hf->e1) << ", \"e2\": " << Fmt17(ph.hf->e2)
        << ", \"e3\": " << Fmt17(ph.hf->e3)
        << ", \"iterations\": " << ph.hf->iterations
        << ", \"converged\": " << (ph.hf->converged ? "true" : "false")
        << "},\n";
  }
  out << "  \"imsrg\": {\"e0\": " << Fmt17(e0)
      << ", \"num_omegas\": " << num_omegas
      << ", \"num_gathers\": " << num_gathers << ", \"stages\": [";
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const StageRecord& st = stages[i];
    out << (i ? ", " : "") << "{\"stage\": " << st.stage
        << ", \"generator\": " << JsonString(st.generator)
        << ", \"steps\": " << st.result.steps
        << ", \"num_gathers\": " << st.result.num_gathers
        << ", \"s\": " << Fmt17(st.result.s)
        << ", \"e0\": " << Fmt17(st.result.e0)
        << ", \"norm_eta\": " << Fmt17(st.result.norm_eta)
        << ", \"stop\": " << JsonString(StopName(st.result.reason)) << "}";
  }
  out << "]";
  if (triples) {
    out << ", \"perturbative_triples\": {\"de\": " << Fmt17(triples->de)
        << ", \"e0_plus_de\": " << Fmt17(triples->e0_plus_de) << "}";
  }
  out << "},\n  \"operators\": {";
  for (std::size_t i = 0; i < ops.size(); ++i) {
    out << (i ? ", " : "") << JsonString(ops[i].name)
        << ": {\"reference\": " << Fmt17(ops[i].reference)
        << ", \"imsrg\": " << Fmt17(ops[i].imsrg)
        << ", \"file\": " << JsonString(ops[i].file) << "}";
  }
  out << "}";
  if (radius) {
    const auto rch = [](double rch2) {
      const std::optional<double> r = pipeline::ChargeRadius(rch2);
      return r ? Fmt17(*r) : std::string("null");
    };
    out << ",\n  \"charge_radius\": {\"rso\": {\"reference\": "
        << Fmt17(radius->rso_reference)
        << ", \"imsrg\": " << Fmt17(radius->rso_imsrg)
        << "}, \"rch2\": {\"reference\": " << Fmt17(radius->rch2_reference)
        << ", \"imsrg\": " << Fmt17(radius->rch2_imsrg)
        << "}, \"rch\": {\"reference\": " << rch(radius->rch2_reference)
        << ", \"imsrg\": " << rch(radius->rch2_imsrg) << "}}";
  }
  out << ",\n  \"valence_interaction\": "
      << (valence_file.empty() ? std::string("null") : JsonString(valence_file))
      << "\n}\n";
}

}  // namespace

void Run(const RunConfig& config) {
  const fs::path dir(config.output.directory);
  WriteResolvedConfig(config, dir / "resolved_config.toml");
  RejectPlanned(config);

  const pipeline::PreparedHamiltonian ph = pipeline::PrepareHamiltonian(config);
  if (config.run.task == Task::kHf) {
    WriteResults(dir / "results.json", ph, {}, ph.e_reference, 0, 0, {}, "");
    return;
  }
  const basis::ModelSpace& ms = ph.hno.model_space();

  // ---- IMSRG flow: stage 1 with the core generator, stage 2 (valence) with a
  // new Omega.
  const StepConfig& step = config.imsrg.step;
  flow::SolverOptions so;
  so.smax = step.smax;
  so.dsmax = step.dsmax;
  so.ds_0 = step.ds_0;
  so.domega = step.domega;
  so.omega_norm_max = step.omega_norm_max;
  so.hunter_gatherer = config.imsrg.omega.hunter_gatherer;
  so.eta_criterion = step.eta_criterion;
  const Generator core_gen = config.imsrg.generator.core;
  SmallStepsForStiffGenerators(core_gen, so);

  const bool imsrg3 = config.imsrg3.in_flow;
  FlowLog flow_log(config.output.flow_file.empty()
                       ? dir / "flow.csv"
                       : fs::path(config.output.flow_file),
                   imsrg3);
  // The ODE methods evolve the requested operators alongside H, so these are
  // built first (imsrg++.cc:1074); Magnus replays its Omega stack afterwards.
  const bool magnus = config.imsrg.method == Method::kMagnus;
  // The charge radius (D35) needs Rso next to Rp2: an Rso that was not
  // requested is evaluated internally as one more operator after the requested
  // ones (flowing alongside H for the ODE methods); it gets no file and no
  // `operators` entry.
  std::vector<OperatorRequest> requests = config.operators.list;
  const std::optional<ChargeRadiusSlots> radius_slots =
      PlanChargeRadius(config, requests);
  std::vector<op::Operator> prepared;
  std::vector<OperatorRecord> ops;
  if (!magnus) {
    for (const OperatorRequest& req : requests) {
      prepared.push_back(pipeline::PrepareOperator(req, ph, config));
      OperatorRecord rec;
      rec.name = req.canonical_name;
      rec.reference = prepared.back().zero_body();
      log::Info("{}   HF: {:.6f}", rec.name, rec.reference);
      ops.push_back(rec);
    }
  }
  flow::BchOptions bch;
  bch.factorized = config.imsrg3.factorized;
  bch.factorized_terms.one_body_intermediates =
      config.imsrg3.factorized_1b_intermediates;
  bch.factorized_terms.two_body_intermediates =
      config.imsrg3.factorized_2b_intermediates;
  const flow::GeneratorOptions core_opts =
      MakeGeneratorOptions(config, core_gen, ms);
  // IMSRG(3)-N7 (imsrg++.cc:704-725, 1048-1050): the IMSRG model space gets
  // the three-body space and H the (zero) three-body part; the MagnusSolver
  // promotes it to rank 3 because the commutator options select IMSRG(3).
  commutator::CommutatorOptions copts;
  op::Operator h_start = ph.hno;
  std::shared_ptr<const basis::ModelSpace> ms_flow = ph.hno.model_space_ptr();
  if (imsrg3) {
    basis::ThreeBodyCuts cuts;
    cuts.e3max = config.basis.e3max;
    cuts.emax_3body = -1;
    if (config.imsrg3.de3max) cuts.de3max = *config.imsrg3.de3max;
    const std::shared_ptr<const basis::ModelSpace> ms3 =
        ph.hno.model_space_ptr()->WithThreeBody(cuts);
    log::Info(
        "IMSRG(3)-N7: e3max = {}, dE3max = {}, 3-body states kept: {} out of "
        "{}",
        cuts.e3max, cuts.de3max, ms3->three_body_space()->NumRows(),
        ms3->three_body()->SumChannelKets());
    ms_flow = ms3;
    h_start = op::WithModelSpace(ph.hno, ms3);
    copts.terms = commutator::TermMask::Imsrg3N7();
    copts.threebody_threshold = config.imsrg3.threebody_threshold;
  }
  Engine solver =
      magnus ? Engine(flow::MagnusSolver(h_start, so, core_opts, bch, copts))
             : Engine(flow::OdeSolver(config.imsrg.method == Method::kFlowRk4
                                          ? flow::OdeMethod::kRk4
                                          : flow::OdeMethod::kDopri5,
                                      ph.hno, prepared, so, step.ode_tolerance,
                                      core_opts));
  flow_log.set_hamiltonian(
      [&solver]() -> const op::Operator& { return solver.hamiltonian(); });
  std::vector<StageRecord> stages;
  log::Info("Setting generator to {}", ToString(core_gen));
  stages.push_back(
      {1, std::string(ToString(core_gen)), solver.Solve(std::ref(flow_log))});

  // ---- Perturbative triples after stage 1 (imsrg++.cc:1100-1108, before the
  // second stage; H(s) and the Omega stack are those of stage 1).
  std::optional<TriplesRecord> triples;
  if (config.imsrg.perturbative_triples && magnus &&
      stages.back().result.reason != flow::StopReason::kDiverged) {
    flow::TriplesOptions to;
    if (config.imsrg3.de3max) to.de3max = *config.imsrg3.de3max;
    to.legacy_omega_order = config.imsrg3.legacy_omega_order;
    const flow::TriplesResult tr =
        flow::PerturbativeTriples(solver.magnus_solver(), to);
    const double e0_stage1 = solver.hamiltonian().zero_body();
    log::Info(
        "Perturbative triples: dE3max = {}, 3-body states kept: {} out of {}",
        to.de3max, tr.kept_rows, tr.total_kets);
    log::Info("Perturbative triples:  {:16.8f} -> {:.8f}", tr.de,
              e0_stage1 + tr.de);
    triples = TriplesRecord{tr.de, e0_stage1 + tr.de};
  }

  const bool second_stage =
      config.imsrg.steps > 1 && config.valence.has_valence_space();
  if (second_stage &&
      stages.back().result.reason != flow::StopReason::kDiverged) {
    // imsrg++.cc:1122-1143: smax doubled for the second Magnus stage only (not
    // for the flow methods), new Omega.
    const Generator val_gen = config.imsrg.generator.valence;
    // Derived from the solver's options so that the domega bump persists.
    flow::SolverOptions so2 = solver.solver_options();
    so2.smax = (magnus ? 2.0 : 1.0) * step.smax;
    if (config.imsrg.method == Method::kFlow) {
      // The flow parameter continues across the stages (M8-Q40), but imsrg++'s
      // adaptive method restarts at s = 0 and so integrates [0, smax] again
      // even when stage 1 ran to smax. The flow equation is autonomous, so
      // the equivalent range here is [s_1, s_1 + smax] (M8-Q50).
      so2.smax = stages.back().result.s + step.smax;
    }
    SmallStepsForStiffGenerators(val_gen, so2);
    log::Info("Setting generator to {}", ToString(val_gen));
    solver.set_generator(MakeGeneratorOptions(config, val_gen, ms));
    solver.set_solver_options(so2);
    stages.push_back(
        {2, std::string(ToString(val_gen)), solver.Solve(std::ref(flow_log))});
  }
  for (const StageRecord& st : stages) {
    log::Info(
        "stage {} ({}): {} steps, {} Omega gathers, s = {}, E0 = {:.9f}, "
        "stop: {}",
        st.stage, st.generator, st.result.steps, st.result.num_gathers,
        st.result.s, st.result.e0, StopName(st.result.reason));
  }
  const bool diverged =
      stages.back().result.reason == flow::StopReason::kDiverged;

  // ---- Final Hamiltonian, re-normal ordered w.r.t. the core for ensemble
  // references.
  const bool renormal = NeedsRenormalOrdering(ms);
  // IMSRG(3): the residual three-body part is dropped here, as in imsrg++.
  const op::Operator h_flow = TwoBodyPart(solver.hamiltonian());
  op::Operator h_final = renormal ? RenormalOrderToCore(h_flow) : h_flow;
  const double e0 = solver.hamiltonian().zero_body();
  if (config.valence.has_valence_space()) {
    log::Info("Core Energy = {:.6f} (valence-space zero body {:.6f})", e0,
              h_final.zero_body());
  } else {
    log::Info("Core Energy = {:.6f}", e0);
  }

  std::string valence_file;
  const std::string stem =
      config.output.stem.empty() ? "imsrg" : config.output.stem;
  if (config.valence.has_valence_space()) {
    valence_file = (dir / (stem + ".snt")).string();
    WriteValenceFile(valence_file, h_final, config, ph,
                     io::ValenceWriteMode::kHamiltonian);
  }

  // ---- Operators: HF-basis, normal-ordered operators transformed with the
  // Omega stack.
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const OperatorRequest& req = requests[i];
    op::Operator hf;
    if (magnus) {
      hf = pipeline::PrepareOperator(req, ph, config);
      OperatorRecord rec;
      rec.name = req.canonical_name;
      rec.reference = hf.zero_body();
      log::Info("{}   HF: {:.6f}", rec.name, rec.reference);
      ops.push_back(rec);
    }
    OperatorRecord& rec = ops.at(i);
    if (imsrg3 && magnus) hf = op::WithModelSpace(hf, ms_flow);
    op::Operator O = TwoBodyPart(solver.Transformed(i, hf));
    if (renormal) O = RenormalOrderToCore(O);
    rec.imsrg = O.zero_body();
    log::Info("{}   IMSRG: {:.8f}", rec.name, rec.imsrg);
    if (i >= config.operators.list.size()) continue;  // internal Rso
    rec.file = (dir / (stem + "_" + rec.name + ".snt")).string();
    WriteValenceFile(rec.file, O, config, ph,
                     io::ValenceWriteMode::kIsospinAveragedOperator);
  }
  std::optional<ChargeRadiusRecord> radius;
  if (radius_slots) {
    const OperatorRecord& rp2 = ops.at(radius_slots->rp2);
    const OperatorRecord& rso = ops.at(radius_slots->rso);
    ChargeRadiusRecord rr;
    rr.rso_reference = rso.reference;
    rr.rso_imsrg = rso.imsrg;
    rr.rch2_reference = pipeline::ChargeRadiusSquared(
        rp2.reference, rso.reference, ph.mass_number, ph.proton_number);
    rr.rch2_imsrg = pipeline::ChargeRadiusSquared(
        rp2.imsrg, rso.imsrg, ph.mass_number, ph.proton_number);
    log::Info(
        "Charge radius^2 (Rp2 + Rso/Z + nucleon terms): HF {:.6f} fm^2, "
        "IMSRG {:.6f} fm^2",
        rr.rch2_reference, rr.rch2_imsrg);
    radius = rr;
  }
  ops.resize(config.operators.list.size());  // drop the internal Rso

  WriteResults(dir / "results.json", ph, stages, e0, solver.num_omegas(),
               solver.num_gathers(), ops, valence_file, triples, radius);
  if (diverged) {
    throw NumericalError(
        "the IMSRG flow diverged (see the flow log); results.json was written");
  }
}

}  // namespace imsrg::driver
