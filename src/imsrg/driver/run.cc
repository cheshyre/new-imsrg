// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The pipeline of the `imsrg` driver (design/13 section 9, design/16 section
// 6.1, imsrg++.cc): stage 1 (pipeline/hamiltonian.cc) -> IMSRG flow (1 or 2
// Magnus stages) -> re-normal ordering for ensemble references -> transformed
// operators -> outputs (results.json, flow CSV, .snt).
#include "imsrg/driver/run.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/options.h"
#include "imsrg/driver/config/compat.h"
#include "imsrg/driver/config/writer.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/charge_radius.h"
#include "imsrg/driver/pipeline/checkpoint.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/driver/pipeline/mpi_sync.h"
#include "imsrg/driver/pipeline/operator_frame.h"
#include "imsrg/driver/pipeline/prepared_state.h"
#include "imsrg/driver/termination.h"
#include "imsrg/driver/transform/transform_task.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/flow/ode.h"
#include "imsrg/flow/triples.h"
#include "imsrg/io/valence_writers.h"
#include "imsrg/mbpt/energies.h"
#include "imsrg/mbpt/natural_orbitals.h"
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
  if (c.imsrg.space && (c.imsrg3.active() || c.imsrg.perturbative_triples)) {
    Planned("[imsrg.space] with IMSRG(3) or perturbative triples");
  }
  if (c.imsrg3.at_end) Planned("imsrg3.at_end");
  if (c.imsrg3.in_flow) {
    if (c.imsrg3.variant != Imsrg3Variant::kN7) {
      Planned("imsrg3.variant other than \"n7\"");
    }
    if (c.imsrg3.only_2b_omega) Planned("imsrg3.only_2b_omega");
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
  if (c.output.write_omega && c.imsrg.method != Method::kMagnus) {
    throw ConfigError(
        "output.write_omega requires imsrg.method = \"magnus\" (the flow "
        "methods keep no Omega)");
  }
  if (c.output.write_omega && c.imsrg3.in_flow) {
    Planned("output.write_omega with imsrg3.in_flow (three-body Omegas)");
  }
  if (!c.checkpoint.directory.empty()) {
    if (c.imsrg.method != Method::kMagnus) {
      Planned("[checkpoint] for imsrg.method other than \"magnus\"");
    }
    if (c.imsrg.omega.hunter_gatherer) {
      Planned("[checkpoint] with imsrg.omega.hunter_gatherer");
    }
    if (c.imsrg3.in_flow) Planned("[checkpoint] with imsrg3.in_flow");
    if (c.imsrg.perturbative_triples) {
      Planned("[checkpoint] with imsrg.perturbative_triples");
    }
  }
  if (c.checkpoint.restart && c.checkpoint.directory.empty()) {
    throw ConfigError("checkpoint.restart requires checkpoint.directory");
  }
  if (!c.operators.first_order.empty()) Planned("operators.first_order");
  if (!c.operators.from_file.empty()) Planned("operators.from_file");
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
  /// `write_file`: false on the ranks that write no files (every rank but rank
  /// 0): they open nothing but still evaluate every value of a step, so that
  /// collective evaluations stay matched across the ranks.
  FlowLog(const fs::path& path, bool three_body, bool write_file)
      : three_body_(three_body), write_file_(write_file) {
    if (write_file_) {
      if (path.has_parent_path()) fs::create_directories(path.parent_path());
      out_.open(path);
      if (!out_) {
        throw InputError("cannot write the flow file '" + path.string() + "'");
      }
      WriteHeader();
    }
    log::Info(
        "{:>5} {:>5} {:>12} {:>18} {:>14} {:>14} {:>14} {:>14} {:>4} {:>10}",
        "stage", "step", "s", "E0", "|eta_1b|", "|eta_2b|", "|Omega_1b|",
        "|Omega_2b|", "nOm", "wall(s)");
  }
  void WriteHeader() {
    out_ << "stage,istep,s,ds,E0,norm_H,norm_eta_1b,norm_eta_2b,"
         << (three_body_ ? "norm_eta_3b," : "")
         << "norm_eta,norm_omega_1b,norm_omega_2b,"
         << (three_body_ ? "norm_omega_3b," : "")
         << "norm_omega,num_omegas,E_MP2,wall_s\n";
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
    if (write_file_) WriteRow(st, e_mp2);
    log::Info(
        "{:>5} {:>5} {:>12.5f} {:>18.9f} {:>14.9f} {:>14.9f} {:>14.9f} "
        "{:>14.9f} {:>4} "
        "{:>10.3f}",
        st.stage, st.istep, st.s, st.e0, st.norm_eta_1b, st.norm_eta_2b,
        st.norm_omega_1b, st.norm_omega_2b, st.num_omegas, st.wall_seconds);
  }

 private:
  void WriteRow(const flow::FlowStatus& st, double e_mp2) {
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
  }

  bool three_body_ = false;
  bool write_file_ = true;
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
  /// `on_step` is used by the Magnus solver only (checkpoints).
  flow::SolveResult Solve(
      const flow::MagnusSolver::StatusCallback& cb,
      const flow::MagnusSolver::StepCallback& on_step = {}) {
    if (auto* m = std::get_if<flow::MagnusSolver>(&v_)) {
      return m->Solve(cb, on_step);
    }
    return std::get<flow::OdeSolver>(v_).Solve(cb);
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
  double e0_plus_de = 0.0;      ///< E0 of the stage-1 H(s) plus de.
  std::int64_t kept_kets = 0;   ///< 3B kets kept by the dE3max/OccNat3 cuts.
  std::int64_t total_kets = 0;  ///< The same before the cuts.
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

/// Writes `O` as a valence-space file; only rank 0 writes (the device-to-host
/// sync, a possible collective of later phases, runs on every rank first).
void WriteValenceFile(const fs::path& path, op::Operator O, const RunConfig& c,
                      const pipeline::OperatorFrame& frame,
                      io::ValenceWriteMode mode,
                      const comm::Communicator& comm) {
  O.SyncToHost();
  if (!comm.is_root()) return;
  const io::ValenceWriteOptions o =
      pipeline::MakeValenceWriteOptions(c, frame, mode);
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

/// Writes the schema version, the reference energies and the HF block.
void WriteResultsHeader(std::ostream& out,
                        const pipeline::PreparedHamiltonian& ph) {
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
}

/// Writes the `imsrg` object: energy, Omega counts, stages, triples.
void WriteImsrgBlock(std::ostream& out, const std::vector<StageRecord>& stages,
                     double e0, std::size_t num_omegas, int num_gathers,
                     const std::optional<TriplesRecord>& triples) {
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
        << ", \"e0_plus_de\": " << Fmt17(triples->e0_plus_de)
        << ", \"kept_kets\": " << triples->kept_kets
        << ", \"total_kets\": " << triples->total_kets << "}";
  }
  out << "}";
}

/// Writes the `operators` object (reference and IMSRG values, file names).
void WriteOperatorsBlock(std::ostream& out,
                         const std::vector<OperatorRecord>& ops) {
  out << ",\n  \"operators\": {";
  for (std::size_t i = 0; i < ops.size(); ++i) {
    out << (i ? ", " : "") << JsonString(ops[i].name)
        << ": {\"reference\": " << Fmt17(ops[i].reference)
        << ", \"imsrg\": " << Fmt17(ops[i].imsrg)
        << ", \"file\": " << JsonString(ops[i].file) << "}";
  }
  out << "}";
}

/// Writes the `charge_radius` object (rso, rch2 and rch, reference and IMSRG).
void WriteChargeRadiusBlock(std::ostream& out,
                            const ChargeRadiusRecord& radius) {
  const auto rch = [](double rch2) {
    const std::optional<double> r = pipeline::ChargeRadius(rch2);
    return r ? Fmt17(*r) : std::string("null");
  };
  out << ",\n  \"charge_radius\": {\"rso\": {\"reference\": "
      << Fmt17(radius.rso_reference)
      << ", \"imsrg\": " << Fmt17(radius.rso_imsrg)
      << "}, \"rch2\": {\"reference\": " << Fmt17(radius.rch2_reference)
      << ", \"imsrg\": " << Fmt17(radius.rch2_imsrg)
      << "}, \"rch\": {\"reference\": " << rch(radius.rch2_reference)
      << ", \"imsrg\": " << rch(radius.rch2_imsrg) << "}}";
}

void WriteResults(const fs::path& path, const pipeline::PreparedHamiltonian& ph,
                  const std::vector<StageRecord>& stages, double e0,
                  std::size_t num_omegas, int num_gathers,
                  const std::vector<OperatorRecord>& ops,
                  const std::string& valence_file,
                  const std::optional<TriplesRecord>& triples = {},
                  const std::optional<ChargeRadiusRecord>& radius = {}) {
  std::ofstream out(path);
  if (!out) throw InputError("cannot write '" + path.string() + "'");
  WriteResultsHeader(out, ph);
  WriteImsrgBlock(out, stages, e0, num_omegas, num_gathers, triples);
  WriteOperatorsBlock(out, ops);
  if (radius) WriteChargeRadiusBlock(out, *radius);
  out << ",\n  \"valence_interaction\": "
      << (valence_file.empty() ? std::string("null") : JsonString(valence_file))
      << "\n}\n";
}

// ---- Helpers of Run().

/// Builds the stage-1 solver options from the configured step.
flow::SolverOptions MakeSolverOptions(const RunConfig& config) {
  const StepConfig& step = config.imsrg.step;
  flow::SolverOptions so;
  so.smax = step.smax;
  so.dsmax = step.dsmax;
  so.ds_0 = step.ds_0;
  so.domega = step.domega;
  so.omega_norm_max = step.omega_norm_max;
  so.hunter_gatherer = config.imsrg.omega.hunter_gatherer;
  so.eta_criterion = step.eta_criterion;
  SmallStepsForStiffGenerators(config.imsrg.generator.core, so);
  return so;
}

/// Builds the BCH options (factorized IMSRG(3) terms) of the configuration.
flow::BchOptions MakeBchOptions(const RunConfig& config) {
  flow::BchOptions bch;
  bch.factorized = config.imsrg3.factorized;
  bch.factorized_terms.one_body_intermediates =
      config.imsrg3.factorized_1b_intermediates;
  bch.factorized_terms.two_body_intermediates =
      config.imsrg3.factorized_2b_intermediates;
  return bch;
}

/// Natural occupations n_i (one per orbit) for the OccNat3 cut
/// (imsrg++.cc:570): those of the MBPT(2) density matrix of `ph.hno`, also in
/// an HF-basis run; the NAT step's values are reused for `basis.kind = "nat"`.
/// Shared by the IMSRG(3) flow and the perturbative triples.
std::vector<double> CutOccupations(const pipeline::PreparedHamiltonian& ph) {
  if (!ph.nat_occupations.empty()) return ph.nat_occupations;
  op::Operator hno = ph.hno;  // Mbpt2DensityMatrix syncs its argument
  const linalg::Matrix<double> rho = mbpt::Mbpt2DensityMatrix(hno);
  return mbpt::ComputeNaturalOrbitals(rho, ph.hno.model_space()).occupations;
}

/// The Hamiltonian and model space the flow starts from.
struct FlowStart {
  op::Operator h_start;
  std::shared_ptr<const basis::ModelSpace> ms_flow;
};

/// Builds the flow start. IMSRG(3)-N7 (imsrg++.cc:704-725, 1048-1050): the
/// IMSRG model space gets the three-body space and H the (zero) three-body
/// part; the MagnusSolver promotes it to rank 3 because the commutator
/// options (`copts`, set here) select IMSRG(3).
FlowStart MakeFlowStart(const RunConfig& config,
                        const pipeline::PreparedHamiltonian& ph,
                        commutator::CommutatorOptions& copts) {
  FlowStart start{ph.hno, ph.hno.model_space_ptr()};
  if (config.imsrg3.in_flow) {
    basis::ThreeBodyCuts cuts;
    cuts.e3max = config.basis.e3max;
    cuts.emax_3body = -1;
    if (config.imsrg3.de3max) cuts.de3max = *config.imsrg3.de3max;
    if (config.imsrg3.occ_nat3_cut) {
      cuts.occnat3_cut = *config.imsrg3.occ_nat3_cut;
      cuts.natural_occupations = CutOccupations(ph);
    }
    const std::shared_ptr<const basis::ModelSpace> ms3 =
        ph.hno.model_space_ptr()->WithThreeBody(cuts);
    log::Info(
        "IMSRG(3)-N7: e3max = {}, dE3max = {}, OccNat3Cut = {:e}, 3-body "
        "states kept: {} out of {}",
        cuts.e3max, cuts.de3max, cuts.occnat3_cut,
        ms3->three_body_space()->NumRows(),
        ms3->three_body()->SumChannelKets());
    start.ms_flow = ms3;
    start.h_start = op::WithModelSpace(ph.hno, ms3);
    copts.terms = commutator::TermMask::Imsrg3N7();
    copts.threebody_threshold = config.imsrg3.threebody_threshold;
  }
  pipeline::ApplyCompatCommutatorOptions(config, copts);
  return start;
}

/// Prepares the requested operators of an ODE-method run (they evolve
/// alongside H) and appends their HF records to `ops`.
std::vector<op::Operator> PrepareFlowingOperators(
    const std::vector<OperatorRequest>& requests,
    const pipeline::PreparedHamiltonian& ph, const RunConfig& config,
    const comm::Communicator& comm, std::vector<OperatorRecord>& ops) {
  std::vector<op::Operator> prepared;
  const pipeline::OperatorFrame frame = pipeline::MakeOperatorFrame(ph, config);
  for (const OperatorRequest& req : requests) {
    prepared.push_back(pipeline::PrepareOperator(req, frame));
    pipeline::BroadcastOperator(prepared.back(), comm);
    OperatorRecord rec;
    rec.name = req.canonical_name;
    rec.reference = prepared.back().zero_body();
    log::Info("{}   HF: {:.6f}", rec.name, rec.reference);
    ops.push_back(rec);
  }
  return prepared;
}

/// Evaluates the perturbative triples correction of the stage-1 H(s).
TriplesRecord EvaluatePerturbativeTriples(
    Engine& solver, const pipeline::PreparedHamiltonian& ph,
    const RunConfig& config) {
  flow::TriplesOptions to;
  if (config.imsrg3.de3max) to.de3max = *config.imsrg3.de3max;
  if (config.imsrg3.occ_nat3_cut) {
    to.occnat3_cut = *config.imsrg3.occ_nat3_cut;
    to.natural_occupations = CutOccupations(ph);
  }
  const flow::TriplesResult tr =
      flow::PerturbativeTriples(solver.magnus_solver(), to);
  const double e0_stage1 = solver.hamiltonian().zero_body();
  log::Info(
      "Perturbative triples: dE3max = {}, OccNat3Cut = {:e}, 3-body states "
      "kept: {} out of {}",
      to.de3max, to.occnat3_cut, tr.kept_rows, tr.total_kets);
  log::Info("Perturbative triples:  {:16.8f} -> {:.8f}", tr.de,
            e0_stage1 + tr.de);
  return TriplesRecord{tr.de, e0_stage1 + tr.de, tr.kept_rows, tr.total_kets};
}

/// Runs stage 2 (the valence generator with a new Omega) and records it.
void RunValenceStage(Engine& solver, FlowLog& flow_log, const RunConfig& config,
                     const basis::ModelSpace& ms,
                     std::vector<StageRecord>& stages,
                     const flow::MagnusSolver::StepCallback& on_step) {
  const StepConfig& step = config.imsrg.step;
  // New Omega. D47: stage 2 integrates [s_1, s_1 + smax] for every method
  // (s continues across the stages); imsrg++ doubled smax for Magnus only
  // (imsrg++.cc:1122-1143), which left a flow_rk4 stage 2 empty after a
  // stage 1 that reached smax.
  const Generator val_gen = config.imsrg.generator.valence;
  // Derived from the solver's options so that the domega bump persists.
  flow::SolverOptions so2 = solver.solver_options();
  so2.smax = stages.back().result.s + step.smax;
  SmallStepsForStiffGenerators(val_gen, so2);
  log::Info("Setting generator to {}", ToString(val_gen));
  solver.set_generator(MakeGeneratorOptions(config, val_gen, ms));
  solver.set_solver_options(so2);
  stages.push_back({2, std::string(ToString(val_gen)),
                    solver.Solve(std::ref(flow_log), on_step)});
}

/// Transforms the requested operators to the end of the flow, writes their
/// valence files (rank 0 only) and fills the IMSRG values of `ops` (on every
/// rank). For Magnus the HF-basis operators are built on every rank and then
/// broadcast from rank 0, so that all ranks replay the Omegas on identical
/// bits.
void TransformOperators(const std::vector<OperatorRequest>& requests,
                        const pipeline::OperatorFrame& frame,
                        const RunConfig& config, Engine& solver, bool magnus,
                        const std::shared_ptr<const basis::ModelSpace>& ms_flow,
                        const fs::path& dir, const std::string& stem,
                        const comm::Communicator& comm,
                        std::vector<OperatorRecord>& ops) {
  const bool imsrg3 = config.imsrg3.in_flow;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const OperatorRequest& req = requests[i];
    op::Operator hf;
    if (magnus) {
      hf = pipeline::PrepareOperator(req, frame);
      pipeline::BroadcastOperator(hf, comm);
      OperatorRecord rec;
      rec.name = req.canonical_name;
      rec.reference = hf.zero_body();
      log::Info("{}   HF: {:.6f}", rec.name, rec.reference);
      ops.push_back(rec);
    }
    OperatorRecord& rec = ops.at(i);
    if (imsrg3 && magnus) hf = op::WithModelSpace(hf, ms_flow);
    const op::Operator O =
        pipeline::FinishOperator(solver.Transformed(i, hf), frame);
    rec.imsrg = O.zero_body();
    log::Info("{}   IMSRG: {:.8f}", rec.name, rec.imsrg);
    if (i >= config.operators.list.size()) continue;  // internal Rso
    rec.file =
        (dir / pipeline::OperatorFileName(stem, rec.name, O.spec())).string();
    if (O.spec().is_scalar()) {
      WriteValenceFile(rec.file, O, config, frame,
                       io::ValenceWriteMode::kIsospinAveragedOperator, comm);
    } else if (comm.is_root()) {
      pipeline::WriteTensorSnt(rec.file, O, config, frame);
    }
  }
}

/// Builds the charge-radius record from the Rp2 and Rso entries of `ops`.
ChargeRadiusRecord MakeChargeRadiusRecord(
    const std::vector<OperatorRecord>& ops, const ChargeRadiusSlots& slots,
    const pipeline::PreparedHamiltonian& ph) {
  const OperatorRecord& rp2 = ops.at(slots.rp2);
  const OperatorRecord& rso = ops.at(slots.rso);
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
  return rr;
}

/// Builds the stage-1 solver engine: Magnus from the flow start, or the ODE
/// solver evolving the `prepared` operators alongside H.
Engine MakeEngine(const RunConfig& config,
                  const pipeline::PreparedHamiltonian& ph,
                  const FlowStart& start,
                  const std::vector<op::Operator>& prepared,
                  const commutator::CommutatorOptions& copts,
                  const comm::Communicator& comm) {
  const flow::SolverOptions so = MakeSolverOptions(config);
  const flow::BchOptions bch = MakeBchOptions(config);
  const flow::GeneratorOptions core_opts = MakeGeneratorOptions(
      config, config.imsrg.generator.core, ph.hno.model_space());
  const bool magnus = config.imsrg.method == Method::kMagnus;
  const commutator::WorkspaceOptions wopts{.comm = comm};
  return magnus ? Engine(flow::MagnusSolver(start.h_start, so, core_opts, bch,
                                            copts, wopts))
                : Engine(flow::OdeSolver(
                      config.imsrg.method == Method::kFlowRk4
                          ? flow::OdeMethod::kRk4
                          : flow::OdeMethod::kDopri5,
                      ph.hno, prepared, so, config.imsrg.step.ode_tolerance,
                      core_opts, commutator::CommutatorOptions{}, wopts));
}

/// Runs stage 1, the perturbative triples and the optional valence stage,
/// appending the stage records and logging their summary.
/// With `[checkpoint]` every Magnus step goes through `ckpt`; a `restored`
/// checkpoint (checkpoint.restart) skips the stages it has finished and
/// continues its stage from the saved step (design/13 section 7.3).
void RunStages(Engine& solver, FlowLog& flow_log, const RunConfig& config,
               const pipeline::PreparedHamiltonian& ph,
               const basis::ModelSpace& ms, bool magnus,
               std::vector<StageRecord>& stages,
               std::optional<TriplesRecord>& triples,
               pipeline::FlowCheckpointer& ckpt,
               std::optional<pipeline::Checkpoint> restored,
               const comm::Communicator& comm) {
  const Generator core_gen = config.imsrg.generator.core;
  std::vector<pipeline::FinishedStage> finished;
  // After every Magnus step: a termination request (SIGTERM, design/00 D75)
  // on any rank stops all ranks alike (one small allreduce per step) after a
  // checkpoint; otherwise the regular checkpoint schedule.
  const flow::MagnusSolver::StepCallback on_step =
      [&](const flow::MagnusLoopState& st) {
        const std::int64_t flag = TerminationRequested() ? 1 : 0;
        if (comm.AllreduceMax(flag) > 0) {
          std::string where = "no checkpoint (checkpoint.directory is empty)";
          if (ckpt.enabled()) {
            ckpt.WriteNow(solver.magnus_solver(), st, finished);
            where = "checkpoint written to '" + config.checkpoint.directory +
                    "'; continue with checkpoint.restart = true";
          }
          throw InterruptedError("termination requested (SIGTERM) at stage " +
                                 std::to_string(st.stage) + ", step " +
                                 std::to_string(st.istep) + ": " + where);
        }
        if (ckpt.enabled()) ckpt.OnStep(solver.magnus_solver(), st, finished);
      };
  int start_stage = 1;
  if (restored) {
    start_stage = restored->loop.stage;
    for (const pipeline::FinishedStage& f : restored->stages) {
      stages.push_back({f.stage, f.generator, f.result});
    }
    finished = restored->stages;
    const Generator g =
        start_stage == 1 ? core_gen : config.imsrg.generator.valence;
    ckpt.Resume(*restored);
    solver.magnus_solver().Restore(
        restored->loop, MakeGeneratorOptions(config, g, ms),
        std::move(restored->omegas), std::move(restored->h_saved),
        std::move(restored->h));
    log::Info("Restarting stage {} ({}) from the checkpoint", start_stage,
              ToString(g));
  }
  if (start_stage == 1) {
    log::Info("Setting generator to {}", ToString(core_gen));
    stages.push_back({1, std::string(ToString(core_gen)),
                      solver.Solve(std::ref(flow_log), on_step)});
    finished.push_back(
        {stages.back().stage, stages.back().generator, stages.back().result});
  }

  // ---- Perturbative triples after stage 1 (imsrg++.cc:1100-1108, before the
  // second stage; H(s) and the Omega stack are those of stage 1).
  if (config.imsrg.perturbative_triples && magnus && start_stage == 1 &&
      stages.back().result.reason != flow::StopReason::kDiverged) {
    triples = EvaluatePerturbativeTriples(solver, ph, config);
  }

  const bool second_stage =
      config.imsrg.steps > 1 && config.valence.has_valence_space();
  if (start_stage == 2) {
    // The restored solver carries the stage-2 generator, step options and
    // Omega stack.
    stages.push_back({2, std::string(ToString(config.imsrg.generator.valence)),
                      solver.Solve(std::ref(flow_log), on_step)});
  } else if (second_stage &&
             stages.back().result.reason != flow::StopReason::kDiverged) {
    RunValenceStage(solver, flow_log, config, ms, stages, on_step);
  }
  for (const StageRecord& st : stages) {
    log::Info(
        "stage {} ({}): {} steps, {} Omega gathers, s = {}, E0 = {:.9f}, "
        "stop: {}",
        st.stage, st.generator, st.result.steps, st.result.num_gathers,
        st.result.s, st.result.e0, StopName(st.result.reason));
  }
}

/// Writes the final valence file and the transformed operators, then
/// results.json; throws NumericalError after writing if the flow diverged.
void FinishRun(const RunConfig& config, const pipeline::PreparedHamiltonian& ph,
               Engine& solver, const FlowStart& start, bool magnus,
               const std::vector<OperatorRequest>& requests,
               const std::optional<ChargeRadiusSlots>& radius_slots,
               const std::vector<StageRecord>& stages,
               const std::optional<TriplesRecord>& triples,
               const comm::Communicator& comm,
               std::vector<OperatorRecord>& ops) {
  // Output stage rule (review note M4): every value that is printed or written
  // is computed on ALL ranks, before any is_root() branch; only the file
  // writes are root-only. Never compute (norms, expectation values,
  // transformations) inside a root-only branch: in phase 2 op::Norm and the
  // gathers become collectives that every rank must call.
  const fs::path dir(config.output.directory);
  const bool diverged =
      stages.back().result.reason == flow::StopReason::kDiverged;

  // ---- Final Hamiltonian, re-normal ordered w.r.t. the core for ensemble
  // references.
  const pipeline::OperatorFrame frame = pipeline::MakeOperatorFrame(ph, config);
  // IMSRG(3): the residual three-body part is dropped here, as in imsrg++.
  const op::Operator h_final =
      pipeline::FinishOperator(solver.hamiltonian(), frame);
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
    WriteValenceFile(valence_file, h_final, config, frame,
                     io::ValenceWriteMode::kHamiltonian, comm);
  }

  // ---- Operators: HF-basis, normal-ordered operators transformed with the
  // Omega stack.
  TransformOperators(requests, frame, config, solver, magnus, start.ms_flow,
                     dir, stem, comm, ops);
  // ---- output.write_omega (design/50 C6): the Omega stack plus what a later
  // transformation needs (basis C, occupations, manifest).
  if (config.output.write_omega && comm.is_root()) {
    const linalg::Matrix<double>* c = nullptr;
    if (ph.nat) {
      c = &ph.nat->C;
    } else if (ph.hf) {
      c = &ph.hf->C;
    }
    pipeline::WriteOmegaStack(dir, stem, solver.magnus_solver().omegas(),
                              config.output.omega_format, c,
                              frame.ms_normal_ordering.get());
    if (config.output.omega_second_format) {
      const bool native =
          config.output.omega_format == OperatorFileKind::kNative;
      pipeline::WriteOmegaStack(
          dir, stem + (native ? "_me2jp" : "_native"),
          solver.magnus_solver().omegas(),
          native ? OperatorFileKind::kMe2jp : OperatorFileKind::kNative, c,
          frame.ms_normal_ordering.get());
    }
  }
  std::optional<ChargeRadiusRecord> radius;
  if (radius_slots) radius = MakeChargeRadiusRecord(ops, *radius_slots, ph);
  ops.resize(config.operators.list.size());  // drop the internal Rso

  if (comm.is_root()) {
    WriteResults(dir / "results.json", ph, stages, e0, solver.num_omegas(),
                 solver.num_gathers(), ops, valence_file, triples, radius);
  }
  if (diverged) {
    throw NumericalError(
        "the IMSRG flow diverged (see the flow log); results.json was written");
  }
}

/// Stage 1 or its stored result (design/00 D75): with checkpoint.restart and
/// a `prepared/` in the own checkpoint directory, it is read from there
/// (`from_own_checkpoint` is set); else with checkpoint.prepared_from from
/// that directory; else `PrepareHamiltonian` runs. Every rank reads the same
/// files, so all hold identical bits without a broadcast. Unless it came
/// from the own directory, the result is written to `<checkpoint
/// dir>/prepared/` (root only) for later restarts.
pipeline::PreparedHamiltonian ObtainPreparedHamiltonian(
    const RunConfig& config, const comm::Communicator& comm,
    bool& from_own_checkpoint) {
  const fs::path ckpt_dir(config.checkpoint.directory);
  const bool flow_task = config.run.task != Task::kHf;
  from_own_checkpoint = false;
  pipeline::PreparedHamiltonian ph;
  if (flow_task && config.checkpoint.restart && !ckpt_dir.empty() &&
      pipeline::HasPreparedState(ckpt_dir)) {
    ph = pipeline::ReadPreparedState(ckpt_dir, config);
    from_own_checkpoint = true;
    log::Info("restart: prepared Hamiltonian read from {}",
              pipeline::PreparedStateDirectory(ckpt_dir).string());
    return ph;
  }
  if (flow_task && !config.checkpoint.prepared_from.empty()) {
    const fs::path from(config.checkpoint.prepared_from);
    if (!pipeline::HasPreparedState(from)) {
      throw InputError(
          "checkpoint.prepared_from: no prepared Hamiltonian in '" +
          pipeline::PreparedStateDirectory(from).string() + "'");
    }
    ph = pipeline::ReadPreparedState(from, config);
    log::Info("prepared Hamiltonian read from {} (checkpoint.prepared_from)",
              pipeline::PreparedStateDirectory(from).string());
  } else {
    if (flow_task && config.checkpoint.restart) {
      log::Info(
          "restart: no prepared Hamiltonian in {}, preparing it (readers, HF, "
          "NAT)",
          pipeline::PreparedStateDirectory(ckpt_dir).string());
    }
    ph = pipeline::PrepareHamiltonian(config);
    // HF and the normal ordering use parallel reductions: the ranks agree
    // only to rounding, so all start the flow from the bits of rank 0.
    // Nothing in the driver derives the model space from NAT occupations yet
    // (imsrg3.occ_nat3_cut is E-PLANNED); when it does, broadcast it here
    // too.
    pipeline::BroadcastOperator(ph.hno, comm);
  }
  if (flow_task && !ckpt_dir.empty() && comm.is_root()) {
    pipeline::WritePreparedState(ckpt_dir, ph, config);
  }
  return ph;
}

}  // namespace

void Run(const RunConfig& config, const comm::Communicator& comm) {
  // Only rank 0 prints (Logger default) and writes files; every rank runs the
  // whole pipeline on replicated data.
  log::DefaultLogger().set_rank(comm.rank());
  const fs::path dir(config.output.directory);
  if (comm.is_root()) WriteResolvedConfig(config, dir / "resolved_config.toml");
  if (config.run.task == Task::kTransform) {
    // The operator farm over stored Omegas (design/50 C10): no HF, no flow.
    transform::RunTransform(config, comm);
    return;
  }
  RejectPlanned(config);

  bool from_own_checkpoint = false;
  pipeline::PreparedHamiltonian ph =
      ObtainPreparedHamiltonian(config, comm, from_own_checkpoint);
  if (config.run.task == Task::kHf) {
    if (comm.is_root()) {
      WriteResults(dir / "results.json", ph, {}, ph.e_reference, 0, 0, {}, "");
    }
    return;
  }
  const basis::ModelSpace& ms = ph.hno.model_space();

  // ---- IMSRG flow: stage 1 with the core generator, stage 2 (valence) with a
  // new Omega.
  FlowLog flow_log(config.output.flow_file.empty()
                       ? dir / "flow.csv"
                       : fs::path(config.output.flow_file),
                   config.imsrg3.in_flow, comm.is_root());
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
    prepared = PrepareFlowingOperators(requests, ph, config, comm, ops);
  }
  commutator::CommutatorOptions copts;
  const FlowStart start = MakeFlowStart(config, ph, copts);
  Engine solver = MakeEngine(config, ph, start, prepared, copts, comm);
  flow_log.set_hamiltonian(
      [&solver]() -> const op::Operator& { return solver.hamiltonian(); });
  std::vector<StageRecord> stages;
  std::optional<TriplesRecord> triples;
  pipeline::FlowCheckpointer ckpt(config, comm.is_root());
  std::optional<pipeline::Checkpoint> restored;
  if (config.checkpoint.restart) {
    if (from_own_checkpoint &&
        !fs::exists(fs::path(config.checkpoint.directory) / "state.txt")) {
      // Killed after the preparation but before the first flow checkpoint.
      log::Info(
          "restart: no flow checkpoint in {} yet, starting the flow from the "
          "prepared Hamiltonian",
          config.checkpoint.directory);
    } else {
      // Every rank reads the same files (identical bits on all ranks).
      restored = pipeline::ReadCheckpoint(config, start.ms_flow);
    }
  }
  RunStages(solver, flow_log, config, ph, ms, magnus, stages, triples, ckpt,
            std::move(restored), comm);
  FinishRun(config, ph, solver, start, magnus, requests, radius_slots, stages,
            triples, comm, ops);
}

}  // namespace imsrg::driver
