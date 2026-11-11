// The pipeline of the `imsrg` driver (design/13 section 9, design/16 section 6.1, imsrg++.cc):
// stage 1 (pipeline/hamiltonian.cpp) -> IMSRG flow (1 or 2 Magnus stages) -> re-normal ordering
// for ensemble references -> transformed operators -> outputs (results.json, flow CSV, .snt).
#include "imsrg/driver/run.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "imsrg/base/error.hpp"
#include "imsrg/base/log.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/driver/config/writer.hpp"
#include "imsrg/driver/pipeline/hamiltonian.hpp"
#include "imsrg/flow/magnus.hpp"
#include "imsrg/io/valence_writers.hpp"
#include "imsrg/operator/normal_ordering.hpp"

namespace imsrg::driver {

namespace {

namespace fs = std::filesystem;

[[noreturn]] void planned(const std::string& what) {
  throw ConfigError("E-PLANNED: " + what + " is not implemented yet in this build");
}

/// Options of the pipeline that are not implemented yet (design/16 phases): fail before any work.
void reject_planned(const RunConfig& c) {
  if (c.run.task == Task::Mp3) planned("run.task = \"mp3\"");
  if (c.run.task == Task::FciExport) planned("run.task = \"fci_export\"");
  if (c.imsrg.method != Method::Magnus) planned("imsrg.method other than \"magnus\"");
  if (c.imsrg.omega.hunter_gatherer) planned("imsrg.omega.hunter_gatherer");
  if (c.imsrg.use_brueckner_bch) planned("imsrg.use_brueckner_bch");
  if (c.imsrg.goose_tank) planned("imsrg.goose_tank");
  if (c.imsrg.perturbative_triples) planned("imsrg.perturbative_triples");
  if (c.imsrg.space) planned("[imsrg.space] (IMSRG model-space truncation)");
  if (c.imsrg3.active() || c.imsrg3.factorized) planned("IMSRG(3) ([imsrg3])");
  if (c.output.valence_file_format == ValenceFileFormat::Antoine) {
    planned("output.valence_file_format = \"antoine\"");
  }
  if (c.output.write_omega) planned("output.write_omega");
  if (!c.operators.first_order.empty()) planned("operators.first_order");
  if (!c.operators.from_file.empty()) planned("operators.from_file");
}

/// driver::Generator -> flow::GeneratorKind (the enums list the same generators, M8a-Q1).
flow::GeneratorKind generator_kind(Generator g) {
  switch (g) {
    case Generator::White:
      return flow::GeneratorKind::White;
    case Generator::Atan:
      return flow::GeneratorKind::Atan;
    case Generator::ImaginaryTime:
      return flow::GeneratorKind::ImaginaryTime;
    case Generator::Wegner:
      return flow::GeneratorKind::Wegner;
    case Generator::QtransferAtan:
      return flow::GeneratorKind::QtransferAtan;
    case Generator::ShellModel:
      return flow::GeneratorKind::ShellModel;
    case Generator::ShellModelAtan:
      return flow::GeneratorKind::ShellModelAtan;
    case Generator::ShellModelWegner:
      return flow::GeneratorKind::ShellModelWegner;
    case Generator::ShellModelImaginaryTime:
      return flow::GeneratorKind::ShellModelImaginaryTime;
    case Generator::ShellModelAtanNpnh:
      return flow::GeneratorKind::ShellModelAtanNpnh;
    case Generator::HartreeFock:
      return flow::GeneratorKind::HartreeFock;
    case Generator::OnePa:
      return flow::GeneratorKind::OnePa;
  }
  throw InvariantError("unknown driver::Generator");
}

flow::GeneratorOptions generator_options(const RunConfig& c, Generator g,
                                         const basis::ModelSpace& ms) {
  flow::GeneratorOptions o;
  o.kind = generator_kind(g);
  if (o.kind == flow::GeneratorKind::ShellModelAtanNpnh || o.kind == flow::GeneratorKind::OnePa) {
    planned("generator \"" + std::string(to_string(g)) + "\"");
  }
  switch (c.imsrg.denominator.partitioning) {
    case Partitioning::EpsteinNesbet:
      o.partitioning = flow::Partitioning::EpsteinNesbet;
      break;
    case Partitioning::MollerPlesset:
      o.partitioning = flow::Partitioning::MollerPlesset;
      break;
    case Partitioning::MpIsospin:
      o.partitioning = flow::Partitioning::MpIsospin;
      break;
  }
  o.qtransfer_n = c.imsrg.generator.qtransfer_n;
  o.denominator_delta = c.imsrg.denominator.delta;
  if (const auto& orb = c.imsrg.denominator.delta_orbit) {
    if (*orb == "all") {
      o.denominator_delta_all = true;
    } else {
      o.denominator_delta_orbit = ms.basis().index_of(basis::parse_orbit_label(*orb));
    }
  }
  return o;
}

/// imsrg++.cc:1080-1090 and 1135-1140: imaginary-time and Wegner generators use small steps.
void small_steps_for_stiff_generators(Generator g, flow::SolverOptions& so) {
  const std::string_view name = to_string(g);
  const bool stiff = name.find("imaginary") != std::string_view::npos ||
                     name.find("wegner") != std::string_view::npos;
  if (stiff && so.ds_0 > 1e-2) {
    so.ds_0 = 1e-4;
    so.dsmax = 1e-2;
  }
}

std::string fmt17(double x) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char ch : s) {
    if (ch == '"' || ch == '\\') out += '\\';
    out += ch;
  }
  return out + "\"";
}

const char* stop_name(flow::StopReason r) {
  switch (r) {
    case flow::StopReason::Converged:
      return "converged";
    case flow::StopReason::SmaxReached:
      return "smax";
    case flow::StopReason::Diverged:
      return "diverged";
  }
  return "unknown";
}

/// The flow log: a CSV file (design/13 section 7) plus one stdout line per step.
class FlowLog {
 public:
  explicit FlowLog(const fs::path& path) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    out_.open(path);
    if (!out_) throw InputError("cannot write the flow file '" + path.string() + "'");
    out_ << "stage,istep,s,ds,E0,norm_H,norm_eta_1b,norm_eta_2b,norm_eta,norm_omega_1b,"
            "norm_omega_2b,norm_omega,num_omegas,wall_s\n";
    log::info("{:>5} {:>5} {:>12} {:>18} {:>14} {:>14} {:>14} {:>14} {:>4} {:>10}", "stage", "step",
              "s", "E0", "|eta_1b|", "|eta_2b|", "|Omega_1b|", "|Omega_2b|", "nOm", "wall(s)");
  }
  void operator()(const flow::FlowStatus& st) {
    out_ << st.stage << ',' << st.istep << ',' << fmt17(st.s) << ',' << fmt17(st.ds) << ','
         << fmt17(st.e0) << ',' << fmt17(st.norm_h) << ',' << fmt17(st.norm_eta_1b) << ','
         << fmt17(st.norm_eta_2b) << ',' << fmt17(st.norm_eta) << ',' << fmt17(st.norm_omega_1b)
         << ',' << fmt17(st.norm_omega_2b) << ',' << fmt17(st.norm_omega) << ',' << st.num_omegas
         << ',' << fmt17(st.wall_seconds) << '\n';
    out_.flush();
    log::info(
        "{:>5} {:>5} {:>12.5f} {:>18.9f} {:>14.9f} {:>14.9f} {:>14.9f} {:>14.9f} {:>4} "
        "{:>10.3f}",
        st.stage, st.istep, st.s, st.e0, st.norm_eta_1b, st.norm_eta_2b, st.norm_omega_1b,
        st.norm_omega_2b, st.num_omegas, st.wall_seconds);
  }

 private:
  std::ofstream out_;
};

struct StageRecord {
  int stage = 0;
  std::string generator;
  flow::SolveResult result;
};

struct OperatorRecord {
  std::string name;
  double reference = 0.0;
  double imsrg = 0.0;
  std::string file;
};

/// Occupations of the core (1 for core orbits, 0 otherwise).
std::vector<double> core_occupations(const basis::ModelSpace& ms) {
  std::vector<double> occ(static_cast<std::size_t>(ms.num_orbits()), 0.0);
  for (const OrbitIndex a : ms.partition().core()) occ[a.to_size()] = 1.0;
  return occ;
}

/// imsrg++.cc:1221-1277: with a valence space, re-normal order w.r.t. the core if the reference
/// used for normal ordering is not the core (ensemble / targeted normal ordering).
bool needs_renormal_ordering(const basis::ModelSpace& ms) {
  if (ms.partition().is_single_reference()) return false;
  const std::vector<double> core = core_occupations(ms);
  const auto occ = ms.reference().occupations();
  for (std::size_t a = 0; a < core.size(); ++a) {
    if (std::abs(occ[a] - core[a]) > 1e-12) return true;
  }
  return false;
}

/// Undo the normal ordering w.r.t. the reference, then normal order w.r.t. the core.
op::Operator renormal_order_to_core(const op::Operator& O) {
  const op::Operator vacuum = op::undo_normal_order(O);
  const std::vector<double> core = core_occupations(O.model_space());
  return op::normal_order(vacuum, core, op::NormalOrderDirection::ToReference);
}

io::TokyoWriteOptions tokyo_options(const RunConfig& c, const pipeline::PreparedHamiltonian& ph,
                                    io::ValenceWriteMode mode) {
  io::TokyoWriteOptions o;
  o.mode = mode;
  o.file_2n = c.interaction.two_body.file.empty() ? "" : c.interaction.two_body.file.front();
  o.file_3n = c.interaction.three_body ? c.interaction.three_body->file : "none";
  o.e1max = c.basis.emax;
  o.e2max = 2 * c.basis.emax;
  o.e3max = c.basis.e3max;
  o.hbar_omega = c.basis.hw;
  o.a_ref = static_cast<std::int32_t>(ph.ms_bare->reference().target_mass());
  o.z_ref = static_cast<std::int32_t>(ph.ms_bare->reference().target_Z());
  o.a_target = static_cast<double>(ph.mass_number);
  o.version = c.provenance.version.empty() ? "imsrg" : "imsrg " + c.provenance.version;
  return o;
}

void write_valence_file(const fs::path& path, op::Operator O, const RunConfig& c,
                        const pipeline::PreparedHamiltonian& ph, io::ValenceWriteMode mode) {
  O.sync_to_host();
  const io::ValenceWriteOptions o = tokyo_options(c, ph, mode);
  if (c.output.valence_file_format == ValenceFileFormat::Nushellx) {
    fs::path base = path;
    io::write_nushellx_int(base.replace_extension(".int"), O, o);
    io::write_nushellx_sps(base.replace_extension(".sp"), O, o);
  } else {
    io::write_tokyo(path, O, o);
  }
}

void write_results(const fs::path& path, const pipeline::PreparedHamiltonian& ph,
                   const std::vector<StageRecord>& stages, double e0, std::size_t num_omegas,
                   const std::vector<OperatorRecord>& ops, const std::string& valence_file) {
  std::ofstream out(path);
  if (!out) throw InputError("cannot write '" + path.string() + "'");
  out << "{\n  \"e_reference\": " << fmt17(ph.e_reference) << ",\n";
  if (ph.hf) {
    out << "  \"hf\": {\"e_hf\": " << fmt17(ph.hf->e_hf) << ", \"e1\": " << fmt17(ph.hf->e1)
        << ", \"e2\": " << fmt17(ph.hf->e2) << ", \"e3\": " << fmt17(ph.hf->e3)
        << ", \"iterations\": " << ph.hf->iterations
        << ", \"converged\": " << (ph.hf->converged ? "true" : "false") << "},\n";
  }
  out << "  \"imsrg\": {\"e0\": " << fmt17(e0) << ", \"num_omegas\": " << num_omegas
      << ", \"stages\": [";
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const StageRecord& st = stages[i];
    out << (i ? ", " : "") << "{\"stage\": " << st.stage
        << ", \"generator\": " << json_string(st.generator) << ", \"steps\": " << st.result.steps
        << ", \"s\": " << fmt17(st.result.s) << ", \"e0\": " << fmt17(st.result.e0)
        << ", \"norm_eta\": " << fmt17(st.result.norm_eta)
        << ", \"stop\": " << json_string(stop_name(st.result.reason)) << "}";
  }
  out << "]},\n  \"operators\": {";
  for (std::size_t i = 0; i < ops.size(); ++i) {
    out << (i ? ", " : "") << json_string(ops[i].name)
        << ": {\"reference\": " << fmt17(ops[i].reference) << ", \"imsrg\": " << fmt17(ops[i].imsrg)
        << ", \"file\": " << json_string(ops[i].file) << "}";
  }
  out << "},\n  \"valence_interaction\": "
      << (valence_file.empty() ? std::string("null") : json_string(valence_file)) << "\n}\n";
}

}  // namespace

void run(const RunConfig& config) {
  const fs::path dir(config.output.directory);
  write_resolved_config(config, dir / "resolved_config.toml");
  reject_planned(config);

  const pipeline::PreparedHamiltonian ph = pipeline::prepare_hamiltonian(config);
  if (config.run.task == Task::Hf) {
    write_results(dir / "results.json", ph, {}, ph.e_reference, 0, {}, "");
    return;
  }
  const basis::ModelSpace& ms = ph.hno.model_space();

  // ---- IMSRG flow: stage 1 with the core generator, stage 2 (valence) with a new Omega.
  const StepConfig& step = config.imsrg.step;
  flow::SolverOptions so;
  so.smax = step.smax;
  so.dsmax = step.dsmax;
  so.ds_0 = step.ds_0;
  so.domega = step.domega;
  so.omega_norm_max = step.omega_norm_max;
  so.eta_criterion = step.eta_criterion;
  const Generator core_gen = config.imsrg.generator.core;
  small_steps_for_stiff_generators(core_gen, so);

  FlowLog flow_log(config.output.flow_file.empty() ? dir / "flow.csv"
                                                   : fs::path(config.output.flow_file));
  flow::MagnusSolver solver(ph.hno, so, generator_options(config, core_gen, ms));
  std::vector<StageRecord> stages;
  log::info("Setting generator to {}", to_string(core_gen));
  stages.push_back({1, std::string(to_string(core_gen)), solver.solve(std::ref(flow_log))});

  const bool second_stage = config.imsrg.steps > 1 && config.valence.has_valence_space();
  if (second_stage && stages.back().result.reason != flow::StopReason::Diverged) {
    // imsrg++.cc:1122-1143: smax doubled for the second Magnus stage, new Omega.
    const Generator val_gen = config.imsrg.generator.valence;
    flow::SolverOptions so2 = so;
    so2.smax = 2.0 * step.smax;
    small_steps_for_stiff_generators(val_gen, so2);
    log::info("Setting generator to {}", to_string(val_gen));
    solver.set_generator(generator_options(config, val_gen, ms));
    solver.set_smax(so2.smax);
    stages.push_back({2, std::string(to_string(val_gen)), solver.solve(std::ref(flow_log))});
  }
  for (const StageRecord& st : stages) {
    log::info("stage {} ({}): {} steps, s = {}, E0 = {:.9f}, stop: {}", st.stage, st.generator,
              st.result.steps, st.result.s, st.result.e0, stop_name(st.result.reason));
  }
  const bool diverged = stages.back().result.reason == flow::StopReason::Diverged;

  // ---- Final Hamiltonian, re-normal ordered w.r.t. the core for ensemble references.
  const bool renormal = needs_renormal_ordering(ms);
  op::Operator h_final =
      renormal ? renormal_order_to_core(solver.hamiltonian()) : solver.hamiltonian();
  const double e0 = solver.hamiltonian().zero_body();
  if (config.valence.has_valence_space()) {
    log::info("Core Energy = {:.6f} (valence-space zero body {:.6f})", e0, h_final.zero_body());
  } else {
    log::info("Core Energy = {:.6f}", e0);
  }

  std::string valence_file;
  const std::string stem = config.output.stem.empty() ? "imsrg" : config.output.stem;
  if (config.valence.has_valence_space()) {
    valence_file = (dir / (stem + ".snt")).string();
    write_valence_file(valence_file, h_final, config, ph, io::ValenceWriteMode::Hamiltonian);
  }

  // ---- Operators: HF-basis, normal-ordered operators transformed with the Omega stack.
  std::vector<OperatorRecord> ops;
  for (const OperatorRequest& req : config.operators.list) {
    op::Operator O = pipeline::prepare_operator(req, ph, config);
    OperatorRecord rec;
    rec.name = req.canonical_name;
    rec.reference = O.zero_body();
    log::info("{}   HF: {:.6f}", rec.name, rec.reference);
    O = solver.transform(O);
    if (renormal) O = renormal_order_to_core(O);
    rec.imsrg = O.zero_body();
    log::info("{}   IMSRG: {:.8f}", rec.name, rec.imsrg);
    rec.file = (dir / (stem + "_" + rec.name + ".snt")).string();
    write_valence_file(rec.file, O, config, ph, io::ValenceWriteMode::IsospinAveragedOperator);
    ops.push_back(rec);
  }

  write_results(dir / "results.json", ph, stages, e0, solver.omegas().size(), ops, valence_file);
  if (diverged) {
    throw NumericalError("the IMSRG flow diverged (see the flow log); results.json was written");
  }
}

}  // namespace imsrg::driver
