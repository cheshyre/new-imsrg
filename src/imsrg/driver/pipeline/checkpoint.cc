// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/pipeline/checkpoint.h"

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/driver/config/writer.h"
#include "imsrg/io/mexj.h"
#include "imsrg/io/operator_binary.h"

namespace imsrg::driver::pipeline {
namespace {

namespace fs = std::filesystem;

double NowSeconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// Exact text form of a double (hexfloat), read back with `ParseDouble`.
std::string Hex(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%a", x);
  return buf;
}

double ParseDouble(const std::string& s, const std::string& what) {
  char* end = nullptr;
  const double x = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0') {
    throw InputError("checkpoint state: bad number '" + s + "' for " + what);
  }
  return x;
}

std::uint64_t Fnv1a(const std::string& s) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const char ch : s) {
    h ^= static_cast<unsigned char>(ch);
    h *= 0x100000001b3ULL;
  }
  return h;
}

const char* StopReasonName(flow::StopReason r) {
  switch (r) {
    case flow::StopReason::kConverged:
      return "converged";
    case flow::StopReason::kSmaxReached:
      return "smax";
    case flow::StopReason::kDiverged:
      return "diverged";
  }
  return "smax";
}

flow::StopReason ParseStopReason(const std::string& s) {
  if (s == "converged") return flow::StopReason::kConverged;
  if (s == "diverged") return flow::StopReason::kDiverged;
  if (s == "smax") return flow::StopReason::kSmaxReached;
  throw InputError("checkpoint state: bad stop reason '" + s + "'");
}

/// Removes every file that belongs to a stored operator `stem`.
void RemoveStored(const fs::path& stem) {
  std::error_code ec;
  for (const char* ext : {".opb", ".me1j", ".me2jp.bin"}) {
    fs::path p = stem;
    p += ext;
    fs::remove(p, ec);
  }
}

std::string Index(int k) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%03d", k);
  return buf;
}

}  // namespace

std::vector<std::string> WriteStoredOperator(const fs::path& stem,
                                             const op::Operator& op,
                                             OperatorFileKind kind,
                                             std::string_view name) {
  const std::string base = stem.filename().string();
  if (kind == OperatorFileKind::kNative) {
    fs::path p = stem;
    p += ".opb";
    io::WriteOperatorBinary(p, op, name);
    return {base + ".opb"};
  }
  fs::path p1 = stem;
  p1 += ".me1j";
  fs::path p2 = stem;
  p2 += ".me2jp.bin";
  io::WriteMe1j(p1, op, name);
  io::WriteMe2jp(p2, op, name);
  return {base + ".me1j", base + ".me2jp.bin"};
}

op::Operator ReadStoredOperator(const fs::path& stem,
                                std::shared_ptr<const basis::ModelSpace> ms,
                                OperatorFileKind kind, Hermiticity h) {
  if (kind == OperatorFileKind::kNative) {
    fs::path p = stem;
    p += ".opb";
    return io::ReadOperatorBinary(p, std::move(ms));
  }
  fs::path p1 = stem;
  p1 += ".me1j";
  fs::path p2 = stem;
  p2 += ".me2jp.bin";
  return io::ReadMe1jMe2jp(p1, p2, std::move(ms), h);
}

std::uint64_t FlowConfigHash(const RunConfig& config) {
  RunConfig c = config;
  c.output = OutputConfig{};
  c.checkpoint = CheckpointConfig{};
  c.run.log_level = TaskConfig{}.log_level;
  return Fnv1a(ToToml(c, /*include_provenance=*/false));
}

FlowCheckpointer::FlowCheckpointer(const RunConfig& config, bool write_files)
    : enabled_(!config.checkpoint.directory.empty()),
      write_files_(write_files),
      dir_(config.checkpoint.directory),
      interval_s_(60.0 * config.checkpoint.interval_minutes),
      format_(config.checkpoint.format),
      abort_after_steps_(config.checkpoint.abort_after_steps),
      config_hash_(FlowConfigHash(config)),
      last_write_(NowSeconds()) {
  if (enabled_ && write_files_) fs::create_directories(dir_);
}

void FlowCheckpointer::Resume(const Checkpoint& c) {
  closed_written_ = c.omegas.size() - 1;
  h_saved_written_ =
      c.omegas.size() > 1 ? static_cast<int>(c.omegas.size()) - 1 : -1;
  // Continue the sequence after the restored one (read from state.txt).
  std::ifstream in(dir_ / "state.txt");
  std::string key;
  while (in >> key) {
    if (key == "sequence") {
      in >> sequence_;
      break;
    }
    std::string rest;
    std::getline(in, rest);
  }
  last_write_ = NowSeconds();
}

void FlowCheckpointer::OnStep(const flow::MagnusSolver& solver,
                              const flow::MagnusLoopState& state,
                              const std::vector<FinishedStage>& finished) {
  if (!enabled_) return;
  int flow_step = state.istep;
  for (const FinishedStage& f : finished) flow_step += f.result.steps;
  const bool abort_now =
      abort_after_steps_ > 0 && flow_step == abort_after_steps_;
  if (abort_now || NowSeconds() - last_write_ >= interval_s_) {
    Write(solver, state, finished);
  }
  if (abort_now) {
    throw RuntimeError(
        "checkpoint.abort_after_steps = " + std::to_string(abort_after_steps_) +
        " reached (testing aid): checkpoint written to '" + dir_.string() +
        "'");
  }
}

void FlowCheckpointer::Write(const flow::MagnusSolver& solver,
                             const flow::MagnusLoopState& state,
                             const std::vector<FinishedStage>& finished) {
  const std::vector<op::Operator>& omegas = solver.omegas();
  const std::size_t n = omegas.size();
  const int seq = sequence_ + 1;
  if (write_files_) {
    const auto t0 = NowSeconds();
    for (std::size_t k = closed_written_; k + 1 < n; ++k) {
      WriteStoredOperator(dir_ / ("omega_" + Index(static_cast<int>(k))),
                          omegas[k], format_, "Omega " + std::to_string(k));
    }
    const int cur = static_cast<int>(n) - 1;
    if (n > 1 && h_saved_written_ != cur) {
      WriteStoredOperator(dir_ / ("h_saved_" + Index(cur)),
                          solver.saved_hamiltonian(), format_, "H_saved");
    }
    WriteStoredOperator(dir_ / ("omega_cur_" + std::to_string(seq)),
                        omegas.back(), format_, "Omega (current)");
    WriteStoredOperator(dir_ / ("h_" + std::to_string(seq)),
                        solver.hamiltonian(), format_, "H(s)");
    std::ostringstream st;
    st << "# imsrg flow checkpoint v1\n";
    st << "config_hash " << config_hash_ << "\n";
    st << "format "
       << (format_ == OperatorFileKind::kNative ? "native" : "me2jp") << "\n";
    st << "sequence " << seq << "\n";
    st << "num_omegas " << n << "\n";
    const flow::SolverOptions& so = state.solver;
    st << "loop " << state.stage << " " << state.istep << " " << Hex(state.s)
       << " " << Hex(state.ds) << " " << Hex(state.e_last) << " "
       << state.loop_gathers << " " << state.num_gathers << "\n";
    st << "solver " << Hex(so.smax) << " " << Hex(so.dsmax) << " "
       << Hex(so.ds_0) << " " << Hex(so.domega) << " " << Hex(so.omega_norm_max)
       << " " << Hex(so.eta_criterion) << " " << so.magnus_adaptive << " "
       << so.hunter_gatherer << "\n";
    for (const FinishedStage& f : finished) {
      st << "stage " << f.stage << " " << f.generator << " " << f.result.steps
         << " " << Hex(f.result.s) << " " << Hex(f.result.e0) << " "
         << Hex(f.result.norm_eta) << " " << f.result.num_gathers << " "
         << StopReasonName(f.result.reason) << "\n";
    }
    st << "# readable: stage " << state.stage << " step " << state.istep
       << " s " << state.s << " E0 " << solver.hamiltonian().zero_body()
       << "\n";
    io::WriteFileAtomically(dir_ / "state.txt",
                            [&](std::ostream& out) { out << st.str(); });
    // The previous sequence's per-step files are no longer referenced.
    if (sequence_ > 0) {
      RemoveStored(dir_ / ("omega_cur_" + std::to_string(sequence_)));
      RemoveStored(dir_ / ("h_" + std::to_string(sequence_)));
    }
    log::Info(
        "checkpoint {} written to {} (stage {}, step {}, s = {}, {} Omegas, "
        "{:.1f} s)",
        seq, dir_.string(), state.stage, state.istep, state.s, n,
        NowSeconds() - t0);
  }
  closed_written_ = n - 1;
  if (n > 1) h_saved_written_ = static_cast<int>(n) - 1;
  sequence_ = seq;
  ++num_written_;
  last_write_ = NowSeconds();
}

Checkpoint ReadCheckpoint(const RunConfig& config,
                          std::shared_ptr<const basis::ModelSpace> ms) {
  const fs::path dir(config.checkpoint.directory);
  const fs::path state_path = dir / "state.txt";
  std::ifstream in(state_path);
  if (!in) {
    throw InputError("checkpoint.restart: no checkpoint state '" +
                     state_path.string() + "'");
  }
  Checkpoint c;
  std::uint64_t hash = 0;
  std::string format;
  int seq = 0;
  std::size_t n = 0;
  bool have_loop = false, have_solver = false;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    std::vector<std::string> v;
    for (std::string t; ls >> t;) v.push_back(t);
    auto need = [&](std::size_t k) {
      if (v.size() != k) {
        throw InputError("checkpoint state '" + state_path.string() +
                         "': bad line '" + line + "'");
      }
    };
    if (key == "config_hash") {
      need(1);
      hash = std::stoull(v[0]);
    } else if (key == "format") {
      need(1);
      format = v[0];
    } else if (key == "sequence") {
      need(1);
      seq = std::stoi(v[0]);
    } else if (key == "num_omegas") {
      need(1);
      n = static_cast<std::size_t>(std::stoul(v[0]));
    } else if (key == "loop") {
      need(7);
      c.loop.stage = std::stoi(v[0]);
      c.loop.istep = std::stoi(v[1]);
      c.loop.s = ParseDouble(v[2], "s");
      c.loop.ds = ParseDouble(v[3], "ds");
      c.loop.e_last = ParseDouble(v[4], "e_last");
      c.loop.loop_gathers = std::stoi(v[5]);
      c.loop.num_gathers = std::stoi(v[6]);
      have_loop = true;
    } else if (key == "solver") {
      need(8);
      flow::SolverOptions& so = c.loop.solver;
      so.smax = ParseDouble(v[0], "smax");
      so.dsmax = ParseDouble(v[1], "dsmax");
      so.ds_0 = ParseDouble(v[2], "ds_0");
      so.domega = ParseDouble(v[3], "domega");
      so.omega_norm_max = ParseDouble(v[4], "omega_norm_max");
      so.eta_criterion = ParseDouble(v[5], "eta_criterion");
      so.magnus_adaptive = v[6] == "1";
      so.hunter_gatherer = v[7] == "1";
      have_solver = true;
    } else if (key == "stage") {
      need(8);
      FinishedStage f;
      f.stage = std::stoi(v[0]);
      f.generator = v[1];
      f.result.steps = std::stoi(v[2]);
      f.result.s = ParseDouble(v[3], "stage s");
      f.result.e0 = ParseDouble(v[4], "stage e0");
      f.result.norm_eta = ParseDouble(v[5], "stage norm_eta");
      f.result.num_gathers = std::stoi(v[6]);
      f.result.reason = ParseStopReason(v[7]);
      c.stages.push_back(f);
    }
  }
  if (!have_loop || !have_solver || n == 0 || seq == 0 ||
      (format != "native" && format != "me2jp")) {
    throw InputError("checkpoint state '" + state_path.string() +
                     "' is incomplete");
  }
  if (hash != FlowConfigHash(config)) {
    throw InputError(
        "checkpoint.restart: the checkpoint in '" + dir.string() +
        "' was written with another flow configuration (config hash " +
        std::to_string(hash) + ", now " +
        std::to_string(FlowConfigHash(config)) + ")");
  }
  const OperatorFileKind kind =
      format == "native" ? OperatorFileKind::kNative : OperatorFileKind::kMe2jp;
  for (std::size_t k = 0; k + 1 < n; ++k) {
    c.omegas.push_back(
        ReadStoredOperator(dir / ("omega_" + Index(static_cast<int>(k))), ms,
                           kind, Hermiticity::kAntiHermitian));
  }
  c.omegas.push_back(
      ReadStoredOperator(dir / ("omega_cur_" + std::to_string(seq)), ms, kind,
                         Hermiticity::kAntiHermitian));
  if (n > 1) {
    c.h_saved =
        ReadStoredOperator(dir / ("h_saved_" + Index(static_cast<int>(n) - 1)),
                           ms, kind, Hermiticity::kHermitian);
  }
  c.h = ReadStoredOperator(dir / ("h_" + std::to_string(seq)), ms, kind,
                           Hermiticity::kHermitian);
  log::Info("read checkpoint {} from {}: stage {}, step {}, s = {}, {} Omegas",
            seq, dir.string(), c.loop.stage, c.loop.istep, c.loop.s, n);
  return c;
}

void WriteOmegaStack(const fs::path& dir, const std::string& stem,
                     const std::vector<op::Operator>& omegas,
                     OperatorFileKind kind, const linalg::Matrix<double>* c) {
  fs::create_directories(dir);
  const basis::ModelSpace& ms = omegas.front().model_space();
  std::vector<std::vector<std::string>> files;
  for (std::size_t k = 0; k < omegas.size(); ++k) {
    files.push_back(WriteStoredOperator(
        dir / (stem + "_omega_" + Index(static_cast<int>(k))), omegas[k], kind,
        "Omega " + std::to_string(k)));
  }
  if (c != nullptr) {
    io::WriteFileAtomically(
        dir / (stem + "_basis.bin"), [&](std::ostream& out) {
          const auto r = static_cast<std::uint64_t>(c->rows());
          const auto cc = static_cast<std::uint64_t>(c->cols());
          out.write(reinterpret_cast<const char*>(&r), 8);
          out.write(reinterpret_cast<const char*>(&cc), 8);
          for (std::int64_t j = 0; j < c->cols(); ++j) {
            for (std::int64_t i = 0; i < c->rows(); ++i) {
              const double x = (*c)(i, j);
              out.write(reinterpret_cast<const char*>(&x), 8);
            }
          }
        });
  }
  io::WriteFileAtomically(dir / (stem + "_occ.dat"), [&](std::ostream& out) {
    out << "# n l 2j 2tz occupation (IMSRG model space)\n";
    for (std::int32_t a = 0; a < ms.num_orbits(); ++a) {
      const basis::Orbit& o = ms.basis().orbit(OrbitIndex(a));
      out << o.n << " " << o.l << " " << o.j2.twice << " " << o.tz2.twice << " "
          << Hex(ms.reference().occupation(OrbitIndex(a))) << "\n";
    }
  });
  io::WriteFileAtomically(dir / (stem + "_omegas.toml"), [&](std::ostream&
                                                                 out) {
    out << "# Omega stack of an imsrg flow: apply in this order,\n"
        << "# O' = e^Om_n ... e^Om_0 O e^-Om_0 ... e^-Om_n (one BCH each)\n";
    out << "format = \""
        << (kind == OperatorFileKind::kNative ? "native" : "me2jp") << "\"\n";
    out << "emax = " << ms.basis().emax() << "\n";
    out << "e2max = " << 2 * ms.basis().emax() << "\n";
    out << "hw = " << ms.basis().hbar_omega() << "\n";
    out << "hermiticity = \"anti-hermitian\"\n";
    out << "num_omegas = " << omegas.size() << "\n";
    out << "layout_hash = \"" << io::ModelSpaceLayoutHash(ms) << "\"\n";
    out << "occupations = \"" << stem << "_occ.dat\"\n";
    if (c != nullptr)
      out << "basis_transformation = \"" << stem << "_basis.bin\"\n";
    out << "files = [\n";
    for (const auto& f : files) {
      out << "  [";
      for (std::size_t i = 0; i < f.size(); ++i)
        out << (i ? ", " : "") << "\"" << f[i] << "\"";
      out << "],\n";
    }
    out << "]\n";
  });
  log::Info("wrote {} Omega(s) to {} ({})", omegas.size(), dir.string(),
            kind == OperatorFileKind::kNative ? "native" : "me2jp");
}

}  // namespace imsrg::driver::pipeline
