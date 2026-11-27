// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/transform/transform_task.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/driver/config/compat.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/io/operator_binary.h"
#include "imsrg/io/valence_writers.h"

namespace imsrg::driver::transform {

namespace fs = std::filesystem;

namespace {

/// Checks that the occupations of `ms` equal those of the file bit for bit.
void CheckOccupations(const basis::ModelSpace& ms, const fs::path& file) {
  const std::vector<double> occ =
      OccupationsInBasis(ReadOccupations(file), ms.basis());
  for (std::int32_t a = 0; a < ms.num_orbits(); ++a) {
    if (occ[static_cast<std::size_t>(a)] !=
        ms.reference().occupation(OrbitIndex(a))) {
      throw InputError(
          "transform: the occupations of the IMSRG model space differ from '" +
          file.string() +
          "' (is the configuration the one of the flow that wrote the "
          "Omegas?)");
    }
  }
}

/// Parses one work-list name; nullopt with `error` set if it does not parse.
std::optional<OperatorRequest> ParseItem(const std::string& name,
                                         const RunConfig& config,
                                         std::string& error) {
  auto r = ParseOperatorSpec(name, ModelSpaceInfo{.emax = config.basis.emax});
  if (!r) {
    error.clear();
    for (const ConfigIssue& issue : r.error()) {
      error += (error.empty() ? "" : "; ") + issue.ToString();
    }
    return std::nullopt;
  }
  if (r.value().size() != 1) {
    error = "'" + name + "' expands to " + std::to_string(r.value().size()) +
            " operators; list them one per line";
    return std::nullopt;
  }
  return r.value().front();
}

/// Seconds since `t0`.
double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

/// The lines of a text file (empty if it cannot be read).
std::vector<std::string> ReadLines(const fs::path& path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  return lines;
}

}  // namespace

pipeline::OperatorFrame MakeFrameFromManifest(const RunConfig& config,
                                              const OmegaManifest& manifest) {
  pipeline::OperatorFrame f;
  f.ms_bare = pipeline::MakeModelSpace(config);
  const basis::SingleParticleBasis& bb = f.ms_bare->basis();
  if (manifest.basis_transformation) {
    f.c = ReadBasisTransformation(*manifest.basis_transformation);
    if (f.c->rows() != bb.num_orbits() || f.c->cols() != bb.num_orbits()) {
      throw InputError("transform: the basis transformation is " +
                       std::to_string(f.c->rows()) + " x " +
                       std::to_string(f.c->cols()) +
                       ", the basis.emax space has " +
                       std::to_string(bb.num_orbits()) + " orbits");
    }
  } else if (config.basis.kind != BasisKind::kOscillator) {
    throw InputError("transform: the manifest has no basis_transformation");
  }
  // Normal-ordering space: ms_bare with the stored occupations (the same
  // object if they are the configuration's, as in-process without HF
  // re-filling; WithModelSpace copies data unchanged either way).
  if (manifest.normal_ordering_occupations) {
    if (manifest.normal_ordering_emax &&
        *manifest.normal_ordering_emax != config.basis.emax) {
      throw InputError("transform: basis.emax differs from the manifest's");
    }
    std::vector<double> occ = OccupationsInBasis(
        ReadOccupations(*manifest.normal_ordering_occupations), bb);
    bool same = true;
    for (std::int32_t a = 0; a < bb.num_orbits(); ++a) {
      same = same && occ[static_cast<std::size_t>(a)] ==
                         f.ms_bare->reference().occupation(OrbitIndex(a));
    }
    f.ms_normal_ordering =
        same ? f.ms_bare
             : f.ms_bare->WithReference(basis::Reference::FromOccupations(
                   bb, std::move(occ), f.ms_bare->reference().name()));
  } else if (!config.imsrg.space) {
    std::vector<double> occ =
        OccupationsInBasis(ReadOccupations(manifest.occupations), bb);
    f.ms_normal_ordering =
        f.ms_bare->WithReference(basis::Reference::FromOccupations(
            bb, std::move(occ), f.ms_bare->reference().name()));
  } else {
    throw InputError(
        "transform: with [imsrg.space] the manifest must name "
        "normal_ordering_occupations (written since design/50 C10)");
  }
  f.ms_imsrg = config.imsrg.space ? pipeline::MakeImsrgModelSpace(
                                        config, *f.ms_normal_ordering)
                                  : f.ms_normal_ordering;
  const basis::ModelSpace& mi = *f.ms_imsrg;
  if (io::ModelSpaceLayoutHash(mi) != manifest.layout_hash ||
      mi.basis().emax() != manifest.emax ||
      mi.basis().hbar_omega() != manifest.hw) {
    throw InputError(
        "transform: the IMSRG model space of the configuration does not match "
        "the Omega manifest (emax, hw or layout hash)");
  }
  CheckOccupations(mi, manifest.occupations);
  if (pipeline::NeedsRenormalOrdering(mi)) {
    f.ms_core = f.ms_imsrg->WithReference(basis::Reference::FromOccupations(
        mi.basis(), pipeline::CoreOccupations(mi), "core"));
  }
  f.mass_number = config.basis.mass_number;
  f.proton_number = f.ms_bare->reference().target_Z();
  f.tensor_no.fork79e9330_diagonal_bug =
      CompatOptionsFor(config.run.compat).tensor_no_fork_diagonal_bug;
  return f;
}

commutator::CommutatorOptions TransformCommutatorOptions(
    const RunConfig& config) {
  commutator::CommutatorOptions copts;
  pipeline::ApplyCompatCommutatorOptions(config, copts);
  return copts;
}

fs::path TransformOutputPath(const TransformContext& ctx,
                             const std::string& name, bool tensor) {
  return ctx.out_dir / (ctx.stem + name + (tensor ? "_2b" : "") + ".snt");
}

fs::path TransformItem(const OperatorRequest& request,
                       const TransformContext& ctx) {
  const pipeline::OperatorFrame& frame = *ctx.frame;
  op::Operator O =
      flow::WithParticleRank(pipeline::PrepareOperator(request, frame), 2);
  // e^Omega_n ... e^Omega_0 O e^-Omega_0 ... e^-Omega_n, one BCH series per
  // Omega (flow::MagnusSolver::Transform; imsrg++ Transform_Partial).
  for (std::size_t k = 0; k < ctx.omegas->size(); ++k) {
    const op::Operator omega = ctx.omegas->Omega(k);
    O = flow::BchTransform(O, omega, ctx.bch, ctx.copts, *ctx.workspace);
  }
  op::Operator F = pipeline::FinishOperator(O, frame);
  F.SyncToHost();
  const bool tensor = !F.spec().is_scalar();
  const fs::path path =
      TransformOutputPath(ctx, request.canonical_name, tensor);
  // Written under a temporary name and renamed: an existing output is
  // complete (transform.skip_existing).
  fs::path tmp = path;
  tmp += ".tmp";
  if (tensor) {
    pipeline::WriteTensorSnt(tmp, F, *ctx.config, frame);
  } else {
    io::WriteTokyo(tmp, F,
                   pipeline::MakeValenceWriteOptions(
                       *ctx.config, frame,
                       pipeline::ScalarOperatorWriteMode(*ctx.config)));
  }
  if (ctx.config->transform.dump_native) {
    // Full precision for device-vs-host checks (the .snt has 6-9 digits);
    // written (atomically) before the .snt is renamed into place, so an
    // existing .snt (transform.skip_existing) implies its .opb.
    fs::path opb = path;
    opb.replace_extension(".opb");
    io::WriteOperatorBinary(opb, F, request.canonical_name);
  }
  fs::rename(tmp, path);
  return path;
}

TransformReport RunTransform(const RunConfig& config,
                             const comm::Communicator& comm) {
  if (config.transform.manifest.empty()) {
    throw ConfigError("run.task = \"transform\" needs transform.manifest");
  }
  if (config.imsrg3.in_flow || config.imsrg3.factorized) {
    throw ConfigError(
        "E-PLANNED: run.task = \"transform\" with IMSRG(3) or factorized "
        "BCH Omegas is not implemented yet in this build");
  }
  if (config.output.valence_file_format != ValenceFileFormat::kTokyo) {
    throw ConfigError(
        "run.task = \"transform\" writes tokyo files only "
        "(output.valence_file_format)");
  }
  const auto t_start = std::chrono::steady_clock::now();
  const OmegaManifest manifest = ReadOmegaManifest(config.transform.manifest);
  const pipeline::OperatorFrame frame = MakeFrameFromManifest(config, manifest);

  // ---- Work list and static assignment (identical on every rank).
  std::vector<std::string> names;
  for (const OperatorRequest& r : config.operators.list) {
    names.push_back(r.canonical_name);
  }
  if (!config.transform.work_list.empty()) {
    for (std::string& n : ReadWorkList(config.transform.work_list)) {
      names.push_back(std::move(n));
    }
  }
  std::vector<std::optional<OperatorRequest>> requests;
  std::vector<std::string> parse_errors(names.size());
  std::vector<double> costs;
  for (std::size_t i = 0; i < names.size(); ++i) {
    requests.push_back(ParseItem(names[i], config, parse_errors[i]));
    costs.push_back(requests.back()
                        ? ItemCost(*requests.back(), *frame.ms_imsrg,
                                   manifest.omega_stems.size())
                        : 0.0);
  }
  const std::vector<int> owner = AssignStatic(costs, comm.size());

  const fs::path out_dir(config.output.directory);
  const fs::path failure_path =
      config.transform.failure_manifest.empty()
          ? out_dir / "failed.txt"
          : fs::path(config.transform.failure_manifest);
  if (comm.is_root()) {
    fs::create_directories(out_dir);
    // Every item is "not finished" until the end: a killed job leaves a
    // complete list for resubmission.
    std::vector<Failure> pending;
    for (const std::string& n : names) {
      pending.push_back({n, "job did not complete"});
    }
    WriteFailureManifest(failure_path, pending);
  }

  // ---- Omegas once per node in shared memory.
  const comm::Communicator node = comm.SplitShared();
  const SharedOmegaStack omegas(manifest, frame.ms_imsrg, node);
  log::Info(
      "transform: {} work items on {} ranks, {} Omega(s) ({}), shared window "
      "{:.3f} GB per node, setup {:.1f} s",
      names.size(), comm.size(), omegas.size(),
      manifest.format == OperatorFileKind::kNative ? "native" : "me2jp",
      static_cast<double>(omegas.bytes()) * 1.0e-9, SecondsSince(t_start));

  commutator::Workspace workspace(frame.ms_imsrg);
  TransformContext ctx;
  ctx.config = &config;
  ctx.frame = &frame;
  ctx.omegas = &omegas;
  ctx.copts = TransformCommutatorOptions(config);
  ctx.workspace = &workspace;
  ctx.out_dir = out_dir;
  ctx.stem = config.output.stem;

  TransformReport report;
  report.num_items = names.size();
  report.omega_window_bytes = omegas.bytes();
  std::vector<std::int32_t> ok(names.size(), 0);
  std::vector<Failure> mine;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (owner[i] != comm.rank()) continue;
    ++report.num_assigned;
    if (!requests[i]) {
      mine.push_back({names[i], parse_errors[i]});
      continue;
    }
    try {
      if (config.transform.skip_existing) {
        bool done = false;
        for (const bool tensor : {false, true}) {
          const fs::path p =
              TransformOutputPath(ctx, requests[i]->canonical_name, tensor);
          fs::path opb = p;
          opb.replace_extension(".opb");
          // With dump_native an item is done only with its .opb as well
          // (a .snt of a run without dump_native does not count).
          if (fs::exists(p) &&
              (!config.transform.dump_native || fs::exists(opb))) {
            report.written.push_back(p);
            done = true;
          }
        }
        if (done) {
          ok[i] = 1;
          continue;
        }
      }
      const auto t_item = std::chrono::steady_clock::now();
      report.written.push_back(TransformItem(*requests[i], ctx));
      ok[i] = 1;
      log::Info("transform: {} done in {:.2f} s", names[i],
                SecondsSince(t_item));
    } catch (const std::exception& e) {
      mine.push_back({names[i], e.what()});
      log::Warn("transform: {} FAILED: {}", names[i], e.what());
    }
  }

  // ---- Failures: the status table by one reduction, the reasons through
  // per-rank files (only ranks with failures write one).
  comm.AllreduceSum(std::span<std::int32_t>(ok));
  fs::path rank_file = failure_path;
  rank_file += ".rank" + std::to_string(comm.rank());
  if (!mine.empty()) WriteFailureManifest(rank_file, mine);
  comm.Barrier();
  std::vector<Failure> all;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (ok[i] != 0) continue;
    std::string reason = "failed on rank " + std::to_string(owner[i]);
    if (owner[i] == comm.rank()) {
      for (const Failure& f : mine) {
        if (f.name == names[i]) reason = f.reason;
      }
    } else if (comm.is_root()) {
      fs::path rf = failure_path;
      rf += ".rank" + std::to_string(owner[i]);
      for (const std::string& line : ReadLines(rf)) {
        const auto tab = line.find('\t');
        if (tab != std::string::npos && line.substr(0, tab) == names[i]) {
          reason = line.substr(tab + 1);
        }
      }
    }
    all.push_back({names[i], reason});
  }
  if (comm.is_root()) {
    WriteFailureManifest(failure_path, all);
    for (int r = 0; r < comm.size(); ++r) {
      fs::path rf = failure_path;
      rf += ".rank" + std::to_string(r);
      std::error_code ec;
      fs::remove(rf, ec);
    }
  }
  log::Info("transform: {} of {} items finished, {} failed ({}), {:.1f} s",
            names.size() - all.size(), names.size(), all.size(),
            failure_path.string(), SecondsSince(t_start));
  report.failures = std::move(all);
  comm.Barrier();
  return report;
}

}  // namespace imsrg::driver::transform
