// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_TRANSFORM_TRANSFORM_TASK_H_
#define IMSRG_DRIVER_TRANSFORM_TRANSFORM_TASK_H_
/// @file
/// `run.task = "transform"` (design/50 C10, design/16 section 2.11b): the
/// operator farm over stored Omegas. No HF and no flow: the operator frame
/// (model spaces, C, occupations) is rebuilt from the configuration and the
/// files of `output.write_omega`, every rank transforms its static share of
/// the work list and writes the `.snt` files, and rank 0 writes the failure
/// manifest.
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/driver/config/schema.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/operator_frame.h"
#include "imsrg/driver/transform/farm.h"
#include "imsrg/driver/transform/omega_store.h"
#include "imsrg/flow/bch.h"

namespace imsrg::driver::transform {

/// The operator frame of a stored flow (what `pipeline::MakeOperatorFrame`
/// gives in-process, rebuilt without HF): `ms_bare` from the configuration,
/// C from the manifest's basis file, the normal-ordering space = `ms_bare`
/// with the stored basis.emax occupations, the IMSRG space from
/// `[imsrg.space]` (or the normal-ordering space), the core space if
/// `pipeline::NeedsRenormalOrdering`, compat options of `config.run.compat`.
/// Checks the IMSRG space against the manifest (layout hash, emax, hw,
/// occupations, bit for bit).
/// @throws InputError / ConfigError on any mismatch or missing file.
pipeline::OperatorFrame MakeFrameFromManifest(const RunConfig& config,
                                              const OmegaManifest& manifest);

/// The commutator options of the flow that wrote the Omegas (IMSRG(2) terms;
/// 9j threshold of the compat profile), as `driver::Run` builds them.
commutator::CommutatorOptions TransformCommutatorOptions(
    const RunConfig& config);

/// What one work item needs; shared read-only by the items of a rank (the
/// workspace is the rank's scratch).
struct TransformContext {
  const RunConfig* config = nullptr;
  const pipeline::OperatorFrame* frame = nullptr;
  const SharedOmegaStack* omegas = nullptr;
  flow::BchOptions bch;
  commutator::CommutatorOptions copts;
  commutator::Workspace* workspace = nullptr;
  std::filesystem::path out_dir;
  std::string stem;  ///< file prefix: `<stem><name>.snt`
};

/// File of a finished operator: `<out_dir>/<stem><name>.snt` for scalars,
/// `..._2b.snt` for tensors (the fork's `<intfile><op>` names).
std::filesystem::path TransformOutputPath(const TransformContext& ctx,
                                          const std::string& name, bool tensor);

/// The per-item computation, the unit a device version replaces: build the
/// operator at basis.emax, change basis, normal order, truncate
/// (`pipeline::PrepareOperator`), then e^Omega_k ... O ... e^-Omega_k for k
/// = 0, 1, ... with one `flow::BchTransform` each (as
/// `flow::MagnusSolver::Transform`), `pipeline::FinishOperator`, and write
/// the `.snt` (scalar writer in isospin-averaged operator mode, or the
/// tensor writer). Returns the file written. Rank-local (no collectives).
/// @throws whatever the steps throw (the farm records it as a failure).
std::filesystem::path TransformItem(const OperatorRequest& request,
                                    const TransformContext& ctx);

/// Outcome of `RunTransform` on this rank.
struct TransformReport {
  std::size_t num_items = 0;                   ///< work items in total
  std::size_t num_assigned = 0;                ///< items of this rank
  std::vector<std::filesystem::path> written;  ///< files of this rank
  std::vector<Failure> failures;       ///< all ranks' failures (every rank)
  std::size_t omega_window_bytes = 0;  ///< shared window per node
};

/// `run.task = "transform"`. Collective over `comm`: node communicators
/// (`SplitShared`), the shared Omega window, the static assignment
/// (`ItemCost`, `AssignStatic`), the items of this rank (each in its own
/// try/catch), then the failures gathered on rank 0, which writes the failure
/// manifest. Logs one line per item.
/// @throws ConfigError if `transform.manifest` is empty or the configuration
///   cannot be transformed (IMSRG(3), factorized BCH); InputError for bad
///   manifest files (on every rank).
TransformReport RunTransform(const RunConfig& config,
                             const comm::Communicator& comm);

}  // namespace imsrg::driver::transform

#endif  // IMSRG_DRIVER_TRANSFORM_TRANSFORM_TASK_H_
