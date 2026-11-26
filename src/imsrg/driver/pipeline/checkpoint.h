// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_PIPELINE_CHECKPOINT_H_
#define IMSRG_DRIVER_PIPELINE_CHECKPOINT_H_
/// @file
/// Stored operators (Omega dumps) and Magnus flow checkpoints (design/50 C6,
/// C7; design/13 section 7.3; keys `output.write_omega`, `[checkpoint]` of
/// design/16).
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/driver/config/schema.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/operator/operator.h"

namespace imsrg::driver::pipeline {

/// Writes `op` as `<stem>.opb` (native) or `<stem>.me1j` + `<stem>.me2jp.bin`
/// (ME1J + float32 binary ME2JP, scalar operators only). Atomic per file.
/// Returns the file names written (without directories).
std::vector<std::string> WriteStoredOperator(const std::filesystem::path& stem,
                                             const op::Operator& op,
                                             OperatorFileKind kind,
                                             std::string_view name);

/// Reads an operator written by `WriteStoredOperator` onto `ms`; for ME2JP
/// the hermiticity `h` comes from the caller (it is not in the format).
op::Operator ReadStoredOperator(const std::filesystem::path& stem,
                                std::shared_ptr<const basis::ModelSpace> ms,
                                OperatorFileKind kind, Hermiticity h);

/// FNV-1a hash of everything in `config` that determines the flow: the
/// resolved TOML without `[output]`, `[checkpoint]`, `run.log_level` and the
/// provenance. A restart refuses a checkpoint written with another hash.
std::uint64_t FlowConfigHash(const RunConfig& config);

/// A finished stage as recorded in a checkpoint.
struct FinishedStage {
  int stage = 0;
  std::string generator;
  flow::SolveResult result;
};

/// The contents of a checkpoint directory.
struct Checkpoint {
  flow::MagnusLoopState loop;         ///< State after the last saved step.
  std::vector<FinishedStage> stages;  ///< Stages completed before it.
  std::vector<op::Operator> omegas;   ///< The Omega stack, oldest first.
  op::Operator h_saved;               ///< Empty for a one-entry stack.
  op::Operator h;                     ///< H(s).
};

/// Writes Magnus checkpoints into `checkpoint.directory` (design/13 7.3):
///   * `omega_<k>.*` once for every closed Omega (k < stack size - 1),
///   * `h_saved_<k>.*` once for the H the current Omega k acts on,
///   * `omega_cur_<seq>.*`, `h_<seq>.*` (current Omega, H(s)) every time,
///   * `state.txt` last (atomically; it names the files of its sequence
///     number), then the files of the previous sequence are removed.
/// So the directory always holds one complete checkpoint. Only the root rank
/// writes (`write_files`); every rank calls `OnStep` with the same arguments.
class FlowCheckpointer {
 public:
  FlowCheckpointer(const RunConfig& config, bool write_files);
  FlowCheckpointer(const FlowCheckpointer&) = delete;
  FlowCheckpointer& operator=(const FlowCheckpointer&) = delete;
  FlowCheckpointer(FlowCheckpointer&&) = default;
  FlowCheckpointer& operator=(FlowCheckpointer&&) = default;
  ~FlowCheckpointer() = default;

  /// True if `[checkpoint]` is enabled (a directory is given).
  bool enabled() const noexcept { return enabled_; }
  /// Continues the numbering of a restored checkpoint (closed Omegas already
  /// on disk are not written again).
  void Resume(const Checkpoint& c);
  /// After every step: writes a checkpoint when `interval_minutes` of wall
  /// time have passed since the last one (always for interval 0), and throws
  /// RuntimeError right after the checkpoint of flow step
  /// `abort_after_steps` (counted over all stages; testing aid).
  void OnStep(const flow::MagnusSolver& solver,
              const flow::MagnusLoopState& state,
              const std::vector<FinishedStage>& finished);
  /// Number of checkpoints written so far by this object.
  int num_written() const noexcept { return num_written_; }

 private:
  void Write(const flow::MagnusSolver& solver,
             const flow::MagnusLoopState& state,
             const std::vector<FinishedStage>& finished);

  bool enabled_ = false;
  bool write_files_ = false;
  std::filesystem::path dir_;
  double interval_s_ = 0.0;
  OperatorFileKind format_ = OperatorFileKind::kNative;
  int abort_after_steps_ = 0;
  std::uint64_t config_hash_ = 0;
  double last_write_ = 0.0;
  int sequence_ = 0;                ///< Sequence number of the last checkpoint.
  std::size_t closed_written_ = 0;  ///< Closed Omegas already on disk.
  int h_saved_written_ = -1;        ///< Omega index whose H_saved is on disk.
  int num_written_ = 0;
};

/// Reads the checkpoint in `config.checkpoint.directory` onto `ms` (the
/// IMSRG model space of the restarted run).
/// @throws InputError if it is missing, incomplete, or was written with
///   another flow configuration (`FlowConfigHash`).
Checkpoint ReadCheckpoint(const RunConfig& config,
                          std::shared_ptr<const basis::ModelSpace> ms);

/// `output.write_omega`: writes the Omega stack as `<stem>_omega_<k>.*`, the
/// basis transformation `<stem>_basis.bin` (u64 rows, cols, then the
/// column-major C of the HO -> HF/NAT basis, f64; omitted if `c` is null),
/// the occupations `<stem>_occ.dat` (`n l 2j 2tz occupation`, IMSRG model
/// space) and the manifest `<stem>_omegas.toml` (format, emax, e2max, hw,
/// hermiticity, the files in the order to apply them, the model-space hash).
void WriteOmegaStack(const std::filesystem::path& dir, const std::string& stem,
                     const std::vector<op::Operator>& omegas,
                     OperatorFileKind kind, const linalg::Matrix<double>* c);

}  // namespace imsrg::driver::pipeline

#endif  // IMSRG_DRIVER_PIPELINE_CHECKPOINT_H_
