// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_TRANSFORM_OMEGA_STORE_H_
#define IMSRG_DRIVER_TRANSFORM_OMEGA_STORE_H_
/// @file
/// Stored Omegas for `run.task = "transform"` (design/50 C10): the manifest
/// written by `pipeline::WriteOmegaStack`, the basis data stored with it, and
/// the Omega stack held ONCE per node in MPI shared memory.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/driver/config/schema.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/operator/operator.h"

namespace imsrg::driver::transform {

/// The contents of `<stem>_omegas.toml` (paths resolved against its
/// directory).
struct OmegaManifest {
  OperatorFileKind format = OperatorFileKind::kNative;
  std::int32_t emax = 0;   ///< emax of the IMSRG model space
  std::int32_t e2max = 0;  ///< 2 emax
  double hw = 0.0;
  std::uint64_t layout_hash = 0;  ///< `io::ModelSpaceLayoutHash` (IMSRG space)
  std::filesystem::path occupations;  ///< IMSRG-space occupations
  /// basis.emax space occupations (absent in manifests written before C10).
  std::optional<std::filesystem::path> normal_ordering_occupations;
  std::optional<std::int32_t> normal_ordering_emax;
  /// C(HO orbit, HF/NAT orbit); absent for the oscillator basis.
  std::optional<std::filesystem::path> basis_transformation;
  /// The Omega files without extension (`.opb`, or `.me1j` + `.me2jp.bin`),
  /// in the order to apply them.
  std::vector<std::filesystem::path> omega_stems;
};

/// Parses a manifest of `pipeline::WriteOmegaStack` (a TOML subset: `key =
/// value` lines and the `files = [ [...], ... ]` array).
/// @throws InputError for a missing file, an unknown format or a missing key.
OmegaManifest ReadOmegaManifest(const std::filesystem::path& path);

/// One line of an occupation file (`n l 2j 2tz occupation`).
struct OccupationRow {
  std::int32_t n = 0;
  std::int32_t l = 0;
  std::int32_t j2 = 0;
  std::int32_t tz2 = 0;
  double occupation = 0.0;  ///< exact (hexfloat in the file)
};

/// Reads an occupation file of `WriteOmegaStack`.
/// @throws InputError if it cannot be read or a line is malformed.
std::vector<OccupationRow> ReadOccupations(const std::filesystem::path& path);

/// The occupations of `rows` in the orbit order of `basis` (orbits missing
/// from `rows` get 0).
/// @throws InputError if a row names an orbit that `basis` lacks.
std::vector<double> OccupationsInBasis(const std::vector<OccupationRow>& rows,
                                       const basis::SingleParticleBasis& basis);

/// Reads `<stem>_basis.bin` (u64 rows, u64 cols, column-major f64).
/// @throws InputError if the file is missing or truncated.
linalg::Matrix<double> ReadBasisTransformation(
    const std::filesystem::path& path);

/// The Omega stack, held once per node in MPI shared memory: rank 0 of the
/// node communicator reads every Omega file (native or ME2JP, as the
/// manifest says) and packs [zero body, one-body buffer, two-body buffer]
/// of each into one shared window; the other ranks only wait (`Barrier`).
/// `Omega(k)` copies Omega k out of the window into an operator of the
/// calling rank (the step where a device version copies to the GPU). Memory:
/// one window of num_omegas x (1 + norb^2 + two-body buffer) doubles per
/// node, plus one materialized Omega per rank while it is used.
class SharedOmegaStack {
 public:
  /// Collective over `node` (a `Communicator::SplitShared` communicator).
  /// `ms` is the IMSRG model space the Omegas live on (layout and, for
  /// native files, reference hash are checked by the reader).
  /// @throws InputError if a file is missing or does not match `ms`.
  SharedOmegaStack(const OmegaManifest& manifest,
                   std::shared_ptr<const basis::ModelSpace> ms,
                   const comm::Communicator& node);
  SharedOmegaStack(const SharedOmegaStack&) = delete;
  SharedOmegaStack& operator=(const SharedOmegaStack&) = delete;
  SharedOmegaStack(SharedOmegaStack&&) = default;
  SharedOmegaStack& operator=(SharedOmegaStack&&) = default;
  ~SharedOmegaStack() = default;

  /// Number of Omegas.
  std::size_t size() const noexcept { return num_; }
  /// Bytes of the shared window (per node).
  std::size_t bytes() const noexcept { return memory_.size(); }
  /// Omega k (0 = first applied) as an anti-Hermitian scalar operator of
  /// particle rank 2 on the model space, synced to the device. Bit-identical
  /// to the operator the reader returned.
  op::Operator Omega(std::size_t k) const;

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  op::OperatorSpec spec_;
  std::size_t num_ = 0;
  std::size_t n1_ = 0;  ///< doubles of the one-body buffer
  std::size_t n2_ = 0;  ///< doubles of the two-body buffer
  comm::SharedMemory memory_;
};

}  // namespace imsrg::driver::transform

#endif  // IMSRG_DRIVER_TRANSFORM_OMEGA_STORE_H_
