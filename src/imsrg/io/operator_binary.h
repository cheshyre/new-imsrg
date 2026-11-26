// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_IO_OPERATOR_BINARY_H_
#define IMSRG_IO_OPERATOR_BINARY_H_
/// @file
/// Native float64 binary file of one operator (design/19 section 3.2): the
/// exact storage of Omegas and flow checkpoints (design/50 C6/C7). A
/// self-describing header, the raw storage buffers and a checksum; written
/// atomically (temporary file + rename). Little-endian targets only.
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::io {

/// FNV-1a 64 hash of the storage layout of a model space: the orbits (n, l,
/// 2j, 2tz) in index order, hw (bit pattern) and the two-body channels (2J,
/// parity, 2Tz, kets as orbit pairs). Independent of the occupations. Two
/// model spaces with the same hash give operators with identical buffers.
std::uint64_t ModelSpaceLayoutHash(const basis::ModelSpace& ms);

/// FNV-1a 64 hash of the reference occupations (bit patterns, index order).
std::uint64_t ReferenceHash(const basis::ModelSpace& ms);

/// Header of a native operator file.
struct OperatorFileHeader {
  std::uint32_t version = 1;        ///< Format version.
  std::uint64_t layout_hash = 0;    ///< `ModelSpaceLayoutHash`.
  std::uint64_t reference_hash = 0;  ///< `ReferenceHash`.
  op::OperatorSpec spec;            ///< Rank, symmetry, hermiticity.
  std::int32_t emax = 0;            ///< Basis emax.
  std::int32_t e2max = 0;  ///< 2 emax (the basis has no e2max cut).
  double hw = 0.0;         ///< Oscillator frequency, MeV.
  std::string label;       ///< Free text (operator name, stage, ...).
  std::uint64_t one_body_count = 0;  ///< Doubles of the one-body buffer.
  std::uint64_t two_body_count = 0;  ///< Doubles of the two-body buffer.
};

/// Writes `op` (particle rank <= 2) to `path`. Layout (little-endian):
/// magic "IMSRGOPB", u32 version, u32 header length, the header fields (u64
/// hashes, i32 particle_rank, rank_J, parity, delta_Tz, hermiticity,
/// delta_particle_number, emax, e2max, f64 hw, u32 label length + bytes, u64
/// counts), then f64 zero body, the one-body and the two-body buffer (raw, as
/// stored), then a u64 FNV-1a checksum of everything after the magic.
/// Written to `<path>.tmp.<pid>` and renamed onto `path`, so readers never
/// see a partial file. Syncs `op` to the host.
/// @throws NotImplementedError for particle rank 3; InputError if the file
///   cannot be written.
void WriteOperatorBinary(const std::filesystem::path& path,
                         const op::Operator& op, std::string_view label = {});

/// Reads the header of a native operator file.
/// @throws InputError for a missing, truncated or foreign file.
OperatorFileHeader ReadOperatorBinaryHeader(const std::filesystem::path& path);

/// Options of `ReadOperatorBinary`.
struct OperatorBinaryReadOptions {
  /// Reject a file whose reference hash differs from the model space's.
  bool check_reference = true;
};

/// Reads a native operator file into an operator on `ms`. Checks the layout
/// hash (always), the reference hash (unless disabled), the buffer counts,
/// the exact file size and the checksum.
/// @throws InputError on any mismatch, naming the file.
op::Operator ReadOperatorBinary(const std::filesystem::path& path,
                                std::shared_ptr<const basis::ModelSpace> ms,
                                const OperatorBinaryReadOptions& options = {});

/// Writes a file atomically: `write` fills a stream opened on
/// `<path>.tmp.<pid>` (binary mode), which is then renamed onto `path`.
/// @throws InputError if the file cannot be written or renamed.
void WriteFileAtomically(const std::filesystem::path& path,
                         const std::function<void(std::ostream&)>& write);

}  // namespace imsrg::io

#endif  // IMSRG_IO_OPERATOR_BINARY_H_
